"""
步骤4：基础检索+Rerank问答链路

职责：实现完整问答流程：用户提问 → 向量检索 → Reranker 精排 → LLM 生成答案

知识点讲解：
1. Reranker 的作用：向量检索是"粗排"（Bi-Encoder，问题和文档分别编码后比余弦相似度），
   Reranker 是"精排"（Cross-Encoder，把问题和文档拼在一起编码，能捕捉两者之间的交互），
   所以精度远高于向量检索，是 RAG 中性价比最高的一环。

2. 为什么 LLM 用 ChatOpenAI 而不是 OpenAI？
   - LangChain 有两个类：
     * OpenAI      → 调用旧版 /completions 端点（仅支持 completion 式补全）
     * ChatOpenAI  → 调用 /chat/completions 端点（支持 messages 角色对话）
   - DashScope 的 OpenAI 兼容模式只提供 /chat/completions，
     所以必须用 ChatOpenAI，否则会报 404 "Unsupported model XXX for OpenAI compatibility mode"
   - 这是本项目踩过的一个真实坑：类选错，模型名再对也没用

3. 为什么用 messages（角色化）而不是纯字符串 prompt？
   - system 消息设定身份与约束（"只依据上下文回答"）
   - human 消息承载动态内容（检索到的上下文 + 用户问题）
   - 角色分离让模型更稳定地遵守指令，是生产环境的标准写法
"""
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVE_TOP_K,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RERANKER_DEVICE,
    RERANKER_LOCAL_PATH,
)
from preprocess.storage.store import search

# ========== LLM 初始化（ChatOpenAI：/chat/completions）==========
# 懒加载单例：import 时不构造，首次调用 get_llm() 才创建。
# 好处1：Key 未配置/DashScope 欠费时 import 本模块不报错，离线链路照常可用；
# 好处2：Agent 层测试可通过 monkey-patch get_llm 注入桩 LLM，零成本验证图逻辑
_llm = None


def get_llm():
    """获取 LLM 实例（单例，懒加载）—— 所有 LLM 调用的统一出口"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    return _llm

# ========== Reranker 初始化（Cross-Encoder 精排模型）==========
# 懒加载：只有真正用到时才加载模型，避免 import 阶段就占用显存
_reranker = None


def get_reranker():
    """获取 Reranker 实例（单例，懒加载）"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[Rerank] 正在加载重排模型（本地路径: {RERANKER_LOCAL_PATH}）...")
        _reranker = CrossEncoder(
            RERANKER_LOCAL_PATH if RERANKER_LOCAL_PATH else RERANKER_MODEL,
            device=RERANKER_DEVICE,
        )
        print("[Rerank] 重排模型加载完成")
    return _reranker


# ========== Prompt 模板（system + human 角色分离）==========
SYSTEM_PROMPT = """你是一个企业知识库智能助手。
请严格依据【参考资料】回答用户问题，遵守以下规则：
1. 只使用参考资料中的信息，不要编造、不要引入外部知识；
2. 如果参考资料不足以回答，请明确说明"根据现有资料无法回答"；
3. 回答要准确、简洁，分点作答时保持条理清晰。"""

HUMAN_PROMPT = """【参考资料】
{context}

【用户问题】
{question}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


# ========== 环节1：检索 + 重排（不依赖 LLM，可独立测试）==========
def retrieve_and_rerank(query: str) -> list[dict[str, Any]]:
    """
    向量检索 → Reranker 精排，返回 Top N 结果

    返回：[{"text": ..., "vector_score": 0.30, "rerank_score": 5.12,
            "doc_name": ..., "chunk_index": ...}, ...]
    其中 vector_score 是余弦相似度，rerank_score 是 Cross-Encoder 原始分（无固定范围）
    """
    # 1) 向量检索：召回 Top K（粗排）
    print(f"[QA] 向量检索 Top {RETRIEVE_TOP_K} ...")
    candidates = search(query, top_k=RETRIEVE_TOP_K)
    if not candidates:
        return []

    # 2) Reranker 精排：对 (问题, 文档) 逐对打分
    print(f"[QA] Reranker 精排 Top {RERANK_TOP_N} ...")
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # 3) 按重排分降序，取前 N
    merged = []
    for candidate, score in zip(candidates, scores):
        merged.append({
            "text": candidate["text"],
            "vector_score": candidate["score"],
            "rerank_score": float(score),
            "doc_name": candidate["doc_name"],
            "chunk_index": candidate["chunk_index"],
        })
    merged.sort(key=lambda x: x["rerank_score"], reverse=True)
    return merged[:RERANK_TOP_N]


# ========== 环节2：上下文组装 ==========
def build_context(reranked: list[dict[str, Any]]) -> str:
    """
    把重排后的片段拼成上下文，并标注来源（强制溯源的基础）

    格式：每段前加 [来源: 文档名#块号]，方便 LLM 引用、也方便人工核查
    """
    parts = []
    for i, item in enumerate(reranked, 1):
        parts.append(
            f"[片段{i} 来源: {item['doc_name']}#{item['chunk_index']}]\n{item['text']}"
        )
    return "\n\n".join(parts)


# ========== 环节3：LLM 生成答案 ==========
def generate_answer(query: str, context: str) -> str:
    """调用 LLM 基于上下文生成答案（需要 DashScope 账户可用）"""
    messages = PROMPT.format_messages(context=context, question=query)
    response = get_llm().invoke(messages)
    return response.content


# ========== 完整链路 ==========
def qa_chain(query: str, verbose: bool = True) -> str:
    """
    完整问答链路：用户提问 → 向量检索 → Reranker 精排 → LLM 生成答案
    """
    print(f"[QA] 用户问题: {query}")
    reranked = retrieve_and_rerank(query)
    if not reranked:
        return "向量库为空，请先执行入库操作。"

    context = build_context(reranked)
    print("[QA] LLM 生成答案 ...")
    return generate_answer(query, context)


# ========== 运行测试 ==========
if __name__ == "__main__":
    from preprocess.parsing.parser import parse_document
    from preprocess.chunking.chunker import chunk_text
    from preprocess.storage.store import add_chunks, clear_collection, get_collection_count
    from config.settings import DATA_DIR

    # ---------- 准备：解析 → 切分 → 入库 ----------
    clear_collection()
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    chunks = chunk_text(parse_document(doc_path))
    add_chunks(chunks, doc_name="smart_office_manual")
    print(f"\n[测试] 向量库文档数: {get_collection_count()}")

    # ---------- 测试A：检索 + Rerank（不依赖 LLM）----------
    query = "请假审批流程是什么？"
    print("\n" + "=" * 64)
    print("【测试A】向量粗排 vs Reranker 精排 对比")
    print("=" * 64)
    print(f"查询: {query}")

    # 先看纯向量检索的原始顺序
    raw = search(query, top_k=RETRIEVE_TOP_K)
    print("\n--- 向量检索原始顺序（粗排）---")
    for i, r in enumerate(raw, 1):
        print(f"  {i}. vector_score={r['score']:.4f} "
              f"| 块{r['chunk_index']} | {r['text'][:45].replace(chr(10), ' ')}...")

    # 再看重排后的顺序
    reranked = retrieve_and_rerank(query)
    print("\n--- Reranker 精排后顺序（送给 LLM 的 Top N）---")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. rerank_score={r['rerank_score']:.4f} "
              f"(vector={r['vector_score']:.4f}) | 块{r['chunk_index']}")
        print(f"      {r['text'][:60].replace(chr(10), ' ')}...")

    print("\n" + "=" * 64)
    print("【结论】对比两次顺序，若重排把真正相关的片段提到前面，说明 Reranker 生效")
    print("=" * 64)

    # ---------- 测试B：LLM 生成（需账户可用，暂缓）----------
    print("\n【测试B】LLM 生成答案 —— 需要 DashScope 账户余额，暂跳过")
    print("完整代码已就绪，充值后直接运行 qa_chain(query) 即可。")
