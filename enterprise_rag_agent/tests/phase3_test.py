"""
阶段3-步骤5：Agent 图路由测试（桩 LLM 版）

目的：不调用真实 LLM（DashScope 欠费也能跑），验证 LangGraph 图的
      结构与路由逻辑完全正确 —— 这是"编排层"的单元测试。

核心手法：monkey-patch（猴子补丁）注入桩 LLM
    agent.nodes.get_llm = lambda: fake_llm
  nodes 模块里所有 LLM 调用都经 get_llm() 出口，替换后节点拿到的就是桩。
  这正是 nodes.py 把 LLM 收口到 get_llm() 的原因：可测试性（Testability）。

桩的行为策略（按 Prompt 关键词识别调用类型）：
    Prompt 含"判断该片段是否相关" → 打分调用 → 按预设返回 yes / no
    Prompt 含"改写后的问题"       → 改写调用 → 返回预设的改写问题
    Prompt 含"参考资料"           → 生成调用 → 返回预设的模拟答案

两个场景覆盖全部路由分支：
    场景A：LLM 判所有子块相关     → 路由直达 generate（无改写循环）
    场景B：LLM 判所有子块不相关   → rewrite 重试至上限 → 兜底 generate
    （检索节点是真实运行的 —— Qdrant 本地库不花钱，顺带验证了检索接入正常）

运行方式（项目根目录下）：
    python -m tests.phase3_test
"""
import sys
from typing import Any

# ---------- 导入待测模块 ----------
import agent.nodes as nodes
from agent.graph import build_graph
from config.settings import AGENT_MAX_REWRITES, DATA_DIR


# ========== 桩 LLM：替代 ChatOpenAI，记录调用、按关键词返回预设内容 ==========
class FakeLLM:
    """最小桩：只实现节点用到的 invoke()，接口对齐 LangChain 的 .invoke(list[BaseMessage])"""

    def __init__(self, grade_verdict: str = "yes",
                 rewritten_question: str = "GPS定位打卡的精度要求是多少？",
                 fake_answer: str = "这是桩LLM生成的模拟答案。"):
        self.grade_verdict = grade_verdict
        self.rewritten_question = rewritten_question
        self.fake_answer = fake_answer
        # 调用计数（用于断言路由走到了哪些节点）
        self.grade_calls = 0
        self.rewrite_calls = 0
        self.generate_calls = 0

    def invoke(self, messages: list) -> Any:
        # 拼接所有消息文本作为识别依据
        text = "\n".join(getattr(m, "content", str(m)) for m in messages)
        if "判断该片段是否相关" in text:          # grade_documents 节点
            self.grade_calls += 1
            return _Msg(self.grade_verdict)
        if "改写后的问题" in text:                 # rewrite_query 节点
            self.rewrite_calls += 1
            return _Msg(self.rewritten_question)
        if "参考资料" in text:                     # generate 节点（复用 qa_chain.PROMPT）
            self.generate_calls += 1
            return _Msg(self.fake_answer)
        raise ValueError(f"FakeLLM 收到无法识别的调用，请检查 Prompt 关键词。前100字符: {text[:100]}")


class _Msg:
    """模拟 LLM 响应对象：节点只读 .content 属性"""
    def __init__(self, content: str):
        self.content = content


# ========== 测试前准备：确保向量库有数据（检索节点要真实运行）==========
def ensure_vector_db_ready():
    from preprocess.storage.qdrant_store import get_counts
    counts = get_counts()
    if counts["children"] > 0 and counts["parents"] > 0:
        print(f"[准备] Qdrant 已有数据（子块 {counts['children']} / 父块 {counts['parents']}），直接复用")
        return
    print("[准备] Qdrant 为空，重建并入库（需加载 BGE-M3，仅首次）...")
    from preprocess.parsing.structured_parser import parse_document_structured
    from preprocess.chunking.parent_child_chunker import build_parent_child_chunks
    from preprocess.storage.qdrant_store import recreate_collections, add_parent_child_chunks
    recreate_collections()
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    parents, children = build_parent_child_chunks(
        parse_document_structured(doc_path), doc_name="smart_office_manual"
    )
    add_parent_child_chunks(parents, children)
    print(f"[准备] 入库完成: {get_counts()}")


# ========== 场景A：全部相关 → 直达生成 ==========
def test_case_a():
    print("\n" + "=" * 70)
    print("【场景A】LLM 判定所有子块相关 → 预期：直达 generate，无改写循环")
    print("=" * 70)

    fake = FakeLLM(grade_verdict="yes")
    nodes.get_llm = lambda: fake                 # ← 注入桩（monkey-patch）
    app = build_graph()

    answer = app.invoke({
        "question": "GPS定位打卡的精度是多少？",
        "original_question": "GPS定位打卡的精度是多少？",
        "documents": [], "relevant_docs": [], "rewrite_count": 0, "answer": "",
    })

    # ---- 断言：路由走了预期路径 ----
    assert fake.grade_calls > 0, "应发生过相关性打分调用"
    assert fake.rewrite_calls == 0, "有相关文档时不应触发改写"
    assert fake.generate_calls == 1, "生成应恰好执行 1 次"
    assert answer["rewrite_count"] == 0, "不应发生改写"
    assert len(answer["relevant_docs"]) > 0, "应保留相关子块"
    assert answer["answer"] == fake.fake_answer, "答案应来自桩 LLM"
    print(f"\n[断言通过] 场景A：打分 {fake.grade_calls} 次 → 全部相关 → 直接生成，改写 0 次")
    print(f"[最终答案] {answer['answer']}")


# ========== 场景B：全部不相关 → 改写循环至上限 → 兜底生成 ==========
def test_case_b():
    print("\n" + "=" * 70)
    print(f"【场景B】LLM 判定所有子块不相关 → 预期：改写重试 {AGENT_MAX_REWRITES} 次后兜底生成")
    print("=" * 70)

    fake = FakeLLM(grade_verdict="no")
    nodes.get_llm = lambda: fake
    app = build_graph()

    answer = app.invoke({
        "question": "随便问点什么呀",
        "original_question": "随便问点什么呀",
        "documents": [], "relevant_docs": [], "rewrite_count": 0, "answer": "",
    })

    # ---- 断言：循环与终止条件 ----
    assert fake.grade_calls > 0, "应发生过打分调用"
    assert fake.rewrite_calls == AGENT_MAX_REWRITES, \
        f"改写应恰好执行 {AGENT_MAX_REWRITES} 次，实际 {fake.rewrite_calls} 次"
    assert fake.generate_calls == 1, "兜底生成应恰好执行 1 次"
    assert answer["rewrite_count"] == AGENT_MAX_REWRITES, "改写计数应达到上限"
    assert len(answer["relevant_docs"]) == 0, "全程无相关文档"
    assert answer["answer"] == fake.fake_answer, "兜底也应产出答案（由 Prompt 规则保证内容为'无法回答'）"
    print(f"\n[断言通过] 场景B：改写 {fake.rewrite_calls}/{AGENT_MAX_REWRITES} 次 → 达上限兜底生成，图正常终止")
    print(f"[最终答案] {answer['answer']}")


# ========== 主入口 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("阶段3测试：LangGraph 图路由验证（桩 LLM，不消耗任何 API 费用）")
    print("=" * 70)

    # 先打印图结构（Mermaid 文本，可粘贴到 mermaid.live 渲染）
    app = build_graph()
    print("\n[图结构]\n" + app.get_graph().draw_mermaid())

    ensure_vector_db_ready()

    test_case_a()
    test_case_b()

    print("\n" + "=" * 70)
    print("阶段3全部测试通过 ✓  图结构、条件路由、改写循环、兜底终止均符合预期")
    print("说明：真实 LLM 问答需 DashScope 充值后运行 agent/graph.py 的 ask(question)")
    print("=" * 70)
    sys.exit(0)
