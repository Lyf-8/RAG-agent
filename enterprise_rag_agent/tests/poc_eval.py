"""
步骤5：POC 阶段检索质量评估

背景：
本阶段"LLM 生成答案"环节依赖 DashScope 账户余额（当前暂缓充值），
因此先对 POC 最核心、也最可量化的一环 —— 检索质量 —— 做评估，
同时这个脚本就是阶段6「评估体系」的最小雏形。

评估方法（简化版 Hit@K）：
1. 为每条业务问题人工标注一个"标准答案关键词"（gold keyword）；
2. 该关键词在文档中只出现在唯一正确的片段里；
3. 检查 Top-K 片段中是否命中该关键词 → 命中记 1，否则 0；
4. 对比「向量粗排 Top-K」与「Reranker 精排 Top-N」，量化 Reranker 的实际收益。

这套指标（Hit@K / Recall@K / MRR）就是阶段6要正式建立的业务评估集基础。
"""
from preprocess.parsing.parser import parse_document
from preprocess.chunking.chunker import chunk_text
from preprocess.storage.store import add_chunks, search
from preprocess.storage.store import get_collection_count
from qa_chain import retrieve_and_rerank
from config.settings import DATA_DIR, RETRIEVE_TOP_K, RERANK_TOP_N

# ========== 业务评估集：问题 + 标准答案关键词 ==========
# 关键词取自示例文档中唯一、可判定的表述
EVAL_SET = [
    ("请假1天以内由谁审批？",            "直属主管审批即可生效"),
    ("请假3天以上需要哪些人审批？",       "分管副总审批"),
    ("加班调休的有效期是多久？",          "有效期90天"),
    ("采购20000元以上需要谁审批？",       "总经理审批"),
    ("电子发票需要上传什么格式？",        "PDF原件"),
    ("合同审批的时效要求是多久？",        "3个工作日内处理"),
    ("文档权限分为哪几个等级？",          "评论者"),
    ("90天以上的文档版本如何保留？",      "每周保留最后一个版本"),
    ("系统日志默认保留多少天？",          "180天"),
    ("密码连续错误5次会怎样？",           "锁定账号30分钟"),
    ("审批被驳回后能否重新提交？",        "重新提交"),
    ("GPS定位打卡的精度是多少？",         "50米"),
]


def correct_rank(chunks: list[dict], keyword: str) -> int:
    """
    返回"含标准答案关键词的片段"的排名（1-based）；未出现则返回 0

    为什么不用 Hit@K？
    本 POC 语料只有 4 个块，而 Top-K 一取就是 4 个，Hit@4 恒为 100%，没有区分度。
    改用"正解排名"才能真正反映检索质量：
    - 排名越靠前，说明检索越准（LLM 拿到正确依据的概率越高）
    - Top-1 准确率 = 正解排在第 1 位的比例
    - MRR = 平均 1/rank，是阶段6 的标准指标之一
    """
    for i, c in enumerate(chunks, 1):
        if keyword in c["text"]:
            return i
    return 0


def main():
    # ---------- 1. 准备向量库 ----------
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    chunks = chunk_text(parse_document(doc_path))
    add_chunks(chunks, doc_name="smart_office_manual")
    print(f"\n[Eval] 向量库文档数: {get_collection_count()}")

    # ---------- 2. 逐题评估 ----------
    rows = []
    for idx, (question, keyword) in enumerate(EVAL_SET, 1):
        raw = search(question, top_k=RETRIEVE_TOP_K)
        vec_rank = correct_rank(raw, keyword)

        reranked = retrieve_and_rerank(question)
        rr_rank = correct_rank(reranked, keyword)

        rows.append((idx, question, keyword, vec_rank, rr_rank))

    # ---------- 3. 结果汇总 ----------
    n = len(EVAL_SET)
    print("\n" + "=" * 84)
    print("评估结果（正解片段排名：1 = 排第一；0 = 未召回；- = 不适用）")
    print("=" * 84)
    print(f"{'#':<3}{'问题':<30}{'向量粗排':<12}{'精排后':<12}")
    print("-" * 84)
    for idx, question, keyword, vec_rank, rr_rank in rows:
        v = f"第{vec_rank}位" if vec_rank else "未召回"
        r = f"第{rr_rank}位" if rr_rank else "未召回"
        print(f"{idx:<3}{question:<30}{v:<12}{r:<12}")

    # Top-1 准确率 & MRR
    vec_top1 = sum(1 for r in rows if r[3] == 1) / n
    rr_top1 = sum(1 for r in rows if r[4] == 1) / n
    vec_mrr = sum(1 / r[3] if r[3] else 0 for r in rows) / n
    rr_mrr = sum(1 / r[4] if r[4] else 0 for r in rows) / n

    print("-" * 84)
    print(f"{'指标':<30}{'向量粗排':<12}{'Reranker 精排':<12}")
    print(f"{'Top-1 准确率':<30}{vec_top1:<12.1%}{rr_top1:<12.1%}")
    print(f"{'MRR（平均倒数排名）':<30}{vec_mrr:<12.4f}{rr_mrr:<12.4f}")
    print("=" * 84)

    # ---------- 4. Bad case 分析 ----------
    print("\n【Bad Case 分析】")
    worse = [r for r in rows if r[4] == 0 or (r[3] != 0 and r[4] > r[3])]
    better = [r for r in rows if (r[3] == 0 and r[4] != 0) or (r[4] != 0 and r[3] != 0 and r[4] < r[3])]
    print(f"  Reranker 使排名变好的问题: {len(better)} 条")
    for idx, question, _, v, r in better:
        print(f"    {idx}. {question}: 第{v or '-'}位 → 第{r}位")
    print(f"  Reranker 使排名变差/未召回的问题: {len(worse)} 条")
    for idx, question, _, v, r in worse:
        print(f"    {idx}. {question}: 第{v or '-'}位 → 第{r or '未召回'}")
    print("\n结论：Reranker 的作用是把正确依据【提前】，让 LLM 更容易只看到相关内容；")
    print("      未召回的问题属于传统 RAG 的召回天花板，正是阶段3/4 做查询改写+自检重试的动因。")


if __name__ == "__main__":
    main()
