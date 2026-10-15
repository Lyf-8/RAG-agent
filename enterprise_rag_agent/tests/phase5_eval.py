"""
阶段5-步骤1：检索质量量化评估（Recall@K / Precision@K / MRR）

职责：用 tests/eval_dataset.py 中的人工标注题库，对当前完整检索链路做量化体检，
     全程离线运行（不调用 LLM，零 API 费用）。
"""
from config.settings import DATA_DIR, RERANK_TOP_N
from preprocess.storage.qdrant_store import get_counts
from qa_chain_v2 import hybrid_search, retrieve_and_rerank_v2
from tests.eval_dataset import EVAL_DATASET

# 评估用的 K 值档位：1 看"一击即中"，4 = 送给 LLM 的窗口大小，10 看召回兜底能力
K_LEVELS = (1, 4, 10)


# ========== 基础判据：关键词命中 ==========
def first_relevant_rank(results: list[dict], keyword: str) -> int:
    """首个含标准答案关键词的片段排名（1-based）；未出现返回 0"""
    for i, c in enumerate(results, 1):
        if keyword in c["text"]:
            return i
    return 0


def hits_in_top_k(results: list[dict], keyword: str, k: int) -> int:
    """Top K 中含关键词的片段数量（用于 Precision@K）"""
    return sum(1 for c in results[:k] if keyword in c["text"])


# ========== 三个指标的计算（输入均为逐题统计量的列表）==========
def recall_at_k(ranks: list[int], k: int) -> float:
    """Recall@K：正解首个排名 ≤ K 的问题占比（0=未命中，不计入）"""
    return sum(1 for r in ranks if 0 < r <= k) / len(ranks)


def precision_at_k(hit_counts: list[int], k: int) -> float:
    """Precision@K：逐题 TopK 相关片段数 / K，再取平均"""
    return sum(min(c, k) / k for c in hit_counts) / len(hit_counts)


def mrr(ranks: list[int]) -> float:
    """MRR：首个相关片段排名的倒数平均（未命中计 0）"""
    return sum(1.0 / r if r > 0 else 0.0 for r in ranks) / len(ranks)


# ========== 数据自检：向量库为空则自动重新入库 ==========
def ensure_data_ready():
    counts = get_counts()
    if counts["children"] > 0:
        print(f"[Eval] 检测到已有向量库：子块 {counts['children']} 个，父块 {counts['parents']} 个，直接复用")
        return
    print("[Eval] 向量库为空，执行入库流水线 ...")
    from preprocess.parsing.structured_parser import parse_document_structured
    from preprocess.chunking.parent_child_chunker import build_parent_child_chunks
    from preprocess.storage.qdrant_store import recreate_collections, add_parent_child_chunks
    recreate_collections()
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    sections = parse_document_structured(doc_path)
    parents, children = build_parent_child_chunks(sections, doc_name="smart_office_manual")
    add_parent_child_chunks(parents, children)
    print(f"[Eval] 入库完成：{get_counts()}")


def main():
    ensure_data_ready()

    rows = []  # (题号, 问题, 召回层排名, 精排层排名, 召回层各K命中数, 精排层各K命中数)
    for idx, (question, keyword) in enumerate(EVAL_DATASET, 1):
        print(f"\n----- [{idx}/{len(EVAL_DATASET)}] {question} -----")
        # 召回层：Hybrid 双路 + RRF 融合（未精排的完整候选列表）
        recalled = hybrid_search(question)
        rank_recall = first_relevant_rank(recalled, keyword)
        hits_recall = {k: hits_in_top_k(recalled, keyword, k) for k in K_LEVELS}

        # 精排层：+ Reranker，即最终送给 LLM 的 Top N
        reranked = retrieve_and_rerank_v2(question)
        rank_rerank = first_relevant_rank(reranked, keyword)
        hits_rerank = {k: hits_in_top_k(reranked, keyword, k) for k in (1, RERANK_TOP_N)}

        rows.append((idx, question, rank_recall, rank_rerank, hits_recall, hits_rerank))

    # ---------- 逐题明细 ----------
    print("\n" + "=" * 76)
    print("逐题排名明细（0 = 未命中）")
    print("=" * 76)
    print(f"{'#':<3}{'问题':<30}{'召回层排名(RRF)':<16}{'精排层排名(Top4)':<14}")
    print("-" * 76)
    for idx, q, r1, r2, _, _ in rows:
        print(f"{idx:<3}{q:<30}{r1:<16}{r2:<14}")

    recall_ranks = [r[2] for r in rows]
    rerank_ranks = [r[3] for r in rows]

    # ---------- 召回层汇总 ----------
    print("\n" + "=" * 76)
    print("【召回层】Hybrid 双路 + RRF（考察'捞全'能力，是精排的候选池）")
    print("=" * 76)
    for k in K_LEVELS:
        hits = [r[4][k] for r in rows]
        print(f"  Recall@{k:<2} = {recall_at_k(recall_ranks, k):.1%}    "
              f"Precision@{k:<2} = {precision_at_k(hits, k):.1%}")
    print(f"  MRR      = {mrr(recall_ranks):.4f}")

    # ---------- 精排层汇总 ----------
    print("\n" + "=" * 76)
    print(f"【精排层】+ Reranker（最终送给 LLM 的 Top {RERANK_TOP_N}，考察'排准'能力）")
    print("=" * 76)
    for k in (1, RERANK_TOP_N):
        hits = [r[5][k] for r in rows]
        print(f"  Recall@{k:<2} = {recall_at_k(rerank_ranks, k):.1%}    "
              f"Precision@{k:<2} = {precision_at_k(hits, k):.1%}")
    print(f"  MRR      = {mrr(rerank_ranks):.4f}")
    print("=" * 76)
    print("解读提示：")
    print("  召回层 Recall@10 < 100% → 有题在召回阶段就丢了，应优先改切分/召回；")
    print("  召回层全中但精排层 MRR 低 → 正解在候选池里但排不上去，应调精排/候选数。")


if __name__ == "__main__":
    main()
