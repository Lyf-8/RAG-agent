"""
阶段2-步骤4：Hybrid 混合检索 + RRF 融合 + 父块上下文（升级版问答链路）

职责：串联阶段2全部升级件，形成新版问答链路：
  用户问题 → 稠密/稀疏双路召回 → RRF 融合 → Reranker 精排（子块级）
         → 回取父块 → 组装上下文 → LLM 生成答案
"""
from typing import Any

from config.settings import (
    HYBRID_TOP_K,
    RRF_K,
    RERANK_TOP_N,
    DATA_DIR,
)
from preprocess.storage.qdrant_store import (
    search_dense,
    search_sparse,
    fetch_parents,
)
# 复用阶段1的 Reranker 单例与 LLM 生成函数，避免重复加载模型
from qa_chain import get_reranker, generate_answer


# ========== 环节1：双路召回 + RRF 融合 ==========
def rrf_fuse(dense_results: list[dict], sparse_results: list[dict], k: int = RRF_K) -> list[dict]:
    """
    RRF 倒数排名融合：把两路检索结果按"排名"融合成一个列表

    公式：某条目 RRF 分数 = Σ（每一路中 1 / (k + 该路排名)）
    - 只用一个参数 k=60，无需调权重
    - 两路都命中的条目自动加分 → 共识条目置顶
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for rank, r in enumerate(dense_results, 1):
        scores[r["child_id"]] = scores.get(r["child_id"], 0.0) + 1.0 / (k + rank)
        items[r["child_id"]] = r
    for rank, r in enumerate(sparse_results, 1):
        scores[r["child_id"]] = scores.get(r["child_id"], 0.0) + 1.0 / (k + rank)
        items.setdefault(r["child_id"], r)

    fused = sorted(items.values(), key=lambda x: scores[x["child_id"]], reverse=True)
    for it in fused:
        it["rrf_score"] = scores[it["child_id"]]
    return fused


def hybrid_search(query: str, top_k: int = HYBRID_TOP_K) -> list[dict]:
    """双路召回 → RRF 融合，返回融合后的子块列表"""
    print(f"[Hybrid] 稠密路召回 Top {top_k} ...")
    dense_results = search_dense(query, top_k=top_k)
    print(f"[Hybrid] 稀疏路召回 Top {top_k} ...")
    sparse_results = search_sparse(query, top_k=top_k)
    fused = rrf_fuse(dense_results, sparse_results)
    print(f"[Hybrid] RRF 融合完成：{len(dense_results)} + {len(sparse_results)} 路结果 → {len(fused)} 个去重候选")
    return fused


# ========== 环节2：Reranker 精排（子块级）==========
def retrieve_and_rerank_v2(query: str) -> list[dict[str, Any]]:
    """Hybrid 召回 → Reranker 精排，返回 Top N 子块"""
    candidates = hybrid_search(query)
    if not candidates:
        return []

    print(f"[Hybrid] Reranker 精排 Top {RERANK_TOP_N} ...")
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:RERANK_TOP_N]


# ========== 环节3：子块 → 父块，组装上下文 ==========
def build_context_v2(reranked_children: list[dict[str, Any]]) -> str:
    """
    用命中子块的 parent_id 回取父块原文，拼成 LLM 上下文

    要点：多个子块可能命中同一个父块 → 按 parent_id 去重，
    且保持"精排分最高的子块所对应父块"排最前。
    """
    # 按精排顺序取 parent_id，去重保序
    seen = set()
    parent_ids = []
    for c in reranked_children:
        if c["parent_id"] not in seen:
            seen.add(c["parent_id"])
            parent_ids.append(c["parent_id"])

    parents = fetch_parents(parent_ids)

    parts = []
    for i, pid in enumerate(parent_ids, 1):
        p = parents[pid]
        parts.append(f"[片段{i} 来源: {p['heading_path']}]\n{p['text']}")
    return "\n\n".join(parts)


# ========== 完整链路 ==========
def qa_chain_v2(query: str) -> str:
    """阶段2完整问答链路（LLM 环节需 DashScope 账户可用）"""
    print(f"[QA-v2] 用户问题: {query}")
    reranked = retrieve_and_rerank_v2(query)
    if not reranked:
        return "向量库为空，请先执行入库操作。"
    context = build_context_v2(reranked)
    print("[QA-v2] LLM 生成答案 ...")
    return generate_answer(query, context)


# ========== 运行测试 ==========
if __name__ == "__main__":
    from preprocess.parsing.structured_parser import parse_document_structured
    from preprocess.chunking.parent_child_chunker import build_parent_child_chunks
    from preprocess.storage.qdrant_store import recreate_collections, add_parent_child_chunks

    # ---------- 准备：重建并入库 ----------
    recreate_collections()
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    sections = parse_document_structured(doc_path)
    parents, children = build_parent_child_chunks(sections, doc_name="smart_office_manual")
    add_parent_child_chunks(parents, children)

    # ---------- 测试：三路结果对比（稠密 / 稀疏 / RRF融合）----------
    query = "GPS定位打卡的精度是多少？"
    print("\n" + "=" * 70)
    print(f"查询: {query}")
    print("=" * 70)

    dense_res = search_dense(query)
    sparse_res = search_sparse(query)
    fused = rrf_fuse(dense_res, sparse_res)

    def show(title, results, score_key, n=3):
        print(f"\n--- {title} Top{n} ---")
        for i, r in enumerate(results[:n], 1):
            print(f"  {i}. {score_key}={r[score_key]:.4f} | {r['heading_path']}")

    show("稠密路（语义）", dense_res, "score")
    show("稀疏路（字面）", sparse_res, "score")
    show("RRF 融合", fused, "rrf_score")

    # ---------- 测试：精排 → 父块上下文 ----------
    print("\n" + "-" * 70)
    print("精排后回取父块，组装给 LLM 的上下文：")
    print("-" * 70)
    reranked = retrieve_and_rerank_v2(query)
    context = build_context_v2(reranked)
    print(context)
    print("=" * 70)
    print("说明：LLM 生成环节依赖 DashScope 充值，本测试到此为止（检索链路已完整验证）")
