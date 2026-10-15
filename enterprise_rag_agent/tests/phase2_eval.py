"""
阶段2-步骤5：升级前后检索质量对比评估

职责：用同一套业务评估集，量化对比四种检索链路：
  A. 阶段1：Chroma 稠密粗排（800字符大块）
  B. 阶段2：Qdrant 稠密粗排（200字符子块 + 标题路径增强）
  C. 阶段2：Qdrant Hybrid（稠密+稀疏，RRF 融合）
  D. 阶段2：Hybrid + Reranker 精排（完整新链路）
"""
from config.settings import DATA_DIR, RETRIEVE_TOP_K

# 阶段1链路（Chroma + 大块）
from preprocess.parsing.parser import parse_document
from preprocess.chunking.chunker import chunk_text
from preprocess.storage.store import (
    add_chunks, search as chroma_search, clear_collection, get_collection_count,
)

# 阶段2链路（Qdrant + 父子块 + Hybrid）
from preprocess.parsing.structured_parser import parse_document_structured
from preprocess.chunking.parent_child_chunker import build_parent_child_chunks
from preprocess.storage.qdrant_store import (
    recreate_collections, add_parent_child_chunks, search_dense,
)
from qa_chain_v2 import rrf_fuse, retrieve_and_rerank_v2
from preprocess.storage.qdrant_store import search_sparse

# ========== 评估集（与 tests/poc_eval.py 完全一致，保证公平对比）==========
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
    """含标准答案关键词的片段排名（1-based）；未出现返回 0"""
    for i, c in enumerate(chunks, 1):
        if keyword in c["text"]:
            return i
    return 0


def main():
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"

    # ---------- 1. 双链路入库 ----------
    print("=" * 70)
    print("【入库】阶段1链路（Chroma 大块）")
    print("=" * 70)
    clear_collection()
    chunks = chunk_text(parse_document(doc_path))
    add_chunks(chunks, doc_name="smart_office_manual")
    print(f"[Eval] Chroma 大块数: {get_collection_count()}")

    print("\n" + "=" * 70)
    print("【入库】阶段2链路（Qdrant 父子块双向量）")
    print("=" * 70)
    recreate_collections()
    sections = parse_document_structured(doc_path)
    parents, children = build_parent_child_chunks(sections, doc_name="smart_office_manual")
    add_parent_child_chunks(parents, children)

    # ---------- 2. 逐题跑四种链路 ----------
    rows = []
    for idx, (question, keyword) in enumerate(EVAL_SET, 1):
        # A. 阶段1 稠密大块
        res_a = chroma_search(question, top_k=RETRIEVE_TOP_K)
        rank_a = correct_rank(res_a, keyword)

        # B. 阶段2 稠密子块
        res_b = search_dense(question)
        rank_b = correct_rank(res_b, keyword)

        # C. 阶段2 Hybrid + RRF
        res_c = rrf_fuse(res_b, search_sparse(question))
        rank_c = correct_rank(res_c, keyword)

        # D. 阶段2 Hybrid + RRF + 精排（完整链路）
        res_d = retrieve_and_rerank_v2(question)
        rank_d = correct_rank(res_d, keyword)

        rows.append((idx, question, rank_a, rank_b, rank_c, rank_d))

    # ---------- 3. 汇总 ----------
    n = len(EVAL_SET)

    def top1(col): return sum(1 for r in rows if r[col] == 1) / n
    def mrr(col): return sum(1 / r[col] if r[col] else 0 for r in rows) / n

    print("\n" + "=" * 92)
    print("逐题排名对比（数字=正解片段排名，0=未召回）")
    print("=" * 92)
    print(f"{'#':<3}{'问题':<26}{'A:旧大块':<10}{'B:稠密子块':<12}{'C:+Hybrid':<11}{'D:+精排':<9}")
    print("-" * 92)
    for idx, q, a, b, c, d in rows:
        print(f"{idx:<3}{q:<26}{a:<10}{b:<12}{c:<11}{d:<9}")

    print("-" * 92)
    print(f"{'Top-1 准确率':<29}{top1(2):<10.1%}{top1(3):<12.1%}{top1(4):<11.1%}{top1(5):<9.1%}")
    print(f"{'MRR':<29}{mrr(2):<10.4f}{mrr(3):<12.4f}{mrr(4):<11.4f}{mrr(5):<9.4f}")
    print("=" * 92)
    print("归因解读：")
    print("  A→B：标题路径增强 + 子块聚焦（切分策略收益）")
    print("  B→C：稀疏路补足字面匹配（混合检索收益）")
    print("  C→D：Cross-Encoder 精排（重排收益）")


if __name__ == "__main__":
    main()
