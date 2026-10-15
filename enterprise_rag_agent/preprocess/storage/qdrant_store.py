"""
阶段2-步骤3-B：Qdrant 向量库存储与双路检索

职责：替代阶段1的 Chroma，用 Qdrant 存储子块（稠密+稀疏双向量）和父块原文。
"""
from typing import List, Dict, Any
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client import QdrantClient
from qdrant_client import models

from config.settings import (
    QDRANT_DIR,
    QDRANT_COLLECTION,
    QDRANT_PARENT_COLLECTION,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    HYBRID_TOP_K,
)
from preprocess.embedding.hybrid_embedder import encode_hybrid, encode_query_hybrid

# 客户端单例
_client = None

DENSE_DIM = 1024  # BGE-M3 稠密向量维度


def get_client() -> QdrantClient:
    """获取 Qdrant 客户端（本地文件模式，无需 Docker）"""
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_DIR))
        print(f"[QdrantStore] 本地文件模式客户端已创建，存储路径: {QDRANT_DIR}")
    return _client


def _to_uuid(text_id: str) -> str:
    """字符串 ID → 确定性 UUID（同一个 ID 永远映射到同一个 UUID，保证幂等）"""
    return str(uuid5(NAMESPACE_DNS, text_id))


def recreate_collections():
    """删除并重建两个 Collection（入库前调用，保证数据干净）"""
    client = get_client()
    for name in (QDRANT_COLLECTION, QDRANT_PARENT_COLLECTION):
        if client.collection_exists(name):
            client.delete_collection(name)

    # 子块 Collection：稠密命名向量 + 稀疏命名向量
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(),
        },
    )
    # 父块 Collection：只当 KV 用，挂 1 维占位向量
    client.create_collection(
        collection_name=QDRANT_PARENT_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=1, distance=models.Distance.COSINE),
        },
    )
    print(f"[QdrantStore] Collection 已重建: {QDRANT_COLLECTION}（双向量）, {QDRANT_PARENT_COLLECTION}（父块KV）")


def _sparse_to_qdrant(sparse_dict: Dict) -> models.SparseVector:
    """FlagEmbedding 稀疏字典 → Qdrant SparseVector（key 可能是 str，统一转 int）"""
    indices = [int(k) for k in sparse_dict.keys()]
    values = [float(v) for v in sparse_dict.values()]
    return models.SparseVector(indices=indices, values=values)


def add_parent_child_chunks(parents: List[dict], children: List[dict]) -> int:
    """
    父子块入库：
    - 子块 → 双路编码（dense+sparse）→ rag_hybrid Collection
    - 父块 → 原文直接存 → rag_parents Collection（不编码，不建索引）
    """
    client = get_client()

    # ---------- 1. 父块入库（纯 KV，无需向量化）----------
    parent_points = [
        models.PointStruct(
            id=_to_uuid(p["parent_id"]),
            vector={DENSE_VECTOR_NAME: [0.0]},  # 占位向量，父块不参与向量检索
            payload=p,
        )
        for p in parents
    ]
    client.upsert(collection_name=QDRANT_PARENT_COLLECTION, points=parent_points)
    print(f"[QdrantStore] 父块入库完成: {len(parents)} 条")

    # ---------- 2. 子块双路编码 ----------
    child_texts = [c["text"] for c in children]
    dense_vecs, sparse_vecs = encode_hybrid(child_texts)

    # ---------- 3. 子块入库（稠密+稀疏双向量）----------
    child_points = []
    for c, dense, sparse in zip(children, dense_vecs, sparse_vecs):
        child_points.append(
            models.PointStruct(
                id=_to_uuid(c["child_id"]),
                vector={
                    DENSE_VECTOR_NAME: dense.tolist(),
                    SPARSE_VECTOR_NAME: _sparse_to_qdrant(sparse),
                },
                payload=c,
            )
        )
    client.upsert(collection_name=QDRANT_COLLECTION, points=child_points)
    print(f"[QdrantStore] 子块入库完成: {len(children)} 条（每条含 dense 1024维 + sparse 向量）")
    return len(children)


def _points_to_results(points) -> List[Dict[str, Any]]:
    """Qdrant ScoredPoint 列表 → 统一结果字典"""
    results = []
    for pt in points:
        results.append({
            "child_id": pt.payload["child_id"],
            "parent_id": pt.payload["parent_id"],
            "text": pt.payload["text"],
            "heading_path": pt.payload.get("heading_path", ""),
            "doc_name": pt.payload.get("doc_name", "unknown"),
            "score": float(pt.score),
        })
    return results


def search_dense(query: str, top_k: int = HYBRID_TOP_K) -> List[Dict[str, Any]]:
    """稠密路检索：语义匹配（与阶段1向量检索等价，但检索对象是小子块）"""
    client = get_client()
    dense_q, _ = encode_query_hybrid(query)
    resp = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=dense_q.tolist(),
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )
    return _points_to_results(resp.points)


def search_sparse(query: str, top_k: int = HYBRID_TOP_K) -> List[Dict[str, Any]]:
    """稀疏路检索：字面/关键词匹配（模型学习到的词权重，强化版 BM25）"""
    client = get_client()
    _, sparse_q = encode_query_hybrid(query)
    resp = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=_sparse_to_qdrant(sparse_q),
        using=SPARSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
    )
    return _points_to_results(resp.points)


def fetch_parents(parent_ids: List[str]) -> Dict[str, dict]:
    """按 parent_id 批量取父块原文（KV 查询），返回 {parent_id: payload}"""
    client = get_client()
    points = client.retrieve(
        collection_name=QDRANT_PARENT_COLLECTION,
        ids=[_to_uuid(pid) for pid in parent_ids],
        with_payload=True,
    )
    return {pt.payload["parent_id"]: pt.payload for pt in points}


def get_counts() -> Dict[str, int]:
    """两个 Collection 的点数统计"""
    client = get_client()
    return {
        "children": client.count(QDRANT_COLLECTION).count,
        "parents": client.count(QDRANT_PARENT_COLLECTION).count,
    }


# ========== 运行测试：完整离线入库流水线 ==========
if __name__ == "__main__":
    from preprocess.parsing.structured_parser import parse_document_structured
    from preprocess.chunking.parent_child_chunker import build_parent_child_chunks
    from config.settings import DATA_DIR

    # 完整流水线：结构化解析 → 父子块切分 → 双路编码 → Qdrant 入库
    recreate_collections()

    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    sections = parse_document_structured(doc_path)
    parents, children = build_parent_child_chunks(sections, doc_name="smart_office_manual")
    add_parent_child_chunks(parents, children)

    print(f"\n[测试] 入库统计: {get_counts()}")

    # 双路检索对比测试
    query = "采购20000元以上需要谁审批？"
    print(f"\n[测试] 查询: '{query}'")
    print("\n--- 稠密路（语义）Top3 ---")
    for i, r in enumerate(search_dense(query, top_k=3), 1):
        print(f"  {i}. score={r['score']:.4f} | {r['heading_path']}")
        print(f"     {r['text'][:60]}...")
    print("\n--- 稀疏路（字面）Top3 ---")
    for i, r in enumerate(search_sparse(query, top_k=3), 1):
        print(f"  {i}. score={r['score']:.4f} | {r['heading_path']}")
        print(f"     {r['text'][:60]}...")

    # 父块回取测试
    pid = search_dense(query, top_k=1)[0]["parent_id"]
    parent = fetch_parents([pid])[pid]
    print(f"\n[测试] 子块命中 → 回取父块 {pid}:")
    print(parent["text"])
