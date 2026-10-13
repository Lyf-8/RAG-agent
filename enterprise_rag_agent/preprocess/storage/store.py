"""
步骤3-B：Chroma 向量库存储与检索

职责：将向量 + 文本存入 Chroma 本地向量库，支持相似度检索

知识点讲解：
1. 什么是向量库？
   把每个文本块的向量（1024维）和原文存到一个数据库里。
   检索时，把用户问题的向量也算出来，在库里找"距离最近"的几条。
   Chroma 是最轻量的本地向量库，一个 pip install 就能用，适合 POC。

2. Chroma 的核心概念：
   - Collection（集合）：类似数据库的"表"，一个项目放一个 collection
   - Document（文档）：每条记录 = 向量 + 原文 + 元数据
   - 检索：输入查询向量 → Chroma 返回最相似的 TopK 条

3. 为什么 POC 用 Chroma？
   - 纯本地，无需服务器，零部署成本
   - 自动持久化到磁盘，重启不丢失
   - 数据量 < 10万条时性能足够
   - 阶段2会升级为 Qdrant（支持更大数据量和混合检索）

4. 检索原理：余弦相似度（Cosine Similarity）
   - 两个向量方向越接近，相似度越高（范围 0~1）
   - 我们在 embedder 中已做归一化，Chroma 用余弦距离
"""
from typing import List, Dict, Any
import os

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import CHROMA_DIR, RETRIEVE_TOP_K
from preprocess.embedding.embedder import encode, encode_query

# Chroma 客户端单例
_client = None
_collection = None

# Collection 名称
COLLECTION_NAME = "rag_poc"


def get_client():
    """获取 Chroma 客户端（单例）"""
    global _client
    if _client is None:
        # 确保存储目录存在
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),  # 关闭遥测，不上报使用数据
        )
        print(f"[Store] Chroma 客户端已创建，持久化路径: {CHROMA_DIR}")
    return _client


def get_collection():
    """获取或创建 Collection（单例）"""
    global _collection
    if _collection is None:
        client = get_client()
        # get_or_create：已存在就获取，不存在就创建
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "POC 阶段 RAG 向量库"},
        )
        print(f"[Store] Collection '{COLLECTION_NAME}' 已就绪")
    return _collection


def add_chunks(chunks: List[str], doc_name: str = "unknown") -> int:
    """
    将文本块列表存入向量库

    流程：文本块 → BGE-M3 编码 → 向量 + 原文 + 元数据 → Chroma 存储

    参数：
    - chunks: 文本块列表（来自 chunker.chunk_text()）
    - doc_name: 来源文档名称（存入元数据，方便后续过滤）

    返回：存入的块数
    """
    if not chunks:
        print("[Store] 警告：块列表为空，跳过入库")
        return 0

    collection = get_collection()

    # 1. 编码：文本 → 向量
    print(f"[Store] 正在编码 {len(chunks)} 个文本块...")
    vectors = encode(chunks)

    # 2. 构造 ID 和元数据
    # ID 格式：doc_name_0, doc_name_1, ...（Chroma 要求 ID 唯一）
    ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_name": doc_name, "chunk_index": i} for i in range(len(chunks))]

    # 3. 存入 Chroma
    collection.upsert(
        ids=ids,
        embeddings=vectors.tolist(),     # Chroma 需要 list，不接受 numpy
        documents=chunks,                 # 原文也存进去，检索时直接返回
        metadatas=metadatas,
    )
    print(f"[Store] 入库完成：{len(chunks)} 块，来源文档: {doc_name}")
    return len(chunks)


def search(query: str, top_k: int = RETRIEVE_TOP_K) -> List[Dict[str, Any]]:
    """
    向量检索：用户问题 → Top K 最相关文本块

    流程：查询文本 → BGE-M3 编码 → Chroma 余弦相似度检索 → 返回 Top K

    参数：
    - query: 用户问题
    - top_k: 返回前 K 条（默认10，来自 settings.py）

    返回：[{"text": "块内容", "score": 0.85, "doc_name": "manual.md", "chunk_index": 2}, ...]
    """
    collection = get_collection()

    # 检查库中是否有数据
    if collection.count() == 0:
        print("[Store] 警告：向量库为空，请先执行入库操作")
        return []

    # 1. 编码查询
    query_vector = encode_query(query)

    # 2. Chroma 检索
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # 3. 整理结果格式
    # Chroma 返回格式：{"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
    # 嵌套列表是因为它支持批量查询，我们只查了1条，取 [0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # 距离越小 → 相似度越高；Chroma 默认用余弦距离 = 1 - 余弦相似度
    # 转换为相似度分数（0~1）
    search_results = []
    for i in range(len(docs)):
        similarity = 1 - distances[i]  # 距离 → 相似度
        search_results.append({
            "text": docs[i],
            "score": float(similarity),
            "doc_name": metas[i].get("doc_name", "unknown"),
            "chunk_index": metas[i].get("chunk_index", -1),
        })

    return search_results


def get_collection_count() -> int:
    """获取当前向量库中的文档数量"""
    collection = get_collection()
    return collection.count()


def clear_collection():
    """清空向量库（调试用，慎调）"""
    global _collection, _client
    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
        print("[Store] 已清空 Collection")


# ========== 运行测试 ==========
if __name__ == "__main__":
    from preprocess.parsing.parser import parse_document
    from preprocess.chunking.chunker import chunk_text
    from config.settings import DATA_DIR

    # 完整流水线测试：解析 → 切分 → 编码 → 入库 → 检索

    # 0. 清空旧数据（保证测试干净）
    clear_collection()

    # 1. 解析文档
    doc_path = DATA_DIR / "raw_docs" / "smart_office_manual.md"
    text = parse_document(doc_path)

    # 2. 切分
    chunks = chunk_text(text)

    # 3. 入库（内部自动调用 embedder 编码）
    add_chunks(chunks, doc_name="smart_office_manual")

    # 4. 验证：库中有多少条
    print(f"\n[测试] 向量库中文档数: {get_collection_count()}")

    # 5. 检索测试
    query = "请假审批流程是什么？"
    print(f"\n[测试] 检索查询: '{query}'")

    results = search(query, top_k=3)
    print(f"[测试] 返回 Top {len(results)} 条结果：\n")
    for i, r in enumerate(results):
        print(f"  [{i + 1}] 相似度={r['score']:.4f} | 来源={r['doc_name']} 块{r['chunk_index']}")
        print(f"      内容预览: {r['text'][:100]}...")
        print()
