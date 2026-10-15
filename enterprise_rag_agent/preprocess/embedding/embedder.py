"""
步骤3-A：BGE-M3 文本向量化（Embedder）

职责：将文本块 → 稠密向量（1024维），供 Chroma 向量库存储和检索
"""
from typing import List
import numpy as np

from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_LOCAL_PATH

# 模型实例缓存：全局单例，避免重复加载（加载一次约需10秒）
_model = None


def get_model():
    """
    获取 BGE-M3 模型实例（单例模式）

    模型已通过 ModelScope 下载到本地，直接从本地路径加载（无需联网）
    """
    global _model
    if _model is None:
        print(f"[Embedder] 正在加载模型: {EMBEDDING_MODEL}")
        print(f"[Embedder] 本地路径: {EMBEDDING_LOCAL_PATH}")
        print(f"[Embedder] 设备: {EMBEDDING_DEVICE}（首次加载需10-30秒）...")

        from sentence_transformers import SentenceTransformer
        # 优先用本地路径加载，避免联网下载
        model_path = EMBEDDING_LOCAL_PATH if EMBEDDING_LOCAL_PATH else EMBEDDING_MODEL
        _model = SentenceTransformer(
            model_path,
            device=EMBEDDING_DEVICE,
        )
        print(f"[Embedder] 模型加载完成，向量维度: {_model.get_sentence_embedding_dimension()}")
    return _model


def encode(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    将文本列表 → 向量矩阵

    参数：
    - texts: 文本块列表，如 ["块1内容", "块2内容", ...]
    - batch_size: 批量编码大小，32是显存和速度的平衡点

    返回：
    - numpy 数组，形状 (块数, 1024)，每行是一个块的向量

    使用方式：
        vectors = encode(["你好", "hello"])
        print(vectors.shape)  # (2, 1024)
    """
    model = get_model()
    print(f"[Embedder] 正在编码 {len(texts)} 段文本...")

    # convert_to_numpy=True 直接返回 numpy 数组，无需手动转换
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # 归一化：向量长度=1，余弦相似度=点积
    )
    print(f"[Embedder] 编码完成，向量矩阵形状: {vectors.shape}")
    return vectors


def encode_query(query: str) -> np.ndarray:
    """
    将单个查询文本 → 向量（用于检索时编码用户问题）

    和 encode() 的区别：输入是单个字符串，输出是单个向量
    """
    model = get_model()
    # 注意：encode 接受列表，这里包一层
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0]  # 返回单个向量 (1024,)


# ========== 运行测试 ==========
if __name__ == "__main__":
    # 测试：编码两个句子，看向量形状和相似度
    texts = [
        "智能办公系统支持GPS定位打卡和WiFi打卡",
        "员工到达办公区域后可以完成打卡",
        "采购审批需要根据金额分级审批",
    ]

    vectors = encode(texts)

    print("\n" + "=" * 60)
    print(f"测试文本数: {len(texts)}")
    print(f"向量矩阵形状: {vectors.shape}")
    print(f"向量维度: {vectors.shape[1]}")

    # 计算两两余弦相似度（已归一化，点积=余弦相似度）
    print("\n余弦相似度矩阵：")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = float(np.dot(vectors[i], vectors[j]))
            print(f"  [{i}] vs [{j}]: {sim:.4f}  | {texts[i][:15]}... vs {texts[j][:15]}...")
    print("=" * 60)
    print("结论：相似度越高（接近1），说明语义越相近")
