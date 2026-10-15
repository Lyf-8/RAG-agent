"""
阶段2-步骤3-A：BGE-M3 双路向量化（稠密 + 稀疏）

职责：用同一个 BGE-M3 模型，同时产出两种向量：
- 稠密向量（Dense）：1024 维浮点数，表达"语义相似"
- 稀疏向量（Sparse）：词表维度（25万+）的权重，表达"字面匹配"
"""
from typing import List, Dict, Tuple
import numpy as np

from config.settings import EMBEDDING_LOCAL_PATH, EMBEDDING_DEVICE

# 模型单例（加载约10秒，全局只加载一次）
_model = None


def get_model():
    """获取 BGEM3FlagModel 实例（单例）"""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        print(f"[HybridEmbedder] 正在加载 BGE-M3（本地路径，输出 dense+sparse）...")
        _model = BGEM3FlagModel(
            EMBEDDING_LOCAL_PATH,
            use_fp16=True,          # GPU 上半精度推理：速度翻倍、显存减半，精度损失可忽略
            device=EMBEDDING_DEVICE,
        )
        print("[HybridEmbedder] BGE-M3 加载完成")
    return _model


def encode_hybrid(texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[Dict[int, float]]]:
    """
    文本列表 → (稠密向量矩阵, 稀疏向量列表)

    返回：
    - dense_vecs: numpy 数组 (N, 1024)，已 L2 归一化
    - sparse_vecs: [{token_id: weight, ...}, ...]，每个文本一个稀疏字典
    """
    model = get_model()
    print(f"[HybridEmbedder] 正在双路编码 {len(texts)} 段文本...")
    outputs = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # multi-vector 本阶段不用，关闭省显存
    )
    dense_vecs = outputs["dense_vecs"]
    # FlagEmbedding 输出的 lexical_weights 就是 {token_id: weight} 字典列表
    sparse_vecs = outputs["lexical_weights"]
    print(f"[HybridEmbedder] 编码完成：dense {dense_vecs.shape}，"
          f"sparse 平均非零维数 {np.mean([len(s) for s in sparse_vecs]):.0f}")
    return dense_vecs, sparse_vecs


def encode_query_hybrid(query: str) -> Tuple[np.ndarray, Dict[int, float]]:
    """单个查询 → (稠密向量, 稀疏向量)，检索专用"""
    dense, sparse = encode_hybrid([query])
    return dense[0], sparse[0]


# ========== 运行测试 ==========
if __name__ == "__main__":
    texts = [
        "请假3天以上需要分管副总审批",
        "采购金额20000元以上需总经理审批",
    ]
    dense, sparse = encode_hybrid(texts)

    print("\n" + "=" * 70)
    print(f"稠密向量形状: {dense.shape}（每行 1024 维）")
    print(f"第1条稀疏向量非零项数: {len(sparse[0])}")
    print(f"第1条稀疏向量 Top5 权重项: {sorted(sparse[0].items(), key=lambda x: -x[1])[:5]}")

    # 直观验证：稠密 vs 稀疏 的差异
    # 两句话字面完全不同，稠密相似度应较低
    sim = float(np.dot(dense[0], dense[1]))
    print(f"\n两句稠密余弦相似度: {sim:.4f}（语义不同，应较低）")

    # 稀疏交集：两句没有共同 token，交集应接近空
    common = set(sparse[0]) & set(sparse[1])
    print(f"两句稀疏向量共同 token 数: {len(common)}（字面不重叠，应很少）")
    print("=" * 70)
    print("结论：同一份模型权重，产出两种互补的检索信号")
