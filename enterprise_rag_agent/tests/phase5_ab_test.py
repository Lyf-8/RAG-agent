"""
阶段5-步骤2：A/B 调优测试（单变量对照实验，迭代最优基线参数）

职责：用同一套评估题库（tests/eval_dataset.py），对检索链路的关键参数做对照实验，
     找出"效果不打折、成本最低"的最优基线。全程离线，零 API 费用。
"""
import contextlib
import io

from preprocess.storage.qdrant_store import search_dense, search_sparse
from qa_chain import get_reranker
from qa_chain_v2 import rrf_fuse, build_context_v2
from tests.eval_dataset import EVAL_DATASET
# 复用步骤1的指标函数与数据自检，保证口径完全一致
from tests.phase5_eval import first_relevant_rank, recall_at_k, mrr, ensure_data_ready

BASELINE = {"recall": "hybrid", "top_k": 10, "top_n": 4}   # 当前 settings 基线


# ========== 统一实验执行器：按配置跑完整个题库 ==========
def run_config(recall: str, top_k: int, top_n: int | None) -> list[tuple[str, str, list[dict]]]:
    """
    对 18 题逐题执行指定配置，返回 [(问题, 关键词, 结果列表), ...]
    top_n=None 表示不精排（考察召回层）；否则精排后截取 Top N（完整链路）
    日志全部静默，避免 144 次运行的打印淹没实验报告
    """
    per_question = []
    for question, keyword in EVAL_DATASET:
        with contextlib.redirect_stdout(io.StringIO()):
            # --- 召回层 ---
            if recall == "dense":
                cands = search_dense(question, top_k=top_k)
            elif recall == "sparse":
                cands = search_sparse(question, top_k=top_k)
            else:  # hybrid：双路 + RRF
                cands = rrf_fuse(
                    search_dense(question, top_k=top_k),
                    search_sparse(question, top_k=top_k),
                )
            # --- 排序层（可选）---
            if top_n is not None and cands:
                reranker = get_reranker()
                scores = reranker.predict([(question, c["text"]) for c in cands])
                for c, s in zip(cands, scores):
                    c["rerank_score"] = float(s)
                cands.sort(key=lambda x: x["rerank_score"], reverse=True)
                cands = cands[:top_n]
        per_question.append((question, keyword, cands))
    return per_question


# ========== 两种报告器：召回层 / 完整链路 ==========
def report_recall_layer(name: str, runs) -> list[int]:
    """召回层指标：Recall@1/4/10 + MRR + 平均候选数（精排成本代理）"""
    ranks = [first_relevant_rank(c, kw) for _, kw, c in runs]
    avg_cand = sum(len(c) for _, _, c in runs) / len(runs)
    print(f"  {name:<24} Recall@1={recall_at_k(ranks, 1):6.1%} "
          f"Recall@4={recall_at_k(ranks, 4):6.1%} "
          f"Recall@10={recall_at_k(ranks, 10):6.1%} "
          f"MRR={mrr(ranks):.4f}  平均候选={avg_cand:.1f}")
    return ranks


def report_full_chain(name: str, runs) -> list[int]:
    """完整链路指标：Recall@1 + MRR + 平均上下文字符数（LLM token 成本代理）"""
    ranks = [first_relevant_rank(c, kw) for _, kw, c in runs]
    avg_ctx = sum(len(build_context_v2(c)) for _, _, c in runs) / len(runs)
    print(f"  {name:<24} Recall@1={recall_at_k(ranks, 1):6.1%} "
          f"MRR={mrr(ranks):.4f}  平均上下文={avg_ctx:.0f}字符")
    return ranks


def main():
    ensure_data_ready()
    n = len(EVAL_DATASET)
    print(f"[A/B] 题库 {n} 题，基线 = {BASELINE}，单变量对照\n")

    # ---------- E1：召回策略（top_k=10，不精排）----------
    print("=" * 84)
    print("【E1 召回策略】稠密 vs 稀疏 vs Hybrid —— 单路够吗？（top_k=10，不精排）")
    print("=" * 84)
    e1_dense = run_config("dense", 10, None)
    e1_sparse = run_config("sparse", 10, None)
    e1_hybrid = run_config("hybrid", 10, None)
    report_recall_layer("稠密路单路（语义）", e1_dense)
    report_recall_layer("稀疏路单路（字面）", e1_sparse)
    ranks_e1_hybrid = report_recall_layer("Hybrid+RRF（基线）", e1_hybrid)

    # ---------- E2：候选池 top_k（Hybrid，不精排）----------
    print("\n" + "=" * 84)
    print("【E2 候选池大小】每路 top_k = 5 / 10 / 15 —— 池子多大才够？（Hybrid，不精排）")
    print("=" * 84)
    e2_k5 = run_config("hybrid", 5, None)
    report_recall_layer("top_k=5", e2_k5)
    print(f"  {'top_k=10（基线，同E1）':<24} （复用 E1 Hybrid 行，避免重复实验）")
    e2_k15 = run_config("hybrid", 15, None)
    report_recall_layer("top_k=15", e2_k15)

    # ---------- E3：精排窗口 top_n（完整链路）----------
    print("\n" + "=" * 84)
    print("【E3 精排窗口】top_n = 3 / 4 / 5 —— 给 LLM 多大窗口？（Hybrid@10 + 精排）")
    print("=" * 84)
    e3 = {}
    for top_n in (3, 4, 5):
        e3[top_n] = run_config("hybrid", 10, top_n)
        tag = "（基线）" if top_n == BASELINE["top_n"] else ""
        report_full_chain(f"top_n={top_n}{tag}", e3[top_n])

    # ---------- E4：是否精排（派生对比，零额外运行）----------
    print("\n" + "=" * 84)
    print("【E4 精排的价值】RRF 直接取 Top4 vs Reranker 精排 Top4")
    print("=" * 84)
    rrf_top4 = [(q, kw, c[:4]) for q, kw, c in e1_hybrid]   # 从 E1 派生，不重跑
    report_full_chain("RRF 直接 Top4", rrf_top4)
    print(f"  {'精排 Top4（基线，同E3）':<24} （复用 E3 top_n=4 行）")
    # MRR 对比：RRF Top4 的正解排名 vs 精排后的正解排名
    ranks_rrf = [first_relevant_rank(c, kw) for _, kw, c in rrf_top4]
    ranks_rr = [first_relevant_rank(c, kw) for _, kw, c in e3[4]]
    worse = sum(1 for a, b in zip(ranks_rrf, ranks_rr) if b < a)
    print(f"  → 精排把 {worse}/{n} 题的正解排名前提了（MRR: {mrr(ranks_rrf):.4f} → {mrr(ranks_rr):.4f}）")

    # ---------- 实验结论模板 ----------
    print("\n" + "=" * 84)
    print("【结论】对照基线逐组解读：")
    print("  E1：若单路 Recall@10 低于 Hybrid → 混合检索不是摆设，必须保留；")
    print("  E2：若 top_k=5 与 10 指标持平 → 基线可降至 5，精排成本减半；")
    print("  E3：若 top_n=3 仍满分 → 窗口可收窄，每次问答省一个父块的 token；")
    print("  E4：MRR 差值即精排贡献，接近 0 说明 RRF 排序已够好（本题库特征）。")


if __name__ == "__main__":
    main()
