"""
阶段4-步骤5：CrewAI 子Agent 路由测试（桩版）

目的：不调用真实 LLM、不运行真实 CrewAI 团队（DashScope 欠费也能跑），
      验证"报告型任务 → crew_collaborate 子节点"的新路由接入正确。

桩注入点（两处）：
    1. agent.nodes.get_llm   → FakeLLM（相关性打分恒返回 yes/no）
    2. agent.nodes.run_crew  → 假团队函数（直接返回预设报告文本）
  run_crew 是唯一接触 CrewAI 的函数且为纯函数，替换它即可完全隔离外部框架 ——
  这正是"子Agent 封装成纯函数"带来的可测试性红利。

三个场景覆盖阶段4全部新增/原有路由：
    场景A：报告型问题 + 有相关文档 → crew 节点（crew 被调用、generate 未被调用）
    场景B：普通问答   + 有相关文档 → generate 节点（crew 未被调用）
    场景C：报告型问题 + 无相关文档 → 仍走 rewrite 循环（crew 不抢主流程的路）

运行方式（项目根目录下）：
    python -m tests.phase4_test
"""
import sys
from typing import Any

import agent.nodes as nodes
from agent.graph import build_graph
from config.settings import DATA_DIR


# ========== 桩 LLM：替代 ChatOpenAI ==========
class FakeLLM:
    def __init__(self, grade_verdict: str = "yes", fake_answer: str = "这是桩LLM生成的模拟答案。"):
        self.grade_verdict = grade_verdict
        self.fake_answer = fake_answer
        self.grade_calls = 0
        self.rewrite_calls = 0
        self.generate_calls = 0

    def invoke(self, messages: list) -> Any:
        text = "\n".join(getattr(m, "content", str(m)) for m in messages)
        if "判断该片段是否相关" in text:
            self.grade_calls += 1
            return _Msg(self.grade_verdict)
        if "改写后的问题" in text:
            self.rewrite_calls += 1
            return _Msg("改写后的问题")
        if "参考资料" in text:
            self.generate_calls += 1
            return _Msg(self.fake_answer)
        raise ValueError(f"FakeLLM 收到无法识别的调用。前100字符: {text[:100]}")


class _Msg:
    def __init__(self, content: str):
        self.content = content


# ========== 桩 CrewAI 团队：替代 run_crew ==========
class FakeCrew:
    """记录调用并返回预设报告文本"""
    def __init__(self):
        self.calls = 0
        self.last_task = None
        self.last_context = None

    def __call__(self, task: str, context: str) -> str:
        self.calls += 1
        self.last_task = task
        self.last_context = context
        return "这是桩CrewAI团队生成的模拟报告。"


def _init_state(question: str) -> dict:
    return {
        "question": question,
        "original_question": question,
        "documents": [], "relevant_docs": [], "rewrite_count": 0, "answer": "",
    }


# ========== 场景A：报告型问题 + 有相关文档 → crew ==========
def test_case_a():
    print("\n" + "=" * 70)
    print("【场景A】报告型问题 + 有相关文档 → 预期：走 crew_collaborate，不走 generate")
    print("=" * 70)

    fake_llm = FakeLLM(grade_verdict="yes")
    fake_crew = FakeCrew()
    nodes.get_llm = lambda: fake_llm
    nodes.run_crew = fake_crew                      # ← 隔离真实 CrewAI
    app = build_graph()

    question = "对比正式员工和实习生的考勤制度差异，写一份说明报告"
    answer = app.invoke(_init_state(question))

    assert fake_crew.calls == 1, "CrewAI 团队应恰好被调用 1 次"
    assert fake_llm.generate_calls == 0, "报告型任务不应走 generate"
    assert fake_llm.rewrite_calls == 0, "有相关文档时不应触发改写"
    assert answer["answer"] == "这是桩CrewAI团队生成的模拟报告。", "答案应来自桩 Crew"
    assert fake_crew.last_task == question, "交给 Crew 的任务应是用户原始问题"
    assert len(fake_crew.last_context) > 50, "交给 Crew 的上下文应是真实的父块内容"
    print(f"\n[断言通过] 场景A：crew 调用 {fake_crew.calls} 次，generate 调用 {fake_llm.generate_calls} 次")
    print(f"[Crew收到的任务] {fake_crew.last_task}")
    print(f"[Crew收到的上下文前80字] {fake_crew.last_context[:80]}...")


# ========== 场景B：普通问答 + 有相关文档 → generate ==========
def test_case_b():
    print("\n" + "=" * 70)
    print("【场景B】普通问答 + 有相关文档 → 预期：走 generate，crew 不被调用")
    print("=" * 70)

    fake_llm = FakeLLM(grade_verdict="yes")
    fake_crew = FakeCrew()
    nodes.get_llm = lambda: fake_llm
    nodes.run_crew = fake_crew
    app = build_graph()

    answer = app.invoke(_init_state("GPS定位打卡的精度是多少？"))

    assert fake_crew.calls == 0, "普通问答不应走 crew"
    assert fake_llm.generate_calls == 1, "生成应恰好执行 1 次"
    assert answer["answer"] == fake_llm.fake_answer, "答案应来自桩 LLM"
    print(f"\n[断言通过] 场景B：crew 调用 {fake_crew.calls} 次，generate 调用 {fake_llm.generate_calls} 次")


# ========== 场景C：报告型问题 + 无相关文档 → 仍走 rewrite 循环 ==========
def test_case_c():
    print("\n" + "=" * 70)
    print("【场景C】报告型问题 + 无相关文档 → 预期：rewrite 循环至上限，crew 不抢主流程")
    print("=" * 70)

    fake_llm = FakeLLM(grade_verdict="no")
    fake_crew = FakeCrew()
    nodes.get_llm = lambda: fake_llm
    nodes.run_crew = fake_crew
    app = build_graph()

    answer = app.invoke(_init_state("写一份不存在的制度报告"))

    assert fake_crew.calls == 0, "无相关文档时 crew 不应被调用（资料是空的，交给团队也是幻觉）"
    assert fake_llm.rewrite_calls == 2, "改写应重试至上限"
    assert fake_llm.generate_calls == 1, "兜底仍走 generate"
    assert len(answer["relevant_docs"]) == 0, "全程无相关文档"
    print(f"\n[断言通过] 场景C：改写 {fake_llm.rewrite_calls}/2 次 → 兜底 generate，crew 调用 {fake_crew.calls} 次")


# ========== 主入口 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("阶段4测试：CrewAI 子Agent 路由验证（桩版，零 API 费用）")
    print("=" * 70)

    app = build_graph()
    print("\n[图结构]\n" + app.get_graph().draw_mermaid())

    # 确保向量库有数据（检索节点真实运行）
    from preprocess.storage.qdrant_store import get_counts
    counts = get_counts()
    print(f"\n[准备] Qdrant 数据：子块 {counts['children']} / 父块 {counts['parents']}")
    if counts["children"] == 0:
        print("[准备] Qdrant 为空，重建并入库 ...")
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

    test_case_a()
    test_case_b()
    test_case_c()

    print("\n" + "=" * 70)
    print("阶段4全部测试通过 ✓  报告型走crew、问答型走generate、无资料仍走rewrite循环")
    print("说明：真实 CrewAI 团队运行需 DashScope 充值 + pip install crewai 后调用 agent.graph.ask()")
    print("=" * 70)
    sys.exit(0)
