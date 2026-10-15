"""
阶段3-步骤4：LangGraph 图编排

职责：把节点装配成一张有条件的"流程图"（StateGraph），编译后对外提供 Agent 实例。

图结构（CRAG 风格：检索质量把关 + 不可靠时纠错重试）：

    START
      │
      ▼
   retrieve（Hybrid 检索 + 精排）
      │
      ▼
   grade_documents（LLM 判相关性）
      │
      ▼
   decide_next（条件路由）
      ├─ 有相关文档 ──────────► generate ─► END
      └─ 无相关文档且未达上限 ─► rewrite_query ─► retrieve（循环）
      （达上限时兜底走 generate，Prompt 让 LLM 坦诚"无法回答"）
"""
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent import nodes

# 图的节点名常量（集中定义，避免字符串散落各处写错）
NODE_RETRIEVE = "retrieve"
NODE_GRADE = "grade_documents"
NODE_REWRITE = "rewrite_query"
NODE_GENERATE = "generate"
NODE_CREW = "crew_collaborate"   # 阶段4：CrewAI 多角色子Agent（仅作子节点）


def build_graph():
    """组装并编译 Agent 图（返回可直接 invoke 的 CompiledGraph）"""
    builder = StateGraph(AgentState)

    # 1) 注册节点：名字 → 处理函数
    builder.add_node(NODE_RETRIEVE, nodes.retrieve)
    builder.add_node(NODE_GRADE, nodes.grade_documents)
    builder.add_node(NODE_REWRITE, nodes.rewrite_query)
    builder.add_node(NODE_GENERATE, nodes.generate)
    builder.add_node(NODE_CREW, nodes.crew_collaborate)

    # 2) 固定边：START → retrieve → grade
    builder.add_edge(START, NODE_RETRIEVE)
    builder.add_edge(NODE_RETRIEVE, NODE_GRADE)

    # 3) 条件边：grade 之后由 decide_next 决定去向
    #    rewrite=改写重试；generate=普通问答成稿；crew=报告型任务交给CrewAI团队
    builder.add_conditional_edges(
        NODE_GRADE,
        nodes.decide_next,
        {"rewrite": NODE_REWRITE, "generate": NODE_GENERATE, "crew": NODE_CREW},
    )

    # 4) 循环边：改写后重新检索；两个成稿节点都直接结束
    builder.add_edge(NODE_REWRITE, NODE_RETRIEVE)
    builder.add_edge(NODE_GENERATE, END)
    builder.add_edge(NODE_CREW, END)

    return builder.compile()


# ========== 模块级单例：对外统一入口 ==========
_agent_app = None


def get_agent_app():
    """获取编译好的 Agent 图（单例，懒加载）"""
    global _agent_app
    if _agent_app is None:
        _agent_app = build_graph()
    return _agent_app


def ask(question: str) -> str:
    """对外问答接口：输入问题，返回 Agent 最终答案"""
    init_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "relevant_docs": [],
        "rewrite_count": 0,
        "answer": "",
    }
    final_state = get_agent_app().invoke(init_state)
    return final_state["answer"]


# ========== 运行：打印图结构 ==========
if __name__ == "__main__":
    app = build_graph()
    print("Agent 图的 Mermaid 结构（可粘贴到 mermaid.live 渲染）：\n")
    print(app.get_graph().draw_mermaid())
    print("\n说明：图结构构建成功。完整问答需 DashScope 可用，图路由测试见 tests/phase3_test.py")
