"""
阶段3-步骤1：Agent 图状态定义

职责：定义 LangGraph 全局共享状态 AgentState ——
      图中每个节点读它、每个节点写回它的部分字段，
      LangGraph 负责把各节点的返回值"合并"进同一份状态流经全图。
"""
from typing import Any, TypedDict


class AgentState(TypedDict):
    """LangGraph 全局状态：所有节点共享读写"""

    question: str                  # 当前问题（初始=原始问题，可能被 rewrite 改写）
    original_question: str         # 用户原始问题（generate 时对齐用户意图）
    documents: list[dict[str, Any]]       # 本轮检索召回的子块（未过滤）
    relevant_docs: list[dict[str, Any]]   # LLM 判定"与问题相关"的子块
    rewrite_count: int             # 已改写次数（循环退出条件）
    answer: str                    # 最终生成的答案
