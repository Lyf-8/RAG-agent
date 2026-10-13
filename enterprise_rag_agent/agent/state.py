"""
阶段3-步骤1：Agent 图状态定义

职责：定义 LangGraph 全局共享状态 AgentState ——
      图中每个节点读它、每个节点写回它的部分字段，
      LangGraph 负责把各节点的返回值"合并"进同一份状态流经全图。

知识点讲解：
1. 为什么用 TypedDict 而不是 Pydantic / dataclass？
   - LangGraph 的状态本质是"字典 + 合并规则"，TypedDict 是最轻量的表达：
     只做类型标注（IDE 提示用），运行时就是普通 dict，零开销
   - Pydantic 会做运行时校验和拷贝，对高频流转的图状态反而是负担；
     需要强校验时再升级也不迟（LangGraph 同样支持 Pydantic 模型做状态）

2. 状态合并（Reducer）规则：
   - 默认规则是"覆盖"：节点返回 {"question": 新值} → 状态里的 question 被替换
   - 若某字段需要"追加"而非覆盖（如消息列表），要显式声明 reducer：
       messages: Annotated[list, operator.add]
   - 本项目是线性+条件循环图，各字段都是"最新值覆盖"语义，无需自定义 reducer

3. 字段设计思路（对照 CRAG / Self-RAG 论文的思路做了简化）：
   - question / original_question 分开存：
     rewrite 节点会改写 question 用于检索，但 generate 生成答案时
     要对齐用户的"原始意图"，所以原始问题必须单独保留
   - documents（本轮召回的子块）与 relevant_docs（LLM 判定相关的子块）分开存：
     方便调试时对比"检索到了什么"和"LLM 留下了什么"，归因过滤环节
   - rewrite_count 是循环图的"退出条件计数器"，防止 LLM 一直不满意导致死循环
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
