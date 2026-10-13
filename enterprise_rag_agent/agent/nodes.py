"""
阶段3-步骤3：Agent 图节点实现

职责：实现图的四个节点 + 一个条件路由函数：
      retrieve（检索）→ grade_documents（LLM 判相关性）
      → [条件路由] → rewrite_query（改写重试）/ generate（生成答案）

知识点讲解：
1. 节点函数的统一签名：state 进 → 局部更新字典出
   - 入参：整个 AgentState（只读自己关心的字段）
   - 返回：只返回"本次要更新的字段"，LangGraph 会按覆盖规则合并进全局状态
   - 不要返回整个 state —— 只写自己改的字段，是最小职责原则在图编排里的体现

2. 检索节点直接复用阶段2的 retrieve_and_rerank_v2：
   - Agent 编排层不重复造检索轮子，这正是分阶段架构的收益：
     阶段2把检索质量做到 100%，阶段3只需要"编排调度"
   - 判相关性在子块级做（短文本、打分准），生成上下文用父块（信息完整），
     延续阶段2"子块定位、父块回答"的设计

3. LLM 懒加载 get_llm() 的意义：
   - 所有 LLM 调用都经 get_llm() 取实例，模块加载时不构造 ChatOpenAI
   - 好处1：测试可以 monkey-patch 这个函数注入"桩 LLM"，不花一分钱验证图逻辑
   - 好处2：Key 未配置时 import 本模块不报错，离线链路照常可用
"""
from typing import Any

from config.settings import AGENT_MAX_REWRITES, CREW_TRIGGER_KEYWORDS
from agent.prompts import GRADE_PROMPT, REWRITE_PROMPT
from agent.subagents.crew_team import run_crew
from qa_chain_v2 import retrieve_and_rerank_v2, build_context_v2
from qa_chain import PROMPT, get_llm


# ========== 节点1：检索 ==========
def retrieve(state: dict) -> dict[str, Any]:
    """调用阶段2 Hybrid 混合检索 + Reranker 精排（不依赖 LLM）"""
    question = state["question"]
    print(f"\n[节点 retrieve] 检索: {question}")
    documents = retrieve_and_rerank_v2(question)
    print(f"[节点 retrieve] 召回 {len(documents)} 个子块")
    # 顺手清空上一轮的判定结果，避免改写重试时残留旧判定
    return {"documents": documents, "relevant_docs": []}


# ========== 节点2：LLM 文档相关性过滤 ==========
def grade_documents(state: dict) -> dict[str, Any]:
    """
    用 LLM 逐个判断"子块是否包含回答问题所需信息"，过滤掉不相关的

    为什么需要这一步？向量检索/Reranker 是"文本相似"，
    相似 ≠ 真能回答问题（比如问题问流程A，检索回来的是流程B的介绍）。
    LLM 打分是语义级的"质检关卡"，也是 CRAG 论文的核心思想：
    检索结果可靠才直接生成，不可靠就走纠错分支（改写重试）。
    """
    question = state["question"]
    documents = state["documents"]
    print(f"[节点 grade_documents] LLM 判定 {len(documents)} 个子块的相关性 ...")

    llm = get_llm()
    relevant = []
    for i, doc in enumerate(documents, 1):
        response = llm.invoke(
            GRADE_PROMPT.format_messages(question=question, document=doc["text"])
        )
        verdict = response.content.strip().lower()
        is_relevant = "yes" in verdict
        print(f"    子块{i}: {'相关' if is_relevant else '不相关'}（LLM: {verdict[:20]}）")
        if is_relevant:
            relevant.append(doc)

    print(f"[节点 grade_documents] 过滤后保留 {len(relevant)} 个相关子块")
    return {"relevant_docs": relevant}


# ========== 条件路由：判相关性后走哪条边 ==========
def _is_report_task(question: str) -> bool:
    """启发式判断是否为"报告型任务"（对比/审核/写报告等需要多步骤创作的问题）

    用关键词匹配而非再调一次 LLM 分类：零成本、可解释、规则就在 settings 里可改。
    """
    return any(kw in question for kw in CREW_TRIGGER_KEYWORDS)


def decide_next(state: dict) -> str:
    """
    图的条件边：根据过滤结果决定下一步

    - 有相关文档 且 是报告型任务 → crew（CrewAI 多角色协作成稿，阶段4新增）
    - 有相关文档 且 普通问答   → generate（单 LLM 直接生成）
    - 无相关文档 且 改写次数未达上限 → rewrite（改写问题重试）
    - 无相关文档 且 已达上限 → generate（兜底：带空上下文生成，
      Prompt 中的规则会让 LLM 明确回答"根据现有资料无法回答"，避免死循环）
    """
    if state["relevant_docs"]:
        if _is_report_task(state["original_question"]):
            print("[路由 decide_next] 有相关文档，报告型任务 → crew_collaborate")
            return "crew"
        print("[路由 decide_next] 有相关文档，普通问答 → generate")
        return "generate"
    if state["rewrite_count"] < AGENT_MAX_REWRITES:
        print(f"[路由 decide_next] 无相关文档，改写次数 {state['rewrite_count']}/{AGENT_MAX_REWRITES} → rewrite_query")
        return "rewrite"
    print("[路由 decide_next] 重试已达上限 → generate（兜底回答）")
    return "generate"


# ========== 节点3：查询改写 ==========
def rewrite_query(state: dict) -> dict[str, Any]:
    """LLM 改写问题使其更利于检索，并累加改写计数"""
    question = state["question"]
    print(f"[节点 rewrite_query] 改写前: {question}")
    llm = get_llm()
    response = llm.invoke(REWRITE_PROMPT.format_messages(question=question))
    new_question = response.content.strip()
    print(f"[节点 rewrite_query] 改写后: {new_question}")
    return {"question": new_question, "rewrite_count": state["rewrite_count"] + 1}


# ========== 节点4：生成答案 ==========
def generate(state: dict) -> dict[str, Any]:
    """
    相关子块 → 回取父块上下文 → LLM 生成答案

    两个细节：
    - 用 original_question 而非（可能被改写过的）question 生成，
      保证答案对齐用户的原始意图
    - 上下文复用阶段2的 build_context_v2：子块命中 → 回取完整父块
    """
    # 用原始问题生成；相关子块为空时上下文为空串，Prompt 规则会让 LLM 坦诚"无法回答"
    context = build_context_v2(state["relevant_docs"]) if state["relevant_docs"] else "（无可用参考资料）"
    question = state["original_question"]

    print(f"[节点 generate] 组装上下文 {len(context)} 字符，LLM 生成答案 ...")
    llm = get_llm()
    response = llm.invoke(PROMPT.format_messages(context=context, question=question))
    print("[节点 generate] 答案生成完毕")
    return {"answer": response.content}


# ========== 节点5：CrewAI 多角色协作（阶段4-模式A）==========
def crew_collaborate(state: dict) -> dict[str, Any]:
    """
    报告型任务的成稿节点：把检索+质检后的资料交给 CrewAI 团队（研究→审核→写作）

    子Agent 集成原则的代码体现：
    - 本节点与 retrieve/generate 地位平等：state 进 → 局部更新字典出
    - run_crew 是纯函数，只接收字符串、返回字符串，接触不到图状态与路由
    - 拔掉本节点，主流程（retrieve → grade → generate）不受影响
    """
    context = build_context_v2(state["relevant_docs"])
    question = state["original_question"]

    print(f"[节点 crew_collaborate] 报告型任务，交给 CrewAI 团队（上下文 {len(context)} 字符）...")
    answer = run_crew(task=question, context=context)
    print("[节点 crew_collaborate] CrewAI 团队成稿完毕")
    return {"answer": answer}
