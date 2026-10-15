"""
阶段3-步骤2：Agent 节点提示词模板

职责：集中管理三个 LLM 节点的 Prompt（打分 / 改写 / 生成），
      Prompt 与节点逻辑分离 —— 调优话术只改这里，不碰图结构代码。
"""
from langchain_core.prompts import ChatPromptTemplate

# ========== 节点1：文档相关性打分（grade_documents 用）==========
# 设计要点：只输出 yes/no，附 few-shot 示例钉死判断标准
GRADE_SYSTEM_PROMPT = """你是一个文档相关性判断专家。
给定【用户问题】和【文档片段】，判断该片段是否包含回答问题所需的信息。
判断标准：
- 片段内容与问题直接相关，或包含回答问题必需的关键词/事实 → yes
- 片段只是主题沾边但不含回答所需的具体信息 → no
- 只输出 yes 或 no，不要输出任何其他内容

示例：
用户问题：年假有几天？
片段：员工入职满一年可享受10天年假。 → yes
用户问题：年假有几天？
片段：公司年会的举办时间是每年12月。 → no"""

GRADE_HUMAN_PROMPT = """【用户问题】
{question}

【文档片段】
{document}

请判断该片段是否相关（只输出 yes 或 no）："""

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GRADE_SYSTEM_PROMPT),
    ("human", GRADE_HUMAN_PROMPT),
])

# ========== 节点2：查询改写（rewrite_query 用）==========
REWRITE_SYSTEM_PROMPT = """你是一个检索查询优化专家。
用户的问题在知识库中检索效果不佳，请你改写这个问题以提升检索命中率和准确性。
改写要求：
1. 保持用户的原始意图不变，不要回答问题；
2. 把口语化、有指代不明的表达改写成具体、明确的表述；
3. 补全可能的关键词（如把"那个流程"改成具体流程名）；
4. 只输出改写后的问题本身，不要解释。"""

REWRITE_HUMAN_PROMPT = """【原始问题】
{question}

请输出改写后的问题："""

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM_PROMPT),
    ("human", REWRITE_HUMAN_PROMPT),
])
