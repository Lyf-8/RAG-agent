"""
阶段4-模式A：CrewAI 多角色协作子Agent

职责：把 CrewAI 多角色团队封装成一个纯函数 run_crew(task, context) -> str，
      供 LangGraph 的 crew_collaborate 节点调用。

角色流水线（Process.sequential 顺序执行）：
    研究员 researcher  → 从检索到的资料中整理与任务相关的要点
    审核员 reviewer    → 逐条核对要点是否有资料依据，剔除无依据内容（RAG 防幻觉第二道关）
    写作者 writer      → 把审核后的要点写成结构清晰的最终回答

知识点讲解：
1. 为什么多角色比单 LLM 一次成稿更好？
   - 单 LLM 一次生成要同时兼顾"找要点 + 查事实 + 写文风"，注意力被摊薄，
     长上下文下容易顾此失彼（典型症状：编造资料里没有的细节）
   - 拆成"整理 → 审核 → 成稿"三步，每个角色只做一件事、Prompt 更聚焦，
     且审核员的"对抗性核对"是专门防幻觉的设计 —— 这是 Multi-Agent 的核心价值：
     用角色分工换取单点质量，而不是堆一个更大的模型

2. 手册核心禁忌的落地（"禁止子Agent做主框架"）：
   - 本模块只暴露一个纯函数 run_crew()：输入 task+context 两个字符串，输出一个字符串
   - CrewAI 团队不知道 LangGraph 的存在，拿不到图状态、改不了路由字段
   - LangGraph 主图把它当普通节点调用，拔掉本模块主流程（retrieve→grade→generate）原样能跑
   - 这就是"子Agent 只作子节点"：主流程可控、可追溯、可人工干预

3. CrewAI 的 LLM 接入：
   - CrewAI 内部走 LiteLLM 协议，OpenAI 兼容端点用 "openai/<模型名>" 前缀声明，
     配合 base_url 即可接入 DashScope 兼容模式，与主链路共用同一个 Key
   - crewai 包较重，采用懒加载：只有真实调用 run_crew 时才 import，
     未安装 crewai 时本模块可正常被 import（桩测试不受影响）
"""
from config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


def run_crew(task: str, context: str) -> str:
    """
    运行 CrewAI 多角色团队，产出最终回答

    :param task:    用户的报告型任务（原始问题，如"对比正式员工与实习生的考勤差异"）
    :param context: 检索+质检后的父块上下文（唯一可信资料来源）
    :return:        写作者产出的最终回答文本
    """
    # 懒加载：真实调用时才 import crewai（包重、依赖多，import 阶段不加载）
    from crewai import Agent, Task, Crew, Process, LLM

    # OpenAI 兼容协议接入 DashScope；与主链路共用同一个 Key/端点/模型
    llm = LLM(
        model=f"openai/{LLM_MODEL}",
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )

    # ========== 角色定义 ==========
    researcher = Agent(
        role="企业资料研究员",
        goal="从给定资料中准确找出与任务相关的全部要点，不遗漏、不发挥",
        backstory="你擅长从制度文档中快速定位关键信息，只摘录资料中真实存在的内容。",
        llm=llm,
        verbose=False,
    )
    reviewer = Agent(
        role="合规审核员",
        goal="逐条核对要点是否有资料原文依据，剔除任何无依据、臆测或资料外的内容",
        backstory="你是企业制度审核专家，对'资料里没有的话'零容忍，宁缺毋滥。",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="报告写作者",
        goal="把审核后的要点写成结构清晰、分点明确的最终回答，语言简洁专业",
        backstory="你擅长把零散要点组织成易读的中文说明，分点作答、忠于事实。",
        llm=llm,
        verbose=False,
    )

    # ========== 任务定义（顺序流水线：研究 → 审核 → 成稿）==========
    task_research = Task(
        description=(
            f"【用户任务】\n{task}\n\n【可信资料（唯一信息来源）】\n{context}\n\n"
            "请从资料中整理出与任务相关的全部要点，逐条列出；资料没有的内容不要写。"
        ),
        expected_output="按条列出的要点清单，每条标注对应资料片段编号",
        agent=researcher,
    )
    task_review = Task(
        description=(
            f"对照以下【可信资料】，逐条审核研究员整理的要点：\n{context}\n\n"
            "剔除所有无资料依据的条目，保留条目原样输出；全部无依据时如实说明。"
        ),
        expected_output="审核后的要点清单（仅保留有资料依据的条目）",
        agent=reviewer,
    )
    task_write = Task(
        description=(
            f"基于审核后的要点，针对用户任务【{task}】写出最终回答。"
            "要求：分点作答、条理清晰、忠于资料，不添加资料外的信息。"
        ),
        expected_output="结构清晰的中文最终回答",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, reviewer, writer],
        tasks=[task_research, task_review, task_write],
        process=Process.sequential,   # 顺序流水线：上一个角色的输出是下一个的输入
        verbose=False,
    )

    result = crew.kickoff()
    return str(result)
