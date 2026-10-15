# 企业级 RAG + LangGraph Agent 智能问答系统

> 面向企业私有知识库的智能问答后端，解决大模型"不懂内部文档、易幻觉"的痛点。
> 检索链路可量化（18 题业务评估集 Recall@10 = 100%、MRR = 1.0），Agent 流程可追溯、可人工介入。

---

## 核心特性

| 能力 | 实现 |
|---|---|
| **双路混合检索** | 稠密（BGE-M3）+ 稀疏（BM25）→ RRF 倒数排名融合 → Cross-Encoder 精排 |
| **父子块切分** | 子块（200 字符）负责检索定位，命中后回取父块（600 字符）喂给 LLM |
| **CRAG 纠错编排** | LangGraph 实现：检索结果先经 LLM 质检，不相关则改写重试（≤2 次），超限兜底坦诚作答 |
| **多角色子 Agent** | CrewAI 研究员 → 审核员 → 写作者流水线，按关键词路由分流报告型任务 |
| **工程化封装** | FastAPI 接口 + SQLite 对话持久化 + 结构化异常 |
| **浏览器聊天页** | 单文件 `index.html`，同源部署规避跨域，零依赖 |

---

## Agent 架构（CRAG 纠错式 RAG）

```
START
  │
  ▼
retrieve       ← Hybrid 双路召回（稠密+稀疏）→ RRF 融合 → Reranker 精排
  │
  ▼
grade_documents  ← LLM 判定每个子块是否与问题相关
  │
  ▼
decide_next    ← 条件路由：
  ├─ 有相关文档 ──────────► generate ──────────────────────► END
  │                                          或（报告型任务）
  │                                       crew_collaborate ─► END
  └─ 无相关文档且未达上限 ─► rewrite_query ─► retrieve（循环）
                                       达上限 → generate（兜底坦诚"无法回答"）
```

**设计要点**：子 Agent 只作主图的一个节点（`crew_collaborate`），永不接管主流程——保证可控、可追溯、可人工干预。

---

## 检索链路

```
用户问题
  │
  ├── BGE-M3 稠密向量 → cosine 相似度 Top-K
  ├── BM25 稀疏向量   → 词频匹配 Top-K
  │
  ▼
RRF 倒数排名融合（k=60，仅看排名不看分数，免调参）
  │
  ▼
bge-reranker-v2-m3 Cross-Encoder 精排（问题+子块拼接打分，Top-4）
  │
  ▼
按 parent_id 去重，回取完整父块
  │
  ▼
组装带溯源标记的上下文 → 喂给 LLM
```

---

## 评估结果（18 题人工标注集，全部离线可跑）

| 指标 | 召回层（Hybrid+RRF） | 精排层（+Reranker Top4） |
|---|---|---|
| Recall@1 | 100% | 100% |
| Recall@4 | 100% | 100% |
| MRR | 1.0000 | 1.0000 |

**历史演进**：阶段1 纯稠密 + 800 字符大块 → Reranker 把 Top-1 从 58.3% 救到 100%；阶段2 切分 + 混合检索后，两层全满分，精排在当前语料上增量为 0——说明组件价值取决于上游短板。

**A/B 调优结论**：单变量对照实验覆盖候选池大小、精排窗口、是否启用精排等组合，当前参数（`top_k=10 / rerank_top_n=4 / RRF_K=60`）在题库上已是全满分状态，维持基线不变（等语料规模扩大后重跑调优）。

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.12+、有 CUDA GPU（CPU 可改 settings.py 的 device）
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入：

```bash
DASHSCOPE_API_KEY=你的Key    # 通义千问 DashScope OpenAI 兼容端点
```

`.env` 已在 `.gitignore`，绝不会入库。若账号欠费/不想用阿里云，改 `config/settings.py` 的三行即可切换供应商：
- `LLM_BASE_URL` → 任意 OpenAI 兼容端点
- `LLM_API_KEY` → 新 Key
- `LLM_MODEL` → 对应模型名（如 `glm-4-flash`、`deepseek-chat`）

### 3. 启动服务

```bash
cd enterprise_rag_agent
python -m api.main
```

| 访问点 | 说明 |
|---|---|
| http://127.0.0.1:8000/ui/ | 浏览器聊天界面（直接打字提问，多轮会话） |
| http://127.0.0.1:8000/docs | FastAPI Swagger 接口调试页 |
| http://127.0.0.1:8000/health | 探活：返回 LLM 就绪状态 + 向量库规模 |

### 4. 接口契约

**POST `/chat`**

```json
// Request
{
  "question": "请假审批流程是什么？",
  "session_id": "可选，首次由服务端分配"
}

// Response
{
  "session_id": "f56752cf...",
  "answer": "请假审批流程如下：1. 员工提交申请 2. 直属主管审批（24小时内）...",
  "cost": { "retrieve_ms": 820, "generate_ms": 3120 }
}
```

**GET `/history/{session_id}`** → 按时间正序返回会话全部历史（持久化在 SQLite）

**GET `/health`** → `{"status": "ok", "llm_ready": true, "qdrant": {"children": 20, "parents": 19}}`

### 5. 跑评估（离线、零 API 费用）

```bash
cd enterprise_rag_agent

# 业务评估集：Recall@K / MRR
python -m tests.phase5_eval

# A/B 调优：参数对照实验
python -m tests.phase5_ab_test

# Agent 桩测试（零 API 成本验证图路由）
python -m tests.phase3_test    # CRAG 纠错循环
python -m tests.phase4_test    # CrewAI 路由
python -m tests.phase5_api_test # API 接口契约
```

---

## 目录结构

```
enterprise_rag_agent/
├── api/
│   ├── main.py              # FastAPI 入口（/chat /health /history）
│   ├── conversation.py      # SQLite 对话持久化
│   └── static/index.html   # 浏览器聊天界面（单文件、同源部署）
├── agent/
│   ├── state.py             # AgentState TypedDict（图状态定义）
│   ├── prompts.py           # 三节点提示词（grade / rewrite / generate）
│   ├── nodes.py             # 所有业务节点（retrieve / grade / rewrite / generate / crew_collaborate + decide_next 路由）
│   ├── graph.py             # StateGraph 装配 + ask() 入口
│   └── subagents/crew_team.py  # CrewAI 研究员→审核员→写作者流水线
├── config/settings.py       # 全局参数（模型名 / 向量库路径 / 检索参数 / CrewAI 触发关键词）
├── preprocess/
│   ├── parsing/             # Markdown/Word/PDF 解析 + 结构化标题层级识别
│   ├── chunking/            # 递归切分 + 父子块切分（带标题路径增强）
│   ├── embedding/           # BGE-M3 双向量嵌入（dense + sparse）
│   └── storage/             # Qdrant 本地文件模式存储（子块 Collection + 父块 Collection）
├── qa_chain.py              # 阶段1 POC 基线检索链路（ChromaDB 单向量）
├── qa_chain_v2.py           # 阶段2 企业版检索链路（Hybrid + RRF + 精排 + 父块）
├── tests/
│   ├── eval_dataset.py      # 18 题人工标注评估集
│   ├── phase2_eval.py       # 阶段2 消融对比（POC vs 企业版）
│   ├── phase3_test.py       # 阶段3 CRAG 路由桩测试
│   ├── phase4_test.py       # 阶段4 CrewAI 路由桩测试
│   ├── phase5_eval.py       # 阶段5 业务量化评估
│   ├── phase5_ab_test.py    # A/B 单变量调优
│   └── phase5_api_test.py   # API 接口契约桩测试
└── requirements.txt
```

---

## 技术栈

| 层 | 选型 |
|---|---|
| 编排框架 | LangGraph 1.x（StateGraph + TypedDict 状态机） |
| LLM | DashScope 通义千问（走 OpenAI 兼容协议，便于切换 DeepSeek/GLM 等） |
| Embedding | BGE-M3（FlagEmbedding，输出 dense + sparse 双向量） |
| 精排 | bge-reranker-v2-m3（Cross-Encoder，512 token 输入上限） |
| 向量库 | Qdrant 本地文件模式（无需 Docker） |
| 稀疏检索 | BM25（rank-bm25） |
| 子 Agent | CrewAI（多角色 sequential 流水线） |
| API | FastAPI + Uvicorn |
| 持久化 | SQLite |
| 文档解析 | python-docx / pypdf |

---

## 开发阶段

| 阶段 | 提交 | 内容 |
|---|---|---|
| POC 基线 | `8f69337` | 递归切分 + 单向量 + 纯粗排，Top-1 58.3% |
| 企业版检索 | `1c85c35` | 父子块切分 + Hybrid 混合检索 + RRF + 精排，Top-1 100% |
| LangGraph CRAG | `efdb46e` | 4 节点 + 条件路由，检索质检→改写循环→兜底坦诚 |
| CrewAI 子 Agent | `a15c0a3` | 研究员/审核员/写作者流水线，关键词路由分流报告型任务 |
| 评估 + 工程化 | `5e202e0` | 18 题业务评估集 + A/B 调优 + FastAPI + SQLite 持久化 |
| 浏览器聊天页 | `af060aa` | 单文件 HTML 聊天界面，同源部署规避跨域 |
| 注释精简 | `4d67c3c` | 移除各模块 docstring 中教学性"知识点讲解"段落（纯删除 393 行） |

---

## 运行时注意事项

- **Qdrant 本地文件模式**是单进程独占的：启动 `api.main` 后，另起进程访问同一 `qdrant_db/` 目录会报 `already accessed by another instance`。并发访问需切到 Qdrant Server 模式（Docker 部署）。
- **LLM 懒加载单例**：`get_llm()` 首次调用时才构造，测试可 `monkey-patch` 注入桩实现、零 API 成本验证图路由。
- **PowerShell 5 编码坑**：`Invoke-RestMethod` 遇到没带 `charset` 的响应头时按 Latin-1 解码中文，浏览器和 Python 客户端不受影响。API 数据本身始终是对的。

---

## 后续可选方向

- 多租户 + 权限过滤（不同职级看不同文档）
- 增量文档更新（新文档入库无需重建全量索引）
- 流式输出（SSE / WebSocket）
- 引用溯源在答案中高亮
- Qdrant Server 模式（Docker 部署，支持并发）
- AutoGen 代码沙箱（此前按用户选择跳过的模式 B）
