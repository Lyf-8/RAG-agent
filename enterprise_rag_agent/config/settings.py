"""
全局配置文件 —— 项目唯一的参数入口
所有阶段共享此配置，改参数只改这里，符合工程化规范
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 自动读取项目根目录的 .env 文件（Key 不写死在代码里，也不泄露到 GitHub）
# .env 文件位置：e:\ai agent study\RAG-agent\.env
_DOTENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
# override=True：让 .env 的值覆盖系统环境变量，避免残留的旧值（如模板占位符）干扰
load_dotenv(_DOTENV_PATH, override=True)

# ========== HuggingFace 镜像配置 ==========
# 国内访问 HuggingFace 需走镜像，否则模型下载会超时
# 官方镜像：https://hf-mirror.com
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ========== 模型本地路径（ModelScope 下载，国内速度快）==========
# 模型已通过 ModelScope 下载到本地，避免每次从 HuggingFace 加载超时
# 如果通过 ModelScope 重新下载，路径会自动更新
_MODELSCOPE_BASE = Path(os.path.expanduser("~")) / ".cache" / "modelscope" / "models"
EMBEDDING_LOCAL_PATH = str(_MODELSCOPE_BASE / "BAAI--bge-m3" / "snapshots" / "master")
RERANKER_LOCAL_PATH = str(_MODELSCOPE_BASE / "BAAI--bge-reranker-v2-m3" / "snapshots" / "master")

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录 enterprise_rag_agent/
DATA_DIR = BASE_DIR / "data"                        # 原始企业文档存放处
CHROMA_DIR = BASE_DIR / "chroma_db"                 # POC向量库本地持久化目录（首次运行自动创建）

# ========== LLM 配置（大脑：语义理解/判断/生成/校验）==========
# 通义千问 DashScope，走 OpenAI 兼容协议 —— 好处：一套代码可无缝换 DeepSeek/OpenAI
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")    # 从环境变量读取，Key 绝不写死在代码里
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"  # 主力模型；qwen-max 更强但更贵，qwen-turbo 最便宜
LLM_TEMPERATURE = 0.1         # RAG场景要低随机性，保证答案稳定、可追溯

# ========== Embedding 配置（翻译官：文字 → 向量）==========
EMBEDDING_MODEL = "BAAI/bge-m3"        # HuggingFace模型ID，首次运行自动下载（约2GB）
EMBEDDING_DEVICE = "cuda"               # 你有GPU(torch+cu124)；无GPU环境改成 "cpu"

# ========== Reranker 配置（精排裁判：检索结果二次打分）==========
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # 首次运行自动下载（约1GB）
RERANKER_DEVICE = "cuda"

# ========== POC 基线参数（手册固定基线，阶段1专用）==========
CHUNK_SIZE = 800            # 递归字符切分块大小
CHUNK_OVERLAP = 80          # 重叠10%，防止句子被硬切断
RETRIEVE_TOP_K = 10         # 向量召回 Top10
RERANK_TOP_N = 4            # 重排后筛选 Top4 送给LLM

# ========== 阶段2：父子块切分参数 ==========
PARENT_CHUNK_SIZE = 600     # 父块大小：喂给 LLM 的完整上下文（语义完整的一段）
PARENT_CHUNK_OVERLAP = 60   # 父块重叠
CHILD_CHUNK_SIZE = 200      # 子块大小：只做检索定位，小=向量更聚焦、匹配更准
CHILD_CHUNK_OVERLAP = 40    # 子块重叠

# ========== 阶段2：Qdrant 向量库配置 ==========
QDRANT_DIR = BASE_DIR / "qdrant_db"           # Qdrant 本地文件模式存储目录（无需 Docker）
QDRANT_COLLECTION = "rag_hybrid"              # 子块 Collection（稠密+稀疏双向量）
QDRANT_PARENT_COLLECTION = "rag_parents"      # 父块 Collection（只存原文，取向量库当 KV 用）
DENSE_VECTOR_NAME = "dense"                   # 稠密向量名字（Qdrant 命名向量）
SPARSE_VECTOR_NAME = "sparse"                 # 稀疏向量名字

# ========== 阶段2：Hybrid 混合检索参数 ==========
# 【阶段5 A/B 实测基线】tests/phase5_ab_test.py，18 题题库（20 子块语料）：
#   - top_k=5/10/15 三档 Recall@10 均 100%，top_k=5 时平均候选仅 6.8 个（精排成本减半）
#   - top_n=3 时仍满分，上下文 449 字符（比 top_n=4 省 27% token）
#   - RRF 直接 Top4 与精排 Top4 的 MRR 持平（精排在小语料上增量为 0，
#     其价值在难检索场景：阶段1大块语料上曾把 Top-1 从 58.3% 提到 100%）
# 结论：维持现基线不改 —— 语料会增长（增量更新/多文档），top_k/top_n 的余量
# 是给未来更难检索场景留的安全边际；语料扩大后应重跑 phase5_ab_test 复核。
HYBRID_TOP_K = 10           # 稠密/稀疏每路各召回 Top10
RRF_K = 60                  # RRF 融合常数（业界标准值 60，见 chunker 注释）

# ========== 阶段3：LangGraph Agent 编排参数 ==========
AGENT_MAX_REWRITES = 2      # 查询改写最大重试次数（防死循环；超限后兜底生成"无法回答"）

# ========== 阶段4-模式A：CrewAI 多角色子Agent ==========
# 命中以下任一关键词的问题视为"报告型任务"，路由到 CrewAI 多角色协作节点；
# 否则走普通单 LLM 生成节点。启发式判断，零成本、可解释，比再花一次 LLM 分类更划算
CREW_TRIGGER_KEYWORDS = ["对比", "比较", "差异", "报告", "写一份", "审核", "调研", "梳理", "总结"]

# ========== 检查配置是否就绪（后面每个阶段入口都会调用）==========
def check_llm_ready() -> bool:
    """LLM Key 是否已配置；离线链路（解析/切分/入库）不需要它"""
    return bool(LLM_API_KEY)
