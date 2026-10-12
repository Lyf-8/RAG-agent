"""
全局配置文件 —— 项目唯一的参数入口
所有阶段共享此配置，改参数只改这里，符合工程化规范
"""
import os
from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录 enterprise_rag_agent/
DATA_DIR = BASE_DIR / "data"                        # 原始企业文档存放处
CHROMA_DIR = BASE_DIR / "chroma_db"                 # POC向量库本地持久化目录（首次运行自动创建）

# ========== LLM 配置（大脑：语义理解/判断/生成/校验）==========
# 通义千问 DashScope，走 OpenAI 兼容协议 —— 好处：一套代码可无缝换 DeepSeek/OpenAI
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")    # 从环境变量读取，Key 绝不写死在代码里
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"       # 主力模型；qwen-max 更强但更贵，qwen-turbo 最便宜
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

# ========== 检查配置是否就绪（后面每个阶段入口都会调用）==========
def check_llm_ready() -> bool:
    """LLM Key 是否已配置；离线链路（解析/切分/入库）不需要它"""
    return bool(LLM_API_KEY)
