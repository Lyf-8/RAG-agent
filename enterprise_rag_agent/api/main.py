"""
阶段5-步骤3b：FastAPI 工程化封装（对外服务入口）

职责：把 Agent 能力包装成 HTTP 服务 —— POST /chat 问答、GET /health 探活、
     GET /history/{session_id} 查历史，含请求校验、结构化异常、访问日志。

知识点讲解：
1. 薄封装原则：API 层只做三件事
   - 协议转换：HTTP JSON ↔ Python 函数调用（Pydantic 模型负责校验）
   - 横切关注点：持久化、日志、异常处理（与业务无关的"后勤"）
   - 绝不碰图逻辑：路由、改写、生成全在 agent 层
   好处：图逻辑已有独立的桩测试（phase3/4），API 层只测"接线对不对"；
   将来换协议（gRPC/WebSocket/CLI）时核心代码零改动。
   分层：表现层(api) → 编排层(agent) → 能力层(qa_chain/preprocess)。

2. Pydantic 模型的意义：边界校验前置
   请求体缺字段/类型错，FastAPI 直接返回 422 和逐字段错误信息，
   业务代码里不需要写一行防御性 if —— "在系统边界校验，内部充分信任"。

3. 结构化异常：调用方拿到的是可解析的 JSON（{"detail": ...}），
   而不是一坨 HTML 堆栈页 —— 这是"服务"和"脚本"的分水岭。

4. /health 探活：负载均衡和监控系统靠它判断实例生死；
   顺带暴露 llm_ready / 向量库规模，运维一眼区分"服务挂了"与"LLM欠费"。

运行方式（项目根目录下）：
    python -m api.main          # 启动服务于 http://127.0.0.1:8000
    浏览器打开 http://127.0.0.1:8000/docs 可交互调试（FastAPI 自带 Swagger UI）
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.graph import ask
from api.conversation import init_db, new_session_id, add_message, get_history
from config.settings import check_llm_ready
from preprocess.storage.qdrant_store import get_counts

# ---------- 日志（最小版：输出到控制台，生产可改 FileHandler/接入 ELK）----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_api")

init_db()   # 启动即建表（幂等）

app = FastAPI(title="企业 RAG Agent API", version="0.5.0")


# ========== 请求/响应模型（边界校验）==========
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题，非空")
    session_id: str | None = Field(default=None, description="会话ID；不传则服务端分配新会话")


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


# ========== 接口1：问答 ==========
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or new_session_id()
    logger.info("chat 请求: session=%s 问题=%s", session_id[:8], req.question[:30])

    start = time.perf_counter()
    try:
        answer = ask(req.question)          # 调编排层唯一入口，API 层不碰图内部
    except Exception as e:
        # 结构化错误：不把堆栈泄露给调用方，但日志里保留完整信息
        logger.exception("chat 处理失败: session=%s", session_id[:8])
        raise HTTPException(status_code=500, detail=f"问答处理失败: {type(e).__name__}: {e}")
    elapsed = time.perf_counter() - start

    add_message(session_id, "user", req.question)
    add_message(session_id, "assistant", answer)
    logger.info("chat 完成: session=%s 耗时=%.2fs 答案=%d字", session_id[:8], elapsed, len(answer))
    return ChatResponse(session_id=session_id, answer=answer)


# ========== 接口2：健康探活 ==========
@app.get("/health")
def health() -> dict:
    try:
        counts = get_counts()
    except Exception:
        counts = {"children": -1, "parents": -1}   # 向量库异常不拖垮探活接口
    return {"status": "ok", "llm_ready": check_llm_ready(), "qdrant": counts}


# ========== 接口3：会话历史 ==========
@app.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str) -> HistoryResponse:
    return HistoryResponse(session_id=session_id, messages=get_history(session_id))


# ========== 启动入口 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
