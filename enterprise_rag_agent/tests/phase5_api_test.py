"""
阶段5-步骤3c：API 层测试（TestClient + 桩 LLM，零 API 费用）

目的：验证 FastAPI 封装的"接线"完全正确 —— 接口契约、持久化、异常处理。
     图逻辑本身已由 phase3/phase4 测试覆盖，本测试只测 API 层的职责。

测试策略（与阶段3一脉相承）：
1. 桩 LLM：monkey-patch agent.nodes.get_llm，grade 全判 yes → 直达 generate；
2. 隔离数据库：patch api.conversation.DB_PATH 到系统临时目录，
   测试跑完即删，绝不污染真实 chat_history.db；
3. TestClient：进程内直接调 ASGI 应用，无需真起服务、开端口。

覆盖点：
  ✓ GET  /health     探活返回服务状态与向量库规模
  ✓ POST /chat       无 session_id 时服务端分配；返回答案来自桩 LLM
  ✓ POST /chat       同 session_id 第二轮 → 历史按序累计 4 条（2问2答）
  ✓ GET  /history    角色与顺序正确
  ✓ POST /chat       图执行抛异常 → 500 结构化错误（无堆栈泄露）
  ✓ POST /chat       空问题 → 422 边界校验（Pydantic 自动拦截）

运行方式（项目根目录下）：
    python -m tests.phase5_api_test
"""
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import agent.nodes as nodes
import api.conversation as conversation
from api.main import app
from tests.phase3_test import FakeLLM

# ---------- 隔离测试数据库：临时目录建库，测完即删 ----------
_TMP_DB = Path(tempfile.gettempdir()) / "rag_api_test_chat.db"
conversation.DB_PATH = _TMP_DB
conversation.init_db()

client = TestClient(app)


def main():
    # ========== 1. 探活接口 ==========
    print("=" * 70)
    print("【1】GET /health")
    print("=" * 70)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and "llm_ready" in body and "qdrant" in body
    print(f"[断言通过] {body}")

    # ========== 2. 首轮问答（无 session_id → 服务端分配）==========
    print("\n" + "=" * 70)
    print("【2】POST /chat 首轮（桩 LLM：grade 全 yes → 直达 generate）")
    print("=" * 70)
    fake = FakeLLM(grade_verdict="yes", fake_answer="桩答案：GPS定位精度为50米。")
    nodes.get_llm = lambda: fake

    resp = client.post("/chat", json={"question": "GPS定位打卡的精度是多少？"})
    assert resp.status_code == 200, f"预期200，实际{resp.status_code}: {resp.text}"
    body = resp.json()
    session_id = body["session_id"]
    assert session_id, "服务端应分配 session_id"
    assert body["answer"] == fake.fake_answer, "答案应来自桩 LLM"
    assert fake.rewrite_calls == 0 and fake.generate_calls == 1, "应直达生成、无改写"
    print(f"[断言通过] session={session_id[:8]} 答案={body['answer']}")

    # ========== 3. 同会话第二轮 → 历史累计 ==========
    print("\n" + "=" * 70)
    print("【3】POST /chat 同会话第二轮 → GET /history 应有 4 条（2问2答）")
    print("=" * 70)
    resp = client.post("/chat", json={"question": "蓝牙打卡范围呢？", "session_id": session_id})
    assert resp.status_code == 200 and resp.json()["session_id"] == session_id

    resp = client.get(f"/history/{session_id}")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 4, f"预期4条历史，实际{len(messages)}"
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"], "角色顺序应为 问/答/问/答"
    assert messages[0]["content"] == "GPS定位打卡的精度是多少？", "首条应为第一轮问题"
    print(f"[断言通过] 历史 {len(messages)} 条，角色序列正确，按时间正序")

    # ========== 4. 异常路径：图执行抛错 → 500 结构化错误 ==========
    print("\n" + "=" * 70)
    print("【4】POST /chat 异常路径（桩 ask 抛异常 → 500 + JSON detail）")
    print("=" * 70)
    import api.main as api_main
    original_ask = api_main.ask

    def _boom(_question: str) -> str:
        raise RuntimeError("模拟图执行内部故障")

    api_main.ask = _boom
    try:
        resp = client.post("/chat", json={"question": "会触发故障的问题"})
        assert resp.status_code == 500, f"预期500，实际{resp.status_code}"
        detail = resp.json()["detail"]
        assert "RuntimeError" in detail and "模拟图执行内部故障" in detail
        assert "Traceback" not in resp.text, "不应向调用方泄露堆栈"
        print(f"[断言通过] 500 结构化错误: {detail}")
    finally:
        api_main.ask = original_ask     # 恢复，避免影响后续用例

    # ========== 5. 边界校验：空问题 → 422 ==========
    print("\n" + "=" * 70)
    print("【5】POST /chat 空问题 → 422（Pydantic 边界校验）")
    print("=" * 70)
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422, f"预期422，实际{resp.status_code}"
    print("[断言通过] 空问题被边界拦截，返回 422")

    # ---------- 收尾：清理临时库 ----------
    _TMP_DB.unlink(missing_ok=True)
    print("\n" + "=" * 70)
    print("阶段5-步骤3 API 测试全部通过 ✓（接口契约/持久化/异常/校验，零 API 费用）")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
