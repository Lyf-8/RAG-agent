"""
阶段5-步骤3a：对话持久化（SQLite 最小实现）

职责：把每一次问答按"会话"存进本地 SQLite，支持多轮对话的历史查询。
"""
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime

from config.settings import BASE_DIR

# 库文件路径（模块级常量：测试时 patch 成临时路径即可得到隔离的干净库）
DB_PATH = BASE_DIR / "chat_history.db"


def _conn() -> sqlite3.Connection:
    """开短连接，Row 工厂让查询结果可以按列名访问"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表（幂等：已存在则跳过）。session_id 加索引，按会话查历史是主场景"""
    with closing(_conn()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,          -- user / assistant
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
        )


def new_session_id() -> str:
    """生成会话 ID（uuid4 短横线格式，客户端无会话时由服务端分配）"""
    return uuid.uuid4().hex


def add_message(session_id: str, role: str, content: str) -> None:
    """追加一条消息到指定会话"""
    with closing(_conn()) as conn, conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
        )


def get_history(session_id: str) -> list[dict]:
    """按时间正序取回一个会话的全部消息（只读，无需事务层）"""
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]
