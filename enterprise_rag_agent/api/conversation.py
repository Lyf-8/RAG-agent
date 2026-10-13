"""
阶段5-步骤3a：对话持久化（SQLite 最小实现）

职责：把每一次问答按"会话"存进本地 SQLite，支持多轮对话的历史查询。

知识点讲解：
1. 为什么选 SQLite 而不是 Redis/PostgreSQL？
   - 零配置：一个文件就是一个库，随项目分发，测试时换个路径即是全新库；
   - 单机 Demo/小团队内部工具完全够用（QPS 几十以内无压力）；
   - 工程原则：先用最简方案跑通"持久化"这个能力维度，
     等真要上生产再换外部存储 —— 届时只需改本文件的 4 个函数，API 层不动。
     这就是"存储细节收敛在一个模块里"的价值（依赖倒置）。

2. 表设计为什么只有一张 messages？
   会话本身不需要元数据（标题、所有者都是后续需求），
   session_id 只是客户端传来的分组标识 —— 首条消息到达时会话自然"存在"。
   不建 sessions 表 = 不维护多余状态 = 少一类 bug。

3. 为什么每次操作都开短连接而不是全局长连接？
   sqlite3 连接不是线程安全的；FastAPI 默认多线程处理请求。
   短连接（用完即关）是最省心的线程安全方案，本地文件开销可忽略。

4. 经典坑：with sqlite3.connect() 并不会关闭连接！
   Connection 的 __exit__ 只负责 commit/rollback（事务管理），不关文件句柄。
   句柄泄漏在 Windows 上会导致文件被占用删不掉，长跑服务会越攒越多。
   正确姿势：with closing(_conn()) as conn, conn:
   —— 外层 closing 负责关连接，内层 conn 负责事务，两层各司其职。
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
