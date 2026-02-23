# """
# memory/memory_manager.py
# ────────────────────────
# Async memory manager for the PaddleAurum agent system.
# Handles three memory tiers:
#   1. Short-term  — in-state message window (managed in PaddleAurumState)
#   2. Long-term   — SQLite rows persisted across sessions
#   3. Shared      — key/value store for cross-agent facts
# """
# from __future__ import annotations

# import json
# import logging
# import time
# from typing import Any, Dict, List, Optional

# import aiosqlite

# from config.settings import settings

# logger = logging.getLogger(__name__)
# DB_PATH = settings.sqlite_db_path


# class MemoryManager:
#     """
#     Central async memory interface.
#     All agents should use this class rather than direct DB queries
#     so memory access is audited and window-managed in one place.
#     """

#     def __init__(self, db_path: str = DB_PATH):
#         self.db_path = db_path

#     # ── Initialisation ────────────────────────────────────────────────────────

#     async def init_db(self) -> None:
#         """Create all required tables. Safe to call multiple times (IF NOT EXISTS)."""
#         from tools.db_tool import init_db
#         await init_db()
#         logger.info("MemoryManager: DB initialised at %s", self.db_path)

#     # ── Short-term window management ──────────────────────────────────────────

#     def trim_short_term(
#         self, messages: List[Dict], window: int = None
#     ) -> List[Dict]:
#         """
#         Trim the in-memory message list to the configured window size.
#         Always keeps the system message (index 0) if present.
#         """
#         if window is None:
#             window = settings.short_term_memory_window

#         if len(messages) <= window:
#             return messages

#         # Preserve system message
#         if messages and messages[0].get("role") == "system":
#             return [messages[0]] + messages[-(window - 1):]
#         return messages[-window:]

#     # ── Long-term memory (SQLite) ─────────────────────────────────────────────

#     async def write_session(
#         self, session_id: str, messages: List[Dict]
#     ) -> List[int]:
#         """
#         Persist a list of agent messages to the long-term agent_memory table.
#         Returns list of inserted row IDs for storage in state.long_term_memory_keys.
#         """
#         row_ids = []
#         async with aiosqlite.connect(self.db_path) as db:
#             for msg in messages:
#                 cursor = await db.execute(
#                     "INSERT INTO agent_memory (session_id, agent_id, role, content, timestamp) "
#                     "VALUES (?, ?, ?, ?, ?)",
#                     (
#                         session_id,
#                         msg.get("agent_id", "unknown"),
#                         msg.get("role", "assistant"),
#                         msg.get("content", ""),
#                         msg.get("timestamp", time.time()),
#                     ),
#                 )
#                 row_ids.append(cursor.lastrowid)
#             await db.commit()
#         logger.debug("Persisted %d messages for session %s", len(messages), session_id)
#         return [str(r) for r in row_ids]

#     async def recall(
#         self,
#         agent_id: str,
#         limit: int = 10,
#         session_id: Optional[str] = None,
#     ) -> List[str]:
#         """
#         Recall recent messages for a specific agent.
#         Optionally filter by session_id (cross-session by default).
#         """
#         async with aiosqlite.connect(self.db_path) as db:
#             if session_id:
#                 query = (
#                     "SELECT content FROM agent_memory "
#                     "WHERE agent_id = ? AND session_id = ? "
#                     "ORDER BY timestamp DESC LIMIT ?"
#                 )
#                 params = (agent_id, session_id, limit)
#             else:
#                 query = (
#                     "SELECT content FROM agent_memory "
#                     "WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?"
#                 )
#                 params = (agent_id, limit)
#             async with db.execute(query, params) as cursor:
#                 rows = await cursor.fetchall()
#         return [r[0] for r in rows]

#     async def recall_session(self, session_id: str) -> List[Dict]:
#         """Return all messages for a given session (for replay/audit)."""
#         async with aiosqlite.connect(self.db_path) as db:
#             db.row_factory = aiosqlite.Row
#             async with db.execute(
#                 "SELECT * FROM agent_memory WHERE session_id = ? ORDER BY timestamp ASC",
#                 (session_id,),
#             ) as cursor:
#                 rows = await cursor.fetchall()
#         return [dict(r) for r in rows]

#     async def search_memory(self, keyword: str, limit: int = 10) -> List[Dict]:
#         """Full-text search across all agent memory (SQLite LIKE)."""
#         async with aiosqlite.connect(self.db_path) as db:
#             db.row_factory = aiosqlite.Row
#             async with db.execute(
#                 "SELECT * FROM agent_memory WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
#                 (f"%{keyword}%", limit),
#             ) as cursor:
#                 rows = await cursor.fetchall()
#         return [dict(r) for r in rows]

#     # ── Shared context store ──────────────────────────────────────────────────

#     async def set_shared(self, key: str, value: Any) -> None:
#         """
#         Store a shared context value (cross-agent key-value).
#         Uses the agent_memory table with agent_id='_shared'.
#         """
#         async with aiosqlite.connect(self.db_path) as db:
#             await db.execute(
#                 "INSERT OR REPLACE INTO agent_memory "
#                 "(session_id, agent_id, role, content, timestamp) "
#                 "VALUES ('_shared', '_shared', ?, ?, ?)",
#                 (key, json.dumps(value), time.time()),
#             )
#             await db.commit()

#     async def get_shared(self, key: str) -> Optional[Any]:
#         """Retrieve a shared context value by key."""
#         async with aiosqlite.connect(self.db_path) as db:
#             async with db.execute(
#                 "SELECT content FROM agent_memory "
#                 "WHERE agent_id='_shared' AND role=? "
#                 "ORDER BY timestamp DESC LIMIT 1",
#                 (key,),
#             ) as cursor:
#                 row = await cursor.fetchone()
#         if row:
#             return json.loads(row[0])
#         return None

#     # ── Session summary ───────────────────────────────────────────────────────

#     async def write_session_summary(self, session_id: str, summary: Dict) -> None:
#         """Persist final session summary for future CEO planning context."""
#         await self.set_shared(f"session_summary:{session_id}", summary)
#         logger.info("Session summary written for %s", session_id)

#     async def get_recent_summaries(self, limit: int = 5) -> List[Dict]:
#         """Retrieve the most recent session summaries."""
#         async with aiosqlite.connect(self.db_path) as db:
#             async with db.execute(
#                 "SELECT role, content, timestamp FROM agent_memory "
#                 "WHERE agent_id='_shared' AND role LIKE 'session_summary:%' "
#                 "ORDER BY timestamp DESC LIMIT ?",
#                 (limit,),
#             ) as cursor:
#                 rows = await cursor.fetchall()
#         summaries = []
#         for row in rows:
#             try:
#                 summaries.append({
#                     "session_id": row[0].replace("session_summary:", ""),
#                     "summary": json.loads(row[1]),
#                     "timestamp": row[2],
#                 })
#             except Exception:
#                 continue
#         return summaries

#     # ── Cleanup ───────────────────────────────────────────────────────────────

#     async def purge_old_memory(self, older_than_days: int = 30) -> int:
#         """Delete memory records older than N days. Returns count deleted."""
#         cutoff = time.time() - (older_than_days * 86400)
#         async with aiosqlite.connect(self.db_path) as db:
#             cursor = await db.execute(
#                 "DELETE FROM agent_memory WHERE timestamp < ? AND agent_id != '_shared'",
#                 (cutoff,),
#             )
#             await db.commit()
#             deleted = cursor.rowcount
#         logger.info("Purged %d old memory records (older than %d days)", deleted, older_than_days)
#         return deleted


# # Module-level singleton
# memory_manager = MemoryManager()



























 





# @@@#################################################################################################################






















"""
memory/memory_manager.py
────────────────────────
Async memory manager for the PaddleAurum agent system.
Handles three memory tiers:
  1. Short-term  — in-state message window (managed in PaddleAurumState)
  2. Long-term   — SQLite rows persisted across sessions
  3. Shared      — key/value store for cross-agent facts
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiosqlite

from config.settings import settings

logger = logging.getLogger(__name__)
DB_PATH = settings.sqlite_db_path


class MemoryManager:
    """
    Central async memory interface.
    All agents should use this class rather than direct DB queries
    so memory access is audited and window-managed in one place.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ── Initialisation ────────────────────────────────────────────────────────

    async def init_db(self) -> None:
        """Create all required tables. Safe to call multiple times (IF NOT EXISTS)."""
        from tools.db_tool import init_db
        await init_db()  # creates core tables (agent_memory, etc.)

        # Add shared_context table and FTS5 virtual table for agent_memory
        async with aiosqlite.connect(self.db_path) as db:
            # Shared key-value store (separate table)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS shared_context (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            # FTS5 virtual table for full-text search on agent_memory.content
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS agent_memory_fts USING fts5(
                    content, content=agent_memory, content_rowid=id
                )
            """)
            # Triggers to keep FTS index in sync
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS agent_memory_ai AFTER INSERT ON agent_memory
                BEGIN
                    INSERT INTO agent_memory_fts(rowid, content) VALUES (NEW.id, NEW.content);
                END
            """)
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS agent_memory_ad AFTER DELETE ON agent_memory
                BEGIN
                    INSERT INTO agent_memory_fts(agent_memory_fts, rowid, content) VALUES('delete', OLD.id, OLD.content);
                END
            """)
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS agent_memory_au AFTER UPDATE ON agent_memory
                BEGIN
                    INSERT INTO agent_memory_fts(agent_memory_fts, rowid, content) VALUES('delete', OLD.id, OLD.content);
                    INSERT INTO agent_memory_fts(rowid, content) VALUES (NEW.id, NEW.content);
                END
            """)
            await db.commit()
        logger.info("MemoryManager: extended schema initialised at %s", self.db_path)

    # ── Short-term window management ──────────────────────────────────────────

    def trim_short_term(
        self, messages: List[Dict], window: int = None
    ) -> List[Dict]:
        """
        Trim the in-memory message list to the configured window size.
        Always keeps the system message (index 0) if present.
        """
        if window is None:
            window = settings.short_term_memory_window

        if len(messages) <= window:
            return messages

        # Preserve system message
        if messages and messages[0].get("role") == "system":
            return [messages[0]] + messages[-(window - 1):]
        return messages[-window:]

    # ── Long-term memory (SQLite) ─────────────────────────────────────────────

    async def write_session(
        self, session_id: str, messages: List[Dict]
    ) -> List[int]:
        """
        Persist a list of agent messages to the long-term agent_memory table.
        Returns list of inserted row IDs for storage in state.long_term_memory_keys.
        """
        row_ids = []
        async with aiosqlite.connect(self.db_path) as db:
            for msg in messages:
                cursor = await db.execute(
                    "INSERT INTO agent_memory (session_id, agent_id, role, content, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        msg.get("agent_id", "unknown"),
                        msg.get("role", "assistant"),
                        msg.get("content", ""),
                        msg.get("timestamp", time.time()),
                    ),
                )
                row_ids.append(cursor.lastrowid)
            await db.commit()
        logger.debug("Persisted %d messages for session %s", len(messages), session_id)
        return [str(r) for r in row_ids]

    async def recall(
        self,
        agent_id: str,
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Recall recent messages for a specific agent.
        Optionally filter by session_id (cross-session by default).
        """
        async with aiosqlite.connect(self.db_path) as db:
            if session_id:
                query = (
                    "SELECT content FROM agent_memory "
                    "WHERE agent_id = ? AND session_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?"
                )
                params = (agent_id, session_id, limit)
            else:
                query = (
                    "SELECT content FROM agent_memory "
                    "WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?"
                )
                params = (agent_id, limit)
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def recall_session(self, session_id: str) -> List[Dict]:
        """Return all messages for a given session (for replay/audit)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_memory WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def search_memory(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        Full-text search across all agent memory using FTS5.
        Returns matching rows sorted by relevance.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT am.* FROM agent_memory_fts fts "
                "JOIN agent_memory am ON fts.rowid = am.id "
                "WHERE agent_memory_fts MATCH ? ORDER BY fts.rank LIMIT ?",
                (keyword, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Shared context store (separate table) ────────────────────────────────

    async def set_shared(self, key: str, value: Any) -> None:
        """
        Store a shared context value (cross-agent key-value).
        Uses dedicated shared_context table.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO shared_context (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            await db.commit()

    async def get_shared(self, key: str) -> Optional[Any]:
        """Retrieve a shared context value by key."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT value FROM shared_context WHERE key = ?",
                (key,),
            ) as cursor:
                row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    # ── Session summary ───────────────────────────────────────────────────────

    async def write_session_summary(self, session_id: str, summary: Dict) -> None:
        """Persist final session summary for future CEO planning context."""
        await self.set_shared(f"session_summary:{session_id}", summary)
        logger.info("Session summary written for %s", session_id)

    async def get_recent_summaries(self, limit: int = 5) -> List[Dict]:
        """Retrieve the most recent session summaries."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT key, value, updated_at FROM shared_context "
                "WHERE key LIKE 'session_summary:%' "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        summaries = []
        for key, value_json, ts in rows:
            try:
                summaries.append({
                    "session_id": key.replace("session_summary:", ""),
                    "summary": json.loads(value_json),
                    "timestamp": ts,
                })
            except Exception:
                continue
        return summaries

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def purge_old_memory(self, older_than_days: int = 30) -> int:
        """
        Delete memory records older than N days. Returns count deleted.
        (Excludes shared_context entries, which are kept indefinitely.)
        """
        cutoff = time.time() - (older_than_days * 86400)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM agent_memory WHERE timestamp < ?",
                (cutoff,),
            )
            await db.commit()
            deleted = cursor.rowcount
        logger.info("Purged %d old memory records (older than %d days)", deleted, older_than_days)
        return deleted

    # ── Scheduled purge (optional) ────────────────────────────────────────────

    def schedule_purge(self, interval_days: int = 30) -> None:
        """
        Set up an APScheduler job to automatically purge old memory.
        Call this once during application startup if you want automatic cleanup.
        Requires `apscheduler` to be installed.
        """
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
        except ImportError:
            logger.error("APScheduler not installed. Cannot schedule purge.")
            return

        scheduler = AsyncIOScheduler()

        async def _purge_job():
            await self.purge_old_memory(older_than_days=interval_days)

        scheduler.add_job(
            _purge_job,
            "interval",
            days=interval_days,
            id="memory_purge",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Memory purge scheduled every %d days", interval_days)


# Module-level singleton
memory_manager = MemoryManager()
