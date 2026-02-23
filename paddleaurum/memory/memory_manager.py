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
        await init_db()
        logger.info("MemoryManager: DB initialised at %s", self.db_path)

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
        """Full-text search across all agent memory (SQLite LIKE)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM agent_memory WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Shared context store ──────────────────────────────────────────────────

    async def set_shared(self, key: str, value: Any) -> None:
        """
        Store a shared context value (cross-agent key-value).
        Uses the agent_memory table with agent_id='_shared'.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO agent_memory "
                "(session_id, agent_id, role, content, timestamp) "
                "VALUES ('_shared', '_shared', ?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            await db.commit()

    async def get_shared(self, key: str) -> Optional[Any]:
        """Retrieve a shared context value by key."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT content FROM agent_memory "
                "WHERE agent_id='_shared' AND role=? "
                "ORDER BY timestamp DESC LIMIT 1",
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
                "SELECT role, content, timestamp FROM agent_memory "
                "WHERE agent_id='_shared' AND role LIKE 'session_summary:%' "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        summaries = []
        for row in rows:
            try:
                summaries.append({
                    "session_id": row[0].replace("session_summary:", ""),
                    "summary": json.loads(row[1]),
                    "timestamp": row[2],
                })
            except Exception:
                continue
        return summaries

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def purge_old_memory(self, older_than_days: int = 30) -> int:
        """Delete memory records older than N days. Returns count deleted."""
        cutoff = time.time() - (older_than_days * 86400)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM agent_memory WHERE timestamp < ? AND agent_id != '_shared'",
                (cutoff,),
            )
            await db.commit()
            deleted = cursor.rowcount
        logger.info("Purged %d old memory records (older than %d days)", deleted, older_than_days)
        return deleted


# Module-level singleton
memory_manager = MemoryManager()
