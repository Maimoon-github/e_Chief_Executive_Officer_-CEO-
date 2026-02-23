"""
tools/db_tool.py
────────────────
Async SQLite database tool for all agents.
Handles: customer profiles, purchase history, restock thresholds,
promo logs, and agent memory tables.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiosqlite
from langchain.tools import BaseTool
from pydantic import Field

from config.settings import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.sqlite_db_path

# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS agent_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    timestamp       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id         TEXT PRIMARY KEY,
    email               TEXT,
    name                TEXT,
    purchase_history    TEXT DEFAULT '[]',  -- JSON array
    lifetime_value      REAL DEFAULT 0.0,
    segment             TEXT DEFAULT 'general',
    last_contact        REAL,
    created_at          REAL
);

CREATE TABLE IF NOT EXISTS restock_thresholds (
    product_id      TEXT PRIMARY KEY,
    sku             TEXT,
    product_name    TEXT,
    min_qty         INTEGER DEFAULT 10,
    reorder_qty     INTEGER DEFAULT 50,
    supplier_url    TEXT,
    supplier_price  REAL,
    last_updated    REAL
);

CREATE TABLE IF NOT EXISTS sales_velocity (
    product_id          TEXT PRIMARY KEY,
    units_sold_30d      INTEGER DEFAULT 0,
    units_sold_7d       INTEGER DEFAULT 0,
    avg_daily_sales     REAL DEFAULT 0.0,
    last_calculated     REAL
);

CREATE TABLE IF NOT EXISTS promo_log (
    campaign_id         TEXT PRIMARY KEY,
    campaign_type       TEXT,
    subject_line        TEXT,
    target_segment      TEXT,
    sent_at             REAL,
    sent_count          INTEGER DEFAULT 0,
    open_rate           REAL DEFAULT 0.0,
    click_rate          REAL DEFAULT 0.0,
    revenue_attributed  REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_memory_session   ON agent_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_agent     ON agent_memory(agent_id);
CREATE INDEX IF NOT EXISTS idx_customer_segment ON customer_profiles(segment);
"""


async def init_db() -> None:
    """Create all tables if they do not exist. Call once on startup."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_TABLES_SQL)
        await db.commit()
    logger.info("Database initialised at %s", DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level async helpers
# ─────────────────────────────────────────────────────────────────────────────

async def db_fetch_one(query: str, params: tuple = ()) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def db_fetch_all(query: str, params: tuple = ()) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def db_execute(query: str, params: tuple = ()) -> int:
    """Execute an INSERT/UPDATE/DELETE, return lastrowid."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(query, params)
        await db.commit()
        return cursor.lastrowid


async def db_executemany(query: str, params_list: List[tuple]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(query, params_list)
        await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific operations
# ─────────────────────────────────────────────────────────────────────────────

async def get_customer_profile(customer_id: str) -> Optional[Dict]:
    return await db_fetch_one(
        "SELECT * FROM customer_profiles WHERE customer_id = ?", (customer_id,)
    )


async def upsert_customer_profile(profile: Dict) -> None:
    await db_execute(
        """INSERT INTO customer_profiles
           (customer_id, email, name, purchase_history, lifetime_value,
            segment, last_contact, created_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(customer_id) DO UPDATE SET
               email=excluded.email,
               name=excluded.name,
               purchase_history=excluded.purchase_history,
               lifetime_value=excluded.lifetime_value,
               segment=excluded.segment,
               last_contact=excluded.last_contact
        """,
        (
            profile["customer_id"],
            profile.get("email", ""),
            profile.get("name", ""),
            json.dumps(profile.get("purchase_history", [])),
            profile.get("lifetime_value", 0.0),
            profile.get("segment", "general"),
            profile.get("last_contact", time.time()),
            profile.get("created_at", time.time()),
        ),
    )


async def get_customers_by_segment(segment: str) -> List[Dict]:
    rows = await db_fetch_all(
        "SELECT * FROM customer_profiles WHERE segment = ?", (segment,)
    )
    for row in rows:
        row["purchase_history"] = json.loads(row.get("purchase_history", "[]"))
    return rows


async def get_restock_threshold(product_id: str) -> Optional[Dict]:
    return await db_fetch_one(
        "SELECT * FROM restock_thresholds WHERE product_id = ?", (product_id,)
    )


async def get_all_thresholds() -> List[Dict]:
    return await db_fetch_all("SELECT * FROM restock_thresholds")


async def upsert_threshold(data: Dict) -> None:
    await db_execute(
        """INSERT INTO restock_thresholds
           (product_id, sku, product_name, min_qty, reorder_qty,
            supplier_url, supplier_price, last_updated)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(product_id) DO UPDATE SET
               min_qty=excluded.min_qty,
               reorder_qty=excluded.reorder_qty,
               supplier_url=excluded.supplier_url,
               supplier_price=excluded.supplier_price,
               last_updated=excluded.last_updated
        """,
        (
            data["product_id"], data.get("sku", ""), data.get("product_name", ""),
            data.get("min_qty", 10), data.get("reorder_qty", 50),
            data.get("supplier_url", ""), data.get("supplier_price", 0.0),
            time.time(),
        ),
    )


async def get_sales_velocity(product_id: str) -> Optional[Dict]:
    return await db_fetch_one(
        "SELECT * FROM sales_velocity WHERE product_id = ?", (product_id,)
    )


async def write_agent_memory(session_id: str, agent_id: str, role: str, content: str) -> None:
    await db_execute(
        "INSERT INTO agent_memory (session_id, agent_id, role, content, timestamp) VALUES (?,?,?,?,?)",
        (session_id, agent_id, role, content, time.time()),
    )


async def recall_agent_memory(agent_id: str, limit: int = 10) -> List[str]:
    rows = await db_fetch_all(
        "SELECT content FROM agent_memory WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?",
        (agent_id, limit),
    )
    return [r["content"] for r in rows]


async def log_campaign(campaign: Dict) -> None:
    await db_execute(
        """INSERT INTO promo_log
           (campaign_id, campaign_type, subject_line, target_segment, sent_at, sent_count)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(campaign_id) DO NOTHING
        """,
        (
            campaign["campaign_id"], campaign.get("campaign_type", "email"),
            campaign.get("subject_line", ""), campaign.get("target_segment", ""),
            time.time(), campaign.get("sent_count", 0),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangChain BaseTool wrapper (used inside CrewAI agents)
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseTool(BaseTool):
    name: str = "database"
    description: str = (
        "Query the PaddleAurum SQLite database. "
        "Actions: get_customer, get_customers_by_segment, get_threshold, "
        "get_all_thresholds, get_sales_velocity, recall_memory. "
        "Input: JSON string with 'action' key and relevant parameters."
    )

    async def _arun(self, query: str) -> str:
        try:
            params = json.loads(query)
            action = params.get("action")

            if action == "get_customer":
                result = await get_customer_profile(params["customer_id"])
            elif action == "get_customers_by_segment":
                result = await get_customers_by_segment(params["segment"])
            elif action == "get_threshold":
                result = await get_restock_threshold(params["product_id"])
            elif action == "get_all_thresholds":
                result = await get_all_thresholds()
            elif action == "get_sales_velocity":
                result = await get_sales_velocity(params["product_id"])
            elif action == "recall_memory":
                result = await recall_agent_memory(
                    params["agent_id"], params.get("limit", 10)
                )
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

            return json.dumps(result or {})
        except Exception as exc:
            logger.error("DatabaseTool error: %s", exc)
            return json.dumps({"error": str(exc)})

    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async only via _arun")
