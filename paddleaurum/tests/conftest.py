"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the PaddleAurum test suite.
Provides:
  - Async-capable event loop configuration (pytest-asyncio)
  - Temporary SQLite database per test
  - Pre-seeded customer and threshold fixtures
  - Reusable mock LLM and mock Shopify response helpers
  - State factory shortcut

Install test dependencies:
    pip install pytest pytest-asyncio pytest-cov aiosqlite --break-system-packages

Run all tests:
    pytest                             # uses pytest.ini settings
    pytest -m unit                     # fast unit tests only
    pytest -m "not integration"        # skip live-API tests
    pytest tests/test_agents.py -v     # single module
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
import pytest_asyncio

# ── Ensure project root is importable ────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Event Loop
# ─────────────────────────────────────────────────────────────────────────────

# pytest-asyncio >= 0.21 auto mode handles loop creation per test.
# Set asyncio_mode = auto in pytest.ini — no explicit loop fixture needed.


# ─────────────────────────────────────────────────────────────────────────────
# Temporary Database Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def tmp_db(tmp_path) -> AsyncGenerator[str, None]:
    """
    Create a fresh, initialised SQLite database in a temp directory.
    Overrides settings.sqlite_db_path for the duration of the test.
    Yields the absolute path to the test database file.

    Usage:
        async def test_something(tmp_db):
            profile = await get_customer_profile("C001")
    """
    from config.settings import settings
    from tools.db_tool import DB_PATH as _ORIGINAL_PATH, init_db

    db_path = str(tmp_path / "paddleaurum_test.db")

    # Temporarily redirect the module-level DB_PATH
    import tools.db_tool as db_module
    import memory.memory_manager as mm_module

    _orig_db = db_module.DB_PATH
    _orig_mm = mm_module.DB_PATH
    _orig_settings = settings.sqlite_db_path

    db_module.DB_PATH = db_path
    mm_module.DB_PATH = db_path
    settings.sqlite_db_path = db_path

    await init_db()

    yield db_path

    # Restore originals
    db_module.DB_PATH = _orig_db
    mm_module.DB_PATH = _orig_mm
    settings.sqlite_db_path = _orig_settings


# ─────────────────────────────────────────────────────────────────────────────
# Pre-seeded Data Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def seeded_customers(tmp_db) -> List[Dict]:
    """
    Seed the test database with representative customer profiles.
    Returns the list of inserted customer dicts.
    """
    from tools.db_tool import upsert_customer_profile

    customers = [
        {
            "customer_id": "C001",
            "email": "player@example.com",
            "name": "Alex Smith",
            "purchase_history": json.dumps([
                {"product_id": "1001", "quantity": 1},
                {"product_id": "1002", "quantity": 3},
            ]),
            "lifetime_value": 389.97,
            "segment": "vip",
            "last_contact": time.time(),
            "created_at": time.time() - 86400 * 30,
        },
        {
            "customer_id": "C002",
            "email": "casual@example.com",
            "name": "Jamie Lee",
            "purchase_history": json.dumps([
                {"product_id": "1002", "quantity": 1},
            ]),
            "lifetime_value": 19.99,
            "segment": "general",
            "last_contact": time.time(),
            "created_at": time.time() - 86400 * 10,
        },
        {
            "customer_id": "C003",
            "email": "angry@example.com",
            "name": "Sam Angry",
            "purchase_history": json.dumps([]),
            "lifetime_value": 0.0,
            "segment": "general",
            "last_contact": time.time(),
            "created_at": time.time() - 86400 * 2,
        },
    ]

    for customer in customers:
        await upsert_customer_profile(customer)

    return customers


@pytest_asyncio.fixture
async def seeded_thresholds(tmp_db) -> List[Dict]:
    """
    Seed the test database with restock thresholds for mock products.
    Returns the list of inserted threshold dicts.
    """
    from tools.db_tool import upsert_threshold

    thresholds = [
        {
            "product_id": "1001",
            "sku": "PADDLE-CE-001",
            "product_name": "ProPaddle Carbon Elite",
            "min_qty": 10,
            "reorder_qty": 50,
            "supplier_url": "http://supplier.example.com/carbon-elite",
            "supplier_price": 89.99,
        },
        {
            "product_id": "1002",
            "sku": "BALL-40-003",
            "product_name": "SpeedBall 40-Hole Outdoor 3-Pack",
            "min_qty": 20,
            "reorder_qty": 100,
            "supplier_url": "http://supplier.example.com/speedball",
            "supplier_price": 11.50,
        },
    ]

    for threshold in thresholds:
        await upsert_threshold(threshold)

    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_factory():
    """
    Factory that creates a MagicMock OllamaLLM with a pre-configured ainvoke.

    Usage:
        def test_something(mock_llm_factory):
            llm = mock_llm_factory({"key": "value"})  # JSON response
    """
    def _factory(response: Any) -> MagicMock:
        llm = MagicMock()
        if isinstance(response, (dict, list)):
            llm.ainvoke = AsyncMock(return_value=json.dumps(response))
        else:
            llm.ainvoke = AsyncMock(return_value=str(response))
        return llm
    return _factory


@pytest.fixture
def mock_ceo_llm(mock_llm_factory):
    """
    Pre-configured CEO LLM mock that returns a valid task decomposition.
    """
    return mock_llm_factory([
        {"description": "Handle support queue", "assigned_to": "chat_buddy",   "priority": 1, "required_tools": [], "max_retries": 3},
        {"description": "Scan inventory",       "assigned_to": "stock_scout",  "priority": 2, "required_tools": [], "max_retries": 3},
        {"description": "Draft promo email",    "assigned_to": "promo_general","priority": 3, "required_tools": [], "max_retries": 3},
    ])


@pytest.fixture
def mock_chat_buddy_llm(mock_llm_factory):
    """
    Pre-configured Chat Buddy LLM mock that returns a valid support response.
    """
    return mock_llm_factory({
        "reply_text": "Great question! For beginners we recommend the LightTouch 16mm paddle.",
        "escalate": False,
        "sentiment_score": 0.85,
        "resolution_type": "answered",
    })


@pytest.fixture
def mock_escalation_llm(mock_llm_factory):
    """
    Chat Buddy LLM mock that triggers an escalation response.
    """
    return mock_llm_factory({
        "reply_text": "I'm so sorry to hear this! Let me escalate immediately.",
        "escalate": True,
        "sentiment_score": 0.1,
        "resolution_type": "escalated",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Mock Shopify Data Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_shopify_products() -> List[Dict]:
    """Standard two-product mock matching mock_shopify_response.json."""
    return [
        {
            "id": 1001,
            "title": "ProPaddle Carbon Elite",
            "status": "active",
            "variants": [{"id": 2001, "sku": "PADDLE-CE-001", "inventory_item_id": 3001, "price": "149.99"}],
        },
        {
            "id": 1002,
            "title": "SpeedBall 40-Hole Outdoor 3-Pack",
            "status": "active",
            "variants": [{"id": 2002, "sku": "BALL-40-003", "inventory_item_id": 3002, "price": "19.99"}],
        },
    ]


@pytest.fixture
def mock_inventory_levels() -> List[Dict]:
    """Inventory levels: product 1001 is low-stock (5 < threshold 10)."""
    return [
        {"inventory_item_id": 3001, "location_id": 9001, "available": 5},
        {"inventory_item_id": 3002, "location_id": 9001, "available": 120},
    ]


@pytest.fixture
def mock_shopify_order() -> Dict:
    """A single fulfilled, paid order for product 1001."""
    return {
        "id": "5001",
        "financial_status": "paid",
        "fulfillment_status": "fulfilled",
        "total_price": "149.99",
        "created_at": "2024-06-01T10:00:00Z",
        "line_items": [
            {"product_id": 1001, "name": "ProPaddle Carbon Elite", "quantity": 1, "price": "149.99"}
        ],
    }


@pytest.fixture
def mock_shopify_customer() -> Dict:
    """A single Shopify customer record matching C001."""
    return {
        "id": "C001",
        "email": "player@example.com",
        "first_name": "Alex",
        "last_name": "Smith",
        "orders_count": 3,
        "total_spent": "389.97",
    }


# ─────────────────────────────────────────────────────────────────────────────
# State Factory Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def make_state():
    """
    Factory fixture for creating PaddleAurumState dicts with optional overrides.

    Usage:
        def test_router(make_state):
            state = make_state("My goal", task_queue=[...])
    """
    def _factory(goal: str = "Test goal", **overrides) -> Dict:
        from workflows.state import initial_state
        state = initial_state(goal=goal)
        state.update(overrides)
        return state
    return _factory


@pytest.fixture
def make_task_item():
    """
    Factory fixture for building TaskItem dicts with full control.

    Usage:
        def test_recovery(make_task_item):
            task = make_task_item("chat_buddy", status="failed", retries=2)
    """
    def _factory(
        assigned_to: str,
        status: str = "pending",
        retries: int = 0,
        max_retries: int = 3,
        description: str = "Test task",
        priority: int = 2,
        input_data: Dict = None,
    ) -> Dict:
        from workflows.state import make_task
        t = make_task(description, assigned_to, priority=priority, input_data=input_data or {})
        t["status"] = status
        t["retries"] = retries
        t["max_retries"] = max_retries
        return t
    return _factory


# ─────────────────────────────────────────────────────────────────────────────
# Memory Manager Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def memory_manager(tmp_db):
    """
    MemoryManager instance wired to the temporary test database.

    Usage:
        async def test_recall(memory_manager):
            await memory_manager.write_session("s1", [...])
    """
    from memory.memory_manager import MemoryManager
    mm = MemoryManager(tmp_db)
    await mm.init_db()
    return mm


# ─────────────────────────────────────────────────────────────────────────────
# Mock Fixtures for External Services
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_smtp(monkeypatch):
    """
    Monkeypatches aiosmtplib.send to avoid real SMTP calls.
    Returns the AsyncMock so tests can assert call counts.
    """
    import tools.email_tool as email_module
    mock_send = AsyncMock(return_value=None)
    monkeypatch.setattr(email_module.aiosmtplib, "send", mock_send)
    return mock_send


@pytest.fixture
def mock_ddg_search():
    """
    Monkeypatches ddg_search to return a canned set of results.
    """
    results = [
        {"title": "Best beginner pickleball paddles", "url": "http://example.com/paddles", "body": "Start with a lightweight 16mm paddle."},
        {"title": "Outdoor ball selection guide",     "url": "http://example.com/balls",   "body": "Use 40-hole balls for outdoor courts."},
    ]
    return results


@pytest.fixture
def mock_browser_page():
    """Canned supplier page text for BrowserTool tests."""
    return "ProPaddle Carbon Elite - $89.99 - In Stock - Add to Cart - Qty: 25 available"
