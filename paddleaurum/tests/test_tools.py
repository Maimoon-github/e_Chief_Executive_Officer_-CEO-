"""
tests/test_tools.py
───────────────────
Unit tests for all tool modules.
Uses mocking to avoid real API calls and Shopify dependency.
Run: pytest tests/test_tools.py -v
"""
from __future__ import annotations

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Ensure project root is on path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── DatabaseTool tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_db_init(tmp_path):
    """Database initialises without errors."""
    from config.settings import settings
    test_db = str(tmp_path / "test.db")
    original = settings.sqlite_db_path
    settings.sqlite_db_path = test_db

    from tools.db_tool import init_db
    await init_db()

    import aiosqlite
    async with aiosqlite.connect(test_db) as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
            tables = [r[0] for r in await cur.fetchall()]
    assert "agent_memory" in tables
    assert "customer_profiles" in tables
    assert "restock_thresholds" in tables

    settings.sqlite_db_path = original


@pytest.mark.asyncio
async def test_db_tool_get_customer(tmp_path):
    """DatabaseTool correctly retrieves a customer profile."""
    import aiosqlite
    from config.settings import settings
    test_db = str(tmp_path / "test.db")
    settings.sqlite_db_path = test_db

    from tools.db_tool import init_db, upsert_customer_profile, get_customer_profile
    await init_db()
    await upsert_customer_profile({
        "customer_id": "C999",
        "email": "test@example.com",
        "name": "Test User",
        "segment": "vip",
    })

    profile = await get_customer_profile("C999")
    assert profile is not None
    assert profile["email"] == "test@example.com"
    assert profile["segment"] == "vip"


# ── ShopifyTool tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shopify_tool_get_products():
    """ShopifyTool.get_products returns product list from mocked response."""
    mock_response = {"products": [{"id": 1, "title": "Test Paddle"}]}

    with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_response)):
        from tools.shopify_tool import get_products
        products = await get_products(limit=10)
        assert len(products) == 1
        assert products[0]["title"] == "Test Paddle"


@pytest.mark.asyncio
async def test_shopify_tool_get_order():
    """ShopifyTool.get_order returns order dict from mocked response."""
    mock_order = {"order": {"id": "5001", "financial_status": "paid"}}

    with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_order)):
        from tools.shopify_tool import get_order
        order = await get_order("5001")
        assert order["id"] == "5001"
        assert order["financial_status"] == "paid"


@pytest.mark.asyncio
async def test_shopify_tool_langchain_wrapper():
    """ShopifyTool (BaseTool) dispatches correctly."""
    mock_products = {"products": [{"id": 2, "title": "Ball Pack"}]}

    with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_products)):
        from tools.shopify_tool import ShopifyTool
        tool = ShopifyTool()
        result = await tool._arun(json.dumps({"action": "get_products", "limit": 5}))
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["title"] == "Ball Pack"


# ── SearchTool tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_tool_faq():
    """SearchTool returns non-empty string for FAQ search."""
    mock_results = [{"title": "Best beginner paddles", "url": "http://example.com", "body": "Start with a light paddle."}]

    with patch("tools.search_tool.ddg_search", new=AsyncMock(return_value=mock_results)):
        from tools.search_tool import search_pickleball_faq
        result = await search_pickleball_faq("best paddle for beginners")
        assert "Best beginner paddles" in result


# ── EmailTool tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_email_tool_mock_send():
    """EmailTool send_email returns 'sent' when SMTP succeeds."""
    with patch("tools.email_tool.aiosmtplib.send", new=AsyncMock(return_value=None)):
        from tools.email_tool import send_email
        result = await send_email(
            to="test@example.com",
            subject="Test",
            body_html="<p>Hello</p>",
        )
        assert result["status"] == "sent"


@pytest.mark.asyncio
async def test_email_tool_mock_failure():
    """EmailTool returns 'failed' on SMTP error."""
    with patch("tools.email_tool.aiosmtplib.send", new=AsyncMock(side_effect=Exception("SMTP error"))):
        from tools.email_tool import send_email
        result = await send_email("fail@test.com", "Fail", "<p>x</p>")
        assert result["status"] == "failed"
        assert "SMTP error" in result["error"]


# ── BrowserTool tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_browser_tool_mock():
    """BrowserTool.check_supplier_stock parses mock page text correctly."""
    mock_text = "ProPaddle Carbon Elite - $89.99 - In Stock - Add to Cart"

    with patch("tools.browser_tool._get_page_text", new=AsyncMock(return_value=mock_text)):
        from tools.browser_tool import check_supplier_stock
        result = await check_supplier_stock("http://supplier.example.com", "ProPaddle Carbon Elite")
        assert result["in_stock"] is True
        assert result["price_hint"] == "$89.99"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
