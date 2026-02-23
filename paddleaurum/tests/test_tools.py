# """
# tests/test_tools.py
# ───────────────────
# Unit tests for all tool modules.
# Uses mocking to avoid real API calls and Shopify dependency.
# Run: pytest tests/test_tools.py -v
# """
# from __future__ import annotations

# import json
# import sys
# import os
# from unittest.mock import AsyncMock, MagicMock, patch

# import pytest

# # ── Ensure project root is on path ───────────────────────────────────────────
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# # ── DatabaseTool tests ────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_db_init(tmp_path):
#     """Database initialises without errors."""
#     from config.settings import settings
#     test_db = str(tmp_path / "test.db")
#     original = settings.sqlite_db_path
#     settings.sqlite_db_path = test_db

#     from tools.db_tool import init_db
#     await init_db()

#     import aiosqlite
#     async with aiosqlite.connect(test_db) as db:
#         async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
#             tables = [r[0] for r in await cur.fetchall()]
#     assert "agent_memory" in tables
#     assert "customer_profiles" in tables
#     assert "restock_thresholds" in tables

#     settings.sqlite_db_path = original


# @pytest.mark.asyncio
# async def test_db_tool_get_customer(tmp_path):
#     """DatabaseTool correctly retrieves a customer profile."""
#     import aiosqlite
#     from config.settings import settings
#     test_db = str(tmp_path / "test.db")
#     settings.sqlite_db_path = test_db

#     from tools.db_tool import init_db, upsert_customer_profile, get_customer_profile
#     await init_db()
#     await upsert_customer_profile({
#         "customer_id": "C999",
#         "email": "test@example.com",
#         "name": "Test User",
#         "segment": "vip",
#     })

#     profile = await get_customer_profile("C999")
#     assert profile is not None
#     assert profile["email"] == "test@example.com"
#     assert profile["segment"] == "vip"


# # ── ShopifyTool tests ─────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_shopify_tool_get_products():
#     """ShopifyTool.get_products returns product list from mocked response."""
#     mock_response = {"products": [{"id": 1, "title": "Test Paddle"}]}

#     with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_response)):
#         from tools.shopify_tool import get_products
#         products = await get_products(limit=10)
#         assert len(products) == 1
#         assert products[0]["title"] == "Test Paddle"


# @pytest.mark.asyncio
# async def test_shopify_tool_get_order():
#     """ShopifyTool.get_order returns order dict from mocked response."""
#     mock_order = {"order": {"id": "5001", "financial_status": "paid"}}

#     with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_order)):
#         from tools.shopify_tool import get_order
#         order = await get_order("5001")
#         assert order["id"] == "5001"
#         assert order["financial_status"] == "paid"


# @pytest.mark.asyncio
# async def test_shopify_tool_langchain_wrapper():
#     """ShopifyTool (BaseTool) dispatches correctly."""
#     mock_products = {"products": [{"id": 2, "title": "Ball Pack"}]}

#     with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_products)):
#         from tools.shopify_tool import ShopifyTool
#         tool = ShopifyTool()
#         result = await tool._arun(json.dumps({"action": "get_products", "limit": 5}))
#         data = json.loads(result)
#         assert isinstance(data, list)
#         assert data[0]["title"] == "Ball Pack"


# # ── SearchTool tests ──────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_search_tool_faq():
#     """SearchTool returns non-empty string for FAQ search."""
#     mock_results = [{"title": "Best beginner paddles", "url": "http://example.com", "body": "Start with a light paddle."}]

#     with patch("tools.search_tool.ddg_search", new=AsyncMock(return_value=mock_results)):
#         from tools.search_tool import search_pickleball_faq
#         result = await search_pickleball_faq("best paddle for beginners")
#         assert "Best beginner paddles" in result


# # ── EmailTool tests ───────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_email_tool_mock_send():
#     """EmailTool send_email returns 'sent' when SMTP succeeds."""
#     with patch("tools.email_tool.aiosmtplib.send", new=AsyncMock(return_value=None)):
#         from tools.email_tool import send_email
#         result = await send_email(
#             to="test@example.com",
#             subject="Test",
#             body_html="<p>Hello</p>",
#         )
#         assert result["status"] == "sent"


# @pytest.mark.asyncio
# async def test_email_tool_mock_failure():
#     """EmailTool returns 'failed' on SMTP error."""
#     with patch("tools.email_tool.aiosmtplib.send", new=AsyncMock(side_effect=Exception("SMTP error"))):
#         from tools.email_tool import send_email
#         result = await send_email("fail@test.com", "Fail", "<p>x</p>")
#         assert result["status"] == "failed"
#         assert "SMTP error" in result["error"]


# # ── BrowserTool tests ─────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_browser_tool_mock():
#     """BrowserTool.check_supplier_stock parses mock page text correctly."""
#     mock_text = "ProPaddle Carbon Elite - $89.99 - In Stock - Add to Cart"

#     with patch("tools.browser_tool._get_page_text", new=AsyncMock(return_value=mock_text)):
#         from tools.browser_tool import check_supplier_stock
#         result = await check_supplier_stock("http://supplier.example.com", "ProPaddle Carbon Elite")
#         assert result["in_stock"] is True
#         assert result["price_hint"] == "$89.99"


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])



























# @##############################################################################################















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


# ── APPENDED: Fixture-based tool tests ───────────────────────────────────────
# These tests use conftest.py fixtures (tmp_db, seeded_customers, mock_smtp, etc.)
# for consistent, isolated, database-backed testing.

class TestDatabaseToolWithFixtures:
    """Database tool tests using the shared tmp_db fixture."""

    @pytest.mark.asyncio
    async def test_all_required_tables_created(self, tmp_db):
        """init_db creates all five required tables."""
        import aiosqlite
        async with aiosqlite.connect(tmp_db) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cur:
                tables = {r[0] for r in await cur.fetchall()}

        required = {
            "agent_memory", "customer_profiles",
            "restock_thresholds", "sales_velocity", "promo_log",
        }
        assert required.issubset(tables), f"Missing tables: {required - tables}"

    @pytest.mark.asyncio
    async def test_upsert_customer_idempotent(self, tmp_db, seeded_customers):
        """Upserting the same customer twice does not create duplicates."""
        from tools.db_tool import upsert_customer_profile, get_customer_profile
        import aiosqlite

        updated = {**seeded_customers[0], "lifetime_value": 999.99}
        await upsert_customer_profile(updated)

        profile = await get_customer_profile("C001")
        assert profile["lifetime_value"] == 999.99

        async with aiosqlite.connect(tmp_db) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM customer_profiles WHERE customer_id = 'C001'"
            ) as cur:
                count = (await cur.fetchone())[0]
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_customers_by_segment(self, tmp_db, seeded_customers):
        """get_customers_by_segment filters correctly."""
        from tools.db_tool import get_customers_by_segment

        vip_customers = await get_customers_by_segment("vip")
        general_customers = await get_customers_by_segment("general")

        assert len(vip_customers) == 1
        assert vip_customers[0]["customer_id"] == "C001"
        assert len(general_customers) == 2

    @pytest.mark.asyncio
    async def test_upsert_and_get_threshold(self, tmp_db, seeded_thresholds):
        """get_restock_threshold returns correct threshold for a product."""
        from tools.db_tool import get_restock_threshold

        threshold = await get_restock_threshold("1001")
        assert threshold is not None
        assert threshold["min_qty"] == 10
        assert threshold["reorder_qty"] == 50

    @pytest.mark.asyncio
    async def test_get_all_thresholds(self, tmp_db, seeded_thresholds):
        """get_all_thresholds returns all seeded thresholds."""
        from tools.db_tool import get_all_thresholds

        thresholds = await get_all_thresholds()
        assert len(thresholds) == len(seeded_thresholds)
        product_ids = {t["product_id"] for t in thresholds}
        assert "1001" in product_ids
        assert "1002" in product_ids

    @pytest.mark.asyncio
    async def test_write_and_recall_agent_memory(self, tmp_db):
        """write_agent_memory and recall_agent_memory round-trip correctly."""
        from tools.db_tool import write_agent_memory, recall_agent_memory

        await write_agent_memory("sess-42", "chat_buddy", "assistant", "Handled ticket #7")
        await write_agent_memory("sess-42", "chat_buddy", "assistant", "Handled ticket #8")

        recalled = await recall_agent_memory("chat_buddy", limit=5)
        assert len(recalled) == 2
        assert "ticket #8" in recalled[0]  # Most recent first

    @pytest.mark.asyncio
    async def test_log_campaign(self, tmp_db):
        """log_campaign inserts a promo log entry."""
        import aiosqlite
        from tools.db_tool import log_campaign

        await log_campaign({
            "campaign_id": "CAMP-001",
            "campaign_type": "email",
            "subject_line": "Summer Pickleball Sale 🏓",
            "target_segment": "general",
            "sent_count": 150,
        })

        async with aiosqlite.connect(tmp_db) as db:
            async with db.execute(
                "SELECT subject_line, sent_count FROM promo_log WHERE campaign_id = 'CAMP-001'"
            ) as cur:
                row = await cur.fetchone()

        assert row is not None
        assert row[0] == "Summer Pickleball Sale 🏓"
        assert row[1] == 150

    @pytest.mark.asyncio
    async def test_database_tool_langchain_wrapper(self, tmp_db, seeded_customers):
        """DatabaseTool (BaseTool) dispatches get_customer correctly."""
        from tools.db_tool import DatabaseTool

        tool = DatabaseTool()
        result = await tool._arun(json.dumps({"action": "get_customer", "customer_id": "C001"}))
        data = json.loads(result)
        assert data is not None
        assert data.get("email") == "player@example.com"

    @pytest.mark.asyncio
    async def test_database_tool_unknown_action(self, tmp_db):
        """DatabaseTool returns error dict for unknown action."""
        from tools.db_tool import DatabaseTool

        tool = DatabaseTool()
        result = await tool._arun(json.dumps({"action": "drop_table", "table": "agent_memory"}))
        data = json.loads(result)
        assert "error" in data
        assert "Unknown action" in data["error"]


class TestShopifyToolWithFixtures:
    """Shopify tool tests using shared fixture data."""

    @pytest.mark.asyncio
    async def test_get_products_uses_fixture_data(self, mock_shopify_products):
        """get_products returns all products from fixture."""
        mock_resp = {"products": mock_shopify_products}
        with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_resp)):
            from tools.shopify_tool import get_products
            products = await get_products(limit=50)

        assert len(products) == 2
        titles = [p["title"] for p in products]
        assert "ProPaddle Carbon Elite" in titles
        assert "SpeedBall 40-Hole Outdoor 3-Pack" in titles

    @pytest.mark.asyncio
    async def test_get_inventory_levels_maps_correctly(self, mock_inventory_levels):
        """get_inventory_levels returns inventory_levels list."""
        mock_resp = {"inventory_levels": mock_inventory_levels}
        with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_resp)):
            from tools.shopify_tool import get_inventory_levels
            levels = await get_inventory_levels()

        assert len(levels) == 2
        assert levels[0]["available"] == 5
        assert levels[1]["available"] == 120

    @pytest.mark.asyncio
    async def test_shopify_tool_get_customer_dispatch(self, mock_shopify_customer):
        """ShopifyTool dispatches get_customer action correctly."""
        mock_resp = {"customer": mock_shopify_customer}
        with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_resp)):
            from tools.shopify_tool import ShopifyTool
            tool = ShopifyTool()
            result = await tool._arun(json.dumps({"action": "get_customer", "customer_id": "C001"}))
            data = json.loads(result)

        assert data["email"] == "player@example.com"
        assert data["first_name"] == "Alex"

    @pytest.mark.asyncio
    async def test_shopify_tool_get_order_dispatch(self, mock_shopify_order):
        """ShopifyTool dispatches get_order action correctly."""
        mock_resp = {"order": mock_shopify_order}
        with patch("tools.shopify_tool._shopify_get", new=AsyncMock(return_value=mock_resp)):
            from tools.shopify_tool import ShopifyTool
            tool = ShopifyTool()
            result = await tool._arun(json.dumps({"action": "get_order", "order_id": "5001"}))
            data = json.loads(result)

        assert data["financial_status"] == "paid"
        assert data["fulfillment_status"] == "fulfilled"

    @pytest.mark.asyncio
    async def test_shopify_tool_unknown_action_error(self):
        """ShopifyTool returns error JSON for unrecognised action."""
        from tools.shopify_tool import ShopifyTool
        tool = ShopifyTool()
        result = await tool._arun(json.dumps({"action": "delete_all_products"}))
        data = json.loads(result)
        assert "error" in data
        assert "Unknown action" in data["error"]


class TestEmailToolWithFixtures:
    """Email tool tests using the mock_smtp monkeypatch fixture."""

    @pytest.mark.asyncio
    async def test_send_email_called_once_per_recipient(self, mock_smtp):
        """send_email invokes SMTP send exactly once for a single recipient."""
        from tools.email_tool import send_email
        result = await send_email(
            to="player@example.com",
            subject="Your order shipped!",
            body_html="<p>It's on its way.</p>",
        )
        mock_smtp.assert_awaited_once()
        assert result["status"] == "sent"

    @pytest.mark.asyncio
    async def test_send_promo_campaign_batches_correctly(self, mock_smtp):
        """send_promo_campaign splits 110 recipients into 3 SMTP batches (50+50+10)."""
        from tools.email_tool import send_promo_campaign
        recipients = [f"user{i}@example.com" for i in range(110)]
        result = await send_promo_campaign(
            recipients=recipients,
            subject="Big Paddle Sale 🏓",
            headline="Summer Sale",
            body="Get 20% off all paddles this week.",
            cta_text="Shop Now",
            cta_url="https://paddleaurum.com/sale",
            discount_code="SUMMER20",
        )
        assert result["batches_sent"] == 3          # ceil(110 / 50) = 3
        assert result["total_recipients"] == 110
        assert mock_smtp.await_count == 3

    @pytest.mark.asyncio
    async def test_send_support_reply_structure(self, mock_smtp):
        """send_support_reply produces a valid email with correct subject."""
        from tools.email_tool import send_support_reply
        result = await send_support_reply(
            customer_email="player@example.com",
            customer_name="Alex",
            reply_text="Your refund has been processed.",
            original_subject="Cracked paddle",
        )
        assert result["status"] == "sent"
        # Verify subject contains Re:
        call_args = mock_smtp.call_args
        msg = call_args[0][0]  # first positional arg = MIMEMultipart
        assert "Re:" in msg["Subject"]

    @pytest.mark.asyncio
    async def test_send_alert_targets_admin_email(self, mock_smtp):
        """send_alert sends to the configured alert_email address."""
        from config.settings import settings
        from tools.email_tool import send_alert
        result = await send_alert("Restock Required", "Carbon Elite is at 2 units.")
        assert result["status"] == "sent"
        assert settings.alert_email in result["recipients"]

    @pytest.mark.asyncio
    async def test_email_tool_langchain_alert_dispatch(self, mock_smtp):
        """EmailTool (BaseTool) dispatches alert action correctly."""
        from tools.email_tool import EmailTool
        tool = EmailTool()
        result = await tool._arun(json.dumps({
            "action": "alert",
            "subject": "Test Alert",
            "message": "This is a test.",
        }))
        data = json.loads(result)
        assert data["status"] == "sent"


class TestSearchToolWithFixtures:
    """Search tool tests using mock_ddg_search fixture."""

    @pytest.mark.asyncio
    async def test_search_pickleball_faq_formats_results(self, mock_ddg_search):
        """search_pickleball_faq formats multiple results with titles and URLs."""
        with patch("tools.search_tool.ddg_search", new=AsyncMock(return_value=mock_ddg_search)):
            from tools.search_tool import search_pickleball_faq
            result = await search_pickleball_faq("outdoor ball selection")

        assert "Best beginner pickleball paddles" in result
        assert "http://example.com" in result

    @pytest.mark.asyncio
    async def test_search_competitor_prices_formats_results(self, mock_ddg_search):
        """search_competitor_prices returns formatted competitor pricing string."""
        with patch("tools.search_tool.ddg_search", new=AsyncMock(return_value=mock_ddg_search)):
            from tools.search_tool import search_competitor_prices
            result = await search_competitor_prices("pickleball paddles")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_search_tool_empty_results(self):
        """search_pickleball_faq handles empty DDG results gracefully."""
        with patch("tools.search_tool.ddg_search", new=AsyncMock(return_value=[])):
            from tools.search_tool import search_pickleball_faq
            result = await search_pickleball_faq("xyzzy unknown query")

        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_search_tool_langchain_faq_dispatch(self, mock_ddg_search):
        """SearchTool (BaseTool) dispatches faq_search action."""
        with patch("tools.search_tool.ddg_search", new=AsyncMock(return_value=mock_ddg_search)):
            from tools.search_tool import SearchTool
            tool = SearchTool()
            result = await tool._arun(json.dumps({"action": "faq_search", "query": "beginner paddle"}))

        assert "Best beginner pickleball paddles" in result


class TestBrowserToolWithFixtures:
    """Browser tool tests using mock_browser_page fixture."""

    @pytest.mark.asyncio
    async def test_check_supplier_in_stock_detection(self, mock_browser_page):
        """check_supplier_stock detects in-stock from 'Add to Cart' keyword."""
        with patch("tools.browser_tool._get_page_text", new=AsyncMock(return_value=mock_browser_page)):
            from tools.browser_tool import check_supplier_stock
            result = await check_supplier_stock(
                "http://supplier.example.com/ce001", "ProPaddle Carbon Elite"
            )

        assert result["in_stock"] is True
        assert result["price_hint"] == "$89.99"
        assert result["error"] is False

    @pytest.mark.asyncio
    async def test_check_supplier_out_of_stock(self):
        """check_supplier_stock returns in_stock=False on 'Out of Stock' text."""
        oos_text = "ProPaddle Carbon Elite - $89.99 - Out of Stock - Notify Me"
        with patch("tools.browser_tool._get_page_text", new=AsyncMock(return_value=oos_text)):
            from tools.browser_tool import check_supplier_stock
            result = await check_supplier_stock("http://supplier.example.com", "Carbon Elite")

        assert result["in_stock"] is False

    @pytest.mark.asyncio
    async def test_check_supplier_page_error(self):
        """check_supplier_stock handles Playwright errors gracefully."""
        error_text = "ERROR: Could not scrape http://bad-url.com. Reason: timeout"
        with patch("tools.browser_tool._get_page_text", new=AsyncMock(return_value=error_text)):
            from tools.browser_tool import check_supplier_stock
            result = await check_supplier_stock("http://bad-url.com", "Some Product")

        assert result["error"] is True
        assert result["in_stock"] is None

    @pytest.mark.asyncio
    async def test_scrape_product_listing_extracts_prices(self, mock_browser_page):
        """scrape_product_listing finds price patterns on a page."""
        with patch("tools.browser_tool._get_page_text", new=AsyncMock(return_value=mock_browser_page)):
            from tools.browser_tool import scrape_product_listing
            result = await scrape_product_listing("http://competitor.example.com/paddle")

        assert "$89.99" in result["prices_found"]
        assert result["error"] is False

    @pytest.mark.asyncio
    async def test_browser_tool_langchain_dispatch(self, mock_browser_page):
        """BrowserTool (BaseTool) dispatches check_supplier_stock correctly."""
        with patch("tools.browser_tool._get_page_text", new=AsyncMock(return_value=mock_browser_page)):
            from tools.browser_tool import BrowserTool
            tool = BrowserTool()
            result = await tool._arun(json.dumps({
                "action": "check_supplier_stock",
                "url": "http://supplier.example.com",
                "product_name": "ProPaddle Carbon Elite",
            }))
            data = json.loads(result)

        assert data["in_stock"] is True
        assert data["price_hint"] == "$89.99"

    @pytest.mark.asyncio
    async def test_browser_tool_unknown_action(self):
        """BrowserTool returns error JSON for unknown action."""
        from tools.browser_tool import BrowserTool
        tool = BrowserTool()
        result = await tool._arun(json.dumps({"action": "download_file", "url": "http://x.com/file"}))
        data = json.loads(result)
        assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])