# """
# tests/test_agents.py
# ────────────────────
# Unit tests for all agent node functions.
# Mocks LLM calls and external API calls.
# Run: pytest tests/test_agents.py -v
# """
# from __future__ import annotations

# import json
# import sys
# import os
# from unittest.mock import AsyncMock, MagicMock, patch

# import pytest

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# # ── State factory ─────────────────────────────────────────────────────────────

# def make_state(goal="Test goal", extra_tasks=None):
#     from workflows.state import initial_state, make_task
#     state = initial_state(goal=goal)
#     if extra_tasks:
#         state["task_queue"] = extra_tasks
#     return state


# # ── CEO tests ─────────────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_ceo_decompose_goal():
#     """CEO decomposes a goal into at least one TaskItem."""
#     mock_tasks = [
#         {"description": "Handle support", "assigned_to": "chat_buddy", "priority": 1},
#         {"description": "Check inventory", "assigned_to": "stock_scout", "priority": 2},
#     ]
#     mock_llm = MagicMock()
#     mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_tasks))

#     with patch("agents.ceo.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
#          patch("agents.ceo.memory_manager.get_recent_summaries", new=AsyncMock(return_value=[])):
#         from agents.ceo import decompose_goal
#         tasks = await decompose_goal("Run daily ops", {}, [])
#         assert len(tasks) >= 2
#         assert tasks[0]["assigned_to"] == "chat_buddy"


# @pytest.mark.asyncio
# async def test_ceo_node_first_iteration():
#     """CEO orchestrator node populates task_queue on first call."""
#     state = make_state("Launch new product")
#     state["iteration_count"] = 0

#     mock_tasks = [
#         {"description": "Promote launch", "assigned_to": "promo_general", "priority": 1},
#     ]
#     mock_llm = MagicMock()
#     mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_tasks))

#     with patch("agents.ceo.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
#          patch("agents.ceo.memory_manager.get_recent_summaries", new=AsyncMock(return_value=[])):
#         from agents.ceo import ceo_orchestrator_node
#         result = await ceo_orchestrator_node(state)
#         assert len(result["task_queue"]) >= 1
#         assert result["iteration_count"] == 1


# # ── Chat Buddy tests ──────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_chat_buddy_handles_question():
#     """Chat Buddy returns a valid response for a product question."""
#     mock_response = {
#         "reply_text": "For beginners, we recommend the LightTouch 16mm!",
#         "escalate": False,
#         "sentiment_score": 0.9,
#         "resolution_type": "answered",
#     }
#     mock_llm = MagicMock()
#     mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_response))

#     with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
#          patch("agents.chat_buddy.get_customer_profile", new=AsyncMock(return_value=None)), \
#          patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
#         from agents.chat_buddy import handle_inquiry
#         result = await handle_inquiry("Which paddle for beginners?", "C001")
#         assert result["escalate"] is False
#         assert result["resolution_type"] == "answered"
#         assert "LightTouch" in result["reply_text"]


# @pytest.mark.asyncio
# async def test_chat_buddy_escalates_on_complaint():
#     """Chat Buddy sets escalate=True for a defect complaint."""
#     mock_response = {
#         "reply_text": "I'm so sorry! Let me escalate this to our team.",
#         "escalate": True,
#         "sentiment_score": 0.1,
#         "resolution_type": "escalated",
#     }
#     mock_llm = MagicMock()
#     mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_response))

#     with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
#          patch("agents.chat_buddy.get_customer_profile", new=AsyncMock(return_value=None)), \
#          patch("agents.chat_buddy.get_order", new=AsyncMock(return_value={"id": "5002"})), \
#          patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
#         from agents.chat_buddy import handle_inquiry
#         result = await handle_inquiry("Paddle arrived cracked! Refund!", "C003", "5002")
#         assert result["escalate"] is True


# # ── Stock Scout tests ─────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_stock_scout_detects_low_stock():
#     """Stock Scout flags products below threshold as WARNING or CRITICAL."""
#     mock_products = [{"id": 1001, "title": "Carbon Elite", "variants": [{"inventory_item_id": 3001, "sku": "CE-001"}]}]
#     mock_inventory = [{"inventory_item_id": 3001, "available": 3}]
#     mock_thresholds = [{"product_id": "1001", "min_qty": 10, "reorder_qty": 50, "supplier_url": ""}]
#     mock_velocity = {"avg_daily_sales": 1.5}

#     with patch("agents.stock_scout.get_products", new=AsyncMock(return_value=mock_products)), \
#          patch("agents.stock_scout.get_inventory_levels", new=AsyncMock(return_value=mock_inventory)), \
#          patch("agents.stock_scout.get_all_thresholds", new=AsyncMock(return_value=mock_thresholds)), \
#          patch("agents.stock_scout.get_sales_velocity", new=AsyncMock(return_value=mock_velocity)):
#         from agents.stock_scout import scan_inventory
#         alerts = await scan_inventory()
#         assert len(alerts) >= 1
#         assert alerts[0]["severity"] in ("CRITICAL", "WARNING")
#         assert alerts[0]["current_qty"] == 3


# # ── Recommender tests ─────────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_recommender_generates_sets():
#     """Recommender generates recommendation sets for customers with history."""
#     import numpy as np
#     mock_customers = [
#         {"customer_id": "C001", "purchase_history": [{"product_id": "1001"}], "email": "a@b.com", "segment": "general"},
#         {"customer_id": "C002", "purchase_history": [{"product_id": "1001"}, {"product_id": "1002"}], "email": "c@d.com", "segment": "general"},
#     ]
#     mock_products = [
#         {"id": 1001, "title": "Carbon Paddle", "variants": []},
#         {"id": 1002, "title": "Ball Pack", "variants": []},
#         {"id": 1003, "title": "Bag", "variants": []},
#     ]

#     with patch("agents.recommender.get_customers_by_segment", new=AsyncMock(return_value=mock_customers)), \
#          patch("agents.recommender.get_products", new=AsyncMock(return_value=mock_products)):
#         from agents.recommender import generate_recommendations
#         sets = await generate_recommendations(segment="general")
#         assert len(sets) >= 1
#         assert "recommendations" in sets[0]


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])









































# @###########################################################################################




















"""
tests/test_agents.py
────────────────────
Unit tests for all agent node functions.
Mocks LLM calls and external API calls.
Run: pytest tests/test_agents.py -v
"""
from __future__ import annotations

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── State factory ─────────────────────────────────────────────────────────────

def make_state(goal="Test goal", extra_tasks=None):
    from workflows.state import initial_state, make_task
    state = initial_state(goal=goal)
    if extra_tasks:
        state["task_queue"] = extra_tasks
    return state


# ── CEO tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ceo_decompose_goal():
    """CEO decomposes a goal into at least one TaskItem."""
    mock_tasks = [
        {"description": "Handle support", "assigned_to": "chat_buddy", "priority": 1},
        {"description": "Check inventory", "assigned_to": "stock_scout", "priority": 2},
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_tasks))

    with patch("agents.ceo.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
         patch("agents.ceo.memory_manager.get_recent_summaries", new=AsyncMock(return_value=[])):
        from agents.ceo import decompose_goal
        tasks = await decompose_goal("Run daily ops", {}, [])
        assert len(tasks) >= 2
        assert tasks[0]["assigned_to"] == "chat_buddy"


@pytest.mark.asyncio
async def test_ceo_node_first_iteration():
    """CEO orchestrator node populates task_queue on first call."""
    state = make_state("Launch new product")
    state["iteration_count"] = 0

    mock_tasks = [
        {"description": "Promote launch", "assigned_to": "promo_general", "priority": 1},
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_tasks))

    with patch("agents.ceo.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
         patch("agents.ceo.memory_manager.get_recent_summaries", new=AsyncMock(return_value=[])):
        from agents.ceo import ceo_orchestrator_node
        result = await ceo_orchestrator_node(state)
        assert len(result["task_queue"]) >= 1
        assert result["iteration_count"] == 1


# ── Chat Buddy tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_buddy_handles_question():
    """Chat Buddy returns a valid response for a product question."""
    mock_response = {
        "reply_text": "For beginners, we recommend the LightTouch 16mm!",
        "escalate": False,
        "sentiment_score": 0.9,
        "resolution_type": "answered",
    }
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_response))

    with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
         patch("agents.chat_buddy.get_customer_profile", new=AsyncMock(return_value=None)), \
         patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
        from agents.chat_buddy import handle_inquiry
        result = await handle_inquiry("Which paddle for beginners?", "C001")
        assert result["escalate"] is False
        assert result["resolution_type"] == "answered"
        assert "LightTouch" in result["reply_text"]


@pytest.mark.asyncio
async def test_chat_buddy_escalates_on_complaint():
    """Chat Buddy sets escalate=True for a defect complaint."""
    mock_response = {
        "reply_text": "I'm so sorry! Let me escalate this to our team.",
        "escalate": True,
        "sentiment_score": 0.1,
        "resolution_type": "escalated",
    }
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=json.dumps(mock_response))

    with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
         patch("agents.chat_buddy.get_customer_profile", new=AsyncMock(return_value=None)), \
         patch("agents.chat_buddy.get_order", new=AsyncMock(return_value={"id": "5002"})), \
         patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
        from agents.chat_buddy import handle_inquiry
        result = await handle_inquiry("Paddle arrived cracked! Refund!", "C003", "5002")
        assert result["escalate"] is True


# ── Stock Scout tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stock_scout_detects_low_stock():
    """Stock Scout flags products below threshold as WARNING or CRITICAL."""
    mock_products = [{"id": 1001, "title": "Carbon Elite", "variants": [{"inventory_item_id": 3001, "sku": "CE-001"}]}]
    mock_inventory = [{"inventory_item_id": 3001, "available": 3}]
    mock_thresholds = [{"product_id": "1001", "min_qty": 10, "reorder_qty": 50, "supplier_url": ""}]
    mock_velocity = {"avg_daily_sales": 1.5}

    with patch("agents.stock_scout.get_products", new=AsyncMock(return_value=mock_products)), \
         patch("agents.stock_scout.get_inventory_levels", new=AsyncMock(return_value=mock_inventory)), \
         patch("agents.stock_scout.get_all_thresholds", new=AsyncMock(return_value=mock_thresholds)), \
         patch("agents.stock_scout.get_sales_velocity", new=AsyncMock(return_value=mock_velocity)):
        from agents.stock_scout import scan_inventory
        alerts = await scan_inventory()
        assert len(alerts) >= 1
        assert alerts[0]["severity"] in ("CRITICAL", "WARNING")
        assert alerts[0]["current_qty"] == 3


# ── Recommender tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recommender_generates_sets():
    """Recommender generates recommendation sets for customers with history."""
    import numpy as np
    mock_customers = [
        {"customer_id": "C001", "purchase_history": [{"product_id": "1001"}], "email": "a@b.com", "segment": "general"},
        {"customer_id": "C002", "purchase_history": [{"product_id": "1001"}, {"product_id": "1002"}], "email": "c@d.com", "segment": "general"},
    ]
    mock_products = [
        {"id": 1001, "title": "Carbon Paddle", "variants": []},
        {"id": 1002, "title": "Ball Pack", "variants": []},
        {"id": 1003, "title": "Bag", "variants": []},
    ]

    with patch("agents.recommender.get_customers_by_segment", new=AsyncMock(return_value=mock_customers)), \
         patch("agents.recommender.get_products", new=AsyncMock(return_value=mock_products)):
        from agents.recommender import generate_recommendations
        sets = await generate_recommendations(segment="general")
        assert len(sets) >= 1
        assert "recommendations" in sets[0]


# ── APPENDED: Fixture-based agent tests ───────────────────────────────────────
# These tests leverage conftest.py fixtures for cleaner setup and teardown.

class TestCEOWithFixtures:
    """CEO agent tests using shared conftest fixtures."""

    @pytest.mark.asyncio
    async def test_ceo_fallback_on_invalid_json(self, make_state, make_task_item):
        """CEO falls back to default tasks when LLM returns malformed JSON."""
        state = make_state("Daily ops", iteration_count=0)
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value="NOT VALID JSON {{{{")

        with patch("agents.ceo.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
             patch("agents.ceo.memory_manager.get_recent_summaries", new=AsyncMock(return_value=[])):
            from agents.ceo import decompose_goal
            tasks = await decompose_goal("Daily ops", {}, [])
            # Fallback produces 3 default tasks
            assert len(tasks) == 3
            assigned = {t["assigned_to"] for t in tasks}
            assert "chat_buddy" in assigned
            assert "stock_scout" in assigned

    @pytest.mark.asyncio
    async def test_ceo_synthesis_on_second_iteration(self, make_state, mock_ceo_llm):
        """CEO synthesises final report on iteration_count >= 1."""
        state = make_state("Synthesis test")
        state["iteration_count"] = 1
        state["customer_support_output"] = {"tasks_processed": 2, "resolved": 2, "escalated": 0}
        state["inventory_output"] = {"low_stock_alerts": []}
        state["marketing_output"] = {"campaigns_sent": 1}
        state["recommendation_output"] = {"recommendation_sets": []}

        expected_report = {
            "summary": "All ops ran smoothly.",
            "wins": ["2 support tickets resolved"],
            "blockers": [],
            "next_actions": ["Run promo for VIPs"],
            "timestamp": "2026-02-23T00:00:00Z",
        }

        with patch("agents.ceo.get_llm_async", new=AsyncMock(return_value=mock_ceo_llm)), \
             patch("agents.ceo.memory_manager.write_session_summary", new=AsyncMock()), \
             patch("agents.ceo.memory_manager.get_recent_summaries", new=AsyncMock(return_value=[])):
            mock_ceo_llm.ainvoke = AsyncMock(return_value=json.dumps(expected_report))
            from agents.ceo import synthesise_report
            report = await synthesise_report(state)
            assert "summary" in report
            assert isinstance(report.get("wins"), list)
            assert isinstance(report.get("blockers"), list)

    @pytest.mark.asyncio
    async def test_ceo_short_term_memory_appended(self, make_state, mock_ceo_llm):
        """CEO node appends a message to short_term_memory after decomposition."""
        state = make_state("Memory test", iteration_count=0)
        initial_mem_len = len(state["short_term_memory"])

        with patch("agents.ceo.get_llm_async", new=AsyncMock(return_value=mock_ceo_llm)), \
             patch("agents.ceo.memory_manager.get_recent_summaries", new=AsyncMock(return_value=[])):
            from agents.ceo import ceo_orchestrator_node
            result = await ceo_orchestrator_node(state)

        assert len(result["short_term_memory"]) > initial_mem_len
        last_msg = result["short_term_memory"][-1]
        assert last_msg["agent_id"] == "ceo"
        assert "Decomposed" in last_msg["content"]


class TestChatBuddyWithFixtures:
    """Chat Buddy agent tests using shared conftest fixtures."""

    @pytest.mark.asyncio
    async def test_chat_buddy_with_seeded_customer(
        self, seeded_customers, mock_chat_buddy_llm
    ):
        """Chat Buddy enriches response with customer profile from live test DB."""
        with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_chat_buddy_llm)), \
             patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
            from agents.chat_buddy import handle_inquiry
            result = await handle_inquiry(
                "Which paddle for beginners?", customer_id="C001"
            )
        assert result["escalate"] is False
        assert result["sentiment_score"] >= 0.0
        assert result["resolution_type"] in ("answered", "escalated", "needs_info", "refund_initiated", "order_updated")

    @pytest.mark.asyncio
    async def test_chat_buddy_llm_error_triggers_escalation(self, make_state):
        """Chat Buddy returns escalation=True when LLM call throws."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM offline"))

        with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_llm)), \
             patch("agents.chat_buddy.get_customer_profile", new=AsyncMock(return_value=None)), \
             patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
            from agents.chat_buddy import handle_inquiry
            result = await handle_inquiry("Test message", "C001")

        assert result["escalate"] is True
        assert result["resolution_type"] == "escalated"

    @pytest.mark.asyncio
    async def test_chat_buddy_order_enrichment(self, mock_shopify_order, mock_chat_buddy_llm):
        """Chat Buddy enriches context with Shopify order details."""
        with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_chat_buddy_llm)), \
             patch("agents.chat_buddy.get_customer_profile", new=AsyncMock(return_value=None)), \
             patch("agents.chat_buddy.get_order", new=AsyncMock(return_value=mock_shopify_order)), \
             patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
            from agents.chat_buddy import handle_inquiry
            result = await handle_inquiry("Where is my order?", "C002", "5001")

        assert "resolution_type" in result
        assert isinstance(result["sentiment_score"], float)

    @pytest.mark.asyncio
    async def test_chat_buddy_node_updates_state(self, make_state, make_task_item, mock_chat_buddy_llm):
        """chat_buddy_node processes tasks and updates state correctly."""
        task = make_task_item(
            "chat_buddy",
            input_data={"message": "Which paddle for beginners?", "customer_id": "C001"},
        )
        state = make_state("Support run", task_queue=[task])

        with patch("agents.chat_buddy.get_llm_async", new=AsyncMock(return_value=mock_chat_buddy_llm)), \
             patch("agents.chat_buddy.get_customer_profile", new=AsyncMock(return_value=None)), \
             patch("agents.chat_buddy.write_agent_memory", new=AsyncMock()):
            from agents.chat_buddy import chat_buddy_node
            result = await chat_buddy_node(state)

        assert result["customer_support_output"] is not None
        assert result["customer_support_output"]["tasks_processed"] >= 1


class TestStockScoutWithFixtures:
    """Stock Scout tests using shared conftest fixtures."""

    @pytest.mark.asyncio
    async def test_stock_scout_critical_severity(self, mock_shopify_products, seeded_thresholds):
        """Products with < 7 days stock are flagged CRITICAL."""
        mock_inventory = [
            {"inventory_item_id": 3001, "available": 2},   # 2 units / 1.5 daily = 1.3 days → CRITICAL
            {"inventory_item_id": 3002, "available": 120},
        ]
        mock_velocity = {"avg_daily_sales": 1.5}

        with patch("agents.stock_scout.get_products", new=AsyncMock(return_value=mock_shopify_products)), \
             patch("agents.stock_scout.get_inventory_levels", new=AsyncMock(return_value=mock_inventory)), \
             patch("agents.stock_scout.get_all_thresholds", new=AsyncMock(return_value=seeded_thresholds)), \
             patch("agents.stock_scout.get_sales_velocity", new=AsyncMock(return_value=mock_velocity)):
            from agents.stock_scout import scan_inventory
            alerts = await scan_inventory()

        critical = [a for a in alerts if a["severity"] == "CRITICAL"]
        assert len(critical) >= 1
        assert critical[0]["product_id"] == "1001"

    @pytest.mark.asyncio
    async def test_stock_scout_ok_products_excluded(self, mock_shopify_products, seeded_thresholds):
        """Products above threshold produce no alerts."""
        mock_inventory = [
            {"inventory_item_id": 3001, "available": 150},
            {"inventory_item_id": 3002, "available": 200},
        ]

        with patch("agents.stock_scout.get_products", new=AsyncMock(return_value=mock_shopify_products)), \
             patch("agents.stock_scout.get_inventory_levels", new=AsyncMock(return_value=mock_inventory)), \
             patch("agents.stock_scout.get_all_thresholds", new=AsyncMock(return_value=seeded_thresholds)), \
             patch("agents.stock_scout.get_sales_velocity", new=AsyncMock(return_value=None)):
            from agents.stock_scout import scan_inventory
            alerts = await scan_inventory()

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_stock_scout_node_populates_state(
        self, make_state, mock_shopify_products, seeded_thresholds
    ):
        """stock_scout_node writes inventory_output into state."""
        mock_inventory = [{"inventory_item_id": 3001, "available": 3}, {"inventory_item_id": 3002, "available": 120}]
        mock_velocity = {"avg_daily_sales": 2.0}

        state = make_state("Inventory check")

        with patch("agents.stock_scout.get_products", new=AsyncMock(return_value=mock_shopify_products)), \
             patch("agents.stock_scout.get_inventory_levels", new=AsyncMock(return_value=mock_inventory)), \
             patch("agents.stock_scout.get_all_thresholds", new=AsyncMock(return_value=seeded_thresholds)), \
             patch("agents.stock_scout.get_sales_velocity", new=AsyncMock(return_value=mock_velocity)), \
             patch("agents.stock_scout.check_supplier_stock", new=AsyncMock(return_value={"in_stock": True})):
            from agents.stock_scout import stock_scout_node
            result = await stock_scout_node(state)

        assert result["inventory_output"] is not None
        assert "low_stock_alerts" in result["inventory_output"]
        assert isinstance(result["inventory_output"]["low_stock_alerts"], list)


class TestRecommenderWithFixtures:
    """Recommender tests using shared conftest fixtures."""

    @pytest.mark.asyncio
    async def test_recommender_cross_sell_reason_code(self, mock_shopify_products):
        """Recommender assigns cross_sell reason when customer has purchase history."""
        mock_customers = [
            {"customer_id": "C001", "purchase_history": [{"product_id": "1001"}], "segment": "vip"},
        ]

        with patch("agents.recommender.get_customers_by_segment", new=AsyncMock(return_value=mock_customers)), \
             patch("agents.recommender.get_products", new=AsyncMock(return_value=mock_shopify_products)):
            from agents.recommender import generate_recommendations
            sets = await generate_recommendations(segment="vip")

        assert len(sets) >= 1
        recs = sets[0]["recommendations"]
        assert any(r["reason_code"] in ("cross_sell", "trending", "upsell") for r in recs)

    @pytest.mark.asyncio
    async def test_recommender_empty_segment_returns_empty(self):
        """Recommender returns empty list when segment has no customers."""
        with patch("agents.recommender.get_customers_by_segment", new=AsyncMock(return_value=[])):
            from agents.recommender import generate_recommendations
            sets = await generate_recommendations(segment="nonexistent")

        assert sets == []

    @pytest.mark.asyncio
    async def test_recommender_node_updates_state(self, make_state, make_task_item, mock_shopify_products):
        """recommender_node writes recommendation_output to state."""
        task = make_task_item("recommender", input_data={"segment": "general"})
        state = make_state("Recs run", task_queue=[task])
        mock_customers = [
            {"customer_id": "C001", "purchase_history": [{"product_id": "1001"}], "segment": "general"},
        ]

        with patch("agents.recommender.get_customers_by_segment", new=AsyncMock(return_value=mock_customers)), \
             patch("agents.recommender.get_products", new=AsyncMock(return_value=mock_shopify_products)):
            from agents.recommender import recommender_node
            result = await recommender_node(state)

        assert result["recommendation_output"] is not None
        assert "recommendation_sets" in result["recommendation_output"]
        assert "total_customers_recommended" in result["recommendation_output"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])