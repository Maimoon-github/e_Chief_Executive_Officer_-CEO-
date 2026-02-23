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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
