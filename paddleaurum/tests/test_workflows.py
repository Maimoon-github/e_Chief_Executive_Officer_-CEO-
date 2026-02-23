"""
tests/test_workflows.py
───────────────────────
Integration tests for the LangGraph workflow.
Tests state transitions, routing logic, and error recovery.
Run: pytest tests/test_workflows.py -v
"""
from __future__ import annotations

import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── State helpers ─────────────────────────────────────────────────────────────

def _make_task(assigned_to, status="pending", retries=0, max_retries=3):
    from workflows.state import make_task
    t = make_task("Test task", assigned_to)
    t["status"] = status
    t["retries"] = retries
    t["max_retries"] = max_retries
    return t


# ── Router tests ──────────────────────────────────────────────────────────────

def test_route_from_ceo_support_only():
    """CEO router returns customer_captain for support tasks."""
    from workflows.state import initial_state
    from workflows.router import route_from_ceo
    state = initial_state("Handle support tickets")
    state["task_queue"] = [_make_task("chat_buddy")]
    routes = route_from_ceo(state)
    assert "customer_captain" in routes


def test_route_from_ceo_all_branches():
    """CEO router fans out to all 3 branches when tasks are mixed."""
    from workflows.state import initial_state
    from workflows.router import route_from_ceo
    state = initial_state("Full ops run")
    state["task_queue"] = [
        _make_task("chat_buddy"),
        _make_task("stock_scout"),
        _make_task("recommender"),
    ]
    routes = route_from_ceo(state)
    assert set(routes) == {"customer_captain", "stock_sergeant", "promo_general"}


def test_route_from_ceo_empty_queue():
    """CEO router falls back to aggregator when no tasks exist."""
    from workflows.state import initial_state
    from workflows.router import route_from_ceo
    state = initial_state("Empty goal")
    state["task_queue"] = []
    routes = route_from_ceo(state)
    assert "aggregator" in routes


def test_route_post_aggregation_no_failures():
    """Post-aggregation routes to memory_persist when no failures."""
    from workflows.state import initial_state
    from workflows.router import route_post_aggregation
    state = initial_state("Test")
    state["task_queue"] = [_make_task("chat_buddy", status="done")]
    result = route_post_aggregation(state)
    assert result == "memory_persist"


def test_route_post_aggregation_with_retryable_failure():
    """Post-aggregation routes to error_recovery when retries remain."""
    from workflows.state import initial_state
    from workflows.router import route_post_aggregation
    state = initial_state("Test")
    state["task_queue"] = [_make_task("chat_buddy", status="failed", retries=0, max_retries=3)]
    result = route_post_aggregation(state)
    assert result == "error_recovery"


def test_route_post_aggregation_exhausted_retries():
    """Post-aggregation routes to memory_persist when retries exhausted."""
    from workflows.state import initial_state
    from workflows.router import route_post_aggregation
    state = initial_state("Test")
    state["task_queue"] = [_make_task("chat_buddy", status="failed", retries=3, max_retries=3)]
    result = route_post_aggregation(state)
    assert result == "memory_persist"


def test_route_post_error_recovery_replan():
    """Error recovery re-routes to CEO when tasks pending and iter < 3."""
    from workflows.state import initial_state
    from workflows.router import route_post_error_recovery
    state = initial_state("Test")
    state["task_queue"] = [_make_task("stock_scout", status="pending")]
    state["iteration_count"] = 1
    result = route_post_error_recovery(state)
    assert result == "ceo_orchestrator"


def test_route_post_error_recovery_max_iter():
    """Error recovery skips to memory_persist at max iterations."""
    from workflows.state import initial_state
    from workflows.router import route_post_error_recovery
    state = initial_state("Test")
    state["task_queue"] = [_make_task("stock_scout", status="pending")]
    state["iteration_count"] = 3
    result = route_post_error_recovery(state)
    assert result == "memory_persist"


# ── Error recovery node test ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_error_recovery_resets_task():
    """Error recovery node resets a failed task to pending for retry."""
    from workflows.state import initial_state
    from workflows.graph import error_recovery_node
    state = initial_state("Test")
    task = _make_task("chat_buddy", status="failed", retries=0, max_retries=3)
    task["error_message"] = "Timeout"
    state["task_queue"] = [task]

    # Patch sleep to avoid actual waiting in tests
    with patch("asyncio.sleep", new=AsyncMock()):
        result = await error_recovery_node(state)

    assert result["task_queue"][0]["status"] == "pending"
    assert result["task_queue"][0]["retries"] == 1
    assert len(result["error_log"]) == 1


@pytest.mark.asyncio
async def test_error_recovery_skips_exhausted():
    """Error recovery skips tasks that have hit max_retries."""
    from workflows.state import initial_state
    from workflows.graph import error_recovery_node
    state = initial_state("Test")
    task = _make_task("chat_buddy", status="failed", retries=3, max_retries=3)
    state["task_queue"] = [task]

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await error_recovery_node(state)

    assert result["task_queue"][0]["status"] == "skipped"
    assert len(result["failed_tasks"]) == 1


# ── State schema test ─────────────────────────────────────────────────────────

def test_initial_state_has_required_keys():
    """initial_state() produces a dict with all required PaddleAurumState keys."""
    from workflows.state import initial_state
    state = initial_state("Test goal")
    required_keys = [
        "session_id", "goal", "task_queue", "completed_tasks", "failed_tasks",
        "iteration_count", "shared_context", "short_term_memory", "error_log",
    ]
    for key in required_keys:
        assert key in state, f"Missing key: {key}"
    assert state["goal"] == "Test goal"
    assert state["iteration_count"] == 0
    assert isinstance(state["task_queue"], list)


def test_make_task_defaults():
    """make_task() creates a valid TaskItem with sensible defaults."""
    from workflows.state import make_task
    task = make_task("Do something", "chat_buddy", priority=2)
    assert task["status"] == "pending"
    assert task["retries"] == 0
    assert task["max_retries"] == 3
    assert task["priority"] == 2
    assert task["assigned_to"] == "chat_buddy"


# ── Memory manager test ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_manager_write_recall(tmp_path):
    """MemoryManager writes and recalls agent messages correctly."""
    from config.settings import settings
    settings.sqlite_db_path = str(tmp_path / "mem.db")

    from memory.memory_manager import MemoryManager
    mm = MemoryManager(str(tmp_path / "mem.db"))
    await mm.init_db()

    messages = [
        {"agent_id": "chat_buddy", "role": "assistant", "content": "Test message", "timestamp": time.time()},
    ]
    ids = await mm.write_session("sess-001", messages)
    assert len(ids) == 1

    recalled = await mm.recall("chat_buddy", limit=5)
    assert len(recalled) == 1
    assert recalled[0] == "Test message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
