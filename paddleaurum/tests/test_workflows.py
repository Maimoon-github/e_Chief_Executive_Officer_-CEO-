# """
# tests/test_workflows.py
# ───────────────────────
# Integration tests for the LangGraph workflow.
# Tests state transitions, routing logic, and error recovery.
# Run: pytest tests/test_workflows.py -v
# """
# from __future__ import annotations

# import sys
# import os
# import time
# from unittest.mock import AsyncMock, MagicMock, patch

# import pytest

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# # ── State helpers ─────────────────────────────────────────────────────────────

# def _make_task(assigned_to, status="pending", retries=0, max_retries=3):
#     from workflows.state import make_task
#     t = make_task("Test task", assigned_to)
#     t["status"] = status
#     t["retries"] = retries
#     t["max_retries"] = max_retries
#     return t


# # ── Router tests ──────────────────────────────────────────────────────────────

# def test_route_from_ceo_support_only():
#     """CEO router returns customer_captain for support tasks."""
#     from workflows.state import initial_state
#     from workflows.router import route_from_ceo
#     state = initial_state("Handle support tickets")
#     state["task_queue"] = [_make_task("chat_buddy")]
#     routes = route_from_ceo(state)
#     assert "customer_captain" in routes


# def test_route_from_ceo_all_branches():
#     """CEO router fans out to all 3 branches when tasks are mixed."""
#     from workflows.state import initial_state
#     from workflows.router import route_from_ceo
#     state = initial_state("Full ops run")
#     state["task_queue"] = [
#         _make_task("chat_buddy"),
#         _make_task("stock_scout"),
#         _make_task("recommender"),
#     ]
#     routes = route_from_ceo(state)
#     assert set(routes) == {"customer_captain", "stock_sergeant", "promo_general"}


# def test_route_from_ceo_empty_queue():
#     """CEO router falls back to aggregator when no tasks exist."""
#     from workflows.state import initial_state
#     from workflows.router import route_from_ceo
#     state = initial_state("Empty goal")
#     state["task_queue"] = []
#     routes = route_from_ceo(state)
#     assert "aggregator" in routes


# def test_route_post_aggregation_no_failures():
#     """Post-aggregation routes to memory_persist when no failures."""
#     from workflows.state import initial_state
#     from workflows.router import route_post_aggregation
#     state = initial_state("Test")
#     state["task_queue"] = [_make_task("chat_buddy", status="done")]
#     result = route_post_aggregation(state)
#     assert result == "memory_persist"


# def test_route_post_aggregation_with_retryable_failure():
#     """Post-aggregation routes to error_recovery when retries remain."""
#     from workflows.state import initial_state
#     from workflows.router import route_post_aggregation
#     state = initial_state("Test")
#     state["task_queue"] = [_make_task("chat_buddy", status="failed", retries=0, max_retries=3)]
#     result = route_post_aggregation(state)
#     assert result == "error_recovery"


# def test_route_post_aggregation_exhausted_retries():
#     """Post-aggregation routes to memory_persist when retries exhausted."""
#     from workflows.state import initial_state
#     from workflows.router import route_post_aggregation
#     state = initial_state("Test")
#     state["task_queue"] = [_make_task("chat_buddy", status="failed", retries=3, max_retries=3)]
#     result = route_post_aggregation(state)
#     assert result == "memory_persist"


# def test_route_post_error_recovery_replan():
#     """Error recovery re-routes to CEO when tasks pending and iter < 3."""
#     from workflows.state import initial_state
#     from workflows.router import route_post_error_recovery
#     state = initial_state("Test")
#     state["task_queue"] = [_make_task("stock_scout", status="pending")]
#     state["iteration_count"] = 1
#     result = route_post_error_recovery(state)
#     assert result == "ceo_orchestrator"


# def test_route_post_error_recovery_max_iter():
#     """Error recovery skips to memory_persist at max iterations."""
#     from workflows.state import initial_state
#     from workflows.router import route_post_error_recovery
#     state = initial_state("Test")
#     state["task_queue"] = [_make_task("stock_scout", status="pending")]
#     state["iteration_count"] = 3
#     result = route_post_error_recovery(state)
#     assert result == "memory_persist"


# # ── Error recovery node test ──────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_error_recovery_resets_task():
#     """Error recovery node resets a failed task to pending for retry."""
#     from workflows.state import initial_state
#     from workflows.graph import error_recovery_node
#     state = initial_state("Test")
#     task = _make_task("chat_buddy", status="failed", retries=0, max_retries=3)
#     task["error_message"] = "Timeout"
#     state["task_queue"] = [task]

#     # Patch sleep to avoid actual waiting in tests
#     with patch("asyncio.sleep", new=AsyncMock()):
#         result = await error_recovery_node(state)

#     assert result["task_queue"][0]["status"] == "pending"
#     assert result["task_queue"][0]["retries"] == 1
#     assert len(result["error_log"]) == 1


# @pytest.mark.asyncio
# async def test_error_recovery_skips_exhausted():
#     """Error recovery skips tasks that have hit max_retries."""
#     from workflows.state import initial_state
#     from workflows.graph import error_recovery_node
#     state = initial_state("Test")
#     task = _make_task("chat_buddy", status="failed", retries=3, max_retries=3)
#     state["task_queue"] = [task]

#     with patch("asyncio.sleep", new=AsyncMock()):
#         result = await error_recovery_node(state)

#     assert result["task_queue"][0]["status"] == "skipped"
#     assert len(result["failed_tasks"]) == 1


# # ── State schema test ─────────────────────────────────────────────────────────

# def test_initial_state_has_required_keys():
#     """initial_state() produces a dict with all required PaddleAurumState keys."""
#     from workflows.state import initial_state
#     state = initial_state("Test goal")
#     required_keys = [
#         "session_id", "goal", "task_queue", "completed_tasks", "failed_tasks",
#         "iteration_count", "shared_context", "short_term_memory", "error_log",
#     ]
#     for key in required_keys:
#         assert key in state, f"Missing key: {key}"
#     assert state["goal"] == "Test goal"
#     assert state["iteration_count"] == 0
#     assert isinstance(state["task_queue"], list)


# def test_make_task_defaults():
#     """make_task() creates a valid TaskItem with sensible defaults."""
#     from workflows.state import make_task
#     task = make_task("Do something", "chat_buddy", priority=2)
#     assert task["status"] == "pending"
#     assert task["retries"] == 0
#     assert task["max_retries"] == 3
#     assert task["priority"] == 2
#     assert task["assigned_to"] == "chat_buddy"


# # ── Memory manager test ───────────────────────────────────────────────────────

# @pytest.mark.asyncio
# async def test_memory_manager_write_recall(tmp_path):
#     """MemoryManager writes and recalls agent messages correctly."""
#     from config.settings import settings
#     settings.sqlite_db_path = str(tmp_path / "mem.db")

#     from memory.memory_manager import MemoryManager
#     mm = MemoryManager(str(tmp_path / "mem.db"))
#     await mm.init_db()

#     messages = [
#         {"agent_id": "chat_buddy", "role": "assistant", "content": "Test message", "timestamp": time.time()},
#     ]
#     ids = await mm.write_session("sess-001", messages)
#     assert len(ids) == 1

#     recalled = await mm.recall("chat_buddy", limit=5)
#     assert len(recalled) == 1
#     assert recalled[0] == "Test message"


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])





























# 2#@##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


























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


# ── APPENDED: Fixture-based workflow tests ────────────────────────────────────
# These tests leverage conftest.py fixtures for cleaner, more maintainable
# workflow and integration-level coverage.

class TestRouterWithFixtures:
    """Router tests using make_state and make_task_item fixtures."""

    def test_route_from_ceo_inventory_only(self, make_state, make_task_item):
        """Router returns only stock_sergeant for pure inventory tasks."""
        from workflows.router import route_from_ceo
        state = make_state("Inventory scan", task_queue=[
            make_task_item("stock_scout"),
            make_task_item("stock_sergeant"),
        ])
        routes = route_from_ceo(state)
        assert "stock_sergeant" in routes
        assert "customer_captain" not in routes
        assert "promo_general" not in routes

    def test_route_from_ceo_marketing_only(self, make_state, make_task_item):
        """Router returns only promo_general for pure marketing tasks."""
        from workflows.router import route_from_ceo
        state = make_state("Marketing blitz", task_queue=[
            make_task_item("promo_general"),
            make_task_item("recommender"),
        ])
        routes = route_from_ceo(state)
        assert "promo_general" in routes
        assert "customer_captain" not in routes
        assert "stock_sergeant" not in routes

    def test_route_skips_completed_tasks(self, make_state, make_task_item):
        """Router does not include branches for already-done tasks."""
        from workflows.router import route_from_ceo
        state = make_state("Partial run", task_queue=[
            make_task_item("chat_buddy", status="done"),
            make_task_item("stock_scout", status="pending"),
        ])
        routes = route_from_ceo(state)
        # Only stock_sergeant should be triggered — chat_buddy task is done
        assert "stock_sergeant" in routes
        assert "customer_captain" not in routes

    def test_route_post_aggregation_mixed_statuses(self, make_state, make_task_item):
        """Aggregation routing checks only failed+retryable, ignores done/skipped."""
        from workflows.router import route_post_aggregation
        state = make_state("Mixed run", task_queue=[
            make_task_item("chat_buddy", status="done"),
            make_task_item("stock_scout", status="skipped", retries=3, max_retries=3),
            make_task_item("recommender", status="failed", retries=1, max_retries=3),
        ])
        result = route_post_aggregation(state)
        # recommender failed with retries remaining → error_recovery
        assert result == "error_recovery"

    def test_is_done_all_terminal(self, make_state, make_task_item):
        """is_done() returns True when all tasks are in terminal states."""
        from workflows.router import is_done
        state = make_state("Done check", task_queue=[
            make_task_item("chat_buddy", status="done"),
            make_task_item("stock_scout", status="skipped"),
            make_task_item("recommender", status="failed"),
        ])
        assert is_done(state) is True

    def test_is_done_pending_task(self, make_state, make_task_item):
        """is_done() returns False when any task is still pending."""
        from workflows.router import is_done
        state = make_state("Not done", task_queue=[
            make_task_item("chat_buddy", status="done"),
            make_task_item("stock_scout", status="pending"),
        ])
        assert is_done(state) is False

    def test_is_done_in_progress_task(self, make_state, make_task_item):
        """is_done() returns False when any task is in_progress."""
        from workflows.router import is_done
        state = make_state("In flight", task_queue=[
            make_task_item("promo_general", status="in_progress"),
        ])
        assert is_done(state) is False


class TestErrorRecoveryWithFixtures:
    """Error recovery node tests using make_state and make_task_item fixtures."""

    @pytest.mark.asyncio
    async def test_error_recovery_exponential_backoff_cap(self, make_state, make_task_item):
        """Backoff is capped at 30 seconds regardless of retry count."""
        from workflows.graph import error_recovery_node
        task = make_task_item("chat_buddy", status="failed", retries=10, max_retries=15)
        state = make_state("Retry backoff test", task_queue=[task])

        waited = []
        async def mock_sleep(duration):
            waited.append(duration)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await error_recovery_node(state)

        assert all(w <= 30 for w in waited), f"Backoff exceeded 30s: {waited}"

    @pytest.mark.asyncio
    async def test_error_recovery_multiple_tasks_partial(self, make_state, make_task_item):
        """Error recovery resets retryable tasks and skips exhausted ones."""
        from workflows.graph import error_recovery_node
        state = make_state("Partial retry", task_queue=[
            make_task_item("chat_buddy",   status="failed", retries=0, max_retries=3),
            make_task_item("stock_scout",  status="failed", retries=3, max_retries=3),
            make_task_item("recommender",  status="done"),
        ])

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await error_recovery_node(state)

        statuses = {t["assigned_to"]: t["status"] for t in result["task_queue"]}
        assert statuses["chat_buddy"] == "pending"     # reset for retry
        assert statuses["stock_scout"] == "skipped"    # exhausted
        assert statuses["recommender"] == "done"       # untouched

    @pytest.mark.asyncio
    async def test_error_recovery_populates_error_log(self, make_state, make_task_item):
        """Each retried task produces an error_log entry."""
        from workflows.graph import error_recovery_node
        tasks = [
            make_task_item("chat_buddy",  status="failed", retries=0, max_retries=2),
            make_task_item("stock_scout", status="failed", retries=0, max_retries=2),
        ]
        state = make_state("Log test", task_queue=tasks, error_log=[])

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await error_recovery_node(state)

        assert len(result["error_log"]) == 2

    @pytest.mark.asyncio
    async def test_error_recovery_failed_tasks_list_updated(self, make_state, make_task_item):
        """Exhausted tasks are added to state['failed_tasks']."""
        from workflows.graph import error_recovery_node
        task = make_task_item("recommender", status="failed", retries=3, max_retries=3)
        task["error_message"] = "Vector store unavailable"
        state = make_state("Failed task", task_queue=[task], failed_tasks=[])

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await error_recovery_node(state)

        assert len(result["failed_tasks"]) == 1
        assert result["failed_tasks"][0]["assigned_to"] == "recommender"


class TestStateSchemaWithFixtures:
    """State schema validation tests using make_state and make_task_item fixtures."""

    def test_make_state_fixture_goal(self, make_state):
        """make_state fixture correctly sets goal field."""
        state = make_state("Fixture goal test")
        assert state["goal"] == "Fixture goal test"
        assert state["iteration_count"] == 0

    def test_make_state_override_fields(self, make_state):
        """make_state fixture accepts field overrides."""
        state = make_state("Override test", iteration_count=2, current_step="aggregator")
        assert state["iteration_count"] == 2
        assert state["current_step"] == "aggregator"

    def test_make_task_item_fixture_defaults(self, make_task_item):
        """make_task_item fixture produces a valid TaskItem with correct defaults."""
        task = make_task_item("stock_scout")
        assert task["status"] == "pending"
        assert task["assigned_to"] == "stock_scout"
        assert task["retries"] == 0
        assert task["max_retries"] == 3
        assert isinstance(task["task_id"], str)
        assert len(task["task_id"]) == 8

    def test_make_task_item_status_override(self, make_task_item):
        """make_task_item fixture correctly applies status override."""
        task = make_task_item("chat_buddy", status="in_progress", retries=1)
        assert task["status"] == "in_progress"
        assert task["retries"] == 1

    def test_state_session_id_unique_per_call(self, make_state):
        """Each make_state call produces a unique session_id."""
        s1 = make_state("State A")
        s2 = make_state("State B")
        assert s1["session_id"] != s2["session_id"]

    def test_task_id_unique_per_task(self, make_task_item):
        """Each make_task_item call produces a unique task_id."""
        t1 = make_task_item("chat_buddy")
        t2 = make_task_item("chat_buddy")
        assert t1["task_id"] != t2["task_id"]


class TestMemoryManagerWithFixtures:
    """MemoryManager tests using the memory_manager fixture from conftest."""

    @pytest.mark.asyncio
    async def test_write_and_recall_multiple_agents(self, memory_manager):
        """Recall correctly filters by agent_id across multiple agents."""
        messages = [
            {"agent_id": "chat_buddy",  "role": "assistant", "content": "Handled ticket #1", "timestamp": time.time()},
            {"agent_id": "stock_scout", "role": "assistant", "content": "Scan complete",      "timestamp": time.time()},
            {"agent_id": "chat_buddy",  "role": "assistant", "content": "Handled ticket #2", "timestamp": time.time()},
        ]
        await memory_manager.write_session("sess-multi", messages)

        chat_recalls = await memory_manager.recall("chat_buddy", limit=10)
        scout_recalls = await memory_manager.recall("stock_scout", limit=10)

        assert len(chat_recalls) == 2
        assert len(scout_recalls) == 1
        assert "ticket" in chat_recalls[0]

    @pytest.mark.asyncio
    async def test_trim_short_term_preserves_window(self, memory_manager):
        """trim_short_term keeps the most recent N messages within the window."""
        messages = [
            {"agent_id": f"agent_{i}", "role": "assistant", "content": f"msg {i}", "timestamp": float(i)}
            for i in range(30)
        ]
        trimmed = memory_manager.trim_short_term(messages, window=20)
        assert len(trimmed) == 20
        # Most recent messages should be retained
        assert trimmed[-1]["content"] == "msg 29"

    @pytest.mark.asyncio
    async def test_trim_short_term_preserves_system_message(self, memory_manager):
        """trim_short_term always retains the first system message."""
        messages = [
            {"agent_id": "system", "role": "system", "content": "System prompt", "timestamp": 0.0},
        ] + [
            {"agent_id": f"a{i}", "role": "assistant", "content": f"msg {i}", "timestamp": float(i + 1)}
            for i in range(25)
        ]
        trimmed = memory_manager.trim_short_term(messages, window=10)
        assert trimmed[0]["role"] == "system"
        assert trimmed[0]["content"] == "System prompt"
        assert len(trimmed) == 10

    @pytest.mark.asyncio
    async def test_set_and_get_shared_context(self, memory_manager):
        """set_shared and get_shared correctly store and retrieve cross-agent KV data."""
        await memory_manager.set_shared("low_stock_count", 3)
        value = await memory_manager.get_shared("low_stock_count")
        assert value == 3

    @pytest.mark.asyncio
    async def test_write_session_summary_and_recall(self, memory_manager):
        """write_session_summary persists and get_recent_summaries retrieves it."""
        summary = {
            "summary": "Daily ops completed.",
            "wins": ["3 tickets resolved"],
            "blockers": [],
            "next_actions": ["Run weekend promo"],
            "timestamp": "2026-02-23T00:00:00Z",
        }
        await memory_manager.write_session_summary("sess-sum-001", summary)

        recent = await memory_manager.get_recent_summaries(limit=5)
        assert len(recent) >= 1
        assert recent[0]["summary"]["summary"] == "Daily ops completed."

    @pytest.mark.asyncio
    async def test_recall_by_session_id(self, memory_manager):
        """recall_session returns only messages from the specified session."""
        msgs_a = [{"agent_id": "chat_buddy", "role": "assistant", "content": "Session A msg", "timestamp": time.time()}]
        msgs_b = [{"agent_id": "chat_buddy", "role": "assistant", "content": "Session B msg", "timestamp": time.time()}]

        await memory_manager.write_session("sess-A", msgs_a)
        await memory_manager.write_session("sess-B", msgs_b)

        session_a_msgs = await memory_manager.recall_session("sess-A")
        assert len(session_a_msgs) == 1
        assert session_a_msgs[0]["content"] == "Session A msg"

    @pytest.mark.asyncio
    async def test_purge_old_memory(self, memory_manager):
        """purge_old_memory removes entries older than the cutoff."""
        import aiosqlite
        # Insert an artificially old record
        old_ts = time.time() - (35 * 86400)  # 35 days ago
        async with aiosqlite.connect(memory_manager.db_path) as db:
            await db.execute(
                "INSERT INTO agent_memory (session_id, agent_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                ("old-sess", "chat_buddy", "assistant", "Ancient message", old_ts)
            )
            await db.commit()

        deleted = await memory_manager.purge_old_memory(older_than_days=30)
        assert deleted >= 1

        recalled = await memory_manager.recall("chat_buddy", limit=10, session_id="old-sess")
        assert len(recalled) == 0

    @pytest.mark.asyncio
    async def test_search_memory_keyword(self, memory_manager):
        """search_memory returns records containing the keyword."""
        msgs = [
            {"agent_id": "stock_scout", "role": "assistant", "content": "Carbon Elite is CRITICAL low stock", "timestamp": time.time()},
            {"agent_id": "stock_scout", "role": "assistant", "content": "Ball packs are fine",               "timestamp": time.time()},
        ]
        await memory_manager.write_session("sess-search", msgs)

        results = await memory_manager.search_memory("CRITICAL", limit=10)
        assert len(results) >= 1
        assert all("CRITICAL" in r["content"] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])