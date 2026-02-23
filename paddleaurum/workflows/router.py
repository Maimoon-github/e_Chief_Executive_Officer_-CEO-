# """
# workflows/router.py
# ───────────────────
# Conditional edge functions for the LangGraph workflow.
# Each function takes the current state and returns the next node name(s).
# These are the "brain" of the graph's branching logic.
# """
# from __future__ import annotations

# import logging
# from typing import List, Literal

# from workflows.state import PaddleAurumState

# logger = logging.getLogger(__name__)

# # ── Agent groupings ───────────────────────────────────────────────────────────
# SUPPORT_AGENTS   = {"chat_buddy", "customer_captain"}
# INVENTORY_AGENTS = {"stock_scout", "stock_sergeant"}
# MARKETING_AGENTS = {"promo_general", "recommender"}

# # ── CEO fan-out ───────────────────────────────────────────────────────────────

# def route_from_ceo(state: PaddleAurumState) -> List[str]:
#     """
#     Fan-out after CEO orchestration.
#     Returns a list of team lead node names that have relevant tasks.
#     LangGraph will execute all returned nodes in parallel.
#     """
#     routes = set()

#     for task in state.get("task_queue", []):
#         if task["status"] != "pending":
#             continue
#         agent = task["assigned_to"]

#         if agent in SUPPORT_AGENTS:
#             routes.add("customer_captain")
#         elif agent in INVENTORY_AGENTS:
#             routes.add("stock_sergeant")
#         elif agent in MARKETING_AGENTS:
#             routes.add("promo_general")

#     if not routes:
#         logger.warning("[Router] CEO produced no routable tasks — going to aggregator")
#         routes.add("aggregator")

#     logger.info("[Router] CEO → branches: %s", routes)
#     return list(routes)


# # ── Post-aggregation routing ──────────────────────────────────────────────────

# def route_post_aggregation(
#     state: PaddleAurumState,
# ) -> Literal["error_recovery", "memory_persist"]:
#     """
#     After aggregator collects all branch results:
#     - If any task failed and has retries left → error_recovery
#     - Otherwise → memory_persist
#     """
#     failed_with_retries = [
#         t for t in state.get("task_queue", [])
#         if t["status"] == "failed" and t["retries"] < t["max_retries"]
#     ]

#     if failed_with_retries:
#         logger.info(
#             "[Router] %d failed tasks with retries remaining → error_recovery",
#             len(failed_with_retries),
#         )
#         return "error_recovery"

#     return "memory_persist"


# # ── Post error-recovery routing ───────────────────────────────────────────────

# def route_post_error_recovery(
#     state: PaddleAurumState,
# ) -> Literal["ceo_orchestrator", "memory_persist"]:
#     """
#     After error recovery:
#     - If iteration_count < 3 AND some tasks were reset to pending → re-plan
#     - Otherwise → give up and persist what we have
#     """
#     still_pending = [
#         t for t in state.get("task_queue", [])
#         if t["status"] == "pending"
#     ]

#     if still_pending and state.get("iteration_count", 0) < 3:
#         logger.info(
#             "[Router] %d tasks still pending after error recovery (iter %d) → re-plan",
#             len(still_pending),
#             state["iteration_count"],
#         )
#         return "ceo_orchestrator"

#     logger.info("[Router] Max iterations or no pending tasks → memory_persist")
#     return "memory_persist"


# # ── Terminal check ────────────────────────────────────────────────────────────

# def is_done(state: PaddleAurumState) -> bool:
#     """
#     Helper to check if session should terminate.
#     All tasks must be done, failed, or skipped.
#     """
#     active = [
#         t for t in state.get("task_queue", [])
#         if t["status"] in ("pending", "in_progress")
#     ]
#     return len(active) == 0

























# 2###############################################################################################





























"""
workflows/router.py
───────────────────
Conditional edge functions for the LangGraph workflow.
Each function takes the current state and returns the next node name(s).
These are the "brain" of the graph's branching logic.
"""
from __future__ import annotations

import logging
from typing import List, Literal

from langgraph.types import Send
from workflows.state import PaddleAurumState

logger = logging.getLogger(__name__)

# ── Agent groupings ───────────────────────────────────────────────────────────
SUPPORT_AGENTS   = {"chat_buddy", "customer_captain"}
INVENTORY_AGENTS = {"stock_scout", "stock_sergeant"}
MARKETING_AGENTS = {"promo_general", "recommender"}

# ── CEO fan-out ───────────────────────────────────────────────────────────────

def route_from_ceo(state: PaddleAurumState) -> List[Send]:
    """
    Fan-out after CEO orchestration.
    Returns a list of Send objects targeting team lead nodes that have relevant tasks.
    LangGraph executes all returned Send objects in parallel.
    """
    routes = set()

    for task in state.get("task_queue", []):
        if task["status"] != "pending":
            continue
        agent = task["assigned_to"]

        if agent in SUPPORT_AGENTS:
            routes.add("customer_captain")
        elif agent in INVENTORY_AGENTS:
            routes.add("stock_sergeant")
        elif agent in MARKETING_AGENTS:
            routes.add("promo_general")

    if not routes:
        logger.warning("[Router] CEO produced no routable tasks — going to aggregator")
        routes.add("aggregator")

    logger.info("[Router] CEO → branches: %s", routes)
    # Each Send receives the full current state; the graph will pass it to the target node.
    return [Send(node, state) for node in routes]


# ── Post-aggregation routing ──────────────────────────────────────────────────

def route_post_aggregation(
    state: PaddleAurumState,
) -> Literal["error_recovery", "memory_persist"]:
    """
    After aggregator collects all branch results:
    - If any task failed and has retries left → error_recovery
    - Otherwise → memory_persist
    """
    failed_with_retries = [
        t for t in state.get("task_queue", [])
        if t["status"] == "failed" and t["retries"] < t["max_retries"]
    ]

    if failed_with_retries:
        logger.info(
            "[Router] %d failed tasks with retries remaining → error_recovery",
            len(failed_with_retries),
        )
        return "error_recovery"

    return "memory_persist"


# ── Post error-recovery routing ───────────────────────────────────────────────

def route_post_error_recovery(
    state: PaddleAurumState,
) -> Literal["ceo_orchestrator", "memory_persist"]:
    """
    After error recovery:
    - If iteration_count < 3 AND some tasks were reset to pending → re-plan
    - Otherwise → give up and persist what we have
    """
    still_pending = [
        t for t in state.get("task_queue", [])
        if t["status"] == "pending"
    ]

    if still_pending and state.get("iteration_count", 0) < 3:
        logger.info(
            "[Router] %d tasks still pending after error recovery (iter %d) → re-plan",
            len(still_pending),
            state["iteration_count"],
        )
        return "ceo_orchestrator"

    logger.info("[Router] Max iterations or no pending tasks → memory_persist")
    return "memory_persist"


# ── Terminal check ────────────────────────────────────────────────────────────

def is_done(state: PaddleAurumState) -> bool:
    """
    Helper to check if session should terminate.
    All tasks must be done, failed, or skipped.
    """
    active = [
        t for t in state.get("task_queue", [])
        if t["status"] in ("pending", "in_progress")
    ]
    return len(active) == 0

    