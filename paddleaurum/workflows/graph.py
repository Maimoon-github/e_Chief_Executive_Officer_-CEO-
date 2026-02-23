"""
workflows/graph.py
──────────────────
Builds and compiles the PaddleAurum LangGraph StateGraph.
This is the central wiring module — all nodes and edges are registered here.

Execution topology:
  ceo_orchestrator
       ↓ (fan-out)
  customer_captain  stock_sergeant  promo_general
       ↓                  ↓               ↓
  chat_buddy          stock_scout     recommender
       ↓                  ↓               ↓
              aggregator (barrier)
                    ↓
        error_recovery  OR  memory_persist
                    ↓               ↓
          ceo_orchestrator     done_notify
                                    ↓
                                   END
"""
from __future__ import annotations

import logging

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agents.ceo import ceo_orchestrator_node
from agents.chat_buddy import chat_buddy_node
from agents.customer_captain import customer_captain_node
from agents.promo_general import promo_general_node
from agents.recommender import recommender_node
from agents.stock_scout import stock_scout_node
from agents.stock_sergeant import stock_sergeant_node
from config.settings import settings
from workflows.router import (
    route_from_ceo,
    route_post_aggregation,
    route_post_error_recovery,
)
from workflows.state import PaddleAurumState

logger = logging.getLogger(__name__)


# ── Utility nodes (pure Python, no LLM) ──────────────────────────────────────

async def aggregator_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    Barrier node: waits until all branch outputs are populated.
    Merges parallel results and triggers CEO synthesis.
    """
    import time
    from agents.customer_captain import customer_captain_review_node
    from agents.stock_sergeant import stock_sergeant_review_node

    state["current_step"] = "aggregator"
    logger.info("[Aggregator] Collecting branch results")

    # Post-processing hooks
    state = await customer_captain_review_node(state)
    state = await stock_sergeant_review_node(state)

    # Mark completed tasks
    done_count = sum(1 for t in state["task_queue"] if t["status"] == "done")
    failed_count = sum(1 for t in state["task_queue"] if t["status"] == "failed")
    logger.info("[Aggregator] Tasks: %d done, %d failed", done_count, failed_count)

    state["short_term_memory"].append({
        "agent_id": "aggregator",
        "role": "system",
        "content": f"Aggregation complete. Done: {done_count}, Failed: {failed_count}",
        "timestamp": time.time(),
        "tool_calls": None,
        "tool_results": None,
    })

    # Trigger CEO synthesis by setting iteration_count to signal synthesis phase
    # CEO will synthesise report on next call (iteration_count >= 1)
    return state


async def error_recovery_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    Retry failed tasks with exponential backoff.
    Tasks exceeding max_retries are marked 'skipped'.
    """
    import asyncio
    import time

    state["current_step"] = "error_recovery"
    failed_tasks = [t for t in state["task_queue"] if t["status"] == "failed"]

    for task in failed_tasks:
        if task["retries"] < task["max_retries"]:
            wait = min(2 ** task["retries"], 30)  # cap at 30s
            logger.info(
                "[Error Recovery] Retrying task %s (attempt %d, wait %ds)",
                task["task_id"], task["retries"] + 1, wait,
            )
            await asyncio.sleep(wait)
            task["status"] = "pending"
            task["retries"] += 1
            state["error_log"].append({
                "task_id": task["task_id"],
                "retry_attempt": task["retries"],
                "wait_seconds": wait,
                "timestamp": time.time(),
            })
        else:
            logger.warning("[Error Recovery] Task %s exceeded max retries — skipping", task["task_id"])
            task["status"] = "skipped"
            state["failed_tasks"].append(task)
            state["error_log"].append({
                "task_id": task["task_id"],
                "final_status": "skipped",
                "reason": task.get("error_message", "unknown"),
                "timestamp": time.time(),
            })

    return state


async def memory_persist_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    Persist session memory to SQLite and trim short-term window.
    Also triggers CEO final report synthesis.
    """
    import time
    from memory.memory_manager import memory_manager

    state["current_step"] = "memory_persist"
    logger.info("[Memory] Persisting session %s", state["session_id"])

    # Write long-term memory
    if state["short_term_memory"]:
        row_ids = await memory_manager.write_session(
            state["session_id"], state["short_term_memory"]
        )
        state["long_term_memory_keys"].extend(row_ids)

    # Trim window
    state["short_term_memory"] = memory_manager.trim_short_term(
        state["short_term_memory"]
    )

    # CEO synthesis (sets final_report)
    state = await ceo_orchestrator_node(state)

    return state


async def done_notify_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    Terminal node: log summary, optionally send Slack/email notification.
    """
    import time
    from tools.email_tool import send_alert

    state["current_step"] = "done"
    report = state.get("final_report") or {}

    summary = (
        f"Session: {state['session_id']}\n"
        f"Goal: {state['goal']}\n"
        f"Summary: {report.get('summary', 'No summary.')}\n"
        f"Wins: {report.get('wins', [])}\n"
        f"Blockers: {report.get('blockers', [])}\n"
        f"Next Actions: {report.get('next_actions', [])}\n"
        f"Failed tasks: {len(state.get('failed_tasks', []))}\n"
        f"Error count: {len(state.get('error_log', []))}"
    )

    logger.info("\n%s\n%s\n%s", "=" * 60, summary, "=" * 60)

    # Send admin email (non-blocking, best effort)
    try:
        await send_alert("PaddleAurum Session Complete", summary)
    except Exception as exc:
        logger.warning("Could not send completion email: %s", exc)

    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(use_checkpointer: bool = True) -> StateGraph:
    """
    Build and compile the PaddleAurum LangGraph.

    Args:
        use_checkpointer: If True, attach SQLite checkpointer for persistence.

    Returns:
        Compiled StateGraph ready for async streaming execution.
    """
    workflow = StateGraph(PaddleAurumState)

    # ── Register all nodes ────────────────────────────────────────────────────
    workflow.add_node("ceo_orchestrator",  ceo_orchestrator_node)
    workflow.add_node("customer_captain",  customer_captain_node)
    workflow.add_node("chat_buddy",        chat_buddy_node)
    workflow.add_node("stock_sergeant",    stock_sergeant_node)
    workflow.add_node("stock_scout",       stock_scout_node)
    workflow.add_node("promo_general",     promo_general_node)
    workflow.add_node("recommender",       recommender_node)
    workflow.add_node("aggregator",        aggregator_node)
    workflow.add_node("error_recovery",    error_recovery_node)
    workflow.add_node("memory_persist",    memory_persist_node)
    workflow.add_node("done_notify",       done_notify_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("ceo_orchestrator")

    # ── CEO → parallel team leads (conditional fan-out) ───────────────────────
    workflow.add_conditional_edges(
        "ceo_orchestrator",
        route_from_ceo,
        {
            "customer_captain": "customer_captain",
            "stock_sergeant":   "stock_sergeant",
            "promo_general":    "promo_general",
            "aggregator":       "aggregator",   # fallback if no tasks
        },
    )

    # ── Team Lead → Worker edges ──────────────────────────────────────────────
    workflow.add_edge("customer_captain", "chat_buddy")
    workflow.add_edge("stock_sergeant",   "stock_scout")
    workflow.add_edge("promo_general",    "recommender")

    # ── Workers → Aggregator (all branches converge here) ────────────────────
    workflow.add_edge("chat_buddy",   "aggregator")
    workflow.add_edge("stock_scout",  "aggregator")
    workflow.add_edge("recommender",  "aggregator")

    # ── Aggregator → conditional ──────────────────────────────────────────────
    workflow.add_conditional_edges(
        "aggregator",
        route_post_aggregation,
        {
            "error_recovery": "error_recovery",
            "memory_persist": "memory_persist",
        },
    )

    # ── Error recovery → conditional re-plan or persist ──────────────────────
    workflow.add_conditional_edges(
        "error_recovery",
        route_post_error_recovery,
        {
            "ceo_orchestrator": "ceo_orchestrator",
            "memory_persist":   "memory_persist",
        },
    )

    # ── Linear tail ───────────────────────────────────────────────────────────
    workflow.add_edge("memory_persist", "done_notify")
    workflow.add_edge("done_notify",    END)

    # ── Compile ───────────────────────────────────────────────────────────────
    if use_checkpointer:
        checkpointer = SqliteSaver.from_conn_string(settings.sqlite_db_path)
        compiled = workflow.compile(checkpointer=checkpointer)
    else:
        compiled = workflow.compile()

    logger.info("LangGraph compiled successfully")
    return compiled
