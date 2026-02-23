"""
agents/ceo.py
─────────────
NODE 0 — PaddleAurum CEO Orchestrator

Responsibilities:
  - Receive raw business goal from the trigger layer
  - Use LLM reasoning to decompose the goal into a task queue
  - Write task assignments into PaddleAurumState
  - After execution: synthesise a final strategic report
  - Re-plan if critical tasks fail (up to iteration_count < 3)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import yaml
from crewai import Agent, Crew, Process, Task

from config.settings import settings
from models.ollama_loader import get_llm_async
from workflows.state import PaddleAurumState, TaskItem, make_task

logger = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────

def _load_config() -> tuple[Dict, Dict]:
    with open("config/agents.yaml") as f:
        agents_cfg = yaml.safe_load(f)
    with open("config/tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)
    return agents_cfg, tasks_cfg


# ── Task decomposition ────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """You are the PaddleAurum CEO AI. Your job is to decompose
business goals into discrete actionable tasks for your agent team.
Agent names: chat_buddy, stock_scout, recommender, customer_captain,
stock_sergeant, promo_general.
Always output valid JSON only — no markdown, no explanation."""

_DECOMPOSE_TEMPLATE = """
Goal: {goal}
Store context: {context}
Previous session summaries: {summaries}

Decompose this goal into 3-8 tasks. Each task object must have:
  task_id (string, 8 chars), description (string), assigned_to (string),
  priority (int 1-5), required_tools (list), max_retries (int, default 3).

Assigned_to must be one of: chat_buddy, stock_scout, recommender,
customer_captain, stock_sergeant, promo_general.

JSON array only:
"""


async def decompose_goal(
    goal: str,
    shared_context: Dict,
    recent_summaries: List[Dict],
) -> List[TaskItem]:
    """
    Call the CEO LLM to break a goal into task items.
    Returns a list of TaskItem dicts ready for the state.
    """
    llm = await get_llm_async("ceo")

    prompt = _DECOMPOSE_TEMPLATE.format(
        goal=goal,
        context=json.dumps(shared_context, indent=2)[:800],
        summaries=json.dumps(recent_summaries, indent=2)[:400],
    )

    full_prompt = f"{_DECOMPOSE_SYSTEM}\n\n{prompt}"

    try:
        raw_response = await llm.ainvoke(full_prompt)
        # Strip any accidental markdown fences
        clean = raw_response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        raw_tasks = json.loads(clean)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("CEO decompose_goal failed to parse LLM output: %s", exc)
        # Graceful fallback: create one generic task per branch
        raw_tasks = _fallback_tasks(goal)

    tasks = []
    for t in raw_tasks:
        tasks.append(
            make_task(
                description=t.get("description", goal),
                assigned_to=t.get("assigned_to", "chat_buddy"),
                priority=int(t.get("priority", 3)),
                input_data={"goal": goal, **t},
                required_tools=t.get("required_tools", []),
                max_retries=int(t.get("max_retries", 3)),
            )
        )

    logger.info("CEO decomposed goal into %d tasks", len(tasks))
    return tasks


def _fallback_tasks(goal: str) -> List[Dict]:
    """Minimal fallback if LLM decomposition fails entirely."""
    return [
        {"description": f"Support: {goal}", "assigned_to": "chat_buddy",    "priority": 2},
        {"description": f"Inventory: {goal}", "assigned_to": "stock_scout", "priority": 2},
        {"description": f"Promo: {goal}",     "assigned_to": "recommender", "priority": 3},
    ]


# ── Report synthesis ──────────────────────────────────────────────────────────

_SYNTHESIS_TEMPLATE = """
You are the PaddleAurum CEO. Synthesise an executive report from these team outputs.

Customer Support: {customer_output}
Inventory:        {inventory_output}
Marketing:        {marketing_output}
Recommendations:  {recommendation_output}
Failed tasks:     {failed_tasks}

Return JSON only with keys: summary, wins (list), blockers (list),
next_actions (list), timestamp (ISO string).
"""


async def synthesise_report(state: PaddleAurumState) -> Dict[str, Any]:
    llm = await get_llm_async("ceo")
    prompt = _SYNTHESIS_TEMPLATE.format(
        customer_output=json.dumps(state.get("customer_support_output") or {})[:600],
        inventory_output=json.dumps(state.get("inventory_output") or {})[:600],
        marketing_output=json.dumps(state.get("marketing_output") or {})[:600],
        recommendation_output=json.dumps(state.get("recommendation_output") or {})[:400],
        failed_tasks=json.dumps(state.get("failed_tasks") or [])[:400],
    )

    try:
        raw = await llm.ainvoke(f"{_DECOMPOSE_SYSTEM}\n\n{prompt}")
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        report = json.loads(clean)
    except Exception as exc:
        logger.error("CEO synthesis failed: %s", exc)
        report = {
            "summary": "Execution completed with partial results.",
            "wins": [],
            "blockers": [str(exc)],
            "next_actions": ["Review logs and re-run."],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    return report


# ── LangGraph node function ───────────────────────────────────────────────────

async def ceo_orchestrator_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    LangGraph node for CEO orchestration.
    On first call (iteration_count == 0): decompose goal → populate task_queue.
    On subsequent calls (re-plan): synthesise final report.
    """
    from memory.memory_manager import memory_manager

    state["current_step"] = "ceo_orchestrator"

    if state["iteration_count"] == 0:
        # ── Initial decomposition ──────────────────────────────────────────
        logger.info("[CEO] Decomposing goal: %s", state["goal"])
        recent_summaries = await memory_manager.get_recent_summaries(limit=3)
        tasks = await decompose_goal(
            state["goal"], state["shared_context"], recent_summaries
        )
        state["task_queue"] = tasks

        # Record in short-term memory
        state["short_term_memory"].append({
            "agent_id": "ceo",
            "role": "assistant",
            "content": f"Decomposed goal into {len(tasks)} tasks.",
            "timestamp": time.time(),
            "tool_calls": None,
            "tool_results": None,
        })

    elif state["iteration_count"] >= 1:
        # ── Post-execution synthesis ───────────────────────────────────────
        logger.info("[CEO] Synthesising final report (iteration %d)", state["iteration_count"])
        report = await synthesise_report(state)
        state["final_report"] = report

        await memory_manager.write_session_summary(state["session_id"], report)

    state["iteration_count"] += 1
    return state
