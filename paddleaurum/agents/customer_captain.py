"""
agents/customer_captain.py
──────────────────────────
NODE 1 — Customer Captain (Team Lead)

Responsibilities:
  - Filter and pre-process customer-related tasks from the task queue
  - Delegate to Chat Buddy
  - Review outputs and flag tasks needing human escalation
  - Send escalation emails for unresolved critical issues
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from config.settings import settings
from tools.email_tool import send_alert
from workflows.state import PaddleAurumState, TaskItem

logger = logging.getLogger(__name__)

# Keywords that make a task "support" relevant
SUPPORT_KEYWORDS = {
    "customer", "support", "chat", "email", "inquiry", "complaint",
    "refund", "return", "order", "shipping", "response", "reply"
}


def _is_support_task(task: TaskItem) -> bool:
    """Classify a task as a support task based on assigned_to or keywords."""
    if task["assigned_to"] in ("chat_buddy", "customer_captain"):
        return True
    desc_lower = task["description"].lower()
    return any(kw in desc_lower for kw in SUPPORT_KEYWORDS)


async def review_and_escalate(output: Dict[str, Any]) -> List[Dict]:
    """
    Review Chat Buddy's output for unresolved escalations.
    Send email alerts for any task that requires human intervention.
    Returns list of escalation notices.
    """
    escalations = []
    for result in output.get("results", []):
        task_output = result.get("output_data", {})
        if task_output.get("escalate") and result["status"] == "done":
            inp = result.get("input_data", {})
            customer_id = inp.get("customer_id", "unknown")
            message = inp.get("message", "")
            reply_text = task_output.get("reply_text", "")

            # Fire-and-forget email alert
            try:
                await send_alert(
                    subject=f"Escalation Required — Customer {customer_id}",
                    message=(
                        f"Customer Message: {message[:300]}\n\n"
                        f"Chat Buddy Response: {reply_text[:300]}\n\n"
                        f"Resolution Type: {task_output.get('resolution_type', 'unknown')}\n"
                        f"Sentiment Score: {task_output.get('sentiment_score', 0):.2f}"
                    ),
                )
            except Exception as exc:
                logger.error("Could not send escalation email: %s", exc)

            escalations.append({
                "task_id": result["task_id"],
                "customer_id": customer_id,
                "resolution_type": task_output.get("resolution_type"),
            })

    return escalations


async def customer_captain_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    LangGraph node for Customer Captain.
    Filters support tasks, invokes Chat Buddy, then reviews outputs.
    """
    state["current_step"] = "customer_captain"
    logger.info("[Customer Captain] Reviewing task queue")

    # Count support tasks in queue (Chat Buddy handles them directly in its node)
    support_tasks = [t for t in state["task_queue"] if _is_support_task(t)]

    if not support_tasks:
        logger.info("[Customer Captain] No support tasks found.")
        return state

    logger.info("[Customer Captain] Found %d support tasks — delegating to Chat Buddy", len(support_tasks))

    # Mark tasks for Chat Buddy (if not already assigned)
    for task in state["task_queue"]:
        if task in support_tasks and task["assigned_to"] == "customer_captain":
            task["assigned_to"] = "chat_buddy"

    # After Chat Buddy runs, we'll review in a follow-up call if needed.
    # For now, log the delegation.
    state["short_term_memory"].append({
        "agent_id": "customer_captain",
        "role": "assistant",
        "content": f"Delegated {len(support_tasks)} tasks to Chat Buddy.",
        "timestamp": time.time(),
        "tool_calls": None,
        "tool_results": None,
    })

    return state


async def customer_captain_review_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    Optional post-Chat-Buddy review node.
    Called by aggregator after Chat Buddy completes to handle escalations.
    """
    support_output = state.get("customer_support_output")
    if not support_output:
        return state

    escalations = await review_and_escalate(support_output)

    if escalations:
        state["customer_support_output"]["escalations_sent"] = escalations
        logger.info("[Customer Captain] Sent %d escalation emails", len(escalations))

    return state
