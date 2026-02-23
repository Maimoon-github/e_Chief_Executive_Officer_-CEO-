# tasks/support_tasks.py
"""
tasks/support_tasks.py
──────────────────────
Task builders for customer support.
Loads templates from tasks.yaml and returns TaskItem objects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.prompt_registry import prompt_registry
from workflows.state import TaskItem, make_task

logger = logging.getLogger(__name__)

TASK_KEYS = {
    "handle_customer_inquiry",
    "review_support_queue",
}


def build_handle_inquiry_task(
    message: str,
    customer_id: str = "",
    order_id: str = "",
    priority: int = 1,
    max_retries: int = 3,
) -> TaskItem:
    """
    Build a task to handle a single customer inquiry.
    """
    input_data = {
        "message": message,
        "customer_id": customer_id,
        "order_id": order_id,
    }
    description = prompt_registry.render_task("handle_customer_inquiry", **input_data)
    return make_task(
        description=description,
        assigned_to="chat_buddy",
        priority=priority,
        input_data=input_data,
        required_tools=["shopify_api", "database", "web_search"],
        max_retries=max_retries,
    )


def build_review_queue_task(
    tickets: List[Dict[str, Any]],
    priority: int = 2,
    max_retries: int = 2,
) -> TaskItem:
    """
    Build a task to review the current support ticket queue.
    """
    input_data = {"tickets": tickets}
    description = prompt_registry.render_task("review_support_queue", **input_data)
    return make_task(
        description=description,
        assigned_to="customer_captain",
        priority=priority,
        input_data=input_data,
        required_tools=["database"],
        max_retries=max_retries,
    )


def build_support_tasks_from_goal(goal: str) -> List[TaskItem]:
    """
    Placeholder for goal‑based task generation.
    """
    tasks = []
    if any(kw in goal.lower() for kw in ["customer", "support", "ticket", "inquiry"]):
        # In a real system, the CEO would parse the goal and create specific tasks.
        # For now, we return an empty list and let the CEO handle it.
        pass
    return tasks