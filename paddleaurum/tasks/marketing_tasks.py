# tasks/marketing_tasks.py
"""
tasks/marketing_tasks.py
────────────────────────
Task builders for marketing and promotions.
Loads templates from tasks.yaml and returns TaskItem objects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.prompt_registry import prompt_registry
from workflows.state import TaskItem, make_task

logger = logging.getLogger(__name__)

TASK_KEYS = {
    "draft_promo_campaign",
    "research_competitor_pricing",
}


def build_draft_promo_campaign_task(
    objective: str,
    segment: str = "general",
    segment_size: int = 0,
    recommended_products: Optional[List[Dict[str, Any]]] = None,
    priority: int = 2,
    max_retries: int = 3,
) -> TaskItem:
    """
    Build a task to draft an email campaign.
    """
    input_data = {
        "objective": objective,
        "segment": segment,
        "segment_size": segment_size,
        "recommended_products": recommended_products or [],
        # competitor_context and low_stock will be filled at runtime by Promo General
    }
    description = prompt_registry.render_task("draft_promo_campaign", **input_data)
    return make_task(
        description=description,
        assigned_to="promo_general",
        priority=priority,
        input_data=input_data,
        required_tools=["web_search", "email_sender", "database"],
        max_retries=max_retries,
    )


def build_competitor_pricing_task(
    categories: List[str],
    priority: int = 3,
    max_retries: int = 2,
) -> TaskItem:
    """
    Build a task to research competitor pricing for given categories.
    """
    input_data = {"categories": categories}
    description = prompt_registry.render_task("research_competitor_pricing", **input_data)
    return make_task(
        description=description,
        assigned_to="promo_general",
        priority=priority,
        input_data=input_data,
        required_tools=["web_search"],
        max_retries=max_retries,
    )


def build_marketing_tasks_from_goal(goal: str) -> List[TaskItem]:
    """
    Placeholder for goal‑based task generation.
    """
    # Example: if goal mentions "email campaign", create a draft task.
    tasks = []
    if "campaign" in goal.lower():
        tasks.append(build_draft_promo_campaign_task(objective=goal))
    if "competitor" in goal.lower() or "pricing" in goal.lower():
        tasks.append(build_competitor_pricing_task(categories=["pickleball paddles"]))
    return tasks