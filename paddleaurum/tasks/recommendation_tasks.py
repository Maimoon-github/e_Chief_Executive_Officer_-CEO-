# tasks/recommendation_tasks.py
"""
tasks/recommendation_tasks.py
──────────────────────────────
Task builders for product recommendations.
Loads templates from tasks.yaml and returns TaskItem objects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.prompt_registry import prompt_registry
from workflows.state import TaskItem, make_task

logger = logging.getLogger(__name__)

TASK_KEYS = {
    "generate_recommendations",
}


def build_generate_recommendations_task(
    segment: str = "general",
    priority: int = 2,
    max_retries: int = 3,
) -> TaskItem:
    """
    Build a task to generate personalised product recommendations for a segment.
    """
    input_data = {"segment": segment}
    description = prompt_registry.render_task("generate_recommendations", **input_data)
    return make_task(
        description=description,
        assigned_to="recommender",
        priority=priority,
        input_data=input_data,
        required_tools=["database"],
        max_retries=max_retries,
    )


def build_recommendation_tasks_from_goal(goal: str) -> List[TaskItem]:
    """
    Placeholder for goal‑based task generation.
    """
    tasks = []
    if "recommend" in goal.lower() or "upsell" in goal.lower() or "cross-sell" in goal.lower():
        tasks.append(build_generate_recommendations_task())
    return tasks