# tasks/inventory_tasks.py
"""
tasks/inventory_tasks.py
────────────────────────
Task builders for inventory management.
Loads templates from tasks.yaml and returns TaskItem objects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.prompt_registry import prompt_registry
from workflows.state import TaskItem, make_task

logger = logging.getLogger(__name__)

TASK_KEYS = {
    "scan_inventory_levels",
    "check_supplier_availability",
}


def build_scan_inventory_task(
    priority: int = 2,
    input_data: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> TaskItem:
    """
    Build a task to scan all inventory levels and generate low‑stock alerts.
    """
    description = prompt_registry.render_task(
        "scan_inventory_levels",
        **input_data or {}
    )
    return make_task(
        description=description,
        assigned_to="stock_scout",
        priority=priority,
        input_data=input_data or {},
        required_tools=["shopify_api", "database"],
        max_retries=max_retries,
    )


def build_supplier_check_task(
    product_id: str,
    supplier_url: str,
    priority: int = 3,
    max_retries: int = 2,
) -> TaskItem:
    """
    Build a task to check a specific supplier's stock/price for a product.
    """
    input_data = {
        "product_id": product_id,
        "supplier_url": supplier_url,
    }
    description = prompt_registry.render_task(
        "check_supplier_availability",
        **input_data
    )
    return make_task(
        description=description,
        assigned_to="stock_scout",
        priority=priority,
        input_data=input_data,
        required_tools=["browser_scraper", "database"],
        max_retries=max_retries,
    )


def build_inventory_tasks_from_goal(goal: str) -> List[TaskItem]:
    """
    Higher‑level builder: given a goal related to inventory,
    create an appropriate set of tasks.
    This is a placeholder; in a full implementation it might use the CEO LLM.
    """
    # Simple heuristic: always run a full scan, and for any critical alerts
    # we would later create supplier checks, but that depends on scan output.
    return [build_scan_inventory_task(input_data={"goal": goal})]