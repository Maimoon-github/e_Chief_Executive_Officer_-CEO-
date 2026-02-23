"""
agents/stock_sergeant.py
────────────────────────
NODE 2 — Stock Sergeant (Team Lead)

Responsibilities:
  - Oversee Stock Scout results
  - Approve or escalate restock recommendations
  - Send restock alert emails for CRITICAL items
  - Update shared_context with actionable restock decisions
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from tools.email_tool import send_restock_alert
from workflows.state import PaddleAurumState

logger = logging.getLogger(__name__)

INVENTORY_KEYWORDS = {
    "stock", "inventory", "restock", "reorder", "supply", "warehouse",
    "sold out", "shortage", "level", "quantity", "replenish"
}


def _is_inventory_task(task) -> bool:
    if task["assigned_to"] in ("stock_scout", "stock_sergeant"):
        return True
    desc_lower = task["description"].lower()
    return any(kw in desc_lower for kw in INVENTORY_KEYWORDS)


async def _send_restock_emails(alerts: List[Dict]) -> List[Dict]:
    """Send restock alert emails for CRITICAL items."""
    sent = []
    for alert in alerts:
        if alert.get("severity") == "CRITICAL":
            try:
                await send_restock_alert(
                    product_name=alert["product_name"],
                    current_qty=alert["current_qty"],
                    reorder_qty=alert["reorder_qty"],
                    supplier_url=alert.get("supplier_url", "N/A"),
                )
                sent.append(alert["product_id"])
                logger.info("[Stock Sergeant] Restock alert sent for %s", alert["product_name"])
            except Exception as exc:
                logger.error("Could not send restock email for %s: %s",
                             alert.get("product_name"), exc)
    return sent


async def stock_sergeant_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    LangGraph node for Stock Sergeant.
    Delegates scanning to Stock Scout, then reviews and acts on critical alerts.
    """
    state["current_step"] = "stock_sergeant"

    inventory_tasks = [t for t in state["task_queue"] if _is_inventory_task(t)]
    if inventory_tasks:
        logger.info("[Stock Sergeant] Routing %d inventory tasks to Stock Scout", len(inventory_tasks))
        for task in state["task_queue"]:
            if _is_inventory_task(task) and task["assigned_to"] == "stock_sergeant":
                task["assigned_to"] = "stock_scout"

    state["short_term_memory"].append({
        "agent_id": "stock_sergeant",
        "role": "assistant",
        "content": f"Delegated {len(inventory_tasks)} inventory tasks to Stock Scout.",
        "timestamp": time.time(),
        "tool_calls": None,
        "tool_results": None,
    })
    return state


async def stock_sergeant_review_node(state: PaddleAurumState) -> PaddleAurumState:
    """
    Post-scan review — send emails and update decisions.
    Called after Stock Scout completes.
    """
    inv_output = state.get("inventory_output")
    if not inv_output:
        return state

    alerts = inv_output.get("low_stock_alerts", [])
    emails_sent = await _send_restock_emails(alerts)

    state["inventory_output"]["restock_emails_sent"] = emails_sent
    state["inventory_output"]["reviewed_by"] = "stock_sergeant"

    critical_count = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
    logger.info("[Stock Sergeant] Review complete. %d critical alerts, %d emails sent.",
                critical_count, len(emails_sent))

    return state
