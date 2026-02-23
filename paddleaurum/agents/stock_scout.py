# """
# agents/stock_scout.py
# ─────────────────────
# NODE 2a — Stock Scout (Worker)

# Responsibilities:
#   - Poll Shopify API for all product inventory levels
#   - Compare against reorder thresholds stored in SQLite
#   - Calculate days_until_stockout from sales velocity
#   - Optionally verify supplier availability via browser scraping
#   - Write structured alerts to state["inventory_output"]
# """
# from __future__ import annotations

# import json
# import logging
# import time
# from typing import Any, Dict, List, Optional

# from config.settings import settings
# from tools.browser_tool import check_supplier_stock
# from tools.db_tool import (
#     get_all_thresholds,
#     get_sales_velocity,
#     upsert_threshold,
# )
# from tools.shopify_tool import (
#     get_inventory_levels,
#     get_products,
#     get_sales_by_product,
# )
# from workflows.state import LowStockAlert, PaddleAurumState, TaskItem

# logger = logging.getLogger(__name__)

# # Severity thresholds (days until stockout)
# CRITICAL_DAYS = 7
# WARNING_DAYS = 14


# async def calculate_days_until_stockout(
#     product_id: str, current_qty: int
# ) -> float:
#     """
#     Estimate days until stockout using 30-day sales velocity.
#     Returns float('inf') if no sales data or velocity is 0.
#     """
#     velocity = await get_sales_velocity(product_id)
#     if velocity and velocity.get("avg_daily_sales", 0) > 0:
#         return current_qty / velocity["avg_daily_sales"]

#     # Fallback: calculate from Shopify orders
#     try:
#         units_30d = await get_sales_by_product(product_id, days=30)
#         avg_daily = units_30d / 30.0
#         if avg_daily > 0:
#             return current_qty / avg_daily
#     except Exception as exc:
#         logger.warning("Could not calculate velocity for %s: %s", product_id, exc)

#     return float("inf")


# async def scan_inventory() -> List[LowStockAlert]:
#     """
#     Main inventory scan:
#     1. Fetch all active products from Shopify
#     2. Get inventory levels
#     3. Compare against DB thresholds
#     4. Calculate stockout estimates
#     5. Return sorted alert list
#     """
#     alerts: List[LowStockAlert] = []

#     # Get all products
#     try:
#         products = await get_products(limit=250)
#     except Exception as exc:
#         logger.error("Could not fetch products from Shopify: %s", exc)
#         return alerts

#     # Get all thresholds from DB
#     thresholds = await get_all_thresholds()
#     threshold_map: Dict[str, Dict] = {t["product_id"]: t for t in thresholds}

#     # Get inventory levels
#     try:
#         inventory_levels = await get_inventory_levels()
#     except Exception as exc:
#         logger.error("Could not fetch inventory levels: %s", exc)
#         return alerts

#     # Build inventory_item_id → quantity map
#     inv_map: Dict[str, int] = {
#         str(il["inventory_item_id"]): il.get("available", 0)
#         for il in inventory_levels
#     }

#     for product in products:
#         product_id = str(product.get("id", ""))
#         product_name = product.get("title", "Unknown")

#         # Aggregate quantity across all variants
#         total_qty = 0
#         for variant in product.get("variants", []):
#             inv_item_id = str(variant.get("inventory_item_id", ""))
#             total_qty += inv_map.get(inv_item_id, 0)

#         # Get threshold (default if not set)
#         threshold_data = threshold_map.get(product_id, {})
#         min_qty = threshold_data.get("min_qty", 10)
#         reorder_qty = threshold_data.get("reorder_qty", 50)
#         supplier_url = threshold_data.get("supplier_url", "")

#         if total_qty <= min_qty:
#             days = await calculate_days_until_stockout(product_id, total_qty)

#             if days <= CRITICAL_DAYS:
#                 severity = "CRITICAL"
#             elif days <= WARNING_DAYS:
#                 severity = "WARNING"
#             else:
#                 severity = "OK" if total_qty > min_qty else "WARNING"

#             alerts.append(
#                 LowStockAlert(
#                     product_id=product_id,
#                     sku=product.get("variants", [{}])[0].get("sku", ""),
#                     product_name=product_name,
#                     current_qty=total_qty,
#                     threshold=min_qty,
#                     days_until_stockout=round(days, 1) if days != float("inf") else 999,
#                     severity=severity,
#                     reorder_qty=reorder_qty,
#                     supplier_url=supplier_url,
#                 )
#             )

#     # Sort: CRITICAL first, then WARNING, then by days_until_stockout
#     severity_order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
#     alerts.sort(key=lambda a: (severity_order.get(a["severity"], 2), a["days_until_stockout"]))

#     logger.info(
#         "Inventory scan complete: %d alerts (%d CRITICAL, %d WARNING)",
#         len(alerts),
#         sum(1 for a in alerts if a["severity"] == "CRITICAL"),
#         sum(1 for a in alerts if a["severity"] == "WARNING"),
#     )
#     return alerts


# async def verify_suppliers(alerts: List[LowStockAlert]) -> List[Dict]:
#     """
#     For CRITICAL alerts with a supplier URL, scrape the supplier page.
#     Returns list of supplier check results.
#     """
#     checks = []
#     critical = [a for a in alerts if a["severity"] == "CRITICAL" and a.get("supplier_url")]
#     for alert in critical[:5]:  # Cap at 5 browser scrapes per run
#         try:
#             result = await check_supplier_stock(alert["supplier_url"], alert["product_name"])
#             checks.append(result)
#         except Exception as exc:
#             logger.error("Supplier check failed for %s: %s", alert["product_name"], exc)
#             checks.append({"product": alert["product_name"], "error": str(exc)})
#     return checks


# async def process_inventory_tasks(tasks: List[TaskItem]) -> Dict[str, Any]:
#     """Process all pending inventory tasks."""
#     alerts = await scan_inventory()
#     supplier_checks = await verify_suppliers(alerts)

#     reorder_recommendations = [
#         {
#             "product_id": a["product_id"],
#             "product_name": a["product_name"],
#             "current_qty": a["current_qty"],
#             "reorder_qty": a["reorder_qty"],
#             "days_until_stockout": a["days_until_stockout"],
#             "severity": a["severity"],
#         }
#         for a in alerts if a["severity"] in ("CRITICAL", "WARNING")
#     ]

#     # Mark tasks done
#     for task in tasks:
#         task["status"] = "done"
#         task["completed_at"] = time.time()
#         task["output_data"] = {"alerts_count": len(alerts)}

#     return {
#         "total_products_scanned": len(alerts) + 1,  # approximate
#         "low_stock_alerts": [dict(a) for a in alerts],
#         "reorder_recommendations": reorder_recommendations,
#         "supplier_checks": supplier_checks,
#         "scan_timestamp": time.time(),
#     }


# # ── LangGraph node ────────────────────────────────────────────────────────────

# async def stock_scout_node(state: PaddleAurumState) -> PaddleAurumState:
#     state["current_step"] = "stock_scout"
#     logger.info("[Stock Scout] Starting inventory scan")

#     inventory_tasks = [
#         t for t in state["task_queue"]
#         if t["assigned_to"] in ("stock_scout",) and t["status"] == "pending"
#     ]

#     if not inventory_tasks:
#         # Run a scan anyway as part of the monitoring cycle
#         inventory_tasks = []

#     output = await process_inventory_tasks(inventory_tasks)
#     state["inventory_output"] = output

#     # Surface critical alerts in shared context for other agents
#     critical = [a for a in output["low_stock_alerts"] if a["severity"] == "CRITICAL"]
#     state["shared_context"]["low_stock_alerts"] = critical

#     state["short_term_memory"].append({
#         "agent_id": "stock_scout",
#         "role": "assistant",
#         "content": (
#             f"Inventory scan: {len(output['low_stock_alerts'])} alerts, "
#             f"{len(critical)} critical."
#         ),
#         "timestamp": time.time(),
#         "tool_calls": None,
#         "tool_results": None,
#     })

#     return state































# 23#######################################################################################################
























"""
agents/stock_scout.py
─────────────────────
NODE 2a — Stock Scout (Worker)

Responsibilities:
  - Poll Shopify API for all product inventory levels
  - Compare against reorder thresholds stored in SQLite
  - Calculate days_until_stockout from sales velocity
  - Optionally verify supplier availability via browser scraping
  - Write structured alerts to state["inventory_output"]
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from tools.browser_tool import check_supplier_stock
from tools.db_tool import (
    get_all_thresholds,
    get_sales_velocity,
    upsert_threshold,
)
from tools.shopify_tool import (
    get_inventory_levels,
    get_products,
    get_sales_by_product,
)
from workflows.state import LowStockAlert, PaddleAurumState, TaskItem

logger = logging.getLogger(__name__)

# Severity thresholds (days until stockout)
CRITICAL_DAYS = 7
WARNING_DAYS = 14


async def calculate_days_until_stockout(
    product_id: str, current_qty: int
) -> float:
    """
    Estimate days until stockout using 30-day sales velocity.
    Returns float('inf') if no sales data or velocity is 0.
    """
    velocity = await get_sales_velocity(product_id)
    if velocity and velocity.get("avg_daily_sales", 0) > 0:
        return current_qty / velocity["avg_daily_sales"]

    # Fallback: calculate from Shopify orders
    try:
        units_30d = await get_sales_by_product(product_id, days=30)
        avg_daily = units_30d / 30.0
        if avg_daily > 0:
            return current_qty / avg_daily
    except Exception as exc:
        logger.warning("Could not calculate velocity for %s: %s", product_id, exc)

    return float("inf")


async def scan_inventory() -> Tuple[List[LowStockAlert], int]:
    """
    Main inventory scan:
    1. Fetch all active products from Shopify
    2. Get inventory levels
    3. Compare against DB thresholds
    4. Calculate stockout estimates
    5. Return sorted alert list and total products scanned
    """
    alerts: List[LowStockAlert] = []

    # Get all products
    try:
        products = await get_products(limit=250)
    except Exception as exc:
        logger.error("Could not fetch products from Shopify: %s", exc)
        return alerts, 0

    # Get all thresholds from DB
    thresholds = await get_all_thresholds()
    threshold_map: Dict[str, Dict] = {t["product_id"]: t for t in thresholds}

    # Get inventory levels
    try:
        inventory_levels = await get_inventory_levels()
    except Exception as exc:
        logger.error("Could not fetch inventory levels: %s", exc)
        return alerts, len(products)  # still count products even if inventory fails

    # Build inventory_item_id → quantity map
    inv_map: Dict[str, int] = {
        str(il["inventory_item_id"]): il.get("available", 0)
        for il in inventory_levels
    }

    for product in products:
        product_id = str(product.get("id", ""))
        product_name = product.get("title", "Unknown")

        # Aggregate quantity across all variants
        total_qty = 0
        for variant in product.get("variants", []):
            inv_item_id = str(variant.get("inventory_item_id", ""))
            total_qty += inv_map.get(inv_item_id, 0)

        # Get threshold (default if not set)
        threshold_data = threshold_map.get(product_id, {})
        min_qty = threshold_data.get("min_qty", 10)
        reorder_qty = threshold_data.get("reorder_qty", 50)
        supplier_url = threshold_data.get("supplier_url", "")

        if total_qty <= min_qty:
            days = await calculate_days_until_stockout(product_id, total_qty)

            if days <= CRITICAL_DAYS:
                severity = "CRITICAL"
            elif days <= WARNING_DAYS:
                severity = "WARNING"
            else:
                severity = "OK" if total_qty > min_qty else "WARNING"

            alerts.append(
                LowStockAlert(
                    product_id=product_id,
                    sku=product.get("variants", [{}])[0].get("sku", ""),
                    product_name=product_name,
                    current_qty=total_qty,
                    threshold=min_qty,
                    days_until_stockout=round(days, 1) if days != float("inf") else 999,
                    severity=severity,
                    reorder_qty=reorder_qty,
                    supplier_url=supplier_url,
                )
            )

    # Sort: CRITICAL first, then WARNING, then by days_until_stockout
    severity_order = {"CRITICAL": 0, "WARNING": 1, "OK": 2}
    alerts.sort(key=lambda a: (severity_order.get(a["severity"], 2), a["days_until_stockout"]))

    logger.info(
        "Inventory scan complete: %d alerts (%d CRITICAL, %d WARNING)",
        len(alerts),
        sum(1 for a in alerts if a["severity"] == "CRITICAL"),
        sum(1 for a in alerts if a["severity"] == "WARNING"),
    )
    return alerts, len(products)


async def verify_suppliers(alerts: List[LowStockAlert]) -> List[Dict]:
    """
    For CRITICAL alerts with a supplier URL, scrape the supplier page.
    Returns list of supplier check results.
    """
    checks = []
    critical = [a for a in alerts if a["severity"] == "CRITICAL" and a.get("supplier_url")]
    for alert in critical[:5]:  # Cap at 5 browser scrapes per run
        try:
            result = await check_supplier_stock(alert["supplier_url"], alert["product_name"])
            checks.append(result)
        except Exception as exc:
            logger.error("Supplier check failed for %s: %s", alert["product_name"], exc)
            checks.append({"product": alert["product_name"], "error": str(exc)})
    return checks


async def process_inventory_tasks(tasks: List[TaskItem]) -> Dict[str, Any]:
    """Process all pending inventory tasks."""
    alerts, total_products = await scan_inventory()
    supplier_checks = await verify_suppliers(alerts)

    reorder_recommendations = [
        {
            "product_id": a["product_id"],
            "product_name": a["product_name"],
            "current_qty": a["current_qty"],
            "reorder_qty": a["reorder_qty"],
            "days_until_stockout": a["days_until_stockout"],
            "severity": a["severity"],
        }
        for a in alerts if a["severity"] in ("CRITICAL", "WARNING")
    ]

    # Mark tasks done
    for task in tasks:
        task["status"] = "done"
        task["completed_at"] = time.time()
        task["output_data"] = {"alerts_count": len(alerts)}

    return {
        "total_products_scanned": total_products,
        "low_stock_alerts": [dict(a) for a in alerts],
        "reorder_recommendations": reorder_recommendations,
        "supplier_checks": supplier_checks,
        "scan_timestamp": time.time(),
    }


# ── LangGraph node ────────────────────────────────────────────────────────────

async def stock_scout_node(state: PaddleAurumState) -> PaddleAurumState:
    state["current_step"] = "stock_scout"
    logger.info("[Stock Scout] Starting inventory scan")

    inventory_tasks = [
        t for t in state["task_queue"]
        if t["assigned_to"] in ("stock_scout",) and t["status"] == "pending"
    ]

    if not inventory_tasks:
        # Run a scan anyway as part of the monitoring cycle
        inventory_tasks = []

    output = await process_inventory_tasks(inventory_tasks)
    state["inventory_output"] = output

    # Surface critical alerts in shared context for other agents
    critical = [a for a in output["low_stock_alerts"] if a["severity"] == "CRITICAL"]
    state["shared_context"]["low_stock_alerts"] = critical

    state["short_term_memory"].append({
        "agent_id": "stock_scout",
        "role": "assistant",
        "content": (
            f"Inventory scan: {len(output['low_stock_alerts'])} alerts, "
            f"{len(critical)} critical."
        ),
        "timestamp": time.time(),
        "tool_calls": None,
        "tool_results": None,
    })

    return state