"""
crews/inventory_crew.py
───────────────────────
CrewAI crew for the inventory management team.
Standalone crew for inventory-only runs or testing.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import yaml
from crewai import Agent, Crew, Process, Task

from models.ollama_loader import get_llm
from tools.browser_tool import BrowserTool
from tools.db_tool import DatabaseTool
from tools.email_tool import EmailTool
from tools.shopify_tool import ShopifyTool

logger = logging.getLogger(__name__)


def build_inventory_crew() -> Crew:
    """Build a CrewAI hierarchical crew for inventory scanning."""
    with open("config/agents.yaml") as f:
        cfg = yaml.safe_load(f)

    llm_lead   = get_llm("stock_sergeant")
    llm_worker = get_llm("stock_scout")

    tools = [ShopifyTool(), DatabaseTool(), BrowserTool(), EmailTool()]

    sergeant = Agent(
        role=cfg["stock_sergeant"]["role"],
        goal=cfg["stock_sergeant"]["goal"],
        backstory=cfg["stock_sergeant"]["backstory"],
        llm=llm_lead,
        verbose=False,
        allow_delegation=True,
        max_iter=cfg["stock_sergeant"]["max_iter"],
    )

    scout = Agent(
        role=cfg["stock_scout"]["role"],
        goal=cfg["stock_scout"]["goal"],
        backstory=cfg["stock_scout"]["backstory"],
        tools=tools,
        llm=llm_worker,
        verbose=False,
        allow_delegation=False,
        max_iter=cfg["stock_scout"]["max_iter"],
    )

    scan_task = Task(
        description=(
            "Perform a full inventory scan:\n"
            "1. Use the Shopify API to fetch all active products and inventory levels.\n"
            "2. Check the database for reorder thresholds (default min_qty=10).\n"
            "3. Calculate days_until_stockout using 30-day sales velocity.\n"
            "4. For CRITICAL items (< 7 days), check supplier availability via browser.\n"
            "5. Return a full inventory report as JSON."
        ),
        agent=scout,
        expected_output=(
            "JSON object with: total_products_scanned, low_stock_alerts (list), "
            "reorder_recommendations (list), supplier_checks (list)"
        ),
    )

    return Crew(
        agents=[sergeant, scout],
        tasks=[scan_task],
        process=Process.hierarchical,
        manager_agent=sergeant,
        verbose=False,
    )


async def run_inventory_crew() -> Dict[str, Any]:
    """Run the inventory crew and return structured results."""
    import asyncio

    crew = build_inventory_crew()
    loop = asyncio.get_event_loop()

    try:
        raw = await loop.run_in_executor(None, crew.kickoff)
        clean = str(raw).strip().lstrip("```json").lstrip("```").rstrip("```")
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Inventory crew returned non-JSON")
        return {"raw_output": str(raw), "error": "parse_failed"}
    except Exception as exc:
        logger.error("Inventory crew failed: %s", exc)
        return {"error": str(exc)}
