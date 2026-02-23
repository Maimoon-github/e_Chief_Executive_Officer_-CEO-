# """
# crews/main_crew.py
# ──────────────────
# CEO-level CrewAI crew.
# Provides a pure-CrewAI alternative execution path (without LangGraph)
# for simpler single-objective runs or testing the CEO reasoning alone.
# """
# from __future__ import annotations

# import json
# import logging
# from typing import Any, Dict

# import yaml
# from crewai import Agent, Crew, Process, Task

# from models.ollama_loader import get_llm
# from tools.db_tool import DatabaseTool
# from tools.email_tool import EmailTool
# from tools.search_tool import SearchTool
# from tools.shopify_tool import ShopifyTool

# logger = logging.getLogger(__name__)


# def build_main_crew(goal: str) -> Crew:
#     """
#     Build the full 7-agent hierarchical CrewAI crew.
#     CEO coordinates all team leads and workers in a single crew run.
#     """
#     with open("config/agents.yaml") as f:
#         cfg = yaml.safe_load(f)
#     with open("config/tasks.yaml") as f:
#         tasks_cfg = yaml.safe_load(f)

#     llm_ceo     = get_llm("ceo")
#     llm_lead    = get_llm("customer_captain")
#     llm_worker  = get_llm("chat_buddy")

#     all_tools = [ShopifyTool(), DatabaseTool(), SearchTool(), EmailTool()]

#     def _agent(key: str, tools=None, llm=None) -> Agent:
#         c = cfg[key]
#         return Agent(
#             role=c["role"],
#             goal=c["goal"],
#             backstory=c["backstory"],
#             tools=tools or [],
#             llm=llm or llm_worker,
#             verbose=False,
#             allow_delegation=c.get("allow_delegation", False),
#             max_iter=c.get("max_iter", 5),
#         )

#     ceo             = _agent("ceo",              llm=llm_ceo)
#     customer_captain = _agent("customer_captain", llm=llm_lead)
#     chat_buddy       = _agent("chat_buddy",       tools=all_tools)
#     stock_sergeant   = _agent("stock_sergeant",   llm=llm_lead)
#     stock_scout      = _agent("stock_scout",      tools=all_tools)
#     promo_general    = _agent("promo_general",    llm=llm_lead)
#     recommender      = _agent("recommender",      tools=all_tools)

#     main_task = Task(
#         description=(
#             f"Execute the following business goal for PaddleAurum: {goal}\n\n"
#             "Coordinate all teams:\n"
#             "1. Delegate customer support tasks to Customer Captain → Chat Buddy\n"
#             "2. Delegate inventory tasks to Stock Sergeant → Stock Scout\n"
#             "3. Delegate marketing tasks to Promo General → Recommender\n"
#             "4. Collect all outputs and produce an executive summary.\n"
#             "Return a final JSON report with: summary, wins, blockers, next_actions."
#         ),
#         agent=ceo,
#         expected_output=(
#             "JSON with: summary, wins (list), blockers (list), next_actions (list)"
#         ),
#     )

#     return Crew(
#         agents=[ceo, customer_captain, chat_buddy,
#                 stock_sergeant, stock_scout,
#                 promo_general, recommender],
#         tasks=[main_task],
#         process=Process.hierarchical,
#         manager_agent=ceo,
#         verbose=True,
#     )


# async def run_main_crew(goal: str) -> Dict[str, Any]:
#     """Run the full CEO crew for a goal. Returns parsed report."""
#     import asyncio
#     crew = build_main_crew(goal)
#     loop = asyncio.get_event_loop()
#     try:
#         raw = await loop.run_in_executor(None, crew.kickoff)
#         clean = str(raw).strip().lstrip("```json").lstrip("```").rstrip("```")
#         return json.loads(clean)
#     except Exception as exc:
#         logger.error("Main crew failed: %s", exc)
#         return {"error": str(exc), "goal": goal}



























# 3w#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
























"""
crews/main_crew.py
──────────────────
CEO-level CrewAI crew.
Provides a pure-CrewAI alternative execution path (without LangGraph)
for simpler single-objective runs or testing the CEO reasoning alone.

NOTE: This crew is currently not wired into the main LangGraph workflow (workflows/graph.py).
It exists as a standalone CrewAI alternative for testing or single‑purpose runs.
To integrate, the node functions in agents/ would need to delegate to these crews.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import yaml
from crewai import Agent, Crew, Process, Task

from models.ollama_loader import get_llm
from tools.db_tool import DatabaseTool
from tools.email_tool import EmailTool
from tools.search_tool import SearchTool
from tools.shopify_tool import ShopifyTool

logger = logging.getLogger(__name__)


def build_main_crew(goal: str) -> Crew:
    """
    Build the full 7-agent hierarchical CrewAI crew.
    CEO coordinates all team leads and workers in a single crew run.
    """
    with open("config/agents.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("config/tasks.yaml") as f:
        tasks_cfg = yaml.safe_load(f)

    llm_ceo     = get_llm("ceo")
    llm_lead    = get_llm("customer_captain")
    llm_worker  = get_llm("chat_buddy")

    all_tools = [ShopifyTool(), DatabaseTool(), SearchTool(), EmailTool()]

    def _agent(key: str, tools=None, llm=None) -> Agent:
        c = cfg[key]
        return Agent(
            role=c["role"],
            goal=c["goal"],
            backstory=c["backstory"],
            tools=tools or [],
            llm=llm or llm_worker,
            verbose=False,
            allow_delegation=c.get("allow_delegation", False),
            max_iter=c.get("max_iter", 5),
        )

    ceo             = _agent("ceo",              llm=llm_ceo)
    customer_captain = _agent("customer_captain", llm=llm_lead)
    chat_buddy       = _agent("chat_buddy",       tools=all_tools)
    stock_sergeant   = _agent("stock_sergeant",   llm=llm_lead)
    stock_scout      = _agent("stock_scout",      tools=all_tools)
    promo_general    = _agent("promo_general",    llm=llm_lead)
    recommender      = _agent("recommender",      tools=all_tools)

    main_task = Task(
        description=(
            f"Execute the following business goal for PaddleAurum: {goal}\n\n"
            "Coordinate all teams:\n"
            "1. Delegate customer support tasks to Customer Captain → Chat Buddy\n"
            "2. Delegate inventory tasks to Stock Sergeant → Stock Scout\n"
            "3. Delegate marketing tasks to Promo General → Recommender\n"
            "4. Collect all outputs and produce an executive summary.\n"
            "Return a final JSON report with: summary, wins, blockers, next_actions."
        ),
        agent=ceo,
        expected_output=(
            "JSON with: summary, wins (list), blockers (list), next_actions (list)"
        ),
    )

    return Crew(
        agents=[ceo, customer_captain, chat_buddy,
                stock_sergeant, stock_scout,
                promo_general, recommender],
        tasks=[main_task],
        process=Process.hierarchical,
        manager_agent=ceo,
        verbose=True,
    )


async def run_main_crew(goal: str) -> Dict[str, Any]:
    """Run the full CEO crew for a goal. Returns parsed report."""
    import asyncio
    crew = build_main_crew(goal)
    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, crew.kickoff)
        clean = str(raw).strip().lstrip("```json").lstrip("```").rstrip("```")
        return json.loads(clean)
    except Exception as exc:
        logger.error("Main crew failed: %s", exc)
        return {"error": str(exc), "goal": goal}