# """
# crews/marketing_crew.py
# ───────────────────────
# CrewAI crew for the marketing team.
# Standalone for marketing-only runs.
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


# def build_marketing_crew(objective: str, segment: str = "general") -> Crew:
#     """Build the marketing crew for a given campaign objective."""
#     with open("config/agents.yaml") as f:
#         cfg = yaml.safe_load(f)

#     llm_lead   = get_llm("promo_general")
#     llm_worker = get_llm("recommender")

#     tools = [ShopifyTool(), DatabaseTool(), SearchTool(), EmailTool()]

#     general = Agent(
#         role=cfg["promo_general"]["role"],
#         goal=cfg["promo_general"]["goal"],
#         backstory=cfg["promo_general"]["backstory"],
#         llm=llm_lead,
#         verbose=False,
#         allow_delegation=True,
#         max_iter=cfg["promo_general"]["max_iter"],
#     )

#     rec_agent = Agent(
#         role=cfg["recommender"]["role"],
#         goal=cfg["recommender"]["goal"],
#         backstory=cfg["recommender"]["backstory"],
#         tools=tools,
#         llm=llm_worker,
#         verbose=False,
#         allow_delegation=False,
#         max_iter=cfg["recommender"]["max_iter"],
#     )

#     rec_task = Task(
#         description=(
#             f"Generate product recommendations for the '{segment}' customer segment.\n"
#             "Query the database for purchase histories. "
#             "Use cosine similarity logic to find cross-sell products. "
#             "Return top 3 product recommendations with confidence scores."
#         ),
#         agent=rec_agent,
#         expected_output=(
#             "JSON list of recommendation sets: "
#             "[{customer_id, recommendations: [{product_id, product_name, confidence_score, reason_code}]}]"
#         ),
#     )

#     campaign_task = Task(
#         description=(
#             f"Using the product recommendations from the Recommender, "
#             f"draft a promotional email campaign for the following objective: {objective}.\n"
#             f"Target segment: {segment}.\n"
#             "Research competitor pricing with the search tool. "
#             "Draft the campaign and send it to the segment's email list via the email tool."
#         ),
#         agent=general,
#         expected_output=(
#             "JSON with: subject_line, headline, body, cta_text, cta_url, "
#             "discount_code, campaigns_sent (int)"
#         ),
#     )

#     return Crew(
#         agents=[general, rec_agent],
#         tasks=[rec_task, campaign_task],
#         process=Process.hierarchical,
#         manager_agent=general,
#         verbose=False,
#     )


# async def run_marketing_crew(
#     objective: str, segment: str = "general"
# ) -> Dict[str, Any]:
#     import asyncio
#     crew = build_marketing_crew(objective, segment)
#     loop = asyncio.get_event_loop()
#     try:
#         raw = await loop.run_in_executor(None, crew.kickoff)
#         clean = str(raw).strip().lstrip("```json").lstrip("```").rstrip("```")
#         return json.loads(clean)
#     except Exception as exc:
#         logger.error("Marketing crew failed: %s", exc)
#         return {"error": str(exc)}






















# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
















"""
crews/marketing_crew.py
───────────────────────
CrewAI crew for the marketing team.
Standalone for marketing-only runs.

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


def build_marketing_crew(objective: str, segment: str = "general") -> Crew:
    """Build the marketing crew for a given campaign objective."""
    with open("config/agents.yaml") as f:
        cfg = yaml.safe_load(f)

    llm_lead   = get_llm("promo_general")
    llm_worker = get_llm("recommender")

    tools = [ShopifyTool(), DatabaseTool(), SearchTool(), EmailTool()]

    general = Agent(
        role=cfg["promo_general"]["role"],
        goal=cfg["promo_general"]["goal"],
        backstory=cfg["promo_general"]["backstory"],
        llm=llm_lead,
        verbose=False,
        allow_delegation=True,
        max_iter=cfg["promo_general"]["max_iter"],
    )

    rec_agent = Agent(
        role=cfg["recommender"]["role"],
        goal=cfg["recommender"]["goal"],
        backstory=cfg["recommender"]["backstory"],
        tools=tools,
        llm=llm_worker,
        verbose=False,
        allow_delegation=False,
        max_iter=cfg["recommender"]["max_iter"],
    )

    rec_task = Task(
        description=(
            f"Generate product recommendations for the '{segment}' customer segment.\n"
            "Query the database for purchase histories. "
            "Use cosine similarity logic to find cross-sell products. "
            "Return top 3 product recommendations with confidence scores."
        ),
        agent=rec_agent,
        expected_output=(
            "JSON list of recommendation sets: "
            "[{customer_id, recommendations: [{product_id, product_name, confidence_score, reason_code}]}]"
        ),
    )

    campaign_task = Task(
        description=(
            f"Using the product recommendations from the Recommender, "
            f"draft a promotional email campaign for the following objective: {objective}.\n"
            f"Target segment: {segment}.\n"
            "Research competitor pricing with the search tool. "
            "Draft the campaign and send it to the segment's email list via the email tool."
        ),
        agent=general,
        expected_output=(
            "JSON with: subject_line, headline, body, cta_text, cta_url, "
            "discount_code, campaigns_sent (int)"
        ),
    )

    return Crew(
        agents=[general, rec_agent],
        tasks=[rec_task, campaign_task],
        process=Process.hierarchical,
        manager_agent=general,
        verbose=False,
    )


async def run_marketing_crew(
    objective: str, segment: str = "general"
) -> Dict[str, Any]:
    import asyncio
    crew = build_marketing_crew(objective, segment)
    loop = asyncio.get_event_loop()
    try:
        raw = await loop.run_in_executor(None, crew.kickoff)
        clean = str(raw).strip().lstrip("```json").lstrip("```").rstrip("```")
        return json.loads(clean)
    except Exception as exc:
        logger.error("Marketing crew failed: %s", exc)
        return {"error": str(exc)}