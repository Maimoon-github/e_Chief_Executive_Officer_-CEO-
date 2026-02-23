# """
# crews/support_crew.py
# ─────────────────────
# CrewAI crew for the customer support team.
# Used as an alternative entry point for support-only runs
# (e.g., real-time chat webhook without the full graph).
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


# def build_support_crew(message: str, customer_id: str = "", order_id: str = "") -> Crew:
#     """Build a CrewAI hierarchical crew for a single support inquiry."""
#     with open("config/agents.yaml") as f:
#         cfg = yaml.safe_load(f)

#     llm_lead   = get_llm("customer_captain")
#     llm_worker = get_llm("chat_buddy")

#     tools = [ShopifyTool(), DatabaseTool(), SearchTool(), EmailTool()]

#     captain = Agent(
#         role=cfg["customer_captain"]["role"],
#         goal=cfg["customer_captain"]["goal"],
#         backstory=cfg["customer_captain"]["backstory"],
#         llm=llm_lead,
#         verbose=False,
#         allow_delegation=True,
#         max_iter=cfg["customer_captain"]["max_iter"],
#     )

#     buddy = Agent(
#         role=cfg["chat_buddy"]["role"],
#         goal=cfg["chat_buddy"]["goal"],
#         backstory=cfg["chat_buddy"]["backstory"],
#         tools=tools,
#         llm=llm_worker,
#         verbose=False,
#         allow_delegation=False,
#         max_iter=cfg["chat_buddy"]["max_iter"],
#     )

#     task = Task(
#         description=(
#             f"Handle this customer inquiry:\n"
#             f"Message: {message}\n"
#             f"Customer ID: {customer_id or 'unknown'}\n"
#             f"Order ID: {order_id or 'none'}\n\n"
#             "Fetch relevant Shopify order details and customer profile. "
#             "Respond with JSON: reply_text, escalate (bool), "
#             "sentiment_score (float), resolution_type (string)."
#         ),
#         agent=buddy,
#         expected_output=(
#             "JSON object with reply_text, escalate, "
#             "sentiment_score, resolution_type"
#         ),
#     )

#     return Crew(
#         agents=[captain, buddy],
#         tasks=[task],
#         process=Process.hierarchical,
#         manager_agent=captain,
#         verbose=False,
#     )


# async def run_support_crew(
#     message: str,
#     customer_id: str = "",
#     order_id: str = "",
# ) -> Dict[str, Any]:
#     """
#     Convenience async wrapper to run the support crew for a single inquiry.
#     Returns the parsed JSON result dict.
#     """
#     import asyncio

#     crew = build_support_crew(message, customer_id, order_id)

#     loop = asyncio.get_event_loop()
#     try:
#         raw_result = await loop.run_in_executor(None, crew.kickoff)
#         clean = str(raw_result).strip().lstrip("```json").lstrip("```").rstrip("```")
#         return json.loads(clean)
#     except json.JSONDecodeError:
#         logger.warning("Support crew returned non-JSON: %s", raw_result)
#         return {
#             "reply_text": str(raw_result),
#             "escalate": False,
#             "sentiment_score": 0.5,
#             "resolution_type": "answered",
#         }
#     except Exception as exc:
#         logger.error("Support crew failed: %s", exc)
#         return {
#             "reply_text": "We're experiencing issues. Our team will be in touch shortly.",
#             "escalate": True,
#             "sentiment_score": 0.3,
#             "resolution_type": "escalated",
#         }































#2###################################################################################














"""
crews/support_crew.py
─────────────────────
CrewAI crew for the customer support team.
Used as an alternative entry point for support-only runs
(e.g., real-time chat webhook without the full graph).

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


def build_support_crew(message: str, customer_id: str = "", order_id: str = "") -> Crew:
    """Build a CrewAI hierarchical crew for a single support inquiry."""
    with open("config/agents.yaml") as f:
        cfg = yaml.safe_load(f)

    llm_lead   = get_llm("customer_captain")
    llm_worker = get_llm("chat_buddy")

    tools = [ShopifyTool(), DatabaseTool(), SearchTool(), EmailTool()]

    captain = Agent(
        role=cfg["customer_captain"]["role"],
        goal=cfg["customer_captain"]["goal"],
        backstory=cfg["customer_captain"]["backstory"],
        llm=llm_lead,
        verbose=False,
        allow_delegation=True,
        max_iter=cfg["customer_captain"]["max_iter"],
    )

    buddy = Agent(
        role=cfg["chat_buddy"]["role"],
        goal=cfg["chat_buddy"]["goal"],
        backstory=cfg["chat_buddy"]["backstory"],
        tools=tools,
        llm=llm_worker,
        verbose=False,
        allow_delegation=False,
        max_iter=cfg["chat_buddy"]["max_iter"],
    )

    task = Task(
        description=(
            f"Handle this customer inquiry:\n"
            f"Message: {message}\n"
            f"Customer ID: {customer_id or 'unknown'}\n"
            f"Order ID: {order_id or 'none'}\n\n"
            "Fetch relevant Shopify order details and customer profile. "
            "Respond with JSON: reply_text, escalate (bool), "
            "sentiment_score (float), resolution_type (string)."
        ),
        agent=buddy,
        expected_output=(
            "JSON object with reply_text, escalate, "
            "sentiment_score, resolution_type"
        ),
    )

    return Crew(
        agents=[captain, buddy],
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=captain,
        verbose=False,
    )


async def run_support_crew(
    message: str,
    customer_id: str = "",
    order_id: str = "",
) -> Dict[str, Any]:
    """
    Convenience async wrapper to run the support crew for a single inquiry.
    Returns the parsed JSON result dict.
    """
    import asyncio

    crew = build_support_crew(message, customer_id, order_id)

    loop = asyncio.get_event_loop()
    try:
        raw_result = await loop.run_in_executor(None, crew.kickoff)
        clean = str(raw_result).strip().lstrip("```json").lstrip("```").rstrip("```")
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Support crew returned non-JSON: %s", raw_result)
        return {
            "reply_text": str(raw_result),
            "escalate": False,
            "sentiment_score": 0.5,
            "resolution_type": "answered",
        }
    except Exception as exc:
        logger.error("Support crew failed: %s", exc)
        return {
            "reply_text": "We're experiencing issues. Our team will be in touch shortly.",
            "escalate": True,
            "sentiment_score": 0.3,
            "resolution_type": "escalated",
        }