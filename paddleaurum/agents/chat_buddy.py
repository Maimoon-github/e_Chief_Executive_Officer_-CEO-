"""
agents/chat_buddy.py
────────────────────
NODE 1a — Chat Buddy (Worker)

Responsibilities:
  - Answer product questions ("which paddle for beginners?")
  - Resolve order issues (status, refunds, exchanges)
  - Detect sentiment; escalate hostile/complex cases
  - Write output to state["customer_support_output"]
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task

from config.settings import settings
from models.ollama_loader import get_llm_async
from tools.db_tool import DatabaseTool, get_customer_profile, write_agent_memory
from tools.email_tool import EmailTool
from tools.search_tool import SearchTool
from tools.shopify_tool import ShopifyTool, get_order
from workflows.state import PaddleAurumState, TaskItem

logger = logging.getLogger(__name__)


# ── Core response logic ───────────────────────────────────────────────────────

_CHAT_SYSTEM = """You are Chat Buddy, the 24/7 support agent for PaddleAurum,
a pickleball e-commerce brand. You are knowledgeable, friendly, and concise.
You know every paddle SKU, ball type, and shipping policy.
When you don't know something, say so and escalate.
Always respond with valid JSON only."""

_CHAT_TEMPLATE = """
Customer message: "{message}"
Customer ID: {customer_id}
Order ID (if any): {order_id}
Customer history: {history}
Order details (if fetched): {order_details}
Web search results (if relevant): {search_results}

Respond with JSON containing:
  reply_text (string, max 200 words, friendly pickleball voice),
  escalate (boolean — true if requires human or refund > $50),
  sentiment_score (float 0.0-1.0, 1.0 = very positive),
  resolution_type (one of: answered | escalated | refund_initiated | order_updated | needs_info).

JSON only:"""


async def handle_inquiry(
    message: str,
    customer_id: str = "",
    order_id: str = "",
) -> Dict[str, Any]:
    """
    Process a single customer inquiry end-to-end.
    Fetches enrichment data, calls LLM, returns structured response.
    """
    llm = await get_llm_async("chat_buddy")

    # ── Enrich: customer history ───────────────────────────────────────────
    history = {}
    if customer_id:
        profile = await get_customer_profile(customer_id)
        if profile:
            history = {
                "name": profile.get("name"),
                "segment": profile.get("segment"),
                "lifetime_value": profile.get("lifetime_value"),
                "purchase_count": len(json.loads(profile.get("purchase_history", "[]"))),
            }

    # ── Enrich: order details ──────────────────────────────────────────────
    order_details = {}
    if order_id:
        try:
            order = await get_order(order_id)
            if order:
                order_details = {
                    "id": order.get("id"),
                    "status": order.get("financial_status"),
                    "fulfillment_status": order.get("fulfillment_status"),
                    "total": order.get("total_price"),
                    "created_at": order.get("created_at"),
                    "items": [
                        {"name": i.get("name"), "qty": i.get("quantity")}
                        for i in order.get("line_items", [])
                    ],
                }
        except Exception as exc:
            logger.warning("Could not fetch order %s: %s", order_id, exc)

    # ── Enrich: search (if question-type message) ──────────────────────────
    search_results = ""
    question_keywords = ["which", "best", "recommend", "difference", "compare", "what is", "how"]
    if any(kw in message.lower() for kw in question_keywords):
        from tools.search_tool import search_pickleball_faq
        try:
            search_results = await search_pickleball_faq(message)
        except Exception:
            pass

    # ── Call LLM ───────────────────────────────────────────────────────────
    prompt = f"{_CHAT_SYSTEM}\n\n" + _CHAT_TEMPLATE.format(
        message=message,
        customer_id=customer_id or "unknown",
        order_id=order_id or "none",
        history=json.dumps(history),
        order_details=json.dumps(order_details),
        search_results=search_results[:500] if search_results else "none",
    )

    try:
        raw = await llm.ainvoke(prompt)
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        response = json.loads(clean)
    except Exception as exc:
        logger.error("Chat Buddy LLM error: %s", exc)
        response = {
            "reply_text": "I'm having trouble processing your request right now. "
                          "Our team will get back to you within 1 hour.",
            "escalate": True,
            "sentiment_score": 0.5,
            "resolution_type": "escalated",
        }

    # ── Persist to memory ─────────────────────────────────────────────────
    if customer_id:
        await write_agent_memory(
            "session_chat", "chat_buddy", "assistant",
            f"Customer {customer_id}: {message[:100]} → {response.get('resolution_type')}"
        )

    return response


# ── Process all support tasks from state ─────────────────────────────────────

async def process_support_tasks(tasks: List[TaskItem]) -> Dict[str, Any]:
    """Run handle_inquiry for each pending support task."""
    results = []
    for task in tasks:
        task["status"] = "in_progress"
        inp = task.get("input_data", {})
        try:
            result = await handle_inquiry(
                message=inp.get("message", inp.get("description", "")),
                customer_id=inp.get("customer_id", ""),
                order_id=inp.get("order_id", ""),
            )
            task["output_data"] = result
            task["status"] = "done"
            task["completed_at"] = time.time()
        except Exception as exc:
            logger.error("Support task %s failed: %s", task["task_id"], exc)
            task["status"] = "failed"
            task["error_message"] = str(exc)
            task["retries"] += 1
        results.append(task)

    resolved = sum(1 for r in results if r["status"] == "done")
    escalated = sum(
        1 for r in results
        if r["status"] == "done" and r.get("output_data", {}).get("escalate", False)
    )
    return {
        "tasks_processed": len(results),
        "resolved": resolved,
        "escalated": escalated,
        "results": results,
    }


# ── LangGraph node ────────────────────────────────────────────────────────────

async def chat_buddy_node(state: PaddleAurumState) -> PaddleAurumState:
    state["current_step"] = "chat_buddy"
    logger.info("[Chat Buddy] Processing support tasks")

    support_tasks = [
        t for t in state["task_queue"]
        if t["assigned_to"] in ("chat_buddy",) and t["status"] == "pending"
    ]

    if not support_tasks:
        logger.info("[Chat Buddy] No tasks assigned.")
        return state

    output = await process_support_tasks(support_tasks)
    state["customer_support_output"] = output

    # Update task statuses in queue
    task_map = {t["task_id"]: t for t in output["results"]}
    for i, task in enumerate(state["task_queue"]):
        if task["task_id"] in task_map:
            state["task_queue"][i] = task_map[task["task_id"]]

    # Append to short-term memory
    state["short_term_memory"].append({
        "agent_id": "chat_buddy",
        "role": "assistant",
        "content": f"Processed {output['tasks_processed']} support tasks, "
                   f"{output['resolved']} resolved, {output['escalated']} escalated.",
        "timestamp": time.time(),
        "tool_calls": None,
        "tool_results": None,
    })

    return state
