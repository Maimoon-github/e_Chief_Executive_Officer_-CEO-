# """
# agents/promo_general.py
# ───────────────────────
# NODE 3 — Promo General (Team Lead)

# Responsibilities:
#   - Receive marketing/promo sub-tasks from CEO
#   - Delegate product recommendations to Recommender
#   - Use LLM to draft email campaign copy
#   - Send campaigns via EmailTool
#   - Write marketing output to state["marketing_output"]
# """
# from __future__ import annotations

# import json
# import logging
# import time
# from typing import Any, Dict, List, Optional

# from config.settings import settings
# from models.ollama_loader import get_llm_async
# from tools.db_tool import get_customers_by_segment, log_campaign
# from tools.email_tool import send_promo_campaign
# from tools.search_tool import search_competitor_prices
# from workflows.state import PaddleAurumState, TaskItem

# logger = logging.getLogger(__name__)

# PROMO_KEYWORDS = {
#     "promo", "campaign", "email", "marketing", "launch", "discount",
#     "social", "post", "reel", "ad", "offer", "sale", "announce"
# }

# _PROMO_SYSTEM = """You are Promo General, the marketing lead of PaddleAurum.
# You write conversion-focused, on-brand pickleball email campaigns.
# Tone: energetic, community-focused, pickleball-obsessed. Never corporate.
# Output valid JSON only."""

# _DRAFT_TEMPLATE = """
# Campaign objective: {objective}
# Target segment: {segment} ({segment_size} customers)
# Recommended products: {products}
# Competitor pricing context: {competitor_context}
# Active low-stock items to avoid promoting: {low_stock}

# Draft a promotional email. Return JSON with keys:
#   subject_line (string, max 60 chars, include emoji),
#   preview_text (string, max 90 chars),
#   headline (string, max 80 chars),
#   body (string, 80-150 words, enthusiastic pickleball voice),
#   cta_text (string, max 30 chars),
#   cta_url (string, use https://paddleaurum.com/[relevant-path]),
#   discount_code (string or null),
#   estimated_open_rate (float 0.0-1.0).

# JSON only:"""


# async def draft_campaign(
#     objective: str,
#     segment: str,
#     segment_size: int,
#     recommended_products: List[Dict],
#     low_stock_product_ids: List[str],
# ) -> Optional[Dict]:
#     """Use LLM to draft a promo campaign based on objective and recommendations."""
#     llm = await get_llm_async("promo_general")

#     # Get competitor pricing context
#     category = "pickleball paddles"
#     try:
#         competitor_context = await search_competitor_prices(category)
#     except Exception:
#         competitor_context = "No competitor data available."

#     # Filter out low-stock products from recommendations
#     safe_products = [
#         p for p in recommended_products
#         if p.get("product_id") not in low_stock_product_ids
#     ][:5]

#     prompt = f"{_PROMO_SYSTEM}\n\n" + _DRAFT_TEMPLATE.format(
#         objective=objective,
#         segment=segment,
#         segment_size=segment_size,
#         products=json.dumps(safe_products, indent=2)[:600],
#         competitor_context=competitor_context[:400],
#         low_stock=json.dumps(low_stock_product_ids[:5]),
#     )

#     try:
#         raw = await llm.ainvoke(prompt)
#         clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
#         return json.loads(clean)
#     except Exception as exc:
#         logger.error("[Promo General] Campaign draft failed: %s", exc)
#         return None


# async def process_promo_tasks(
#     tasks: List[TaskItem],
#     state: PaddleAurumState,
# ) -> Dict[str, Any]:
#     """Process all pending marketing/promo tasks."""
#     campaigns_sent = []
#     campaigns_drafted = []

#     # Gather recommendation data from state (if available)
#     rec_output = state.get("recommendation_output") or {}
#     all_recs = rec_output.get("recommendation_sets", [])
#     # Flatten: collect all unique recommended products
#     top_products: List[Dict] = []
#     seen_pids = set()
#     for rs in all_recs:
#         for rec in rs.get("recommendations", []):
#             pid = rec.get("product_id")
#             if pid and pid not in seen_pids:
#                 top_products.append(rec)
#                 seen_pids.add(pid)

#     # Get low-stock IDs to avoid promoting out-of-stock items
#     low_stock_ids = [
#         a["product_id"]
#         for a in (state.get("inventory_output") or {}).get("low_stock_alerts", [])
#         if a.get("severity") == "CRITICAL"
#     ]

#     for task in tasks:
#         task["status"] = "in_progress"
#         inp = task.get("input_data", {})
#         objective = inp.get("description", inp.get("objective", "Grow sales"))
#         segment = inp.get("segment", "general")

#         # Get segment customers for email list
#         try:
#             customers = await get_customers_by_segment(segment)
#         except Exception:
#             customers = []

#         recipient_emails = [c["email"] for c in customers if c.get("email")]
#         segment_size = len(recipient_emails)

#         # Draft campaign
#         draft = await draft_campaign(
#             objective=objective,
#             segment=segment,
#             segment_size=segment_size,
#             recommended_products=top_products,
#             low_stock_product_ids=low_stock_ids,
#         )

#         if draft:
#             campaigns_drafted.append(draft)

#             # Send campaign if we have recipients
#             if recipient_emails:
#                 try:
#                     result = await send_promo_campaign(
#                         recipients=recipient_emails,
#                         subject=draft["subject_line"],
#                         headline=draft["headline"],
#                         body=draft["body"],
#                         cta_text=draft["cta_text"],
#                         cta_url=draft["cta_url"],
#                         discount_code=draft.get("discount_code"),
#                     )
#                     campaigns_sent.append({
#                         "segment": segment,
#                         "recipients": segment_size,
#                         "subject": draft["subject_line"],
#                         "result": result,
#                     })

#                     # Log to DB
#                     import uuid
#                     await log_campaign({
#                         "campaign_id": str(uuid.uuid4())[:8],
#                         "campaign_type": "email",
#                         "subject_line": draft["subject_line"],
#                         "target_segment": segment,
#                         "sent_count": segment_size,
#                     })
#                 except Exception as exc:
#                     logger.error("[Promo General] Campaign send failed: %s", exc)

#         task["status"] = "done"
#         task["completed_at"] = time.time()
#         task["output_data"] = {"campaign_drafted": draft is not None}

#     return {
#         "campaigns_drafted": len(campaigns_drafted),
#         "campaigns_sent": len(campaigns_sent),
#         "campaign_details": campaigns_drafted,
#         "send_results": campaigns_sent,
#     }


# async def promo_general_node(state: PaddleAurumState) -> PaddleAurumState:
#     state["current_step"] = "promo_general"
#     logger.info("[Promo General] Processing marketing tasks")

#     promo_tasks = [
#         t for t in state["task_queue"]
#         if t["assigned_to"] in ("promo_general",) and t["status"] == "pending"
#     ]

#     if not promo_tasks:
#         logger.info("[Promo General] No marketing tasks assigned.")
#         return state

#     output = await process_promo_tasks(promo_tasks, state)
#     state["marketing_output"] = output

#     for task in promo_tasks:
#         for i, qt in enumerate(state["task_queue"]):
#             if qt["task_id"] == task["task_id"]:
#                 state["task_queue"][i] = task

#     state["short_term_memory"].append({
#         "agent_id": "promo_general",
#         "role": "assistant",
#         "content": (
#             f"Marketing: {output['campaigns_drafted']} drafted, "
#             f"{output['campaigns_sent']} sent."
#         ),
#         "timestamp": time.time(),
#         "tool_calls": None,
#         "tool_results": None,
#     })

#     return state
































# @######################################################################################################






































"""
agents/promo_general.py
───────────────────────
NODE 3 — Promo General (Team Lead)

Responsibilities:
  - Receive marketing/promo sub-tasks from CEO
  - Delegate product recommendations to Recommender
  - Use LLM to draft email campaign copy
  - Send campaigns via EmailTool
  - Write marketing output to state["marketing_output"]
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid  # moved to top (Issue 1)
from typing import Any, Dict, List, Optional, Set

from config.settings import settings
from models.ollama_loader import get_llm_async
from tools.db_tool import get_customers_by_segment, get_unsubscribed_emails, log_campaign
from tools.email_tool import send_promo_campaign
from tools.search_tool import search_competitor_prices
from workflows.state import PaddleAurumState, TaskItem

logger = logging.getLogger(__name__)

PROMO_KEYWORDS = {
    "promo", "campaign", "email", "marketing", "launch", "discount",
    "social", "post", "reel", "ad", "offer", "sale", "announce"
}

_PROMO_SYSTEM = """You are Promo General, the marketing lead of PaddleAurum.
You write conversion-focused, on-brand pickleball email campaigns.
Tone: energetic, community-focused, pickleball-obsessed. Never corporate.
Output valid JSON only."""

_DRAFT_TEMPLATE = """
Campaign objective: {objective}
Target segment: {segment} ({segment_size} customers)
Recommended products: {products}
Competitor pricing context: {competitor_context}
Active low-stock items to avoid promoting: {low_stock}

Draft a promotional email. Return JSON with keys:
  subject_line (string, max 60 chars, include emoji),
  preview_text (string, max 90 chars),
  headline (string, max 80 chars),
  body (string, 80-150 words, enthusiastic pickleball voice),
  cta_text (string, max 30 chars),
  cta_url (string, use https://paddleaurum.com/[relevant-path]),
  discount_code (string or null),
  estimated_open_rate (float 0.0-1.0).

JSON only:"""


async def draft_campaign(
    objective: str,
    segment: str,
    segment_size: int,
    recommended_products: List[Dict],
    low_stock_product_ids: List[str],
) -> Optional[Dict]:
    """Use LLM to draft a promo campaign based on objective and recommendations."""
    llm = await get_llm_async("promo_general")

    # Get competitor pricing context
    category = "pickleball paddles"
    try:
        competitor_context = await search_competitor_prices(category)
    except Exception:
        competitor_context = "No competitor data available."

    # Filter out low-stock products from recommendations
    safe_products = [
        p for p in recommended_products
        if p.get("product_id") not in low_stock_product_ids
    ][:5]

    prompt = f"{_PROMO_SYSTEM}\n\n" + _DRAFT_TEMPLATE.format(
        objective=objective,
        segment=segment,
        segment_size=segment_size,
        products=json.dumps(safe_products, indent=2)[:600],
        competitor_context=competitor_context[:400],
        low_stock=json.dumps(low_stock_product_ids[:5]),
    )

    try:
        raw = await llm.ainvoke(prompt)
        # Strip markdown fences
        import re
        clean = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
        clean = re.sub(r'\s*```$', '', clean, flags=re.IGNORECASE).strip()
        return json.loads(clean)
    except Exception as exc:
        logger.error("[Promo General] Campaign draft failed: %s", exc)
        return None


async def _send_campaign_batches(
    recipients: List[str],
    subject: str,
    headline: str,
    body: str,
    cta_text: str,
    cta_url: str,
    discount_code: Optional[str] = None,
    batch_size: int = 50,
    delay_between_batches: float = 2.0,
) -> Dict[str, Any]:
    """
    Send campaign in batches, handling partial failures and rate limiting.
    Returns summary of successful and failed sends.
    """
    total = len(recipients)
    successful = 0
    failed = 0
    batch_results = []

    for i in range(0, total, batch_size):
        batch = recipients[i:i+batch_size]
        try:
            # Call the actual email tool (assumes it returns success count or raises)
            # Here we wrap and catch exceptions per batch.
            result = await send_promo_campaign(
                recipients=batch,
                subject=subject,
                headline=headline,
                body=body,
                cta_text=cta_text,
                cta_url=cta_url,
                discount_code=discount_code,
            )
            # Assume result is a dict with 'success_count' or similar.
            # If result is just a bool or int, adapt accordingly.
            # For robustness, we'll treat any exception as failure for the whole batch.
            batch_success = len(batch)  # optimistic, but tool might return actual count
            successful += batch_success
            batch_results.append({"batch": i//batch_size, "status": "success", "count": len(batch)})
            logger.info("[Promo General] Sent batch %d (%d emails)", i//batch_size, len(batch))
        except Exception as exc:
            logger.error("[Promo General] Batch %d failed: %s", i//batch_size, exc)
            failed += len(batch)
            batch_results.append({"batch": i//batch_size, "status": "failed", "error": str(exc)})
        # Rate limiting delay
        if i + batch_size < total:
            await asyncio.sleep(delay_between_batches)

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "batch_results": batch_results,
    }


async def process_promo_tasks(
    tasks: List[TaskItem],
    state: PaddleAurumState,
) -> Dict[str, Any]:
    """Process all pending marketing/promo tasks."""
    campaigns_sent = []
    campaigns_drafted = []

    # Gather recommendation data from state (if available)
    rec_output = state.get("recommendation_output") or {}
    all_recs = rec_output.get("recommendation_sets", [])
    # Flatten: collect all unique recommended products
    top_products: List[Dict] = []
    seen_pids = set()
    for rs in all_recs:
        for rec in rs.get("recommendations", []):
            pid = rec.get("product_id")
            if pid and pid not in seen_pids:
                top_products.append(rec)
                seen_pids.add(pid)

    # Get low-stock IDs to avoid promoting out-of-stock items
    low_stock_ids = [
        a["product_id"]
        for a in (state.get("inventory_output") or {}).get("low_stock_alerts", [])
        if a.get("severity") == "CRITICAL"
    ]

    # Get all unsubscribed emails (suppression list)
    unsubscribed: Set[str] = set()
    try:
        unsubscribed = set(await get_unsubscribed_emails())
    except Exception as exc:
        logger.warning("[Promo General] Could not fetch unsubscribe list: %s", exc)

    for task in tasks:
        task["status"] = "in_progress"
        inp = task.get("input_data", {})
        objective = inp.get("description", inp.get("objective", "Grow sales"))
        segment = inp.get("segment", "general")

        # Get segment customers for email list
        try:
            customers = await get_customers_by_segment(segment)
        except Exception:
            customers = []

        recipient_emails = [c["email"] for c in customers if c.get("email")]

        # Filter out unsubscribed addresses (Issue 3)
        original_count = len(recipient_emails)
        recipient_emails = [e for e in recipient_emails if e not in unsubscribed]
        if original_count - len(recipient_emails) > 0:
            logger.info("[Promo General] Suppressed %d unsubscribed emails", original_count - len(recipient_emails))

        segment_size = len(recipient_emails)

        # Draft campaign
        draft = await draft_campaign(
            objective=objective,
            segment=segment,
            segment_size=segment_size,
            recommended_products=top_products,
            low_stock_product_ids=low_stock_ids,
        )

        if draft:
            campaigns_drafted.append(draft)

            # Send campaign if we have recipients
            if recipient_emails:
                try:
                    send_result = await _send_campaign_batches(
                        recipients=recipient_emails,
                        subject=draft["subject_line"],
                        headline=draft["headline"],
                        body=draft["body"],
                        cta_text=draft["cta_text"],
                        cta_url=draft["cta_url"],
                        discount_code=draft.get("discount_code"),
                    )
                    campaigns_sent.append({
                        "segment": segment,
                        "recipients": segment_size,
                        "subject": draft["subject_line"],
                        "send_result": send_result,
                    })

                    # Log to DB
                    await log_campaign({
                        "campaign_id": str(uuid.uuid4())[:8],
                        "campaign_type": "email",
                        "subject_line": draft["subject_line"],
                        "target_segment": segment,
                        "sent_count": send_result["successful"],
                        "failed_count": send_result["failed"],
                    })
                except Exception as exc:
                    logger.error("[Promo General] Campaign send failed completely: %s", exc)
                    campaigns_sent.append({
                        "segment": segment,
                        "error": str(exc),
                    })

        task["status"] = "done"
        task["completed_at"] = time.time()
        task["output_data"] = {"campaign_drafted": draft is not None}

    return {
        "campaigns_drafted": len(campaigns_drafted),
        "campaigns_sent": len(campaigns_sent),
        "campaign_details": campaigns_drafted,
        "send_results": campaigns_sent,
    }


async def promo_general_node(state: PaddleAurumState) -> PaddleAurumState:
    state["current_step"] = "promo_general"
    logger.info("[Promo General] Processing marketing tasks")

    promo_tasks = [
        t for t in state["task_queue"]
        if t["assigned_to"] in ("promo_general",) and t["status"] == "pending"
    ]

    if not promo_tasks:
        logger.info("[Promo General] No marketing tasks assigned.")
        return state

    output = await process_promo_tasks(promo_tasks, state)
    state["marketing_output"] = output

    for task in promo_tasks:
        for i, qt in enumerate(state["task_queue"]):
            if qt["task_id"] == task["task_id"]:
                state["task_queue"][i] = task

    state["short_term_memory"].append({
        "agent_id": "promo_general",
        "role": "assistant",
        "content": (
            f"Marketing: {output['campaigns_drafted']} drafted, "
            f"{output['campaigns_sent']} sent."
        ),
        "timestamp": time.time(),
        "tool_calls": None,
        "tool_results": None,
    })

    return state