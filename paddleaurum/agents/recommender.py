# """
# agents/recommender.py
# ─────────────────────
# NODE 3a — Recommender (Worker)

# Responsibilities:
#   - Load customer purchase histories from SQLite
#   - Build product co-purchase vectors
#   - Compute cosine similarity to find affinity
#   - Return top-N cross-sell / upsell recommendations per customer
#   - Write recommendation_sets to state["recommendation_output"]
# """
# from __future__ import annotations

# import json
# import logging
# import time
# from collections import defaultdict
# from typing import Any, Dict, List, Tuple

# import numpy as np

# from tools.db_tool import get_customers_by_segment
# from tools.shopify_tool import get_products
# from workflows.state import PaddleAurumState, RecommendationSet, TaskItem

# logger = logging.getLogger(__name__)

# TOP_N = 3  # recommendations per customer


# # ── Cosine similarity helpers ─────────────────────────────────────────────────

# def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
#     """Compute cosine similarity between two sparse vectors."""
#     norm_a = np.linalg.norm(vec_a)
#     norm_b = np.linalg.norm(vec_b)
#     if norm_a == 0 or norm_b == 0:
#         return 0.0
#     return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# def _build_product_index(products: List[Dict]) -> Dict[str, int]:
#     """Map product_id → integer index for vectorisation."""
#     return {str(p["id"]): idx for idx, p in enumerate(products)}


# def _build_customer_vectors(
#     customers: List[Dict], product_index: Dict[str, int]
# ) -> Dict[str, np.ndarray]:
#     """
#     Build a binary purchase vector for each customer.
#     Vector length = number of distinct products.
#     """
#     n_products = len(product_index)
#     vectors: Dict[str, np.ndarray] = {}
#     for customer in customers:
#         cid = customer["customer_id"]
#         history = customer.get("purchase_history", [])
#         if isinstance(history, str):
#             history = json.loads(history)
#         vec = np.zeros(n_products, dtype=np.float32)
#         for item in history:
#             pid = str(item.get("product_id", item) if isinstance(item, dict) else item)
#             if pid in product_index:
#                 vec[product_index[pid]] = 1.0
#         vectors[cid] = vec
#     return vectors


# def _build_product_cooccurrence(
#     vectors: Dict[str, np.ndarray]
# ) -> np.ndarray:
#     """
#     Build a product co-purchase matrix.
#     cooccurrence[i][j] = number of customers who bought both product i and j.
#     """
#     if not vectors:
#         return np.array([])
#     n = len(next(iter(vectors.values())))
#     matrix = np.zeros((n, n), dtype=np.float32)
#     for vec in vectors.values():
#         bought = np.where(vec > 0)[0]
#         for i in bought:
#             for j in bought:
#                 if i != j:
#                     matrix[i][j] += 1.0
#     return matrix


# def _recommend_for_customer(
#     customer_vec: np.ndarray,
#     cooccurrence: np.ndarray,
#     product_index: Dict[str, int],
#     products: List[Dict],
#     top_n: int = TOP_N,
# ) -> List[Dict]:
#     """
#     For a customer, score all unowned products by co-purchase affinity.
#     Returns top_n recommendations with confidence score and reason_code.
#     """
#     reverse_index = {v: k for k, v in product_index.items()}
#     n = len(product_index)

#     owned = set(np.where(customer_vec > 0)[0])
#     scores = np.zeros(n)

#     for owned_idx in owned:
#         if owned_idx < len(cooccurrence):
#             scores += cooccurrence[owned_idx]

#     # Zero out owned products
#     for idx in owned:
#         scores[idx] = 0.0

#     if scores.sum() == 0:
#         # No co-purchase data — fall back to top products by index (popularity)
#         top_indices = [i for i in range(n) if i not in owned][:top_n]
#         return [
#             {
#                 "product_id": reverse_index.get(i, ""),
#                 "product_name": products[i].get("title", "") if i < len(products) else "",
#                 "confidence_score": 0.1,
#                 "reason_code": "trending",
#             }
#             for i in top_indices
#         ]

#     top_indices = np.argsort(scores)[::-1][:top_n]
#     max_score = scores.max() or 1.0
#     recs = []
#     for idx in top_indices:
#         if scores[idx] <= 0:
#             continue
#         pid = reverse_index.get(int(idx), "")
#         recs.append({
#             "product_id": pid,
#             "product_name": products[int(idx)].get("title", "") if int(idx) < len(products) else "",
#             "confidence_score": round(float(scores[idx]) / max_score, 3),
#             "reason_code": "cross_sell" if len(owned) > 0 else "upsell",
#         })
#     return recs


# # ── Main recommendation pipeline ─────────────────────────────────────────────

# async def generate_recommendations(
#     segment: str = "general",
#     max_customers: int = 100,
# ) -> List[RecommendationSet]:
#     """
#     Full recommendation pipeline for a customer segment.
#     """
#     logger.info("[Recommender] Generating recommendations for segment: %s", segment)

#     # Load data
#     customers = await get_customers_by_segment(segment)
#     if not customers:
#         logger.warning("[Recommender] No customers found for segment: %s", segment)
#         return []

#     customers = customers[:max_customers]

#     try:
#         products = await get_products(limit=250)
#     except Exception as exc:
#         logger.error("[Recommender] Could not load products: %s", exc)
#         return []

#     if not products:
#         return []

#     product_index = _build_product_index(products)
#     customer_vectors = _build_customer_vectors(customers, product_index)
#     cooccurrence = _build_product_cooccurrence(customer_vectors)

#     recommendation_sets: List[RecommendationSet] = []
#     for customer in customers:
#         cid = customer["customer_id"]
#         vec = customer_vectors.get(cid, np.zeros(len(product_index)))
#         recs = _recommend_for_customer(vec, cooccurrence, product_index, products)
#         if recs:
#             recommendation_sets.append(
#                 RecommendationSet(
#                     customer_id=cid,
#                     recommendations=recs,
#                 )
#             )

#     logger.info("[Recommender] Generated %d recommendation sets", len(recommendation_sets))
#     return recommendation_sets


# async def process_recommendation_tasks(tasks: List[TaskItem]) -> Dict[str, Any]:
#     """Run recommendation generation for all pending rec tasks."""
#     segments = set()
#     for task in tasks:
#         seg = task.get("input_data", {}).get("segment", "general")
#         segments.add(seg)

#     all_sets = []
#     for seg in segments or {"general"}:
#         sets = await generate_recommendations(segment=seg)
#         all_sets.extend([dict(s) for s in sets])

#     for task in tasks:
#         task["status"] = "done"
#         task["completed_at"] = time.time()
#         task["output_data"] = {"recommendation_count": len(all_sets)}

#     return {
#         "recommendation_sets": all_sets,
#         "total_customers_recommended": len(all_sets),
#         "segments_processed": list(segments or {"general"}),
#     }


# # ── LangGraph node ────────────────────────────────────────────────────────────

# async def recommender_node(state: PaddleAurumState) -> PaddleAurumState:
#     state["current_step"] = "recommender"
#     logger.info("[Recommender] Processing recommendation tasks")

#     rec_tasks = [
#         t for t in state["task_queue"]
#         if t["assigned_to"] in ("recommender",) and t["status"] == "pending"
#     ]

#     output = await process_recommendation_tasks(rec_tasks)
#     state["recommendation_output"] = output

#     for task in rec_tasks:
#         for i, qt in enumerate(state["task_queue"]):
#             if qt["task_id"] == task["task_id"]:
#                 state["task_queue"][i] = task

#     state["short_term_memory"].append({
#         "agent_id": "recommender",
#         "role": "assistant",
#         "content": (
#             f"Generated {output['total_customers_recommended']} recommendation sets "
#             f"across {output['segments_processed']} segments."
#         ),
#         "timestamp": time.time(),
#         "tool_calls": None,
#         "tool_results": None,
#     })

#     return state
































# @###############################################################################################




























"""
agents/recommender.py
─────────────────────
NODE 3a — Recommender (Worker)

Responsibilities:
  - Load customer purchase histories from SQLite
  - Build product co-purchase vectors
  - Compute cosine similarity to find affinity
  - Return top-N cross-sell / upsell recommendations per customer
  - Write recommendation_sets to state["recommendation_output"]
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from config.settings import settings
from tools.db_tool import get_customers_by_segment
from tools.shopify_tool import get_products
from workflows.state import PaddleAurumState, RecommendationSet, TaskItem

logger = logging.getLogger(__name__)

TOP_N = 3  # recommendations per customer


# ── Cosine similarity helpers ─────────────────────────────────────────────────

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two sparse vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _build_product_index(products: List[Dict]) -> Dict[str, int]:
    """Map product_id → integer index for vectorisation."""
    return {str(p["id"]): idx for idx, p in enumerate(products)}


def _build_customer_vectors(
    customers: List[Dict], product_index: Dict[str, int]
) -> Dict[str, np.ndarray]:
    """
    Build a purchase vector for each customer, weighted by quantity.
    Vector length = number of distinct products.
    """
    n_products = len(product_index)
    vectors: Dict[str, np.ndarray] = {}
    for customer in customers:
        cid = customer["customer_id"]
        history = customer.get("purchase_history", [])
        if isinstance(history, str):
            history = json.loads(history)
        vec = np.zeros(n_products, dtype=np.float32)
        for item in history:
            # item can be product_id string or dict with 'product_id' and optional 'quantity'
            if isinstance(item, dict):
                pid = str(item.get("product_id", ""))
                qty = float(item.get("quantity", 1))
            else:
                pid = str(item)
                qty = 1.0
            if pid in product_index:
                vec[product_index[pid]] += qty  # weighted by quantity (Issue 2)
        vectors[cid] = vec
    return vectors


def _build_product_cooccurrence(
    vectors: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Build a product co-purchase matrix using vectorised dot product.
    cooccurrence[i][j] = total quantity weight of customers who bought both i and j.
    """
    if not vectors:
        return np.array([])

    # Stack all customer vectors into a matrix of shape (m, n)
    customer_ids = list(vectors.keys())
    n = len(next(iter(vectors.values())))
    m = len(customer_ids)
    C = np.zeros((m, n), dtype=np.float32)
    for i, cid in enumerate(customer_ids):
        C[i, :] = vectors[cid]

    # Co-occurrence = C^T @ C   (faster than nested loops, Issue 1)
    # This gives a dense matrix where entry (i,j) is sum over customers of weight_i * weight_j.
    # Since weights are positive, this correctly accumulates weighted co-purchases.
    cooccurrence = C.T @ C
    return cooccurrence


def _recommend_for_customer(
    customer_vec: np.ndarray,
    cooccurrence: np.ndarray,
    product_index: Dict[str, int],
    products: List[Dict],
    top_n: int = TOP_N,
) -> List[Dict]:
    """
    For a customer, score all unowned products by co-purchase affinity.
    Returns top_n recommendations with confidence score and reason_code.
    """
    reverse_index = {v: k for k, v in product_index.items()}
    n = len(product_index)

    owned = set(np.where(customer_vec > 0)[0])
    scores = np.zeros(n)

    for owned_idx in owned:
        if owned_idx < len(cooccurrence):
            scores += cooccurrence[owned_idx]

    # Zero out owned products
    for idx in owned:
        scores[idx] = 0.0

    if scores.sum() == 0:
        # No co-purchase data — fall back to top products by index (popularity)
        top_indices = [i for i in range(n) if i not in owned][:top_n]
        return [
            {
                "product_id": reverse_index.get(i, ""),
                "product_name": products[i].get("title", "") if i < len(products) else "",
                "confidence_score": 0.1,
                "reason_code": "trending",
            }
            for i in top_indices
        ]

    top_indices = np.argsort(scores)[::-1][:top_n]
    max_score = scores.max() or 1.0
    recs = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        pid = reverse_index.get(int(idx), "")
        recs.append({
            "product_id": pid,
            "product_name": products[int(idx)].get("title", "") if int(idx) < len(products) else "",
            "confidence_score": round(float(scores[idx]) / max_score, 3),
            "reason_code": "cross_sell" if len(owned) > 0 else "upsell",
        })
    return recs


# ── Main recommendation pipeline ─────────────────────────────────────────────

async def generate_recommendations(
    segment: str = "general",
) -> List[RecommendationSet]:
    """
    Full recommendation pipeline for a customer segment.
    Uses configurable max_customers from settings (Issue 3).
    """
    logger.info("[Recommender] Generating recommendations for segment: %s", segment)

    # Load data
    customers = await get_customers_by_segment(segment)
    if not customers:
        logger.warning("[Recommender] No customers found for segment: %s", segment)
        return []

    # Apply configurable limit (Issue 3)
    max_cust = settings.max_customers_recommender
    customers = customers[:max_cust]

    try:
        products = await get_products(limit=250)
    except Exception as exc:
        logger.error("[Recommender] Could not load products: %s", exc)
        return []

    if not products:
        return []

    product_index = _build_product_index(products)
    customer_vectors = _build_customer_vectors(customers, product_index)
    cooccurrence = _build_product_cooccurrence(customer_vectors)

    recommendation_sets: List[RecommendationSet] = []
    for customer in customers:
        cid = customer["customer_id"]
        vec = customer_vectors.get(cid, np.zeros(len(product_index)))
        recs = _recommend_for_customer(vec, cooccurrence, product_index, products)
        if recs:
            recommendation_sets.append(
                RecommendationSet(
                    customer_id=cid,
                    recommendations=recs,
                )
            )

    logger.info("[Recommender] Generated %d recommendation sets", len(recommendation_sets))
    return recommendation_sets


async def process_recommendation_tasks(tasks: List[TaskItem]) -> Dict[str, Any]:
    """Run recommendation generation for all pending rec tasks."""
    segments = set()
    for task in tasks:
        seg = task.get("input_data", {}).get("segment", "general")
        segments.add(seg)

    all_sets = []
    for seg in segments or {"general"}:
        sets = await generate_recommendations(segment=seg)
        all_sets.extend([dict(s) for s in sets])

    for task in tasks:
        task["status"] = "done"
        task["completed_at"] = time.time()
        task["output_data"] = {"recommendation_count": len(all_sets)}

    return {
        "recommendation_sets": all_sets,
        "total_customers_recommended": len(all_sets),
        "segments_processed": list(segments or {"general"}),
    }


# ── LangGraph node ────────────────────────────────────────────────────────────

async def recommender_node(state: PaddleAurumState) -> PaddleAurumState:
    state["current_step"] = "recommender"
    logger.info("[Recommender] Processing recommendation tasks")

    rec_tasks = [
        t for t in state["task_queue"]
        if t["assigned_to"] in ("recommender",) and t["status"] == "pending"
    ]

    output = await process_recommendation_tasks(rec_tasks)
    state["recommendation_output"] = output

    for task in rec_tasks:
        for i, qt in enumerate(state["task_queue"]):
            if qt["task_id"] == task["task_id"]:
                state["task_queue"][i] = task

    state["short_term_memory"].append({
        "agent_id": "recommender",
        "role": "assistant",
        "content": (
            f"Generated {output['total_customers_recommended']} recommendation sets "
            f"across {output['segments_processed']} segments."
        ),
        "timestamp": time.time(),
        "tool_calls": None,
        "tool_results": None,
    })

    return state