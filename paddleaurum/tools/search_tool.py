# """
# tools/search_tool.py
# ────────────────────
# Async DuckDuckGo search tool. Free, no API key needed.
# Used by Chat Buddy (FAQ answers) and Promo General (competitor pricing).
# """
# from __future__ import annotations

# import asyncio
# import json
# import logging
# from typing import List, Optional

# from duckduckgo_search import DDGS
# from langchain.tools import BaseTool

# logger = logging.getLogger(__name__)


# async def ddg_search(query: str, max_results: int = 5) -> List[dict]:
#     """
#     Async wrapper around DuckDuckGo search.
#     Runs the synchronous DDGS call in a thread pool to avoid blocking.
#     """
#     def _sync_search() -> List[dict]:
#         results = []
#         with DDGS() as ddgs:
#             for r in ddgs.text(query, max_results=max_results):
#                 results.append({
#                     "title": r.get("title", ""),
#                     "url":   r.get("href", ""),
#                     "body":  r.get("body", ""),
#                 })
#         return results

#     loop = asyncio.get_event_loop()
#     try:
#         return await loop.run_in_executor(None, _sync_search)
#     except Exception as exc:
#         logger.error("DuckDuckGo search failed for query '%s': %s", query, exc)
#         return []


# async def search_pickleball_faq(question: str) -> str:
#     """
#     Search for an answer to a pickleball product question.
#     Returns a formatted snippet for Chat Buddy to use.
#     """
#     query = f"pickleball {question} site:reddit.com OR site:pickleballcentral.com OR site:usapickleball.org"
#     results = await ddg_search(query, max_results=3)
#     if not results:
#         return "No results found."
#     lines = []
#     for r in results:
#         lines.append(f"**{r['title']}**\n{r['body'][:300]}\n{r['url']}")
#     return "\n\n".join(lines)


# async def search_competitor_prices(category: str) -> str:
#     """
#     Search competitor pricing for a pickleball product category.
#     Used by Promo General and Stock Sergeant.
#     """
#     query = f"best {category} pickleball price 2024 site:amazon.com OR site:pickleballcentral.com"
#     results = await ddg_search(query, max_results=5)
#     if not results:
#         return "No competitor pricing found."
#     lines = []
#     for r in results:
#         lines.append(f"{r['title']}: {r['body'][:200]}")
#     return "\n".join(lines)


# async def search_supplier(supplier_name: str, product: str) -> str:
#     """Search for a supplier's current stock and pricing for a product."""
#     query = f"{supplier_name} {product} pickleball wholesale stock"
#     results = await ddg_search(query, max_results=3)
#     return json.dumps(results)


# # ─────────────────────────────────────────────────────────────────────────────
# # LangChain BaseTool wrapper
# # ─────────────────────────────────────────────────────────────────────────────

# class SearchTool(BaseTool):
#     name: str = "web_search"
#     description: str = (
#         "Search the web for pickleball product information, FAQs, "
#         "competitor pricing, and supplier availability. "
#         "Actions: faq_search, competitor_prices, supplier_search, general_search. "
#         "Input: JSON string with 'action' and 'query' keys."
#     )

#     async def _arun(self, query: str) -> str:
#         try:
#             params = json.loads(query)
#             action = params.get("action", "general_search")
#             search_query = params.get("query", "")

#             if action == "faq_search":
#                 return await search_pickleball_faq(search_query)
#             elif action == "competitor_prices":
#                 return await search_competitor_prices(search_query)
#             elif action == "supplier_search":
#                 return await search_supplier(
#                     params.get("supplier", ""), search_query
#                 )
#             else:
#                 results = await ddg_search(search_query, params.get("max_results", 5))
#                 return json.dumps(results)
#         except Exception as exc:
#             logger.error("SearchTool error: %s", exc)
#             return json.dumps({"error": str(exc)})

#     def _run(self, query: str) -> str:
#         raise NotImplementedError("Use async _arun only")



























# @#############################################################################################























"""
tools/search_tool.py
────────────────────
Async DuckDuckGo search tool. Free, no API key needed.
Used by Chat Buddy (FAQ answers) and Promo General (competitor pricing).
Includes optional result reranking using a cross-encoder for improved FAQ accuracy.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any

from duckduckgo_search import DDGS
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

# ── Optional cross-encoder for reranking ──────────────────────────────────────
_RERANKER = None
_RERANKER_LOCK = asyncio.Lock()

async def _get_reranker():
    """Lazy load cross-encoder model (sentence-transformers) if available."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    async with _RERANKER_LOCK:
        if _RERANKER is not None:
            return _RERANKER
        try:
            from sentence_transformers import CrossEncoder
            # Use a lightweight model suitable for CPU
            _RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder loaded for search reranking")
        except ImportError:
            logger.warning("sentence-transformers not installed; reranking disabled")
            _RERANKER = False  # sentinel
        except Exception as e:
            logger.error("Failed to load cross-encoder: %s", e)
            _RERANKER = False
        return _RERANKER


async def rerank_results(query: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Rerank search results using a cross-encoder if available.
    Returns top_k results sorted by relevance score (descending).
    If reranker not available, returns original results truncated to top_k.
    """
    reranker = await _get_reranker()
    if not reranker or not results:
        return results[:top_k]

    # Prepare pairs for the cross-encoder
    pairs = [(query, r.get("body", "") or r.get("title", "")) for r in results]
    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.error("Reranking failed: %s", e)
        return results[:top_k]

    # Combine scores with results and sort
    scored = list(zip(results, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:top_k]]


def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicates based on URL."""
    seen = set()
    unique = []
    for r in results:
        url = r.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)
    return unique


async def ddg_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Async wrapper around DuckDuckGo search.
    Runs the synchronous DDGS call in a thread pool to avoid blocking.
    """
    def _sync_search() -> List[dict]:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url":   r.get("href", ""),
                    "body":  r.get("body", ""),
                })
        return results

    # Use get_running_loop() as recommended for Python 3.10+ (Issue 1)
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(None, _sync_search)
        return deduplicate_results(results)
    except Exception as exc:
        logger.error("DuckDuckGo search failed for query '%s': %s", query, exc)
        return []


async def search_pickleball_faq(question: str) -> str:
    """
    Search for an answer to a pickleball product question.
    Returns a formatted snippet for Chat Buddy to use.
    Results are reranked for better relevance.
    """
    query = f"pickleball {question} site:reddit.com OR site:pickleballcentral.com OR site:usapickleball.org"
    results = await ddg_search(query, max_results=10)  # fetch more for reranking
    if not results:
        return "No results found."

    # Rerank and take top 3 (Issue 2 + Improvement)
    reranked = await rerank_results(question, results, top_k=3)

    lines = []
    for r in reranked:
        lines.append(f"**{r['title']}**\n{r['body'][:300]}\n{r['url']}")
    return "\n\n".join(lines)


async def search_competitor_prices(category: str) -> str:
    """
    Search competitor pricing for a pickleball product category.
    Used by Promo General and Stock Sergeant.
    """
    query = f"best {category} pickleball price 2024 site:amazon.com OR site:pickleballcentral.com"
    results = await ddg_search(query, max_results=5)
    if not results:
        return "No competitor pricing found."
    lines = []
    for r in results:
        lines.append(f"{r['title']}: {r['body'][:200]}")
    return "\n".join(lines)


async def search_supplier(supplier_name: str, product: str) -> str:
    """Search for a supplier's current stock and pricing for a product."""
    query = f"{supplier_name} {product} pickleball wholesale stock"
    results = await ddg_search(query, max_results=3)
    return json.dumps(results)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain BaseTool wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for pickleball product information, FAQs, "
        "competitor pricing, and supplier availability. "
        "Actions: faq_search, competitor_prices, supplier_search, general_search. "
        "Input: JSON string with 'action' and 'query' keys."
    )

    async def _arun(self, query: str) -> str:
        try:
            params = json.loads(query)
            action = params.get("action", "general_search")
            search_query = params.get("query", "")

            if action == "faq_search":
                return await search_pickleball_faq(search_query)
            elif action == "competitor_prices":
                return await search_competitor_prices(search_query)
            elif action == "supplier_search":
                return await search_supplier(
                    params.get("supplier", ""), search_query
                )
            else:
                results = await ddg_search(search_query, params.get("max_results", 5))
                return json.dumps(results)
        except Exception as exc:
            logger.error("SearchTool error: %s", exc)
            return json.dumps({"error": str(exc)})

    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async _arun only")