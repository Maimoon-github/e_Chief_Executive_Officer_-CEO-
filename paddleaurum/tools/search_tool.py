"""
tools/search_tool.py
────────────────────
Async DuckDuckGo search tool. Free, no API key needed.
Used by Chat Buddy (FAQ answers) and Promo General (competitor pricing).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from duckduckgo_search import DDGS
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


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

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _sync_search)
    except Exception as exc:
        logger.error("DuckDuckGo search failed for query '%s': %s", query, exc)
        return []


async def search_pickleball_faq(question: str) -> str:
    """
    Search for an answer to a pickleball product question.
    Returns a formatted snippet for Chat Buddy to use.
    """
    query = f"pickleball {question} site:reddit.com OR site:pickleballcentral.com OR site:usapickleball.org"
    results = await ddg_search(query, max_results=3)
    if not results:
        return "No results found."
    lines = []
    for r in results:
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
