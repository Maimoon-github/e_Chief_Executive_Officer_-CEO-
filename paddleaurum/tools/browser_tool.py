"""
tools/browser_tool.py
─────────────────────
Async Playwright browser tool for scraping supplier websites.
Used by Stock Scout to verify stock availability and pricing directly
from supplier pages when no API is available.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


async def _get_page_text(url: str, timeout_ms: int = 15000) -> str:
    """
    Navigate to a URL, wait for DOM content, and return visible text.
    Playwright must be installed: `playwright install chromium`
    """
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            )
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            text = await page.inner_text("body")
            await browser.close()
            return text[:5000]  # cap at 5k chars for LLM context
    except Exception as exc:
        logger.error("Playwright error scraping %s: %s", url, exc)
        return f"ERROR: Could not scrape {url}. Reason: {exc}"


async def check_supplier_stock(supplier_url: str, product_name: str) -> Dict:
    """
    Scrape a supplier page and extract stock status + price hints.
    Returns a structured dict consumed by Stock Scout.
    """
    text = await _get_page_text(supplier_url)
    if text.startswith("ERROR"):
        return {
            "url": supplier_url,
            "product": product_name,
            "in_stock": None,
            "price_hint": None,
            "raw_text_snippet": text,
            "error": True,
        }

    # Heuristic stock detection
    lower = text.lower()
    in_stock: Optional[bool] = None
    if any(kw in lower for kw in ["in stock", "add to cart", "available", "qty"]):
        in_stock = True
    elif any(kw in lower for kw in ["out of stock", "unavailable", "sold out", "back-order"]):
        in_stock = False

    # Heuristic price extraction — find first $ amount near product name
    price_hint: Optional[str] = None
    price_matches = re.findall(r"\$[\d,]+\.?\d{0,2}", text)
    if price_matches:
        price_hint = price_matches[0]

    return {
        "url": supplier_url,
        "product": product_name,
        "in_stock": in_stock,
        "price_hint": price_hint,
        "raw_text_snippet": text[:500],
        "error": False,
    }


async def scrape_product_listing(url: str) -> Dict:
    """
    Generic product page scraper — extracts title, price, and description.
    Useful for competitive intelligence by Promo General.
    """
    text = await _get_page_text(url)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Grab first substantial line as title candidate
    title_candidate = lines[0] if lines else "Unknown"

    # Price patterns
    prices = re.findall(r"\$[\d,]+\.?\d{0,2}", text)

    return {
        "url": url,
        "title_candidate": title_candidate[:100],
        "prices_found": prices[:5],
        "text_snippet": " ".join(lines[:10])[:600],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangChain BaseTool wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BrowserTool(BaseTool):
    name: str = "browser_scraper"
    description: str = (
        "Scrape supplier websites and product pages using a headless browser. "
        "Actions: check_supplier_stock, scrape_product_listing. "
        "Input: JSON string with 'action', 'url', and optionally 'product_name' keys."
    )

    async def _arun(self, query: str) -> str:
        try:
            params = json.loads(query)
            action = params.get("action")

            if action == "check_supplier_stock":
                result = await check_supplier_stock(
                    params["url"], params.get("product_name", "")
                )
            elif action == "scrape_product_listing":
                result = await scrape_product_listing(params["url"])
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

            return json.dumps(result)
        except Exception as exc:
            logger.error("BrowserTool error: %s", exc)
            return json.dumps({"error": str(exc)})

    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async _arun only")
