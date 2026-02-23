# """
# tools/browser_tool.py
# ─────────────────────
# Async Playwright browser tool for scraping supplier websites.
# Used by Stock Scout to verify stock availability and pricing directly
# from supplier pages when no API is available.
# """
# from __future__ import annotations

# import json
# import logging
# import re
# from typing import Dict, Optional

# from langchain.tools import BaseTool

# logger = logging.getLogger(__name__)


# async def _get_page_text(url: str, timeout_ms: int = 15000) -> str:
#     """
#     Navigate to a URL, wait for DOM content, and return visible text.
#     Playwright must be installed: `playwright install chromium`
#     """
#     try:
#         from playwright.async_api import async_playwright
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             page = await browser.new_page(
#                 user_agent=(
#                     "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
#                     "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
#                 )
#             )
#             await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
#             text = await page.inner_text("body")
#             await browser.close()
#             return text[:5000]  # cap at 5k chars for LLM context
#     except Exception as exc:
#         logger.error("Playwright error scraping %s: %s", url, exc)
#         return f"ERROR: Could not scrape {url}. Reason: {exc}"


# async def check_supplier_stock(supplier_url: str, product_name: str) -> Dict:
#     """
#     Scrape a supplier page and extract stock status + price hints.
#     Returns a structured dict consumed by Stock Scout.
#     """
#     text = await _get_page_text(supplier_url)
#     if text.startswith("ERROR"):
#         return {
#             "url": supplier_url,
#             "product": product_name,
#             "in_stock": None,
#             "price_hint": None,
#             "raw_text_snippet": text,
#             "error": True,
#         }

#     # Heuristic stock detection
#     lower = text.lower()
#     in_stock: Optional[bool] = None
#     if any(kw in lower for kw in ["in stock", "add to cart", "available", "qty"]):
#         in_stock = True
#     elif any(kw in lower for kw in ["out of stock", "unavailable", "sold out", "back-order"]):
#         in_stock = False

#     # Heuristic price extraction — find first $ amount near product name
#     price_hint: Optional[str] = None
#     price_matches = re.findall(r"\$[\d,]+\.?\d{0,2}", text)
#     if price_matches:
#         price_hint = price_matches[0]

#     return {
#         "url": supplier_url,
#         "product": product_name,
#         "in_stock": in_stock,
#         "price_hint": price_hint,
#         "raw_text_snippet": text[:500],
#         "error": False,
#     }


# async def scrape_product_listing(url: str) -> Dict:
#     """
#     Generic product page scraper — extracts title, price, and description.
#     Useful for competitive intelligence by Promo General.
#     """
#     text = await _get_page_text(url)
#     lines = [l.strip() for l in text.splitlines() if l.strip()]

#     # Grab first substantial line as title candidate
#     title_candidate = lines[0] if lines else "Unknown"

#     # Price patterns
#     prices = re.findall(r"\$[\d,]+\.?\d{0,2}", text)

#     return {
#         "url": url,
#         "title_candidate": title_candidate[:100],
#         "prices_found": prices[:5],
#         "text_snippet": " ".join(lines[:10])[:600],
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # LangChain BaseTool wrapper
# # ─────────────────────────────────────────────────────────────────────────────

# class BrowserTool(BaseTool):
#     name: str = "browser_scraper"
#     description: str = (
#         "Scrape supplier websites and product pages using a headless browser. "
#         "Actions: check_supplier_stock, scrape_product_listing. "
#         "Input: JSON string with 'action', 'url', and optionally 'product_name' keys."
#     )

#     async def _arun(self, query: str) -> str:
#         try:
#             params = json.loads(query)
#             action = params.get("action")

#             if action == "check_supplier_stock":
#                 result = await check_supplier_stock(
#                     params["url"], params.get("product_name", "")
#                 )
#             elif action == "scrape_product_listing":
#                 result = await scrape_product_listing(params["url"])
#             else:
#                 return json.dumps({"error": f"Unknown action: {action}"})

#             return json.dumps(result)
#         except Exception as exc:
#             logger.error("BrowserTool error: %s", exc)
#             return json.dumps({"error": str(exc)})

#     def _run(self, query: str) -> str:
#         raise NotImplementedError("Use async _arun only")






























# 2########################################################################################

























"""
tools/browser_tool.py
─────────────────────
Async Playwright browser tool for scraping supplier websites.
Uses a persistent browser instance to avoid launching a new browser per call.
Includes negation-aware stock detection and improved timeout handling.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Dict, Optional

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

# ── Persistent browser singleton ─────────────────────────────────────────────
_browser = None
_browser_lock = asyncio.Lock()


async def _get_browser():
    """Lazy initialisation of a persistent Playwright browser instance."""
    global _browser
    async with _browser_lock:
        if _browser is None:
            try:
                from playwright.async_api import async_playwright
                # Store both playwright and browser
                p = await async_playwright().start()
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--disable-dev-shm-usage']
                )
                # Keep playwright object to avoid garbage collection
                _browser = (p, browser)
                logger.info("Playwright browser launched (persistent)")
            except Exception as e:
                logger.error("Failed to launch Playwright: %s", e)
                raise
        return _browser[1]  # return browser instance


async def _close_browser():
    """Close the persistent browser (call during shutdown)."""
    global _browser
    async with _browser_lock:
        if _browser:
            p, browser = _browser
            await browser.close()
            await p.stop()
            _browser = None
            logger.info("Playwright browser closed")


# ── Enhanced stock detection (negation-aware) ───────────────────────────────

def _check_in_stock(text: str) -> Optional[bool]:
    """
    Determine if product is in stock based on page text.
    Returns True if likely in stock, False if likely out, None if uncertain.
    Negation‑aware: avoids false positives like 'not out of stock'.
    """
    lower = text.lower()

    # Define stock phrases with polarity and optional negation check
    in_stock_phrases = [
        "in stock",
        "add to cart",
        "available",
        "qty",
        "quantity",
        "buy now",
    ]
    out_of_stock_phrases = [
        "out of stock",
        "unavailable",
        "sold out",
        "back-order",
        "backorder",
        "pre-order",  # sometimes indicates not immediately available
    ]

    # Check for negation near out-of-stock phrases
    for phrase in out_of_stock_phrases:
        # Use regex with word boundaries to find phrase, and check for negation within 10 chars before
        pattern = re.compile(rf'\b{re.escape(phrase)}\b')
        for match in pattern.finditer(lower):
            start = max(0, match.start() - 15)
            preceding = lower[start:match.start()]
            if not re.search(r'\b(not|no|never|isn\'t|aren\'t|won\'t)\b', preceding):
                # No negation before the phrase -> out of stock
                return False
            # If negation present, this match is not a true out-of-stock indicator

    # If no out-of-stock phrase with positive polarity, check in-stock phrases
    for phrase in in_stock_phrases:
        if phrase in lower:
            return True

    # If still uncertain
    return None


async def _get_page_text(url: str, timeout_ms: int = 15000) -> str:
    """
    Navigate to a URL using a persistent browser, wait for DOM content,
    and return visible text. Timeout applied to the whole operation.
    """
    try:
        browser = await _get_browser()
        # Create a new page for each request (lightweight)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        try:
            # Apply timeout to the whole navigation + text extraction
            await asyncio.wait_for(
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded"),
                timeout=timeout_ms / 1000 + 5  # extra seconds for safety
            )
            text = await page.inner_text("body")
            return text[:5000]  # cap at 5k chars for LLM context
        finally:
            await page.close()  # always close page
    except asyncio.TimeoutError:
        logger.error("Timeout scraping %s", url)
        return f"ERROR: Timeout scraping {url}."
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

    in_stock = _check_in_stock(text)

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