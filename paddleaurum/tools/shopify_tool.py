"""
tools/shopify_tool.py
─────────────────────
Async Shopify Admin REST API wrapper.
Handles auth, rate-limit backoff, and common e-commerce operations:
products, inventory, orders, customers, and fulfillments.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain.tools import BaseTool

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Rate limit constants ──────────────────────────────────────────────────────
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Core async HTTP client
# ─────────────────────────────────────────────────────────────────────────────

async def _shopify_get(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """
    Perform an authenticated GET against the Shopify Admin API.
    Handles 429 rate-limits with exponential backoff.
    """
    url = f"{settings.shopify_base_url}/{endpoint}"
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    url, headers=settings.shopify_headers, params=params or {}
                )
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", backoff))
                    logger.warning("Shopify 429 — waiting %.1fs (attempt %d)", wait, attempt + 1)
                    await asyncio.sleep(wait)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Shopify HTTP error %s on %s: %s", exc.response.status_code, url, exc)
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2
        except httpx.RequestError as exc:
            logger.error("Shopify request error on %s: %s", url, exc)
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Shopify GET {endpoint} failed after {MAX_RETRIES} attempts")


async def _shopify_post(endpoint: str, body: Dict) -> Dict:
    """Perform an authenticated POST (create/update resource)."""
    url = f"{settings.shopify_base_url}/{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=settings.shopify_headers, json=body)
        resp.raise_for_status()
        return resp.json()


async def _shopify_put(endpoint: str, body: Dict) -> Dict:
    """Perform an authenticated PUT (update resource)."""
    url = f"{settings.shopify_base_url}/{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.put(url, headers=settings.shopify_headers, json=body)
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Domain operations
# ─────────────────────────────────────────────────────────────────────────────

async def get_products(limit: int = 50, status: str = "active") -> List[Dict]:
    """Fetch paginated product list."""
    data = await _shopify_get("products.json", {"limit": limit, "status": status})
    return data.get("products", [])


async def get_product(product_id: str) -> Optional[Dict]:
    """Fetch a single product by ID."""
    data = await _shopify_get(f"products/{product_id}.json")
    return data.get("product")


async def get_inventory_levels(location_id: Optional[str] = None) -> List[Dict]:
    """Fetch inventory levels for all variants, optionally filtered by location."""
    params = {"limit": 250}
    if location_id:
        params["location_id"] = location_id
    data = await _shopify_get("inventory_levels.json", params)
    return data.get("inventory_levels", [])


async def get_inventory_for_product(product_id: str) -> List[Dict]:
    """Return inventory levels for all variants of a product."""
    product = await get_product(product_id)
    if not product:
        return []
    inventory_item_ids = [str(v["inventory_item_id"]) for v in product.get("variants", [])]
    if not inventory_item_ids:
        return []
    data = await _shopify_get(
        "inventory_levels.json",
        {"inventory_item_ids": ",".join(inventory_item_ids), "limit": 250},
    )
    return data.get("inventory_levels", [])


async def get_order(order_id: str) -> Optional[Dict]:
    """Fetch a single order by ID."""
    data = await _shopify_get(f"orders/{order_id}.json")
    return data.get("order")


async def get_recent_orders(limit: int = 50, status: str = "any") -> List[Dict]:
    """Fetch recent orders sorted newest first."""
    data = await _shopify_get(
        "orders.json", {"limit": limit, "status": status, "order": "created_at DESC"}
    )
    return data.get("orders", [])


async def get_customer(customer_id: str) -> Optional[Dict]:
    """Fetch a single Shopify customer."""
    data = await _shopify_get(f"customers/{customer_id}.json")
    return data.get("customer")


async def search_customers(query: str) -> List[Dict]:
    """Search customers by email or name."""
    data = await _shopify_get("customers/search.json", {"query": query, "limit": 10})
    return data.get("customers", [])


async def update_inventory_level(
    inventory_item_id: str, location_id: str, available: int
) -> Dict:
    """Set the available quantity for a variant at a location."""
    return await _shopify_post(
        "inventory_levels/set.json",
        {
            "location_id": int(location_id),
            "inventory_item_id": int(inventory_item_id),
            "available": available,
        },
    )


async def create_product_metafield(
    product_id: str, namespace: str, key: str, value: str
) -> Dict:
    """Add a metafield to a product (e.g., recommendation tags)."""
    return await _shopify_post(
        f"products/{product_id}/metafields.json",
        {"metafield": {"namespace": namespace, "key": key, "value": value, "type": "single_line_text_field"}},
    )


async def get_sales_by_product(product_id: str, days: int = 30) -> int:
    """
    Approximate 30-day units sold by scanning recent order line items.
    Note: For production, prefer Shopify Analytics API or a cached table.
    """
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    orders = await _shopify_get(
        "orders.json",
        {"created_at_min": cutoff, "status": "any", "limit": 250, "fields": "line_items"},
    )
    total = 0
    for order in orders.get("orders", []):
        for item in order.get("line_items", []):
            if str(item.get("product_id")) == str(product_id):
                total += item.get("quantity", 0)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# LangChain BaseTool wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ShopifyTool(BaseTool):
    name: str = "shopify_api"
    description: str = (
        "Interact with the PaddleAurum Shopify store. "
        "Actions: get_products, get_product, get_inventory_levels, "
        "get_inventory_for_product, get_order, get_recent_orders, "
        "get_customer, search_customers, get_sales_by_product. "
        "Input: JSON string with 'action' key and relevant parameters."
    )

    async def _arun(self, query: str) -> str:
        try:
            params = json.loads(query)
            action = params.get("action")

            dispatch = {
                "get_products":              lambda: get_products(params.get("limit", 50)),
                "get_product":               lambda: get_product(params["product_id"]),
                "get_inventory_levels":      lambda: get_inventory_levels(params.get("location_id")),
                "get_inventory_for_product": lambda: get_inventory_for_product(params["product_id"]),
                "get_order":                 lambda: get_order(params["order_id"]),
                "get_recent_orders":         lambda: get_recent_orders(params.get("limit", 50)),
                "get_customer":              lambda: get_customer(params["customer_id"]),
                "search_customers":          lambda: search_customers(params["query"]),
                "get_sales_by_product":      lambda: get_sales_by_product(
                    params["product_id"], params.get("days", 30)
                ),
            }

            if action not in dispatch:
                return json.dumps({"error": f"Unknown action: {action}"})

            result = await dispatch[action]()
            return json.dumps(result or {})
        except Exception as exc:
            logger.error("ShopifyTool error: %s", exc)
            return json.dumps({"error": str(exc)})

    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async _arun only")
