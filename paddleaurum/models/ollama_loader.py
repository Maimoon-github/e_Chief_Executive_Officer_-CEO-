# """
# models/ollama_loader.py
# ───────────────────────
# Factory that creates and caches LangChain-compatible Ollama LLM instances.
# Handles model availability checks, fallback selection, and connection errors.
# """
# from __future__ import annotations

# import asyncio
# import logging
# from functools import lru_cache
# from typing import Optional

# import httpx
# from langchain_ollama import OllamaLLM

# from config.settings import settings
# from models.model_config import (
#     AGENT_MODEL_MAP,
#     AGENT_TEMP_MAP,
#     MODEL_FAST,
#     TIMEOUT_CEO,
#     TIMEOUT_LEAD,
#     TIMEOUT_WORKER,
# )

# logger = logging.getLogger(__name__)

# # ── Agent-name → timeout mapping ─────────────────────────────────────────────
# _TIMEOUT_MAP: dict[str, int] = {
#     "ceo":              TIMEOUT_CEO,
#     "customer_captain": TIMEOUT_LEAD,
#     "stock_sergeant":   TIMEOUT_LEAD,
#     "promo_general":    TIMEOUT_LEAD,
#     "chat_buddy":       TIMEOUT_WORKER,
#     "stock_scout":      TIMEOUT_WORKER,
#     "recommender":      TIMEOUT_WORKER,
# }


# async def check_ollama_health() -> bool:
#     """Ping Ollama server to confirm it is reachable."""
#     try:
#         async with httpx.AsyncClient(timeout=5.0) as client:
#             resp = await client.get(f"{settings.ollama_base_url}/api/tags")
#             return resp.status_code == 200
#     except Exception as exc:
#         logger.error("Ollama health check failed: %s", exc)
#         return False


# async def list_available_models() -> list[str]:
#     """Return list of models currently pulled in Ollama."""
#     try:
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             resp = await client.get(f"{settings.ollama_base_url}/api/tags")
#             resp.raise_for_status()
#             data = resp.json()
#             return [m["name"].split(":")[0] for m in data.get("models", [])]
#     except Exception as exc:
#         logger.warning("Could not list Ollama models: %s", exc)
#         return []


# async def resolve_model(preferred: str) -> str:
#     """
#     Return preferred model if available, else fallback.
#     Ensures we never crash agents due to a missing model.
#     """
#     available = await list_available_models()
#     if not available:
#         logger.warning("No models listed — using preferred %s regardless.", preferred)
#         return preferred
#     if preferred in available:
#         return preferred
#     fallback = settings.ollama_fallback_model
#     if fallback in available:
#         logger.warning("Model %s not found — falling back to %s.", preferred, fallback)
#         return fallback
#     # Last resort: first available model
#     logger.warning("Fallback %s also missing — using %s.", fallback, available[0])
#     return available[0]


# @lru_cache(maxsize=16)
# def _build_llm(model: str, temperature: float, timeout: int) -> OllamaLLM:
#     """
#     Build and cache an OllamaLLM instance.
#     lru_cache means identical (model, temp, timeout) combos share an instance.
#     """
#     return OllamaLLM(
#         model=model,
#         base_url=settings.ollama_base_url,
#         temperature=temperature,
#         timeout=timeout,
#     )


# def get_llm(
#     agent_name: str,
#     model_override: Optional[str] = None,
#     temperature_override: Optional[float] = None,
# ) -> OllamaLLM:
#     """
#     Public factory used by all agents.

#     Args:
#         agent_name: One of the agent keys defined in model_config.AGENT_MODEL_MAP.
#         model_override: Force a specific model (e.g. for testing).
#         temperature_override: Override default temperature.

#     Returns:
#         Configured OllamaLLM instance (cached).
#     """
#     model = model_override or AGENT_MODEL_MAP.get(agent_name, MODEL_FAST)
#     temperature = temperature_override if temperature_override is not None \
#         else AGENT_TEMP_MAP.get(agent_name, 0.3)
#     timeout = _TIMEOUT_MAP.get(agent_name, 60)

#     logger.debug("LLM requested for agent=%s model=%s temp=%.1f", agent_name, model, temperature)
#     return _build_llm(model, temperature, timeout)


# async def get_llm_async(
#     agent_name: str,
#     model_override: Optional[str] = None,
# ) -> OllamaLLM:
#     """
#     Async variant — resolves model availability before building.
#     Use this in node functions where model resolution is critical.
#     """
#     preferred = model_override or AGENT_MODEL_MAP.get(agent_name, MODEL_FAST)
#     resolved_model = await resolve_model(preferred)
#     temperature = AGENT_TEMP_MAP.get(agent_name, 0.3)
#     timeout = _TIMEOUT_MAP.get(agent_name, 60)
#     return _build_llm(resolved_model, temperature, timeout)























# W####@####################################################################


















"""
models/ollama_loader.py
───────────────────────
Factory that creates and caches LangChain-compatible Ollama LLM instances.
Handles model availability checks, fallback selection, and connection errors.
"""
from __future__ import annotations

import asyncio
import logging
import time
from fractions import Fraction
from functools import lru_cache
from typing import Optional, Tuple

import httpx
from langchain_ollama import OllamaLLM

from config.settings import settings
from models.model_config import (
    AGENT_MODEL_MAP,
    AGENT_TEMP_MAP,
    MODEL_FAST,
    TIMEOUT_CEO,
    TIMEOUT_LEAD,
    TIMEOUT_WORKER,
)

logger = logging.getLogger(__name__)

# ── Agent-name → timeout mapping ─────────────────────────────────────────────
_TIMEOUT_MAP: dict[str, int] = {
    "ceo":              TIMEOUT_CEO,
    "customer_captain": TIMEOUT_LEAD,
    "stock_sergeant":   TIMEOUT_LEAD,
    "promo_general":    TIMEOUT_LEAD,
    "chat_buddy":       TIMEOUT_WORKER,
    "stock_scout":      TIMEOUT_WORKER,
    "recommender":      TIMEOUT_WORKER,
}

# ── Cache for model availability ─────────────────────────────────────────────
_available_models_cache: Tuple[list[str], float] | None = None
_available_models_cache_ttl: int = 300  # seconds
_available_models_lock = asyncio.Lock()

# Cache for resolved model per preferred model
_resolved_model_cache: dict[str, Tuple[str, float]] = {}  # preferred -> (resolved, timestamp)
_resolved_model_cache_ttl: int = 300
_resolved_model_lock = asyncio.Lock()


async def check_ollama_health() -> bool:
    """Ping Ollama server to confirm it is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            return resp.status_code == 200
    except Exception as exc:
        logger.error("Ollama health check failed: %s", exc)
        return False


async def list_available_models() -> list[str]:
    """Return list of models currently pulled in Ollama, with caching."""
    global _available_models_cache
    async with _available_models_lock:
        now = time.time()
        if _available_models_cache is not None:
            cached_models, timestamp = _available_models_cache
            if now - timestamp < _available_models_cache_ttl:
                return cached_models

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{settings.ollama_base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                _available_models_cache = (models, now)
                return models
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []


async def resolve_model(preferred: str) -> str:
    """
    Return preferred model if available, else fallback.
    Caches the result for each preferred model for a TTL.
    """
    async with _resolved_model_lock:
        now = time.time()
        if preferred in _resolved_model_cache:
            resolved, timestamp = _resolved_model_cache[preferred]
            if now - timestamp < _resolved_model_cache_ttl:
                return resolved

        available = await list_available_models()
        if not available:
            logger.warning("No models listed — using preferred %s regardless.", preferred)
            resolved = preferred
        elif preferred in available:
            resolved = preferred
        else:
            fallback = settings.ollama_fallback_model
            if fallback in available:
                logger.warning("Model %s not found — falling back to %s.", preferred, fallback)
                resolved = fallback
            else:
                # Last resort: first available model
                logger.warning("Fallback %s also missing — using %s.", fallback, available[0])
                resolved = available[0]

        _resolved_model_cache[preferred] = (resolved, now)
        return resolved


def _make_cache_key(model: str, temperature: float, timeout: int) -> tuple:
    """
    Create a hashable cache key from parameters.
    Uses Fraction for exact representation of temperature.
    """
    # Convert temperature to Fraction to avoid float hash issues
    temp_frac = Fraction(temperature).limit_denominator(1000)
    return (model, temp_frac, timeout)


@lru_cache(maxsize=16)
def _build_llm(model: str, temperature_frac: Fraction, timeout: int) -> OllamaLLM:
    """
    Build and cache an OllamaLLM instance.
    The temperature is passed as a Fraction to ensure hash stability.
    """
    # Convert back to float for the actual model
    temperature = float(temperature_frac)
    return OllamaLLM(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        timeout=timeout,
    )


def get_llm(
    agent_name: str,
    model_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
) -> OllamaLLM:
    """
    Public factory used by all agents (synchronous version).
    Does NOT resolve model availability; use get_llm_async for production.

    Args:
        agent_name: One of the agent keys defined in model_config.AGENT_MODEL_MAP.
        model_override: Force a specific model (e.g. for testing).
        temperature_override: Override default temperature.

    Returns:
        Configured OllamaLLM instance (cached).
    """
    model = model_override or AGENT_MODEL_MAP.get(agent_name, MODEL_FAST)
    temperature = temperature_override if temperature_override is not None \
        else AGENT_TEMP_MAP.get(agent_name, 0.3)
    timeout = _TIMEOUT_MAP.get(agent_name, 60)

    logger.debug("LLM requested for agent=%s model=%s temp=%.1f", agent_name, model, temperature)
    key = _make_cache_key(model, temperature, timeout)
    return _build_llm(key[0], key[1], key[2])


async def get_llm_async(
    agent_name: str,
    model_override: Optional[str] = None,
) -> OllamaLLM:
    """
    Async variant — resolves model availability before building.
    Use this in node functions where model resolution is critical.
    """
    preferred = model_override or AGENT_MODEL_MAP.get(agent_name, MODEL_FAST)
    resolved_model = await resolve_model(preferred)
    temperature = AGENT_TEMP_MAP.get(agent_name, 0.3)
    timeout = _TIMEOUT_MAP.get(agent_name, 60)

    key = _make_cache_key(resolved_model, temperature, timeout)
    return _build_llm(key[0], key[1], key[2])