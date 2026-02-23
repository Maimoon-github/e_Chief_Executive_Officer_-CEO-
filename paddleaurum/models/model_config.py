"""
models/model_config.py
──────────────────────
Model name constants and capability tiers for Ollama-backed LLMs.
All agent files import MODEL_* constants from here — never hardcode
model names in agent logic.
"""
from __future__ import annotations

# ── Primary Models ────────────────────────────────────────────────────────────
# High-capability: used by CEO and Team Leads for complex reasoning
MODEL_PRIMARY = "llama3"          # Llama 3 8B — good balance
MODEL_PRIMARY_LARGE = "llama3:70b"  # For production servers with >40 GB RAM

# Fast / lightweight: used by workers for structured extraction tasks
MODEL_FAST = "mistral"            # Mistral 7B — fast, lean
MODEL_FAST_ALT = "gemma:7b"       # Fallback if Mistral unavailable

# Code-specialised (optional, for tool-writing tasks)
MODEL_CODE = "codellama"

# ── Timeout Configuration (seconds) ──────────────────────────────────────────
TIMEOUT_CEO = 90        # CEO reasoning is most complex
TIMEOUT_LEAD = 60       # Team leads moderate complexity
TIMEOUT_WORKER = 45     # Workers structured extraction, faster
TIMEOUT_FALLBACK = 30   # Fallback model is lighter, quicker

# ── Temperature Settings ─────────────────────────────────────────────────────
TEMP_REASONING = 0.3    # Low temp for deterministic task decomposition
TEMP_CREATIVE = 0.7     # Higher for email copy and product descriptions
TEMP_EXTRACTION = 0.1   # Near-zero for structured JSON extraction

# ── Context Window Limits ─────────────────────────────────────────────────────
MAX_TOKENS_CEO = 4096
MAX_TOKENS_WORKER = 2048

# ── Role → Model Mapping ──────────────────────────────────────────────────────
AGENT_MODEL_MAP: dict[str, str] = {
    "ceo":              MODEL_PRIMARY,
    "customer_captain": MODEL_PRIMARY,
    "stock_sergeant":   MODEL_PRIMARY,
    "promo_general":    MODEL_PRIMARY,
    "chat_buddy":       MODEL_FAST,
    "stock_scout":      MODEL_FAST,
    "recommender":      MODEL_FAST,
}

AGENT_TEMP_MAP: dict[str, float] = {
    "ceo":              TEMP_REASONING,
    "customer_captain": TEMP_CREATIVE,
    "stock_sergeant":   TEMP_EXTRACTION,
    "promo_general":    TEMP_CREATIVE,
    "chat_buddy":       TEMP_CREATIVE,
    "stock_scout":      TEMP_EXTRACTION,
    "recommender":      TEMP_EXTRACTION,
}
