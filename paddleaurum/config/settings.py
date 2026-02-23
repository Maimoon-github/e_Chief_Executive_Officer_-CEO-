"""
config/settings.py
──────────────────
Centralised configuration loaded from .env via Pydantic-Settings.
All agents and tools import `settings` from here — never raw os.environ.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Shopify ───────────────────────────────────────────────────────────────
    shopify_store_url: str = Field(default="https://paddleaurum.myshopify.com")
    shopify_api_key: str = Field(default="")
    shopify_api_secret: str = Field(default="")
    shopify_api_version: str = Field(default="2024-01")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3")
    ollama_fallback_model: str = Field(default="mistral")

    # ── Email ─────────────────────────────────────────────────────────────────
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_pass: str = Field(default="")
    smtp_from_name: str = Field(default="PaddleAurum Operations")

    # ── Notifications ─────────────────────────────────────────────────────────
    alert_email: str = Field(default="admin@paddleaurum.com")
    slack_webhook_url: str = Field(default="")

    # ── Database ──────────────────────────────────────────────────────────────
    sqlite_db_path: str = Field(default="paddleaurum/memory/agent_memory.db")

    # ── System ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    max_agent_retries: int = Field(default=3)
    agent_timeout_seconds: int = Field(default=60)
    short_term_memory_window: int = Field(default=20)
    cron_interval_minutes: int = Field(default=30)

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="paddleaurum")

    @property
    def shopify_base_url(self) -> str:
        return f"{self.shopify_store_url}/admin/api/{self.shopify_api_version}"

    @property
    def shopify_headers(self) -> dict:
        return {
            "X-Shopify-Access-Token": self.shopify_api_key,
            "Content-Type": "application/json",
        }


# Singleton — import this everywhere
settings = Settings()
