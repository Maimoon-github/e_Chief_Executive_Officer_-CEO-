# """
# config/settings.py
# ──────────────────
# Centralised configuration loaded from .env via Pydantic-Settings.
# All agents and tools import `settings` from here — never raw os.environ.
# """
# from __future__ import annotations

# from pydantic import Field
# from pydantic_settings import BaseSettings, SettingsConfigDict


# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         case_sensitive=False,
#         extra="ignore",
#     )

#     # ── Shopify ───────────────────────────────────────────────────────────────
#     shopify_store_url: str = Field(default="https://paddleaurum.myshopify.com")
#     shopify_api_key: str = Field(default="")
#     shopify_api_secret: str = Field(default="")
#     shopify_api_version: str = Field(default="2024-01")

#     # ── Ollama ────────────────────────────────────────────────────────────────
#     ollama_base_url: str = Field(default="http://localhost:11434")
#     ollama_model: str = Field(default="llama3")
#     ollama_fallback_model: str = Field(default="mistral")

#     # ── Email ─────────────────────────────────────────────────────────────────
#     smtp_host: str = Field(default="smtp.gmail.com")
#     smtp_port: int = Field(default=587)
#     smtp_user: str = Field(default="")
#     smtp_pass: str = Field(default="")
#     smtp_from_name: str = Field(default="PaddleAurum Operations")

#     # ── Notifications ─────────────────────────────────────────────────────────
#     alert_email: str = Field(default="admin@paddleaurum.com")
#     slack_webhook_url: str = Field(default="")

#     # ── Database ──────────────────────────────────────────────────────────────
#     sqlite_db_path: str = Field(default="paddleaurum/memory/agent_memory.db")

#     # ── System ────────────────────────────────────────────────────────────────
#     log_level: str = Field(default="INFO")
#     max_agent_retries: int = Field(default=3)
#     agent_timeout_seconds: int = Field(default=60)
#     short_term_memory_window: int = Field(default=20)
#     cron_interval_minutes: int = Field(default=30)

#     # ── LangSmith ─────────────────────────────────────────────────────────────
#     langchain_tracing_v2: bool = Field(default=False)
#     langchain_api_key: str = Field(default="")
#     langchain_project: str = Field(default="paddleaurum")

#     @property
#     def shopify_base_url(self) -> str:
#         return f"{self.shopify_store_url}/admin/api/{self.shopify_api_version}"

#     @property
#     def shopify_headers(self) -> dict:
#         return {
#             "X-Shopify-Access-Token": self.shopify_api_key,
#             "Content-Type": "application/json",
#         }


# # Singleton — import this everywhere
# settings = Settings()
















# @########################################################################################
























"""
config/settings.py
──────────────────
Centralised configuration loaded from .env via Pydantic-Settings.
All agents and tools import `settings` from here — never raw os.environ.
"""
from __future__ import annotations

from pydantic import Field, model_validator
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
    shopify_api_key: str = Field(default="", min_length=0)  # empty allowed, but will warn at startup
    shopify_api_secret: str = Field(default="", min_length=0)
    shopify_api_version: str = Field(default="2024-01")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3", min_length=1)
    ollama_fallback_model: str = Field(default="mistral", min_length=1)

    # ── Email ─────────────────────────────────────────────────────────────────
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = Field(default="", min_length=0)
    smtp_pass: str = Field(default="", min_length=0)
    smtp_from_name: str = Field(default="PaddleAurum Operations")

    # ── Notifications ─────────────────────────────────────────────────────────
    alert_email: str = Field(default="admin@paddleaurum.com")
    slack_webhook_url: str = Field(default="")

    # ── Database ──────────────────────────────────────────────────────────────
    sqlite_db_path: str = Field(default="paddleaurum/memory/agent_memory.db")

    # ── System ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    max_agent_retries: int = Field(default=3, ge=1, le=10)
    agent_timeout_seconds: int = Field(default=60, ge=5, le=300)
    short_term_memory_window: int = Field(default=20, ge=1, le=100)
    cron_interval_minutes: int = Field(default=30, ge=1, le=1440)

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="", min_length=0)
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

    @model_validator(mode='after')
    def validate_required_configs(self) -> 'Settings':
        """
        Startup validation: warn about missing credentials that could affect certain features.
        This does not raise errors, only logs warnings.
        """
        import warnings

        if not self.shopify_api_key or not self.shopify_api_secret:
            warnings.warn(
                "Shopify API credentials are missing. Inventory sync and order operations will fail.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not self.smtp_user or not self.smtp_pass:
            warnings.warn(
                "SMTP credentials are missing. Email sending (support replies, campaigns, alerts) will fail.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not self.ollama_base_url or not self.ollama_model:
            warnings.warn(
                "Ollama configuration is incomplete. LLM calls will fail.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Optional: warn if alert_email is default but may be incorrect
        if self.alert_email == "admin@paddleaurum.com":
            warnings.warn(
                "Alert email is set to default value. Update it in .env to receive operational alerts.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self


# Singleton — import this everywhere
settings = Settings()