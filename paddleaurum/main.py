"""
main.py
───────
PaddleAurum Agentic AI System — Entry Point

Usage:
  # Single run
  python main.py run --goal "Launch new carbon fiber paddle line"

  # CRON / autonomous mode (every 30 mins)
  python main.py cron

  # Dry run (graph validation, no execution)
  python main.py run --goal "Test" --dry-run

  # Direct support query
  python main.py support --message "Which paddle for beginners?" --customer-id C001

  # Inventory scan only
  python main.py inventory

  # Test LLM connection
  python main.py health
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("paddleaurum/logs/agent_runs.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Quieten noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("crewai").setLevel(logging.WARNING)

logger = logging.getLogger("paddleaurum.main")
console = Console()
app = typer.Typer(
    name="paddleaurum",
    help="🏓 PaddleAurum Autonomous Agent System",
    add_completion=False,
)


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    goal: str = typer.Option(
        "Run daily PaddleAurum operations: monitor inventory, handle support, run promos.",
        "--goal", "-g",
        help="Business goal to execute",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate graph without executing"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id", help="LangGraph thread ID"),
):
    """Execute the full multi-agent workflow for a given goal."""
    from workflows.runner import run_paddleaurum

    async def _run():
        return await run_paddleaurum(
            goal=goal,
            trigger_source="manual",
            thread_id=thread_id,
            dry_run=dry_run,
        )

    asyncio.run(_run())


@app.command()
def cron(
    goal_template: str = typer.Option(
        "Run daily PaddleAurum operations for {timestamp}: "
        "check inventory, handle support queue, run personalised promos.",
        "--goal-template",
        help="Goal template with optional {timestamp} placeholder",
    ),
    interval: int = typer.Option(30, "--interval", "-i", help="Run interval in minutes"),
):
    """Start autonomous CRON mode — runs the workflow on a recurring schedule."""
    from workflows.runner import schedule_recurring

    console.print(f"[bold green]Starting CRON mode — every {interval} minutes[/bold green]")
    scheduler = schedule_recurring(goal_template, interval_minutes=interval)

    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        scheduler.shutdown()
        console.print("[yellow]CRON scheduler stopped.[/yellow]")


@app.command()
def support(
    message: str = typer.Option(..., "--message", "-m", help="Customer message"),
    customer_id: str = typer.Option("", "--customer-id", help="Customer ID"),
    order_id: str = typer.Option("", "--order-id", help="Order ID"),
):
    """Run a single customer support query directly through Chat Buddy."""
    from agents.chat_buddy import handle_inquiry

    async def _run():
        result = await handle_inquiry(message, customer_id, order_id)
        _print_support_result(result)

    asyncio.run(_run())


@app.command()
def inventory():
    """Run a standalone inventory scan via Stock Scout."""
    from agents.stock_scout import scan_inventory

    async def _run():
        alerts = await scan_inventory()
        _print_inventory_alerts(alerts)

    asyncio.run(_run())


@app.command()
def health():
    """Check system health: Ollama connection, database, Shopify API."""
    async def _run():
        from models.ollama_loader import check_ollama_health, list_available_models
        from tools.shopify_tool import get_products

        console.print("\n[bold]PaddleAurum Health Check[/bold]\n")

        # Ollama
        ollama_ok = await check_ollama_health()
        models = await list_available_models() if ollama_ok else []
        _status("Ollama", ollama_ok, f"Models: {', '.join(models) or 'none pulled'}")

        # Database
        try:
            from memory.memory_manager import memory_manager
            await memory_manager.init_db()
            _status("SQLite DB", True, "Initialised")
        except Exception as exc:
            _status("SQLite DB", False, str(exc))

        # Shopify (only if API key set)
        from config.settings import settings
        if settings.shopify_api_key:
            try:
                products = await get_products(limit=1)
                _status("Shopify API", True, f"{len(products)} product(s) reachable")
            except Exception as exc:
                _status("Shopify API", False, str(exc))
        else:
            _status("Shopify API", None, "No API key configured (check .env)")

        console.print("")

    asyncio.run(_run())


@app.command()
def init_db():
    """Initialise the SQLite database schema."""
    async def _run():
        from memory.memory_manager import memory_manager
        await memory_manager.init_db()
        console.print("[bold green]✓ Database initialised[/bold green]")

    asyncio.run(_run())


# ── Pretty print helpers ──────────────────────────────────────────────────────

def _status(name: str, ok: Optional[bool], detail: str):
    if ok is True:
        icon, color = "✓", "green"
    elif ok is False:
        icon, color = "✗", "red"
    else:
        icon, color = "?", "yellow"
    console.print(f"  [{color}]{icon}[/{color}] [bold]{name}[/bold]: {detail}")


def _print_support_result(result: dict):
    from rich.panel import Panel
    sentiment = result.get("sentiment_score", 0.5)
    escalate = result.get("escalate", False)
    resolution = result.get("resolution_type", "unknown")
    reply = result.get("reply_text", "")

    color = "green" if not escalate else "yellow"
    console.print(Panel(
        f"[bold]Resolution:[/bold] {resolution}\n"
        f"[bold]Escalate:[/bold] {'⚠️ Yes' if escalate else '✓ No'}\n"
        f"[bold]Sentiment:[/bold] {sentiment:.2f}\n\n"
        f"[italic]{reply}[/italic]",
        title=f"[{color}]💬 Chat Buddy Response[/{color}]",
    ))


def _print_inventory_alerts(alerts: list):
    if not alerts:
        console.print("[green]✓ No inventory alerts — all stock levels healthy.[/green]")
        return

    table = Table(title="📦 Inventory Alerts")
    table.add_column("Product", style="bold")
    table.add_column("SKU")
    table.add_column("Qty", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Days Left", justify="right")
    table.add_column("Severity", justify="center")

    for alert in alerts:
        severity = alert.get("severity", "OK")
        color = {"CRITICAL": "red", "WARNING": "yellow", "OK": "green"}.get(severity, "white")
        days = alert.get("days_until_stockout", 999)
        days_str = f"{days:.0f}" if days < 999 else "∞"
        table.add_row(
            alert.get("product_name", "Unknown")[:30],
            alert.get("sku", ""),
            str(alert.get("current_qty", 0)),
            str(alert.get("threshold", 0)),
            days_str,
            f"[{color}]{severity}[/{color}]",
        )

    console.print(table)


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
