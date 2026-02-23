"""
workflows/runner.py
───────────────────
Async runner for the PaddleAurum agent system.
Entry point for programmatic invocation, CRON scheduling, and webhooks.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings
from memory.memory_manager import memory_manager
from workflows.graph import build_graph
from workflows.state import PaddleAurumState, initial_state

logger = logging.getLogger(__name__)
console = Console()


# ── Core run function ─────────────────────────────────────────────────────────

async def run_paddleaurum(
    goal: str,
    trigger_source: str = "manual",
    thread_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute a full PaddleAurum agentic workflow for a given goal.

    Args:
        goal: High-level business objective string.
        trigger_source: 'cron' | 'webhook' | 'manual'
        thread_id: LangGraph thread ID for checkpointing (auto-generated if None).
        dry_run: If True, build graph but don't execute (for testing).

    Returns:
        Final PaddleAurumState dict after workflow completes.
    """
    if thread_id is None:
        thread_id = f"session-{uuid.uuid4().hex[:8]}"

    # ── Init DB ────────────────────────────────────────────────────────────
    await memory_manager.init_db()

    # ── Build graph ────────────────────────────────────────────────────────
    graph = build_graph(use_checkpointer=not dry_run)

    if dry_run:
        console.print(Panel(
            f"[yellow]DRY RUN — graph built, not executing.[/yellow]\n"
            f"Goal: {goal}\nThread: {thread_id}",
            title="PaddleAurum Dry Run",
        ))
        return {}

    # ── Create initial state ───────────────────────────────────────────────
    state = initial_state(goal=goal, trigger_source=trigger_source)
    config = {"configurable": {"thread_id": thread_id}}

    console.print(Panel(
        f"[bold green]Starting PaddleAurum Workflow[/bold green]\n"
        f"Session: [cyan]{state['session_id']}[/cyan]\n"
        f"Goal: [white]{goal}[/white]\n"
        f"Source: [yellow]{trigger_source}[/yellow]",
        title="🏓 PaddleAurum Agent System",
    ))

    start_time = time.time()
    final_state: Dict[str, Any] = {}

    # ── Stream execution ───────────────────────────────────────────────────
    try:
        async for event in graph.astream(state, config=config):
            node_name = list(event.keys())[0] if event else "unknown"
            node_state = event.get(node_name, {})
            current_step = node_state.get("current_step", node_name)

            _print_step(current_step, node_state)
            final_state = node_state

        elapsed = time.time() - start_time
        _print_completion(final_state, elapsed)

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error("Workflow execution failed after %.1fs: %s", elapsed, exc)
        console.print(
            Panel(
                f"[red]Workflow failed: {exc}[/red]\nElapsed: {elapsed:.1f}s",
                title="❌ Error",
            )
        )
        raise

    return final_state


def _print_step(step_name: str, state: Dict) -> None:
    """Print a progress line for each completed node."""
    icons = {
        "ceo_orchestrator": "🧠",
        "customer_captain": "👥",
        "chat_buddy":       "💬",
        "stock_sergeant":   "📦",
        "stock_scout":      "🔍",
        "promo_general":    "📣",
        "recommender":      "⭐",
        "aggregator":       "🔀",
        "error_recovery":   "🔧",
        "memory_persist":   "💾",
        "done_notify":      "✅",
    }
    icon = icons.get(step_name, "→")
    task_count = len(state.get("task_queue", []))
    done_count = sum(1 for t in state.get("task_queue", []) if t.get("status") == "done")
    console.print(
        f"  {icon} [bold]{step_name}[/bold] "
        f"[dim]({done_count}/{task_count} tasks done)[/dim]"
    )


def _print_completion(state: Dict, elapsed: float) -> None:
    """Print final report summary."""
    report = state.get("final_report") or {}
    error_count = len(state.get("error_log", []))
    failed_count = len(state.get("failed_tasks", []))

    summary = report.get("summary", "Workflow completed.")
    wins = report.get("wins", [])
    next_actions = report.get("next_actions", [])

    console.print(Panel(
        f"[bold green]{summary}[/bold green]\n\n"
        + (f"[green]✓ Wins:[/green] {', '.join(wins[:3])}\n" if wins else "")
        + (f"[cyan]→ Next:[/cyan] {', '.join(next_actions[:3])}\n" if next_actions else "")
        + f"\n[dim]Elapsed: {elapsed:.1f}s | Errors: {error_count} | Failed tasks: {failed_count}[/dim]",
        title="🏓 PaddleAurum Workflow Complete",
    ))


# ── Streaming helper ──────────────────────────────────────────────────────────

async def stream_paddleaurum(
    goal: str,
    trigger_source: str = "manual",
    thread_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async generator version of run_paddleaurum.
    Yields each node's output dict as it completes.
    Useful for webhook streaming responses.
    """
    if thread_id is None:
        thread_id = f"session-{uuid.uuid4().hex[:8]}"

    await memory_manager.init_db()
    graph = build_graph()
    state = initial_state(goal=goal, trigger_source=trigger_source)
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream(state, config=config):
        yield event


# ── CRON scheduler ────────────────────────────────────────────────────────────

def schedule_recurring(goal_template: str, interval_minutes: int = None) -> None:
    """
    Set up APScheduler to run the workflow on a recurring schedule.
    Call this once from main.py to enable autonomous cron mode.
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    if interval_minutes is None:
        interval_minutes = settings.cron_interval_minutes

    scheduler = AsyncIOScheduler()

    async def _job():
        goal = goal_template.format(timestamp=time.strftime("%Y-%m-%d %H:%M"))
        await run_paddleaurum(goal, trigger_source="cron")

    scheduler.add_job(
        _job,
        "interval",
        minutes=interval_minutes,
        id="paddleaurum_main",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("CRON scheduler started: every %d minutes", interval_minutes)
    return scheduler
