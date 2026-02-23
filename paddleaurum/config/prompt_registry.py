"""
config/prompt_registry.py
─────────────────────────
PromptRegistry class: loads agent system prompts from agents.yaml
and task templates from tasks.yaml. Provides methods to retrieve
system prompts and render task descriptions with runtime placeholders.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Singleton registry for all prompt templates."""

    _instance: Optional[PromptRegistry] = None
    _agents: Dict[str, Any] = {}
    _tasks: Dict[str, Any] = {}

    def __new__(cls) -> PromptRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load YAML files on first instantiation."""
        base = Path(__file__).parent
        agents_path = base / "agents.yaml"
        tasks_path = base / "tasks.yaml"

        try:
            with open(agents_path, "r", encoding="utf-8") as f:
                self._agents = yaml.safe_load(f) or {}
            logger.info("Loaded %d agent prompts from %s", len(self._agents), agents_path)
        except Exception as e:
            logger.error("Failed to load agents.yaml: %s", e)
            self._agents = {}

        try:
            with open(tasks_path, "r", encoding="utf-8") as f:
                self._tasks = yaml.safe_load(f) or {}
            logger.info("Loaded %d task templates from %s", len(self._tasks), tasks_path)
        except Exception as e:
            logger.error("Failed to load tasks.yaml: %s", e)
            self._tasks = {}

    def get_system_prompt(self, agent_name: str) -> str:
        """
        Return the system prompt for the given agent.
        Falls back to a generic prompt if agent or system_prompt key missing.
        """
        agent = self._agents.get(agent_name)
        if agent and "system_prompt" in agent:
            return agent["system_prompt"].strip()
        logger.warning("No system_prompt found for agent '%s', using generic fallback", agent_name)
        return f"You are the {agent_name} agent for PaddleAurum. Be helpful and concise."

    def render_task(self, task_name: str, **kwargs: Any) -> str:
        """
        Retrieve the task description template and substitute placeholders.
        Raises KeyError if task_name not found.
        """
        task = self._tasks.get(task_name)
        if not task:
            raise KeyError(f"Task template '{task_name}' not found in tasks.yaml")
        template = task.get("description", "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error("Missing placeholder %s for task '%s'", e, task_name)
            raise
        except Exception as e:
            logger.error("Failed to render task '%s': %s", task_name, e)
            raise

    def get_task_expected_output(self, task_name: str) -> str:
        """Return the expected_output schema for a task (optional)."""
        task = self._tasks.get(task_name)
        return task.get("expected_output", "") if task else ""


# Global singleton instance
prompt_registry = PromptRegistry()