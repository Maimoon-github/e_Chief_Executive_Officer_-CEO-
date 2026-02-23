# """
# workflows/state.py
# ──────────────────
# Central state schema for the LangGraph workflow.
# All nodes read from and write to PaddleAurumState.
# The schema is the single source of truth for inter-agent data flow.
# """
# from __future__ import annotations

# import time
# import uuid
# from typing import Any, Dict, List, Optional
# from typing_extensions import TypedDict


# # ── Sub-schemas ───────────────────────────────────────────────────────────────

# class AgentMessage(TypedDict):
#     """A single message produced or consumed by an agent."""
#     agent_id: str
#     role: str            # system | user | assistant | tool
#     content: str
#     timestamp: float
#     tool_calls: Optional[List[Dict[str, Any]]]
#     tool_results: Optional[List[Dict[str, Any]]]


# class TaskItem(TypedDict):
#     """
#     A discrete unit of work produced by the CEO and assigned to an agent.
#     Lifecycle: pending → in_progress → done | failed | skipped
#     """
#     task_id: str
#     description: str
#     assigned_to: str           # agent key: chat_buddy, stock_scout, etc.
#     priority: int              # 1 (highest) – 5 (lowest)
#     status: str                # pending | in_progress | done | failed | skipped
#     input_data: Dict[str, Any]
#     output_data: Optional[Dict[str, Any]]
#     required_tools: List[str]
#     retries: int
#     max_retries: int
#     error_message: Optional[str]
#     created_at: float
#     completed_at: Optional[float]


# class LowStockAlert(TypedDict):
#     product_id: str
#     sku: str
#     product_name: str
#     current_qty: int
#     threshold: int
#     days_until_stockout: float
#     severity: str              # CRITICAL | WARNING | OK
#     reorder_qty: int
#     supplier_url: Optional[str]


# class RecommendationSet(TypedDict):
#     customer_id: str
#     recommendations: List[Dict[str, Any]]  # [{product_id, product_name, confidence_score, reason_code}]


# class CampaignDraft(TypedDict):
#     subject_line: str
#     preview_text: str
#     headline: str
#     body: str
#     cta_text: str
#     cta_url: str
#     discount_code: Optional[str]
#     target_segment: str
#     estimated_open_rate: float


# # ── Master State ──────────────────────────────────────────────────────────────

# class PaddleAurumState(TypedDict):
#     # ── Session identity ──────────────────────────────────────────────────────
#     session_id: str
#     trigger_source: str        # cron | webhook | manual
#     execution_mode: str        # async

#     # ── Goal & Planning ───────────────────────────────────────────────────────
#     goal: str
#     task_queue: List[TaskItem]
#     completed_tasks: List[TaskItem]
#     failed_tasks: List[TaskItem]
#     current_step: str
#     iteration_count: int       # re-plan counter, max 3 before hard stop

#     # ── Shared cross-agent context ────────────────────────────────────────────
#     shared_context: Dict[str, Any]
#     """
#     Expected keys populated during execution:
#     - store_name, active_products, current_promotions
#     - low_stock_alerts, open_tickets, customer_segments
#     """

#     # ── Branch outputs ────────────────────────────────────────────────────────
#     customer_support_output: Optional[Dict[str, Any]]
#     inventory_output: Optional[Dict[str, Any]]
#     marketing_output: Optional[Dict[str, Any]]
#     recommendation_output: Optional[Dict[str, Any]]
#     final_report: Optional[Dict[str, Any]]

#     # ── Memory ────────────────────────────────────────────────────────────────
#     short_term_memory: List[AgentMessage]
#     long_term_memory_keys: List[str]    # SQLite row IDs for recall queries

#     # ── Observability ─────────────────────────────────────────────────────────
#     error_log: List[Dict[str, Any]]


# # ── Factory ───────────────────────────────────────────────────────────────────

# def initial_state(goal: str, trigger_source: str = "manual") -> PaddleAurumState:
#     """Create a fresh PaddleAurumState for a new session."""
#     return PaddleAurumState(
#         session_id=str(uuid.uuid4()),
#         trigger_source=trigger_source,
#         execution_mode="async",
#         goal=goal,
#         task_queue=[],
#         completed_tasks=[],
#         failed_tasks=[],
#         current_step="start",
#         iteration_count=0,
#         shared_context={
#             "store_name": "PaddleAurum",
#         },
#         customer_support_output=None,
#         inventory_output=None,
#         marketing_output=None,
#         recommendation_output=None,
#         final_report=None,
#         short_term_memory=[],
#         long_term_memory_keys=[],
#         error_log=[],
#     )


# def make_task(
#     description: str,
#     assigned_to: str,
#     priority: int = 3,
#     input_data: Optional[Dict[str, Any]] = None,
#     required_tools: Optional[List[str]] = None,
#     max_retries: int = 3,
# ) -> TaskItem:
#     """Helper to build a TaskItem with sensible defaults."""
#     return TaskItem(
#         task_id=str(uuid.uuid4())[:8],
#         description=description,
#         assigned_to=assigned_to,
#         priority=priority,
#         status="pending",
#         input_data=input_data or {},
#         output_data=None,
#         required_tools=required_tools or [],
#         retries=0,
#         max_retries=max_retries,
#         error_message=None,
#         created_at=time.time(),
#         completed_at=None,
#     )
























# @###################################################################





















"""
workflows/state.py
──────────────────
Central state schema for the LangGraph workflow.
All nodes read from and write to PaddleAurumState.
The schema is the single source of truth for inter-agent data flow.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class AgentMessage(TypedDict):
    """A single message produced or consumed by an agent."""
    agent_id: str
    role: str            # system | user | assistant | tool
    content: str
    timestamp: float
    tool_calls: Optional[List[Dict[str, Any]]]
    tool_results: Optional[List[Dict[str, Any]]]


class TaskItem(TypedDict):
    """
    A discrete unit of work produced by the CEO and assigned to an agent.
    Lifecycle: pending → in_progress → done | failed | skipped
    """
    task_id: str
    description: str
    assigned_to: str           # agent key: chat_buddy, stock_scout, etc.
    priority: int              # 1 (highest) – 5 (lowest)
    status: str                # pending | in_progress | done | failed | skipped
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    required_tools: List[str]
    retries: int
    max_retries: int
    error_message: Optional[str]
    created_at: float
    completed_at: Optional[float]


class LowStockAlert(TypedDict):
    product_id: str
    sku: str
    product_name: str
    current_qty: int
    threshold: int
    days_until_stockout: float
    severity: str              # CRITICAL | WARNING | OK
    reorder_qty: int
    supplier_url: Optional[str]


class RecommendationSet(TypedDict):
    customer_id: str
    recommendations: List[Dict[str, Any]]  # [{product_id, product_name, confidence_score, reason_code}]


class CampaignDraft(TypedDict):
    subject_line: str
    preview_text: str
    headline: str
    body: str
    cta_text: str
    cta_url: str
    discount_code: Optional[str]
    target_segment: str
    estimated_open_rate: float


# ── Master State ──────────────────────────────────────────────────────────────

class PaddleAurumState(TypedDict):
    # ── Session identity ──────────────────────────────────────────────────────
    session_id: str
    trigger_source: str        # cron | webhook | manual
    execution_mode: str        # async

    # ── Goal & Planning ───────────────────────────────────────────────────────
    goal: str
    task_queue: List[TaskItem]
    completed_tasks: List[TaskItem]
    failed_tasks: List[TaskItem]
    current_step: str
    iteration_count: int       # re-plan counter, max 3 before hard stop

    # ── Shared cross-agent context ────────────────────────────────────────────
    shared_context: Dict[str, Any]
    """
    Expected keys populated during execution:
    - store_name, active_products, current_promotions
    - low_stock_alerts, open_tickets, customer_segments
    """

    # ── Branch outputs ────────────────────────────────────────────────────────
    customer_support_output: Optional[Dict[str, Any]]
    inventory_output: Optional[Dict[str, Any]]
    marketing_output: Optional[Dict[str, Any]]
    recommendation_output: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]

    # ── Memory ────────────────────────────────────────────────────────────────
    short_term_memory: List[AgentMessage]
    long_term_memory_keys: List[str]    # SQLite row IDs for recall queries

    # ── Observability ─────────────────────────────────────────────────────────
    error_log: List[Dict[str, Any]]


# ── Factory ───────────────────────────────────────────────────────────────────

def initial_state(goal: str, trigger_source: str = "manual") -> PaddleAurumState:
    """Create a fresh PaddleAurumState for a new session."""
    return PaddleAurumState(
        session_id=str(uuid.uuid4()),
        trigger_source=trigger_source,
        execution_mode="async",
        goal=goal,
        task_queue=[],
        completed_tasks=[],
        failed_tasks=[],
        current_step="start",
        iteration_count=0,
        shared_context={
            "store_name": "PaddleAurum",
        },
        customer_support_output=None,
        inventory_output=None,
        marketing_output=None,
        recommendation_output=None,
        final_report=None,
        short_term_memory=[],
        long_term_memory_keys=[],
        error_log=[],
    )


def make_task(
    description: str,
    assigned_to: str,
    priority: int = 3,
    input_data: Optional[Dict[str, Any]] = None,
    required_tools: Optional[List[str]] = None,
    max_retries: int = 3,
) -> TaskItem:
    """Helper to build a TaskItem with sensible defaults."""
    return TaskItem(
        task_id=str(uuid.uuid4())[:8],
        description=description,
        assigned_to=assigned_to,
        priority=priority,
        status="pending",
        input_data=input_data or {},
        output_data=None,
        required_tools=required_tools or [],
        retries=0,
        max_retries=max_retries,
        error_message=None,
        created_at=time.time(),
        completed_at=None,
    )
