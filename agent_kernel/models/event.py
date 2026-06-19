"""Provenance event model.

Events are append-only JSONL records. The schema captures enough context to
answer four questions later:
- what decision happened
- in which notebook/cell/kernel context
- against which budget/quota snapshot
- what downstream artifacts and rewards should be attributed to it
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_kernel.models.budget import Budget, QuotaSnapshot


class EventType(StrEnum):
    task_created = "task.created"
    task_admitted = "task.admitted"
    task_spawn_requested = "task.spawn.requested"
    task_spawned = "task.spawned"
    notebook_materialized = "notebook.materialized"
    notebook_execution_started = "notebook.execution.started"
    notebook_execution_completed = "notebook.execution.completed"
    cell_execution_started = "cell.execution.started"
    cell_execution_completed = "cell.execution.completed"
    cell_execution_error = "cell.execution.error"
    llm_call_started = "llm.call.started"
    llm_call_completed = "llm.call.completed"
    artifact_emitted = "artifact.emitted"
    budget_debited = "budget.debited"
    budget_refunded = "budget.refunded"
    quota_blocked = "quota.blocked"
    spawn_denied = "spawn.denied"
    task_completed = "task.completed"
    task_failed = "task.failed"
    task_cancelled = "task.cancelled"


class EventStatus(StrEnum):
    ok = "ok"
    error = "error"
    denied = "denied"
    cancelled = "cancelled"


class ProvenanceEvent(BaseModel):
    """One line of the append-only provenance JSONL."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    ts: str  # RFC 3339
    event_type: EventType
    task_id: str
    run_id: str | None = None
    parent_task_id: str | None = None
    notebook_path: str | None = None
    kernel_name: str | None = None
    session_id: str | None = None
    cell_id: str | None = None
    decision_id: str | None = None
    budget_before: Budget | None = None
    budget_after: Budget | None = None
    quota_snapshot: QuotaSnapshot | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)
    status: EventStatus = EventStatus.ok
    error: dict[str, Any] | None = None
