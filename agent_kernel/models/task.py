"""Task models.

A task is a notebook descriptor plus policy state plus lineage. A spawned
child task always receives a reservation from its parent's remaining budget
and lineage metadata linking ``parent_task_id``, ``spawn_index``, and
``decision_event_id``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_kernel.models.budget import Budget


class TaskStatus(StrEnum):
    draft = "draft"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskSpec(BaseModel):
    """Durable task descriptor + current ledger state.

    This object is the row-level state record for a task. Status transitions
    and budget updates are produced by the scheduler from typed events; the
    state store persists snapshots of this model atomically.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    notebook_path: str
    kernel_name: str
    policy_profile: str = "local-dev"
    template_name: str | None = None
    parent_task_id: str | None = None
    spawn_index: int | None = None
    decision_event_id: str | None = None
    depth: int = Field(default=0, ge=0)
    status: TaskStatus = TaskStatus.draft
    reserved_budget: Budget = Field(default_factory=Budget)
    spent_budget: Budget = Field(default_factory=Budget)
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    child_task_ids: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    executed_notebook_path: str | None = None
    error: dict[str, Any] | None = None


class SpawnSpec(BaseModel):
    """Spawn request from a parent task."""

    model_config = ConfigDict(extra="forbid")

    template_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    kernel_name: str = "python3"
    reserved_budget: Budget | None = None
    tags: list[str] = Field(default_factory=list)
