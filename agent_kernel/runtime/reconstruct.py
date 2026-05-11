"""Pure state reconstruction from a stream of provenance events.

This is the inverse of ``Scheduler.advance``: take an event stream and
produce the current ``TaskSpec`` snapshot for every task that appears.

Used by:
- the M1 integration gate (crash/replay reconstruction)
- the M9 replay script
- the scheduler at startup to rebuild in-memory state
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable

from agent_kernel.models.event import EventType, ProvenanceEvent
from agent_kernel.models.task import TaskSpec, TaskStatus


def reconstruct_tasks(events: Iterable[ProvenanceEvent]) -> dict[str, TaskSpec]:
    """Replay events into a ``{task_id: TaskSpec}`` snapshot map.

    The reconstruction is a deterministic fold over the event stream. Events
    are applied in order; unknown event types are ignored (forward compat).
    """
    tasks: dict[str, TaskSpec] = {}
    for ev in events:
        tid = ev.task_id
        spec = tasks.get(tid)

        if ev.event_type == EventType.task_created:
            payload = ev.payload or {}
            ts = TaskSpec.model_validate(payload["task_spec"])
            tasks[tid] = ts
            continue

        if spec is None:
            # Skip events for tasks we haven't seen the creation of.
            continue

        updates: dict[str, object] = {"updated_at": ev.ts}

        if ev.event_type == EventType.task_admitted:
            updates["status"] = TaskStatus.queued
        elif ev.event_type == EventType.notebook_execution_started:
            updates["status"] = TaskStatus.running
        elif ev.event_type == EventType.task_completed:
            updates["status"] = TaskStatus.completed
            if "executed_notebook_path" in ev.payload:
                updates["executed_notebook_path"] = ev.payload["executed_notebook_path"]
        elif ev.event_type == EventType.task_failed:
            updates["status"] = TaskStatus.failed
            if ev.error is not None:
                updates["error"] = ev.error
        elif ev.event_type == EventType.task_cancelled:
            updates["status"] = TaskStatus.cancelled
        elif ev.event_type == EventType.task_spawned:
            child_id = ev.payload.get("child_task_id")
            if child_id and child_id not in spec.child_task_ids:
                updates["child_task_ids"] = [*spec.child_task_ids, child_id]
        elif ev.event_type == EventType.budget_debited:
            delta = ev.payload.get("delta") if ev.payload else None
            if ev.budget_after is not None and delta is not None:
                from agent_kernel.models.budget import Budget

                d = Budget.model_validate(delta)
                updates["spent_budget"] = spec.spent_budget.add(d)
        elif ev.event_type == EventType.budget_refunded and ev.payload and "delta" in ev.payload:
            from agent_kernel.models.budget import Budget

            d = Budget.model_validate(ev.payload["delta"])
            # Refund reduces reserved_budget.
            with contextlib.suppress(ValueError):
                updates["reserved_budget"] = spec.reserved_budget.subtract(d)

        if updates:
            tasks[tid] = spec.model_copy(update=updates)

    return tasks
