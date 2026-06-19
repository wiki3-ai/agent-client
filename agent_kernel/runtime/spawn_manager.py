"""Parent → child task spawn lifecycle.

The spawn manager is the glue between:
- the policy engine (``can_spawn`` decision, budget reservation arithmetic)
- the materializer (lineage-stamped child notebook on disk)
- the scheduler (child task admission + execution + refund)

A spawn:
1. Records a ``task.spawn.requested`` event on the parent.
2. Asks the policy engine whether the spawn is allowed under the active
   ``PolicyProfile``. If denied, emits ``spawn.denied`` and returns.
3. Reserves the requested budget from the parent (parent's reservation is
   reduced by the reserved amount; refund flows back to parent when child
   completes with unspent budget).
4. Materializes the child notebook with lineage metadata via the
   materializer.
5. Creates the child ``TaskSpec`` via the scheduler with the reserved
   budget and lineage.
6. Emits ``task.spawned``. The caller is responsible for running the child.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_kernel.models.budget import Budget
from agent_kernel.models.event import EventStatus, EventType
from agent_kernel.models.task import SpawnSpec, TaskSpec
from agent_kernel.runtime import policy_engine
from agent_kernel.runtime.materializer import materialize
from agent_kernel.runtime.scheduler import Scheduler
from agent_kernel.util import new_id, now_iso


@dataclass
class SpawnResult:
    """Outcome of a spawn request."""

    allowed: bool
    reason: str
    child_task: TaskSpec | None = None
    decision_event_id: str | None = None


class SpawnManager:
    """High-level spawn lifecycle over a scheduler."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def spawn(self, parent_task_id: str, spec: SpawnSpec) -> SpawnResult:
        parent = self.scheduler.get_task(parent_task_id)
        if parent is None:
            raise KeyError(parent_task_id)

        profile = self.scheduler.profile

        # Default reservation from profile if caller didn't specify.
        reserved = spec.reserved_budget or Budget(
            wall_ms=profile.reservation_policy.spawn_wall_ms_default,
            llm_usd_micro=profile.reservation_policy.spawn_llm_usd_micro_default,
        )

        # Pre-emptive decision event so the request itself is auditable.
        decision_id = new_id("dec")
        self.scheduler._emit(
            EventType.task_spawn_requested,
            task_id=parent_task_id,
            payload={
                "template_name": spec.template_name,
                "kernel_name": spec.kernel_name,
                "reserved_budget": reserved.model_dump(mode="json"),
                "decision_id": decision_id,
            },
            decision_id=decision_id,
        )

        decision = policy_engine.can_spawn(parent, profile, reserved)
        if not decision.allowed:
            self.scheduler._emit(
                EventType.spawn_denied,
                task_id=parent_task_id,
                status=EventStatus.denied,
                payload={"reason": decision.reason, "decision_id": decision_id},
                decision_id=decision_id,
            )
            return SpawnResult(allowed=False, reason=decision.reason, decision_event_id=decision_id)

        # --- Reserve from parent ---------------------------------------------
        # Parent loses reservation equal to ``reserved``; child gains it.
        # Failure-mode refund will flow back to parent when child terminates.
        new_parent_reserved = parent.reserved_budget.subtract(reserved)
        updated_parent = parent.model_copy(
            update={"reserved_budget": new_parent_reserved, "updated_at": now_iso()}
        )
        self.scheduler._persist(updated_parent)
        self.scheduler._emit(
            EventType.budget_debited,
            task_id=parent_task_id,
            budget_before=parent.reserved_budget,
            budget_after=new_parent_reserved,
            payload={
                "delta": reserved.model_dump(mode="json"),
                "reason": "spawn_reservation",
            },
        )

        # --- Materialize child notebook -------------------------------------
        spawn_index = len(parent.child_task_ids)
        depth = parent.depth + 1
        spawns_dir = self.scheduler.workspace.spawns_dir
        spawns_dir.mkdir(parents=True, exist_ok=True)
        child_id = new_id("task")
        child_path: Path = spawns_dir / f"child-{spawn_index:04d}-{child_id}.ipynb"

        materialize(
            spec.template_name,
            parameters=spec.parameters,
            kernel_name=spec.kernel_name,
            target_path=child_path,
            task_id=child_id,
            parent_task_id=parent_task_id,
            spawn_index=spawn_index,
            decision_event_id=decision_id,
            depth=depth,
            policy_profile=profile.name,
            events=self.scheduler.events,
        )

        # --- Register the child task with the scheduler ---------------------
        child = self.scheduler.create_task(
            notebook_path=str(child_path),
            kernel_name=spec.kernel_name,
            reserved_budget=reserved,
            parameters=spec.parameters,
            parent_task_id=parent_task_id,
            depth=depth,
            tags=list(spec.tags),
            task_id=child_id,
        )
        # Backfill the spawn lineage fields on the child TaskSpec.
        child = child.model_copy(
            update={"spawn_index": spawn_index, "decision_event_id": decision_id}
        )
        self.scheduler._persist(child)

        # --- Record on parent ------------------------------------------------
        updated_parent = updated_parent.model_copy(
            update={
                "child_task_ids": [*parent.child_task_ids, child_id],
                "updated_at": now_iso(),
            }
        )
        self.scheduler._persist(updated_parent)

        self.scheduler._emit(
            EventType.task_spawned,
            task_id=parent_task_id,
            payload={
                "child_task_id": child_id,
                "child_notebook_path": str(child_path),
                "spawn_index": spawn_index,
                "depth": depth,
                "decision_id": decision_id,
            },
            decision_id=decision_id,
        )
        return SpawnResult(
            allowed=True, reason="ok", child_task=child, decision_event_id=decision_id
        )
