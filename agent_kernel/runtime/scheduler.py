"""In-process asyncio task scheduler.

The scheduler is the operational core that consumes typed ``TaskSpec``
records and runs them through the M2 ``NotebookRunner``. It owns:

- a bounded queue + per-policy concurrency-slot semaphore
- admission decisions delegated to ``policy_engine``
- budget reservation / debit / refund accounting
- status transitions (draft → queued → running → completed/failed/cancelled)
- provenance emission for every transition

All durable side-effects flow through the injected ``JSONLEventStore`` and
``AtomicJSONStore`` instances; the scheduler itself holds only in-memory
state that can be reconstructed by replaying the event log.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_kernel.models.budget import Budget, QuotaSnapshot
from agent_kernel.models.event import EventStatus, EventType, ProvenanceEvent
from agent_kernel.models.policy import DEFAULT_PROFILES, PolicyProfile
from agent_kernel.models.task import TaskSpec, TaskStatus
from agent_kernel.runtime import policy_engine
from agent_kernel.runtime.notebook_runner import (
    NotebookRunFailed,
    NotebookRunner,
    RunResult,
)
from agent_kernel.storage import AtomicJSONStore, JSONLEventStore, WorkspaceLayout
from agent_kernel.util import new_id, now_iso

# A run-fn is the unit of work executed inside the scheduler's concurrency
# slot. Default implementation is ``NotebookRunner.run``; tests inject fakes.
RunFn = Callable[[TaskSpec], Awaitable[RunResult | None]]


@dataclass
class _SchedulerState:
    tasks: dict[str, TaskSpec] = field(default_factory=dict)
    running: set[str] = field(default_factory=set)
    queued: list[str] = field(default_factory=list)
    cancellations: set[str] = field(default_factory=set)


class Scheduler:
    """Async task scheduler over a workspace."""

    def __init__(
        self,
        workspace: WorkspaceLayout,
        *,
        profile: PolicyProfile | None = None,
        run_fn: RunFn | None = None,
        fsync_events: bool = False,
    ) -> None:
        self.workspace = workspace
        self.workspace.ensure()
        self.profile = profile or DEFAULT_PROFILES["local-dev"]
        self.events = JSONLEventStore(workspace.events_dir, fsync=fsync_events)
        self.tasks_store = AtomicJSONStore(workspace.tasks_dir)
        self._state = _SchedulerState()
        self._lock = threading.Lock()
        self._slot_semaphore = asyncio.Semaphore(self.profile.kernel_slots_total)
        self._run_fn: RunFn = run_fn or self._default_run_fn

    # ------------------------------------------------------------ public API

    def create_task(
        self,
        *,
        notebook_path: str | Path,
        kernel_name: str = "python3",
        reserved_budget: Budget | None = None,
        parameters: dict[str, Any] | None = None,
        parent_task_id: str | None = None,
        depth: int = 0,
        tags: list[str] | None = None,
        task_id: str | None = None,
    ) -> TaskSpec:
        """Create a new task and persist its initial state + ``task.created`` event."""
        tid = task_id or new_id("task")
        now = now_iso()
        # Default reservation: full policy budget for top-level tasks; child
        # tasks must explicitly pass ``reserved_budget`` (M5 spawn manager).
        if reserved_budget is None:
            reserved_budget = self.profile.budgets.model_copy()
        task = TaskSpec(
            task_id=tid,
            notebook_path=str(notebook_path),
            kernel_name=kernel_name,
            policy_profile=self.profile.name,
            parent_task_id=parent_task_id,
            depth=depth,
            status=TaskStatus.draft,
            reserved_budget=reserved_budget,
            parameters=parameters or {},
            tags=list(tags or []),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._state.tasks[tid] = task
        self.tasks_store.write(tid, task)
        self._emit(
            EventType.task_created,
            task_id=tid,
            notebook_path=task.notebook_path,
            kernel_name=task.kernel_name,
            parent_task_id=task.parent_task_id,
            payload={"task_spec": task.model_dump(mode="json")},
        )
        return task

    def get_task(self, task_id: str) -> TaskSpec | None:
        with self._lock:
            t = self._state.tasks.get(task_id)
        return t or self.tasks_store.read(task_id, TaskSpec)

    def list_events(self, task_id: str | None = None) -> list[ProvenanceEvent]:
        events = self.events.list_events()
        if task_id is None:
            return events
        return [e for e in events if e.task_id == task_id]

    def cancel(self, task_id: str) -> None:
        """Request cancellation. Effective on next scheduler tick / cell boundary."""
        with self._lock:
            self._state.cancellations.add(task_id)

    # -------------------------------------------------------------- run task

    async def run_task(self, task_id: str) -> TaskSpec:
        """Admit + run a task to completion, returning its final ``TaskSpec``."""
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(task_id)

        # Admission
        with self._lock:
            self._state.queued.append(task_id)
            quota = self._quota_snapshot_locked()
        decision = policy_engine.can_admit(task, self.profile, quota)
        if not decision.allowed:
            self._emit(
                EventType.quota_blocked,
                task_id=task_id,
                payload={"reason": decision.reason},
                quota_snapshot=quota,
                status=EventStatus.denied,
            )
        # Even if quota is full, we'll wait on the semaphore (which provides
        # the actual back-pressure). The event above records the snapshot at
        # the moment of attempted admission.

        self._update_status(task_id, TaskStatus.queued)
        self._emit(EventType.task_admitted, task_id=task_id)

        # Check cancellation pre-run
        if self._is_cancelled(task_id):
            return await self._finalize_cancelled(task_id)

        # Wait for a kernel slot
        async with self._slot_semaphore:
            with self._lock:
                self._state.queued.remove(task_id)
                self._state.running.add(task_id)
            try:
                if self._is_cancelled(task_id):
                    return await self._finalize_cancelled(task_id)

                self._update_status(task_id, TaskStatus.running)
                task = self.get_task(task_id)
                assert task is not None
                result = await self._run_fn(task)
                # On success the run fn already emitted the cell events;
                # finalize budget accounting and unspent-refund.
                return self._finalize_completed(task_id, result)
            except NotebookRunFailed as exc:
                return self._finalize_failed(task_id, exc)
            except Exception as exc:  # pragma: no cover  (defensive)
                return self._finalize_failed(
                    task_id, NotebookRunFailed("?", f"unhandled:{type(exc).__name__}:{exc}")
                )
            finally:
                with self._lock:
                    self._state.running.discard(task_id)

    # ----------------------------------------------------------- run helpers

    async def _default_run_fn(self, task: TaskSpec) -> RunResult:
        """Run the task's notebook on a thread to avoid blocking the loop."""
        runner = NotebookRunner(self.events, self.workspace.runs_dir)

        def _run_sync() -> RunResult:
            return runner.run(
                task.notebook_path,
                task_id=task.task_id,
                kernel_name=task.kernel_name,
                timeout=max(1, task.reserved_budget.wall_ms // 1000) or 60,
            )

        return await asyncio.to_thread(_run_sync)

    # ----------------------------------------------------- terminal transitions

    def _finalize_completed(self, task_id: str, result: RunResult | None) -> TaskSpec:
        """Mark task complete and refund any unspent reservation."""
        task = self.get_task(task_id)
        assert task is not None
        # Refund the unspent reservation back (refund event records the delta).
        refundable = policy_engine.remaining_reservation(task)
        if any(getattr(refundable, f) > 0 for f in refundable.__class__.model_fields):
            transition = policy_engine.refund(task, refundable)
            task = task.model_copy(
                update={
                    "reserved_budget": transition.reserved_budget,
                    "spent_budget": transition.spent_budget,
                }
            )
            self._persist(task)
            self._emit(
                EventType.budget_refunded,
                task_id=task_id,
                budget_before=task.reserved_budget.add(refundable),
                budget_after=task.reserved_budget,
                payload={"delta": refundable.model_dump(mode="json"), "reason": "task_completed"},
            )
        executed_path = str(result.executed_notebook_path) if result else None
        self._update_status(
            task_id,
            TaskStatus.completed,
            executed_notebook_path=executed_path,
        )
        # Note: the NotebookRunner already emitted task.completed; we don't
        # double-emit. But we DO update the persisted snapshot.
        return self.get_task(task_id)  # type: ignore[return-value]

    def _finalize_failed(self, task_id: str, exc: NotebookRunFailed) -> TaskSpec:
        task = self.get_task(task_id)
        assert task is not None
        refundable = policy_engine.remaining_reservation(task)
        if any(getattr(refundable, f) > 0 for f in refundable.__class__.model_fields):
            transition = policy_engine.refund(task, refundable)
            task = task.model_copy(
                update={
                    "reserved_budget": transition.reserved_budget,
                    "spent_budget": transition.spent_budget,
                }
            )
            self._persist(task)
            self._emit(
                EventType.budget_refunded,
                task_id=task_id,
                payload={"delta": refundable.model_dump(mode="json"), "reason": "task_failed"},
            )
        self._update_status(
            task_id,
            TaskStatus.failed,
            error={"name": exc.reason, "cell_id": exc.cell_id, "run_id": exc.run_id},
        )
        # NotebookRunner already emitted task.failed for nbclient-level errors.
        return self.get_task(task_id)  # type: ignore[return-value]

    async def _finalize_cancelled(self, task_id: str) -> TaskSpec:
        task = self.get_task(task_id)
        assert task is not None
        refundable = policy_engine.remaining_reservation(task)
        transition = policy_engine.refund(task, refundable)
        task = task.model_copy(
            update={
                "reserved_budget": transition.reserved_budget,
                "spent_budget": transition.spent_budget,
            }
        )
        self._persist(task)
        self._emit(
            EventType.budget_refunded,
            task_id=task_id,
            payload={"delta": refundable.model_dump(mode="json"), "reason": "task_cancelled"},
        )
        self._update_status(task_id, TaskStatus.cancelled)
        self._emit(
            EventType.task_cancelled,
            task_id=task_id,
            status=EventStatus.cancelled,
        )
        # Clean up queue/cancellation set
        with self._lock:
            if task_id in self._state.queued:
                self._state.queued.remove(task_id)
            self._state.cancellations.discard(task_id)
        return self.get_task(task_id)  # type: ignore[return-value]

    # ------------------------------------------------------- internal helpers

    def _is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._state.cancellations

    def _quota_snapshot_locked(self) -> QuotaSnapshot:
        return QuotaSnapshot(
            kernel_slots_used=len(self._state.running),
            kernel_slots_total=self.profile.kernel_slots_total,
            running_tasks=len(self._state.running),
            queued_tasks=len(self._state.queued),
        )

    def _update_status(self, task_id: str, status: TaskStatus, **updates: Any) -> None:
        with self._lock:
            current = self._state.tasks.get(task_id)
        if current is None:
            current = self.tasks_store.read(task_id, TaskSpec)
        if current is None:
            return
        updated = current.model_copy(update={"status": status, "updated_at": now_iso(), **updates})
        with self._lock:
            self._state.tasks[task_id] = updated
        self._persist(updated)

    def _persist(self, task: TaskSpec) -> None:
        with self._lock:
            self._state.tasks[task.task_id] = task
        self.tasks_store.write(task.task_id, task)

    def _emit(self, event_type: EventType, **kwargs: Any) -> ProvenanceEvent:
        ev = ProvenanceEvent(
            event_id=new_id("evt"),
            ts=now_iso(),
            event_type=event_type,
            **kwargs,
        )
        self.events.append(ev)
        return ev
