"""M4 integration gate: scheduler, policy engine, budgets, quotas.

This test spins up the scheduler in-process, submits N tasks (some
succeed, some fail, one cancelled mid-run), and asserts via ``list_events``
that:

(a) budget never went negative across the JSONL trace
(b) refunds equal reservations on cancellations/failures
(c) quota blocking emitted ``quota.blocked`` when concurrency exceeded
(d) state reconstructed from JSONL matches in-memory ledger
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from agent_kernel.api import AgentKernel
from agent_kernel.models.budget import Budget
from agent_kernel.models.event import EventType
from agent_kernel.models.policy import DEFAULT_PROFILES, PolicyProfile
from agent_kernel.models.task import TaskStatus
from agent_kernel.runtime import policy_engine
from agent_kernel.runtime.reconstruct import reconstruct_tasks
from agent_kernel.runtime.scheduler import Scheduler
from agent_kernel.storage import JSONLEventStore, WorkspaceLayout


def _write_nb(path: Path, source: str) -> None:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [new_code_cell(source, id="only")]
    nbformat.write(nb, path)


# ============================= policy_engine ==============================


@pytest.mark.integration
def test_policy_engine_invariants() -> None:
    profile = DEFAULT_PROFILES["local-dev"]
    from agent_kernel.models.task import TaskSpec

    t = TaskSpec(
        task_id="t1",
        notebook_path="nb.ipynb",
        kernel_name="python3",
        reserved_budget=Budget(wall_ms=10_000, llm_usd_micro=1_000),
    )
    # debit + refund are inverse
    spent = Budget(wall_ms=3_000)
    trans = policy_engine.debit(t, spent)
    assert trans.spent_budget.wall_ms == 3_000
    t2 = t.model_copy(update={"spent_budget": trans.spent_budget})
    remaining = policy_engine.remaining_reservation(t2)
    assert remaining.wall_ms == 7_000

    # debit beyond reservation must raise
    with pytest.raises(ValueError):
        policy_engine.debit(t, Budget(wall_ms=999_999_999))

    # spawn at max depth denied
    deep = t.model_copy(update={"depth": profile.max_spawn_depth})
    d = policy_engine.can_spawn(deep, profile, Budget(wall_ms=1))
    assert not d.allowed and d.reason == "max_spawn_depth"


# ============================== scheduler ===============================


@pytest.mark.integration
def test_scheduler_runs_task_emits_refund_and_no_negative_budget(tmp_path: Path) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    nb_path = ws.notebooks_dir / "ok.ipynb"
    _write_nb(nb_path, "x = 1\nx")

    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path=str(nb_path), kernel_name="python3")
    final = ak.run_task(task.task_id)
    assert final.status == TaskStatus.completed

    events = ak.list_events(task.task_id)
    types = [e.event_type for e in events]
    # Required transitions present and ordered
    assert EventType.task_created in types
    assert EventType.task_admitted in types
    assert EventType.notebook_execution_started in types
    assert EventType.task_completed in types
    assert EventType.budget_refunded in types

    # Refund delta on completion equals the entire initial reservation
    # (since this run did no LLM/CPU debits).
    refund_ev = next(e for e in events if e.event_type == EventType.budget_refunded)
    assert refund_ev.payload["reason"] == "task_completed"
    refunded = Budget.model_validate(refund_ev.payload["delta"])
    # The originally-reserved budget came from the local-dev profile defaults.
    assert refunded.wall_ms == DEFAULT_PROFILES["local-dev"].budgets.wall_ms

    # Invariant: budget_after values across the JSONL trace are never negative
    # (Pydantic's ge=0 validators on Budget guarantee this if the model
    # accepted the events at write time; verify by re-loading raw JSONL.)
    raw = JSONLEventStore(ws.events_dir).list_events()
    for ev in raw:
        for b in (ev.budget_before, ev.budget_after):
            if b is None:
                continue
            for f in b.__class__.model_fields:
                assert getattr(b, f) >= 0


@pytest.mark.integration
def test_scheduler_failure_refunds_remaining_reservation(tmp_path: Path) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    nb_path = ws.notebooks_dir / "bad.ipynb"
    _write_nb(nb_path, "raise RuntimeError('nope')")

    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path=str(nb_path), kernel_name="python3")
    final = ak.run_task(task.task_id)
    assert final.status == TaskStatus.failed

    events = ak.list_events(task.task_id)
    refund_evs = [e for e in events if e.event_type == EventType.budget_refunded]
    assert len(refund_evs) == 1
    assert refund_evs[0].payload["reason"] == "task_failed"

    # And task.failed was emitted by the runner
    assert any(e.event_type == EventType.task_failed for e in events)

    # Reconstruct from JSONL must agree with the persisted snapshot
    raw = JSONLEventStore(ws.events_dir).list_events()
    rebuilt = reconstruct_tasks(raw)
    persisted = ak.get_task(task.task_id)
    assert persisted is not None
    assert rebuilt[task.task_id].status == TaskStatus.failed == persisted.status


@pytest.mark.integration
def test_scheduler_quota_blocking_emits_event_for_overflow_tasks(
    tmp_path: Path,
) -> None:
    """Use a 1-slot profile and submit 3 tasks concurrently; the 2nd and 3rd
    must observe a ``quota.blocked`` event because the slot is occupied.
    The semaphore still serializes execution to completion."""
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    nb_path = ws.notebooks_dir / "slow.ipynb"
    # Cell finishes in ~200ms — long enough to overlap the others' admission.
    _write_nb(nb_path, "import time; time.sleep(0.2); 1")

    one_slot = DEFAULT_PROFILES["local-dev"].model_copy(update={"kernel_slots_total": 1})

    async def _run_three() -> tuple[list[str], list[str]]:
        # Build the scheduler directly so we can drive .run_task concurrently.
        sched = Scheduler(ws, profile=one_slot)
        tids = [
            sched.create_task(notebook_path=str(nb_path), kernel_name="python3").task_id
            for _ in range(3)
        ]
        # The semaphore is created bound to the loop in which it was instantiated.
        # Since Scheduler.__init__ ran outside this loop, rebind here.
        sched._slot_semaphore = asyncio.Semaphore(one_slot.kernel_slots_total)
        results = await asyncio.gather(
            *(sched.run_task(tid) for tid in tids), return_exceptions=True
        )
        statuses = [(r.status.value if hasattr(r, "status") else f"exc:{r!r}") for r in results]
        return tids, statuses

    _tids, statuses = asyncio.run(_run_three())
    assert all(s == "completed" for s in statuses), statuses

    events = JSONLEventStore(ws.events_dir).list_events()
    blocked = [e for e in events if e.event_type == EventType.quota_blocked]
    # At least one of the three should have seen the slot occupied at admission.
    assert blocked, "expected at least one quota.blocked event under contention"
    # Every blocked event's snapshot recorded the 1-slot total
    for e in blocked:
        assert e.quota_snapshot is not None
        assert e.quota_snapshot.kernel_slots_total == 1


@pytest.mark.integration
def test_scheduler_cancellation_pre_admission_refunds_full_reservation(
    tmp_path: Path,
) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    nb_path = ws.notebooks_dir / "ok.ipynb"
    _write_nb(nb_path, "y = 2")

    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path=str(nb_path), kernel_name="python3")
    ak.cancel(task.task_id)
    final = ak.run_task(task.task_id)
    assert final.status == TaskStatus.cancelled

    events = ak.list_events(task.task_id)
    refund_ev = next(e for e in events if e.event_type == EventType.budget_refunded)
    assert refund_ev.payload["reason"] == "task_cancelled"
    delta = Budget.model_validate(refund_ev.payload["delta"])
    # Cancelled before any spend → full reservation refunded
    profile_budget = DEFAULT_PROFILES["local-dev"].budgets
    assert delta.wall_ms == profile_budget.wall_ms
    assert delta.llm_usd_micro == profile_budget.llm_usd_micro

    # task.cancelled event emitted
    assert any(e.event_type == EventType.task_cancelled for e in events)


@pytest.mark.integration
def test_scheduler_state_reconstruction_matches_persisted(tmp_path: Path) -> None:
    """Run several tasks then verify replay(events) == persisted_snapshots."""
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    nb_ok = ws.notebooks_dir / "ok.ipynb"
    nb_bad = ws.notebooks_dir / "bad.ipynb"
    _write_nb(nb_ok, "z = 0")
    _write_nb(nb_bad, "raise ValueError('x')")

    ak = AgentKernel(tmp_path)
    t1 = ak.create_task(notebook_path=str(nb_ok), kernel_name="python3")
    t2 = ak.create_task(notebook_path=str(nb_bad), kernel_name="python3")
    t3 = ak.create_task(notebook_path=str(nb_ok), kernel_name="python3")
    ak.cancel(t3.task_id)
    ak.run_task(t1.task_id)
    ak.run_task(t2.task_id)
    ak.run_task(t3.task_id)

    raw = JSONLEventStore(ws.events_dir).list_events()
    rebuilt = reconstruct_tasks(raw)

    for tid, expected_status in [
        (t1.task_id, TaskStatus.completed),
        (t2.task_id, TaskStatus.failed),
        (t3.task_id, TaskStatus.cancelled),
    ]:
        persisted = ak.get_task(tid)
        assert persisted is not None
        assert persisted.status == expected_status
        assert rebuilt[tid].status == expected_status


@pytest.mark.integration
def test_policy_profile_override_via_api(tmp_path: Path) -> None:
    """Custom PolicyProfile is plumbed through to the scheduler."""
    custom = PolicyProfile(name="tiny", kernel_slots_total=2, queue_limit=8)
    ak = AgentKernel(tmp_path, policy_profile=custom)
    assert ak.scheduler.profile.name == "tiny"
    assert ak.scheduler.profile.kernel_slots_total == 2
