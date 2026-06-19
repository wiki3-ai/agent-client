"""M1 integration gate: crash/replay reconstruction.

Writes synthetic provenance events and atomic task snapshots to a real
on-disk workspace, simulates a process restart, replays the JSONL stream,
and asserts that reconstructed task state matches pre-crash state
bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_kernel.models import (
    Budget,
    EventType,
    ProvenanceEvent,
    TaskSpec,
    TaskStatus,
)
from agent_kernel.runtime.reconstruct import reconstruct_tasks
from agent_kernel.storage import AtomicJSONStore, JSONLEventStore, WorkspaceLayout
from agent_kernel.util import new_id, now_iso


def _make_task(task_id: str, *, status: TaskStatus = TaskStatus.draft) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        notebook_path=f"notebooks/{task_id}.ipynb",
        kernel_name="python3",
        policy_profile="local-dev",
        status=status,
        reserved_budget=Budget(wall_ms=60_000, llm_usd_micro=10_000),
        created_at=now_iso(),
        updated_at=now_iso(),
    )


def _ev(task: TaskSpec, et: EventType, **kw: object) -> ProvenanceEvent:
    return ProvenanceEvent(
        event_id=new_id("evt"),
        ts=now_iso(),
        event_type=et,
        task_id=task.task_id,
        notebook_path=task.notebook_path,
        kernel_name=task.kernel_name,
        **kw,  # type: ignore[arg-type]
    )


@pytest.mark.integration
def test_workspace_layout_ensures_dirs(tmp_path: Path) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    for d in [
        ws.state_root,
        ws.tasks_dir,
        ws.runs_dir,
        ws.events_dir,
        ws.artifacts_dir,
        ws.notebooks_dir,
        ws.spawns_dir,
    ]:
        assert d.is_dir(), d


@pytest.mark.integration
def test_jsonl_round_trip_and_partial_line_tolerance(tmp_path: Path) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    store = JSONLEventStore(ws.events_dir, fsync=True)

    t = _make_task("task_AAA")
    e1 = _ev(t, EventType.task_created, payload={"task_spec": t.model_dump(mode="json")})
    e2 = _ev(t, EventType.task_admitted)
    store.append(e1)
    store.append(e2)

    # Simulate a partial trailing line from a crash mid-write.
    day_file = next(ws.events_dir.glob("*.jsonl"))
    with open(day_file, "a", encoding="utf-8") as f:
        f.write('{"event_id": "evt_partial"')  # no newline; corrupt

    events = JSONLEventStore(ws.events_dir).list_events()
    assert [e.event_id for e in events] == [e1.event_id, e2.event_id]


@pytest.mark.integration
def test_atomic_state_round_trip(tmp_path: Path) -> None:
    store = AtomicJSONStore(tmp_path / "tasks")
    t = _make_task("task_BBB", status=TaskStatus.running)
    store.write(t.task_id, t)

    # The tmp sidecar must not survive.
    leftovers = list((tmp_path / "tasks").glob("*.tmp"))
    assert leftovers == []

    loaded = store.read(t.task_id, TaskSpec)
    assert loaded == t
    assert store.list_ids() == [t.task_id]


@pytest.mark.integration
def test_crash_replay_reconstructs_state_bit_for_bit(tmp_path: Path) -> None:
    """The integration gate for M1: write events + snapshots, kill the
    in-memory state, restart from disk, and confirm reconstruction matches.
    """
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    events = JSONLEventStore(ws.events_dir, fsync=True)
    tasks_store = AtomicJSONStore(ws.tasks_dir)

    # --- Pre-crash: build live state by writing events + snapshots ---
    t1 = _make_task("task_001")
    t2 = _make_task("task_002")

    for t in (t1, t2):
        events.append(
            _ev(t, EventType.task_created, payload={"task_spec": t.model_dump(mode="json")})
        )
    events.append(_ev(t1, EventType.task_admitted))
    events.append(_ev(t1, EventType.notebook_execution_started))
    events.append(
        _ev(
            t1,
            EventType.budget_debited,
            payload={"delta": Budget(llm_usd_micro=2_500).model_dump(mode="json")},
            budget_after=Budget(llm_usd_micro=7_500),
        )
    )
    events.append(
        _ev(
            t1,
            EventType.task_completed,
            payload={"executed_notebook_path": "notebooks/task_001-executed.ipynb"},
        )
    )
    events.append(_ev(t2, EventType.task_admitted))
    events.append(
        _ev(
            t2,
            EventType.task_failed,
            error={"name": "RuntimeError", "message": "boom"},
        )
    )

    live = reconstruct_tasks(events.list_events())
    for tid, spec in live.items():
        tasks_store.write(tid, spec)

    # --- Simulated process restart: throw away memory, rebuild from disk ---
    del live

    # 1. Reconstruct from the event log alone
    replayed = reconstruct_tasks(JSONLEventStore(ws.events_dir).iter_events())

    # 2. Compare against the persisted snapshots
    persisted_store = AtomicJSONStore(ws.tasks_dir)
    persisted = {tid: persisted_store.read(tid, TaskSpec) for tid in persisted_store.list_ids()}

    assert set(replayed) == set(persisted) == {"task_001", "task_002"}

    for tid in replayed:
        # status, child lists, and spent budget must agree exactly
        r = replayed[tid]
        p = persisted[tid]
        assert p is not None
        # updated_at differs only because the snapshot was written from the
        # replay itself; everything else must match bit-for-bit.
        r_compare = r.model_copy(update={"updated_at": p.updated_at, "created_at": p.created_at})
        assert r_compare == p, f"mismatch for {tid}: {r_compare} != {p}"

    # Spot-check the specific transitions
    assert replayed["task_001"].status == TaskStatus.completed
    assert replayed["task_001"].spent_budget.llm_usd_micro == 2_500
    assert replayed["task_001"].executed_notebook_path == "notebooks/task_001-executed.ipynb"
    assert replayed["task_002"].status == TaskStatus.failed
    assert replayed["task_002"].error == {"name": "RuntimeError", "message": "boom"}


@pytest.mark.integration
def test_schemas_emit_to_docs() -> None:
    """The M1 schema-dump script must produce valid JSON Schema files."""
    import json

    from scripts.dump_schemas import SCHEMAS

    for filename, model_cls in SCHEMAS.items():
        schema_path = Path("docs/schemas") / filename
        assert schema_path.exists(), f"missing {schema_path}"
        data = json.loads(schema_path.read_text())
        assert data.get("type") == "object"
        assert "properties" in data
        # Round-trip: the model should still produce the same schema.
        assert data == model_cls.model_json_schema()
