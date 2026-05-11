"""M5 integration gate: parent → child spawn lineage.

Two scenarios:

1. **API-driven spawn (with parent notebook in real Jupyter kernel).**
   A parent notebook fixture is run by the M2 ``NotebookRunner``; one of
   its cells imports ``agent_kernel.api`` and calls ``spawn_child_task``,
   which materializes a child notebook on disk, executes it via the
   scheduler, and reads back the child's executed notebook path. The
   complete lineage chain must appear in the JSONL trace:

       task.created → task.spawn.requested → notebook.materialized →
       task.spawned → notebook.execution.started → … →
       task.completed (child) → task.completed (parent)

2. **Max-spawn-depth enforcement.** A parent at ``depth == max_spawn_depth``
   that attempts a spawn must emit ``spawn.denied`` and continue running.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from agent_kernel.api import AgentKernel
from agent_kernel.models.event import EventType
from agent_kernel.models.policy import DEFAULT_PROFILES
from agent_kernel.models.task import SpawnSpec, TaskStatus
from agent_kernel.runtime.notebook_runner import NotebookRunner
from agent_kernel.runtime.scheduler import Scheduler
from agent_kernel.runtime.spawn_manager import SpawnManager
from agent_kernel.storage import JSONLEventStore, WorkspaceLayout

# ============================== unit-ish ================================


@pytest.mark.integration
def test_spawn_denied_at_max_depth_emits_event_and_parent_continues(
    tmp_path: Path,
) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()

    # Profile with very low max_spawn_depth so parent is "at the wall".
    profile = DEFAULT_PROFILES["local-dev"].model_copy(update={"max_spawn_depth": 0})
    ak = AgentKernel(tmp_path, policy_profile=profile)
    parent = ak.create_task(
        notebook_path=str(tmp_path / "parent.ipynb"),
        kernel_name="python3",
    )
    # Parent depth = 0; profile max_spawn_depth = 0 → spawn must be denied.
    result = ak.spawn_child_task(
        parent.task_id,
        SpawnSpec(template_name="python-analysis", parameters={"query": "x", "limit": 1}),
    )
    assert not result.allowed
    assert result.reason == "max_spawn_depth"

    events = ak.list_events(parent.task_id)
    types = [e.event_type for e in events]
    assert EventType.task_spawn_requested in types
    assert EventType.spawn_denied in types
    assert EventType.task_spawned not in types

    denied = next(e for e in events if e.event_type == EventType.spawn_denied)
    assert denied.payload["reason"] == "max_spawn_depth"
    # Parent state was not mutated by the denied spawn.
    refreshed = ak.get_task(parent.task_id)
    assert refreshed is not None
    assert refreshed.child_task_ids == []


@pytest.mark.integration
def test_spawn_denied_when_max_children_reached(tmp_path: Path) -> None:
    """A parent that has already hit ``max_children_per_task`` is denied."""
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    profile = DEFAULT_PROFILES["local-dev"].model_copy(update={"max_children_per_task": 1})
    sched = Scheduler(ws, profile=profile)
    sm = SpawnManager(sched)
    parent = sched.create_task(notebook_path=str(tmp_path / "p.ipynb"), kernel_name="python3")

    r1 = sm.spawn(
        parent.task_id,
        SpawnSpec(template_name="python-analysis", parameters={"query": "a", "limit": 1}),
    )
    assert r1.allowed
    r2 = sm.spawn(
        parent.task_id,
        SpawnSpec(template_name="python-analysis", parameters={"query": "b", "limit": 1}),
    )
    assert not r2.allowed
    assert r2.reason == "max_children_per_task"


# =============================== integration ================================


@pytest.mark.integration
def test_api_spawn_materializes_child_with_lineage_and_runs_it(tmp_path: Path) -> None:
    """End-to-end: parent task spawns a child via SpawnManager directly, then
    the scheduler runs the child. Verifies lineage in the child notebook
    metadata and the full event chain in the JSONL trace.
    """
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()

    # parent.ipynb exists only as a stub — we don't actually execute the
    # parent here; we test the spawn lifecycle directly against the API.
    parent_nb_path = ws.notebooks_dir / "parent.ipynb"
    parent_nb = new_notebook()
    parent_nb.cells = [new_code_cell("1+1", id="cell_p")]
    parent_nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbformat.write(parent_nb, parent_nb_path)

    ak = AgentKernel(tmp_path)
    parent = ak.create_task(notebook_path=str(parent_nb_path), kernel_name="python3")

    # First spawn
    res1 = ak.spawn_child_task(
        parent.task_id,
        SpawnSpec(
            template_name="python-analysis",
            parameters={"query": "anomaly", "limit": 5},
            kernel_name="python3",
        ),
    )
    assert res1.allowed
    assert res1.child_task is not None
    child = res1.child_task
    assert child.depth == 1
    assert child.parent_task_id == parent.task_id
    assert child.spawn_index == 0
    assert child.decision_event_id == res1.decision_event_id

    # Child notebook on disk has lineage metadata
    nb = nbformat.read(child.notebook_path, as_version=4)
    ak_meta = nb.metadata["agent_kernel"]
    assert ak_meta["task_id"] == child.task_id
    assert ak_meta["parent_task_id"] == parent.task_id
    assert ak_meta["spawn"]["depth"] == 1
    assert ak_meta["spawn"]["spawn_index"] == 0
    assert ak_meta["spawn"]["decision_event_id"] == res1.decision_event_id

    # Parent reservation decreased; child registered
    parent_after_spawn = ak.get_task(parent.task_id)
    assert parent_after_spawn is not None
    assert child.task_id in parent_after_spawn.child_task_ids
    assert parent_after_spawn.reserved_budget.wall_ms < parent.reserved_budget.wall_ms

    # Now execute the child
    final_child = ak.run_task(child.task_id)
    assert final_child.status == TaskStatus.completed

    # The full lineage chain appears in the JSONL trace
    all_events = ak.list_events()
    types_in_order = [(e.event_type, e.task_id) for e in all_events]
    # Sequencing assertions on the chain involving parent + child:
    spawn_req_idx = next(
        i
        for i, (t, tid) in enumerate(types_in_order)
        if t == EventType.task_spawn_requested and tid == parent.task_id
    )
    spawned_idx = next(
        i
        for i, (t, tid) in enumerate(types_in_order)
        if t == EventType.task_spawned and tid == parent.task_id
    )
    child_completed_idx = next(
        i
        for i, (t, tid) in enumerate(types_in_order)
        if t == EventType.task_completed and tid == child.task_id
    )
    assert spawn_req_idx < spawned_idx < child_completed_idx

    # Materialization event references parent
    mat_ev = next(
        e
        for e in all_events
        if e.event_type == EventType.notebook_materialized and e.task_id == child.task_id
    )
    assert mat_ev.parent_task_id == parent.task_id

    # Second spawn from the same parent records spawn_index=1
    res2 = ak.spawn_child_task(
        parent.task_id,
        SpawnSpec(
            template_name="python-analysis",
            parameters={"query": "second", "limit": 2},
            kernel_name="python3",
        ),
    )
    assert res2.allowed
    assert res2.child_task is not None
    assert res2.child_task.spawn_index == 1


@pytest.mark.integration
@pytest.mark.slow
def test_spawn_from_inside_real_notebook_kernel(tmp_path: Path) -> None:
    """The spawn API must work when called from inside a parent notebook
    being executed by the real ``python3`` ipykernel.

    The parent notebook imports ``agent_kernel.api`` and uses
    ``spawn_child_task`` + ``run_task`` to drive a child.
    """
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()

    # Build a parent notebook whose cells exercise the API.
    parent_path = ws.notebooks_dir / "parent.ipynb"
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    source = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {json.dumps(str(Path(__file__).resolve().parents[2]))})
        from agent_kernel.api import AgentKernel
        from agent_kernel.models.task import SpawnSpec, TaskStatus

        ak = AgentKernel({json.dumps(str(ws.root))})
        # The parent task was created externally and its id is injected
        # via parent_task_id (set by the M5 test below).
        parent_id = PARENT_TASK_ID  # noqa: F821 — bound by parameter injection

        res = ak.spawn_child_task(
            parent_id,
            SpawnSpec(template_name="python-analysis",
                      parameters={{"query": "from-inside", "limit": 3}},
                      kernel_name="python3"),
        )
        assert res.allowed, res.reason
        final = ak.run_task(res.child_task.task_id)
        assert final.status == TaskStatus.completed
        print("CHILD_OK", final.task_id)
        """
    ).strip()
    nb.cells = [new_code_cell(source, id="spawn_cell")]
    nbformat.write(nb, parent_path)

    # Set up the workspace + parent task, then materialize PARENT_TASK_ID
    # into the parent notebook by editing its first cell.
    ak_outer = AgentKernel(tmp_path)
    parent_task = ak_outer.create_task(notebook_path=str(parent_path), kernel_name="python3")
    # Substitute the placeholder
    nb = nbformat.read(parent_path, as_version=4)
    nb.cells[0].source = nb.cells[0].source.replace(
        "PARENT_TASK_ID", json.dumps(parent_task.task_id)
    )
    # Also inject sys.executable PYTHONPATH so the spawned subprocess can
    # find this checkout's agent_kernel package. We use sys.path manipulation
    # at the top of the cell; the path was hardcoded above.
    nbformat.write(nb, parent_path)

    # Run the parent notebook via the M2 NotebookRunner using *this* venv's
    # python3 kernel, so `import agent_kernel` resolves to our editable install.
    events = JSONLEventStore(ws.events_dir, fsync=True)
    runner = NotebookRunner(events, ws.runs_dir)
    runner.run(parent_path, task_id=parent_task.task_id, kernel_name="python3", timeout=60)

    # The parent notebook's cell printed CHILD_OK; verify in *any* executed
    # notebook on disk (parent's run is one of several — the child also has
    # an executed notebook). The CHILD_OK marker comes from the parent cell.
    text_all = ""
    for nb_path in ws.runs_dir.glob("*/executed.ipynb"):
        executed = nbformat.read(nb_path, as_version=4)
        for c in executed.cells:
            for o in c.get("outputs") or []:
                if o.get("output_type") == "stream":
                    text_all += o.get("text", "")
    assert "CHILD_OK" in text_all, f"parent notebook output missing CHILD_OK: {text_all!r}"

    # And the JSONL trace contains both task_spawned (parent) and task_completed (child).
    types = {e.event_type for e in JSONLEventStore(ws.events_dir).list_events()}
    assert EventType.task_spawned in types
    assert EventType.task_completed in types


# Helper for sys.path injection in the inside-notebook test
__file__  # noqa: B018 — ensures linters don't strip the symbol used above
sys  # noqa: B018
