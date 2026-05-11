"""M2 integration gate: NotebookRunner end-to-end against real python3 kernel.

Each test runs against a real ``python3`` ipykernel (no mocking of nbclient
or the kernel) and asserts:
- executed notebook is written to ``runs/<run_id>/executed.ipynb``
- JSONL trace contains the expected event sequence with correct cell IDs
- error fixture produces ``cell.execution.error`` and ``task.failed``
- timeout fixture honors the configured timeout

Also runs the CLI as a subprocess to gate the public CLI surface.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from agent_kernel.models.event import EventType
from agent_kernel.runtime.notebook_runner import NotebookRunFailed, NotebookRunner
from agent_kernel.storage import JSONLEventStore, WorkspaceLayout
from tests.fixtures.build_notebooks import write_all


@pytest.fixture
def workspace_with_fixtures(tmp_path: Path) -> WorkspaceLayout:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    write_all(ws.notebooks_dir)
    return ws


def _run(ws: WorkspaceLayout, notebook_name: str, *, timeout: int = 30) -> tuple:
    events = JSONLEventStore(ws.events_dir, fsync=True)
    runner = NotebookRunner(events, ws.runs_dir)
    result = runner.run(
        ws.notebooks_dir / notebook_name,
        task_id="task_TEST",
        kernel_name="python3",
        timeout=timeout,
    )
    return result, events.list_events()


@pytest.mark.integration
def test_passing_notebook_executes_and_emits_full_trace(
    workspace_with_fixtures: WorkspaceLayout,
) -> None:
    ws = workspace_with_fixtures
    result, events = _run(ws, "passing.ipynb")

    # Executed notebook on disk
    assert result.status == "completed"
    assert result.executed_notebook_path.exists()
    nb = nbformat.read(result.executed_notebook_path, as_version=4)
    # Every code cell has execution_count populated
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 3
    for c in code_cells:
        assert c.execution_count is not None, f"cell {c.id} not executed"

    # Event sequence shape
    types = [e.event_type for e in events]
    assert types[0] == EventType.notebook_execution_started
    assert types[-2] == EventType.notebook_execution_completed
    assert types[-1] == EventType.task_completed

    # cell_started + cell_completed for each of the 3 cells, in order
    cell_started = [e for e in events if e.event_type == EventType.cell_execution_started]
    cell_completed = [e for e in events if e.event_type == EventType.cell_execution_completed]
    assert [e.cell_id for e in cell_started] == ["cell_one", "cell_two", "cell_three"]
    assert [e.cell_id for e in cell_completed] == ["cell_one", "cell_two", "cell_three"]

    # Each completed event carries a duration_ms
    for e in cell_completed:
        assert isinstance(e.payload.get("duration_ms"), int)
        assert e.payload["duration_ms"] >= 0

    # No cell errors
    assert not any(e.event_type == EventType.cell_execution_error for e in events)

    # Run directory layout
    assert (result.run_dir / "executed.ipynb").exists()
    assert (result.run_dir / "stdout.log").exists()
    assert (result.run_dir / "stderr.log").exists()


@pytest.mark.integration
def test_erroring_notebook_emits_cell_error_and_task_failed(
    workspace_with_fixtures: WorkspaceLayout,
) -> None:
    ws = workspace_with_fixtures
    events_store = JSONLEventStore(ws.events_dir, fsync=True)
    runner = NotebookRunner(events_store, ws.runs_dir)

    with pytest.raises(NotebookRunFailed) as exc_info:
        runner.run(
            ws.notebooks_dir / "erroring.ipynb",
            task_id="task_ERR",
            kernel_name="python3",
            timeout=30,
        )
    assert exc_info.value.cell_id == "cell_bad"

    events = events_store.list_events()
    types = [e.event_type for e in events]
    assert EventType.cell_execution_error in types
    assert EventType.task_failed in types
    assert EventType.task_completed not in types

    err_event = next(e for e in events if e.event_type == EventType.cell_execution_error)
    assert err_event.cell_id == "cell_bad"
    assert err_event.error["ename"] == "RuntimeError"
    assert "boom from fixture" in (err_event.error.get("evalue") or "")

    failed_event = next(e for e in events if e.event_type == EventType.task_failed)
    assert failed_event.error["kind"] == "cell_error"
    assert failed_event.payload["failed_cell_id"] == "cell_bad"

    # The executed notebook should still be written, with cell_good completed
    # and cell_bad carrying an error output.
    run_dir = ws.runs_dir / failed_event.run_id
    nb = nbformat.read(run_dir / "executed.ipynb", as_version=4)
    by_id = {c.id: c for c in nb.cells}
    assert by_id["cell_good"].execution_count is not None
    bad_outputs = by_id["cell_bad"].outputs
    assert any(o.get("output_type") == "error" for o in bad_outputs)


@pytest.mark.integration
@pytest.mark.slow
def test_timeout_notebook_honors_per_cell_timeout(
    workspace_with_fixtures: WorkspaceLayout,
) -> None:
    ws = workspace_with_fixtures
    events_store = JSONLEventStore(ws.events_dir, fsync=True)
    runner = NotebookRunner(events_store, ws.runs_dir)

    with pytest.raises(NotebookRunFailed):
        runner.run(
            ws.notebooks_dir / "timeout.ipynb",
            task_id="task_TIMEOUT",
            kernel_name="python3",
            timeout=2,  # cell sleeps 60s; must trip
        )
    events = events_store.list_events()
    failed = next(e for e in events if e.event_type == EventType.task_failed)
    assert failed.error["kind"] == "timeout"
    # cell_fast completed; cell_slow timed out
    completed_cells = [
        e.cell_id for e in events if e.event_type == EventType.cell_execution_completed
    ]
    assert "cell_fast" in completed_cells


@pytest.mark.integration
def test_cli_run_subcommand_against_passing_notebook(
    workspace_with_fixtures: WorkspaceLayout,
) -> None:
    """Public CLI surface gate."""
    ws = workspace_with_fixtures
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_kernel.cli",
            "run",
            str(ws.notebooks_dir / "passing.ipynb"),
            "--workspace",
            str(ws.root),
            "--kernel",
            "python3",
            "--timeout",
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK task_id=" in result.stdout
    # At least one events file produced
    assert list(ws.events_dir.glob("*.jsonl"))


@pytest.mark.integration
def test_cli_run_returns_nonzero_on_failure(
    workspace_with_fixtures: WorkspaceLayout,
) -> None:
    ws = workspace_with_fixtures
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_kernel.cli",
            "run",
            str(ws.notebooks_dir / "erroring.ipynb"),
            "--workspace",
            str(ws.root),
            "--kernel",
            "python3",
            "--timeout",
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAILED" in result.stderr
