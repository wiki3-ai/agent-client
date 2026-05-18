"""Milestone 6 acceptance test: budget exhaustion ends task gracefully."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebook_agent import Budget, BudgetExhaustedError, BudgetTracker, create_root_task
from notebook_agent._notebook_io import (
    make_notebook,
    parameters_cell,
    write_notebook,
    write_result_cell,
)
from notebook_agent.notebook_exec import execute_notebook


def _hello_notebook(path: Path) -> Path:
    nb = make_notebook(
        [
            parameters_cell(['name = "World"']),
            ("code", "result = {'message': f'Hello, {name}'}", []),
            write_result_cell(),
        ]
    )
    return write_notebook(nb, path)


def test_budget_exhausted_records_graceful_stop(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(
        runs_root,
        title="Capped task",
        request="run twice but only one allowed",
        budget=Budget(max_notebook_executions=1),
    )
    nb = _hello_notebook(task.task_notebook)
    tracker = BudgetTracker(task.budget)

    # First execution succeeds.
    res = execute_notebook(nb, output_path=task.executed_notebook, run_dir=task.directory, event_log=task.event_log(), budget=tracker)
    assert res.success

    # Second one refused BEFORE Papermill runs.
    second_out = task.directory / "second.ipynb"
    with pytest.raises(BudgetExhaustedError) as ei:
        execute_notebook(nb, output_path=second_out, run_dir=task.directory, event_log=task.event_log(), budget=tracker)
    assert ei.value.resource == "notebook_executions"
    # The second .ipynb must NOT be created since Papermill never ran.
    assert not second_out.exists()

    # The task records exhaustion and we set the status accordingly.
    task.update_status("budget_exhausted")
    m = json.loads(task.manifest_json.read_text())
    m["budget_used"] = tracker.snapshot()["used"]
    m["budget_remaining"] = tracker.snapshot()["remaining"]
    m["exhaustion"] = {"resource": ei.value.resource, "requested": ei.value.requested, "remaining": ei.value.remaining}
    task.write_manifest(m)
    m2 = json.loads(task.manifest_json.read_text())
    assert m2["status"] == "budget_exhausted"
    assert m2["exhaustion"]["resource"] == "notebook_executions"
    assert m2["budget_used"]["max_notebook_executions"] == 1
    assert m2["budget_remaining"]["max_notebook_executions"] == 0
