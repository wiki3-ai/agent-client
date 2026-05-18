"""Milestone 2 acceptance test: real Papermill notebook execution."""

from __future__ import annotations

import json
from pathlib import Path

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
            ("code", "message = f'Hello, {name}'\nresult = {'message': message}", []),
            write_result_cell(),
        ]
    )
    return write_notebook(nb, path)


def _failing_notebook(path: Path) -> Path:
    nb = make_notebook(
        [
            parameters_cell(),
            ("code", "raise ValueError('intentional failure')", []),
        ]
    )
    return write_notebook(nb, path)


def test_papermill_execution_success_with_budget_and_events(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(runs_root, title="Hello task", request="Say hi", budget=Budget(max_notebook_executions=2))
    nb_path = _hello_notebook(task.task_notebook)
    tracker = BudgetTracker(task.budget)

    result = execute_notebook(
        nb_path,
        parameters={"name": "World"},
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=tracker,
    )

    assert result.success is True
    assert task.executed_notebook.exists()
    assert task.result_json.exists()
    payload = json.loads(task.result_json.read_text())
    assert payload == {"message": "Hello, World"}

    # Budget incremented.
    assert tracker.snapshot()["used"]["max_notebook_executions"] == 1

    # Manifest status can be set to success.
    task.update_status("success")
    assert json.loads(task.manifest_json.read_text())["status"] == "success"

    # Event log records execution.
    events = task.event_log().read()
    kinds = [e["event"] for e in events]
    assert "notebook_execution_started" in kinds
    assert "notebook_execution_finished" in kinds


def test_papermill_execution_failure_captured(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(runs_root, title="Boom task", request="fail please")
    nb_path = _failing_notebook(task.task_notebook)
    tracker = BudgetTracker(task.budget)

    result = execute_notebook(
        nb_path,
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=tracker,
    )

    assert result.success is False
    assert result.error is not None
    assert "ValueError" in (result.error.get("type", "") + result.error.get("message", ""))

    task.update_status("failed")
    assert json.loads(task.manifest_json.read_text())["status"] == "failed"

    events_kinds = [e["event"] for e in task.event_log().read()]
    assert "notebook_execution_failed" in events_kinds


def test_budget_blocks_second_execution(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(runs_root, title="Capped", request="r", budget=Budget(max_notebook_executions=1))
    nb_path = _hello_notebook(task.task_notebook)
    tracker = BudgetTracker(task.budget)
    res = execute_notebook(
        nb_path,
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=tracker,
    )
    assert res.success is True

    # Second call must be refused before Papermill runs.
    try:
        execute_notebook(
            nb_path,
            output_path=task.directory / "second.ipynb",
            run_dir=task.directory,
            event_log=task.event_log(),
            budget=tracker,
        )
    except BudgetExhaustedError as e:
        assert e.resource == "notebook_executions"
    else:
        raise AssertionError("expected BudgetExhaustedError")
