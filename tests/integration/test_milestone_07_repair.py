"""Milestone 7 acceptance test: repair loop."""

from __future__ import annotations

import json
from pathlib import Path

from notebook_agent import Budget, BudgetTracker, create_root_task
from notebook_agent._notebook_io import make_notebook, parameters_cell, write_notebook
from notebook_agent.litellm_client import LiteLLMClient
from notebook_agent.notebook_exec import execute_notebook
from notebook_agent.repair import repair_and_rerun


def _missing_dir_notebook(path: Path) -> Path:
    # This notebook writes to a path inside a directory that does not yet
    # exist (the "subdir/" under outputs is not created). It deliberately
    # references the path with a hard-coded relative location to make the
    # diagnostic detection of "No such file or directory" trigger.
    nb = make_notebook(
        [
            parameters_cell(),
            (
                "code",
                (
                    "from pathlib import Path\n"
                    "_target = Path(output_dir) / 'subdir' / 'data.txt'\n"
                    "_target.write_text('hello')  # fails: 'subdir' missing\n"
                ),
                [],
            ),
            (
                "code",
                (
                    "import json\n"
                    "from pathlib import Path\n"
                    "Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
                    "(Path(output_dir) / 'result.json').write_text(json.dumps({'ok': True}))\n"
                ),
                ["write_result"],
            ),
        ]
    )
    return write_notebook(nb, path)


def _undefined_name_notebook(path: Path) -> Path:
    nb = make_notebook(
        [
            parameters_cell(),
            ("code", "result = {'value': some_undefined_var}\n", []),
            (
                "code",
                (
                    "import json\n"
                    "from pathlib import Path\n"
                    "Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
                    "(Path(output_dir) / 'result.json').write_text(json.dumps(result))\n"
                ),
                [],
            ),
        ]
    )
    return write_notebook(nb, path)


def test_repair_missing_output_directory(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(
        runs_root,
        title="Repair me",
        request="produce a file in a missing dir",
        budget=Budget(max_notebook_executions=4, max_repair_attempts=2),
    )
    nb = _missing_dir_notebook(task.task_notebook)
    tracker = BudgetTracker(task.budget)

    res = execute_notebook(
        nb,
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=tracker,
    )
    assert not res.success
    task.update_status("failed")

    outcome = repair_and_rerun(task, res, budget=tracker)
    assert outcome.repair_task is not None
    assert outcome.repaired, outcome.repaired_result.error if outcome.repaired_result else None
    assert outcome.strategy == "deterministic"
    # The repair child task exists with success status.
    assert outcome.repair_task.directory.exists()
    assert json.loads(outcome.repair_task.task_json.read_text())["status"] == "success"
    # Parent manifest records both initial failure and repaired success.
    pm = json.loads(task.manifest_json.read_text())
    assert pm["repairs"]
    rec = pm["repairs"][0]
    assert rec["success"] is True
    assert rec["diagnosis"]["kind"] == "missing_output_dir"


def test_repair_undefined_name_with_fake_llm(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    task = create_root_task(
        runs_root,
        title="Repair name error",
        request="patch in a missing variable",
        budget=Budget(max_notebook_executions=4, max_repair_attempts=2),
    )
    nb = _undefined_name_notebook(task.task_notebook)
    tracker = BudgetTracker(task.budget)

    res = execute_notebook(
        nb,
        output_path=task.executed_notebook,
        run_dir=task.directory,
        event_log=task.event_log(),
        budget=tracker,
    )
    assert not res.success

    fake_llm = LiteLLMClient(provider="fake", fake_response="some_undefined_var = 0")
    outcome = repair_and_rerun(task, res, budget=tracker, llm=fake_llm)
    assert outcome.repaired, outcome.repaired_result.error if outcome.repaired_result else None
    # FakeProvider path returns a deterministic default, which we apply as a
    # deterministic patch — the strategy may be either "deterministic" or
    # "llm" depending on which branch triggered. Both are acceptable; what
    # matters is that the rerun succeeds and is recorded.
    assert outcome.strategy in ("deterministic", "llm")
