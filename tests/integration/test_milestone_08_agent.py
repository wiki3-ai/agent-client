"""Milestone 8 acceptance test: end-to-end agent loop with echo skill."""

from __future__ import annotations

import json
from pathlib import Path

from notebook_agent.agent import run_task


def test_run_task_echo_end_to_end(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    result = run_task(
        "Use the echo skill to echo hello graph agent",
        runs_root=runs_root,
        parameters={"message": "hello graph agent"},
    )
    assert result.success, result.error
    assert result.stage_used == "transform"
    assert result.result_payload == {"message": "hello graph agent"}

    # Filesystem layout.
    task = result.task
    for required in (task.task_json, task.manifest_json, task.readme, task.events_log, task.result_json, task.answer_md, task.executed_notebook, task.task_notebook):
        assert required.exists(), f"missing {required}"

    # Final answer matches the echoed message.
    assert task.answer_md.read_text().strip() == "hello graph agent"

    # Manifest captures the stage decision.
    m = json.loads(task.manifest_json.read_text())
    assert m["status"] == "success"
    assert m["stage_used"] == "transform"
    sd = m["stage_decision"]
    assert sd["chosen"] == "transform"
    assert sd["retrieve"]["attempted"] is True
    assert sd["transform"]["attempted"] is True
    assert sd["generate"]["attempted"] is False

    # Event log includes retrieve, transform-equivalent, execute, summarize.
    kinds = [e["event"] for e in task.event_log().read()]
    assert "retrieval_started" in kinds
    assert "retrieval_finished" in kinds
    assert "notebook_execution_started" in kinds
    assert "notebook_execution_finished" in kinds
    assert "task_finished" in kinds


def test_run_task_no_matching_skill_records_clear_failure(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    result = run_task(
        "xyzzy123 plugh quux frobnicate",
        runs_root=runs_root,
        skill_dirs=[],  # only built-in skills available; none of these tokens match
    )
    assert not result.success
    assert result.stage_used == "generate"
    assert result.error is not None
    assert result.error["type"] == "NoSkillFound"
    m = json.loads(result.task.manifest_json.read_text())
    assert m["status"] == "failed"
    assert m["stage_decision"]["generate"]["attempted"] in (False, True)
