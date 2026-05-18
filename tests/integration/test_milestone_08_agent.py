"""End-to-end agent loop with the echo skill driven by a DSPy DummyLM.

Every LLM-driven decision (plan, skill selection, parameter extraction) is a
``dspy.Predict`` call; we script the DummyLM with one answer per call.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy  # type: ignore[import-untyped]
from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

from notebook_agent.agent import run_task


def test_run_task_echo_end_to_end(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"

    # Scripted answers, one per DSPy Predict call in order:
    # 1) planner   -> {"plan": "..."}
    # 2) chooser   -> {"chosen_skill_id": "core.echo"}
    # 3) extractor -> {"parameters_json": '{"message": "hello graph agent"}'}
    dspy.configure(lm=DummyLM([
        {"plan": "- echo the message\n- return the result"},
        {"chosen_skill_id": "core.echo"},
        {"parameters_json": '{"message": "hello graph agent"}'},
    ]))

    result = run_task(
        "Use the echo skill to echo hello graph agent",
        runs_root=runs_root,
    )
    assert result.success, result.error
    assert result.stage_used == "transform"
    assert result.result_payload == {"message": "hello graph agent"}

    task = result.task
    for required in (
        task.task_json, task.manifest_json, task.readme, task.events_log,
        task.result_json, task.answer_md, task.executed_notebook, task.task_notebook,
    ):
        assert required.exists(), f"missing {required}"

    assert task.answer_md.read_text().strip() == "hello graph agent"

    m = json.loads(task.manifest_json.read_text())
    assert m["status"] == "success"
    assert m["stage_used"] == "transform"
    sd = m["stage_decision"]
    assert sd["chosen"] == "transform"
    assert sd["retrieve"]["attempted"] is True
    assert sd["transform"]["attempted"] is True
    assert sd["generate"]["attempted"] is False

    kinds = [e["event"] for e in task.event_log().read()]
    assert {"retrieval_started", "retrieval_finished",
            "notebook_execution_started", "notebook_execution_finished",
            "task_finished"} <= set(kinds)


def test_run_task_chooser_says_none_falls_through_to_generate(tmp_path: Path) -> None:
    """If the DSPy chooser returns 'none', the agent generates code instead."""
    runs_root = tmp_path / "runs"

    snippet = "result = {'value': 'frobnicated'}\n"
    dspy.configure(lm=DummyLM([
        {"plan": "- pick a strategy\n- emit code"},
        {"chosen_skill_id": "none"},
        {"python_code": snippet},
    ]))

    result = run_task(
        "xyzzy123 plugh quux frobnicate",
        runs_root=runs_root,
        skill_dirs=[],
    )
    assert result.success, result.error
    assert result.stage_used == "generate"
    assert result.result_payload == {"value": "frobnicated"}

    m = json.loads(result.task.manifest_json.read_text())
    assert m["stage_decision"]["generate"]["attempted"] is True
