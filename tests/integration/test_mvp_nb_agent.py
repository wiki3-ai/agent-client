"""MVP acceptance tests for ``nb-agent.md``.

Every LLM step is a DSPy ``Predict`` call. Tests script the DummyLM with one
answer dict per call, in the order the agent makes them:

* generate path  : planner, chooser('none'), code_generator
* transform path : planner, chooser(skill_id), parameters
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy  # type: ignore[import-untyped]
import nbformat
from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

from notebook_agent import root_template_path, run_task
from notebook_agent.magics import task_cell_magic, task_line_magic


def _configure_generate_dummy(snippet: str) -> None:
    """Script DummyLM for the **generate** path: plan, none, code."""
    dspy.configure(lm=DummyLM([
        {"plan": "- think\n- write code\n- run"},
        {"chosen_skill_id": "none"},
        {"python_code": snippet},
    ]))


# ---------------------------------------------------------------------------
# 1. Zero-config run_task(prompt)
# ---------------------------------------------------------------------------


def test_run_task_zero_config_word_count(tmp_path: Path) -> None:
    """Spec flagship: run_task generates+executes a word-count notebook for a
    prompt with no pre-authored skill. Result must equal 6."""
    snippet = (
        "text = 'hello from the graph notebook agent'\n"
        "result = {'word_count': len(text.split())}\n"
    )
    _configure_generate_dummy(snippet)
    result = run_task(
        "Create and execute a notebook that counts the words in: hello from the graph notebook agent",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
    )
    assert result.success, result.error
    assert result.stage_used == "generate"
    assert isinstance(result.result_payload, dict)
    assert result.result_payload.get("word_count") == 6


# ---------------------------------------------------------------------------
# 2. Notebook-native persistence
# ---------------------------------------------------------------------------


def test_executed_notebook_carries_canonical_state(tmp_path: Path) -> None:
    snippet = "result = {'word_count': len('foo bar baz'.split())}\n"
    _configure_generate_dummy(snippet)
    result = run_task(
        "count words: foo bar baz",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
    )
    assert result.success
    nb = nbformat.read(str(result.task.executed_notebook), as_version=4)
    md = nb.metadata.get("notebook_agent") or {}
    assert md.get("task_id") == result.task.task_id
    assert md.get("stage_used") == "generate"
    assert md.get("result") == result.result_payload
    assert md.get("plan"), "plan/TODO list should be persisted in notebook metadata"
    assert md.get("generated"), "generated snippet should be recorded for provenance"
    in_nb = nbformat.read(str(result.task.task_notebook), as_version=4)
    in_md = in_nb.metadata.get("notebook_agent") or {}
    assert in_md.get("stage") == "generate"
    assert "result" in in_md.get("generated", {}).get("source", "")


# ---------------------------------------------------------------------------
# 3. %task / %%task magics
# ---------------------------------------------------------------------------


class _FakeIP:
    def __init__(self) -> None:
        self.user_ns: dict = {}


class _StubResult:
    def __init__(self, prompt: str) -> None:
        self.answer = f"stub:{prompt}"
        self.success = True


def test_line_magic_dispatches_to_run_task(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_task(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return _StubResult(prompt)

    monkeypatch.setattr("notebook_agent.magics.run_task", fake_run_task)
    ip = _FakeIP()
    rendered = task_line_magic("count words: foo bar baz", ip=ip)
    assert captured["prompt"] == "count words: foo bar baz"
    assert rendered is not None
    assert ip.user_ns.get("_last_task_result") is not None


def test_cell_magic_dispatches_to_run_task(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_task(prompt, **kwargs):
        captured["prompt"] = prompt
        return _StubResult(prompt)

    monkeypatch.setattr("notebook_agent.magics.run_task", fake_run_task)
    ip = _FakeIP()
    body = "do the thing\nwith multiple lines"
    task_cell_magic("", body, ip=ip)
    assert captured["prompt"] == body


def test_magic_continue_flag_passes_prior_result(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_task(prompt, **kwargs):
        captured["continue_from"] = kwargs.get("continue_from")
        return _StubResult(prompt)

    monkeypatch.setattr("notebook_agent.magics.run_task", fake_run_task)
    ip = _FakeIP()
    prior = _StubResult("first")
    ip.user_ns["_last_task_result"] = prior
    task_line_magic("--continue try again", ip=ip)
    assert captured["continue_from"] is prior


# ---------------------------------------------------------------------------
# 4. Continuation
# ---------------------------------------------------------------------------


def test_continuation_records_parent(tmp_path: Path) -> None:
    _configure_generate_dummy("result = {'word_count': 3}\n")
    first = run_task(
        "count words: alpha beta gamma",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
    )
    assert first.success
    _configure_generate_dummy("result = {'word_count': 3, 'note': 'retry'}\n")
    follow = run_task(
        "that's wrong, try again",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
        continue_from=first,
    )
    assert follow.success
    m = json.loads(follow.task.manifest_json.read_text())
    assert m.get("continued_from") == str(first.task.directory)
    assert len(follow.plan) >= len(first.plan)


# ---------------------------------------------------------------------------
# 5. max_autonomous_turns
# ---------------------------------------------------------------------------


def test_max_autonomous_turns_zero_blocks_generation(tmp_path: Path) -> None:
    # DummyLM with no answers is fine — we should never reach it.
    dspy.configure(lm=DummyLM([{"plan": ""}]))
    result = run_task(
        "do a thing with no skill",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
        max_autonomous_turns=0,
    )
    assert not result.success
    assert result.error is not None
    assert (
        result.error.get("type") == "BudgetExhaustedError"
        or "turn" in (result.error.get("message", "").lower())
    )


# ---------------------------------------------------------------------------
# 6. Root Papermill template
# ---------------------------------------------------------------------------


def test_root_template_is_a_valid_parameterized_notebook() -> None:
    p = root_template_path()
    assert p.exists(), f"missing root template at {p}"
    nb = nbformat.read(str(p), as_version=4)
    tagged = [c for c in nb.cells if "parameters" in (c.metadata.get("tags") or [])]
    assert tagged, "root template must contain a Papermill parameters cell"
    src = tagged[0].source
    for required in ("prompt", "runs_root", "max_autonomous_turns"):
        assert required in src, f"parameters cell must expose {required!r}"
