"""MVP acceptance tests for the spec in ``nb-agent.md``.

These tests cover the four properties the focused spec calls out explicitly:

1. ``run_task(prompt)`` works with **no other required arguments**.
2. The agent generates and executes code for tasks with no pre-authored skill
   (the word-count test from the spec).
3. ``%task`` / ``%%task`` magics route through the same code path.
4. Continuation reuses prior task state.
5. ``max_autonomous_turns`` is enforced.
6. Canonical task state is present in the executed notebook's metadata
   (notebook-native persistence, not sidecar JSON only).
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat

from notebook_agent import root_template_path, run_task
from notebook_agent.litellm_client import LiteLLMClient
from notebook_agent.magics import task_cell_magic, task_line_magic

WORD_COUNT_SNIPPET = (
    "```python\n"
    "text = 'hello from the graph notebook agent'\n"
    "result = {'word_count': len(text.split())}\n"
    "```\n"
)


def _fake_codegen_llm(snippet: str = WORD_COUNT_SNIPPET) -> LiteLLMClient:
    """A LiteLLM client whose every completion returns *snippet*."""
    return LiteLLMClient(provider="fake", fake_response=snippet)


# ---------------------------------------------------------------------------
# 1. Zero-config run_task(prompt)
# ---------------------------------------------------------------------------


def test_run_task_zero_config_word_count(tmp_path: Path) -> None:
    """The spec's flagship test: run_task generates+executes a word-count notebook
    for a prompt with no pre-authored skill. Result must equal 6."""
    result = run_task(
        "Create and execute a notebook that counts the words in: hello from the graph notebook agent",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
        llm=_fake_codegen_llm(),
    )
    assert result.success, result.error
    assert result.stage_used == "generate"
    assert isinstance(result.result_payload, dict)
    assert result.result_payload.get("word_count") == 6


# ---------------------------------------------------------------------------
# 2. Notebook-native persistence (no sidecar required)
# ---------------------------------------------------------------------------


def test_executed_notebook_carries_canonical_state(tmp_path: Path) -> None:
    result = run_task(
        "count words: foo bar baz",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
        llm=_fake_codegen_llm(
            "```python\nresult = {'word_count': len('foo bar baz'.split())}\n```"
        ),
    )
    assert result.success
    nb = nbformat.read(str(result.task.executed_notebook), as_version=4)
    md = nb.metadata.get("notebook_agent") or {}
    # Per nb-agent.md: machine-readable state lives in notebook metadata.
    assert md.get("task_id") == result.task.task_id
    assert md.get("stage_used") == "generate"
    assert md.get("result") == result.result_payload
    assert md.get("plan"), "plan/TODO list should be persisted in notebook metadata"
    assert md.get("generated"), "generated snippet should be recorded for provenance"
    # And the task notebook (input) carries the same generated snippet on its
    # own metadata so a user opening that file sees what the agent wrote.
    in_nb = nbformat.read(str(result.task.task_notebook), as_version=4)
    in_md = in_nb.metadata.get("notebook_agent") or {}
    assert in_md.get("stage") == "generate"
    assert "result" in in_md.get("generated", {}).get("source", "")


# ---------------------------------------------------------------------------
# 3. %task / %%task magics route through run_task
# ---------------------------------------------------------------------------


class _FakeIP:
    """Tiny IPython-shell stand-in for testing the magics in isolation."""

    def __init__(self) -> None:
        self.user_ns: dict = {}


def test_line_magic_dispatches_to_run_task(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_run_task(prompt, **kwargs):  # noqa: ANN001
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return _StubResult(prompt)

    monkeypatch.setattr("notebook_agent.magics.run_task", fake_run_task)
    ip = _FakeIP()
    rendered = task_line_magic("count words: foo bar baz", ip=ip)
    assert captured["prompt"] == "count words: foo bar baz"
    assert rendered is not None
    assert ip.user_ns.get("_last_task_result") is not None


def test_cell_magic_dispatches_to_run_task(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_run_task(prompt, **kwargs):  # noqa: ANN001
        captured["prompt"] = prompt
        return _StubResult(prompt)

    monkeypatch.setattr("notebook_agent.magics.run_task", fake_run_task)
    ip = _FakeIP()
    body = "do the thing\nwith multiple lines"
    task_cell_magic("", body, ip=ip)
    assert captured["prompt"] == body


def test_magic_continue_flag_passes_prior_result(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_task(prompt, **kwargs):  # noqa: ANN001
        captured["continue_from"] = kwargs.get("continue_from")
        return _StubResult(prompt)

    monkeypatch.setattr("notebook_agent.magics.run_task", fake_run_task)
    ip = _FakeIP()
    prior = _StubResult("first")
    ip.user_ns["_last_task_result"] = prior
    task_line_magic("--continue try again", ip=ip)
    assert captured["continue_from"] is prior


# ---------------------------------------------------------------------------
# 4. Continuation reuses prior task state
# ---------------------------------------------------------------------------


def test_continuation_records_parent(tmp_path: Path) -> None:
    first = run_task(
        "count words: alpha beta gamma",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
        llm=_fake_codegen_llm(
            "```python\nresult = {'word_count': 3}\n```"
        ),
    )
    assert first.success
    follow = run_task(
        "that's wrong, try again",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
        llm=_fake_codegen_llm(
            "```python\nresult = {'word_count': 3, 'note': 'retry'}\n```"
        ),
        continue_from=first,
    )
    assert follow.success
    # Manifest records the parent directory.
    m = json.loads(follow.task.manifest_json.read_text())
    assert m.get("continued_from") == str(first.task.directory)
    # Plan was extended, not replaced.
    assert len(follow.plan) >= len(first.plan)


# ---------------------------------------------------------------------------
# 5. max_autonomous_turns is enforced
# ---------------------------------------------------------------------------


def test_max_autonomous_turns_zero_blocks_generation(tmp_path: Path) -> None:
    result = run_task(
        "do a thing with no skill",
        runs_root=tmp_path / "runs",
        skill_dirs=[],
        llm=_fake_codegen_llm(),
        max_autonomous_turns=0,
    )
    assert not result.success
    assert result.error is not None
    # Either no turns available OR budget exhausted: both are acceptable.
    assert result.error.get("type") in {"BudgetExhaustedError"} or "turn" in (result.error.get("message", "").lower())


# ---------------------------------------------------------------------------
# 6. Root Papermill template ships with the package
# ---------------------------------------------------------------------------


def test_root_template_is_a_valid_parameterized_notebook() -> None:
    p = root_template_path()
    assert p.exists(), f"missing root template at {p}"
    nb = nbformat.read(str(p), as_version=4)
    # Must contain a Papermill 'parameters' cell.
    tagged = [c for c in nb.cells if "parameters" in (c.metadata.get("tags") or [])]
    assert tagged, "root template must contain a Papermill parameters cell"
    src = tagged[0].source
    for required in ("prompt", "runs_root", "max_autonomous_turns"):
        assert required in src, f"parameters cell must expose {required!r}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubResult:
    """Minimal stand-in for AgentResult used by the magic tests."""

    def __init__(self, prompt: str) -> None:
        self.answer = f"stub:{prompt}"
        self.success = True
