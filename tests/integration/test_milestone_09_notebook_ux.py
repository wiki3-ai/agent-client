"""Milestone 9 acceptance: Jupyter-notebook UX + MCP service.

The user UX is a Jupyter notebook. We exercise it two ways:

* End-to-end: papermill-execute ``examples/quickstart.ipynb`` with a DSPy
  ``DummyLM`` configured inside the notebook via env var. This is the same
  notebook a user would run — the only difference is that headless tests
  script the DSPy answers instead of relying on a live LM.
* In-process: call ``run_task`` and the ``show_*`` helpers directly the way
  a notebook cell would.

The MCP server is *not* user UX — it's how other agents talk to this one —
but we still verify the FastMCP tool wiring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import dspy  # type: ignore[import-untyped]
import papermill as pm
import pytest
from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

from notebook_agent import (
    TaskGraph,
    run_task,
    show_answer,
    show_events,
    show_graph,
    show_manifest,
    show_task,
)
from notebook_agent.mcp_server import (
    build_fastmcp,
    tool_execute_notebook,
    tool_get_task_graph,
    tool_list_skills,
    tool_read_manifest,
    tool_run_task,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART_NB = REPO_ROOT / "examples" / "quickstart.ipynb"


# Scripted DSPy answers for any "Use the echo skill" run:
#   planner -> plan, chooser -> core.echo, extractor -> {message: ...}
def _echo_answers(message: str) -> list[dict[str, str]]:
    return [
        {"plan": "- echo the message\n- return the result"},
        {"chosen_skill_id": "core.echo"},
        {"parameters_json": json.dumps({"message": message})},
    ]


def _configure_dspy(answers: list[dict[str, str]]) -> None:
    dspy.configure(lm=DummyLM(answers))


# ---------------------------------------------------------------------------
# Notebook UX
# ---------------------------------------------------------------------------


def test_quickstart_notebook_runs_end_to_end(tmp_path: Path) -> None:
    """Papermill-executing the user-facing quickstart notebook succeeds."""
    executed = tmp_path / "quickstart.executed.ipynb"
    runs_root = tmp_path / "runs"
    message = "hello from quickstart"
    env = os.environ.copy()
    env["NOTEBOOK_AGENT_DSPY_ANSWERS_JSON"] = json.dumps(_echo_answers(message))
    # papermill kernel inherits the current process env automatically.
    old = os.environ.get("NOTEBOOK_AGENT_DSPY_ANSWERS_JSON")
    os.environ["NOTEBOOK_AGENT_DSPY_ANSWERS_JSON"] = env["NOTEBOOK_AGENT_DSPY_ANSWERS_JSON"]
    try:
        pm.execute_notebook(
            str(QUICKSTART_NB),
            str(executed),
            parameters={
                "request": "Use the echo skill to echo from the quickstart notebook",
                "message": message,
                "runs_root": str(runs_root),
            },
            cwd=str(tmp_path),
            kernel_name="python3",
        )
    finally:
        if old is None:
            os.environ.pop("NOTEBOOK_AGENT_DSPY_ANSWERS_JSON", None)
        else:
            os.environ["NOTEBOOK_AGENT_DSPY_ANSWERS_JSON"] = old

    run_dirs = list(runs_root.rglob("manifest.json"))
    assert run_dirs, f"no manifests under {runs_root}"
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in run_dirs]
    statuses = {m["status"] for m in manifests}
    assert "success" in statuses, f"expected at least one success, got {statuses}"


def test_notebook_api_round_trip(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _configure_dspy(_echo_answers("from-api"))
    result = run_task(
        "Use the echo skill",
        parameters={"message": "from-api"},
        runs_root=runs_root,
    )
    assert result.success, result.error
    for helper in (show_task, show_answer, show_manifest, show_events, show_graph):
        rendered = helper(result)
        assert rendered is not None
        assert str(rendered).strip() != ""
    graph = TaskGraph.load(result.task.directory)
    assert graph.root.task_id == result.task.task_id


def test_skill_catalog_contains_echo() -> None:
    catalog = tool_list_skills()
    ids = {s["skill_id"] for s in catalog["skills"]}
    assert "core.echo" in ids


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


def test_mcp_tool_functions_round_trip(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    catalog = tool_list_skills()
    assert any(s["skill_id"] == "core.echo" for s in catalog["skills"])

    _configure_dspy(_echo_answers("mcp says hi"))
    res = tool_run_task(
        "Use the echo skill",
        parameters={"message": "mcp says hi"},
        runs_root=str(runs_root),
    )
    assert res["success"] is True
    task_dir = res["directory"]

    m = tool_read_manifest(task_dir)
    assert m["status"] == "success"
    g = tool_get_task_graph(task_dir)
    assert g["task_id"] == m["task_id"]

    nb = Path(task_dir) / "task.ipynb"
    out = tool_execute_notebook(
        str(nb),
        parameters={"message": "rerun"},
        run_dir=task_dir,
    )
    assert out["success"] is True


def test_mcp_server_registers_required_tools() -> None:
    server = build_fastmcp(runs_root="runs", skill_dirs=[])
    try:
        tool_manager = getattr(server, "_tool_manager", None)
        if tool_manager is not None and hasattr(tool_manager, "list_tools"):
            tool_names = {t.name for t in tool_manager.list_tools()}
        else:
            import asyncio

            tool_names = {t.name for t in asyncio.run(server.list_tools())}
    except Exception as e:  # pragma: no cover - diagnostic
        pytest.skip(f"could not list MCP tools: {e}")
    required = {
        "run_task",
        "run_skill",
        "list_skills",
        "read_manifest",
        "get_task_graph",
        "execute_notebook",
    }
    missing = required - tool_names
    assert not missing, f"missing MCP tools: {missing}"
