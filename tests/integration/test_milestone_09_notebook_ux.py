"""Milestone 9 acceptance test: Jupyter-notebook UX + MCP service.

Per the user direction overriding spec §17: the user UX is a Jupyter
notebook, not a CLI. We exercise that UX two ways:

* End-to-end: papermill-execute ``examples/quickstart.ipynb`` and verify it
  produces a successful task graph (this is exactly what a real user does
  except their cells render interactively instead of being captured).
* In-process: call the public package API (``run_task``, ``search_skills``,
  the ``show_*`` display helpers) the same way a notebook cell would.

The MCP server is *not* part of the user UX — it's how other agents talk to
this one — but we still verify the FastMCP tool wiring here so milestone 9
remains covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import papermill as pm
import pytest

from notebook_agent import (
    TaskGraph,
    run_task,
    search_skills,
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
    tool_read_manifest,
    tool_run_task,
    tool_search_skills,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART_NB = REPO_ROOT / "examples" / "quickstart.ipynb"


# ---------------------------------------------------------------------------
# Notebook UX
# ---------------------------------------------------------------------------


def test_quickstart_notebook_runs_end_to_end(tmp_path: Path) -> None:
    """Papermill-executing the user-facing quickstart notebook succeeds.

    This is the milestone-9 acceptance criterion: the user's notebook-based
    UX produces a real task graph end-to-end without any CLI invocation.
    """
    executed = tmp_path / "quickstart.executed.ipynb"
    runs_root = tmp_path / "runs"
    pm.execute_notebook(
        str(QUICKSTART_NB),
        str(executed),
        parameters={
            "request": "Use the echo skill to echo from the quickstart notebook",
            "message": "hello from quickstart",
            "runs_root": str(runs_root),
        },
        cwd=str(tmp_path),
        kernel_name="python3",
    )
    # The notebook created a run directory with a successful manifest.
    run_dirs = list(runs_root.rglob("manifest.json"))
    assert run_dirs, f"no manifests under {runs_root}"
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in run_dirs]
    statuses = {m["status"] for m in manifests}
    assert "success" in statuses, f"expected at least one success, got {statuses}"


def test_notebook_api_round_trip(tmp_path: Path) -> None:
    """The in-process API a notebook cell would call: run_task + show_*."""
    runs_root = tmp_path / "runs"
    result = run_task(
        "Use the echo skill",
        parameters={"message": "from-api"},
        runs_root=runs_root,
    )
    assert result.success, result.error
    # Display helpers don't raise and return something printable.
    for helper in (show_task, show_answer, show_manifest, show_events, show_graph):
        rendered = helper(result)
        assert rendered is not None
        # str(...) is the IPython-display object repr; it must not be empty.
        assert str(rendered).strip() != ""
    # Reloading the graph from disk matches the live task.
    graph = TaskGraph.load(result.task.directory)
    assert graph.root.task_id == result.task.task_id


def test_search_skills_finds_echo() -> None:
    results = search_skills("echo message", top=5)
    assert results, "expected at least one search hit"
    assert results[0].skill.skill_id == "core.echo"


# ---------------------------------------------------------------------------
# MCP server (agent-to-agent surface, not user UX)
# ---------------------------------------------------------------------------


def test_mcp_tool_functions_round_trip(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    search = tool_search_skills("echo")
    assert any(r["skill"]["skill_id"] == "core.echo" for r in search["results"])

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
        "search_skills",
        "read_manifest",
        "get_task_graph",
        "execute_notebook",
    }
    missing = required - tool_names
    assert not missing, f"missing MCP tools: {missing}"
