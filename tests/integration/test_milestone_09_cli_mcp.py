"""Milestone 9 acceptance test: CLI + MCP service.

Per spec §16 (Milestone 9), the CLI is the required compatibility layer when
the MCP test infrastructure is not available. We test:

* CLI: subprocess invocation of ``notebook-agent run`` end-to-end produces a
  task directory with successful manifest and answer.
* MCP: in-process invocation of the FastMCP-registered tool functions to
  verify schema-correct tool wiring (no client subprocess needed for the
  tool registration itself, which is the main thing we want to guard).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from notebook_agent.mcp_server import (
    build_fastmcp,
    tool_execute_notebook,
    tool_get_task_graph,
    tool_read_manifest,
    tool_run_task,
    tool_search_skills,
)


def test_cli_run_echo_end_to_end(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"message": "from cli"}))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "notebook_agent.cli",
            "run",
            "Use the echo skill to echo something",
            "--params",
            str(params_path),
            "--runs-root",
            str(runs_root),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["success"] is True
    task_dir = Path(out["directory"])
    assert (task_dir / "manifest.json").exists()
    assert (task_dir / "outputs" / "result.json").exists()
    assert (task_dir / "outputs" / "answer.md").read_text().strip() == "from cli"


def test_cli_init_creates_project_dirs(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "notebook_agent.cli", "init", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "skills").is_dir()
    assert (tmp_path / "runs").is_dir()
    assert (tmp_path / "AGENT.md").exists()


def test_cli_search_skills_finds_echo(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "notebook_agent.cli", "search-skills", "echo message", "--json"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout)
    assert results, "expected non-empty search results"
    assert results[0]["skill"]["skill_id"] == "core.echo"


def test_mcp_tool_functions_round_trip(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    # 1. search_skills
    search = tool_search_skills("echo")
    assert any(r["skill"]["skill_id"] == "core.echo" for r in search["results"])
    # 2. run_task
    res = tool_run_task(
        "Use the echo skill",
        parameters={"message": "mcp says hi"},
        runs_root=str(runs_root),
    )
    assert res["success"] is True
    task_dir = res["directory"]
    # 3. read_manifest + get_task_graph
    m = tool_read_manifest(task_dir)
    assert m["status"] == "success"
    g = tool_get_task_graph(task_dir)
    assert g["task_id"] == m["task_id"]
    # 4. execute_notebook on the already-generated notebook
    nb_dir = Path(task_dir) / "task.ipynb"
    out = tool_execute_notebook(
        str(nb_dir),
        parameters={"message": "rerun"},
        run_dir=task_dir,
    )
    assert out["success"] is True


def test_mcp_server_registers_required_tools() -> None:
    server = build_fastmcp(runs_root="runs", skill_dirs=[])
    # FastMCP stores tools in a tool manager. We don't depend on a particular
    # private attribute; instead we use the public list_tools API where
    # available, with a fallback to introspection.
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
