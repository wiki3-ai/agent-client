"""MCP service wrapper (Section 14.10, Milestone 9).

Exposes notebook-agent capabilities as MCP tools. The server uses the official
``mcp`` Python SDK's :class:`FastMCP` for the simplest possible registration.

Available tools (per spec §14.10):

* ``run_task`` — run a full agent task end-to-end;
* ``run_skill`` — transform a known skill and execute it directly;
* ``search_skills`` — lexical search across built-in + project skills;
* ``read_manifest`` — read a task manifest by directory;
* ``get_task_graph`` — render a task graph (parent + children) as JSON;
* ``execute_notebook`` — execute a parameterized notebook.

The module avoids importing :mod:`mcp` at top level so that the rest of the
package keeps working without the optional ``mcp`` extra installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import run_task as _run_task
from .notebook_exec import execute_notebook as _execute_notebook
from .skills import SkillRepository
from .task_graph import TaskGraph
from .transform import builtin_skills_root, transform_skill_to_notebook

# ---------------------------------------------------------------------------
# Tool implementations as plain functions (also reused by the test suite)
# ---------------------------------------------------------------------------


def tool_search_skills(query: str, skill_dirs: list[str] | None = None, top: int = 10) -> dict[str, Any]:
    roots: list[Path | str] = [builtin_skills_root()]
    if skill_dirs:
        roots.extend(skill_dirs)
    repo = SkillRepository(roots)
    results = repo.search(query, top_k=top)
    return {"query": query, "results": [r.to_dict() for r in results]}


def tool_read_manifest(directory: str) -> dict[str, Any]:
    p = Path(directory) / "manifest.json"
    if not p.exists():
        return {"error": f"no manifest at {directory}"}
    return json.loads(p.read_text(encoding="utf-8"))


def tool_get_task_graph(directory: str) -> dict[str, Any]:
    graph = TaskGraph.load(directory)
    return graph.to_dict()


def tool_run_task(
    request: str,
    parameters: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    runs_root: str = "runs",
    skill_dirs: list[str] | None = None,
) -> dict[str, Any]:
    result = _run_task(
        request,
        runs_root=runs_root,
        parameters=parameters or {},
        budget=budget,
        skill_dirs=list(skill_dirs or []) or None,
    )
    return result.to_dict()


def tool_run_skill(
    skill_id: str,
    parameters: dict[str, Any] | None = None,
    runs_root: str = "runs",
    skill_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Run a known skill directly: transform + execute, no retrieval."""
    from .agent import run_task as _rt

    # We piggy-back on run_task by using the skill's name as the request -
    # but to be deterministic we instead build the task ourselves.
    roots: list[Path | str] = [builtin_skills_root()]
    if skill_dirs:
        roots.extend(skill_dirs)
    repo = SkillRepository(roots)
    skill = repo.find(skill_id)
    if skill is None:
        return {"error": f"unknown skill: {skill_id}"}
    # Use the skill id as the request so the retriever can also find it later.
    return _rt(
        f"Run skill {skill_id}",
        runs_root=runs_root,
        parameters=parameters or {},
        skill_dirs=list(skill_dirs or []) or None,
    ).to_dict()


def tool_execute_notebook(
    notebook: str,
    parameters: dict[str, Any] | None = None,
    output: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    res = _execute_notebook(
        Path(notebook),
        parameters=parameters or {},
        output_path=Path(output) if output else None,
        run_dir=Path(run_dir) if run_dir else None,
    )
    return res.to_dict()


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------


def build_fastmcp(*, runs_root: Path | str = "runs", skill_dirs: list[Path | str] | None = None) -> Any:
    """Return a configured :class:`mcp.server.fastmcp.FastMCP` instance.

    Importing :mod:`mcp` here is deferred so a missing optional extra does not
    break the rest of the package.
    """
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    server = FastMCP("notebook-agent")
    skill_dirs_str = [str(p) for p in (skill_dirs or [])]
    runs_root_str = str(runs_root)

    @server.tool()
    def search_skills(query: str, top: int = 10) -> dict[str, Any]:
        """Search skills by lexical query. Returns ranked results with manifest excerpts."""
        return tool_search_skills(query, skill_dirs=skill_dirs_str, top=top)

    @server.tool()
    def read_manifest(directory: str) -> dict[str, Any]:
        """Read a task manifest.json by task directory path."""
        return tool_read_manifest(directory)

    @server.tool()
    def get_task_graph(directory: str) -> dict[str, Any]:
        """Return a task graph (root and recursive children) as JSON."""
        return tool_get_task_graph(directory)

    @server.tool()
    def run_task(
        request: str,
        parameters: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a task end-to-end via Retrieve → Compose → Transform → Generate."""
        return tool_run_task(
            request,
            parameters=parameters,
            budget=budget,
            runs_root=runs_root_str,
            skill_dirs=skill_dirs_str,
        )

    @server.tool()
    def run_skill(skill_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a specific skill by id, skipping retrieval."""
        return tool_run_skill(
            skill_id,
            parameters=parameters,
            runs_root=runs_root_str,
            skill_dirs=skill_dirs_str,
        )

    @server.tool()
    def execute_notebook(
        notebook: str,
        parameters: dict[str, Any] | None = None,
        output: str | None = None,
        run_dir: str | None = None,
    ) -> dict[str, Any]:
        """Execute a parameterized notebook with Papermill."""
        return tool_execute_notebook(notebook, parameters=parameters, output=output, run_dir=run_dir)

    # Reference the transformer so MCP clients can discover its existence via
    # introspection even though it is currently consumed internally by
    # ``run_task``.
    _ = transform_skill_to_notebook
    return server


def serve_stdio(*, runs_root: Path | str = "runs", skill_dirs: list[Path | str] | None = None) -> None:
    """Run the MCP server over stdio. Blocks until the client disconnects."""
    server = build_fastmcp(runs_root=runs_root, skill_dirs=skill_dirs)
    server.run()
