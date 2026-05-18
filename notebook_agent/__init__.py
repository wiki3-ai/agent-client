"""Notebook-native budget-aware Retrieve/Compose/Transform/Generate agent.

The user UX for this package is a **Jupyter notebook**. There is no
user-facing CLI. A typical session in a notebook looks like::

    from notebook_agent import run_task, show_task, show_answer
    result = run_task("Use the echo skill to echo hello",
                      parameters={"message": "hello"})
    show_task(result)      # → rich Markdown summary
    show_answer(result)    # → the rendered answer

The MCP server (for other agents, not humans) is launched with
``python -m notebook_agent.mcp_server`` and is not part of the user UX.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import AgentResult, run_task
from .budget import Budget, BudgetExhaustedError, BudgetTracker
from .display import (
    show_answer,
    show_events,
    show_graph,
    show_manifest,
    show_notebook,
    show_result,
    show_task,
)
from .events import EventLog
from .magics import load_ipython_extension, unload_ipython_extension
from .skills import SkillRepository
from .task_graph import Task, TaskGraph, create_root_task


def search_skills(query: str, *, skill_dirs: list[str] | None = None, top: int = 10):
    """Convenience wrapper: search built-in + user-supplied skills for *query*.

    Returns a list of :class:`notebook_agent.skills.SkillSearchResult` objects,
    sorted by descending score.
    """
    from pathlib import Path

    from .skills import DEFAULT_SKILL_DIRS
    from .transform import builtin_skills_root

    roots: list[Path | str] = [builtin_skills_root(), *DEFAULT_SKILL_DIRS]
    roots.extend(Path(d) for d in (skill_dirs or []))
    repo = SkillRepository(roots=roots)
    return repo.search(query, top_k=top)


def root_template_path():
    """Return the path to the bundled Papermill root agent notebook template."""
    from pathlib import Path

    return Path(__file__).resolve().parent / "templates" / "root_agent.ipynb"


__all__ = [
    "AgentResult",
    "Budget",
    "BudgetExhaustedError",
    "BudgetTracker",
    "EventLog",
    "SkillRepository",
    "Task",
    "TaskGraph",
    "__version__",
    "create_root_task",
    "load_ipython_extension",
    "root_template_path",
    "run_task",
    "search_skills",
    "show_answer",
    "show_events",
    "show_graph",
    "show_manifest",
    "show_notebook",
    "show_result",
    "show_task",
    "unload_ipython_extension",
]
