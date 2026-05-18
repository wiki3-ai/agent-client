"""IPython/Jupyter display helpers for the notebook-native UX.

The user UX for this agent is a Jupyter notebook. After calling
:func:`notebook_agent.run_task` (or loading an existing task with
:class:`notebook_agent.TaskGraph`), use these helpers to render a task, its
manifest, its event log, its answer, or its full graph inside the notebook.

All helpers return an IPython display object (``Markdown`` / ``JSON`` /
``HTML``). When run outside of a notebook they also have a ``str(...)``
representation so they remain useful in plain Python sessions.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .agent import AgentResult
from .task_graph import Task, TaskGraph

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_task(target: Task | TaskGraph | AgentResult | Path | str) -> Task:
    """Normalize a variety of inputs into a :class:`Task` instance."""
    if isinstance(target, Task):
        return target
    if isinstance(target, TaskGraph):
        return target.root
    if isinstance(target, AgentResult):
        return target.task
    p = Path(target)
    if not (p / "task.json").exists():
        raise FileNotFoundError(f"No task.json at {p}")
    t = Task.load(p)
    if t is None:
        raise FileNotFoundError(f"Failed to load task at {p}")
    return t


def _md(text: str) -> Any:
    """Return an IPython Markdown object, or a string fallback if IPython is
    unavailable."""
    try:
        from IPython.display import Markdown  # type: ignore[import-not-found]

        return Markdown(text)
    except Exception:  # pragma: no cover - IPython is a hard dep but guard anyway
        return text


def _json_obj(obj: Any) -> Any:
    """Return an IPython JSON display object, or the raw dict otherwise."""
    try:
        from IPython.display import JSON  # type: ignore[import-not-found]

        return JSON(obj, expanded=True)
    except Exception:  # pragma: no cover
        return obj


def _html(text: str) -> Any:
    try:
        from IPython.display import HTML  # type: ignore[import-not-found]

        return HTML(text)
    except Exception:  # pragma: no cover
        return text


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def show_task(target: Task | TaskGraph | AgentResult | Path | str) -> Any:
    """Render a high-level summary of a task as Markdown.

    Suitable for the first cell after a ``run_task`` call. Shows status,
    stage, directory, request, and a link to the answer if present.
    """
    task = _resolve_task(target)
    lines = [
        f"### Task `{task.task_id}` — {task.title}",
        "",
        f"- **Status:** `{task.status}`",
        f"- **Stage used:** `{task.stage_used or '—'}`",
        f"- **Directory:** `{task.directory}`",
        f"- **Created at:** {task.created_at}",
    ]
    if task.parent_task_id:
        lines.append(f"- **Parent task:** `{task.parent_task_id}`")
    lines += ["", "**Request:**", "", "> " + task.request.replace("\n", "\n> ")]
    if task.answer_md.exists():
        answer = task.answer_md.read_text(encoding="utf-8").strip()
        lines += ["", "**Answer:**", "", answer]
    return _md("\n".join(lines))


def show_manifest(target: Task | TaskGraph | AgentResult | Path | str) -> Any:
    """Render ``manifest.json`` as an IPython JSON tree."""
    task = _resolve_task(target)
    return _json_obj(task.read_manifest())


def show_result(target: Task | TaskGraph | AgentResult | Path | str) -> Any:
    """Render ``outputs/result.json`` as an IPython JSON tree."""
    task = _resolve_task(target)
    if not task.result_json.exists():
        return _md(f"_No `result.json` for task `{task.task_id}`._")
    return _json_obj(json.loads(task.result_json.read_text(encoding="utf-8")))


def show_answer(target: Task | TaskGraph | AgentResult | Path | str) -> Any:
    """Render ``outputs/answer.md`` as Markdown."""
    task = _resolve_task(target)
    if not task.answer_md.exists():
        return _md(f"_No `answer.md` for task `{task.task_id}`._")
    return _md(task.answer_md.read_text(encoding="utf-8"))


def show_events(
    target: Task | TaskGraph | AgentResult | Path | str, *, tail: int | None = 20
) -> Any:
    """Render the JSONL event log as a Markdown table.

    Set ``tail`` to ``None`` to show all events.
    """
    task = _resolve_task(target)
    events = task.event_log().read()
    if tail is not None:
        events = events[-tail:]
    if not events:
        return _md("_No events recorded._")
    rows = ["| Time | Event | Details |", "|---|---|---|"]
    for e in events:
        ts = e.get("ts", "")
        kind = e.get("event", "")
        details = {k: v for k, v in e.items() if k not in {"ts", "event"}}
        rendered = json.dumps(details, default=str)
        # Escape pipe characters that would break the markdown table.
        rendered = rendered.replace("|", "\\|")
        rows.append(f"| `{ts}` | **{kind}** | `{rendered}` |")
    return _md("\n".join(rows))


def show_graph(target: Task | TaskGraph | AgentResult | Path | str) -> Any:
    """Render the task graph (root + descendants) as a Markdown bullet tree."""
    if isinstance(target, TaskGraph):
        graph = target
    else:
        task = _resolve_task(target)
        graph = TaskGraph.load(task.directory)

    lines: list[str] = []

    def _walk(task: Task, depth: int) -> None:
        prefix = "  " * depth + "- "
        status = task.status
        lines.append(
            f"{prefix}**{task.title}** — `{task.status}` "
            f"(`{task.directory.name}`){' ✅' if status == 'success' else ''}"
        )
        for c in task.children:
            _walk(c, depth + 1)

    _walk(graph.root, 0)
    return _md("\n".join(lines))


def show_notebook(target: Task | TaskGraph | AgentResult | Path | str) -> Any:
    """Return a link to the executed notebook inside the task directory.

    In a Jupyter UI this renders as a clickable link.
    """
    task = _resolve_task(target)
    nb = task.executed_notebook if task.executed_notebook.exists() else task.task_notebook
    if not nb.exists():
        return _md(f"_No notebook for task `{task.task_id}`._")
    rel = nb
    safe = html.escape(str(rel))
    return _html(f'<a href="{safe}" target="_blank"><code>{safe}</code></a>')


__all__ = [
    "show_answer",
    "show_events",
    "show_graph",
    "show_manifest",
    "show_notebook",
    "show_result",
    "show_task",
]
