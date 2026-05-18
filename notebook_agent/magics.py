"""IPython magics for the notebook-native UX.

Loaded with ``%load_ext notebook_agent`` (or automatically via
:func:`notebook_agent.load_ipython_extension`). Registers:

* ``%task <prompt>`` — line magic. Runs ``run_task(prompt)`` and renders the
  result in the notebook.
* ``%%task`` — cell magic. The cell body is the multi-line prompt.

The magics route through the **same** :func:`notebook_agent.run_task`
function the public API exposes, so behaviour is identical.

Continuation is supported via the ``--continue`` flag, which reuses the
most-recent :class:`AgentResult` from the IPython user namespace (stored as
``_last_task_result`` after every magic invocation).
"""

from __future__ import annotations

from typing import Any

from .agent import AgentResult, run_task


def _render(result: AgentResult) -> Any:
    """Try to produce a rich notebook display, fall back to plain text."""
    try:
        from .display import show_task

        return show_task(result)
    except Exception:
        return result.answer or "(no answer)"


def _stash(ip: Any, result: AgentResult) -> None:
    """Stash the result in the user namespace for continuation."""
    try:
        ip.user_ns["_last_task_result"] = result
    except Exception:
        pass


def _resolve_continuation(ip: Any, flag: bool) -> AgentResult | None:
    if not flag:
        return None
    return ip.user_ns.get("_last_task_result") if ip is not None else None


def _parse_line(line: str) -> tuple[bool, str]:
    """Return ``(continue_flag, remaining_prompt)``."""
    parts = line.strip().split(maxsplit=1)
    if parts and parts[0] in {"--continue", "-c"}:
        return True, parts[1] if len(parts) > 1 else ""
    return False, line.strip()


def task_line_magic(line: str, *, ip: Any | None = None) -> Any:
    """Implementation of ``%task <prompt>``. Exposed for testability."""
    cont_flag, prompt = _parse_line(line)
    if not prompt:
        return "Usage: %task <prompt>  (or %task --continue <feedback>)"
    cf = _resolve_continuation(ip, cont_flag)
    result = run_task(prompt, continue_from=cf)
    if ip is not None:
        _stash(ip, result)
    return _render(result)


def task_cell_magic(line: str, cell: str, *, ip: Any | None = None) -> Any:
    """Implementation of ``%%task``. The cell body is the prompt."""
    cont_flag, _ = _parse_line(line)
    prompt = cell.strip()
    if not prompt:
        return "Usage: %%task\\n<multi-line prompt>"
    cf = _resolve_continuation(ip, cont_flag)
    result = run_task(prompt, continue_from=cf)
    if ip is not None:
        _stash(ip, result)
    return _render(result)


def load_ipython_extension(ipython: Any) -> None:
    """Entry point for ``%load_ext notebook_agent``."""
    from IPython.core.magic import (  # type: ignore[import-not-found]
        Magics,
        cell_magic,
        line_magic,
        magics_class,
    )

    @magics_class
    class _TaskMagics(Magics):
        @line_magic("task")
        def _line(self, line: str) -> Any:
            return task_line_magic(line, ip=self.shell)

        @cell_magic("task")
        def _cell(self, line: str, cell: str) -> Any:
            return task_cell_magic(line, cell, ip=self.shell)

    ipython.register_magics(_TaskMagics)


def unload_ipython_extension(ipython: Any) -> None:  # pragma: no cover - rarely called
    # IPython doesn't provide a clean unregister API for magics; this is a stub.
    pass


__all__ = [
    "load_ipython_extension",
    "task_cell_magic",
    "task_line_magic",
    "unload_ipython_extension",
]
