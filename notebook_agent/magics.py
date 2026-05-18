"""IPython magics for the notebook-native UX.

Loaded with ``%load_ext notebook_agent`` (or automatically via
:func:`notebook_agent.load_ipython_extension`). Registers:

* ``%task <prompt>`` — line magic. Runs ``run_task(prompt)`` and renders the
  result in the notebook.
* ``%%task`` — cell magic. The cell body is the multi-line prompt.

The magics route through the **same** :func:`notebook_agent.run_task`
function the public API exposes, so behaviour is identical.

Flags accepted on the magic line (before the prompt):

* ``--continue`` / ``-c`` — reuse the most recent task as context.
* ``--max-tokens N`` — override the LM token budget for this call.
* ``--temperature F`` — override the LM sampling temperature for this call.

Continuation is supported via the ``--continue`` flag, which reuses the
most-recent :class:`AgentResult` from the IPython user namespace (stored as
``_last_task_result`` after every magic invocation).
"""

from __future__ import annotations

import shlex
from typing import Any

from .agent import AgentResult, run_task
from .dspy_lm import using_client
from .litellm_client import LiteLLMClient
from .notebook_init import get_notebook_config


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


def _parse_line(line: str) -> tuple[bool, dict[str, Any], str]:
    """Parse the magic line.

    Returns ``(continue_flag, llm_overrides, remaining_prompt)``. Unknown
    leading tokens are treated as part of the prompt, so plain prompts like
    ``%task what is 1+1`` still work.
    """
    try:
        tokens = shlex.split(line.strip())
    except ValueError:
        return False, {}, line.strip()

    cont = False
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"--continue", "-c"}:
            cont = True
            i += 1
            continue
        if tok in {"--max-tokens", "--max_tokens"} and i + 1 < len(tokens):
            try:
                overrides["max_tokens"] = int(tokens[i + 1])
                i += 2
                continue
            except ValueError:
                break
        if tok == "--temperature" and i + 1 < len(tokens):
            try:
                overrides["temperature"] = float(tokens[i + 1])
                i += 2
                continue
            except ValueError:
                break
        # First non-flag token: the rest of the line is the prompt.
        break
    prompt = " ".join(tokens[i:]).strip()
    return cont, overrides, prompt


def _llm_from_overrides(overrides: dict[str, Any]) -> LiteLLMClient | None:
    if not overrides:
        return None
    return LiteLLMClient(**overrides)


def _run_with_overrides(prompt: str, *, cf: AgentResult | None, overrides: dict[str, Any]) -> AgentResult:
    """Run a task, applying any LM overrides only for the duration of this call."""
    nb = get_notebook_config()
    common: dict[str, Any] = {
        "continue_from": cf,
        "max_autonomous_turns": nb.max_autonomous_turns,
        "runs_root": nb.runs_root,
    }
    if nb.skill_dirs:
        common["skill_dirs"] = list(nb.skill_dirs)

    client = _llm_from_overrides(overrides)
    if client is None:
        # Use whatever notebook-level client (if any) is already configured
        # on dspy.settings.lm via init_notebook().
        return run_task(prompt, **common)
    # Temporarily switch DSPy's LM, then restore so the override is per-call.
    with using_client(client):
        return run_task(prompt, llm=client, **common)


def task_line_magic(line: str, *, ip: Any | None = None) -> Any:
    """Implementation of ``%task <prompt>``. Exposed for testability."""
    cont_flag, overrides, prompt = _parse_line(line)
    if not prompt:
        return "Usage: %task [--max-tokens N] [--temperature F] [--continue] <prompt>"
    cf = _resolve_continuation(ip, cont_flag)
    result = _run_with_overrides(prompt, cf=cf, overrides=overrides)
    if ip is not None:
        _stash(ip, result)
    return _render(result)


def task_cell_magic(line: str, cell: str, *, ip: Any | None = None) -> Any:
    """Implementation of ``%%task``. The cell body is the prompt."""
    cont_flag, overrides, _ = _parse_line(line)
    prompt = cell.strip()
    if not prompt:
        return "Usage: %%task [--max-tokens N] [--temperature F] [--continue]\\n<multi-line prompt>"
    cf = _resolve_continuation(ip, cont_flag)
    result = _run_with_overrides(prompt, cf=cf, overrides=overrides)
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
