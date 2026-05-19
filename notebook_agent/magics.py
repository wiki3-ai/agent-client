"""IPython magics for the notebook-native UX.

Loaded with ``%load_ext notebook_agent`` (or automatically via
:func:`notebook_agent.load_ipython_extension`). Registers:

* ``%task <prompt>`` — line magic. Runs ``run_task(prompt)`` and renders the
  result in the notebook.
* ``%%task`` — cell magic. The cell body is the multi-line prompt.
* ``%agent_init [key=value …]`` — reconfigure the notebook-wide
  :class:`NotebookConfig` (model, max_tokens, temperature, …). Loading the
  extension already calls :func:`init_notebook` once with env-derived
  defaults, so a bare notebook works without any boilerplate; this magic
  is for the override case.

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

import ast
import shlex
from typing import Any

from .agent import AgentResult, run_task
from .dspy_lm import using_client
from .litellm_client import LiteLLMClient
from .notebook_init import get_notebook_config, init_notebook
from .progress import ProgressRenderer


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
        if tok in {"--timeout", "--request-timeout", "--request_timeout"} and i + 1 < len(tokens):
            try:
                overrides["request_timeout"] = float(tokens[i + 1])
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
    with ProgressRenderer():
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
    """Entry point for ``%load_ext notebook_agent``.

    Auto-initializes the notebook with env-derived defaults so a bare
    notebook can call ``%task`` immediately. Users who want different
    settings can run ``%agent_init key=value …`` (or call
    :func:`notebook_agent.init_notebook` directly with full kwargs).
    """
    from IPython.core.magic import (  # type: ignore[import-not-found]
        Magics,
        cell_magic,
        line_magic,
        magics_class,
    )

    # Default initialization — picks up NOTEBOOK_AGENT_* env vars.
    try:
        init_notebook()
    except Exception as exc:  # pragma: no cover - shouldn't happen
        print(f"notebook_agent: default init failed: {exc!r}")

    @magics_class
    class _TaskMagics(Magics):
        @line_magic("task")
        def _line(self, line: str) -> Any:
            return task_line_magic(line, ip=self.shell)

        @cell_magic("task")
        def _cell(self, line: str, cell: str) -> Any:
            return task_cell_magic(line, cell, ip=self.shell)

        @line_magic("agent_init")
        def _init(self, line: str) -> Any:
            return agent_init_magic(line)

    ipython.register_magics(_TaskMagics)


def agent_init_magic(line: str) -> Any:
    """``%agent_init [key=value …]`` — reconfigure the notebook.

    Each ``key=value`` token is parsed with :func:`ast.literal_eval` (so
    ``model=lm_studio/foo`` is treated as a string, ``max_tokens=32000`` as
    an int, ``temperature=0.7`` as a float, ``skill_dirs=['/x','/y']`` as a
    list). Keys with no recognised init kwarg are ignored. Calling with no
    arguments resets to env-derived defaults.
    """
    kwargs = _parse_kv_pairs(line)
    client = init_notebook(**kwargs)
    cfg = get_notebook_config()
    return {
        "client": client.to_dict(),
        "max_autonomous_turns": cfg.max_autonomous_turns,
        "runs_root": cfg.runs_root,
        "skill_dirs": cfg.skill_dirs,
    }


_INIT_KWARGS = {
    "provider", "model", "base_url", "api_key",
    "max_tokens", "temperature", "request_timeout", "reasoning_effort",
    "max_autonomous_turns", "runs_root", "skill_dirs",
}


def _parse_kv_pairs(line: str) -> dict[str, Any]:
    """Parse a ``%agent_init`` line into kwargs.

    Accepts shell-style tokens, each of the form ``key=value``. Values are
    parsed with :func:`ast.literal_eval` if possible (so quoting works for
    strings, lists, etc.), falling back to the raw string.
    """
    try:
        tokens = shlex.split(line.strip())
    except ValueError:
        tokens = line.strip().split()
    out: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        key, _, raw = tok.partition("=")
        key = key.strip()
        if key not in _INIT_KWARGS:
            continue
        raw = raw.strip()
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def unload_ipython_extension(ipython: Any) -> None:  # pragma: no cover - rarely called
    # IPython doesn't provide a clean unregister API for magics; this is a stub.
    pass


__all__ = [
    "agent_init_magic",
    "load_ipython_extension",
    "task_cell_magic",
    "task_line_magic",
    "unload_ipython_extension",
]
