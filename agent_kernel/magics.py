"""``%agent`` magic dispatcher used by the control kernel.

Magics are intentionally line-oriented, shlex-parsed, and self-documenting.
This module is the single source of truth for the magic surface so the
kernel implementation stays a thin wrapper.

Supported magics::

    %agent help
    %agent policy show
    %agent quota
    %agent task new <notebook_path> [--kernel python3]
    %agent task status <task_id>
    %agent task list
    %agent run <task_id>
    %agent spawn <parent_task_id> <template> [--param key=value]...
                                            [--kernel python3]
    %agent ledger tail [N] [--task <task_id>]
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from agent_kernel.api import AgentKernel
from agent_kernel.models.task import SpawnSpec


class MagicError(ValueError):
    """Raised on malformed ``%agent`` invocations."""


def is_agent_magic(line: str) -> bool:
    return line.lstrip().startswith("%agent")


def dispatch(line: str, ak: AgentKernel) -> str:
    """Run a single ``%agent`` magic line and return a human-readable result.

    Returns a JSON-shaped string for programmatic parsing in tests; the
    kernel echoes it back to the client via ``stream``.
    """
    line = line.lstrip()
    if not line.startswith("%agent"):
        raise MagicError(f"not an agent magic: {line!r}")
    parts = shlex.split(line[len("%agent") :].strip())
    if not parts:
        return _help()

    cmd, args = parts[0], parts[1:]
    handler = _HANDLERS.get(cmd)
    if handler is None:
        raise MagicError(f"unknown agent magic: {cmd!r}; try '%agent help'")
    return handler(args, ak)


# --------------------------------------------------------------- handlers


def _help(_args: list[str] | None = None, _ak: AgentKernel | None = None) -> str:
    return json.dumps({"ok": True, "magics": list(_HANDLERS)})


def _policy(args: list[str], ak: AgentKernel) -> str:
    if not args or args[0] != "show":
        raise MagicError("usage: %agent policy show")
    profile = ak.scheduler.profile
    return json.dumps({"ok": True, "profile": profile.model_dump(mode="json")})


def _quota(_args: list[str], ak: AgentKernel) -> str:
    snap = ak.scheduler._quota_snapshot_locked()
    return json.dumps({"ok": True, "quota": snap.model_dump(mode="json")})


def _task(args: list[str], ak: AgentKernel) -> str:
    if not args:
        raise MagicError("usage: %agent task {new|status|list} [...]")
    sub, rest = args[0], args[1:]
    if sub == "new":
        return _task_new(rest, ak)
    if sub == "status":
        return _task_status(rest, ak)
    if sub == "list":
        return _task_list(rest, ak)
    raise MagicError(f"unknown task subcommand: {sub!r}")


def _task_new(args: list[str], ak: AgentKernel) -> str:
    if not args:
        raise MagicError("usage: %agent task new <notebook_path> [--kernel python3]")
    notebook_path = args[0]
    kernel_name = "python3"
    i = 1
    while i < len(args):
        if args[i] == "--kernel" and i + 1 < len(args):
            kernel_name = args[i + 1]
            i += 2
        else:
            raise MagicError(f"unknown arg: {args[i]!r}")
    task = ak.create_task(notebook_path=notebook_path, kernel_name=kernel_name)
    return json.dumps({"ok": True, "task_id": task.task_id})


def _task_status(args: list[str], ak: AgentKernel) -> str:
    if len(args) != 1:
        raise MagicError("usage: %agent task status <task_id>")
    t = ak.get_task(args[0])
    if t is None:
        return json.dumps({"ok": False, "error": "not_found"})
    return json.dumps({"ok": True, "task": t.model_dump(mode="json")})


def _task_list(_args: list[str], ak: AgentKernel) -> str:
    ids = ak.scheduler.tasks_store.list_ids()
    return json.dumps({"ok": True, "task_ids": ids})


def _run(args: list[str], ak: AgentKernel) -> str:
    if len(args) != 1:
        raise MagicError("usage: %agent run <task_id>")
    final = ak.run_task(args[0])
    return json.dumps({"ok": True, "task_id": final.task_id, "status": final.status.value})


def _spawn(args: list[str], ak: AgentKernel) -> str:
    if len(args) < 2:
        raise MagicError(
            "usage: %agent spawn <parent_task_id> <template> [--param k=v]... [--kernel python3]"
        )
    parent_id = args[0]
    template = args[1]
    parameters: dict[str, Any] = {}
    kernel_name = "python3"
    i = 2
    while i < len(args):
        if args[i] == "--param" and i + 1 < len(args):
            k, _, v = args[i + 1].partition("=")
            # Parse value as JSON when possible, fall back to string.
            try:
                parameters[k] = json.loads(v)
            except json.JSONDecodeError:
                parameters[k] = v
            i += 2
        elif args[i] == "--kernel" and i + 1 < len(args):
            kernel_name = args[i + 1]
            i += 2
        else:
            raise MagicError(f"unknown arg: {args[i]!r}")
    result = ak.spawn_child_task(
        parent_id,
        SpawnSpec(template_name=template, parameters=parameters, kernel_name=kernel_name),
    )
    return json.dumps(
        {
            "ok": result.allowed,
            "reason": result.reason,
            "child_task_id": result.child_task.task_id if result.child_task else None,
        }
    )


def _ledger(args: list[str], ak: AgentKernel) -> str:
    if not args or args[0] != "tail":
        raise MagicError("usage: %agent ledger tail [N] [--task <task_id>]")
    args = args[1:]
    n = 20
    task_id: str | None = None
    i = 0
    while i < len(args):
        if args[i] == "--task" and i + 1 < len(args):
            task_id = args[i + 1]
            i += 2
        else:
            try:
                n = int(args[i])
            except ValueError as exc:
                raise MagicError(f"unknown arg: {args[i]!r}") from exc
            i += 1
    events = ak.list_events(task_id)[-n:]
    return json.dumps(
        {
            "ok": True,
            "events": [
                {"ts": e.ts, "type": e.event_type.value, "task_id": e.task_id} for e in events
            ],
        }
    )


_HANDLERS = {
    "help": _help,
    "policy": _policy,
    "quota": _quota,
    "task": _task,
    "run": _run,
    "spawn": _spawn,
    "ledger": _ledger,
}


# ---------------------------------------------------------- workspace helper


def workspace_from_env() -> Path:
    """Resolve the workspace directory from ``AGENT_KERNEL_WORKSPACE`` env var.

    Falls back to the current working directory if unset.
    """
    import os

    return Path(os.environ.get("AGENT_KERNEL_WORKSPACE", os.getcwd())).resolve()
