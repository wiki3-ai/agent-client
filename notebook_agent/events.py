"""Append-only JSONL event log (Section 9 of the spec).

In addition to the on-disk JSONL stream, this module exposes a tiny
process-local **subscription bus**: callers (the IPython magic, tests,
external observers) can register listeners via :func:`subscribe` and
receive every event the moment it is appended. This is the mechanism the
``%task`` magic uses to stream live progress into the user's notebook
output cell — without it, a long LM call looks like a hang.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ._clock import iso_now

# ---- process-local event subscription bus ----

EventListener = Callable[[dict[str, Any]], None]
_LISTENERS: list[EventListener] = []


def subscribe(fn: EventListener) -> EventListener:
    """Register a listener for every event appended to any :class:`EventLog`.

    Returns the same callable so it can be used as a decorator.
    """
    if fn not in _LISTENERS:
        _LISTENERS.append(fn)
    return fn


def unsubscribe(fn: EventListener) -> None:
    """Remove a previously-registered listener. Silently ignores unknown."""
    try:
        _LISTENERS.remove(fn)
    except ValueError:
        pass


def _broadcast(event: dict[str, Any]) -> None:
    # Defensive: listener errors must never break the writer.
    for fn in list(_LISTENERS):
        try:
            fn(event)
        except Exception:  # noqa: BLE001
            pass


class EventLog:
    """Append-only JSONL event log writer/reader.

    Each call to :meth:`append` writes a single JSON object on its own line
    with an ``ts`` field added if not provided.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file if it does not yet exist so that consumers can tail it
        # right after a task is created.
        self.path.touch(exist_ok=True)

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append a single event. Returns the full event dict that was written."""
        payload: dict[str, Any] = {"ts": iso_now(), "event": event}
        payload.update(fields)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=_json_default) + "\n")
        _broadcast(payload)
        return payload

    def read(self) -> list[dict[str, Any]]:
        """Read all events from disk. Lines that fail to parse are skipped."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "__fspath__"):
        return str(value)
    return str(value)
