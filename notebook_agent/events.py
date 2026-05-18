"""Append-only JSONL event log (Section 9 of the spec)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._clock import iso_now


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
