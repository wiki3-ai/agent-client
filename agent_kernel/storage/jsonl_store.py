"""Append-only JSONL event store.

Provides a thread-safe writer for ``ProvenanceEvent`` instances. Files are
named ``YYYY-MM-DD.jsonl`` and rolled over at UTC day boundaries. Each line
is a single JSON object terminated by ``\\n``.

Crash safety: writes happen with line-at-a-time ``f.write()`` followed by an
optional ``flush()`` + ``os.fsync()``. Readers tolerate a trailing partial
line (skipped) per JSON Lines convention.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from agent_kernel.models.event import ProvenanceEvent


class JSONLEventStore:
    """Append-only event log with daily file rotation."""

    def __init__(self, events_dir: str | Path, *, fsync: bool = False) -> None:
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._lock = threading.Lock()

    def _path_for(self, ts: str) -> Path:
        # ts is RFC 3339 (e.g. "2026-05-11T05:18:37.919Z"); take date prefix
        date_prefix = ts[:10] if len(ts) >= 10 else datetime.now(UTC).strftime("%Y-%m-%d")
        return self.events_dir / f"{date_prefix}.jsonl"

    def append(self, event: ProvenanceEvent) -> None:
        line = event.model_dump_json(exclude_none=False)
        path = self._path_for(event.ts)
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            if self._fsync:
                f.flush()
                os.fsync(f.fileno())

    def iter_events(self) -> Iterator[ProvenanceEvent]:
        """Yield events in chronological filename order, skipping partial lines."""
        files = sorted(p for p in self.events_dir.glob("*.jsonl") if p.is_file())
        for path in files:
            with open(path, encoding="utf-8") as f:
                for raw in f:
                    if not raw.endswith("\n"):
                        # Tolerate trailing partial line on crash recovery.
                        continue
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        # Skip corrupt lines but do not abort recovery.
                        continue
                    yield ProvenanceEvent.model_validate(obj)

    def list_events(self) -> list[ProvenanceEvent]:
        return list(self.iter_events())
