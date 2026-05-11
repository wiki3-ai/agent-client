#!/usr/bin/env python3
"""Replay a JSONL trace and reconstruct ledger state.

Usage:
    python scripts/replay.py <workspace>            # reconstruct from all JSONL files
    python scripts/replay.py <events.jsonl> ...     # replay specific files

Prints a single JSON document to stdout:
    {
      "task_count": N,
      "tasks": [ <TaskSpec.model_dump>, ... ],
      "event_count": M,
      "sha256": "...",
    }

The ``sha256`` is the digest of the canonical line-sorted JSONL content,
used by the M9 regression-replay test to assert bit-exact reproduction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from agent_kernel.runtime.reconstruct import reconstruct_tasks
from agent_kernel.storage import JSONLEventStore


def _iter_paths(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            # Workspace layout: workspace/.agent_kernel/events/*.jsonl
            candidates = list((p / ".agent_kernel" / "events").glob("*.jsonl")) or list(
                p.glob("*.jsonl")
            )
            paths.extend(sorted(candidates))
        else:
            paths.append(p)
    return paths


def replay(paths: list[Path]) -> dict:
    if not paths:
        raise SystemExit("usage: replay.py <workspace_or_jsonl> [...]")
    # Build a temporary JSONLEventStore-compatible directory by passing the
    # parent of the first file. Easier: iterate the events ourselves.
    parent_dirs = {p.parent for p in paths}
    if len(parent_dirs) != 1:
        raise SystemExit(f"all JSONL files must be in the same directory; got {parent_dirs}")
    store = JSONLEventStore(next(iter(parent_dirs)))
    events = list(store.iter_events())
    tasks = reconstruct_tasks(events)

    # Stable hash of the redacted JSONL content (line-sorted by ts then event_id).
    sorted_events = sorted(events, key=lambda e: (e.ts, e.event_id))
    canon = "\n".join(
        json.dumps(
            json.loads(e.model_dump_json(exclude_none=False)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for e in sorted_events
    )
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()

    return {
        "task_count": len(tasks),
        "tasks": [t.model_dump(mode="json") for t in tasks.values()],
        "event_count": len(events),
        "sha256": digest,
    }


def main(argv: list[str]) -> int:
    result = replay(_iter_paths(argv[1:]))
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
