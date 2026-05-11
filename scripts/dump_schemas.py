"""Dump JSON Schemas for durable Pydantic models to ``docs/schemas/``.

Usage:
    python scripts/dump_schemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_kernel.models import (
    Budget,
    ExecutableCellArtifact,
    PolicyProfile,
    ProvenanceEvent,
    TaskSpec,
)

SCHEMAS = {
    "task.schema.json": TaskSpec,
    "budget.schema.json": Budget,
    "provenance-event.schema.json": ProvenanceEvent,
    "executable-cell-artifact.schema.json": ExecutableCellArtifact,
    "policy-profile.schema.json": PolicyProfile,
}


def main(out_dir: Path | None = None) -> None:
    out = out_dir or Path(__file__).resolve().parent.parent / "docs" / "schemas"
    out.mkdir(parents=True, exist_ok=True)
    for filename, model_cls in SCHEMAS.items():
        schema = model_cls.model_json_schema()
        (out / filename).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {filename}")


if __name__ == "__main__":
    main()
