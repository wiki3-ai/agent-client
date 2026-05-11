"""Notebook materializer: produce a runnable notebook from a template + parameters.

The materializer:
- loads the template via ``template_registry.load_template``
- sets ``metadata.kernelspec`` and ``metadata.language_info`` for the target kernel
- writes ``metadata.agent_kernel.{task_id, parent_task_id, template_name, ...}``
- injects parameters via the two-channel ``parameter_injection`` module
- writes the materialized notebook to ``target_path``
- emits a ``notebook.materialized`` provenance event

All durable fields are mirrored into notebook metadata so the executed
notebook itself remains a self-describing artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode

from agent_kernel.models.event import EventType, ProvenanceEvent
from agent_kernel.runtime.parameter_injection import inject_parameters
from agent_kernel.runtime.template_registry import load_template
from agent_kernel.storage.jsonl_store import JSONLEventStore
from agent_kernel.util import new_id, now_iso


def materialize(
    template: str,
    *,
    parameters: dict[str, Any],
    kernel_name: str,
    target_path: str | Path,
    task_id: str,
    parent_task_id: str | None = None,
    spawn_index: int | None = None,
    decision_event_id: str | None = None,
    depth: int = 0,
    policy_profile: str = "local-dev",
    events: JSONLEventStore | None = None,
) -> Path:
    """Materialize a notebook on disk and return its path."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    nb: NotebookNode = load_template(template)

    # Set kernel metadata for the target kernel
    nb.metadata["kernelspec"] = {"name": kernel_name, "display_name": kernel_name}
    if kernel_name in ("python3", "python"):
        nb.metadata["language_info"] = {"name": "python"}

    # Inject parameters (both channels)
    inject_parameters(nb, parameters, kernel_name=kernel_name)

    # Stamp lineage metadata
    ak = nb.metadata.setdefault("agent_kernel", {})
    ak.update(
        {
            "task_id": task_id,
            "parent_task_id": parent_task_id,
            "template_name": template,
            "policy_profile": policy_profile,
            "spawn": {
                "depth": depth,
                "spawn_index": spawn_index,
                "decision_event_id": decision_event_id,
            },
        }
    )

    nbformat.write(nb, target_path)

    # Provenance
    if events is not None:
        checksum = _checksum(target_path)
        events.append(
            ProvenanceEvent(
                event_id=new_id("evt"),
                ts=now_iso(),
                event_type=EventType.notebook_materialized,
                task_id=task_id,
                parent_task_id=parent_task_id,
                notebook_path=str(target_path),
                kernel_name=kernel_name,
                payload={
                    "template_name": template,
                    "template_checksum": checksum,
                    "parameter_keys": sorted(parameters.keys()),
                    "depth": depth,
                    "spawn_index": spawn_index,
                },
            )
        )

    return target_path


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def get_input_metadata(nb_path: Path) -> dict[str, Any]:
    """Read ``metadata.agent_kernel.inputs`` from a notebook on disk."""
    nb = nbformat.read(nb_path, as_version=4)
    ak = nb.metadata.get("agent_kernel") or {}
    inputs = ak.get("inputs") or {}
    # Ensure JSON-serializable values
    return json.loads(json.dumps(inputs))
