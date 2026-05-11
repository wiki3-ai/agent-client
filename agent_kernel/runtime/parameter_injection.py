"""Parameter injection: kernel-specific executable bindings for materialized notebooks.

Two-channel model (spec §Per-kernel parameter injection):

1. **Universal metadata channel**: every materialized notebook records
   parameters at ``metadata.agent_kernel.inputs``. This is kernel-agnostic
   and guarantees the parameters travel with the notebook.

2. **Executable binding channel**: if a kernel-specific injector exists, a
   code cell tagged ``["injected-parameters", "agent-kernel"]`` is inserted
   immediately after any cell tagged ``parameters`` (or as the first cell
   if no such cell exists), per the Papermill convention.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from nbformat import NotebookNode
from nbformat.v4 import new_code_cell


class Injector(Protocol):
    """Pluggable per-kernel parameter injector."""

    language: str

    def render(self, parameters: dict[str, Any]) -> str:
        """Return the source text of the injected-parameters cell."""
        ...


class PythonInjector:
    """Render Python-language ``name = json.loads(...)`` assignments.

    Uses JSON literal embedding so any JSON-serializable parameter is safely
    round-tripped without f-string quoting hazards.
    """

    language = "python"

    def render(self, parameters: dict[str, Any]) -> str:
        if not parameters:
            return "# agent-kernel: no parameters\n"
        lines = ["# agent-kernel injected parameters"]
        for name, value in parameters.items():
            if not _is_valid_python_identifier(name):
                raise ValueError(f"parameter name {name!r} is not a valid Python identifier")
            literal = json.dumps(value)
            lines.append(f"{name} = {literal}")
        return "\n".join(lines) + "\n"


class MetadataOnlyInjector:
    """Fallback injector that emits no executable cell.

    Used when no language-specific injector is registered for the target
    kernel. The metadata channel still carries the parameters.
    """

    language = "unknown"

    def render(self, parameters: dict[str, Any]) -> str:
        return ""


def _is_valid_python_identifier(name: str) -> bool:
    return name.isidentifier() and not name.startswith("__")


# --------------------------------------------------------------------- registry

_REGISTRY: dict[str, Injector] = {
    "python3": PythonInjector(),
    "python": PythonInjector(),
}


def register_injector(kernel_name: str, injector: Injector) -> None:
    _REGISTRY[kernel_name] = injector


def get_injector(kernel_name: str) -> Injector:
    return _REGISTRY.get(kernel_name, MetadataOnlyInjector())


# ---------------------------------------------------------------- nb mutation


_INJECTED_TAGS = ["injected-parameters", "agent-kernel"]


def inject_parameters(
    nb: NotebookNode,
    parameters: dict[str, Any],
    *,
    kernel_name: str,
) -> NotebookNode:
    """Inject parameters into ``nb`` in place and return the same object.

    - Always writes ``nb.metadata.agent_kernel.inputs = parameters``.
    - If a kernel-specific injector is registered AND it returns a non-empty
      source, inserts an injected-parameters cell after the first cell
      tagged ``parameters``, or as the first cell if no such cell exists.
    - If a previous ``injected-parameters`` cell already exists (from a
      prior materialization), it is replaced rather than duplicated.
    """
    ak = nb.metadata.setdefault("agent_kernel", {})
    ak["inputs"] = parameters

    injector = get_injector(kernel_name)
    source = injector.render(parameters)
    if not source.strip():
        return nb

    new_cell = new_code_cell(source, id="injected_parameters")
    new_cell.metadata["tags"] = list(_INJECTED_TAGS)

    # Replace existing injected-parameters cell, if any
    for i, cell in enumerate(nb.cells):
        tags = cell.get("metadata", {}).get("tags", []) or []
        if "injected-parameters" in tags and "agent-kernel" in tags:
            nb.cells[i] = new_cell
            return nb

    # Otherwise, insert after the first cell tagged "parameters"
    for i, cell in enumerate(nb.cells):
        tags = cell.get("metadata", {}).get("tags", []) or []
        if "parameters" in tags:
            nb.cells.insert(i + 1, new_cell)
            return nb

    # No anchor — prepend
    nb.cells.insert(0, new_cell)
    return nb
