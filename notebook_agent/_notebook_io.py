"""Small helpers for constructing notebooks programmatically.

This module is used internally by :mod:`notebook_agent.transform` and by tests.
Building notebooks with :mod:`nbformat` directly keeps us free of any external
template machinery.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# Standard kernel choice: use the current python via ipykernel. Tests register
# this kernelspec via :func:`ensure_python_kernelspec`.
DEFAULT_KERNEL_NAME = "python3"


def make_notebook(cells: list[dict | tuple[str, str] | tuple[str, str, list[str]]]) -> nbformat.NotebookNode:
    """Build a notebook from a list of cell specs.

    Each entry is either a cell dict, or a ``(kind, source[, tags])`` tuple
    where ``kind`` is ``"code"`` or ``"markdown"``.
    """
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "name": DEFAULT_KERNEL_NAME,
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    for spec in cells:
        if isinstance(spec, dict):
            nb.cells.append(spec)
            continue
        kind = spec[0]
        source = spec[1]
        tags = list(spec[2]) if len(spec) > 2 else []
        if kind == "code":
            cell = new_code_cell(source=source)
        elif kind == "markdown":
            cell = new_markdown_cell(source=source)
        else:
            raise ValueError(f"Unknown cell kind: {kind}")
        if tags:
            cell.metadata["tags"] = tags
        nb.cells.append(cell)
    return nb


def write_notebook(nb: nbformat.NotebookNode, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return p


def parameters_cell(extra_lines: list[str] | None = None) -> tuple[str, str, list[str]]:
    """Return a Papermill-tagged ``parameters`` cell.

    Includes the standard parameters defined in Section 12 of the spec.
    """
    src_lines = [
        "# parameters",
        "task_id = None",
        "parent_task_id = None",
        "input_payload = {}",
        'output_dir = "./outputs"',
        'run_dir = "."',
        "budget = {}",
    ]
    if extra_lines:
        src_lines.extend(extra_lines)
    return ("code", "\n".join(src_lines), ["parameters"])


def write_result_cell(result_expr: str = "result") -> tuple[str, str, list[str]]:
    """Standard result-writing cell that dumps ``result`` to ``outputs/result.json``."""
    src = (
        "import json, os\n"
        "from pathlib import Path\n"
        f"_output_dir = Path(output_dir)\n"
        "_output_dir.mkdir(parents=True, exist_ok=True)\n"
        f"_payload = {result_expr}\n"
        '(_output_dir / "result.json").write_text(json.dumps(_payload, indent=2))\n'
    )
    return ("code", src, ["write_result"])
