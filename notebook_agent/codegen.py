"""Wrap an LLM-generated Python snippet in a Papermill-executable notebook.

The snippet is produced upstream by the DSPy ``NotebookAgentProgram``'s
``code_generator`` predictor (see :mod:`notebook_agent.program`). This module
only knows how to take that snippet and assemble a runnable notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._notebook_io import (
    DEFAULT_KERNEL_NAME,
    make_notebook,
    parameters_cell,
    write_notebook,
    write_result_cell,
)


@dataclass
class GeneratedCode:
    """A single Python snippet produced by the DSPy ``code_generator``."""

    source: str
    request: str
    plan: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "request": self.request, "plan": list(self.plan)}


def validate_snippet(source: str) -> None:
    """Cheap static checks before we hand the code to Papermill.

    Raises :class:`ValueError` (or :class:`SyntaxError`) on bad input.
    """
    if not source or not source.strip():
        raise ValueError("generated snippet is empty")
    compile(source, "<generated>", "exec")
    if "result" not in source:
        raise ValueError("generated snippet does not assign `result`")


def build_generated_notebook(
    request: str,
    code: GeneratedCode,
    output_path: Path | str,
    *,
    plan: list[str] | None = None,
) -> Path:
    """Wrap *code* in a self-contained parameterized notebook.

    The notebook has, in order:

    1. A title/markdown cell describing the task and the plan.
    2. The Papermill ``parameters`` cell.
    3. A setup cell that ensures ``output_dir`` exists.
    4. The generated code cell (tagged ``generated``).
    5. The standard write-result cell that dumps ``result`` to
       ``outputs/result.json``.
    6. A smoke cell that asserts ``result.json`` exists.

    Provenance (the generated source, the plan, the original request) is
    stored on notebook metadata under ``notebook_agent`` so downstream
    consumers can read it without re-parsing cells.
    """
    output_path = Path(output_path)
    plan_md = ""
    plan_items = plan or code.plan
    if plan_items:
        plan_md = "\n\n## Plan\n\n" + "\n".join(f"- [ ] {item}" for item in plan_items)

    setup_src = (
        "import json\n"
        "from pathlib import Path\n"
        "Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
    )
    body_src = (
        f"# --- generated code for: {request!r} ---\n"
        f"{code.source.rstrip()}\n"
        "if not isinstance(result, dict):\n"
        "    result = {'value': result}\n"
    )
    smoke_src = (
        "from pathlib import Path\n"
        "assert (Path(output_dir) / 'result.json').exists(), 'result.json not written'\n"
    )

    cells = [
        ("markdown", f"# Generated task\n\n**Request:** {request}{plan_md}", []),
        parameters_cell([]),
        ("code", setup_src, ["setup"]),
        ("code", body_src, ["generated"]),
        write_result_cell(),
        ("code", smoke_src, ["smoke"]),
    ]
    nb = make_notebook(cells)
    nb.metadata.setdefault("notebook_agent", {})
    nb.metadata["notebook_agent"].update(
        {
            "stage": "generate",
            "request": request,
            "generated": code.to_dict(),
            "plan": list(plan_items),
            "kernel": DEFAULT_KERNEL_NAME,
        }
    )
    return write_notebook(nb, output_path)


__all__ = ["GeneratedCode", "build_generated_notebook", "validate_snippet"]
