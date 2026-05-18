"""Papermill notebook execution wrapper (Section 14.4 of the spec)."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nbformat
import papermill as pm

from ._clock import iso_now
from .budget import BudgetTracker
from .events import EventLog


@dataclass
class NotebookExecutionResult:
    """Structured result of a single notebook execution."""

    success: bool
    notebook_path: Path
    executed_path: Path
    parameters: dict[str, Any]
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    started_at: str = ""
    finished_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "notebook_path": str(self.notebook_path),
            "executed_path": str(self.executed_path),
            "parameters": self.parameters,
            "result": self.result,
            "error": self.error,
            "stdout_path": str(self.stdout_path) if self.stdout_path else None,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            **self.extra,
        }


def execute_notebook(
    notebook_path: Path | str,
    *,
    parameters: dict[str, Any] | None = None,
    output_path: Path | str | None = None,
    run_dir: Path | str | None = None,
    result_json: Path | str | None = None,
    logs_dir: Path | str | None = None,
    event_log: EventLog | None = None,
    budget: BudgetTracker | None = None,
    kernel_name: str | None = None,
    extra_papermill_kwargs: dict[str, Any] | None = None,
) -> NotebookExecutionResult:
    """Execute a parameterized notebook using Papermill.

    The wrapper:
    - charges 1 unit against ``notebook_executions`` budget *before* running
      (raises :class:`BudgetExhaustedError` if exhausted);
    - writes stdout/stderr capture from notebook cells into ``logs_dir``;
    - records ``notebook_execution_started`` / ``_finished`` / ``_failed`` events;
    - tries to load ``outputs/result.json`` (or the path supplied) as the result;
    - returns a :class:`NotebookExecutionResult` capturing success/failure.
    """
    notebook_path = Path(notebook_path)
    if output_path is None:
        output_path = notebook_path.with_name(
            notebook_path.stem.replace("task", "executed")
            if "task" in notebook_path.stem
            else notebook_path.stem + ".executed"
        ).with_suffix(".ipynb")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = dict(parameters or {})
    if run_dir is not None:
        run_dir = Path(run_dir)
        parameters.setdefault("run_dir", str(run_dir))
        if result_json is None:
            result_json = run_dir / "outputs" / "result.json"
        if logs_dir is None:
            logs_dir = run_dir / "logs"
    if result_json is not None:
        result_json = Path(result_json)
        parameters.setdefault("output_dir", str(result_json.parent))
    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

    # Budget check / charge BEFORE doing real work.
    if budget is not None:
        budget.spend("notebook_executions", 1)

    started_at = iso_now()
    if event_log is not None:
        event_log.append(
            "notebook_execution_started",
            notebook=str(notebook_path),
            parameters=parameters,
        )

    kwargs: dict[str, Any] = {
        "input_path": str(notebook_path),
        "output_path": str(output_path),
        "parameters": parameters,
        "progress_bar": False,
        "log_output": False,
        "report_mode": False,
    }
    if kernel_name:
        kwargs["kernel_name"] = kernel_name
    if extra_papermill_kwargs:
        kwargs.update(extra_papermill_kwargs)

    error: dict[str, Any] | None = None
    try:
        pm.execute_notebook(**kwargs)
        success = True
    except Exception as exc:  # noqa: BLE001 - we want any exception captured
        success = False
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    finished_at = iso_now()

    # Capture stdout/stderr from executed cells.
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    if logs_dir is not None and output_path.exists():
        try:
            stdout_text, stderr_text, cell_error = _extract_streams_and_error(output_path)
        except Exception:  # nbformat parsing failure should not blow up the wrapper
            stdout_text, stderr_text, cell_error = "", "", None
        if stdout_text:
            stdout_path = logs_dir / "stdout.log"
            stdout_path.write_text(stdout_text, encoding="utf-8")
        if stderr_text:
            stderr_path = logs_dir / "stderr.log"
            stderr_path.write_text(stderr_text, encoding="utf-8")
        if not success and error is not None and cell_error and not error.get("cell_error"):
            error["cell_error"] = cell_error

    # Attempt to read result.json (only if execution succeeded - failed runs may
    # have written partial output but we still try).
    result_payload: dict[str, Any] | None = None
    if result_json is not None and result_json.exists():
        try:
            result_payload = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result_payload = None

    if event_log is not None:
        if success:
            event_log.append(
                "notebook_execution_finished",
                notebook=str(notebook_path),
                success=True,
            )
        else:
            event_log.append(
                "notebook_execution_failed",
                notebook=str(notebook_path),
                error=(error or {}).get("type"),
                message=(error or {}).get("message"),
            )
            event_log.append(
                "notebook_execution_finished",
                notebook=str(notebook_path),
                success=False,
            )

    return NotebookExecutionResult(
        success=success,
        notebook_path=notebook_path,
        executed_path=output_path,
        parameters=parameters,
        result=result_payload,
        error=error,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=started_at,
        finished_at=finished_at,
    )


def _extract_streams_and_error(executed_notebook: Path) -> tuple[str, str, dict[str, Any] | None]:
    """Pull stdout/stderr text and any cell error from an executed notebook."""
    nb = nbformat.read(str(executed_notebook), as_version=4)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    cell_error: dict[str, Any] | None = None
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []) or []:
            otype = out.get("output_type")
            if otype == "stream":
                if out.get("name") == "stderr":
                    stderr_parts.append(out.get("text", ""))
                else:
                    stdout_parts.append(out.get("text", ""))
            elif otype == "error" and cell_error is None:
                cell_error = {
                    "ename": out.get("ename"),
                    "evalue": out.get("evalue"),
                    "traceback": out.get("traceback", []),
                }
    return ("".join(stdout_parts), "".join(stderr_parts), cell_error)
