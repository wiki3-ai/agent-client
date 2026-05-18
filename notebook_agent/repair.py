"""Notebook repair loop (Section 14.8, Milestone 7).

Implements deterministic repairs for a small set of known failure modes plus
an optional LiteLLM-backed repair path. Repairs are represented as child tasks
under the failed task.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat

from .budget import BudgetTracker
from .events import EventLog
from .notebook_exec import NotebookExecutionResult, execute_notebook
from .task_graph import Task

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


@dataclass
class FailureDiagnosis:
    kind: str
    detail: str
    payload: dict[str, Any]


_MISSING_DIR_PAT = re.compile(
    r"\[Errno 2\] No such file or directory: ['\"]([^'\"]+)['\"]"
)
_NAME_ERROR_PAT = re.compile(r"NameError: name ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"] is not defined")
_MODULE_NOT_FOUND_PAT = re.compile(r"ModuleNotFoundError: No module named ['\"]([A-Za-z0-9_.]+)['\"]")


def diagnose_failure(result: NotebookExecutionResult) -> FailureDiagnosis | None:
    """Look at a failed :class:`NotebookExecutionResult` and classify the error."""
    if result.success or result.error is None:
        return None
    err = result.error
    text = " ".join(
        str(x)
        for x in (
            err.get("type", ""),
            err.get("message", ""),
            err.get("traceback", ""),
            json.dumps((err.get("cell_error") or {}).get("traceback") or []),
            (err.get("cell_error") or {}).get("evalue", "") or "",
        )
    )
    m = _MISSING_DIR_PAT.search(text)
    if m:
        return FailureDiagnosis(kind="missing_output_dir", detail=m.group(1), payload={"path": m.group(1)})
    m = _NAME_ERROR_PAT.search(text)
    if m:
        return FailureDiagnosis(kind="undefined_name", detail=m.group(1), payload={"name": m.group(1)})
    m = _MODULE_NOT_FOUND_PAT.search(text)
    if m:
        return FailureDiagnosis(kind="missing_import", detail=m.group(1), payload={"module": m.group(1)})
    return FailureDiagnosis(kind="unknown", detail=(err.get("message") or "").splitlines()[0][:200], payload=err)


# ---------------------------------------------------------------------------
# Deterministic patches
# ---------------------------------------------------------------------------


_DIR_FIX_PREFIX = "# notebook-agent auto-repair: ensure output directory exists\n"


def _patch_notebook(notebook_path: Path, diagnosis: FailureDiagnosis) -> bool:
    """Apply an in-place deterministic patch. Returns True if patched."""
    nb = nbformat.read(str(notebook_path), as_version=4)
    patched = False

    if diagnosis.kind == "missing_output_dir":
        fix = (
            _DIR_FIX_PREFIX
            + "from pathlib import Path as _P\n"
            f"_P({diagnosis.payload['path']!r}).parent.mkdir(parents=True, exist_ok=True)\n"
            f"_P({diagnosis.payload['path']!r}).mkdir(parents=True, exist_ok=True) if not _P({diagnosis.payload['path']!r}).suffix else None\n"
        )
        # Prepend a new code cell after the parameters cell (or at the top).
        insert_at = 0
        for i, c in enumerate(nb.cells):
            if c.get("cell_type") == "code" and "parameters" in (c.metadata.get("tags") or []):
                insert_at = i + 1
                break
        new_cell = nbformat.v4.new_code_cell(source=fix)
        new_cell.metadata["tags"] = ["auto_repair"]
        nb.cells.insert(insert_at, new_cell)
        patched = True
    elif diagnosis.kind == "missing_import":
        module = diagnosis.payload["module"]
        fix = f"{_DIR_FIX_PREFIX}import {module}  # noqa: F401\n"
        new_cell = nbformat.v4.new_code_cell(source=fix)
        new_cell.metadata["tags"] = ["auto_repair"]
        # Insert right after parameters cell.
        insert_at = 0
        for i, c in enumerate(nb.cells):
            if c.get("cell_type") == "code" and "parameters" in (c.metadata.get("tags") or []):
                insert_at = i + 1
                break
        nb.cells.insert(insert_at, new_cell)
        patched = True

    if patched:
        with notebook_path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    return patched


# ---------------------------------------------------------------------------
# Repair entry point
# ---------------------------------------------------------------------------


@dataclass
class RepairOutcome:
    repaired: bool
    diagnosis: FailureDiagnosis | None
    repair_task: Task | None
    repaired_result: NotebookExecutionResult | None
    strategy: str  # "deterministic", "llm", "none"


def repair_and_rerun(
    parent_task: Task,
    failed_result: NotebookExecutionResult,
    *,
    budget: BudgetTracker,
    parameters: dict[str, Any] | None = None,
    program: Any | None = None,
) -> RepairOutcome:
    """Attempt to repair a failed notebook execution and re-run it.

    Repairs are deterministic: the diagnosis identifies one of a known set of
    failure shapes (missing dir, undefined name, missing module, …) and the
    corresponding patcher edits the notebook. If a ``program`` (a
    :class:`notebook_agent.program.NotebookAgentProgram`) is supplied and the
    failure is an ``undefined_name``, the program's ``repairer`` predictor is
    consulted for a one-line fix.
    """
    diagnosis = diagnose_failure(failed_result)
    parent_log = parent_task.event_log()
    parent_log.append(
        "repair_started",
        notebook=str(failed_result.notebook_path),
        diagnosis=(diagnosis.kind if diagnosis else None),
    )

    if diagnosis is None:
        parent_log.append("repair_finished", success=False, reason="no_diagnosis")
        return RepairOutcome(False, None, None, None, "none")

    # Budget for repair.
    if not budget.can_spend("repair_attempts", 1):
        parent_log.append("repair_finished", success=False, reason="budget_exhausted")
        return RepairOutcome(False, diagnosis, None, None, "none")
    budget.spend("repair_attempts", 1)

    # Create child task representing the repair.
    repair_task = parent_task.create_child(
        title=f"Repair: {diagnosis.kind}",
        request=f"Repair failed notebook ({diagnosis.kind}: {diagnosis.detail}).",
        parameters={"original_notebook": str(failed_result.notebook_path)},
    )
    repair_task.update_status("running")

    # Copy notebook into the repair task for an audit trail.
    repaired_nb_path = repair_task.task_notebook
    repaired_nb_path.write_bytes(failed_result.notebook_path.read_bytes())

    strategy = "deterministic"
    patched = _patch_notebook(repaired_nb_path, diagnosis)
    if not patched and program is not None:
        try:
            patched = _program_patch_notebook(repaired_nb_path, diagnosis, program)
            strategy = "dspy" if patched else strategy
        except Exception:  # noqa: BLE001
            patched = False
    if not patched:
        repair_task.update_status("failed")
        parent_log.append("repair_finished", success=False, reason="no_patch_strategy")
        return RepairOutcome(False, diagnosis, repair_task, None, "none")

    # Re-run the patched notebook.
    rerun_log = EventLog(repair_task.events_log)
    rerun = execute_notebook(
        repaired_nb_path,
        parameters=parameters or failed_result.parameters,
        output_path=repair_task.executed_notebook,
        run_dir=repair_task.directory,
        event_log=rerun_log,
        budget=budget,
    )
    # Update repair task status + manifest.
    repair_task.update_status("success" if rerun.success else "failed")
    m = repair_task.read_manifest()
    m.update(
        {
            "stage_used": "repair",
            "outputs": {"executed_notebook": str(repair_task.executed_notebook)},
            "diagnosis": {"kind": diagnosis.kind, "detail": diagnosis.detail},
            "strategy": strategy,
            "initial_failure": {"type": (failed_result.error or {}).get("type"), "message": (failed_result.error or {}).get("message")},
            "repaired_success": rerun.success,
        }
    )
    repair_task.write_manifest(m)

    parent_log.append(
        "repair_finished",
        success=rerun.success,
        strategy=strategy,
        repair_task_id=repair_task.task_id,
    )

    # Bubble repaired outputs up into parent's manifest record.
    parent_manifest = parent_task.read_manifest()
    parent_manifest.setdefault("repairs", []).append(
        {
            "repair_task_id": repair_task.task_id,
            "strategy": strategy,
            "diagnosis": {"kind": diagnosis.kind, "detail": diagnosis.detail},
            "success": rerun.success,
        }
    )
    parent_task.write_manifest(parent_manifest)

    return RepairOutcome(
        repaired=rerun.success,
        diagnosis=diagnosis,
        repair_task=repair_task,
        repaired_result=rerun,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# LLM-backed patch
# ---------------------------------------------------------------------------


def _program_patch_notebook(
    notebook_path: Path,
    diagnosis: FailureDiagnosis,
    program: Any,
) -> bool:
    """Ask the DSPy ``repairer`` predictor for a one-line fix.

    Conservative: only applied to ``undefined_name`` failures, where the
    suggested fix must be a simple assignment to the missing name. Any other
    shape is refused.
    """
    if diagnosis.kind != "undefined_name":
        return False
    name = diagnosis.payload["name"]
    error_text = (
        f"NameError: name '{name}' is not defined. "
        f"Provide a single Python statement that defines `{name}` "
        "with a reasonable default value."
    )
    fix = (program.repair(error_text) or "").strip().strip("`")
    if not fix:
        return False
    if not re.match(rf"^\s*{re.escape(name)}\s*=", fix):
        return False

    nb = nbformat.read(str(notebook_path), as_version=4)
    cell = nbformat.v4.new_code_cell(source=f"# notebook-agent dspy-repair\n{fix}\n")
    cell.metadata["tags"] = ["auto_repair", "dspy_repair"]
    insert_at = 0
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") == "code" and "parameters" in (c.metadata.get("tags") or []):
            insert_at = i + 1
            break
    nb.cells.insert(insert_at, cell)
    with notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return True
