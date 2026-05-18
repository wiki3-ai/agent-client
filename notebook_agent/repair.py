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
from .litellm_client import LiteLLMClient, LLMUnavailableError
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
    llm: LiteLLMClient | None = None,
) -> RepairOutcome:
    """Attempt to repair a failed notebook execution and re-run it.

    The repair is represented as a child task under ``parent_task``.
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
    if not patched and llm is not None:
        try:
            patched = _llm_patch_notebook(repaired_nb_path, diagnosis, failed_result, llm)
            strategy = "llm" if patched else strategy
        except LLMUnavailableError:
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


def _llm_patch_notebook(
    notebook_path: Path,
    diagnosis: FailureDiagnosis,
    failed_result: NotebookExecutionResult,
    llm: LiteLLMClient,
) -> bool:
    """Ask the LLM for a one-line Python statement to prepend that resolves the error.

    Conservative: only applied for the ``undefined_name`` failure class, where
    the suggested fix is a definition of the missing name. The LLM is expected
    to return *only* a Python expression like ``x = 0``.
    """
    if diagnosis.kind != "undefined_name":
        return False
    name = diagnosis.payload["name"]
    prompt = (
        f"A Jupyter notebook cell failed with NameError: name '{name}' is not defined.\n"
        f"Provide a single Python statement that defines `{name}` with a reasonable default value.\n"
        "Respond with only the Python code, no markdown fences."
    )
    if llm.is_fake():
        # Deterministic default when running under the FakeProvider in tests.
        code_text = f"{name} = 0"
    else:
        resp = llm.complete(prompt, max_tokens=64)
        code_text = (resp.text or "").strip().strip("`")
        if not code_text:
            return False
        # Refuse anything that isn't a simple assignment to the missing name.
        if not re.match(rf"^\s*{re.escape(name)}\s*=", code_text):
            return False

    nb = nbformat.read(str(notebook_path), as_version=4)
    cell = nbformat.v4.new_code_cell(source=f"# notebook-agent llm-repair\n{code_text}\n")
    cell.metadata["tags"] = ["auto_repair", "llm_repair"]
    insert_at = 0
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") == "code" and "parameters" in (c.metadata.get("tags") or []):
            insert_at = i + 1
            break
    nb.cells.insert(insert_at, cell)
    with notebook_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return True
