"""Notebook runner: execute a notebook with nbclient and emit provenance.

This module owns the boundary between agent-kernel and ``nbclient``:
- read the notebook with ``nbformat``
- wire every nbclient hook to ``ProvenanceEvent`` emission on a
  ``JSONLEventStore``
- write the executed notebook + stdout/stderr to ``runs/<run_id>/``
- raise ``NotebookRunFailed`` on any cell error or timeout, with the
  appropriate ``cell.execution.error`` / ``task.failed`` events already
  written

The runner is intentionally **synchronous** at the public-API boundary; it
internally calls ``NotebookClient.execute()`` which manages its own asyncio
loop.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from nbformat import NotebookNode

from agent_kernel.models.event import EventStatus, EventType, ProvenanceEvent
from agent_kernel.storage.jsonl_store import JSONLEventStore
from agent_kernel.util import new_id, now_iso


class NotebookRunFailed(RuntimeError):
    """Raised when a notebook execution ends in cell error or timeout.

    All relevant provenance events are emitted **before** this is raised so
    callers can inspect the ledger to determine the failure cause.
    """

    def __init__(self, run_id: str, reason: str, cell_id: str | None = None) -> None:
        super().__init__(f"run {run_id} failed: {reason}")
        self.run_id = run_id
        self.reason = reason
        self.cell_id = cell_id


@dataclass(frozen=True)
class RunResult:
    """The outcome of one notebook run."""

    run_id: str
    task_id: str
    executed_notebook_path: Path
    run_dir: Path
    status: str  # "completed" | "failed"
    duration_ms: int
    cell_count: int
    error: dict[str, Any] | None = None


class NotebookRunner:
    """Execute a notebook with provenance emission.

    Parameters
    ----------
    events:
        Where to append provenance events. The runner does not own the file
        lifecycle; the caller is responsible for keeping the store alive.
    runs_dir:
        Filesystem directory that will receive ``<run_id>/`` subdirectories.
    """

    def __init__(self, events: JSONLEventStore, runs_dir: str | Path) -> None:
        self._events = events
        self._runs_dir = Path(runs_dir)
        self._runs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ run
    def run(
        self,
        notebook_path: str | Path,
        *,
        task_id: str,
        kernel_name: str = "python3",
        timeout: int = 60,
        startup_timeout: int = 30,
        cwd: str | Path | None = None,
    ) -> RunResult:
        notebook_path = Path(notebook_path)
        if not notebook_path.exists():
            raise FileNotFoundError(notebook_path)

        nb: NotebookNode = nbformat.read(notebook_path, as_version=4)

        run_id = new_id("run")
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        executed_path = run_dir / "executed.ipynb"

        # Emit notebook.execution.started
        self._emit(
            EventType.notebook_execution_started,
            task_id=task_id,
            run_id=run_id,
            notebook_path=str(notebook_path),
            kernel_name=kernel_name,
            payload={
                "cwd": str(cwd) if cwd else None,
                "timeout": timeout,
                "startup_timeout": startup_timeout,
                "cell_count": len(nb.cells),
            },
        )

        # Wire hooks. Each hook captures task/run/notebook context via closures.
        cell_start_ts: dict[int, float] = {}

        def _hash(text: str) -> str:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        def on_cell_start(cell: NotebookNode, cell_index: int) -> None:
            cell_start_ts[cell_index] = time.monotonic()
            self._emit(
                EventType.cell_execution_started,
                task_id=task_id,
                run_id=run_id,
                notebook_path=str(notebook_path),
                kernel_name=kernel_name,
                cell_id=cell.get("id"),
                payload={
                    "cell_index": cell_index,
                    "cell_type": cell.get("cell_type"),
                    "cell_hash": _hash(cell.get("source", "")),
                },
            )

        def on_cell_complete(cell: NotebookNode, cell_index: int) -> None:
            start = cell_start_ts.pop(cell_index, None)
            duration_ms = int((time.monotonic() - start) * 1000) if start else None
            outputs = cell.get("outputs", []) or []
            self._emit(
                EventType.cell_execution_completed,
                task_id=task_id,
                run_id=run_id,
                notebook_path=str(notebook_path),
                kernel_name=kernel_name,
                cell_id=cell.get("id"),
                payload={
                    "cell_index": cell_index,
                    "duration_ms": duration_ms,
                    "output_count": len(outputs),
                    "output_types": [o.get("output_type") for o in outputs],
                },
            )

        def on_cell_error(
            cell: NotebookNode, cell_index: int, execute_reply: dict[str, Any]
        ) -> None:
            content = execute_reply.get("content", {}) if execute_reply else {}
            self._emit(
                EventType.cell_execution_error,
                task_id=task_id,
                run_id=run_id,
                notebook_path=str(notebook_path),
                kernel_name=kernel_name,
                cell_id=cell.get("id"),
                status=EventStatus.error,
                error={
                    "ename": content.get("ename"),
                    "evalue": content.get("evalue"),
                    "traceback": content.get("traceback"),
                },
                payload={"cell_index": cell_index},
            )

        client = NotebookClient(
            nb,
            timeout=timeout,
            startup_timeout=startup_timeout,
            kernel_name=kernel_name,
            allow_errors=False,
            record_timing=True,
            resources={"metadata": {"path": str(cwd) if cwd else str(notebook_path.parent)}},
            on_cell_start=on_cell_start,
            on_cell_complete=on_cell_complete,
            on_cell_error=on_cell_error,
        )

        t0 = time.monotonic()
        error: dict[str, Any] | None = None
        status = "completed"
        failed_cell_id: str | None = None
        try:
            client.execute()
        except CellTimeoutError as exc:
            status = "failed"
            error = {"name": "CellTimeoutError", "message": str(exc), "kind": "timeout"}
            failed_cell_id = self._find_failing_cell_id(nb)
        except CellExecutionError as exc:
            status = "failed"
            error = {
                "name": getattr(exc, "ename", "CellExecutionError"),
                "message": getattr(exc, "evalue", str(exc)),
                "kind": "cell_error",
            }
            failed_cell_id = self._find_failing_cell_id(nb)
        finally:
            # Save the (partially) executed notebook regardless of outcome.
            nbformat.write(nb, executed_path)
            # Empty stdout/stderr placeholders for symmetry with the run layout
            (run_dir / "stdout.log").touch(exist_ok=True)
            (run_dir / "stderr.log").touch(exist_ok=True)

        duration_ms = int((time.monotonic() - t0) * 1000)

        # Terminal events
        if status == "completed":
            self._emit(
                EventType.notebook_execution_completed,
                task_id=task_id,
                run_id=run_id,
                notebook_path=str(notebook_path),
                kernel_name=kernel_name,
                payload={"duration_ms": duration_ms},
            )
            self._emit(
                EventType.task_completed,
                task_id=task_id,
                run_id=run_id,
                notebook_path=str(notebook_path),
                kernel_name=kernel_name,
                payload={
                    "executed_notebook_path": str(executed_path),
                    "duration_ms": duration_ms,
                },
            )
        else:
            self._emit(
                EventType.task_failed,
                task_id=task_id,
                run_id=run_id,
                notebook_path=str(notebook_path),
                kernel_name=kernel_name,
                status=EventStatus.error,
                error=error,
                payload={
                    "executed_notebook_path": str(executed_path),
                    "duration_ms": duration_ms,
                    "failed_cell_id": failed_cell_id,
                },
            )

        result = RunResult(
            run_id=run_id,
            task_id=task_id,
            executed_notebook_path=executed_path,
            run_dir=run_dir,
            status=status,
            duration_ms=duration_ms,
            cell_count=len(nb.cells),
            error=error,
        )

        if status == "failed":
            raise NotebookRunFailed(
                run_id, error["name"] if error else "unknown", cell_id=failed_cell_id
            )
        return result

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _find_failing_cell_id(nb: NotebookNode) -> str | None:
        """Return the id of the first code cell whose outputs include an error."""
        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            for out in cell.get("outputs", []) or []:
                if out.get("output_type") == "error":
                    return cell.get("id")
        return None

    def _emit(self, event_type: EventType, **kwargs: Any) -> ProvenanceEvent:
        ev = ProvenanceEvent(
            event_id=new_id("evt"),
            ts=now_iso(),
            event_type=event_type,
            **kwargs,
        )
        self._events.append(ev)
        return ev
