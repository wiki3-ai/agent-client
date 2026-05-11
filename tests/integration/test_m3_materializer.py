"""M3 integration gate: template materialization + parameter injection.

End-to-end test:
1. Materialize a notebook from the ``python-analysis`` template with
   parameters via ``materialize()``.
2. Execute the materialized notebook through the M2 ``NotebookRunner``
   against the real ``python3`` kernel.
3. Assert that:
   - ``metadata.agent_kernel.inputs`` matches the parameters
   - the injected-parameters cell exists and is tagged correctly
   - injected variables are observable in the executed cell outputs
   - a ``notebook.materialized`` provenance event was emitted
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from agent_kernel.models.event import EventType
from agent_kernel.runtime.materializer import get_input_metadata, materialize
from agent_kernel.runtime.notebook_runner import NotebookRunner
from agent_kernel.runtime.parameter_injection import (
    MetadataOnlyInjector,
    PythonInjector,
    inject_parameters,
)
from agent_kernel.runtime.template_registry import list_templates, load_template
from agent_kernel.storage import JSONLEventStore, WorkspaceLayout


@pytest.mark.integration
def test_python_injector_renders_assignments() -> None:
    src = PythonInjector().render({"query": "top anomalies", "limit": 20, "flags": [1, 2]})
    assert 'query = "top anomalies"' in src
    assert "limit = 20" in src
    assert "flags = [1, 2]" in src


@pytest.mark.integration
def test_metadata_only_injector_returns_empty() -> None:
    assert MetadataOnlyInjector().render({"x": 1}) == ""


@pytest.mark.integration
def test_inject_parameters_replaces_existing_injected_cell() -> None:
    nb = load_template("python-analysis")
    inject_parameters(nb, {"query": "first", "limit": 1}, kernel_name="python3")
    inject_parameters(nb, {"query": "second", "limit": 2}, kernel_name="python3")
    injected_cells = [
        c for c in nb.cells if "injected-parameters" in (c.get("metadata", {}).get("tags") or [])
    ]
    assert len(injected_cells) == 1
    assert 'query = "second"' in injected_cells[0].source
    assert nb.metadata["agent_kernel"]["inputs"]["query"] == "second"


@pytest.mark.integration
def test_template_registry_lists_builtin() -> None:
    assert "python-analysis" in list_templates()


@pytest.mark.integration
def test_materialize_and_execute_end_to_end(tmp_path: Path) -> None:
    """The M3 gate: materialize → execute → assert observable parameters."""
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    events = JSONLEventStore(ws.events_dir, fsync=True)

    target = ws.notebooks_dir / "child-0001.ipynb"
    parameters = {"query": "top anomalies", "limit": 7}
    materialize(
        "python-analysis",
        parameters=parameters,
        kernel_name="python3",
        target_path=target,
        task_id="task_MAT",
        events=events,
    )

    # The materialized notebook on disk carries inputs in metadata
    assert get_input_metadata(target) == parameters

    nb_before = nbformat.read(target, as_version=4)
    # Lineage metadata stamped
    ak = nb_before.metadata["agent_kernel"]
    assert ak["task_id"] == "task_MAT"
    assert ak["template_name"] == "python-analysis"
    # Injected-parameters cell sits immediately after the parameters anchor
    cell_tags = [(c.get("metadata", {}).get("tags") or []) for c in nb_before.cells]
    param_idx = next(i for i, t in enumerate(cell_tags) if "parameters" in t)
    injected_idx = next(i for i, t in enumerate(cell_tags) if "injected-parameters" in t)
    assert injected_idx == param_idx + 1

    # notebook.materialized event emitted
    mat_events = [
        e for e in events.list_events() if e.event_type == EventType.notebook_materialized
    ]
    assert len(mat_events) == 1
    assert mat_events[0].payload["template_name"] == "python-analysis"
    assert mat_events[0].payload["parameter_keys"] == ["limit", "query"]
    assert mat_events[0].payload["template_checksum"]

    # Now execute it with the real python3 kernel
    runner = NotebookRunner(events, ws.runs_dir)
    result = runner.run(target, task_id="task_MAT", kernel_name="python3", timeout=30)
    assert result.status == "completed"

    # The injected parameters must be observable in the executed outputs
    executed = nbformat.read(result.executed_notebook_path, as_version=4)
    echo_cell = next(c for c in executed.cells if c.get("id") == "echo")
    stream_text = "".join(
        o.get("text", "") for o in echo_cell.outputs if o.get("output_type") == "stream"
    )
    assert "query='top anomalies'" in stream_text
    assert "limit=7" in stream_text

    result_cell = next(c for c in executed.cells if c.get("id") == "result")
    # The `result` cell evaluates to a dict; its repr should be in execute_result
    text = "".join(
        d.get("text/plain", "")
        for o in result_cell.outputs
        if o.get("output_type") == "execute_result"
        for d in [o.get("data", {})]
    )
    assert "'query': 'top anomalies'" in text
    assert "'limit': 7" in text


@pytest.mark.integration
def test_unknown_kernel_uses_metadata_only_fallback(tmp_path: Path) -> None:
    """Materializing for an unknown kernel must still produce a valid notebook
    with parameters in metadata, just no executable injected cell."""
    target = tmp_path / "fake-kernel.ipynb"
    materialize(
        "python-analysis",
        parameters={"query": "x", "limit": 3},
        kernel_name="fake-lang-9000",
        target_path=target,
        task_id="task_FB",
    )
    nb = nbformat.read(target, as_version=4)
    assert nb.metadata["kernelspec"]["name"] == "fake-lang-9000"
    # No injected-parameters cell since no injector is registered
    has_injected = any(
        "injected-parameters" in (c.get("metadata", {}).get("tags") or []) for c in nb.cells
    )
    assert not has_injected
    # But metadata channel still carries the inputs
    assert nb.metadata["agent_kernel"]["inputs"] == {"query": "x", "limit": 3}
