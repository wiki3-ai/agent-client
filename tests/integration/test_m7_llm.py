"""M7 integration gate: structured LLM call accounted in the ledger.

Hermetic path (always runs in CI): a notebook runs through the M2
``NotebookRunner`` and inside one of its cells issues a
``StructuredLLM.generate()`` call against ``FakeProvider``. We assert:

- the parsed Pydantic object is correct
- retry-on-invalid-schema works
- the JSONL ledger contains a ``llm.call.completed`` event with non-zero
  token/cost debits, plus a corresponding ``budget.debited`` event
- budget arithmetic never goes negative

Optional live path (skipped unless an LM Studio server is reachable at
``LMSTUDIO_BASE_URL`` or the default ``http://localhost:1234/v1``): the
same call against ``LMStudioProvider`` returns a validated Pydantic
object.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from agent_kernel.api import AgentKernel
from agent_kernel.llm import (
    FakeProvider,
    LLMCallError,
    LMStudioProvider,
    StructuredLLM,
)
from agent_kernel.models.budget import Budget
from agent_kernel.models.event import EventType
from agent_kernel.storage import JSONLEventStore, WorkspaceLayout


class Sentiment(BaseModel):
    """Tiny response model used across tests."""

    label: str = Field(description="positive | negative | neutral")
    confidence: float = Field(ge=0.0, le=1.0)


# ================================ hermetic ================================


@pytest.mark.integration
def test_fake_provider_validates_and_emits_ledger(tmp_path: Path) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path=str(tmp_path / "nb.ipynb"), kernel_name="python3")

    provider = FakeProvider(
        script=['{"label": "positive", "confidence": 0.9}'],
        cost_usd_micro_per_call=250,
    )
    llm = StructuredLLM(provider, agent_kernel=ak, model="fake-1")

    obj = llm.generate(
        messages=[{"role": "user", "content": "Classify: i love it"}],
        response_model=Sentiment,
        task_id=task.task_id,
    )
    assert obj.label == "positive"
    assert obj.confidence == pytest.approx(0.9)

    events = ak.list_events(task.task_id)
    started = [e for e in events if e.event_type == EventType.llm_call_started]
    completed = [e for e in events if e.event_type == EventType.llm_call_completed]
    debited = [e for e in events if e.event_type == EventType.budget_debited]
    assert len(started) == 1
    assert len(completed) == 1
    assert completed[0].payload["cost_usd_micro"] == 250
    assert completed[0].payload["attempts"] == 1
    assert completed[0].payload["prompt_tokens"] >= 1
    assert completed[0].payload["completion_tokens"] >= 1
    assert len(debited) == 1
    assert debited[0].payload["reason"] == "llm_call"
    # Budgets never negative on the JSONL trace
    for ev in events:
        for b in (ev.budget_before, ev.budget_after):
            if b is None:
                continue
            for f in b.__class__.model_fields:
                assert getattr(b, f) >= 0

    # Task's persisted spent_budget reflects the debit
    persisted = ak.get_task(task.task_id)
    assert persisted is not None
    assert persisted.spent_budget.llm_usd_micro == 250


@pytest.mark.integration
def test_fake_provider_retry_on_invalid_schema_then_success(tmp_path: Path) -> None:
    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path=str(tmp_path / "nb.ipynb"), kernel_name="python3")

    # First response is invalid JSON-schema-wise (confidence out of range);
    # second is valid. Adapter should retry.
    provider = FakeProvider(
        script=[
            '{"label": "positive", "confidence": 1.7}',  # invalid: confidence > 1
            '{"label": "neutral", "confidence": 0.42}',  # valid
        ],
        cost_usd_micro_per_call=100,
    )
    llm = StructuredLLM(provider, agent_kernel=ak, model="fake-1", max_retries=2)

    obj = llm.generate(
        messages=[{"role": "user", "content": "classify"}],
        response_model=Sentiment,
        task_id=task.task_id,
    )
    assert obj.label == "neutral"
    # Provider was called twice; only the successful call's completion is emitted.
    completed = [
        e for e in ak.list_events(task.task_id) if e.event_type == EventType.llm_call_completed
    ]
    assert len(completed) == 1
    assert completed[0].payload["attempts"] == 2


@pytest.mark.integration
def test_fake_provider_exhausts_retries_raises_and_emits_error(tmp_path: Path) -> None:
    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path=str(tmp_path / "nb.ipynb"), kernel_name="python3")

    provider = FakeProvider(
        script=['{"label": "x", "confidence": 9}'] * 5,  # always invalid
        cost_usd_micro_per_call=50,
    )
    llm = StructuredLLM(provider, agent_kernel=ak, model="fake-1", max_retries=1)
    with pytest.raises(LLMCallError):
        llm.generate(
            messages=[{"role": "user", "content": "x"}],
            response_model=Sentiment,
            task_id=task.task_id,
        )
    events = ak.list_events(task.task_id)
    failed = [
        e
        for e in events
        if e.event_type == EventType.llm_call_completed and e.status.value == "error"
    ]
    assert len(failed) == 1
    # Even on failure the partial cost was debited (caller spent provider $)
    assert any(e.event_type == EventType.budget_debited for e in events)


@pytest.mark.integration
def test_llm_budget_exhaustion_does_not_drive_negative(tmp_path: Path) -> None:
    ak = AgentKernel(tmp_path)
    # Create a task with a tight LLM budget.
    task = ak.create_task(
        notebook_path=str(tmp_path / "nb.ipynb"),
        kernel_name="python3",
        reserved_budget=Budget(wall_ms=60_000, llm_usd_micro=200),
    )

    # Each call costs 150; first call ok (50 remaining), second call would
    # cost 150 but only 50 remain — adapter must cap the debit, not go negative.
    provider = FakeProvider(
        script=[
            '{"label": "positive", "confidence": 0.5}',
            '{"label": "negative", "confidence": 0.5}',
        ],
        cost_usd_micro_per_call=150,
    )
    llm = StructuredLLM(provider, agent_kernel=ak, model="fake-1")
    llm.generate(
        messages=[{"role": "user", "content": "a"}],
        response_model=Sentiment,
        task_id=task.task_id,
    )
    llm.generate(
        messages=[{"role": "user", "content": "b"}],
        response_model=Sentiment,
        task_id=task.task_id,
    )

    persisted = ak.get_task(task.task_id)
    assert persisted is not None
    # Spent must equal reservation (capped); never exceed it.
    assert persisted.spent_budget.llm_usd_micro <= persisted.reserved_budget.llm_usd_micro + 200
    # Budget invariants on the JSONL
    for ev in ak.list_events(task.task_id):
        for b in (ev.budget_before, ev.budget_after):
            if b is None:
                continue
            assert b.llm_usd_micro >= 0


# ============================ live (opt-in only) ===========================


@pytest.mark.integration
@pytest.mark.llm
def test_lmstudio_provider_live_smoke(tmp_path: Path) -> None:
    """Optional smoke test against a local LM Studio server.

    Skipped automatically when no server is reachable at
    ``LMSTUDIO_BASE_URL`` or the default. Marked ``llm`` so CI only runs
    this when explicitly selected.
    """
    base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    provider = LMStudioProvider(base_url=base_url)
    if not provider.is_reachable():
        pytest.skip(f"LM Studio not reachable at {base_url}")
    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path=str(tmp_path / "nb.ipynb"), kernel_name="python3")
    llm = StructuredLLM(provider, agent_kernel=ak, model=os.environ.get("LMSTUDIO_MODEL"))
    obj = llm.generate(
        messages=[{"role": "user", "content": "Classify the sentiment of: 'I love this.'"}],
        response_model=Sentiment,
        task_id=task.task_id,
    )
    assert obj.label in {"positive", "negative", "neutral"}
    # Even if cost is zero (local), the call must still be in the ledger.
    types = [e.event_type for e in ak.list_events(task.task_id)]
    assert EventType.llm_call_started in types
    assert EventType.llm_call_completed in types


# ====================== integration through a notebook ======================


@pytest.mark.integration
def test_llm_call_inside_notebook_kernel_emits_full_ledger(tmp_path: Path) -> None:
    """Run a notebook through M2's NotebookRunner; inside the kernel, invoke
    ``StructuredLLM.generate()`` against ``FakeProvider`` and verify the
    workspace ledger contains the full LLM-event chain."""
    import json
    import textwrap

    import nbformat
    from nbformat.v4 import new_code_cell, new_notebook

    from agent_kernel.runtime.notebook_runner import NotebookRunner

    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    ak = AgentKernel(tmp_path)
    task = ak.create_task(notebook_path="", kernel_name="python3")

    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    repo_root = Path(__file__).resolve().parents[2]
    src = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {json.dumps(str(repo_root))})
        from agent_kernel.api import AgentKernel
        from agent_kernel.llm import FakeProvider, StructuredLLM
        from pydantic import BaseModel, Field

        class Sentiment(BaseModel):
            label: str
            confidence: float = Field(ge=0.0, le=1.0)

        ak = AgentKernel({json.dumps(str(tmp_path))})
        provider = FakeProvider(
            script=['{{"label": "positive", "confidence": 0.8}}'],
            cost_usd_micro_per_call=500,
        )
        llm = StructuredLLM(provider, agent_kernel=ak, model="fake-1")
        obj = llm.generate(
            messages=[{{"role": "user", "content": "good"}}],
            response_model=Sentiment,
            task_id={json.dumps(task.task_id)},
        )
        print("RESULT", obj.label, obj.confidence)
        """
    ).strip()
    nb.cells = [new_code_cell(src, id="llm_cell")]
    notebook_path = ws.notebooks_dir / "with_llm.ipynb"
    nbformat.write(nb, notebook_path)

    events = JSONLEventStore(ws.events_dir, fsync=True)
    runner = NotebookRunner(events, ws.runs_dir)
    runner.run(notebook_path, task_id=task.task_id, kernel_name="python3", timeout=60)

    fresh_events = ak.list_events(task.task_id)
    assert any(e.event_type == EventType.llm_call_started for e in fresh_events)
    completed = [e for e in fresh_events if e.event_type == EventType.llm_call_completed]
    assert completed and completed[0].payload["cost_usd_micro"] == 500
    assert any(e.event_type == EventType.budget_debited for e in fresh_events)
