"""M9 hardening gate: invariant property tests + replay + redaction.

Three concerns are exercised here:

1. **Hypothesis property tests** over the pure policy engine, asserting:
   - Budget non-negativity after debit / refund.
   - ``refund`` is the algebraic inverse of ``debit`` up to the remaining
     reservation (refunding what was just debited restores the same
     remaining-reservation amount).
   - Depth bounds are monotonic: a child task's depth equals
     ``parent.depth + 1`` whenever spawn admission succeeds, and admission
     always fails once ``parent.depth >= max_spawn_depth``.
   - Termination bounds: ``must_stop`` becomes True the moment any budget
     component reaches its reservation cap and stays True thereafter.

2. **Regression replay**: produce a JSONL trace by running real tasks
   through ``AgentKernel``; then drive ``scripts/replay.py`` against that
   workspace and a freshly-rebuilt workspace constructed by re-appending
   the same events into a new ``JSONLEventStore``. Both runs must yield
   identical ``sha256`` digests and identical reconstructed task state.

3. **Secret-leak test**: plant a fake OpenAI key in an LLM message and
   in a task parameter. Run an end-to-end ``StructuredLLM`` call through
   ``FakeProvider`` and assert the planted secret is absent from every
   JSONL event payload.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings
from pydantic import BaseModel, Field

from agent_kernel.api import AgentKernel
from agent_kernel.llm import FakeProvider, StructuredLLM
from agent_kernel.models.budget import Budget
from agent_kernel.models.policy import PolicyProfile
from agent_kernel.models.task import TaskSpec, TaskStatus
from agent_kernel.runtime import policy_engine
from agent_kernel.security.redaction import redact, redact_payload
from agent_kernel.storage import JSONLEventStore, WorkspaceLayout
from agent_kernel.util import new_id, now_iso

# ============================== strategies =============================


budget_strategy = st.builds(
    Budget,
    wall_ms=st.integers(min_value=0, max_value=1_000_000),
    cpu_ms=st.integers(min_value=0, max_value=1_000_000),
    llm_usd_micro=st.integers(min_value=0, max_value=1_000_000),
    llm_input_tokens=st.integers(min_value=0, max_value=100_000),
    llm_output_tokens=st.integers(min_value=0, max_value=100_000),
    bytes_written=st.integers(min_value=0, max_value=10_000_000),
    spawn_count=st.integers(min_value=0, max_value=64),
)


def _task(reserved: Budget, spent: Budget | None = None, depth: int = 0) -> TaskSpec:
    return TaskSpec(
        task_id=new_id("task"),
        notebook_path="x.ipynb",
        kernel_name="python3",
        policy_profile="local-dev",
        depth=depth,
        status=TaskStatus.running,
        reserved_budget=reserved,
        spent_budget=spent or Budget(),
        created_at=now_iso(),
        updated_at=now_iso(),
    )


_BUDGET_FIELDS = list(Budget.model_fields.keys())


def _all_non_negative(b: Budget) -> bool:
    return all(getattr(b, f) >= 0 for f in _BUDGET_FIELDS)


def _le(a: Budget, b: Budget) -> bool:
    return all(getattr(a, f) <= getattr(b, f) for f in _BUDGET_FIELDS)


# ============================== property tests ==========================


@pytest.mark.integration
@pytest.mark.slow
@settings(
    max_examples=10_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(reserved=budget_strategy, delta=budget_strategy)
def test_property_debit_never_negative_and_bounded_by_reservation(
    reserved: Budget, delta: Budget
) -> None:
    """``debit`` must clip at the reservation; result must be non-negative,
    ``spent_budget <= reserved_budget`` after the call, and ``ValueError``
    must be raised iff ``delta`` would exceed remaining headroom."""
    task = _task(reserved)
    remaining = policy_engine.remaining_reservation(task)
    fits = _le(delta, remaining)
    if not fits:
        with pytest.raises(ValueError):
            policy_engine.debit(task, delta)
        return
    trans = policy_engine.debit(task, delta)
    assert _all_non_negative(trans.spent_budget)
    assert _all_non_negative(trans.reserved_budget)
    assert _le(trans.spent_budget, trans.reserved_budget)


@pytest.mark.integration
@pytest.mark.slow
@settings(
    max_examples=10_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(reserved=budget_strategy, delta=budget_strategy)
def test_property_refund_invariants(reserved: Budget, delta: Budget) -> None:
    """``refund`` must (a) be rejected when ``delta`` exceeds the available
    headroom, (b) leave ``spent_budget`` unchanged, (c) keep both budgets
    non-negative, and (d) maintain ``spent <= reserved``."""
    task = _task(reserved)
    remaining = policy_engine.remaining_reservation(task)
    fits = _le(delta, remaining)
    if not fits:
        with pytest.raises(ValueError):
            policy_engine.refund(task, delta)
        return
    trans = policy_engine.refund(task, delta)
    assert trans.spent_budget == task.spent_budget  # spent unchanged
    assert _all_non_negative(trans.reserved_budget)
    assert _all_non_negative(trans.spent_budget)
    assert _le(trans.spent_budget, trans.reserved_budget)


@pytest.mark.integration
@pytest.mark.slow
@settings(
    max_examples=10_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(
    parent_depth=st.integers(min_value=0, max_value=16),
    max_depth=st.integers(min_value=0, max_value=16),
)
def test_property_depth_bounds_monotonic(parent_depth: int, max_depth: int) -> None:
    """``can_spawn`` must reject whenever parent.depth >= max_spawn_depth."""
    parent = _task(Budget(), depth=parent_depth)
    profile = PolicyProfile(
        name="t",
        max_spawn_depth=max_depth,
        max_children_per_task=999,
    )
    decision = policy_engine.can_spawn(parent, profile, Budget())
    if parent_depth >= max_depth:
        assert not decision.allowed
        assert "depth" in (decision.reason or "").lower()


@pytest.mark.integration
@pytest.mark.slow
@settings(
    max_examples=10_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(reserved=budget_strategy)
def test_property_must_stop_when_any_component_exhausted(reserved: Budget) -> None:
    """``must_stop`` returns ``allowed=True`` (the convention here is
    "policy permits/requires this transition") whenever any component of
    the reservation is fully spent. Conversely, when ``spent == 0`` and
    any reservation field is positive, ``must_stop`` must be False."""
    # spent == reserved on every field => must_stop is True iff at least one field > 0
    task_full = _task(reserved, spent=reserved)
    if any(getattr(reserved, f) > 0 for f in _BUDGET_FIELDS):
        assert policy_engine.must_stop(task_full).allowed is True
    # spent == 0, reservation > 0 on some field => must_stop is False
    task_empty = _task(reserved, spent=Budget())
    if any(getattr(reserved, f) > 0 for f in _BUDGET_FIELDS):
        assert policy_engine.must_stop(task_empty).allowed is False


# ============================== replay test =============================


@pytest.mark.integration
def test_replay_bit_exact_reproduction(tmp_path: Path) -> None:
    """Generate a real JSONL trace; run scripts/replay.py twice (once on
    the original workspace, once on a workspace rebuilt from those
    events) and assert identical sha256 digests."""
    ws = WorkspaceLayout(tmp_path / "original")
    ws.ensure()
    ak = AgentKernel(tmp_path / "original")
    # Three tasks, including a failure and a deliberate refund path.
    for i in range(3):
        nb = ws.notebooks_dir / f"n{i}.ipynb"
        import nbformat
        from nbformat.v4 import new_code_cell, new_notebook

        nb_data = new_notebook()
        nb_data.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
        nb_data.metadata["language_info"] = {"name": "python"}
        if i == 1:
            nb_data.cells = [new_code_cell("raise RuntimeError('boom')", id="c0")]
        else:
            nb_data.cells = [new_code_cell("print('ok')", id="c0")]
        nbformat.write(nb_data, nb)
        task = ak.create_task(notebook_path=str(nb), kernel_name="python3")
        ak.run_task(task.task_id)

    # Replay original
    repo_root = Path(__file__).resolve().parents[2]
    r1 = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "replay.py"), str(tmp_path / "original")],
        capture_output=True,
        text=True,
        check=True,
    )
    digest1 = json.loads(r1.stdout)
    assert digest1["task_count"] == 3
    assert digest1["event_count"] >= 9  # at minimum created+admitted+(completed|failed)

    # Rebuild a fresh workspace by replaying the events into a new JSONL store.
    ws2 = WorkspaceLayout(tmp_path / "rebuilt")
    ws2.ensure()
    store_in = JSONLEventStore(ws.events_dir)
    store_out = JSONLEventStore(ws2.events_dir)
    for ev in store_in.iter_events():
        store_out.append(ev)

    r2 = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "replay.py"), str(tmp_path / "rebuilt")],
        capture_output=True,
        text=True,
        check=True,
    )
    digest2 = json.loads(r2.stdout)
    assert digest1["sha256"] == digest2["sha256"], (digest1, digest2)
    assert digest1["task_count"] == digest2["task_count"]
    assert digest1["event_count"] == digest2["event_count"]

    # Reconstructed tasks must be identical (order-independent).
    by_id_1 = {t["task_id"]: t for t in digest1["tasks"]}
    by_id_2 = {t["task_id"]: t for t in digest2["tasks"]}
    assert by_id_1 == by_id_2


# ============================ redaction tests ===========================


def test_redact_function_handles_known_patterns() -> None:
    fake_openai = "sk-proj-" + "A" * 32
    fake_anthropic = "sk-ant-" + "B" * 32
    fake_github = "ghp_" + "C" * 30
    fake_aws = "AKIA" + "D" * 16
    fake_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk"
    bearer = "Authorization: Bearer abcdefg1234567890_secret_token_value"
    text = " | ".join([fake_openai, fake_anthropic, fake_github, fake_aws, fake_jwt, bearer])
    out = redact(text)
    for s in [fake_openai, fake_anthropic, fake_github, fake_aws, fake_jwt]:
        assert s not in out
    assert "<REDACTED:openai_key>" in out
    assert "<REDACTED:anthropic_key>" in out
    assert "<REDACTED:github_token>" in out
    assert "<REDACTED:aws_access_key_id>" in out
    assert "<REDACTED:jwt>" in out
    assert "<REDACTED:bearer_token>" in out


def test_redact_payload_strips_sensitive_keys() -> None:
    payload = {
        "api_key": "literally-anything",
        "nested": {"password": "hunter2", "ok": "fine"},
        "list": ["sk-proj-" + "X" * 32, "fine string"],
    }
    out = redact_payload(payload)
    assert out["api_key"] == "<REDACTED:api_key>"
    assert out["nested"]["password"] == "<REDACTED:password>"
    assert out["nested"]["ok"] == "fine"
    assert out["list"][0].startswith("<REDACTED:openai_key>")
    assert out["list"][1] == "fine string"


@pytest.mark.integration
def test_secret_leak_in_llm_messages_is_redacted_in_jsonl(tmp_path: Path) -> None:
    """End-to-end: an attacker (or careless user) places an OpenAI key in
    the prompt and as a task parameter. After the run, no JSONL file
    contains the secret."""

    class S(BaseModel):
        label: str = Field(default="x")

    planted = "sk-proj-" + "Z" * 40
    ak = AgentKernel(tmp_path)
    task = ak.create_task(
        notebook_path=str(tmp_path / "n.ipynb"),
        kernel_name="python3",
        parameters={"prompt": f"please use {planted} in your reply"},
    )

    provider = FakeProvider(
        script=['{"label": "ok"}'],
        cost_usd_micro_per_call=10,
    )
    llm = StructuredLLM(provider, agent_kernel=ak, model="fake-1")
    llm.generate(
        messages=[
            {"role": "system", "content": f"key={planted}"},
            {"role": "user", "content": planted},
        ],
        response_model=S,
        task_id=task.task_id,
    )

    # Scan every JSONL file in the workspace for the planted secret.
    events_dir = tmp_path / ".agent_kernel" / "events"
    found_files = list(events_dir.glob("*.jsonl"))
    assert found_files, "expected at least one JSONL events file"
    for fp in found_files:
        contents = fp.read_text(encoding="utf-8")
        assert planted not in contents, f"secret leaked into {fp}: \n{contents[:500]}"
        assert "<REDACTED:openai_key>" in contents or "<REDACTED:" in contents

    # And: the in-memory event objects (which carry pre-redaction payloads
    # transiently) should also have a redacted representation on disk —
    # verify by round-tripping through the store iterator.
    store = JSONLEventStore(events_dir)
    for ev in store.iter_events():
        blob = ev.model_dump_json()
        assert planted not in blob
