"""Live Papermill tests that exercise the *actual* user UX.

These tests build a user-shaped notebook on disk (``%load_ext
notebook_agent`` + ``%task <prompt>``), then drive it through Papermill
against live LM Studio — i.e. **the exact pipeline a user runs**:

    Papermill → Jupyter kernel → IPython → %task → run_task → DSPy → LM Studio

What this catches that the in-process live tests cannot:

* Kernel-boundary issues (magic registration, module import in the kernel).
* Real-clock latency of thinking models (no fakes, no DummyLM).
* That nothing on the client side cancels / times out a slow LM call
  (the user's reported "5-minute hang" symptom). Per the spec we enforce
  only ``max_autonomous_turns`` plus lack-of-progress — never wall-time
  or token caps.

Gated by the ``live`` marker; skipped when LM Studio is unreachable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import nbformat
import papermill as pm
import pytest

pytestmark = pytest.mark.live


LIVE_BASE_URL = os.environ.get(
    "NOTEBOOK_AGENT_BASE_URL", "http://host.docker.internal:1234/v1"
)
LIVE_API_KEY = os.environ.get("NOTEBOOK_AGENT_API_KEY", "lm-studio")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_loaded_model() -> str:
    """Skip the module entirely if LM Studio has no loaded model."""
    from notebook_agent.litellm_client import LiteLLMClient
    from notebook_agent.model_settings import pick_loaded_model

    client = LiteLLMClient(base_url=LIVE_BASE_URL, api_key=LIVE_API_KEY)
    try:
        loaded = pick_loaded_model(client)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"LM Studio probe failed: {exc!r}")
    if loaded is None:
        pytest.skip(f"No model loaded at {LIVE_BASE_URL}")
    return loaded.id


def _make_user_notebook(cells: list[str]) -> dict[str, Any]:
    """Build an nbformat dict shaped like a real user notebook.

    First code cell carries the Papermill ``parameters`` tag so callers can
    inject overrides (currently used only by environment-style params).
    """
    nb: dict[str, Any] = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # parameters cell — Papermill needs at least one tagged cell to
    # inject parameters. We keep it empty by default; tests override
    # globals like ``request`` if they need to vary inputs per-run.
    nb["cells"].append(
        {
            "cell_type": "code",
            "id": uuid4().hex[:8],
            "metadata": {"tags": ["parameters"]},
            "source": "# papermill parameters\n",
            "outputs": [],
            "execution_count": None,
        }
    )
    for src in cells:
        nb["cells"].append(
            {
                "cell_type": "code",
                "id": uuid4().hex[:8],
                "metadata": {},
                "source": src,
                "outputs": [],
                "execution_count": None,
            }
        )
    return nb


def _write_notebook(path: Path, nb: dict[str, Any]) -> Path:
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def _flatten_outputs(nb: nbformat.NotebookNode) -> str:
    """Concatenate all stdout/stream/display text from every code cell."""
    parts: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []) or []:
            t = out.get("output_type")
            if t == "stream":
                parts.append(out.get("text", ""))
            elif t in {"display_data", "execute_result"}:
                data = out.get("data", {}) or {}
                if "text/plain" in data:
                    val = data["text/plain"]
                    parts.append(val if isinstance(val, str) else "\n".join(val))
                if "text/markdown" in data:
                    val = data["text/markdown"]
                    parts.append(val if isinstance(val, str) else "\n".join(val))
    return "\n".join(parts)


def _env_for_kernel(tmp_path: Path) -> dict[str, str]:
    """Env passed through to the Papermill kernel.

    Critically: NO ``NOTEBOOK_AGENT_REQUEST_TIMEOUT``. We're testing that
    the agent does NOT cancel slow LM calls from the client side.
    """
    env = os.environ.copy()
    env["NOTEBOOK_AGENT_BASE_URL"] = LIVE_BASE_URL
    env["NOTEBOOK_AGENT_API_KEY"] = LIVE_API_KEY
    # Force hermetic session/runs roots under tmp_path so the test leaves
    # nothing behind in the workspace.
    env["NOTEBOOK_AGENT_SESSIONS_ROOT"] = str(tmp_path / "sessions")
    env["NOTEBOOK_AGENT_RUNS_ROOT"] = str(tmp_path / "runs")
    # Make sure nothing inherits a stale timeout from the shell that ran
    # pytest. The whole point of this test is "no client-side cancels".
    env.pop("NOTEBOOK_AGENT_REQUEST_TIMEOUT", None)
    env.pop("NOTEBOOK_AGENT_MAX_TOKENS", None)
    env.pop("NOTEBOOK_AGENT_REASONING_EFFORT", None)
    # Do NOT pin a model — mirror real user UX: whatever is loaded in
    # LM Studio at runtime is what the agent should use.
    env.pop("NOTEBOOK_AGENT_MODEL", None)
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_papermill_user_notebook_easy_task(
    live_loaded_model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The minimal user UX: ``%load_ext`` + one ``%task`` cell.

    Asserts the kernel finishes both cells without raising and that the
    %task cell produces some visible output (the agent's answer).
    """
    # Patch the env in this process — papermill kernel inherits it.
    for k, v in _env_for_kernel(tmp_path).items():
        monkeypatch.setenv(k, v)

    nb = _make_user_notebook(
        [
            "%load_ext notebook_agent\n",
            "%task what is two plus two\n",
        ]
    )
    nb_in = _write_notebook(tmp_path / "user_easy.ipynb", nb)
    nb_out = tmp_path / "user_easy.executed.ipynb"

    pm.execute_notebook(
        str(nb_in),
        str(nb_out),
        parameters={},
        cwd=str(tmp_path),
        kernel_name="python3",
    )

    executed = nbformat.read(nb_out, as_version=4)
    # Every cell must have run (execution_count set).
    code_cells = [c for c in executed.cells if c.cell_type == "code"]
    for c in code_cells:
        assert c.execution_count is not None, f"cell did not execute: {c.source!r}"

    output_text = _flatten_outputs(executed)
    # The agent's answer rendering goes through ``show_answer`` (Markdown
    # display_data) so SOMETHING should be in the cell output.
    assert output_text.strip(), (
        "executed notebook had no cell output; see " f"{nb_out}"
    )
    # And a manifest must have been written.
    runs_root = tmp_path / "runs"
    manifests = list(runs_root.rglob("manifest.json"))
    assert manifests, f"no run manifest under {runs_root}"


def test_papermill_user_notebook_no_client_timeout(
    live_loaded_model: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user's reported regression: a hard-for-the-model prompt.

    With the default config (``request_timeout=None``, ``max_tokens=None``)
    the agent must NOT raise a timeout/cancelled error mid-call. It is
    allowed to take a long time; it is allowed to surface a failed
    result if the model produces empty content. It is NOT allowed to
    truncate the LM call or to mark the run ``cancelled`` because of a
    client-side wall-clock cap.
    """
    for k, v in _env_for_kernel(tmp_path).items():
        monkeypatch.setenv(k, v)

    nb = _make_user_notebook(
        [
            "%load_ext notebook_agent\n",
            # A prompt that historically caused Gemma-4 to burn 10k+
            # reasoning tokens. We don't care about the answer's
            # correctness — we care that no client-side cancellation
            # happens.
            "%task when is Mardi Gras in 2072\n",
        ]
    )
    nb_in = _write_notebook(tmp_path / "user_hard.ipynb", nb)
    nb_out = tmp_path / "user_hard.executed.ipynb"

    pm.execute_notebook(
        str(nb_in),
        str(nb_out),
        parameters={},
        cwd=str(tmp_path),
        kernel_name="python3",
    )

    executed = nbformat.read(nb_out, as_version=4)
    output_text = _flatten_outputs(executed)

    # Specific symptoms of a client-side cancellation. None of these may
    # appear in the cell output OR in the run's event log.
    forbidden = [
        "ReadTimeout",
        "ReadTimeoutError",
        "HTTPSConnectionPool",
        "client disconnected",
        "Client disconnected",
        "Operation timed out",
        "504",
    ]
    for needle in forbidden:
        assert needle not in output_text, (
            f"client-side timeout symptom {needle!r} in cell output; see {nb_out}"
        )
    runs_root = tmp_path / "runs"
    for ev_path in runs_root.rglob("events.jsonl"):
        text = ev_path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, (
                f"client-side timeout symptom {needle!r} in {ev_path}"
            )
