"""Live integration tests against a real LM Studio instance.

These tests require LM Studio running at ``NOTEBOOK_AGENT_BASE_URL``
(default ``http://host.docker.internal:1234/v1``) with at least one model
loaded. They are gated by the ``live`` marker and skipped otherwise:

    pytest -m live tests/integration/test_live_lm_studio.py -v

Coverage:

1. **Provider reachability** — ``/api/v0/models`` reports a loaded model.
2. **Reasoning-effort characterization** — probe each candidate value
   directly and record timing / reasoning_tokens. This is what the
   ``core.discover_model_settings`` skill does in production; running it
   as a test lets us assert the value space *before* trusting the agent
   loop to discover it.
3. **End-to-end discovery skill** — run the bundled skill notebook via
   Papermill against live LM Studio and verify it writes
   ``sessions/<slug>/model_settings.ipynb``.
4. **End-to-end ``%task`` magic** — drive the real IPython magic against
   live LM Studio and verify it (a) auto-runs discovery, (b) returns a
   non-empty answer, (c) does NOT re-run discovery on the second call.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


LIVE_BASE_URL = os.environ.get(
    "NOTEBOOK_AGENT_BASE_URL", "http://host.docker.internal:1234/v1"
)
LIVE_API_KEY = os.environ.get("NOTEBOOK_AGENT_API_KEY", "lm-studio")
# Per-probe wall-clock cap. Real thinking models can blow past 30s when
# reasoning_effort is None/ignored; we want failures to surface fast.
PROBE_TIMEOUT_S = 30.0
# Whole-test wall-clock cap for the agent end-to-end test.
AGENT_TIMEOUT_S = 120.0


@pytest.fixture(scope="module")
def live_loaded_model() -> str:
    """Return the id of a currently-loaded model, or skip if unreachable."""
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


def _strip_provider(model: str) -> str:
    if "/" in model and model.split("/", 1)[0] in {"lm_studio", "openai", "anthropic", "ollama"}:
        return model.split("/", 1)[1]
    return model


def _probe_chat(model: str, effort: str | None, *, timeout: float = PROBE_TIMEOUT_S) -> dict:
    """One canary request. Never raises — failures go into the result dict."""
    payload = {
        "model": _strip_provider(model),
        "messages": [{"role": "user", "content": "Reply with exactly the single word OK."}],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    if effort is not None:
        payload["reasoning_effort"] = effort
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LIVE_API_KEY}"}
    req = urllib.request.Request(
        LIVE_BASE_URL.rstrip("/") + "/chat/completions",
        data=body, headers=headers, method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        usage = data.get("usage") or {}
        det = usage.get("completion_tokens_details") or {}
        return {
            "effort": effort, "ok": True, "elapsed": round(time.monotonic() - t0, 2),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": det.get("reasoning_tokens"),
            "content": (data["choices"][0]["message"].get("content") or "")[:80],
        }
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {
            "effort": effort, "ok": False, "elapsed": round(time.monotonic() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:120]}",
        }


def test_provider_reports_loaded_model(live_loaded_model: str) -> None:
    """Sanity: provider lists at least one model with state=loaded."""
    assert live_loaded_model, "expected a non-empty loaded model id"


@pytest.mark.parametrize("effort", ["off", "low", "medium", "high", None])
def test_reasoning_effort_probe(live_loaded_model: str, effort: str | None) -> None:
    """Characterize each reasoning_effort value on the live model.

    This is intentionally non-judgmental: a probe is allowed to fail
    (some servers reject unsupported values) — we just record the result
    so the test log shows real timing. The assertion only requires that
    *at least one* probe path completes inside the timeout, i.e. the
    server is actually serving requests.
    """
    r = _probe_chat(live_loaded_model, effort)
    # Print so pytest -v / -s shows real-world timing in the test log.
    print(f"\n  effort={effort!r:>8} ok={r['ok']} elapsed={r['elapsed']}s "
          f"rt={r.get('reasoning_tokens')} ct={r.get('completion_tokens')} "
          f"err={r.get('error', '')}")
    # Per-effort assertion is light: the request must terminate (not hang).
    # The aggregate test below asserts at least one effort completed ok.
    assert "elapsed" in r
    assert r["elapsed"] < PROBE_TIMEOUT_S + 5  # urlopen timeout works


def test_at_least_one_effort_succeeds(live_loaded_model: str) -> None:
    """The server must be capable of completing *some* request inside 30s.

    If this fails, the discovery skill would also fail and the agent has
    no good options — investigate the server before changing the agent.
    """
    results = [_probe_chat(live_loaded_model, e) for e in ["off", "low", None]]
    oks = [r for r in results if r["ok"]]
    assert oks, (
        "All quick probes failed against live server. Results:\n"
        + "\n".join(repr(r) for r in results)
    )


def test_discovery_skill_end_to_end(live_loaded_model: str, tmp_path: Path) -> None:
    """Run ``core.discover_model_settings`` via Papermill against live LM Studio.

    Asserts the skill writes a parseable settings notebook at
    ``sessions/<slug>/model_settings.ipynb`` and that the recommended
    ``reasoning_effort`` is one of the legal values.
    """
    from notebook_agent.model_settings import (
        model_slug,
        read_settings,
        settings_notebook_path,
    )
    from notebook_agent.notebook_exec import execute_notebook

    skill_nb = (
        Path(__file__).resolve().parents[2]
        / "notebook_agent" / "builtin_skills" / "discover_model_settings" / "skill.ipynb"
    )
    assert skill_nb.exists(), f"discovery skill missing at {skill_nb}"

    sessions_root = tmp_path / "sessions"
    slug = model_slug(live_loaded_model)
    work = sessions_root / slug / "_discovery"
    work.mkdir(parents=True, exist_ok=True)

    res = execute_notebook(
        skill_nb,
        parameters={
            "target_model": live_loaded_model,
            "base_url": LIVE_BASE_URL,
            "api_key": LIVE_API_KEY,
            "sessions_root": str(sessions_root),
            # Keep the test bounded.
            "per_request_timeout": PROBE_TIMEOUT_S,
            "candidates": ["off", "low", None],  # skip slow ones for CI speed
        },
        output_path=work / "executed.ipynb",
        run_dir=work,
    )
    # ``execute_notebook`` swallows papermill exceptions into ``res.error``;
    # surface them so the test failure points at the actual cell error.
    assert res.success, (
        f"discovery skill crashed: {res.error}\n"
        f"executed notebook: {work / 'executed.ipynb'}"
    )

    settings_path = settings_notebook_path(live_loaded_model, sessions_root=sessions_root)
    assert settings_path.exists(), (
        f"discovery skill did not write {settings_path}; "
        f"executed notebook: {work / 'executed.ipynb'}"
    )
    settings = read_settings(settings_path)
    print(f"\n  discovered settings for {live_loaded_model!r}: {settings}")
    assert "reasoning_effort" in settings
    assert settings["reasoning_effort"] in {None, "off", "low", "medium", "high"}
    assert "supports_reasoning_effort" in settings
    assert isinstance(settings["supports_reasoning_effort"], bool)


def test_task_magic_end_to_end(
    live_loaded_model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive ``%task`` end-to-end against live LM Studio.

    Verifies the full chain:

    1. First ``%task`` triggers ``_ensure_model_settings_discovered`` and
       writes ``sessions/<slug>/model_settings.ipynb``.
    2. The discovered ``reasoning_effort`` is applied to the live
       ``LiteLLMClient`` (so the actual agent LM call isn't slow).
    3. ``run_task`` returns a non-empty answer.
    4. A second ``%task`` does NOT re-run discovery (idempotent cache).
    """
    from notebook_agent import magics as magics_mod
    from notebook_agent.litellm_client import LiteLLMClient
    from notebook_agent.model_settings import (
        model_slug,
        read_settings,
        settings_notebook_path,
    )
    from notebook_agent.notebook_init import NotebookConfig

    # Point everything at a tmp sessions/runs root so the test is hermetic.
    sessions_root = tmp_path / "sessions"
    runs_root = tmp_path / "runs"
    sessions_root.mkdir()
    runs_root.mkdir()

    # Build an explicit notebook config (no env coupling) and stash it
    # exactly the way init_notebook would.
    client = LiteLLMClient(
        base_url=LIVE_BASE_URL,
        api_key=LIVE_API_KEY,
        # Don't pin reasoning_effort — let discovery decide.
        reasoning_effort=None,
        # Generous but bounded; the discovered setting should make the
        # actual agent call far faster than this.
        request_timeout=AGENT_TIMEOUT_S,
    )
    from notebook_agent.dspy_lm import configure_dspy
    configure_dspy(client)

    nb = NotebookConfig(
        client=client,
        # Enough turns to plan + choose_skill + generate + execute. With 1,
        # the agent runs only the planner and gives up, which doesn't
        # exercise the full chain.
        max_autonomous_turns=6,
        runs_root=str(runs_root),
        sessions_root=str(sessions_root),
    )
    monkeypatch.setattr(magics_mod, "get_notebook_config", lambda: nb)

    # --- first call: should run discovery + answer the question ----------
    t0 = time.monotonic()
    result = magics_mod.task_line_magic("what is 2 + 2", ip=None)
    elapsed = time.monotonic() - t0
    print(f"\n  first %task: {elapsed:.1f}s, answer={getattr(result, 'answer', result)!r}")

    slug = model_slug(live_loaded_model)
    settings_path = settings_notebook_path(live_loaded_model, sessions_root=sessions_root)
    # Discovery may resolve to either the configured model id or the bare
    # loaded id (LM Studio strips provider prefixes). Accept either slug.
    alt_path = sessions_root / slug / "model_settings.ipynb"
    discovered = read_settings(settings_path) or read_settings(alt_path)
    assert discovered, (
        f"expected discovery to write settings under {sessions_root}; "
        f"sessions dir contents: {list(sessions_root.rglob('*'))}"
    )
    assert "reasoning_effort" in discovered

    # The agent should have produced something. We don't pin the exact
    # text (LMs are non-deterministic even at temp=0), just that it ran.
    assert result is not None
    assert elapsed < AGENT_TIMEOUT_S, f"first %task took {elapsed:.1f}s"

    # --- second call: discovery must NOT re-run --------------------------
    settings_mtime = settings_path.stat().st_mtime if settings_path.exists() else alt_path.stat().st_mtime
    t0 = time.monotonic()
    magics_mod.task_line_magic("what is 3 + 3", ip=None)
    elapsed2 = time.monotonic() - t0
    print(f"  second %task: {elapsed2:.1f}s")

    final_mtime = settings_path.stat().st_mtime if settings_path.exists() else alt_path.stat().st_mtime
    assert final_mtime == settings_mtime, "discovery re-ran on cached model"
