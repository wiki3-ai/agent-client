"""Magic hook → discovery skill → cached settings, end-to-end (mocked LM).

The real discovery skill probes a live OpenAI-compatible endpoint. Here we
stub both the loaded-models lookup and the Papermill execution so the test
runs offline yet still verifies the *wiring*: that the hook detects a
missing settings file, invokes ``execute_notebook`` with the right
parameters, then re-reads the freshly-written notebook and applies its
``reasoning_effort`` to the live :class:`LiteLLMClient`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notebook_agent import magics as magics_mod
from notebook_agent.litellm_client import LiteLLMClient
from notebook_agent.model_settings import LoadedModel, write_settings
from notebook_agent.notebook_init import NotebookConfig


@pytest.fixture
def nb_with_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NotebookConfig:
    """A notebook config pointing at a tmp sessions root, no LM Studio."""
    client = LiteLLMClient(
        provider="lm_studio",
        model="lm_studio/test/gemma-stub",
        base_url="http://stub.invalid:1234/v1",
        api_key="not-used",
    )
    nb = NotebookConfig(
        client=client,
        sessions_root=str(tmp_path / "sessions"),
    )
    monkeypatch.setattr(magics_mod, "get_notebook_config", lambda: nb)
    # configure_dspy would touch dspy.settings.lm; not needed for this test.
    monkeypatch.setattr(magics_mod, "configure_dspy", lambda c: None)
    return nb


def test_hook_runs_discovery_when_settings_absent(
    nb_with_client: NotebookConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """First call: for ``provider=lm_studio`` the hook must write the
    passive settings stub *without* invoking the Papermill probe.

    Rationale: LM Studio's OpenAI-compat endpoint silently ignores the
    ``reasoning_effort`` field (lmstudio-bug-tracker#988), so any probe
    that tries to characterize the value space is meaningless and just
    adds latency to the first ``%task`` call. The hook records that fact
    in the canonical settings notebook and returns immediately.
    """
    from notebook_agent import model_settings as ms_mod
    from notebook_agent import notebook_exec as nbx_mod
    from notebook_agent.model_settings import read_settings, settings_notebook_path

    # Pretend LM Studio reports our model as loaded.
    monkeypatch.setattr(
        ms_mod,
        "pick_loaded_model",
        lambda client, prefer=None: LoadedModel(id=prefer or client.model, state="loaded"),
    )
    monkeypatch.setattr(
        "notebook_agent.model_settings.pick_loaded_model",
        lambda client, prefer=None: LoadedModel(id=prefer or client.model, state="loaded"),
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "execute_notebook must NOT be called for lm_studio provider; "
            "reasoning_effort is ignored upstream so probing is pointless"
        )

    monkeypatch.setattr(nbx_mod, "execute_notebook", boom)

    magics_mod._ensure_model_settings_discovered()

    # The passive stub must be written, capturing why we didn't probe.
    assert nb_with_client.client is not None
    target = settings_notebook_path(
        nb_with_client.client.model,
        sessions_root=Path(nb_with_client.sessions_root),
    )
    assert target.exists(), f"hook did not write {target}"
    settings = read_settings(target)
    assert settings["supports_reasoning_effort"] is False
    assert settings["reasoning_effort"] is None
    assert settings["model"] == "lm_studio/test/gemma-stub"
    assert "lmstudio-bug-tracker" in settings.get("notes", "")


def test_hook_is_idempotent_when_settings_already_cached(
    nb_with_client: NotebookConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second invocation (settings file present) must NOT re-run papermill."""
    from notebook_agent import model_settings as ms_mod
    from notebook_agent import notebook_exec as nbx_mod
    from notebook_agent.model_settings import settings_notebook_path

    monkeypatch.setattr(
        ms_mod,
        "pick_loaded_model",
        lambda client, prefer=None: LoadedModel(id=prefer or client.model, state="loaded"),
    )
    monkeypatch.setattr(
        "notebook_agent.model_settings.pick_loaded_model",
        lambda client, prefer=None: LoadedModel(id=prefer or client.model, state="loaded"),
    )

    # Pre-seed cached settings on disk.
    assert nb_with_client.client is not None
    target = settings_notebook_path(
        nb_with_client.client.model, sessions_root=Path(nb_with_client.sessions_root)
    )
    write_settings(target, {"reasoning_effort": "low"})

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("execute_notebook must not be called when settings exist")

    monkeypatch.setattr(nbx_mod, "execute_notebook", boom)

    magics_mod._ensure_model_settings_discovered()  # must not raise
