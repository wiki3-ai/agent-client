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
    """First call: hook should invoke execute_notebook and apply the
    discovered ``reasoning_effort`` to the live client."""
    from notebook_agent import model_settings as ms_mod
    from notebook_agent import notebook_exec as nbx_mod

    # Pretend LM Studio reports our model as loaded.
    monkeypatch.setattr(
        ms_mod,
        "pick_loaded_model",
        lambda client, prefer=None: LoadedModel(id=prefer or client.model, state="loaded"),
    )
    # Re-bind the symbol the hook imports lazily, too.
    monkeypatch.setattr(
        "notebook_agent.model_settings.pick_loaded_model",
        lambda client, prefer=None: LoadedModel(id=prefer or client.model, state="loaded"),
    )

    calls: list[dict] = []

    def fake_execute_notebook(
        skill_nb: Path,
        *,
        parameters: dict,
        output_path: Path,
        run_dir: Path,
        **_: object,
    ) -> Path:
        calls.append(
            {
                "skill_nb": Path(skill_nb),
                "parameters": dict(parameters),
                "output_path": Path(output_path),
                "run_dir": Path(run_dir),
            }
        )
        # Simulate the skill's side-effect: write the canonical settings
        # notebook the hook will read back.
        from notebook_agent.model_settings import settings_notebook_path

        target = settings_notebook_path(
            parameters["target_model"],
            sessions_root=Path(parameters["sessions_root"]),
        )
        write_settings(target, {"reasoning_effort": "off", "supports_reasoning_effort": True})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("{}")
        return Path(output_path)

    monkeypatch.setattr(nbx_mod, "execute_notebook", fake_execute_notebook)

    magics_mod._ensure_model_settings_discovered()

    assert len(calls) == 1, "discovery skill should run exactly once"
    call = calls[0]
    assert call["skill_nb"].name == "skill.ipynb"
    assert call["skill_nb"].parent.name == "discover_model_settings"
    assert call["parameters"]["target_model"] == "lm_studio/test/gemma-stub"
    assert call["parameters"]["base_url"] == "http://stub.invalid:1234/v1"
    assert call["parameters"]["sessions_root"] == str(Path(nb_with_client.sessions_root))

    # The hook should have applied the discovered setting to the live client.
    assert nb_with_client.client is not None
    assert nb_with_client.client.reasoning_effort == "off"

    out = capsys.readouterr().out
    assert "Discovering settings" in out
    assert "reasoning_effort='off'" in out


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
