"""Unit tests for :mod:`notebook_agent.model_settings`."""

from __future__ import annotations

from pathlib import Path

import pytest

from notebook_agent.model_settings import (
    model_slug,
    read_settings,
    settings_notebook_path,
    write_settings,
)


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("lm_studio/google/gemma-4-31b", "google-gemma-4-31b"),
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("anthropic/claude-opus-4.7", "claude-opus-4.7"),
        ("ollama/llama3", "llama3"),
        ("bare-model-id", "bare-model-id"),
        ("Some Model With Spaces", "some-model-with-spaces"),
    ],
)
def test_model_slug_normalizes_provider_prefixes_and_punctuation(
    model_id: str, expected: str
) -> None:
    assert model_slug(model_id) == expected


def test_settings_notebook_path_is_under_sessions_root(tmp_path: Path) -> None:
    p = settings_notebook_path("lm_studio/google/gemma-4-31b", sessions_root=tmp_path)
    assert p == tmp_path / "google-gemma-4-31b" / "model_settings.ipynb"


def test_write_then_read_round_trips_parameters(tmp_path: Path) -> None:
    p = tmp_path / "settings" / "model_settings.ipynb"
    settings = {
        "model": "google/gemma-4-31b",
        "reasoning_effort": "off",
        "max_tokens": 8192,
        "temperature": 0.0,
        "supports_reasoning_effort": False,
    }
    write_settings(p, settings, notes="Probed against LM Studio 0.3.x")
    assert p.exists()
    loaded = read_settings(p)
    assert loaded == settings


def test_read_settings_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert read_settings(tmp_path / "nope.ipynb") == {}


def test_read_settings_ignores_non_literal_parameters_cell(tmp_path: Path) -> None:
    # Hand-craft a notebook with a parameters cell that includes a call —
    # only the literal assignments should be picked up.
    import json

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"tags": ["parameters"]},
                "source": [
                    "import os\n",
                    "reasoning_effort = 'off'\n",
                    "computed = os.environ.get('X', 'y')\n",
                    "model = 'gemma'\n",
                ],
                "outputs": [],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = tmp_path / "model_settings.ipynb"
    p.write_text(json.dumps(nb))
    loaded = read_settings(p)
    assert loaded == {"reasoning_effort": "off", "model": "gemma"}
