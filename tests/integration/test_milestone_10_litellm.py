"""Milestone 10 acceptance test: LiteLLM client and DSPy stubs.

Per spec §16 / §18:
* Non-generation workflows pass without LM Studio (covered by all other tests).
* Generation attempts fail gracefully with a clear unavailable-model error.
* When the FakeProvider is configured, calls succeed deterministically and are
  recorded in ``logs/lm_calls.jsonl``.
* The live LM-Studio test is marked ``live`` and skipped unless explicitly
  enabled by setting ``NOTEBOOK_AGENT_LIVE_LM=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from notebook_agent.dspy_modules import (
    NotebookRepairer,
    ResultSynthesizer,
    SkillRetriever,
    SkillToNotebookTransformer,
    TaskRouter,
)
from notebook_agent.litellm_client import LiteLLMClient, LLMUnavailableError
from notebook_agent.skills import SkillRepository
from notebook_agent.transform import builtin_skills_root


def test_fake_provider_round_trip_and_logging(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "lm_calls.jsonl"
    client = LiteLLMClient(provider="fake", lm_calls_log=log, fake_response="hello back")
    resp = client.complete("hello?")
    assert resp.text == "hello back"
    assert resp.provider == "fake"
    # Log file written.
    assert log.exists()
    lines = log.read_text().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["provider"] == "fake"
    assert payload["success"] is True
    assert "api_key" not in payload  # secrets must not leak


def test_unconfigured_lm_studio_fails_gracefully(tmp_path: Path) -> None:
    # Point at a definitely-unreachable URL so the call must fail. The wrapper
    # should normalize that into LLMUnavailableError and still write a log line.
    log = tmp_path / "logs" / "lm_calls.jsonl"
    client = LiteLLMClient(
        provider="lm_studio",
        base_url="http://127.0.0.1:1/",  # impossible port
        api_key="lm-studio",
        lm_calls_log=log,
    )
    with pytest.raises(LLMUnavailableError):
        client.complete("hello", max_tokens=4)
    assert log.exists()
    last = json.loads(log.read_text().splitlines()[-1])
    assert last["success"] is False
    assert "error" in last
    # API key must never appear in the log line.
    assert "lm-studio" not in log.read_text()


def test_dspy_stubs_with_fake_llm(tmp_path: Path) -> None:
    repo = SkillRepository([builtin_skills_root()])
    fake = LiteLLMClient(provider="fake", fake_response="done.")

    router = TaskRouter(llm=fake)
    assert router("Echo hi")["strategy"] == "retrieve_first"

    retriever = SkillRetriever(repo, llm=fake)
    skills = retriever("echo message")
    assert any(s.skill_id == "core.echo" for s in skills)

    transformer = SkillToNotebookTransformer(llm=fake)
    out = transformer(skills[0], tmp_path / "skill.ipynb")
    assert out.exists()

    repairer = NotebookRepairer(llm=fake)
    assert repairer("NameError: name 'x' is not defined") == "done."

    syn = ResultSynthesizer(llm=fake)
    # When there is a 'message' field, the synthesizer returns it directly.
    assert syn({"message": "ahoy"}) == "ahoy"
    # When there is no 'message' field, it falls back to the LLM (FakeProvider).
    assert syn({"value": 7}) == "done."


@pytest.mark.live
def test_live_lm_studio_completion(tmp_path: Path) -> None:
    """Live LM Studio test (only runs when explicitly enabled)."""
    if not os.environ.get("NOTEBOOK_AGENT_LIVE_LM"):
        pytest.skip("NOTEBOOK_AGENT_LIVE_LM not set")
    log = tmp_path / "lm_calls.jsonl"
    client = LiteLLMClient(provider="lm_studio", lm_calls_log=log)
    resp = client.complete("Say 'pong' and nothing else.", max_tokens=8)
    assert resp.text
    assert log.exists()
