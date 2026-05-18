"""DSPy module stubs (Section 14.12).

These are intentionally minimal placeholders. They expose the signatures the
spec calls out so future work can plug DSPy in without touching the agent
loop. When invoked without DSPy installed, the stubs degrade to deterministic
behavior based on the configured :class:`LiteLLMClient` (which itself can use
the FakeProvider for tests).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .litellm_client import LiteLLMClient, LLMUnavailableError
from .skills import Skill, SkillRepository

# ---------------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------------


@dataclass
class _ModuleBase:
    llm: LiteLLMClient | None = None

    def _llm(self) -> LiteLLMClient:
        return self.llm or LiteLLMClient()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class TaskRouter(_ModuleBase):
    """Decide which top-level strategy to use for a request.

    The default stub always picks ``"retrieve"`` first; a future DSPy
    implementation would replace this with a learned router.
    """

    def __call__(self, request: str) -> dict[str, Any]:
        return {"request": request, "strategy": "retrieve_first"}


class SkillRetriever(_ModuleBase):
    """Wrap :class:`SkillRepository` to mirror a future DSPy signature."""

    def __init__(self, repo: SkillRepository, **kwargs: Any) -> None:  # noqa: D401
        super().__init__(**kwargs)
        self.repo = repo

    def __call__(self, query: str, top_k: int = 5) -> list[Skill]:
        return [r.skill for r in self.repo.search(query, top_k=top_k)]


class SkillToNotebookTransformer(_ModuleBase):
    """Default transformer: just calls :func:`notebook_agent.transform.transform_skill_to_notebook`."""

    def __call__(self, skill: Skill, output_path: Any) -> Any:
        from .transform import transform_skill_to_notebook

        return transform_skill_to_notebook(skill, output_path)


class NotebookRepairer(_ModuleBase):
    """Provide a diagnostic-only repair suggestion via the configured LLM (if any)."""

    def __call__(self, error_text: str) -> str:
        try:
            resp = self._llm().complete(
                "Suggest a single-line Python fix for this notebook error.\n\n" + error_text,
                max_tokens=64,
            )
            return resp.text.strip()
        except LLMUnavailableError:
            return ""


class ResultSynthesizer(_ModuleBase):
    """Turn a structured result into a short human answer."""

    def __call__(self, result_payload: dict[str, Any]) -> str:
        if "message" in result_payload and isinstance(result_payload["message"], str):
            return result_payload["message"]
        try:
            return self._llm().complete(
                "Summarize this JSON in a one-line answer:\n" + json.dumps(result_payload, sort_keys=True),
                max_tokens=64,
            ).text
        except LLMUnavailableError:
            return json.dumps(result_payload, sort_keys=True)
