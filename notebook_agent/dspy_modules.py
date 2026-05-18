"""DSPy module stubs (Section 14.12).

These are intentionally minimal placeholders. They expose the signatures the
spec calls out so future work can plug DSPy in without touching the agent
loop. When invoked without DSPy installed, the stubs degrade to deterministic
behavior based on the configured :class:`LiteLLMClient` (which itself can use
the FakeProvider for tests).
"""

from __future__ import annotations

import json
import re
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


class ParameterExtractor(_ModuleBase):
    """Infer skill parameter values from a natural-language request.

    Given a request string and a :class:`~notebook_agent.skills.Skill`, asks
    the LLM to produce a JSON object whose keys match the skill's input schema.
    Returned values are merged with any caller-supplied parameters (caller wins
    on conflicts).

    If no LLM is configured, or the LLM response cannot be parsed, returns
    an empty dict so the caller's existing parameters (and notebook defaults)
    are used unchanged.
    """

    SYSTEM_PROMPT = (
        "You are a parameter-extraction assistant. Your ONLY job is to produce "
        "a JSON object that fills in the named parameters from the user's request. "
        "Output one JSON object and nothing else: no prose, no markdown, no code fences, "
        "no explanations. Resolve any dynamic expressions in the request (e.g. "
        "'today', 'now', 'current date') to concrete string values using the "
        "current date provided in the user message."
    )

    def __call__(self, request: str, skill: Skill) -> dict[str, Any]:
        self.last_error: str | None = None
        schema = skill.manifest.get("input_schema") or {}
        props = (schema if isinstance(schema, dict) else {}).get("properties") or {}
        if not props:
            return {}

        schema_lines = []
        for name, spec in props.items():
            if isinstance(spec, dict):
                typ = spec.get("type", "any")
                desc = spec.get("description", "")
                schema_lines.append(f"  - {name} ({typ}){': ' + desc if desc else ''}")
            else:
                schema_lines.append(f"  - {name}")

        from datetime import date
        today = date.today().isoformat()
        prompt = (
            f"Current date: {today}\n\n"
            f"Request: {request}\n\n"
            "Parameters to fill:\n" + "\n".join(schema_lines) + "\n\n"
            "Respond with a single JSON object mapping each parameter name to its value."
        )
        try:
            resp = self._llm().complete(
                prompt,
                system=self.SYSTEM_PROMPT,
                # Thinking models (e.g. Gemma) need substantial headroom: they
                # stream chain-of-thought into ``content`` before emitting the
                # final JSON. 256 tokens is not enough.
                max_tokens=2048,
            )
            text = resp.text.strip()
            # Strip markdown code fences if the model added them anyway.
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
            data = _extract_json_object(text)
            if isinstance(data, dict):
                # Keep only declared keys to avoid leaking reasoning artifacts.
                return {k: v for k, v in data.items() if k in props}
            return {}
        except LLMUnavailableError as exc:
            self.last_error = f"LLM unavailable: {exc!s}"
        except json.JSONDecodeError as exc:
            self.last_error = f"could not parse JSON from LLM response: {exc!s}"
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"parameter extraction failed: {type(exc).__name__}: {exc!s}"
        return {}


def _extract_json_object(text: str) -> Any:
    """Find and parse the *last* balanced ``{...}`` object in ``text``.

    Thinking models often emit several JSON-looking fragments inside their
    reasoning; the final answer is conventionally the last one. We scan from
    the right so we pick that, and we balance braces so we never grab a
    truncated tail.
    """
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)
    # Quick path: whole string parses.
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Walk right-to-left looking for a closing brace, then find the matching
    # opening brace honoring nesting and string literals.
    for end in range(len(s) - 1, -1, -1):
        if s[end] != "}":
            continue
        depth = 0
        in_str = False
        esc = False
        for start in range(end, -1, -1):
            ch = s[start]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    candidate = s[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try next closing brace
        # else: didn't balance, try the next earlier '}'.
    raise json.JSONDecodeError("no JSON object found", s, 0)
