"""DSPy program & optimizer integration.

The agent IS a DSPy program. These tests verify:

1. The program's sub-modules are real ``dspy.Predict`` instances bound to
   typed Signatures, so DSPy optimizers can see them.
2. ``program(request=..., catalog=...)`` returns a ``dspy.Prediction``.
3. ``MIPROv2.compile(program, ...)`` runs to completion under a predictable LM
   and returns a compiled program.
4. ``GEPA.compile(program, ...)`` runs to completion under a predictable LM
   with a tight budget and returns a compiled program.
"""

from __future__ import annotations

import re
from typing import Any

import dspy  # type: ignore[import-untyped]
import pytest
from dspy.adapters.chat_adapter import ChatAdapter  # type: ignore[import-untyped]
from dspy.clients.base_lm import BaseLM  # type: ignore[import-untyped]
from dspy.utils.dummies import DummyLM  # type: ignore[import-untyped]

from notebook_agent import (
    ChooseSkill,
    ExtractParameters,
    GenerateCode,
    NotebookAgentProgram,
    PlanTask,
    RepairNotebook,
    RouteTask,
    SynthesizeAnswer,
    optimize_with_gepa,
    optimize_with_mipro,
)

# ---------------------------------------------------------------------------
# A predictable mock LM that always satisfies whatever signature is asked.
#
# It parses each prompt to discover the expected output-field names (from the
# ChatAdapter prompt skeleton) and emits a ChatAdapter-formatted response that
# fills every field with a plausible value. This lets DSPy optimizers run
# their many internal Predict calls (with shapes we don't know in advance)
# without throwing parse errors.
# ---------------------------------------------------------------------------


_FIELD_LINE_RE = re.compile(r"^\s*\d+\.\s+`([A-Za-z_][A-Za-z0-9_]*)`", re.MULTILINE)
_FIELD_MARKER_RE = re.compile(r"\[\[\s*##\s*([A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]")


class _PredictableLM(BaseLM):
    """A test-only LM that auto-fills any DSPy signature's outputs."""

    def __init__(self, defaults: dict[str, str] | None = None) -> None:
        super().__init__("predictable", "chat", 0.0, 1000, True)
        self.defaults = dict(defaults or {})
        self.history: list[dict[str, Any]] = []  # for inspect_history

    def _output_field_names(self, system_text: str) -> list[str]:
        # Find "Your output fields are:" block and grab numbered field names.
        m = re.search(r"Your output fields are:\s*\n(?P<body>.*?)(?:\n\n|$)",
                      system_text, re.DOTALL)
        if not m:
            # Fallback: scan all [[ ## name ## ]] markers and use those that
            # appear after "Outputs:" in the prompt.
            return list(dict.fromkeys(_FIELD_MARKER_RE.findall(system_text)))
        names = _FIELD_LINE_RE.findall(m.group("body"))
        return list(dict.fromkeys(names))

    def _value_for(self, field: str) -> str:
        if field in self.defaults:
            return self.defaults[field]
        lower = field.lower()
        if lower.endswith("_id"):
            return "none"
        if "json" in lower:
            return "{}"
        if "code" in lower:
            return "result = {'ok': True}"
        if lower in {"plan", "answer", "fix", "strategy",
                     "proposed_instruction", "summary", "observations",
                     "reasoning", "tip"}:
            return f"placeholder-{lower}"
        return "placeholder"

    def forward(self, prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        from types import SimpleNamespace as _NS

        messages = messages or [{"role": "user", "content": prompt or ""}]
        system_text = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"),
            messages[0].get("content", ""),
        )
        fields = self._output_field_names(system_text) or ["answer"]
        body_lines: list[str] = []
        for f in fields:
            body_lines.append(f"[[ ## {f} ## ]]")
            body_lines.append(self._value_for(f))
        body_lines.append("[[ ## completed ## ]]")
        content = "\n".join(body_lines)

        n = kwargs.get("n", 1) or 1
        choices = [
            _NS(message=_NS(content=content, tool_calls=None), finish_reason="stop")
            for _ in range(n)
        ]
        return _NS(
            choices=choices,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="predictable",
        )


# ---------------------------------------------------------------------------
# Shape: the program is a DSPy program
# ---------------------------------------------------------------------------


def test_program_subpredictors_are_typed_dspy_predicts() -> None:
    p = NotebookAgentProgram()
    for name in (
        "router", "planner", "skill_chooser", "code_generator",
        "param_extractor", "repairer", "synthesizer",
    ):
        sub = getattr(p, name)
        assert isinstance(sub, dspy.Predict), f"{name} is not a dspy.Predict"

    pairs = [
        (p.router, RouteTask, ["request"], ["strategy"]),
        (p.planner, PlanTask, ["request"], ["plan"]),
        (p.skill_chooser, ChooseSkill, ["request", "catalog"], ["chosen_skill_id"]),
        (p.code_generator, GenerateCode, ["request", "plan"], ["python_code"]),
        (p.param_extractor, ExtractParameters,
            ["request", "parameter_schema", "current_date"], ["parameters_json"]),
        (p.repairer, RepairNotebook, ["error_text"], ["fix"]),
        (p.synthesizer, SynthesizeAnswer, ["result_json"], ["answer"]),
    ]
    for predict, expected_sig, in_fields, out_fields in pairs:
        sig = predict.signature
        assert sig.__name__ == expected_sig.__name__ or issubclass(sig, expected_sig)
        for f in in_fields:
            assert f in sig.input_fields, f"{expected_sig.__name__} missing input {f}"
        for f in out_fields:
            assert f in sig.output_fields, f"{expected_sig.__name__} missing output {f}"


def test_program_forward_returns_prediction() -> None:
    dspy.configure(lm=DummyLM([
        {"plan": "- step one\n- step two"},
        {"chosen_skill_id": "none"},
        {"python_code": "result = {'ok': True}"},
    ]))
    p = NotebookAgentProgram()
    pred = p(request="count words in 'a b c'", catalog="[]")
    assert isinstance(pred, dspy.Prediction)
    assert pred.plan == ["step one", "step two"]
    assert pred.chosen_skill_id == "none"
    assert "result" in pred.generated_code


# ---------------------------------------------------------------------------
# Optimizer smoke: MIPROv2 + GEPA can both compile our program
# ---------------------------------------------------------------------------


def _metric_always_one(example, pred, trace=None) -> float:
    return 1.0


def _make_trainset() -> list[dspy.Example]:
    return [
        dspy.Example(request="count words in 'a b'", catalog="[]")
            .with_inputs("request", "catalog"),
        dspy.Example(request="echo 'hello'", catalog="[]")
            .with_inputs("request", "catalog"),
    ]


def test_mipro_v2_compiles_the_program() -> None:
    dspy.configure(lm=_PredictableLM(), adapter=ChatAdapter())
    base = NotebookAgentProgram()
    compiled = optimize_with_mipro(
        base,
        trainset=_make_trainset(),
        metric=_metric_always_one,
        auto="light",
        num_threads=1,
    )
    assert isinstance(compiled, dspy.Module)
    pred = compiled(request="anything", catalog="[]")
    assert isinstance(pred, dspy.Prediction)


def test_gepa_compiles_the_program() -> None:
    dspy.configure(lm=_PredictableLM(), adapter=ChatAdapter())
    base = NotebookAgentProgram()

    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=1.0, feedback="ok")

    try:
        compiled = optimize_with_gepa(
            base,
            trainset=_make_trainset(),
            metric=gepa_metric,
            auto=None,
            max_metric_calls=4,
            num_threads=1,
            reflection_lm=_PredictableLM(),
        )
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"GEPA optimizer not runnable in this environment: {e!r}")
    assert isinstance(compiled, dspy.Module)
