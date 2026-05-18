"""The agent expressed as a DSPy program (signatures + composed module).

This is the **optimization surface** of ``notebook_agent``. Every LLM-driven
decision the agent makes (route, plan, parameter extraction, code generation,
result synthesis, repair) is a typed :class:`dspy.Signature` evaluated by a
named :class:`dspy.Predict` sub-module of :class:`NotebookAgentProgram`.

Because every LLM step is a DSPy module with declared inputs and outputs,
DSPy optimizers (``MIPROv2``, ``GEPA``, ``BootstrapFewShot``, …) can compile
this program against a trainset + metric. See
:mod:`notebook_agent.optimize` for the wrappers.

The program *does not* run notebooks itself — it produces the structured
intentions (plan, generated source, parameters, answer text) that the
:func:`notebook_agent.run_task` orchestrator stitches together with the
Papermill executor.

Typical use::

    from notebook_agent import NotebookAgentProgram, configure_dspy
    from notebook_agent.litellm_client import LiteLLMClient

    configure_dspy(LiteLLMClient())          # uses LM Studio via env vars
    prog = NotebookAgentProgram()
    pred = prog(request="Count words in: hello world")
    pred.plan, pred.generated_code, pred.answer
"""

from __future__ import annotations

import dspy  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class ChooseSkill(dspy.Signature):
    """Pick the local skill that best matches the user's request.

    The catalog is a JSON list of objects with ``skill_id``, ``name``,
    ``description``, and ``tags``. Output the chosen ``skill_id`` exactly,
    or the literal string ``"none"`` when no skill in the catalog applies
    well enough that running it would help the user.
    """

    request: str = dspy.InputField(desc="the user's natural-language request")
    catalog: str = dspy.InputField(
        desc="JSON list of available skills, each with skill_id/name/description/tags"
    )
    chosen_skill_id: str = dspy.OutputField(
        desc="skill_id from the catalog, or 'none' if no skill applies"
    )


class RouteTask(dspy.Signature):
    """Decide the top-level strategy for handling the user's request.

    Possible strategies:
      * ``retrieve_first`` — try to match an existing local skill before any LLM work
      * ``generate`` — go straight to LLM code generation
    """

    request: str = dspy.InputField(desc="the user's natural-language request")
    strategy: str = dspy.OutputField(
        desc="one of: retrieve_first, generate"
    )


class PlanTask(dspy.Signature):
    """Draft a short, visible TODO plan (3-6 bullets) for the user's request."""

    request: str = dspy.InputField(desc="the user's natural-language request")
    plan: str = dspy.OutputField(
        desc="newline-separated TODO items; keep each item short and actionable"
    )


class GenerateCode(dspy.Signature):
    """Write a small Python snippet that solves the request.

    The snippet must bind its answer to a variable named ``result`` that is
    JSON-serialisable. Use only the Python standard library. Do not call
    ``input()``, do not access the network, do not write files. The snippet
    runs inside a Jupyter notebook cell.
    """

    request: str = dspy.InputField(desc="the user's natural-language request")
    plan: str = dspy.InputField(desc="the agent's short TODO plan for this task")
    python_code: str = dspy.OutputField(
        desc="a single Python snippet (no fences) that assigns to `result`"
    )


class ExtractParameters(dspy.Signature):
    """Infer the parameter values for a chosen skill from the request.

    Output a single JSON object whose keys match the parameter names exactly.
    """

    request: str = dspy.InputField(desc="the user's natural-language request")
    parameter_schema: str = dspy.InputField(
        desc="parameter schema as a bulleted list: '- name (type): description'"
    )
    current_date: str = dspy.InputField(
        desc="ISO date the agent considers 'today' for relative-time resolution"
    )
    parameters_json: str = dspy.OutputField(
        desc="a single JSON object mapping parameter name -> value"
    )


class RepairNotebook(dspy.Signature):
    """Suggest a minimal fix for a failed notebook cell.

    The fix should be one or two lines of Python that, when prepended to the
    failing cell, resolves the error.
    """

    error_text: str = dspy.InputField(desc="the cell-error traceback / message")
    fix: str = dspy.OutputField(desc="one-or-two-line Python fix")


class SynthesizeAnswer(dspy.Signature):
    """Turn the structured ``result`` payload into a short, human answer.

    Prefer a single sentence. If the result has a ``message`` field, you may
    return it verbatim.
    """

    result_json: str = dspy.InputField(desc="the JSON result payload")
    answer: str = dspy.OutputField(desc="short human-readable answer")


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------


class NotebookAgentProgram(dspy.Module):
    """Composed DSPy program for the notebook-native coding agent.

    Each LLM-driven step is exposed as a named sub-module so DSPy optimizers
    can target them individually (``self.router``, ``self.planner``,
    ``self.code_generator``, ``self.param_extractor``, ``self.repairer``,
    ``self.synthesizer``).

    ``forward(request)`` returns a :class:`dspy.Prediction` with the fields
    ``strategy``, ``plan`` (a list of bullets), ``generated_code``, and
    ``answer``. The other sub-modules are invoked on demand by the
    orchestrator in :mod:`notebook_agent.agent`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.router = dspy.Predict(RouteTask)
        self.planner = dspy.Predict(PlanTask)
        self.skill_chooser = dspy.Predict(ChooseSkill)
        self.code_generator = dspy.Predict(GenerateCode)
        self.param_extractor = dspy.Predict(ExtractParameters)
        self.repairer = dspy.Predict(RepairNotebook)
        self.synthesizer = dspy.Predict(SynthesizeAnswer)

    # ------------------------------------------------------------------
    # Top-level forward (used by optimizers)
    # ------------------------------------------------------------------

    def forward(self, request: str, catalog: str = "[]") -> dspy.Prediction:
        plan_pred = self.planner(request=request)
        plan_items = _split_plan(plan_pred.plan)
        choose_pred = self.skill_chooser(request=request, catalog=catalog)
        chosen = str(choose_pred.chosen_skill_id).strip()
        code_pred = self.code_generator(request=request, plan=plan_pred.plan)
        return dspy.Prediction(
            plan=plan_items,
            plan_raw=plan_pred.plan,
            chosen_skill_id=chosen,
            generated_code=_strip_fences(code_pred.python_code),
        )

    # ------------------------------------------------------------------
    # Convenience wrappers used by the orchestrator
    # ------------------------------------------------------------------

    def plan(self, request: str) -> list[str]:
        return _split_plan(self.planner(request=request).plan)

    def route(self, request: str) -> str:
        return str(self.router(request=request).strategy).strip().lower()

    def choose_skill(self, request: str, catalog: str) -> str:
        return str(self.skill_chooser(request=request, catalog=catalog).chosen_skill_id).strip()

    def generate_code(self, request: str, plan: list[str] | str | None = None) -> str:
        plan_str = plan if isinstance(plan, str) else "\n".join(plan or [])
        out = self.code_generator(request=request, plan=plan_str)
        return _strip_fences(out.python_code)

    def extract_parameters(
        self,
        request: str,
        schema_text: str,
        *,
        current_date: str,
    ) -> str:
        out = self.param_extractor(
            request=request,
            parameter_schema=schema_text,
            current_date=current_date,
        )
        return out.parameters_json

    def repair(self, error_text: str) -> str:
        return str(self.repairer(error_text=error_text).fix)

    def synthesize(self, result_json: str) -> str:
        return str(self.synthesizer(result_json=result_json).answer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_plan(text: str) -> list[str]:
    """Split a plan string returned by the LLM into bullets.

    Accepts ``-`` / ``*`` / ``1.`` style bullets, blank-line separated paragraphs,
    or newline-separated items.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip common bullet markers.
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        else:
            # numeric bullets: '1.', '2)', etc.
            if len(line) > 2 and line[0].isdigit():
                i = 1
                while i < len(line) and line[i].isdigit():
                    i += 1
                if i < len(line) and line[i] in ".):":
                    line = line[i + 1 :].strip()
        if line:
            out.append(line)
    return out


def _strip_fences(text: str) -> str:
    """Drop ```python ... ``` fences if the model emitted them."""
    import re

    s = (text or "").strip()
    m = re.search(r"```(?:python|py)?\s*\n(?P<body>.*?)\n```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group("body").strip()
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


__all__ = [
    "ChooseSkill",
    "ExtractParameters",
    "GenerateCode",
    "NotebookAgentProgram",
    "PlanTask",
    "RepairNotebook",
    "RouteTask",
    "SynthesizeAnswer",
    "split_plan",
    "strip_fences",
]


# Re-export the helpers — they're used by the tests and orchestrator.
def split_plan(text: str) -> list[str]:
    return _split_plan(text)


def strip_fences(text: str) -> str:
    return _strip_fences(text)
