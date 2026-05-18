"""End-to-end agent orchestration (DSPy program drives every LLM step).

This module is the only place that knows about the lifecycle of a notebook
task: filesystem layout, plan persistence, skill catalog assembly, the
:class:`~notebook_agent.program.NotebookAgentProgram` invocation, Papermill
execution, and the canonical state written back into the executed notebook.

Every LLM-driven decision goes through DSPy. There is **no** fallback path
that hand-rolls prompts and calls a completion API directly: if the LM is
unreachable, the task fails with a clear error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import dspy  # type: ignore[import-untyped]
import nbformat

from ._clock import iso_now
from .budget import Budget, BudgetExhaustedError, BudgetTracker
from .codegen import GeneratedCode, build_generated_notebook, validate_snippet
from .dspy_lm import configure_dspy
from .litellm_client import LiteLLMClient
from .notebook_exec import NotebookExecutionResult, execute_notebook
from .planner import StageDecision, decompose_request
from .program import NotebookAgentProgram, split_plan
from .repair import repair_and_rerun
from .skills import Skill, SkillRepository
from .task_graph import Task, create_root_task
from .transform import builtin_skills_root, transform_skill_to_notebook


@dataclass
class AgentResult:
    """Final result returned by :func:`run_task`."""

    task: Task
    success: bool
    stage_used: str | None
    manifest: dict[str, Any]
    result_payload: dict[str, Any] | None
    answer: str | None
    execution: NotebookExecutionResult | None = None
    error: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    turns_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "directory": str(self.task.directory),
            "success": self.success,
            "stage_used": self.stage_used,
            "manifest": self.manifest,
            "result": self.result_payload,
            "answer": self.answer,
            "error": self.error,
            "plan": self.plan,
            "turns_used": self.turns_used,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skill_repository(extra_skill_dirs: list[Path | str] | None) -> SkillRepository:
    roots: list[Path | str] = [builtin_skills_root()]
    if extra_skill_dirs:
        roots.extend(extra_skill_dirs)
    cwd_skills = Path.cwd() / "skills"
    if cwd_skills.exists() and cwd_skills not in [Path(r) for r in roots]:
        roots.append(cwd_skills)
    return SkillRepository(roots)


def _ensure_dspy_configured(client: LiteLLMClient | None) -> Any:
    """Make sure ``dspy.settings.lm`` is set; return the active LM.

    If the caller has already configured DSPy (e.g. at the top of their
    notebook), we respect that. Otherwise we configure from ``client`` or
    from env-driven defaults.
    """
    lm = getattr(dspy.settings, "lm", None)
    if lm is not None and client is None:
        return lm
    return configure_dspy(client or LiteLLMClient())


def _looks_like_continuation(prompt: str) -> bool:
    s = (prompt or "").strip().lower()
    triggers = (
        "continue", "keep going", "try again", "not satisfactory",
        "that's wrong", "thats wrong", "wrong", "fix it", "again", "more",
    )
    return any(t in s for t in triggers)


def _resolve_skill(repo: SkillRepository, chosen_id: str) -> Skill | None:
    if not chosen_id or chosen_id.strip().lower() in {"none", "null", "n/a", ""}:
        return None
    return repo.find(chosen_id.strip())


def _summarize(stage: str, skill: Skill | None, payload: dict[str, Any] | None, err: dict[str, Any] | None) -> str:
    if err is not None:
        return f"Task failed: {err.get('type', 'Error')}: {err.get('message', '')}".strip()
    if payload is None:
        return "Task completed; no structured result was produced."
    if "message" in payload and isinstance(payload["message"], str):
        return payload["message"]
    if "answer" in payload and isinstance(payload["answer"], str):
        return payload["answer"]
    if "value" in payload and not isinstance(payload["value"], (dict, list)):
        return str(payload["value"])
    pretty = json.dumps(payload, indent=2, sort_keys=True)
    title = (skill.name if skill else stage.title()) or "Result"
    return f"{title} result:\n\n```json\n{pretty}\n```"


def _write_notebook_state(nb_path: Path, state: dict[str, Any]) -> None:
    """Merge *state* into the executed notebook's metadata under ``notebook_agent``."""
    if not nb_path.exists():
        return
    try:
        nb = nbformat.read(str(nb_path), as_version=4)
    except Exception:
        return
    md = nb.metadata.setdefault("notebook_agent", {})
    for k, v in state.items():
        md[k] = v
    with nb_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def _read_notebook_state(nb_path: Path) -> dict[str, Any]:
    if not nb_path.exists():
        return {}
    try:
        nb = nbformat.read(str(nb_path), as_version=4)
    except Exception:
        return {}
    return dict(nb.metadata.get("notebook_agent") or {})


def _schema_text(skill: Skill) -> str:
    schema = skill.manifest.get("input_schema") or {}
    props = (schema if isinstance(schema, dict) else {}).get("properties") or {}
    if not props:
        return ""
    lines: list[str] = []
    for name, spec in props.items():
        if isinstance(spec, dict):
            typ = spec.get("type", "any")
            desc = spec.get("description", "")
            lines.append(f"- {name} ({typ}){': ' + desc if desc else ''}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _parse_extracted_params(text: str, schema_props: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON object produced by ``ExtractParameters``; keep declared keys only."""
    s = (text or "").strip()
    # Strip markdown fences if any thinking-model leaks them in.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # Fall back to last balanced { ... }.
        data = _last_balanced_object(s)
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in schema_props}


def _last_balanced_object(text: str) -> Any:
    if not text:
        return None
    for end in range(len(text) - 1, -1, -1):
        if text[end] != "}":
            continue
        depth = 0
        in_str = False
        esc = False
        for start in range(end, -1, -1):
            ch = text[start]
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
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_task(
    request: str,
    *,
    runs_root: Path | str = "runs",
    parameters: dict[str, Any] | None = None,
    budget: Budget | dict[str, Any] | None = None,
    title: str | None = None,
    skill_dirs: list[Path | str] | None = None,
    llm: LiteLLMClient | None = None,
    program: NotebookAgentProgram | None = None,
    auto_repair: bool = True,
    continue_from: AgentResult | Task | str | Path | None = None,
    max_autonomous_turns: int = 6,
) -> AgentResult:
    """Run a task end-to-end via the DSPy :class:`NotebookAgentProgram`.

    Required: ``request``. Everything else is optional.

    The agent assumes DSPy is configured (``dspy.configure(lm=...)``); if no
    LM is configured we build one from ``llm`` or from
    ``NOTEBOOK_AGENT_*`` env vars. ``program`` may be supplied to use a
    pre-compiled program (e.g. one returned by ``optimize_with_mipro``).
    """
    parameters = dict(parameters or {})

    if isinstance(budget, dict):
        bdict = dict(budget)
    elif budget is None:
        bdict = {}
    else:
        bdict = budget.to_dict()
    bdict.setdefault("max_autonomous_turns", max_autonomous_turns)
    budget_obj = Budget.from_dict(bdict)

    parent_task: Task | None = None
    prior_state: dict[str, Any] = {}
    if continue_from is not None:
        parent_task = _resolve_continuation(continue_from)
        if parent_task is not None:
            prior_state = _read_notebook_state(parent_task.executed_notebook)

    task = create_root_task(
        runs_root,
        title=title or (request.strip().splitlines()[0] if request.strip() else "task"),
        request=request,
        parameters=parameters,
        budget=budget_obj,
    )
    log = task.event_log()
    log.append("task_started", task_id=task.task_id)
    if parent_task is not None:
        log.append("task_continuation", parent_task_id=parent_task.task_id, parent_dir=str(parent_task.directory))
    task.update_status("running")
    tracker = BudgetTracker(task.budget)
    decision = StageDecision()
    repo = _skill_repository(skill_dirs)

    manifest = task.read_manifest()
    manifest["started_at"] = iso_now()
    manifest["continued_from"] = str(parent_task.directory) if parent_task else None

    # ----- Configure DSPy + build the program. -----
    _ensure_dspy_configured(llm)
    if program is None:
        program = NotebookAgentProgram()

    # ----- Plan (DSPy planner). -----
    if not tracker.can_spend("autonomous_turns", 1):
        error = {"type": "BudgetExhaustedError", "message": "no turns available for planning",
                 "resource": "autonomous_turns"}
        _finalize(task, manifest, decision, tracker, success=False, error=error, log=log, plan=[])
        return AgentResult(task=task, success=False, stage_used=None,
                           manifest=task.read_manifest(), result_payload=None,
                           answer=_summarize("plan", None, None, error),
                           error=error, plan=[], turns_used=0)
    tracker.spend("autonomous_turns", 1)
    try:
        plan = program.plan(request)
    except Exception as exc:  # noqa: BLE001
        err = {"type": type(exc).__name__, "message": str(exc)}
        _finalize(task, manifest, decision, tracker, success=False, error=err, log=log, plan=[])
        return AgentResult(task=task, success=False, stage_used=None,
                           manifest=task.read_manifest(), result_payload=None,
                           answer=_summarize("plan", None, None, err), error=err, plan=[],
                           turns_used=int(tracker.snapshot()["used"]["max_autonomous_turns"]))
    # Continuation: extend the prior plan instead of replacing it.
    if prior_state.get("plan"):
        plan = list(prior_state["plan"]) + [f"Address user feedback: {request.strip()[:60]}"]
    if not plan:
        plan = [t.title for t in decompose_request(request)]
    log.append("plan_created", plan=plan)
    (task.inputs_dir / "todo.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # ----- Choose skill (DSPy skill_chooser). -----
    catalog_records = repo.catalog()
    log.append("retrieval_started", catalog_size=len(catalog_records))
    chosen_skill: Skill | None = None
    if catalog_records:
        if not tracker.can_spend("autonomous_turns", 1):
            decision.record("retrieve", attempted=False, result="turn budget exhausted before chooser")
        else:
            tracker.spend("autonomous_turns", 1)
            try:
                chosen_id = program.choose_skill(request, json.dumps(catalog_records))
            except Exception as exc:  # noqa: BLE001
                decision.record("retrieve", attempted=True, result=f"chooser failed: {exc!s}")
                chosen_id = "none"
            chosen_skill = _resolve_skill(repo, chosen_id)
            decision.record("retrieve", attempted=True,
                            result=(f"chose {chosen_skill.skill_id}" if chosen_skill else f"chose none (raw={chosen_id!r})"))
    else:
        decision.record("retrieve", attempted=False, result="empty catalog")
    log.append("retrieval_finished", chosen=(chosen_skill.skill_id if chosen_skill else None))

    decision.record("compose", attempted=False, result="not implemented (no prior runs index yet)")

    # ----- Transform or Generate. -----
    nb_path: Path | None = None
    generated_code: GeneratedCode | None = None
    error: dict[str, Any] | None = None

    if chosen_skill is not None:
        try:
            nb_path = transform_skill_to_notebook(chosen_skill, task.task_notebook)
            decision.record("transform", attempted=True, result=f"transformed skill {chosen_skill.skill_id}")
            decision.choose("transform")
        except Exception as exc:  # noqa: BLE001
            decision.record("transform", attempted=True, result=f"failed: {exc!s}")
            chosen_skill = None
            nb_path = None

    if nb_path is None:
        if not tracker.can_spend("autonomous_turns", 1):
            error = {"type": "BudgetExhaustedError",
                     "message": "max_autonomous_turns reached before generation",
                     "resource": "autonomous_turns"}
            decision.record("generate", attempted=False, result="turn budget exhausted")
            decision.choose("generate")
            _finalize(task, manifest, decision, tracker, success=False, error=error, log=log, plan=plan)
            return AgentResult(task=task, success=False, stage_used="generate",
                               manifest=task.read_manifest(), result_payload=None,
                               answer=_summarize("generate", None, None, error),
                               error=error, plan=plan,
                               turns_used=int(tracker.snapshot()["used"]["max_autonomous_turns"]))
        tracker.spend("autonomous_turns", 1)
        try:
            log.append("generation_started")
            source = program.generate_code(request, plan)
            validate_snippet(source)
            generated_code = GeneratedCode(source=source, request=request, plan=list(plan))
            log.append("generation_finished", source_chars=len(source))
            nb_path = build_generated_notebook(request, generated_code, task.task_notebook, plan=plan)
            decision.record("generate", attempted=True, result="generated executable notebook via DSPy")
            decision.choose("generate")
        except Exception as exc:  # noqa: BLE001
            err_type = type(exc).__name__
            decision.record("generate", attempted=True, result=f"failed: {err_type}: {exc!s}")
            decision.choose("generate")
            error = {"type": err_type, "message": str(exc)}
            _finalize(task, manifest, decision, tracker, success=False, error=error, log=log, plan=plan)
            return AgentResult(task=task, success=False, stage_used="generate",
                               manifest=task.read_manifest(), result_payload=None,
                               answer=_summarize("generate", None, None, error),
                               error=error, plan=plan,
                               turns_used=int(tracker.snapshot()["used"]["max_autonomous_turns"]))

    # ----- Parameter inference for the Transform path. -----
    if chosen_skill is not None:
        schema_props = ((chosen_skill.manifest.get("input_schema") or {}) if isinstance(chosen_skill.manifest, dict) else {}).get("properties") or {}
        schema_text = _schema_text(chosen_skill)
        if schema_props and schema_text:
            if not tracker.can_spend("autonomous_turns", 1):
                log.append("parameters_skipped", reason="turn budget exhausted")
            else:
                tracker.spend("autonomous_turns", 1)
                try:
                    raw = program.extract_parameters(
                        request, schema_text, current_date=date.today().isoformat(),
                    )
                    inferred = _parse_extracted_params(raw, schema_props)
                except Exception as exc:  # noqa: BLE001
                    log.append("parameters_inferred", inferred={}, error=str(exc))
                    inferred = {}
                # Caller-supplied values win.
                parameters = {**inferred, **parameters}
                log.append("parameters_inferred", inferred=inferred)

    task.stage_used = decision.chosen

    # ----- Execute. -----
    try:
        exec_result = execute_notebook(
            nb_path,
            parameters=parameters,
            output_path=task.executed_notebook,
            run_dir=task.directory,
            event_log=log,
            budget=tracker,
        )
    except BudgetExhaustedError as exc:
        err = {"type": "BudgetExhaustedError", "message": str(exc), "resource": exc.resource}
        _finalize(task, manifest, decision, tracker, success=False, error=err, log=log,
                  status="budget_exhausted", plan=plan)
        return AgentResult(task=task, success=False, stage_used=decision.chosen,
                           manifest=task.read_manifest(), result_payload=None,
                           answer=f"Budget exhausted for {exc.resource}.", error=err, plan=plan,
                           turns_used=int(tracker.snapshot()["used"]["max_autonomous_turns"]))

    # ----- Repair (deterministic + DSPy repairer). -----
    if not exec_result.success and auto_repair and tracker.can_spend("autonomous_turns", 1):
        tracker.spend("autonomous_turns", 1)
        outcome = repair_and_rerun(task, exec_result, budget=tracker, parameters=parameters, program=program)
        if outcome.repaired and outcome.repaired_result is not None:
            exec_result = outcome.repaired_result

    # ----- Finalize. -----
    success = exec_result.success
    result_payload = exec_result.result
    error = None if success else exec_result.error
    answer = _summarize(decision.chosen or "task", chosen_skill, result_payload, error)
    task.answer_md.write_text(answer, encoding="utf-8")
    _finalize(task, manifest, decision, tracker, success=success, error=error, log=log,
              result_path=task.result_json, plan=plan)

    turns_used = int(tracker.snapshot()["used"]["max_autonomous_turns"])
    _write_notebook_state(
        task.executed_notebook,
        {
            "task_id": task.task_id,
            "request": request,
            "plan": plan,
            "stage_used": decision.chosen,
            "stage_decision": decision.to_dict(),
            "result": result_payload,
            "answer": answer,
            "error": error,
            "turns_used": turns_used,
            "max_autonomous_turns": max_autonomous_turns,
            "generated": generated_code.to_dict() if generated_code is not None else None,
            "skill_id": chosen_skill.skill_id if chosen_skill is not None else None,
            "continued_from": str(parent_task.directory) if parent_task else None,
            "status": "success" if success else "failed",
            "finished_at": iso_now(),
        },
    )

    extras: dict[str, Any] = {}
    if generated_code is not None:
        extras["generated_code"] = generated_code.source

    return AgentResult(
        task=task,
        success=success,
        stage_used=decision.chosen,
        manifest=task.read_manifest(),
        result_payload=result_payload,
        answer=answer,
        execution=exec_result,
        error=error,
        extras=extras,
        plan=list(plan),
        turns_used=turns_used,
    )


def _resolve_continuation(ref: AgentResult | Task | str | Path) -> Task | None:
    if isinstance(ref, AgentResult):
        return ref.task
    if isinstance(ref, Task):
        return ref
    return Task.load(Path(ref))


def _finalize(
    task: Task,
    manifest: dict[str, Any],
    decision: StageDecision,
    tracker: BudgetTracker,
    *,
    success: bool,
    error: dict[str, Any] | None,
    log,
    result_path: Path | None = None,
    status: str | None = None,
    plan: list[str] | None = None,
) -> None:
    snap = tracker.snapshot()
    manifest.update(
        {
            "stage_used": decision.chosen,
            "stage_decision": decision.to_dict(),
            "finished_at": iso_now(),
            "budget_used": snap["used"],
            "budget_remaining": snap["remaining"],
            "outputs": (manifest.get("outputs") or {}),
            "plan": list(plan or []),
        }
    )
    if result_path is not None and result_path.exists():
        manifest["outputs"]["result_json"] = str(result_path)
    if error is not None:
        manifest["error"] = error
    final_status = status or ("success" if success else "failed")
    manifest["status"] = final_status
    task.stage_used = decision.chosen
    task.write_manifest(manifest)
    task.update_status(final_status)
    task.write_readme()
    log.append("manifest_updated", status=final_status, stage_used=decision.chosen)
    log.append("task_finished", status=final_status)


# Helper retained for tests that want to inspect what the planner uses when
# DSPy returns nothing.
__all__ = ["AgentResult", "run_task", "split_plan"]
