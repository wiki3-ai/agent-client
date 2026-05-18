"""End-to-end agent orchestration (Section 14.9, MVP per ``nb-agent.md``).

This is the **only** entry point a notebook user is expected to touch::

    from notebook_agent import run_task
    result = run_task("Count the words in 'hello from the graph notebook agent'")

``run_task`` builds an LLM client from environment defaults (LM Studio
compatible via LiteLLM), drafts a short visible plan/TODO list, searches the
local skill repository, and — if no skill matches strongly — asks the LLM to
write a Python snippet and executes it as a generated notebook with Papermill.
The executed notebook is the canonical task record: machine-readable state is
stored in ``nb.metadata["notebook_agent"]`` and visible cells carry the human
narrative.

The function also supports continuation: pass ``continue_from=<prior result>``
together with a feedback prompt ("continue", "that result is wrong, try again",
new instructions) and the agent resumes from the prior notebook state instead
of starting over.

Backwards compatibility
-----------------------
The MVP keeps writing ``task.json``, ``manifest.json``, ``README.md``,
``outputs/result.json``, ``outputs/answer.md`` and ``logs/events.jsonl`` to the
task directory so the existing milestone-1..9 tests, MCP tools, and display
helpers keep working. The notebook is now the authoritative record — sidecar
files are a derivative cache populated from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nbformat

from ._clock import iso_now
from .budget import Budget, BudgetExhaustedError, BudgetTracker
from .codegen import (
    GeneratedCode,
    build_generated_notebook,
    generate_code_for_request,
)
from .litellm_client import LiteLLMClient, LLMUnavailableError
from .notebook_exec import NotebookExecutionResult, execute_notebook
from .planner import StageDecision, TodoItem, decompose_request
from .repair import repair_and_rerun
from .skills import Skill, SkillRepository
from .task_graph import Task, create_root_task
from .transform import builtin_skills_root, transform_skill_to_notebook

# Skill-match score floor. Below this we fall through to the Generate stage
# rather than forcing the prompt onto a weakly-matching skill (which is what
# made the agent appear to do nothing but echo).
_SKILL_MATCH_THRESHOLD = 3.0


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


def _default_llm(task: Task | None = None) -> LiteLLMClient:
    """Build a default LiteLLM client from env vars (LM Studio compatible).

    The client is created lazily; no network call is made here. If the user
    has not set ``NOTEBOOK_AGENT_PROVIDER`` we default to ``"lm_studio"`` —
    which uses ``NOTEBOOK_AGENT_BASE_URL`` / ``DEFAULT_BASE_URL``.
    """
    log = task.lm_calls_log if task is not None else None
    return LiteLLMClient(lm_calls_log=log)


def _looks_like_continuation(prompt: str) -> bool:
    s = (prompt or "").strip().lower()
    triggers = (
        "continue",
        "keep going",
        "try again",
        "not satisfactory",
        "that's wrong",
        "thats wrong",
        "wrong",
        "fix it",
        "again",
        "more",
    )
    return any(t in s for t in triggers)


def _todo_to_plan(items: list[TodoItem]) -> list[str]:
    return [it.title for it in items]


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
    """Merge *state* into the executed notebook's metadata under ``notebook_agent``.

    This is the canonical persistence channel per ``nb-agent.md``.
    """
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
    md = nb.metadata.get("notebook_agent") or {}
    return dict(md)


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
    auto_repair: bool = True,
    continue_from: AgentResult | Task | str | Path | None = None,
    max_autonomous_turns: int = 6,
) -> AgentResult:
    """Run a task end-to-end. The only required argument is ``request``.

    Parameters
    ----------
    request:
        The user's prompt. Everything else is optional.
    runs_root:
        Where to create the per-run directory.
    parameters:
        Optional skill input overrides. The agent infers parameters from the
        prompt when an LLM is available; explicit values take precedence.
    budget:
        Optional :class:`Budget`. Only ``max_autonomous_turns`` is enforced
        as a hard loop limit per the MVP spec; the rest are tracked for
        observability and can be set by power users.
    title:
        Optional human title (defaults to the first line of *request*).
    skill_dirs:
        Extra skill directories to scan in addition to the built-ins and
        ``./skills``.
    llm:
        Optional pre-built :class:`LiteLLMClient`. If omitted, a default
        client is built from environment defaults (LM Studio compatible).
        The notebook user never needs to supply this.
    auto_repair:
        If True, failed notebook executions are passed through the
        deterministic repair loop.
    continue_from:
        An :class:`AgentResult`, :class:`Task`, task directory path, or
        directory string. When present, the new request is treated as a
        continuation of that prior task (feedback, "continue", corrections).
    max_autonomous_turns:
        Maximum number of autonomous LLM-driven turns the agent may take
        before reporting back. Per ``nb-agent.md`` this is the only
        user-facing budget knob. Defaults to 6.
    """
    parameters = dict(parameters or {})

    # ---------- Budget: turn-based first, the rest passthrough ----------
    if isinstance(budget, dict):
        bdict = dict(budget)
    elif budget is None:
        bdict = {}
    else:
        bdict = budget.to_dict()
    bdict.setdefault("max_autonomous_turns", max_autonomous_turns)
    budget_obj = Budget.from_dict(bdict)

    # ---------- Continuation ----------
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
    started_at = iso_now()

    manifest = task.read_manifest()
    manifest["started_at"] = started_at
    manifest["continued_from"] = str(parent_task.directory) if parent_task else None

    # Auto-build an LLM client when caller didn't supply one. This is the key
    # zero-config UX change: a notebook user just calls run_task(prompt).
    if llm is None:
        llm = _default_llm(task)
    elif llm.lm_calls_log is None:
        llm.lm_calls_log = task.lm_calls_log

    # ---------- Plan / TODO ----------
    todo_items = decompose_request(request)
    plan = _todo_to_plan(todo_items)
    if prior_state.get("plan"):
        # Continuation: keep the original plan and append a follow-up step.
        plan = list(prior_state["plan"]) + [f"Address user feedback: {request.strip()[:60]}"]
    log.append("plan_created", plan=plan)
    (task.inputs_dir / "todo.json").write_text(
        json.dumps([it.to_dict() for it in todo_items], indent=2), encoding="utf-8"
    )

    # ---------- Retrieve ----------
    log.append("retrieval_started", query=request)
    search_results = repo.search(request)
    top = search_results[0] if search_results else None
    decision.record(
        "retrieve",
        attempted=True,
        result=(
            f"found {len(search_results)} candidates; top: "
            + (f"{top.skill.skill_id} (score={top.score:.2f})" if top else "(none)")
        ),
    )
    for r in search_results[:3]:
        log.append("artifact_retrieved", skill_id=r.skill.skill_id, score=r.score)
    log.append("retrieval_finished", count=len(search_results))

    chosen_skill: Skill | None = None
    if top is not None and top.score >= _SKILL_MATCH_THRESHOLD:
        chosen_skill = top.skill

    # ---------- Compose ----------
    decision.record("compose", attempted=False, result="not implemented (no prior runs index yet)")

    # ---------- Transform or Generate ----------
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
        # Real Generate path: ask the LLM to write a snippet.
        if not tracker.can_spend("autonomous_turns", 1):
            error = {
                "type": "BudgetExhaustedError",
                "message": "max_autonomous_turns reached before generation",
                "resource": "autonomous_turns",
            }
            decision.record("generate", attempted=False, result="turn budget exhausted")
            decision.choose("generate")
            _finalize(task, manifest, decision, tracker, success=False, error=error, log=log, plan=plan)
            return AgentResult(
                task=task, success=False, stage_used="generate",
                manifest=task.read_manifest(), result_payload=None,
                answer=_summarize("generate", None, None, error),
                error=error, plan=plan, turns_used=int(tracker.snapshot()["used"]["max_autonomous_turns"]),
            )
        try:
            tracker.spend("autonomous_turns", 1)
            log.append("generation_started")
            generated_code = generate_code_for_request(request, llm=llm)
            log.append("generation_finished", source_chars=len(generated_code.source))
            nb_path = build_generated_notebook(request, generated_code, task.task_notebook, plan=plan)
            decision.record("generate", attempted=True, result="generated executable notebook from LLM")
            decision.choose("generate")
        except (LLMUnavailableError, ValueError, Exception) as exc:  # noqa: BLE001
            err_type = type(exc).__name__
            decision.record("generate", attempted=True, result=f"failed: {err_type}: {exc!s}")
            decision.choose("generate")
            error = {"type": err_type, "message": str(exc)}
            _finalize(task, manifest, decision, tracker, success=False, error=error, log=log, plan=plan)
            return AgentResult(
                task=task, success=False, stage_used="generate",
                manifest=task.read_manifest(), result_payload=None,
                answer=_summarize("generate", None, None, error),
                error=error, plan=plan, turns_used=int(tracker.snapshot()["used"]["max_autonomous_turns"]),
            )

    # ---------- Parameter inference (transform path only) ----------
    extractor_error: str | None = None
    if chosen_skill is not None and llm is not None:
        from .dspy_modules import ParameterExtractor

        if not tracker.can_spend("autonomous_turns", 1):
            log.append("parameters_skipped", reason="turn budget exhausted")
        else:
            try:
                tracker.spend("autonomous_turns", 1)
            except BudgetExhaustedError:
                pass
            else:
                extractor = ParameterExtractor(llm=llm)
                inferred = extractor(request, chosen_skill)
                extractor_error = getattr(extractor, "last_error", None)
                parameters = {**inferred, **parameters}
                log.append("parameters_inferred", inferred=inferred, error=extractor_error)

    task.stage_used = decision.chosen

    # ---------- Execute ----------
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
        _finalize(
            task, manifest, decision, tracker, success=False,
            error={"type": "BudgetExhaustedError", "message": str(exc), "resource": exc.resource},
            log=log, status="budget_exhausted", plan=plan,
        )
        return AgentResult(
            task=task, success=False, stage_used=decision.chosen,
            manifest=task.read_manifest(), result_payload=None,
            answer=f"Budget exhausted for {exc.resource}.",
            error={"type": "BudgetExhaustedError", "message": str(exc), "resource": exc.resource},
            plan=plan, turns_used=int(tracker.snapshot()["used"]["max_autonomous_turns"]),
        )

    # ---------- Repair ----------
    if not exec_result.success and auto_repair:
        if tracker.can_spend("autonomous_turns", 1):
            try:
                tracker.spend("autonomous_turns", 1)
            except BudgetExhaustedError:
                pass
            else:
                outcome = repair_and_rerun(task, exec_result, budget=tracker, parameters=parameters, llm=llm)
                if outcome.repaired and outcome.repaired_result is not None:
                    exec_result = outcome.repaired_result

    # ---------- Finalize ----------
    success = exec_result.success
    result_payload = exec_result.result
    error = None if success else exec_result.error
    answer = _summarize(decision.chosen or "task", chosen_skill, result_payload, error)
    task.answer_md.write_text(answer, encoding="utf-8")
    _finalize(task, manifest, decision, tracker, success=success, error=error, log=log,
              result_path=task.result_json, plan=plan)

    # ---------- Notebook-native canonical state ----------
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
    if extractor_error:
        extras["parameter_extractor_error"] = extractor_error
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
        plan=plan,
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
