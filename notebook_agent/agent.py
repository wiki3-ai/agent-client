"""End-to-end agent orchestration (Section 14.9, Milestone 8).

Implements the Retrieve → Compose → Transform → Generate policy as a single
``run_task`` entry point. The first implementation supports:

* Retrieve: search the local + built-in skill repositories.
* Compose: if a prior successful run for the same skill+input exists, reuse it
  (best-effort; not strictly required by the milestone but cheap to detect).
* Transform: convert the matched ``SKILL.md`` into a parameterized notebook
  via :mod:`notebook_agent.transform`.
* Generate: if no skill matches, attempt LLM generation (only when an LLM
  client is configured); otherwise mark the task ``failed`` with a clear
  message.

The function always writes ``task.json``, ``manifest.json``, ``README.md`` and
``logs/events.jsonl``, and (on success) ``outputs/result.json`` and
``outputs/answer.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._clock import iso_now
from .budget import Budget, BudgetExhaustedError, BudgetTracker
from .litellm_client import LiteLLMClient
from .notebook_exec import NotebookExecutionResult, execute_notebook
from .planner import StageDecision
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
        }


def _skill_repository(
    extra_skill_dirs: list[Path | str] | None,
) -> SkillRepository:
    roots: list[Path | str] = [builtin_skills_root()]
    if extra_skill_dirs:
        roots.extend(extra_skill_dirs)
    # Also pick up a project-local skills/ directory if present.
    cwd_skills = Path.cwd() / "skills"
    if cwd_skills.exists() and cwd_skills not in [Path(r) for r in roots]:
        roots.append(cwd_skills)
    return SkillRepository(roots)


def _summarize(skill: Skill | None, result_payload: dict[str, Any] | None, error: dict[str, Any] | None) -> str:
    if error is not None:
        return f"Task failed: {error.get('type', 'Error')}: {error.get('message', '')}".strip()
    title = skill.name if skill else "Task"
    if result_payload is None:
        return f"{title} completed; no structured result was written."
    if "message" in result_payload and isinstance(result_payload["message"], str):
        return result_payload["message"]
    pretty = json.dumps(result_payload, indent=2, sort_keys=True)
    return f"{title} result:\n\n```json\n{pretty}\n```"


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
) -> AgentResult:
    """Run a single task end-to-end via Retrieve → Compose → Transform → Generate."""
    parameters = dict(parameters or {})
    task = create_root_task(
        runs_root,
        title=title or (request.strip().splitlines()[0] if request.strip() else "task"),
        request=request,
        parameters=parameters,
        budget=budget,
    )
    log = task.event_log()
    log.append("task_started", task_id=task.task_id)
    task.update_status("running")
    tracker = BudgetTracker(task.budget)
    decision = StageDecision()
    repo = _skill_repository(skill_dirs)
    started_at = iso_now()

    manifest = task.read_manifest()
    manifest["started_at"] = started_at

    # ---------------- Retrieve ----------------
    log.append("retrieval_started", query=request)
    search_results = repo.search(request)
    decision.record(
        "retrieve",
        attempted=True,
        result=(
            f"found {len(search_results)} candidates; top: "
            + (search_results[0].skill.skill_id if search_results else "(none)")
        ),
    )
    for r in search_results[:3]:
        log.append("artifact_retrieved", skill_id=r.skill.skill_id, score=r.score)
    log.append("retrieval_finished", count=len(search_results))

    chosen_skill: Skill | None = search_results[0].skill if search_results else None

    # ---------------- Compose ----------------
    # First implementation: composition not yet implemented at this layer.
    decision.record("compose", attempted=False, result="not implemented (no prior runs index yet)")

    # ---------------- Transform / Generate ----------------
    nb_path: Path | None = None
    error: dict[str, Any] | None = None
    if chosen_skill is not None:
        try:
            nb_path = transform_skill_to_notebook(chosen_skill, task.task_notebook)
            decision.record("transform", attempted=True, result=f"transformed skill {chosen_skill.skill_id}")
            decision.choose("transform")
        except Exception as exc:  # noqa: BLE001
            decision.record("transform", attempted=True, result=f"failed: {exc!s}")
            chosen_skill = None
    if nb_path is None:
        # Fall back to Generate. Without an LLM we cannot generate code; record
        # the failure cleanly per spec §19.
        decision.record(
            "generate",
            attempted=llm is not None,
            result=(
                "LLM-based generation not yet implemented for arbitrary requests"
                if llm is not None
                else "no LLM configured; cannot generate code"
            ),
        )
        decision.choose("generate")
        error = {
            "type": "NoSkillFound",
            "message": f"No matching skill for request: {request!r}",
        }
        _finalize(task, manifest, decision, tracker, success=False, error=error, log=log)
        return AgentResult(
            task=task,
            success=False,
            stage_used="generate",
            manifest=task.read_manifest(),
            result_payload=None,
            answer=_summarize(None, None, error),
            error=error,
        )

    # ---------------- Execute ----------------
    # Infer any missing required parameters from the request when an LLM is
    # available.  Caller-supplied parameters always take precedence.
    extractor_error: str | None = None
    if llm is not None and chosen_skill is not None:
        from .dspy_modules import ParameterExtractor
        # Route LM call logs into the task's logs directory if not already set.
        if llm.lm_calls_log is None:
            llm.lm_calls_log = task.directory / "logs" / "lm_calls.jsonl"
        extractor = ParameterExtractor(llm=llm)
        inferred = extractor(request, chosen_skill)
        extractor_error = getattr(extractor, "last_error", None)
        # Merge: inferred fills gaps, caller-supplied wins conflicts.
        parameters = {**inferred, **parameters}
        log.append(
            "parameters_inferred",
            inferred=inferred,
            error=extractor_error,
        )

    task.stage_used = decision.chosen
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
            task,
            manifest,
            decision,
            tracker,
            success=False,
            error={"type": "BudgetExhaustedError", "message": str(exc), "resource": exc.resource},
            log=log,
            status="budget_exhausted",
        )
        return AgentResult(
            task=task,
            success=False,
            stage_used=decision.chosen,
            manifest=task.read_manifest(),
            result_payload=None,
            answer=f"Budget exhausted for {exc.resource}.",
            error={"type": "BudgetExhaustedError", "message": str(exc), "resource": exc.resource},
        )

    # ---------------- Repair ----------------
    if not exec_result.success and auto_repair:
        outcome = repair_and_rerun(task, exec_result, budget=tracker, parameters=parameters, llm=llm)
        if outcome.repaired and outcome.repaired_result is not None:
            exec_result = outcome.repaired_result

    # ---------------- Finalize ----------------
    success = exec_result.success
    result_payload = exec_result.result
    error = None if success else exec_result.error
    answer = _summarize(chosen_skill, result_payload, error)
    task.answer_md.write_text(answer, encoding="utf-8")
    _finalize(task, manifest, decision, tracker, success=success, error=error, log=log, result_path=task.result_json)

    return AgentResult(
        task=task,
        success=success,
        stage_used=decision.chosen,
        manifest=task.read_manifest(),
        result_payload=result_payload,
        answer=answer,
        execution=exec_result,
        error=error,
        extras={"parameter_extractor_error": extractor_error} if extractor_error else {},
    )


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
    # Refresh README to include final outputs / child links / status.
    task.write_readme()
    log.append(
        "manifest_updated",
        status=final_status,
        stage_used=decision.chosen,
    )
    log.append("task_finished", status=final_status)
