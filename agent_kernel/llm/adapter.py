"""Structured-LLM adapter with retry-on-validation and budget accounting.

The adapter is provider-agnostic. Concrete providers implement the
``Provider`` Protocol; the adapter handles:

- JSON-mode prompting / schema injection (delegated to the provider)
- Pydantic validation of the returned text
- Retry-on-validation-error (up to ``max_retries``), appending the
  validator's error message to the conversation
- Provenance emission (``llm.call.started`` / ``llm.call.completed``) and
  budget debiting against the active task

Cost arithmetic is integer-precise (``llm_usd_micro`` is micro-USD: $1 =
1_000_000) so the JSONL budget invariant (never negative) holds exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from agent_kernel.models.budget import Budget
from agent_kernel.models.event import EventStatus, EventType, ProvenanceEvent
from agent_kernel.runtime import policy_engine
from agent_kernel.util import new_id, now_iso

if TYPE_CHECKING:
    from agent_kernel.api import AgentKernel

T = TypeVar("T", bound=BaseModel)


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails or fails to validate after retries."""


@dataclass
class LLMUsage:
    """Reported token usage + cost for one provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd_micro: int = 0
    provider: str = ""
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    """Single-shot text-generation provider Protocol."""

    name: str

    def generate_text(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        model: str | None = None,
    ) -> tuple[str, LLMUsage]:
        """Return (raw_text, usage). Raw text must be a JSON string."""
        ...


class StructuredLLM:
    """Adapter that returns validated Pydantic objects with retries + ledger."""

    def __init__(
        self,
        provider: Provider,
        *,
        agent_kernel: AgentKernel | None = None,
        model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.agent_kernel = agent_kernel
        self.model = model
        self.max_retries = max_retries

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[T],
        task_id: str | None = None,
    ) -> T:
        """Send ``messages`` to the provider; parse + validate into ``response_model``."""
        call_id = new_id("llm")
        schema = response_model.model_json_schema()

        # Emit llm.call.started
        if self.agent_kernel and task_id:
            self._emit_started(task_id, call_id, response_model, len(messages))

        last_error: ValidationError | None = None
        last_usage = LLMUsage(provider=self.provider.name, model=self.model or "")
        attempts = 0
        attempt_messages = list(messages)
        while attempts <= self.max_retries:
            attempts += 1
            text, usage = self.provider.generate_text(
                messages=attempt_messages,
                response_schema=schema,
                model=self.model,
            )
            last_usage = usage
            try:
                parsed = response_model.model_validate_json(text)
                # Success: debit budget + emit completion
                if self.agent_kernel and task_id:
                    self._emit_completed(task_id, call_id, usage, attempts, status=EventStatus.ok)
                    self._debit_budget(task_id, usage)
                return parsed
            except ValidationError as exc:
                last_error = exc
                # Inject the validator's error so the provider can self-correct.
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "The previous response did not match the required JSON schema. "
                            f"Errors:\n{exc}\n"
                            "Please respond again with valid JSON matching the schema."
                        ),
                    },
                ]
                continue

        # Out of retries — emit failed completion + debit the cost incurred.
        if self.agent_kernel and task_id:
            self._emit_completed(
                task_id,
                call_id,
                last_usage,
                attempts,
                status=EventStatus.error,
                error=str(last_error),
            )
            self._debit_budget(task_id, last_usage)
        raise LLMCallError(
            f"LLM response failed validation after {attempts} attempts: {last_error}"
        )

    # ------------------------------------------------------------- emission

    def _emit_started(
        self,
        task_id: str,
        call_id: str,
        response_model: type[BaseModel],
        message_count: int,
    ) -> None:
        assert self.agent_kernel is not None
        task = self.agent_kernel.get_task(task_id)
        before = policy_engine.remaining_reservation(task) if task is not None else None
        self.agent_kernel.scheduler._emit(
            EventType.llm_call_started,
            task_id=task_id,
            payload={
                "call_id": call_id,
                "provider": self.provider.name,
                "model": self.model,
                "response_model": response_model.__name__,
                "message_count": message_count,
            },
            budget_before=before,
        )

    def _emit_completed(
        self,
        task_id: str,
        call_id: str,
        usage: LLMUsage,
        attempts: int,
        *,
        status: EventStatus,
        error: str | None = None,
    ) -> None:
        assert self.agent_kernel is not None
        task = self.agent_kernel.get_task(task_id)
        before = policy_engine.remaining_reservation(task) if task else None
        # We compute budget_after as the projected post-debit availability.
        after = None
        if task is not None:
            try:
                trans = policy_engine.debit(task, Budget(llm_usd_micro=usage.cost_usd_micro))
                after_task = task.model_copy(update={"spent_budget": trans.spent_budget})
                after = policy_engine.remaining_reservation(after_task)
            except ValueError:
                after = before  # debit would exceed; reported in error path
        kwargs: dict[str, Any] = {
            "task_id": task_id,
            "payload": {
                "call_id": call_id,
                "provider": self.provider.name,
                "model": self.model,
                "attempts": attempts,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd_micro": usage.cost_usd_micro,
                "error": error,
            },
            "budget_before": before,
            "budget_after": after,
            "status": status,
        }
        self.agent_kernel.scheduler._emit(EventType.llm_call_completed, **kwargs)

    def _debit_budget(self, task_id: str, usage: LLMUsage) -> None:
        assert self.agent_kernel is not None
        if usage.cost_usd_micro <= 0:
            return
        task = self.agent_kernel.get_task(task_id)
        if task is None:
            return
        delta = Budget(llm_usd_micro=usage.cost_usd_micro)
        try:
            trans = policy_engine.debit(task, delta)
        except ValueError:
            # Budget exhausted; emit a budget.debited event with the cap and
            # let the caller (scheduler / must_stop) react.
            cap = policy_engine.remaining_reservation(task)
            trans = policy_engine.debit(task, cap)
            delta = cap

        before_remaining = policy_engine.remaining_reservation(task)
        updated = task.model_copy(
            update={
                "reserved_budget": trans.reserved_budget,
                "spent_budget": trans.spent_budget,
                "updated_at": now_iso(),
            }
        )
        self.agent_kernel.scheduler._persist(updated)
        after_remaining = policy_engine.remaining_reservation(updated)
        self.agent_kernel.scheduler._emit(
            EventType.budget_debited,
            task_id=task_id,
            budget_before=before_remaining,
            budget_after=after_remaining,
            payload={"delta": delta.model_dump(mode="json"), "reason": "llm_call"},
        )


# A small no-op helper for type checkers / unused-import suppression in the
# adapter module without touching the public surface.
_ = ProvenanceEvent  # type: ignore[unused-ignore]
