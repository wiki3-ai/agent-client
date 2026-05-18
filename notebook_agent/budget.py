"""Budget definition and tracker (Section 5 of the spec)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# All resource fields known to the budget. Numbers default to None which means
# "unlimited" so users can pass partial budgets.
_RESOURCE_FIELDS: tuple[str, ...] = (
    "max_wall_time_seconds",
    "max_lm_calls",
    "max_lm_tokens_input",
    "max_lm_tokens_output",
    "max_cost_usd",
    "max_notebook_executions",
    "max_repair_attempts",
    "max_subtasks",
    "max_parallel_subtasks",
    "max_web_searches",
    "max_generated_code_tokens",
    "max_autonomous_turns",
)

# Mapping of resource short-name -> budget field name.
_RESOURCE_MAP: dict[str, str] = {
    "wall_time_seconds": "max_wall_time_seconds",
    "lm_calls": "max_lm_calls",
    "lm_tokens_input": "max_lm_tokens_input",
    "lm_tokens_output": "max_lm_tokens_output",
    "cost_usd": "max_cost_usd",
    "notebook_executions": "max_notebook_executions",
    "repair_attempts": "max_repair_attempts",
    "subtasks": "max_subtasks",
    "parallel_subtasks": "max_parallel_subtasks",
    "web_searches": "max_web_searches",
    "generated_code_tokens": "max_generated_code_tokens",
    "autonomous_turns": "max_autonomous_turns",
}


class BudgetExhaustedError(RuntimeError):
    """Raised when an operation cannot proceed because of budget exhaustion."""

    def __init__(self, resource: str, requested: float, remaining: float | None) -> None:
        super().__init__(
            f"Budget exhausted for {resource}: requested {requested}, remaining {remaining}"
        )
        self.resource = resource
        self.requested = requested
        self.remaining = remaining


@dataclass
class Budget:
    """User-supplied budget limits. None means unlimited."""

    max_wall_time_seconds: float | None = None
    max_lm_calls: int | None = None
    max_lm_tokens_input: int | None = None
    max_lm_tokens_output: int | None = None
    max_cost_usd: float | None = None
    max_notebook_executions: int | None = None
    max_repair_attempts: int | None = None
    max_subtasks: int | None = None
    max_parallel_subtasks: int | None = None
    max_web_searches: int | None = None
    max_generated_code_tokens: int | None = None
    max_autonomous_turns: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Budget:
        if not data:
            return cls()
        # Accept both "max_foo" and "foo" forms.
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in _RESOURCE_FIELDS:
                kwargs[key] = value
            elif key in _RESOURCE_MAP:
                kwargs[_RESOURCE_MAP[key]] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in _RESOURCE_FIELDS}


@dataclass
class BudgetTracker:
    """Tracks spend against a Budget."""

    budget: Budget = field(default_factory=Budget)
    started_at: float = field(default_factory=time.monotonic)
    used: dict[str, float] = field(default_factory=dict)

    # ----- internal helpers -----

    def _field(self, resource: str) -> str:
        if resource in _RESOURCE_FIELDS:
            return resource
        if resource in _RESOURCE_MAP:
            return _RESOURCE_MAP[resource]
        raise KeyError(f"Unknown budget resource: {resource}")

    def _limit(self, resource: str) -> float | None:
        return getattr(self.budget, self._field(resource))

    def _used(self, resource: str) -> float:
        return float(self.used.get(self._field(resource), 0))

    def _wall_time_used(self) -> float:
        return time.monotonic() - self.started_at

    # ----- public API -----

    def remaining(self, resource: str) -> float | None:
        field_name = self._field(resource)
        limit = getattr(self.budget, field_name)
        if limit is None:
            return None
        if field_name == "max_wall_time_seconds":
            return max(0.0, limit - self._wall_time_used())
        return max(0.0, limit - self._used(resource))

    def can_spend(self, resource: str, amount: float = 1) -> bool:
        rem = self.remaining(resource)
        if rem is None:
            return True
        return amount <= rem

    def spend(self, resource: str, amount: float = 1) -> None:
        if not self.can_spend(resource, amount):
            raise BudgetExhaustedError(resource, amount, self.remaining(resource))
        field_name = self._field(resource)
        if field_name == "max_wall_time_seconds":
            # Wall-time spends naturally; we just touch the dict for snapshotting.
            self.used[field_name] = self._wall_time_used()
            return
        self.used[field_name] = self._used(resource) + amount

    def check(self, resource: str, amount: float = 1) -> None:
        """Raise BudgetExhaustedError if the spend would exceed the limit."""
        if not self.can_spend(resource, amount):
            raise BudgetExhaustedError(resource, amount, self.remaining(resource))

    def snapshot(self) -> dict[str, Any]:
        used: dict[str, Any] = {}
        remaining: dict[str, Any] = {}
        for f in _RESOURCE_FIELDS:
            limit = getattr(self.budget, f)
            if f == "max_wall_time_seconds":
                used_val: float = self._wall_time_used()
            else:
                used_val = float(self.used.get(f, 0))
            used[f] = used_val
            remaining[f] = None if limit is None else max(0.0, limit - used_val)
        return {
            "initial": self.budget.to_dict(),
            "used": used,
            "remaining": remaining,
        }
