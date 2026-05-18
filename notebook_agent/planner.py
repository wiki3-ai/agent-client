"""Simple sequential planner and stage decision logic (Sections 4 & 6, 14.6)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .budget import BudgetTracker

STAGES: tuple[str, ...] = ("retrieve", "compose", "transform", "generate")
STAGE_ORDER: dict[str, int] = {s: i for i, s in enumerate(STAGES)}


@dataclass
class StageDecision:
    """Records which stage was chosen and what was tried."""

    chosen: str | None = None
    attempts: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "retrieve": {"attempted": False, "result": None},
            "compose": {"attempted": False, "result": None},
            "transform": {"attempted": False, "result": None},
            "generate": {"attempted": False, "result": None},
        }
    )

    def record(self, stage: str, *, attempted: bool, result: Any = None) -> None:
        if stage not in self.attempts:
            raise KeyError(stage)
        self.attempts[stage] = {"attempted": attempted, "result": result}

    def choose(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"invalid stage: {stage!r}")
        self.chosen = stage

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen, **self.attempts}


@dataclass
class CandidateAction:
    """Candidate action evaluated by the planner."""

    action_id: str
    stage: str
    expected_value: float = 0.5
    expected_cost: dict[str, float] = field(default_factory=dict)
    risk: float = 0.1
    confidence: float = 0.8
    dependencies: list[str] = field(default_factory=list)

    def utility(self, epsilon: float = 0.01) -> float:
        cost_total = sum(self.expected_cost.values()) or 0.0
        return self.expected_value / max(cost_total, epsilon)


def select_action(
    candidates: list[CandidateAction],
    *,
    budget: BudgetTracker | None = None,
) -> CandidateAction | None:
    """Pick the best candidate biased by stage preference + utility.

    Filters out candidates whose costs cannot be afforded under ``budget``.
    """
    feasible: list[CandidateAction] = []
    for c in candidates:
        if budget is not None and not _affordable(c, budget):
            continue
        feasible.append(c)
    if not feasible:
        return None
    # Sort: stage rank first (lower better), then utility desc, then action_id.
    feasible.sort(key=lambda c: (STAGE_ORDER.get(c.stage, 999), -c.utility(), c.action_id))
    return feasible[0]


def _affordable(action: CandidateAction, budget: BudgetTracker) -> bool:
    for resource, amount in action.expected_cost.items():
        # Convert generic cost keys ("time_seconds" -> "wall_time_seconds").
        key = {"time_seconds": "wall_time_seconds"}.get(resource, resource)
        try:
            if not budget.can_spend(key, amount):
                return False
        except KeyError:
            # Unknown resource: assume affordable.
            continue
    return True


# ---------------------------------------------------------------------------
# TODO decomposition
# ---------------------------------------------------------------------------


@dataclass
class TodoItem:
    title: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "description": self.description}


_AND_SPLIT = re.compile(r"\s+(?:and(?:\s+then)?|then|,)\s+", re.IGNORECASE)
_VERB_HINTS = (
    "search",
    "find",
    "retrieve",
    "fetch",
    "load",
    "compose",
    "transform",
    "generate",
    "execute",
    "run",
    "build",
    "create",
    "write",
    "summarize",
    "answer",
    "echo",
    "repair",
    "test",
)


def decompose_request(request: str) -> list[TodoItem]:
    """Decompose a request into a simple sequential TODO list.

    The decomposition is intentionally conservative: it splits on common
    coordinating words ("and", "then", commas) and emits one TODO per clause
    that contains a verb hint. If no clauses are found, a single TODO mirroring
    the original request is returned.
    """
    text = (request or "").strip()
    if not text:
        return []
    clauses = [c.strip() for c in _AND_SPLIT.split(text) if c.strip()]
    items: list[TodoItem] = []
    for clause in clauses:
        first = clause.split()[0].lower() if clause.split() else ""
        if any(v in clause.lower() for v in _VERB_HINTS) or first in _VERB_HINTS:
            title = clause if len(clause) < 60 else clause[:57] + "..."
            items.append(TodoItem(title=title, description=clause))
    if not items:
        items.append(TodoItem(title=text if len(text) < 60 else text[:57] + "...", description=text))
    return items
