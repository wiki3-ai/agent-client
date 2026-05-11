"""Pure policy engine: typed decisions over budget/quota/lineage state.

All functions in this module are deterministic, side-effect-free, and take
typed inputs. They are the "decision logic" half of agent-kernel and
deliberately know nothing about asyncio, filesystems, providers, or
notebooks. Adapters call into here to ask "can this happen?" and apply the
returned ``Decision`` to durable state.

This boundary is the natural seam where, post-MVP, a fixture-driven
differential test suite (ACL2 or hand-built golden vectors) can be bolted
on without refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_kernel.models.budget import Budget, QuotaSnapshot
from agent_kernel.models.policy import PolicyProfile
from agent_kernel.models.task import TaskSpec


@dataclass(frozen=True)
class Decision:
    """Outcome of a policy check."""

    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class BudgetTransition:
    """Result of applying ``debit`` or ``refund`` to a task's budget."""

    reserved_budget: Budget
    spent_budget: Budget


# ----------------------------------------------------------------- admission


def can_admit(task: TaskSpec, profile: PolicyProfile, quota: QuotaSnapshot) -> Decision:
    """Return whether the scheduler may admit ``task`` to running."""
    if task.status.value not in {"queued", "draft"}:
        return Decision(False, f"status_not_admissible:{task.status.value}")
    if quota.kernel_slots_used >= profile.kernel_slots_total:
        return Decision(False, "quota_no_kernel_slots")
    if quota.queued_tasks > profile.queue_limit:
        return Decision(False, "quota_queue_full")
    return Decision(True, "ok")


# --------------------------------------------------------------------- spawn


def can_spawn(parent: TaskSpec, profile: PolicyProfile, request: Budget) -> Decision:
    """Return whether ``parent`` may spawn a child with reservation ``request``."""
    if parent.depth >= profile.max_spawn_depth:
        return Decision(False, "max_spawn_depth")
    if len(parent.child_task_ids) >= profile.max_children_per_task:
        return Decision(False, "max_children_per_task")

    available = _available_budget(parent)
    # Reserve floor: parent must keep at least floor_ratio of its
    # reserved budget for itself (computed per-field on llm_usd_micro and
    # wall_ms which are the most contended fields).
    floor_ratio = profile.reservation_policy.parent_reserve_floor_ratio
    for field in ("wall_ms", "llm_usd_micro"):
        original = getattr(parent.reserved_budget, field)
        requested = getattr(request, field)
        remaining = getattr(available, field) - requested
        if remaining < int(original * floor_ratio):
            return Decision(False, f"reserve_floor:{field}")

    if not available.can_cover(request):
        return Decision(False, "insufficient_budget")
    return Decision(True, "ok")


# ----------------------------------------------------------------- stop check


def must_stop(task: TaskSpec) -> Decision:
    """Return whether the task must terminate (budget exhausted)."""
    available = _available_budget(task)
    for field in available.__class__.model_fields:
        if getattr(available, field) <= 0 and getattr(task.reserved_budget, field) > 0:
            return Decision(True, f"budget_exhausted:{field}")
    return Decision(False, "ok")


# ------------------------------------------------------------ budget arithmetic


def debit(task: TaskSpec, delta: Budget) -> BudgetTransition:
    """Apply a debit to a task's spent budget.

    Spent budget grows monotonically; the reservation itself is unchanged.
    Raises ``ValueError`` if the debit would exceed the reservation, so the
    caller can mark the task must-stop and refund.
    """
    new_spent = task.spent_budget.add(delta)
    # Verify spend never exceeds reservation (non-negative-availability invariant).
    if not task.reserved_budget.can_cover(new_spent):
        raise ValueError(
            f"debit exceeds reservation: spent_after={new_spent} > reserved={task.reserved_budget}"
        )
    return BudgetTransition(reserved_budget=task.reserved_budget, spent_budget=new_spent)


def refund(task: TaskSpec, delta: Budget) -> BudgetTransition:
    """Refund unused reservation from a task back to its parent.

    Returns the new reservation (reduced by ``delta``). ``delta`` must not
    exceed the currently-available (reserved - spent) budget.
    """
    available = _available_budget(task)
    if not available.can_cover(delta):
        raise ValueError(f"refund exceeds available: available={available} < delta={delta}")
    new_reserved = task.reserved_budget.subtract(delta)
    return BudgetTransition(reserved_budget=new_reserved, spent_budget=task.spent_budget)


def remaining_reservation(task: TaskSpec) -> Budget:
    """Return ``reserved_budget - spent_budget`` (the refundable amount on cancel/fail)."""
    return _available_budget(task)


# --------------------------------------------------------------------- helpers


def _available_budget(task: TaskSpec) -> Budget:
    """Return ``reserved - spent``; never negative because debit() guards it."""
    try:
        return task.reserved_budget.subtract(task.spent_budget)
    except ValueError:
        # Defensive: if state were ever inconsistent, treat as zero.
        return Budget()
