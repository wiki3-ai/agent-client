"""Unit tests for the budget tracker."""

from __future__ import annotations

import pytest

from notebook_agent import Budget, BudgetExhaustedError, BudgetTracker


def test_unlimited_when_no_budget() -> None:
    tracker = BudgetTracker(Budget())
    assert tracker.remaining("notebook_executions") is None
    assert tracker.can_spend("notebook_executions", 1_000_000)
    tracker.spend("notebook_executions", 5)


def test_spend_decrements_remaining() -> None:
    tracker = BudgetTracker(Budget.from_dict({"max_notebook_executions": 3}))
    assert tracker.remaining("notebook_executions") == 3
    tracker.spend("notebook_executions")
    assert tracker.remaining("notebook_executions") == 2
    tracker.spend("notebook_executions", 2)
    assert tracker.remaining("notebook_executions") == 0
    assert not tracker.can_spend("notebook_executions")


def test_spend_raises_when_exhausted() -> None:
    tracker = BudgetTracker(Budget(max_notebook_executions=1))
    tracker.spend("notebook_executions")
    with pytest.raises(BudgetExhaustedError) as ei:
        tracker.spend("notebook_executions")
    assert ei.value.resource == "notebook_executions"


def test_check_raises_without_spending() -> None:
    tracker = BudgetTracker(Budget(max_lm_calls=2))
    tracker.check("lm_calls", 2)
    assert tracker.remaining("lm_calls") == 2
    with pytest.raises(BudgetExhaustedError):
        tracker.check("lm_calls", 3)


def test_snapshot_includes_initial_used_remaining() -> None:
    tracker = BudgetTracker(Budget(max_notebook_executions=3, max_lm_calls=10))
    tracker.spend("notebook_executions", 2)
    tracker.spend("lm_calls", 4)
    snap = tracker.snapshot()
    assert snap["initial"]["max_notebook_executions"] == 3
    assert snap["used"]["max_notebook_executions"] == 2
    assert snap["remaining"]["max_notebook_executions"] == 1
    assert snap["used"]["max_lm_calls"] == 4
    assert snap["remaining"]["max_lm_calls"] == 6


def test_budget_from_dict_accepts_both_key_forms() -> None:
    b1 = Budget.from_dict({"max_notebook_executions": 4})
    b2 = Budget.from_dict({"notebook_executions": 4})
    assert b1.max_notebook_executions == 4
    assert b2.max_notebook_executions == 4


def test_unknown_resource_raises() -> None:
    tracker = BudgetTracker(Budget())
    with pytest.raises(KeyError):
        tracker.remaining("not_a_resource")
