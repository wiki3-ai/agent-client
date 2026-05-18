"""Notebook-native budget-aware Retrieve/Compose/Transform/Generate agent."""

from __future__ import annotations

__version__ = "0.1.0"

from .budget import Budget, BudgetExhaustedError, BudgetTracker
from .events import EventLog
from .task_graph import Task, TaskGraph, create_root_task

__all__ = [
    "Budget",
    "BudgetExhaustedError",
    "BudgetTracker",
    "EventLog",
    "Task",
    "TaskGraph",
    "__version__",
    "create_root_task",
]
