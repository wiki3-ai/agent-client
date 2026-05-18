"""Notebook-native budget-aware Retrieve/Compose/Transform/Generate agent.

The user UX for this package is a **Jupyter notebook**. There is no
user-facing CLI. A typical session in a notebook looks like::

    from notebook_agent import run_task, show_task, show_answer
    result = run_task("Use the echo skill to echo hello",
                      parameters={"message": "hello"})
    show_task(result)      # → rich Markdown summary
    show_answer(result)    # → the rendered answer

The MCP server (for other agents, not humans) is launched with
``python -m notebook_agent.mcp_server`` and is not part of the user UX.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .agent import AgentResult, run_task
from .budget import Budget, BudgetExhaustedError, BudgetTracker
from .display import (
    show_answer,
    show_events,
    show_graph,
    show_manifest,
    show_notebook,
    show_result,
    show_task,
)
from .dspy_lm import build_dspy_lm, configure_dspy, using_client
from .events import EventLog
from .litellm_client import LiteLLMClient
from .magics import load_ipython_extension, unload_ipython_extension
from .notebook_init import (
    NotebookConfig,
    get_notebook_config,
    init_notebook,
    notebook_parameters,
)
from .optimize import optimize_with_gepa, optimize_with_mipro
from .program import (
    ChooseSkill,
    ExtractParameters,
    GenerateCode,
    NotebookAgentProgram,
    PlanTask,
    RepairNotebook,
    RouteTask,
    SynthesizeAnswer,
)
from .skills import Skill, SkillRepository
from .task_graph import Task, TaskGraph, create_root_task


def root_template_path():
    """Return the path to the bundled Papermill root agent notebook template."""
    from pathlib import Path

    return Path(__file__).resolve().parent / "templates" / "root_agent.ipynb"


__all__ = [
    "AgentResult",
    "Budget",
    "BudgetExhaustedError",
    "BudgetTracker",
    "ChooseSkill",
    "EventLog",
    "ExtractParameters",
    "GenerateCode",
    "LiteLLMClient",
    "NotebookAgentProgram",
    "NotebookConfig",
    "PlanTask",
    "RepairNotebook",
    "RouteTask",
    "Skill",
    "SkillRepository",
    "SynthesizeAnswer",
    "Task",
    "TaskGraph",
    "__version__",
    "build_dspy_lm",
    "configure_dspy",
    "create_root_task",
    "get_notebook_config",
    "init_notebook",
    "load_ipython_extension",
    "notebook_parameters",
    "optimize_with_gepa",
    "optimize_with_mipro",
    "root_template_path",
    "run_task",
    "show_answer",
    "show_events",
    "show_graph",
    "show_manifest",
    "show_notebook",
    "show_result",
    "show_task",
    "unload_ipython_extension",
    "using_client",
]
