# notebook-agent

A notebook-native, budget-aware coding/task agent that follows the
**Retrieve → Compose → Transform → Generate** processing strategy. Tasks and
subtasks are persisted as a hierarchical filesystem-backed graph of
notebooks, manifests, logs, inputs, outputs, and artifacts; the execution
unit is a [Papermill](https://papermill.readthedocs.io/)-executed Jupyter
notebook.

This package implements [`coding_agent_project_prompt.md`](coding_agent_project_prompt.md)
across all ten milestones.

> **Note**: the parent repository (`wiki3-ai/agent-client`) was used for an
> earlier exploratory implementation. The Python package shipped here is
> named **`notebook_agent`** and is the starting point for what will be
> moved to `wiki3-ai/notebook-agent`.

## UX: Jupyter notebooks

**There is no user-facing CLI.** The user experience for `notebook-agent`
is a regular Python Jupyter notebook — you import the package, call
`run_task(...)`, and use the `show_*` display helpers to inspect the
result inline. Agent skills may exec CLI tools internally (e.g. `git`,
`pytest`, language servers), but that is an implementation detail of a
skill, not part of the UX.

The dedicated "Agent Kernel" — a custom Jupyter kernel that mediates LLM
and tool calls — has been **deferred**. Users run inside a standard
Python kernel for now.

## Install

```bash
pip install -e ".[mcp,dev]"
```

Optional extras:

- `mcp` — MCP server wrapper (used by *other* agents, not by humans)
- `dev` — pytest + ruff

DSPy and Optuna are **required** runtime dependencies — `notebook-agent` IS a
[DSPy program](https://dspy.ai/) (`notebook_agent.NotebookAgentProgram`)
composed of typed `dspy.Signature`s. Every LLM-driven step (plan, skill
selection, code generation, parameter extraction, repair, synthesis) is a
`dspy.Predict` call against that program, which means optimizers like
**MIPROv2** and **GEPA** can compile the agent end-to-end:

```python
from notebook_agent import (
    NotebookAgentProgram, optimize_with_mipro, optimize_with_gepa,
)

base = NotebookAgentProgram()
compiled = optimize_with_mipro(base, trainset=my_examples, metric=my_metric)
# `compiled` is a drop-in replacement; pass it to run_task(..., program=compiled).
```

## Quickstart

Open [`examples/quickstart.ipynb`](examples/quickstart.ipynb) in Jupyter,
or run the same cells yourself:

```python
from notebook_agent import run_task, show_task, show_answer

# The only required argument is the prompt. The agent auto-builds its LLM
# client from environment variables (LM Studio by default) and figures out
# the rest. See nb-agent.md for the focused MVP spec.
result = run_task(
    "Create and execute a notebook that counts the words in: "
    "hello from the graph notebook agent"
)
show_task(result)     # → Markdown summary with plan, stage, result
show_answer(result)   # → the rendered answer
```

You can also use the IPython magics, which route through the same code path:

```python
%load_ext notebook_agent

%task count the words in "hello from the graph notebook agent"

# or for multi-line prompts:
%%task
Build a small parser that extracts the dates from this text and
returns them as a JSON list.
```

To continue an in-progress task with feedback or corrections, pass the
prior result back in:

```python
result2 = run_task("that's wrong, try again — count only unique words",
                   continue_from=result)
```

Or via the magic:

```python
%task --continue try again, count only unique words
```

### Per-call LM overrides

The magics accept a couple of flags before the prompt that override the
LM settings for that one call only (useful when a thinking model needs a
bigger token budget):

```python
%task --max-tokens 32000 explain in detail why ...
%task --temperature 0.7 --continue keep going but more creative
```

Both flags also work with `%%task`. Defaults come from
`NOTEBOOK_AGENT_MAX_TOKENS` (16384) and `NOTEBOOK_AGENT_TEMPERATURE`
(0.0); the override is scoped to the single magic invocation and
does not mutate the notebook-wide DSPy config.

### Notebook initialization (Papermill parameters + DSPy GEPA surface)

For a new user notebook the recommended layout is:

1. `%load_ext notebook_agent` to register the magics.
2. A Papermill **`parameters`**-tagged cell that declares the optimizable
   knobs (`provider`, `model`, `base_url`, `api_key`, `max_tokens`,
   `temperature`, `max_autonomous_turns`, `runs_root`, `skill_dirs`).
3. A follow-up cell that calls `notebook_agent.init_notebook(...)` with
   those names — this builds a `LiteLLMClient`, configures
   `dspy.settings.lm`, and stashes notebook-wide defaults that every
   `%task` / `%%task` magic picks up automatically.
4. Any number of `%task` / `%%task` cells.

See [examples/first.ipynb](examples/first.ipynb) for the canonical layout.

These same parameter names are the **DSPy GEPA hyperparameter search
space**. Call `notebook_agent.notebook_parameters()` to get the schema
(names, types, defaults, bounds). When the optimizer runs, it mutates
the values in the `parameters` cell, re-executes the notebook via
Papermill, and scores each trial against the trajectory captured under
`runs/` (events, manifests, executed notebooks).

The only enforced autonomy limit is `max_autonomous_turns` (default 6) —
the number of LLM-driven steps the agent may take before reporting back to
you. Set it on the call: `run_task(prompt, max_autonomous_turns=10)`.

This produces a run directory like:

```text
runs/2026/05/18/093015-use-the-echo-skill-...
├── task.json
├── manifest.json
├── README.md
├── task.ipynb
├── executed.ipynb
├── inputs/{request.md,parameters.json}
├── outputs/{result.json,answer.md}
├── logs/{events.jsonl,stdout.log,stderr.log,lm_calls.jsonl}
├── artifacts/
└── children/
```

## Public API

Everything a user needs lives at the package top level:

| Function / class | Purpose |
|---|---|
| `run_task(request, *, parameters=…, runs_root=…, budget=…, llm=…)` | Run a task end-to-end (Retrieve → Compose → Transform → Generate). |
| `search_skills(query, *, top=10)` | Lexical search over built-in + user skills. |
| `TaskGraph.load(directory)` | Reload an existing run as a task graph. |
| `Task`, `Budget`, `BudgetTracker` | Data types describing a task and its budget. |
| `show_task` / `show_answer` / `show_manifest` / `show_result` / `show_events` / `show_graph` / `show_notebook` | IPython display helpers for inline rendering. |

## Architecture summary

- **`task_graph`** owns the on-disk task layout (spec §7–§9).
- **`events`** is append-only JSONL.
- **`budget`** tracks `notebook_executions`, `lm_calls`, `repair_attempts`,
  `wall_time_seconds`, etc., and raises `BudgetExhaustedError` *before*
  expensive work happens.
- **`notebook_exec`** runs notebooks via Papermill, captures cell streams
  and errors, and integrates with the budget tracker and event log.
- **`skills`** is a lexical search over `manifest.json` + `SKILL.md` files
  under `builtin_skills/` (shipped) and any user-supplied directories.
- **`transform`** turns a `SKILL.md` into a parameterized notebook with the
  standard spec §12 cells (parameters / setup / validate / execute /
  write_result / manifest_update / smoke).
- **`repair`** classifies common failure modes and applies a deterministic
  patch. LLM-assisted repair is available when a configured `LiteLLMClient`
  is supplied.
- **`agent.run_task`** orchestrates R→C→T→G end-to-end, calls repair on
  failure and writes `outputs/answer.md`.
- **`display`** offers IPython-friendly helpers for inline rendering of
  tasks, manifests, event logs, and the task graph.
- **`mcp_server`** exposes the agent over MCP for **other agents**
  (`run_task`, `run_skill`, `search_skills`, `read_manifest`,
  `get_task_graph`, `execute_notebook`). Launch it with:

  ```bash
  python -m notebook_agent.mcp_server --runs-root runs
  ```

  This is not a user CLI; it's a server for inter-agent communication.
- **`litellm_client`** wraps LiteLLM with a `fake` provider (uses
  LiteLLM's `mock_response` kwarg) so tests and offline workflows stay
  deterministic. LM Studio is the default real backend
  (`NOTEBOOK_AGENT_BASE_URL`, default `http://host.docker.internal:1234/v1`).
- **`dspy_modules`** are stubs (`TaskRouter`, `SkillRetriever`,
  `SkillToNotebookTransformer`, `NotebookRepairer`, `ResultSynthesizer`,
  `ParameterExtractor`) ready to be replaced by DSPy programs.

## Milestone map

| # | Topic | Module | Test |
|---|---|---|---|
| 1 | Filesystem task graph | `task_graph.py`, `events.py` | `tests/integration/test_milestone_01_task_graph.py` |
| 2 | Papermill execution | `notebook_exec.py` | `tests/integration/test_milestone_02_notebook_exec.py` |
| 3 | Skill retrieval | `skills.py` + `builtin_skills/echo/` | `tests/integration/test_milestone_03_skills.py` |
| 4 | TODO decomposition | `planner.py` | `tests/integration/test_milestone_04_planner.py` |
| 5 | SKILL.md → notebook | `transform.py` | `tests/integration/test_milestone_05_transform.py` |
| 6 | Budget manager | `budget.py` | `tests/integration/test_milestone_06_budget.py`, `tests/unit/test_budget.py` |
| 7 | Repair loop | `repair.py` | `tests/integration/test_milestone_07_repair.py` |
| 8 | End-to-end agent | `agent.py` | `tests/integration/test_milestone_08_agent.py` |
| 9 | **Notebook UX** + MCP | `display.py`, `examples/quickstart.ipynb`, `mcp_server.py` | `tests/integration/test_milestone_09_notebook_ux.py` |
| 10 | LiteLLM + DSPy stubs | `litellm_client.py`, `dspy_modules.py` | `tests/integration/test_milestone_10_litellm.py` |

> Spec §17 defined a CLI for milestone 9. That requirement has been
> superseded: the user UX is a Jupyter notebook, demonstrated by
> `examples/quickstart.ipynb` and exercised by the milestone-9 test which
> papermill-executes the quickstart end-to-end.

## Testing

```bash
pytest                                    # all tests (live LM tests skip by default)
pytest -m "not live"                      # unit + integration only
NOTEBOOK_AGENT_LIVE_LM=1 pytest -m live   # exercises real LM Studio
ruff check notebook_agent tests
```

37 tests pass (1 live test is skipped unless LM Studio is reachable).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NOTEBOOK_AGENT_PROVIDER` | `lm_studio` | LLM provider (`lm_studio`, `fake`, or any LiteLLM model prefix) |
| `NOTEBOOK_AGENT_MODEL` | `lm_studio/model-name` | Model identifier passed to LiteLLM |
| `NOTEBOOK_AGENT_BASE_URL` | `http://host.docker.internal:1234/v1` | OpenAI-compatible base URL |
| `NOTEBOOK_AGENT_API_KEY` | `lm-studio` | API key passed to LiteLLM |
| `NOTEBOOK_AGENT_MAX_TOKENS` | `16384` | Max output tokens for every LM call. Bump higher for thinking models that stream long chain-of-thought before the answer. |
| `NOTEBOOK_AGENT_TEMPERATURE` | `0.0` | Sampling temperature passed to the LM. |
| `NOTEBOOK_AGENT_LIVE_LM` | _(unset)_ | Set to `1` to opt in to live LM tests |

## Non-goals (first version)

See spec §27. No multi-agent orchestration, no distributed scheduler, no
embedding search, no automatic GEPA optimization, and no custom Jupyter
"Agent Kernel" yet — those are explicitly deferred until the core RCTG
flow is solid.

## License

MIT (see `LICENSE`).
