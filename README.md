# notebook-agent

A notebook-native, budget-aware coding/task agent that follows the
**Retrieve → Compose → Transform → Generate** processing strategy. Tasks and
subtasks are persisted as a hierarchical filesystem-backed graph of
notebooks, manifests, logs, inputs, outputs, and artifacts; the execution
unit is a [Papermill](https://papermill.readthedocs.io/)-executed Jupyter
notebook.

This package implements [`coding_agent_project_prompt.md`](coding_agent_project_prompt.md)
end-to-end across all ten milestones.

> **Note**: the parent repository (`wiki3-ai/agent-client`) was used for an
> earlier exploratory implementation. The Python package shipped here is
> named **`notebook_agent`** (per §13 of the prompt) and is the starting
> point for what will be moved to `wiki3-ai/notebook-agent`.

## Install

```bash
pip install -e ".[llm,mcp,dev]"
```

Optional extras:
- `llm` — LiteLLM client (LM Studio / OpenAI / etc.)
- `mcp` — MCP server wrapper
- `dev` — pytest + ruff

## Quickstart

```bash
notebook-agent init .
notebook-agent run "Use the echo skill to echo hello graph agent" \
    --params <(echo '{"message":"hello graph agent"}')
notebook-agent graph runs/<YYYY>/<MM>/<DD>/<HHMMSS-task-slug>
notebook-agent manifest runs/<YYYY>/<MM>/<DD>/<HHMMSS-task-slug>
```

You will get a run directory like:

```text
runs/2026/05/18/083512-use-the-echo-skill-...
├── task.json
├── manifest.json
├── README.md
├── task.ipynb
├── executed.ipynb
├── inputs/{request.md,parameters.json}
├── outputs/{result.json,answer.md}
├── logs/{events.jsonl,stdout.log,stderr.log}
├── artifacts/
└── children/
```

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
| 9 | CLI + MCP service | `cli.py`, `mcp_server.py` | `tests/integration/test_milestone_09_cli_mcp.py` |
| 10 | LiteLLM + DSPy stubs | `litellm_client.py`, `dspy_modules.py` | `tests/integration/test_milestone_10_litellm.py` |

## Architecture summary

- **`task_graph`** owns the on-disk task layout (Sections 7–9 of the spec).
- **`events`** is append-only JSONL.
- **`budget`** tracks `notebook_executions`, `lm_calls`, `repair_attempts`,
  `wall_time_seconds`, etc., and raises `BudgetExhaustedError` *before*
  expensive work happens.
- **`notebook_exec`** runs notebooks via Papermill, captures cell streams
  and errors, and integrates with the budget tracker and event log.
- **`skills`** is a lexical search over `manifest.json` + `SKILL.md` files
  under `builtin_skills/` (shipped) and any user-supplied directories.
- **`transform`** turns a `SKILL.md` into a parameterized notebook with the
  standard Section 12 cells (parameters / setup / validate / execute /
  write_result / manifest_update / smoke).
- **`repair`** classifies common failure modes (missing output directory,
  unknown name, missing import) and applies a deterministic patch.
  LLM-assisted repair is available when a configured `LiteLLMClient` is
  supplied.
- **`agent.run_task`** orchestrates R→C→T→G end-to-end, calling repair on
  failure and writing `outputs/answer.md`.
- **`cli`** is a Typer app exposing `init`, `run`, `search-skills`,
  `execute-notebook`, `graph`, `manifest`, and `mcp`.
- **`mcp_server`** exposes the same surface via FastMCP for use by other
  agents (`run_task`, `run_skill`, `search_skills`, `read_manifest`,
  `get_task_graph`, `execute_notebook`).
- **`litellm_client`** wraps LiteLLM with a `fake` provider (uses
  LiteLLM's `mock_response` kwarg) so tests and offline workflows stay
  deterministic. LM Studio is the default real backend
  (`NOTEBOOK_AGENT_BASE_URL`, default `http://host.docker.internal:1234/v1`).
- **`dspy_modules`** are stubs (`TaskRouter`, `SkillRetriever`,
  `SkillToNotebookTransformer`, `NotebookRepairer`, `ResultSynthesizer`)
  ready to be replaced by DSPy programs.

## Testing

```bash
pytest                            # all tests (live ones skip without env var)
pytest -m "not live"              # unit + integration only
NOTEBOOK_AGENT_LIVE_LM=1 pytest -m live  # exercises real LM Studio
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
| `NOTEBOOK_AGENT_LIVE_LM` | _(unset)_ | Set to `1` to opt in to live LM tests |

## Non-goals (first version)

See spec §27. No multi-agent orchestration, no distributed scheduler, no
embedding search, no automatic GEPA optimization yet — those are explicitly
deferred until the core RCTG flow is solid.

## License

MIT (see `LICENSE`).
