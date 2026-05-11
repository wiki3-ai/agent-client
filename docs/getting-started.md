# Getting started with agent-kernel

This guide walks you through what the current MVP of `agent_kernel`
actually does, end-to-end, using only what's been built through
Milestone 9.

> **Note on scope.** The current control kernel is a thin subclass of
> `IPythonKernel` that adds `%agent` magics for orchestration. It is
> **not** yet a ReAct-loop kernel that runs an LLM-driven agent on your
> behalf. The pieces a ReAct loop needs — typed task lifecycle, child
> spawn with lineage, structured LLM calls with budget accounting,
> append-only provenance — are all here and integration-tested. The
> ReAct policy that ties them together is the planned next step (see
> "What's next" at the end of this guide).

## What you get today

| Capability | How you use it | Where it lives |
| --- | --- | --- |
| Execute any notebook with full per-cell provenance | `agent-kernel run` CLI **or** `NotebookRunner` | `agent_kernel.runtime.notebook_runner`, `agent_kernel.cli` |
| Create / queue / run tasks programmatically | `AgentKernel(...).create_task(...)` / `.run_task(...)` | `agent_kernel.api` |
| Spawn child tasks from a parent with budget reservation and lineage | `AgentKernel(...).spawn_child_task(...)` | `agent_kernel.runtime.spawn_manager` |
| Drive tasks from a notebook with `%agent` magics | `%agent task new …`, `%agent run …`, `%agent spawn …`, `%agent ledger tail` | `agent_kernel.kernel`, `agent_kernel.magics` |
| Structured LLM calls validated to Pydantic, with budget debit + provenance | `StructuredLLM(provider, agent_kernel=ak).generate(...)` | `agent_kernel.llm` |
| Append-only JSONL ledger + atomic JSON state snapshots | every event flows through `JSONLEventStore`; state in `<ws>/.agent_kernel/tasks/*.json` | `agent_kernel.storage` |
| HTTP surface over the same API | `agent_kernel_server` Jupyter Server extension | `agent_kernel_server.app` |

## Install

Requires **Python 3.13**.

```bash
git clone https://github.com/wiki3-ai/agent-client
cd agent-client
pip install -e ".[dev,server,llm]"
```

## A workspace, in one paragraph

Every public surface (CLI, Python API, magics, REST) is parameterised
by a **workspace directory**. The workspace owns:

```
<workspace>/
├── notebooks/                       # task notebooks you author or that get materialized
├── runs/<run_id>/
│   ├── executed.ipynb               # the executed notebook on disk
│   ├── stdout.log
│   └── stderr.log
└── .agent_kernel/
    ├── events/YYYY-MM-DD.jsonl      # append-only, redacted provenance ledger
    ├── tasks/<task_id>.json         # current state snapshots (atomic write)
    └── artifacts/                   # executable-cell artifacts
```

You don't have to create it; any of the entry points will call
`WorkspaceLayout(...).ensure()` for you. **The ledger is the source of
truth** — `scripts/replay.py <workspace>` will reconstruct in-memory
state from the JSONL alone.

## Example notebooks

The hands-on tour lives in `examples/`. Each notebook is short and
runnable against a temp workspace. They build on each other; do them
in order the first time through.

| # | Notebook | What it shows |
| --- | --- | --- |
| 01 | [`01_hello_runner.ipynb`](../examples/01_hello_runner.ipynb) | Execute an arbitrary notebook via `NotebookRunner`; inspect the JSONL trace it produced |
| 02 | [`02_python_api_tasks.ipynb`](../examples/02_python_api_tasks.ipynb) | Create + run tasks via `AgentKernel`; show the budget / quota / state-snapshot flow |
| 03 | [`03_spawn_lineage.ipynb`](../examples/03_spawn_lineage.ipynb) | Parent task spawns a child from the `python-analysis` template; lineage chain in the ledger |
| 04 | [`04_control_kernel_magics.ipynb`](../examples/04_control_kernel_magics.ipynb) | Run **on the `agent-kernel` kernel itself** and drive the system with `%agent` magics |
| 05 | [`05_llm_fake_provider.ipynb`](../examples/05_llm_fake_provider.ipynb) | `StructuredLLM` + `FakeProvider` returning a validated Pydantic object, with retry-on-validation and budget debit visible in the ledger |

The first three run on the stock `python3` kernel — they only import
`agent_kernel` as a library, so you don't need to install anything
extra. Notebook 04 needs the agent-kernel kernelspec installed; the
notebook itself shows you the one-liner.

## Quickstart: 30 seconds at the shell

```bash
mkdir -p /tmp/ak-demo && cd /tmp/ak-demo

# Author a tiny notebook
cat > hello.ipynb <<'JSON'
{"cells":[{"cell_type":"code","metadata":{},"source":"print('hi')","execution_count":null,"outputs":[]}],
 "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"}},
 "nbformat":4,"nbformat_minor":5}
JSON

# Run it through agent-kernel — produces executed notebook + JSONL ledger
agent-kernel run hello.ipynb --workspace .

# Inspect the ledger
cat .agent_kernel/events/*.jsonl | head -n 20
```

You'll see events like `task.created`, `notebook.execution.started`,
`cell.execution.started`, `cell.execution.completed`,
`notebook.execution.completed`, `task.completed`. That's the substrate
every higher layer is built on.

## Quickstart: same thing, Python API

```python
from agent_kernel.api import AgentKernel

ak = AgentKernel(workspace="/tmp/ak-demo")
task = ak.create_task(notebook_path="hello.ipynb", kernel_name="python3")
final = ak.run_task(task.task_id)
print(final.status, final.executed_notebook_path)

for e in ak.list_events(task.task_id):
    print(e.ts, e.event_type.value)
```

## Quickstart: same thing, in the control kernel

```text
%agent task new hello.ipynb
{"ok": true, "task_id": "task_…"}

%agent run task_…
{"ok": true, "task_id": "task_…", "status": "completed"}

%agent ledger tail 5
{"ok": true, "events": [{"ts": "…", "type": "task.completed", "task_id": "task_…"}, …]}
```

Install the kernelspec first:

```bash
python -m agent_kernel install --user
```

Then point JupyterLab (or `nbclient`) at the `agent-kernel` kernel and
the cell magics above will work.

## Policy profiles, budgets, and quotas

`PolicyProfile` is the single object that controls admission and
spending. Two are shipped: `local-dev` (generous, for development) and
`research` (tighter caps). You can inspect the active profile and the
current quota snapshot from a notebook:

```text
%agent policy show
%agent quota
```

Or from Python:

```python
ak.scheduler.profile.model_dump()
ak.scheduler._quota_snapshot_locked().model_dump()
```

The scheduler **reserves** the full profile budget for a new top-level
task; spawned children get a reservation **carved out of the parent's
remaining budget** (`SpawnManager` enforces `max_spawn_depth`,
`max_children_per_task`, and `parent_reserve_floor_ratio`). Failures
and cancellations **refund** the unspent reservation. The integer-
precise debit math means the JSONL ledger's running budget is exactly
non-negative — `tests/integration/test_m4_scheduler.py` proves this
across success / fail / cancel mixes.

## Structured LLM calls

`StructuredLLM` accepts any provider that implements the `Provider`
Protocol. Two ship:

- **`FakeProvider`** — deterministic, in-process. Used by all hermetic
  integration tests in CI. You can give it a `script` (list of JSON
  strings to return in order) or a `handler` callback.
- **`LMStudioProvider`** — OpenAI-compatible HTTP to a local LM Studio
  server (default `http://localhost:1234/v1`). `is_reachable()` lets
  you skip cleanly when LM Studio isn't running.

A successful `generate()` emits `llm.call.started` →
`llm.call.completed` (with `prompt_tokens`, `completion_tokens`,
`cost_usd_micro`, `attempts`, `budget_before`, `budget_after`) →
`budget.debited`. Retry-on-validation-error is built in:
`StructuredLLM(max_retries=2)` will append the validator's complaint to
the conversation and ask the model to try again.

See `examples/05_llm_fake_provider.ipynb` for the full flow.

## Inspecting and replaying the ledger

The JSONL files in `.agent_kernel/events/` are the durable truth.
Useful one-liners:

```bash
# Quick tail
tail -n 20 .agent_kernel/events/*.jsonl | python -m json.tool --json-lines

# Filter to one task
jq -c 'select(.task_id == "task_abc123")' .agent_kernel/events/*.jsonl

# Reconstruct ledger state + stable digest
python scripts/replay.py .
```

Every event is run through `agent_kernel.security.redact_payload`
before it hits disk — OpenAI / Anthropic / GitHub / AWS / JWT / bearer
/ private-key patterns and field names like `api_key`, `password`,
`token` are scrubbed at the storage boundary. This is defense-in-depth,
not a substitute for sourcing secrets from env vars in the first place.

## What's next (ReAct policy on top of this substrate)

The current `AgentControlKernel` is a thin IPython subclass with
`%agent` magics. The intended next step is to make the kernel itself
run a **ReAct loop**: a top-level task notebook describes the goal and
available tools, and the kernel drives an LLM through observe → think
→ act → repeat cycles, spawning child tasks (each its own notebook)
for tool calls and child reasoning steps. The pieces this needs are
already in place:

- Typed task lifecycle and scheduler → `agent_kernel.api`
- Parent → child spawn with budget reservation and lineage →
  `agent_kernel.runtime.spawn_manager`
- Structured LLM calls with retry, validation, and budget accounting →
  `agent_kernel.llm`
- Append-only provenance for everything → `agent_kernel.storage`

A subsequent milestone will add a `ReActPolicy` runtime in
`agent_kernel.runtime` that consumes these and exposes itself through a
`%agent agent run …` magic plus a Python entry point. The
integration-test gate will be: drive a fixture goal notebook through
the control kernel against `FakeProvider` with a scripted think/act
trace, assert the JSONL ledger contains the full ReAct chain.

Until that lands, you can already prototype a ReAct loop in user code
by combining `StructuredLLM.generate()` with `ak.spawn_child_task()` —
see `examples/03_spawn_lineage.ipynb` and `examples/05_llm_fake_provider.ipynb`
for the building blocks.
