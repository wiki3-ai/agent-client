"""Build the example notebooks from in-line cell sources.

Run from the repo root::

    python examples/_build_notebooks.py

Re-runnable; overwrites the .ipynb files in this directory.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).parent


def _nb(cells: list) -> dict:
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
    )
    return nb


def _nb_agent_kernel(cells: list) -> dict:
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"name": "agent-kernel", "display_name": "agent-kernel"},
            "language_info": {"name": "python"},
        },
    )
    return nb


def build_01() -> dict:
    return _nb(
        [
            new_markdown_cell(
                "# 01 — Hello, runner\n"
                "\n"
                "Execute a notebook through `agent_kernel.runtime.notebook_runner.NotebookRunner` "
                "and inspect the JSONL provenance ledger it produced.\n"
                "\n"
                "This is the smallest useful thing the system does: take a notebook on disk, "
                "execute it via `nbclient` against a real Jupyter kernel, capture per-cell "
                "provenance, and write an executed-notebook artifact to a run-scoped directory.\n"
                "\n"
                "**Kernel:** standard `python3`. We use `agent_kernel` purely as a library here."
            ),
            new_code_cell(
                "import json, tempfile, shutil\n"
                "from pathlib import Path\n"
                "\n"
                "from agent_kernel.runtime.notebook_runner import NotebookRunner\n"
                "from agent_kernel.storage import JSONLEventStore, WorkspaceLayout\n"
                "from agent_kernel.util import new_id\n"
                "\n"
                "# Use a throwaway workspace so this notebook is hermetic and re-runnable.\n"
                "workspace = Path(tempfile.mkdtemp(prefix='ak-ex01-'))\n"
                "ws = WorkspaceLayout(workspace); ws.ensure()\n"
                "print('workspace:', workspace)"
            ),
            new_markdown_cell(
                "## 1. Author a tiny notebook on disk\n"
                "\n"
                "Anything `nbclient` can execute will work. We use three cells: a normal "
                "computation, a print, and a final expression so we can see outputs."
            ),
            new_code_cell(
                "import nbformat\n"
                "from nbformat.v4 import new_notebook, new_code_cell\n"
                "\n"
                "target = workspace / 'hello.ipynb'\n"
                "nbformat.write(new_notebook(\n"
                "    cells=[\n"
                "        new_code_cell('x = 6 * 7'),\n"
                "        new_code_cell(\"print(f'the answer is {x}')\"),\n"
                "        new_code_cell('x'),\n"
                "    ],\n"
                "    metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},\n"
                "), target)\n"
                "print('wrote', target)"
            ),
            new_markdown_cell(
                "## 2. Run it through `NotebookRunner`\n"
                "\n"
                "The runner takes an event store and a runs directory; for every nbclient hook "
                "it emits a typed `ProvenanceEvent` to the JSONL ledger and writes the executed "
                "notebook + captured stdout/stderr to `runs/<run_id>/`."
            ),
            new_code_cell(
                "events = JSONLEventStore(ws.events_dir, fsync=True)\n"
                "runner = NotebookRunner(events, ws.runs_dir)\n"
                "result = runner.run(target, task_id=new_id('task'), kernel_name='python3', timeout=30)\n"
                "print('run_id:        ', result.run_id)\n"
                "print('task_id:       ', result.task_id)\n"
                "print('executed file: ', result.executed_notebook_path)\n"
                "print('exists:        ', Path(result.executed_notebook_path).exists())"
            ),
            new_markdown_cell(
                "## 3. Inspect the executed notebook\n"
                "\n"
                "The third cell evaluated `x`, so its output is `42`. The runner preserves "
                "outputs exactly as `nbclient` produced them."
            ),
            new_code_cell(
                "executed = nbformat.read(result.executed_notebook_path, as_version=4)\n"
                "for i, cell in enumerate(executed.cells):\n"
                "    outs = [o.get('text') or o.get('data', {}).get('text/plain') for o in cell.get('outputs', [])]\n"
                "    print(f'cell {i}: source={cell.source!r:40s}  outputs={outs}')"
            ),
            new_markdown_cell(
                "## 4. Walk the JSONL ledger\n"
                "\n"
                "Every event the runner emitted is on disk as a single JSON line in "
                "`<workspace>/.agent_kernel/events/YYYY-MM-DD.jsonl`. The event types you'll see "
                "for a clean run are: `task.created` (synthesized for the runner's task scope), "
                "`notebook.execution.started`, one `cell.execution.started` + `cell.execution.completed` "
                "per code cell, `notebook.execution.completed`, and `task.completed`."
            ),
            new_code_cell(
                "lines = []\n"
                "for f in sorted(ws.events_dir.glob('*.jsonl')):\n"
                "    lines.extend(f.read_text().splitlines())\n"
                "print(f'{len(lines)} events on disk')\n"
                "for ln in lines:\n"
                "    e = json.loads(ln)\n"
                "    print(f\"{e['ts']}  {e['event_type']:35s}  {e.get('task_id','-')}\")"
            ),
            new_markdown_cell(
                "## 5. Same thing from the CLI\n"
                "\n"
                "If you don't need to embed this in Python code, the same flow is one shell "
                "command:\n"
                "\n"
                "```bash\n"
                "agent-kernel run hello.ipynb --workspace .\n"
                "```\n"
                "\n"
                "This is exercised in `tests/integration/test_m2_notebook_runner.py`."
            ),
            new_code_cell(
                "# Tidy up the temp workspace.\n"
                "shutil.rmtree(workspace, ignore_errors=True)\n"
                "print('cleaned')"
            ),
        ]
    )


def build_02() -> dict:
    return _nb(
        [
            new_markdown_cell(
                "# 02 — Tasks, budgets, and the ledger via `AgentKernel`\n"
                "\n"
                "`AgentKernel` is the public Python facade. It wraps the scheduler, the spawn "
                "manager, and the JSONL/state stores into a single object. Use it when you want "
                "tasks as first-class entities (with budget reservation, refund-on-failure, "
                "and policy admission) rather than raw notebook execution.\n"
                "\n"
                "**Kernel:** standard `python3`."
            ),
            new_code_cell(
                "import json, tempfile, shutil\n"
                "from pathlib import Path\n"
                "\n"
                "import nbformat\n"
                "from nbformat.v4 import new_notebook, new_code_cell\n"
                "\n"
                "from agent_kernel.api import AgentKernel\n"
                "\n"
                "workspace = Path(tempfile.mkdtemp(prefix='ak-ex02-'))\n"
                "print('workspace:', workspace)"
            ),
            new_markdown_cell(
                "## 1. Open a workspace under the `local-dev` profile\n"
                "\n"
                "Constructing `AgentKernel(workspace)` opens (or creates) the workspace, loads "
                "the requested `PolicyProfile`, and starts the in-process scheduler. The two "
                "shipped profiles are `local-dev` (generous defaults) and `research` (tighter "
                "caps)."
            ),
            new_code_cell(
                "ak = AgentKernel(workspace, policy_profile='local-dev')\n"
                "print('profile:', ak.scheduler.profile.name)\n"
                "print('initial budget:', ak.scheduler.profile.budgets.model_dump())\n"
                "print('quota:', ak.scheduler._quota_snapshot_locked().model_dump())"
            ),
            new_markdown_cell(
                "## 2. Create + run a task\n"
                "\n"
                "`create_task` writes the initial `TaskSpec` snapshot and emits `task.created`. "
                "`run_task` admits it through the policy engine, executes the notebook via the "
                "runner, and emits the full chain through to `task.completed` (or `task.failed`)."
            ),
            new_code_cell(
                "nb_path = workspace / 'work.ipynb'\n"
                "nbformat.write(new_notebook(\n"
                "    cells=[\n"
                "        new_code_cell('total = sum(range(10))'),\n"
                "        new_code_cell('total'),\n"
                "    ],\n"
                "    metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},\n"
                "), nb_path)\n"
                "\n"
                "task = ak.create_task(notebook_path=str(nb_path), kernel_name='python3')\n"
                "print('created:  ', task.task_id, task.status.value)\n"
                "print('reserved: ', task.reserved_budget.model_dump())\n"
                "\n"
                "final = ak.run_task(task.task_id)\n"
                "print('final:    ', final.status.value)\n"
                "print('spent:    ', final.spent_budget.model_dump())\n"
                "print('executed: ', final.executed_notebook_path)"
            ),
            new_markdown_cell(
                "## 3. Inspect the event chain for this task\n"
                "\n"
                "`list_events(task_id)` filters the JSONL ledger to one task — useful for "
                "test assertions and for live introspection."
            ),
            new_code_cell(
                "for e in ak.list_events(task.task_id):\n"
                "    print(f'{e.ts}  {e.event_type.value:35s}  {(e.payload or {}).get(\"reason\", \"\")}')"
            ),
            new_markdown_cell(
                "## 4. Show the on-disk state snapshot\n"
                "\n"
                "The scheduler atomically writes a snapshot of every task to "
                "`<ws>/.agent_kernel/tasks/<task_id>.json` (temp-write-then-rename). The same "
                "object is recoverable from the JSONL alone — see `scripts/replay.py`."
            ),
            new_code_cell(
                "snapshot = workspace / '.agent_kernel' / 'tasks' / f'{task.task_id}.json'\n"
                "print(snapshot.read_text())"
            ),
            new_markdown_cell(
                "## 5. Run a task that fails — see refund on the budget\n"
                "\n"
                "A failed task **refunds its unspent reservation** so the running budget stays "
                "exact. We'll write a notebook that raises in cell 1 and watch for `task.failed` "
                "plus the corresponding `budget.refunded`."
            ),
            new_code_cell(
                "fail_nb = workspace / 'boom.ipynb'\n"
                "nbformat.write(new_notebook(\n"
                "    cells=[new_code_cell('raise RuntimeError(\"boom\")')],\n"
                "    metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},\n"
                "), fail_nb)\n"
                "\n"
                "failing = ak.create_task(notebook_path=str(fail_nb), kernel_name='python3')\n"
                "out = ak.run_task(failing.task_id)\n"
                "print('final status:', out.status.value)\n"
                "print()\n"
                "events = ak.list_events(failing.task_id)\n"
                "for e in events:\n"
                "    print(f'{e.event_type.value:35s}  {(e.payload or {}).get(\"reason\", \"\")}')"
            ),
            new_code_cell(
                "shutil.rmtree(workspace, ignore_errors=True)"
            ),
        ]
    )


def build_03() -> dict:
    return _nb(
        [
            new_markdown_cell(
                "# 03 — Parent → child spawn with lineage\n"
                "\n"
                "A parent task can spawn a child task that materializes a fresh notebook from "
                "a template, runs against the **parent's remaining budget reservation**, and "
                "feeds its result back to the parent. The spawn manager enforces "
                "`max_spawn_depth`, `max_children_per_task`, and `parent_reserve_floor_ratio` "
                "from the active policy profile.\n"
                "\n"
                "This example shows the parent driving from outside the kernel (Python API) "
                "for clarity; the same call works from inside a running notebook — the "
                "M5 integration test (`test_m5_spawn.py`) exercises that path.\n"
                "\n"
                "**Kernel:** standard `python3`."
            ),
            new_code_cell(
                "import json, tempfile, shutil\n"
                "from pathlib import Path\n"
                "\n"
                "import nbformat\n"
                "from nbformat.v4 import new_notebook, new_code_cell\n"
                "\n"
                "from agent_kernel.api import AgentKernel\n"
                "from agent_kernel.models.task import SpawnSpec\n"
                "from agent_kernel.runtime.template_registry import list_templates\n"
                "\n"
                "workspace = Path(tempfile.mkdtemp(prefix='ak-ex03-'))\n"
                "ak = AgentKernel(workspace)\n"
                "print('templates available:', list_templates())"
            ),
            new_markdown_cell(
                "## 1. Create the parent task\n"
                "\n"
                "The parent's notebook can be anything — for this demo it's a trivial one-cell "
                "notebook; the interesting work happens in the child spawn we issue against it."
            ),
            new_code_cell(
                "parent_nb = workspace / 'parent.ipynb'\n"
                "nbformat.write(new_notebook(\n"
                "    cells=[new_code_cell('print(\"parent setup\")')],\n"
                "    metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},\n"
                "), parent_nb)\n"
                "\n"
                "parent = ak.create_task(notebook_path=str(parent_nb), kernel_name='python3')\n"
                "print('parent task:', parent.task_id, 'depth:', parent.depth)"
            ),
            new_markdown_cell(
                "## 2. Spawn a child from the `python-analysis` template\n"
                "\n"
                "The template ships with the package (`agent_kernel/templates/python-analysis.ipynb`) "
                "and accepts `query` and `limit` parameters. The materializer injects them via the "
                "two-channel model: an executable parameter cell (Python) **and** a metadata "
                "block at `metadata.agent_kernel.inputs` for non-Python kernels."
            ),
            new_code_cell(
                "spawn_result = ak.spawn_child_task(\n"
                "    parent.task_id,\n"
                "    SpawnSpec(\n"
                "        template_name='python-analysis',\n"
                "        parameters={'query': 'recent papers', 'limit': 5},\n"
                "        kernel_name='python3',\n"
                "    ),\n"
                ")\n"
                "print('allowed:        ', spawn_result.allowed)\n"
                "print('reason:         ', spawn_result.reason)\n"
                "child = spawn_result.child_task\n"
                "print('child task:     ', child.task_id)\n"
                "print('  parent:       ', child.parent_task_id)\n"
                "print('  spawn_index:  ', child.spawn_index)\n"
                "print('  depth:        ', child.depth)\n"
                "print('  reserved:     ', child.reserved_budget.model_dump())\n"
                "print('  materialized: ', child.notebook_path)"
            ),
            new_markdown_cell(
                "## 3. Run the child\n"
                "\n"
                "`spawn_child_task` only materializes and reserves; the caller runs the child "
                "explicitly. This keeps spawn pure (no I/O beyond materialize + ledger emit) "
                "and lets the parent decide whether to await or fire-and-forget."
            ),
            new_code_cell(
                "final_child = ak.run_task(child.task_id)\n"
                "print('child final:', final_child.status.value)\n"
                "print('executed:   ', final_child.executed_notebook_path)"
            ),
            new_markdown_cell(
                "## 4. The full lineage chain in the JSONL\n"
                "\n"
                "Walking events filtered to **both** task ids shows the canonical chain:\n"
                "\n"
                "```\n"
                "task.created (parent)\n"
                "task.spawn.requested (parent)\n"
                "notebook.materialized (child)\n"
                "task.created (child)\n"
                "task.spawned (parent, references child)\n"
                "notebook.execution.started (child)\n"
                "cell.execution.* …\n"
                "notebook.execution.completed (child)\n"
                "task.completed (child)\n"
                "```\n"
            ),
            new_code_cell(
                "ids = {parent.task_id, child.task_id}\n"
                "for e in ak.list_events():\n"
                "    if e.task_id in ids:\n"
                "        print(f'{e.event_type.value:35s}  task={e.task_id}')"
            ),
            new_markdown_cell(
                "## 5. Verify the injected parameters made it into the child\n"
                "\n"
                "The materializer wrote `query` and `limit` into the notebook's metadata; the "
                "Python parameter injector also wrote a `# Injected by agent-kernel` cell. We "
                "can read both from the executed notebook."
            ),
            new_code_cell(
                "executed = nbformat.read(final_child.executed_notebook_path, as_version=4)\n"
                "print('metadata.agent_kernel.inputs:', json.dumps(executed.metadata.get('agent_kernel', {}).get('inputs', {}), indent=2))\n"
                "print()\n"
                "for c in executed.cells:\n"
                "    if 'injected-parameters' in (c.metadata.get('tags') or []):\n"
                "        print('--- injected cell ---')\n"
                "        print(c.source)\n"
                "        break"
            ),
            new_code_cell("shutil.rmtree(workspace, ignore_errors=True)"),
        ]
    )


def build_04() -> dict:
    # NOTE: this notebook is designed to run on the agent-kernel kernel.
    return _nb_agent_kernel(
        [
            new_markdown_cell(
                "# 04 — Driving the system with `%agent` magics\n"
                "\n"
                "This notebook runs on the **`agent-kernel`** kernel itself (not `python3`). "
                "Install it once with:\n"
                "\n"
                "```bash\n"
                "python -m agent_kernel install --user\n"
                "```\n"
                "\n"
                "Then in JupyterLab choose the `agent-kernel` kernel for this notebook.\n"
                "\n"
                "Under the agent-kernel:\n"
                "- Cells starting with `%agent ...` are dispatched through `agent_kernel.magics`.\n"
                "- Any other cell executes as normal Python (the kernel subclasses `IPythonKernel`).\n"
                "- A single `AgentKernel` instance is bound per kernel process, scoped to the\n"
                "  workspace from `$AGENT_KERNEL_WORKSPACE` (defaults to the kernel's cwd).\n"
                "\n"
                "> If you opened this notebook on `python3` by mistake, every `%agent` cell will\n"
                "> fail with `UsageError: Line magic function '%agent' not found.` That's the\n"
                "> signal to switch kernels via *Kernel → Change Kernel*."
            ),
            new_markdown_cell(
                "## 1. What magics are available?\n"
                "\n"
                "`%agent help` lists every dispatcher entry. The handlers are line-oriented, "
                "shlex-parsed, and return JSON for easy programmatic consumption."
            ),
            new_code_cell("%agent help"),
            new_markdown_cell(
                "## 2. Inspect policy and quota"
            ),
            new_code_cell("%agent policy show"),
            new_code_cell("%agent quota"),
            new_markdown_cell(
                "## 3. Author a notebook on disk, then create + run a task for it\n"
                "\n"
                "Because we are on the `agent-kernel` (an IPython subclass), regular Python "
                "still works for setup."
            ),
            new_code_cell(
                "from pathlib import Path\n"
                "import nbformat\n"
                "from nbformat.v4 import new_notebook, new_code_cell\n"
                "\n"
                "nb = Path.cwd() / 'demo.ipynb'\n"
                "nbformat.write(new_notebook(\n"
                "    cells=[new_code_cell(\"print('hello from the child notebook')\")],\n"
                "    metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},\n"
                "), nb)\n"
                "print('wrote', nb)"
            ),
            new_code_cell("%agent task new demo.ipynb --kernel python3"),
            new_markdown_cell(
                "Grab the `task_id` from the JSON output above and run it. (In a real workflow "
                "you'd capture it via Python; for the demo, copy-paste.)"
            ),
            new_code_cell(
                "# Show all task ids currently known to the workspace\n"
                "%agent task list"
            ),
            new_markdown_cell(
                "Use one of those ids in the next two cells. Magics accept a single positional "
                "argument:"
            ),
            new_code_cell(
                "# Replace TASK_ID with one from the list above before executing.\n"
                "# %agent task status TASK_ID\n"
                "# %agent run TASK_ID"
            ),
            new_markdown_cell(
                "## 4. Tail the ledger\n"
                "\n"
                "`%agent ledger tail N` returns the last N events from the JSONL store. Add "
                "`--task TASK_ID` to scope to a single task."
            ),
            new_code_cell("%agent ledger tail 10"),
            new_markdown_cell(
                "## 5. Spawn a child from a template\n"
                "\n"
                "`%agent spawn <parent_task_id> <template> [--param k=v]... [--kernel python3]` "
                "uses the spawn manager from notebook 03 under the hood. Parameters are parsed "
                "as JSON when possible, otherwise as strings.\n"
                "\n"
                "```\n"
                "%agent spawn TASK_ID python-analysis --param query=\"recent papers\" --param limit=5\n"
                "```"
            ),
            new_markdown_cell(
                "## What the kernel is **not** doing yet\n"
                "\n"
                "Right now the kernel is a magic dispatcher plus normal Python. It does **not** "
                "yet drive an LLM-backed ReAct loop on your behalf — you compose the orchestration "
                "explicitly via magics or via the Python API.\n"
                "\n"
                "The substrate for a ReAct kernel is all here (task lifecycle, spawn lineage, "
                "structured LLM with budget accounting, append-only provenance); adding a "
                "`ReActPolicy` runtime + `%agent agent run` magic is the next milestone. See the "
                "*What's next* section of `docs/getting-started.md`."
            ),
        ]
    )


def build_05() -> dict:
    return _nb(
        [
            new_markdown_cell(
                "# 05 — Structured LLM calls (LiteLLM under the hood)\n"
                "\n"
                "`StructuredLLM` is the LLM adapter; the underlying provider logic is "
                "**LiteLLM**. The two presets we ship are thin wrappers over "
                "[`litellm.completion`](https://docs.litellm.ai/docs/completion/):\n"
                "\n"
                "- `FakeProvider` — uses LiteLLM's canonical `mock_response` kwarg "
                "([docs](https://docs.litellm.ai/docs/completion/mock_requests)) to script "
                "responses for hermetic tests. No real network call; no bespoke wire format.\n"
                "- `LMStudioProvider` — routes through LiteLLM's built-in `lm_studio/<model>` "
                "provider against a local LM Studio server. **Auto-detects the loaded model** "
                "from `/v1/models` if you don't pass one.\n"
                "\n"
                "`StructuredLLM` itself adds:\n"
                "\n"
                "1. JSON-schema injection from the Pydantic `response_model`.\n"
                "2. Validation + **retry-on-validation-error** with the validator message fed "
                "back to the model so it can self-correct.\n"
                "3. Provenance: `llm.call.started` → `llm.call.completed` (with token usage, "
                "cost, attempts, budget before/after) → `budget.debited`.\n"
                "4. Integer-precise micro-USD budget debits (the ledger never goes negative).\n"
                "\n"
                "**Kernel:** standard `python3`."
            ),
            new_code_cell(
                "import json, tempfile, shutil\n"
                "from pathlib import Path\n"
                "\n"
                "from pydantic import BaseModel, Field\n"
                "\n"
                "from agent_kernel.api import AgentKernel\n"
                "from agent_kernel.llm import FakeProvider, StructuredLLM, LLMCallError\n"
                "from agent_kernel.models.event import EventType\n"
                "\n"
                "workspace = Path(tempfile.mkdtemp(prefix='ak-ex05-'))\n"
                "ak = AgentKernel(workspace)\n"
                "task = ak.create_task(notebook_path=str(workspace / 'host.ipynb'), kernel_name='python3')\n"
                "print('task:', task.task_id)"
            ),
            new_markdown_cell(
                "## 1. Define the response schema\n"
                "\n"
                "`response_model` is any `pydantic.BaseModel`. Its `model_json_schema()` is "
                "what the adapter forwards to the provider."
            ),
            new_code_cell(
                "class Sentiment(BaseModel):\n"
                "    label: str = Field(description='positive | negative | neutral')\n"
                "    confidence: float = Field(ge=0.0, le=1.0)\n"
                "\n"
                "Sentiment.model_json_schema()"
            ),
            new_markdown_cell(
                "## 2. Single happy-path call against `FakeProvider`\n"
                "\n"
                "`FakeProvider(script=[...])` returns each JSON string from the script in order. "
                "We pass `task_id` so the call is accounted to that task's budget."
            ),
            new_code_cell(
                "provider = FakeProvider(\n"
                "    script=['{\"label\": \"positive\", \"confidence\": 0.92}'],\n"
                "    cost_usd_micro_per_call=250,   # $0.00025 per call\n"
                ")\n"
                "llm = StructuredLLM(provider, agent_kernel=ak, model='fake-1')\n"
                "\n"
                "result = llm.generate(\n"
                "    messages=[{'role': 'user', 'content': 'Classify: I love this product.'}],\n"
                "    response_model=Sentiment,\n"
                "    task_id=task.task_id,\n"
                ")\n"
                "print(result)\n"
                "print('type:', type(result).__name__)"
            ),
            new_markdown_cell(
                "## 3. Retry-on-validation-error\n"
                "\n"
                "Script an *invalid* response first, then a valid one. The adapter will see the "
                "first response fail Pydantic validation, append the validator's error to the "
                "conversation, ask the provider again, and succeed on attempt 2. The "
                "`llm.call.completed` event records `attempts=2`."
            ),
            new_code_cell(
                "retry_provider = FakeProvider(\n"
                "    script=[\n"
                "        '{\"label\": \"positive\", \"confidence\": 1.5}',   # invalid: confidence > 1\n"
                "        '{\"label\": \"positive\", \"confidence\": 0.7}',   # valid\n"
                "    ],\n"
                "    cost_usd_micro_per_call=100,\n"
                ")\n"
                "retry_llm = StructuredLLM(retry_provider, agent_kernel=ak, model='fake-1', max_retries=2)\n"
                "\n"
                "out = retry_llm.generate(\n"
                "    messages=[{'role': 'user', 'content': 'Classify: this is fine.'}],\n"
                "    response_model=Sentiment,\n"
                "    task_id=task.task_id,\n"
                ")\n"
                "print(out)"
            ),
            new_markdown_cell(
                "## 4. Look at the ledger entries\n"
                "\n"
                "Per call, you should see one `llm.call.started`, one `llm.call.completed`, "
                "and one `budget.debited`. The completed event carries usage and the "
                "budget_before/after pair, so you can audit cost accounting offline by "
                "replaying JSONL alone."
            ),
            new_code_cell(
                "events = ak.list_events(task.task_id)\n"
                "for e in events:\n"
                "    if e.event_type in (EventType.llm_call_started, EventType.llm_call_completed, EventType.budget_debited):\n"
                "        print(f'{e.event_type.value:25s}  {json.dumps(e.payload, sort_keys=True)[:160]}')"
            ),
            new_markdown_cell(
                "## 5. Out-of-retries failure\n"
                "\n"
                "Three invalid responses in a row with `max_retries=2` (i.e. 3 attempts) will "
                "raise `LLMCallError`. The cost incurred is still debited; a `llm.call.completed` "
                "event is emitted with `status=error`."
            ),
            new_code_cell(
                "always_bad = FakeProvider(\n"
                "    script=[\n"
                "        '{\"label\": \"bad\", \"confidence\": 99}',\n"
                "        '{\"label\": \"bad\", \"confidence\": 99}',\n"
                "        '{\"label\": \"bad\", \"confidence\": 99}',\n"
                "    ],\n"
                "    cost_usd_micro_per_call=50,\n"
                ")\n"
                "bad_llm = StructuredLLM(always_bad, agent_kernel=ak, model='fake-1', max_retries=2)\n"
                "\n"
                "try:\n"
                "    bad_llm.generate(\n"
                "        messages=[{'role': 'user', 'content': '...'}],\n"
                "        response_model=Sentiment,\n"
                "        task_id=task.task_id,\n"
                "    )\n"
                "except LLMCallError as exc:\n"
                "    print('expected failure:', str(exc)[:200])"
            ),
            new_markdown_cell(
                "## Pointer to local LM Studio (auto model selection)\n"
                "\n"
                "Swap the provider line to talk to a real local model. Note that you do **not** "
                "have to name the model — `LMStudioProvider` queries `/v1/models` and uses "
                "whatever LM Studio currently has loaded, which is what you want for quickstart "
                "and ad-hoc testing.\n"
                "\n"
                "```python\n"
                "from agent_kernel.llm import LMStudioProvider\n"
                "\n"
                "provider = LMStudioProvider()                  # default base_url, auto-detect model\n"
                "if provider.is_reachable():\n"
                "    print('LM Studio models loaded:', provider.list_models())\n"
                "    print('will use:               ', provider.resolve_model())\n"
                "    llm = StructuredLLM(provider, agent_kernel=ak)   # model=None — provider picks\n"
                "    # … same generate() call …\n"
                "```\n"
                "\n"
                "Under the hood `LMStudioProvider` routes the call through LiteLLM's built-in "
                "`lm_studio/<model>` provider (it sets `LM_STUDIO_API_BASE` / `LM_STUDIO_API_KEY` "
                "and lets LiteLLM handle the transport, retry, and exception normalization).\n"
                "\n"
                "`is_reachable()` lets you skip cleanly when LM Studio isn't running, which is "
                "exactly what the optional `tests/integration/test_m7_llm.py::test_lmstudio_*` "
                "test does."
            ),
            new_code_cell("shutil.rmtree(workspace, ignore_errors=True)"),
        ]
    )


def main() -> None:
    notebooks = {
        "01_hello_runner.ipynb": build_01(),
        "02_python_api_tasks.ipynb": build_02(),
        "03_spawn_lineage.ipynb": build_03(),
        "04_control_kernel_magics.ipynb": build_04(),
        "05_llm_fake_provider.ipynb": build_05(),
    }
    for name, nb in notebooks.items():
        out = HERE / name
        nbformat.write(nb, out)
        print("wrote", out)


if __name__ == "__main__":
    main()
