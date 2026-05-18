# Prompt for Coding Agent: Build the Notebook-Native Budget-Aware Graph Agent

You are building a Python project that implements a notebook-native coding/task agent. The agent follows the processing strategy **Retrieve → Compose → Transform → Generate**, recursively at every task and subtask level. The execution platform and unit of reproducibility is a **Jupyter notebook executed with Papermill**. The system persists every task and subtask as a hierarchical filesystem-backed graph of notebooks, manifests, logs, inputs, outputs, and artifacts.

The project should be implemented incrementally, with each milestone delivering runnable software and passing both unit tests and live integration tests. Do not only mock behavior. Each module should have at least one live acceptance test that exercises it end-to-end using real files, real notebooks, and real Papermill execution.

## 1. Core goals

Build an agent framework that can:

1. Accept a task request.
2. Create a filesystem-backed task directory.
3. Search/retrieve existing skills, notebooks, docs, or prior runs.
4. Compose existing artifacts when possible.
5. Transform a skill `.md` file into a parameterized Jupyter notebook when needed.
6. Generate new notebook code only when retrieval, composition, or transformation is insufficient.
7. Execute notebooks using Papermill.
8. Track budget usage and enforce budget constraints.
9. Create child task notebooks for subtasks.
10. Persist a traceable parent/child task graph.
11. Run tests and repair failed notebooks within budget.
12. Expose the agent as an MCP service for use by other agents.
13. Prepare the design for future DSPy and GEPA-based self-improvement.

The first implementation should prioritize correctness, reproducibility, and inspectability over sophistication.

## 2. Architectural principle

The central policy is:

```text
Retrieve → Compose → Transform → Generate
```

This applies recursively. Even when code generation appears necessary, the code-generation subtask should itself begin by retrieving relevant skills, examples, tests, templates, and prior runs.

Generation should be treated as the last resort. The system should record why earlier stages were insufficient whenever it enters the Generate stage.

## 3. Terminology

### Task

A user-visible unit of work. Each task has:

- a task ID,
- a filesystem directory,
- input parameters,
- budget,
- status,
- notebook artifacts,
- logs,
- manifest,
- optional child tasks.

### Subtask

A child task linked to a parent task. A subtask has the same structure as a task and is stored in the parent task’s `children/` directory.

### Skill

A reusable capability stored under `skills/`. A skill may start as a `SKILL.md` document and may also have a corresponding `skill.ipynb` notebook.

### Task notebook

A generated or transformed notebook that performs a task. It must be parameterized for Papermill.

### Executed notebook

The output notebook produced by Papermill after execution.

### Manifest

A machine-readable JSON file describing what happened during task execution.

## 4. Required processing stages

### 4.1 Retrieve

Search for existing artifacts before generating anything.

Retrieve candidates from:

- `skills/`,
- `runs/`,
- `examples/`,
- `docs/`,
- test fixtures,
- prior task manifests,
- optional web search module.

Initial implementation can use lexical search and metadata scoring. Embeddings can be added later.

### 4.2 Compose

Combine retrieved artifacts when possible.

Examples:

- chain existing skill notebooks,
- reuse a prior successful task notebook,
- compose a TODO list from existing skill steps,
- call existing notebook executor skill,
- link child notebooks to a parent workflow.

### 4.3 Transform

Adapt an existing artifact.

Examples:

- convert `SKILL.md` to a parameterized notebook,
- add Papermill parameters,
- add tests,
- add manifest emission,
- adapt a notebook to current task parameters,
- modify kernel metadata.

### 4.4 Generate

Generate only what is missing.

Examples:

- missing notebook cells,
- missing tests,
- missing DSPy signatures,
- missing MCP glue,
- missing repair patch.

Every Generate decision must be recorded in the task manifest with a reason.

## 5. Budget-aware execution

Implement an explicit budget model.

A budget should support at least:

```json
{
  "max_wall_time_seconds": 600,
  "max_lm_calls": 20,
  "max_lm_tokens_input": 100000,
  "max_lm_tokens_output": 20000,
  "max_cost_usd": 1.50,
  "max_notebook_executions": 8,
  "max_repair_attempts": 2,
  "max_subtasks": 10,
  "max_parallel_subtasks": 3,
  "max_web_searches": 5,
  "max_generated_code_tokens": 8000
}
```

The first implementation does not need perfect cost accounting, but it must track:

- wall time,
- number of notebook executions,
- number of subtasks,
- number of repair attempts,
- number of web searches if implemented,
- number of LM calls if LiteLLM integration is used.

The budget manager must expose:

```python
can_spend(resource: str, amount: int | float = 1) -> bool
spend(resource: str, amount: int | float = 1) -> None
remaining(resource: str) -> int | float | None
snapshot() -> dict
```

If a task cannot continue because of budget exhaustion, it must:

1. stop gracefully,
2. write the manifest,
3. record the exhaustion reason,
4. preserve partial outputs,
5. update status to `budget_exhausted`.

## 6. Prioritizing under constraints

Implement a simple action selection mechanism.

Candidate action schema:

```json
{
  "action_id": "retrieve-local-skills",
  "stage": "retrieve",
  "expected_value": 0.75,
  "expected_cost": {
    "time_seconds": 10,
    "lm_calls": 0,
    "web_searches": 0,
    "notebook_executions": 0
  },
  "risk": 0.1,
  "confidence": 0.8,
  "dependencies": []
}
```

Initial utility can be simple:

```text
utility = expected_value / max(expected_cost_total, epsilon)
```

Bias action ordering by stage preference:

```text
Retrieve > Compose > Transform > Generate
```

The planner must prefer cheap retrieval before expensive generation.

## 7. Filesystem-backed task graph

The filesystem is the source of truth for task persistence.

Use this hierarchy:

```text
runs/
  YYYY/
    MM/
      DD/
        HHMMSS-task-slug/
          task.json
          manifest.json
          README.md
          task.ipynb
          executed.ipynb
          inputs/
            request.md
            parameters.json
          outputs/
            result.json
            answer.md
          logs/
            events.jsonl
            stdout.log
            stderr.log
            lm_calls.jsonl
          artifacts/
          children/
            001-child-task-slug/
              task.json
              manifest.json
              README.md
              task.ipynb
              executed.ipynb
              inputs/
              outputs/
              logs/
              artifacts/
              children/
```

Top-level task directory names:

```text
HHMMSS-task-slug
```

Child task directory names:

```text
001-child-task-slug
002-child-task-slug
003-child-task-slug
```

Use stable IDs inside JSON. Do not rely on paths alone.

## 8. Required task files

Every task directory must contain these files after creation:

```text
task.json
manifest.json
README.md
inputs/request.md
inputs/parameters.json
logs/events.jsonl
outputs/
artifacts/
children/
```

After notebook execution, it should also contain:

```text
task.ipynb
executed.ipynb
outputs/result.json
```

### 8.1 task.json

Represents intent.

```json
{
  "task_id": "task_...",
  "parent_task_id": null,
  "title": "Build MCP notebook agent",
  "slug": "build-mcp-notebook-agent",
  "request": "...",
  "status": "pending",
  "budget": {},
  "created_at": "2026-05-18T14:30:12-07:00"
}
```

### 8.2 manifest.json

Represents what actually happened.

```json
{
  "task_id": "task_...",
  "parent_task_id": null,
  "status": "success",
  "stage_used": "transform",
  "started_at": "...",
  "finished_at": "...",
  "budget_initial": {},
  "budget_used": {},
  "budget_remaining": {},
  "children": [],
  "outputs": {},
  "tests": {},
  "stage_decision": {
    "chosen": "transform",
    "retrieve": {
      "attempted": true,
      "result": "found matching skill"
    },
    "compose": {
      "attempted": true,
      "result": "no compatible existing notebook"
    },
    "transform": {
      "attempted": true,
      "result": "converted skill markdown to notebook"
    },
    "generate": {
      "attempted": false,
      "result": null
    }
  }
}
```

### 8.3 README.md

Human-readable summary.

It should include:

- title,
- status,
- parent task,
- child tasks,
- budget summary,
- important outputs,
- reproduction command.

Example reproduction command:

```bash
papermill task.ipynb executed.ipynb -f inputs/parameters.yaml
```

## 9. Event log

Each task must maintain an append-only JSONL event log:

```text
logs/events.jsonl
```

Example events:

```json
{"ts":"...","event":"task_created","task_id":"..."}
{"ts":"...","event":"budget_allocated","budget":{"max_notebook_executions":3}}
{"ts":"...","event":"retrieval_started","query":"execute notebook papermill"}
{"ts":"...","event":"artifact_retrieved","path":"skills/core/execute_notebook/SKILL.md","score":0.91}
{"ts":"...","event":"notebook_execution_started","path":"task.ipynb"}
{"ts":"...","event":"notebook_execution_finished","success":true}
{"ts":"...","event":"task_finished","status":"success"}
```

Events should be append-only. Do not rewrite history.

## 10. Skill repository structure

Create this initial layout:

```text
skills/
  core/
    search_local/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    retrieve_artifact/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    todo_decompose/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    compose_subtasks/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    execute_notebook/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    inspect_notebook_result/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    repair_notebook/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    persist_task_graph/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
    summarize_result/
      SKILL.md
      skill.ipynb
      manifest.json
      tests/
```

The absolute required bootstrap skills are:

```text
search_local
retrieve_artifact
todo_decompose
execute_notebook
repair_notebook
persist_task_graph
summarize_result
```

## 11. Skill manifest format

Each skill must include `manifest.json`:

```json
{
  "skill_id": "core.execute_notebook",
  "name": "Execute Notebook",
  "version": "0.1.0",
  "entrypoint": "skill.ipynb",
  "description": "Execute a parameterized Jupyter notebook using Papermill.",
  "input_schema": "input.schema.json",
  "output_schema": "output.schema.json",
  "tags": ["core", "papermill", "execution"]
}
```

## 12. Notebook standards

Every generated or transformed notebook must include:

1. A Papermill `parameters` cell.
2. A setup/import cell.
3. An input validation cell.
4. One or more execution cells.
5. A result writing cell.
6. A manifest update cell.
7. A final lightweight smoke assertion cell when appropriate.

The parameters cell should include at least:

```python
# parameters
task_id = None
parent_task_id = None
input_payload = {}
output_dir = "./outputs"
run_dir = "."
budget = {}
```

The notebook must write structured output to:

```text
outputs/result.json
```

## 13. Python package layout

Use a clean package structure similar to:

```text
notebook_agent/
  __init__.py
  budget.py
  task_graph.py
  events.py
  notebook_exec.py
  skills.py
  planner.py
  transform.py
  repair.py
  agent.py
  mcp_server.py
  litellm_client.py
  dspy_modules.py

tests/
  unit/
  integration/
  live/

examples/
  simple_echo_skill/
  failing_notebook_repair/
```

## 14. Required modules

### 14.1 `budget.py`

Implement:

- `Budget` dataclass or Pydantic model,
- `BudgetTracker`,
- budget snapshotting,
- budget exhaustion errors.

### 14.2 `task_graph.py`

Implement:

- create root task,
- create child task,
- write task JSON,
- write manifest JSON,
- update task status,
- list children,
- load task graph from filesystem.

### 14.3 `events.py`

Implement append-only JSONL event logging.

### 14.4 `notebook_exec.py`

Implement Papermill execution wrapper.

Required behavior:

- execute notebook with parameters,
- capture success/failure,
- write stdout/stderr logs if available,
- update budget usage,
- record events,
- return structured result.

### 14.5 `skills.py`

Implement local skill discovery and retrieval.

Start simple:

- scan `skills/`,
- read `manifest.json`,
- read `SKILL.md`,
- lexical query match,
- rank results.

### 14.6 `planner.py`

Implement simple stage decision logic.

Required behavior:

- prefer retrieve over compose over transform over generate,
- check budget before selecting action,
- record stage decision.

### 14.7 `transform.py`

Implement skill markdown to notebook conversion.

First version can be template-based. It does not need a powerful parser.

### 14.8 `repair.py`

Implement notebook repair loop.

First version may handle simple Python failures:

- missing import,
- undefined variable from parameter mismatch,
- syntax error from generated code,
- missing output directory.

It should use LiteLLM when available, but also support deterministic repair strategies for simple known failures.

### 14.9 `agent.py`

Implement the orchestration API:

```python
run_task(request: str, parameters: dict | None = None, budget: dict | None = None) -> dict
```

It should create a root task, execute the Retrieve → Compose → Transform → Generate policy, and return the final manifest/result.

### 14.10 `mcp_server.py`

Expose an MCP service with tools:

```text
run_task
run_skill
search_skills
read_manifest
get_task_graph
execute_notebook
```

MCP can be a later milestone, but leave interfaces clean enough that it can be added without redesign.

### 14.11 `litellm_client.py`

Implement LiteLLM client wrapper.

Default environment:

```text
LM Studio accessible at http://host.docker.internal:1234/v1
```

Make provider/model configurable through environment variables.

Suggested environment variables:

```bash
NOTEBOOK_AGENT_MODEL=lm_studio/model-name
NOTEBOOK_AGENT_BASE_URL=http://host.docker.internal:1234/v1
NOTEBOOK_AGENT_API_KEY=lm-studio
```

The system must run without LM Studio for non-generation milestones.

### 14.12 `dspy_modules.py`

Stub DSPy modules for future integration:

- TaskRouter,
- SkillRetriever,
- SkillToNotebookTransformer,
- NotebookRepairer,
- ResultSynthesizer.

Do not make DSPy mandatory for milestone 1.

## 15. Testing requirements

Use layered testing:

```text
unit tests
integration tests
live acceptance tests
```

Unit tests are necessary but not sufficient. Every milestone must have a live acceptance test that exercises real filesystem operations and, where applicable, real Papermill notebook execution.

### 15.1 Test tooling

Use:

- `pytest`,
- temporary directories via `tmp_path`,
- real `.ipynb` files generated during tests,
- Papermill for live notebook execution,
- optional `pytest.mark.live` for tests that require LM Studio or network.

### 15.2 Test categories

```text
tests/unit/
  Fast deterministic tests.

tests/integration/
  Filesystem and Papermill tests that do not require external services.

tests/live/
  Tests requiring LM Studio, web access, MCP clients, or other real services.
```

## 16. Milestones and acceptance tests

### Milestone 1: Filesystem task graph

Build:

- task directory creation,
- child task creation,
- task JSON writing,
- manifest writing,
- event logging,
- README writing.

Acceptance test:

```text
Create a root task with one child task in a temporary runs directory.
Verify the exact expected hierarchy exists.
Verify task.json and manifest.json are valid JSON.
Verify child task references parent task ID.
Verify parent manifest lists child task.
Verify events.jsonl contains task_created events.
Reload the graph from disk and verify structure.
```

This must be a real filesystem test, not mocked.

### Milestone 2: Papermill notebook execution

Build:

- notebook execution wrapper,
- parameter passing,
- output notebook creation,
- result JSON detection,
- failure capture.

Acceptance test:

```text
Create a real parameterized notebook that accepts name="World".
Notebook writes outputs/result.json containing {"message":"Hello, World"}.
Execute it with Papermill through the project wrapper.
Verify executed.ipynb exists.
Verify outputs/result.json exists.
Verify result JSON content is correct.
Verify manifest status is success.
Verify budget notebook execution count increments.
```

Also test a failing notebook:

```text
Create a notebook that raises ValueError("intentional failure").
Execute it.
Verify wrapper returns success=false.
Verify error summary is captured.
Verify manifest status is failed.
Verify event log records notebook_execution_failed.
```

### Milestone 3: Local skill retrieval

Build:

- skill manifest loading,
- `SKILL.md` loading,
- lexical search,
- scoring,
- result ranking.

Acceptance test:

```text
Create three fake skills in a temporary skills directory:
- execute_notebook
- search_local
- summarize_result
Search for "run papermill notebook".
Verify execute_notebook is ranked first.
Verify returned result includes manifest metadata and SKILL.md excerpt.
```

### Milestone 4: TODO decomposition and subtask composition

Build:

- simple sequential TODO list generation,
- child task creation from TODO items,
- parent/child linking.

Acceptance test:

```text
Given a task request "search for a skill and execute a notebook",
produce a sequential TODO list with at least search and execute steps.
Create child task directories for each TODO item.
Verify child directories are named with numeric prefixes.
Verify parent manifest lists children in order.
```

### Milestone 5: Skill markdown to notebook transform

Build:

- convert simple `SKILL.md` into a notebook,
- add parameters cell,
- add output writing cell,
- add manifest update cell.

Acceptance test:

```text
Create a simple SKILL.md describing an echo skill.
Transform it into task.ipynb.
Verify the notebook has a parameters cell tagged for Papermill.
Execute it using the notebook executor.
Verify outputs/result.json contains expected echo output.
```

### Milestone 6: Budget manager

Build:

- budget tracker,
- spend/check APIs,
- wall-time checks,
- hard stop behavior.

Acceptance test:

```text
Run a task with max_notebook_executions=1.
Execute one notebook successfully.
Attempt a second notebook execution.
Verify execution is refused before Papermill runs.
Verify task status becomes budget_exhausted or the child action is rejected with a budget error.
Verify manifest records exhaustion reason.
```

### Milestone 7: Repair loop

Build:

- failure inspection,
- deterministic repair for at least one known class of failure,
- retry within budget,
- repair represented as child task.

Acceptance test:

```text
Create a notebook that fails because outputs/ does not exist before writing result.json.
Execute notebook and capture failure.
Run repair.
Repair should patch notebook to create output_dir before writing.
Re-execute patched notebook.
Verify success.
Verify repair child task exists.
Verify manifest records initial failure and repaired success.
```

Optional live LM repair test:

```text
If LM Studio is available, create a notebook with a simple undefined variable error.
Ask repair module to patch using LiteLLM.
Re-execute.
Verify success or produce a clear skipped test if LM Studio is unavailable.
```

### Milestone 8: End-to-end agent loop

Build:

- `run_task`,
- retrieve skill,
- transform or compose,
- execute notebook,
- summarize result.

Acceptance test:

```text
Given a task request "Use the echo skill to echo hello graph agent",
agent should:
1. create root task,
2. retrieve echo skill,
3. transform or compose notebook,
4. execute notebook,
5. write outputs/result.json,
6. write final answer.md,
7. return success manifest.

Verify full task graph exists on disk.
Verify final result is correct.
Verify event log includes retrieve, transform/compose, execute, summarize.
```

### Milestone 9: MCP service

Build:

- MCP server wrapper,
- tools for run_task, search_skills, read_manifest, get_task_graph, execute_notebook.

Acceptance test:

```text
Start the MCP service locally in a test subprocess.
Use an MCP client to call search_skills.
Use an MCP client to call run_task with a simple echo request.
Verify returned manifest and filesystem outputs.
```

If MCP test infrastructure is not yet available, provide a CLI compatibility layer first and test that live.

### Milestone 10: LiteLLM and DSPy integration stubs

Build:

- LiteLLM wrapper,
- configurable model/base URL,
- graceful unavailable behavior,
- DSPy module stubs.

Acceptance test:

```text
Without LM Studio available:
- verify non-generation workflows still pass.
- verify generation attempts fail gracefully with a clear unavailable-model error.

With LM Studio available:
- call LiteLLM wrapper against http://host.docker.internal:1234/v1 or configured URL.
- request a short deterministic completion.
- verify response is captured in logs/lm_calls.jsonl.
```

## 17. CLI requirements

Implement a CLI early. Suggested commands:

```bash
notebook-agent init
notebook-agent run "Echo hello" --budget budget.json
notebook-agent search-skills "papermill notebook"
notebook-agent execute-notebook task.ipynb --params params.json
notebook-agent graph runs/2026/05/18/143012-task
notebook-agent manifest runs/2026/05/18/143012-task
```

CLI acceptance test:

```text
Run notebook-agent init in a temporary project directory.
Run notebook-agent run with a simple echo task.
Verify command exits 0.
Verify run directory exists.
Verify manifest status is success.
Verify output answer exists.
```

## 18. Live integration testing philosophy

Do not rely only on mocks.

A module is accepted only when it has:

1. unit tests for internal logic,
2. integration tests using real files,
3. at least one live acceptance test for its intended real use.

Examples of live tests:

- Real Papermill execution of generated notebooks.
- Real skill search over a temporary skill repository.
- Real task graph persisted to disk and reloaded.
- Real CLI invocation through subprocess.
- Real MCP tool call when MCP module is implemented.
- Real LiteLLM call when LM Studio is available.

External-service live tests may be skipped when unavailable, but the skip condition must be explicit and visible.

Do not silently pass live tests by mocking the external service.

## 19. Error handling requirements

All failures should produce structured outputs.

Do not crash without writing useful state.

On failure, write:

- manifest status,
- error summary,
- traceback path if available,
- event log entry,
- partial outputs if any.

Failure statuses:

```text
pending
running
success
failed
budget_exhausted
cancelled
skipped
```

## 20. Observability requirements

Every significant operation should write an event.

At minimum:

- task_created,
- task_started,
- task_finished,
- child_task_created,
- budget_allocated,
- budget_spent,
- retrieval_started,
- retrieval_finished,
- artifact_retrieved,
- stage_decision_made,
- notebook_execution_started,
- notebook_execution_finished,
- notebook_execution_failed,
- repair_started,
- repair_finished,
- manifest_updated.

LM calls should be logged separately to:

```text
logs/lm_calls.jsonl
```

Do not log secrets or API keys.

## 21. Code quality requirements

Use:

- Python 3.11+,
- type hints,
- dataclasses or Pydantic models for structured data,
- pathlib instead of raw string paths,
- pytest,
- Ruff or equivalent linting,
- clear error classes,
- deterministic tests.

Prefer boring, reliable code.

Do not over-engineer the planner before the notebook execution and task graph are solid.

## 22. Suggested dependencies

Core:

```text
papermill
nbformat
nbclient
pydantic
pytest
typer
rich
python-slugify
```

Optional/future:

```text
litellm
dspy
mcp
fastapi
uvicorn
```

## 23. Minimum viable implementation

The first useful version should support this workflow:

```text
User request
→ create root task directory
→ search local skills
→ find echo skill
→ transform skill.md to task notebook
→ execute with Papermill
→ write result.json
→ write manifest.json
→ write README.md
→ return final answer
```

This should work without any external LLM.

## 24. Built-in echo skill for testing

Create a simple built-in skill:

```text
skills/core/echo/
  SKILL.md
  manifest.json
```

The echo skill accepts:

```json
{
  "message": "hello"
}
```

And returns:

```json
{
  "message": "hello"
}
```

This skill is required for end-to-end testing.

## 25. Acceptance definition for the project

The project is minimally acceptable when all of the following are true:

1. `pytest tests/unit tests/integration` passes.
2. A live Papermill acceptance test passes.
3. CLI can run an echo task end-to-end.
4. The run directory is human-readable and hierarchical.
5. Every task writes `task.json`, `manifest.json`, `README.md`, `logs/events.jsonl`, and `outputs/result.json` when successful.
6. Budget exhaustion is tested and handled gracefully.
7. At least one failed notebook can be repaired and re-executed successfully.
8. The agent records whether it used retrieve, compose, transform, or generate.
9. The system can run without LM Studio for non-generation workflows.
10. LM Studio integration has a live test that runs only when configured and available.

## 26. Future GEPA adapter design notes

Do not implement GEPA immediately unless earlier milestones are solid.

Prepare traces so GEPA can later optimize:

- prompts,
- stage selection logic,
- skill retrieval logic,
- notebook generation logic,
- repair prompts.

Each task graph should preserve enough information for a future GEPA adapter to build reflective datasets:

```json
{
  "inputs": {},
  "candidate_outputs": {},
  "scores": {},
  "failure_feedback": {},
  "trajectory": {}
}
```

Keep this as a future-facing design, not a blocker for MVP.

## 27. Non-goals for the first version

Do not implement these in the MVP:

- sophisticated multi-agent orchestration,
- distributed execution,
- embedding search unless trivial,
- full web browser automation,
- automatic GEPA optimization,
- complex parallel DAG scheduler,
- cloud deployment,
- advanced notebook UI.

The first version should be local, inspectable, reproducible, and testable.

## 28. Development instruction

Proceed milestone by milestone. After each milestone:

1. Run unit tests.
2. Run integration tests.
3. Run the live acceptance test for that milestone.
4. Show the resulting filesystem tree for at least one successful run.
5. Show the relevant manifest excerpt.
6. Do not continue to the next milestone until the current one has a passing acceptance test or a clearly documented blocker.

Prefer small commits and simple implementations.

The guiding question for every module is:

```text
Can a human inspect the filesystem and understand what the agent did, why it did it, what it spent, what failed, and how to reproduce it?
```

If the answer is no, improve the trace, manifest, README, or event log before adding more intelligence.

