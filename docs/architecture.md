# agent-kernel architecture

This document describes the layered architecture of the `agent_kernel`
package and its companion `agent_kernel_server` Jupyter Server extension.
It is intended for contributors and integrators; it is not a tutorial.

## Layers

```
                       ┌────────────────────────────────────────┐
                       │  agent_kernel_server (M8)              │
                       │   • ExtensionApp                       │
                       │   • REST handlers (authenticated)      │
                       └──────────────────┬─────────────────────┘
                                          │
   ┌──────────────────────────────────────┴───────────────────────────────┐
   │                       agent_kernel.api  (M4)                         │
   │   AgentKernel: create_task / run_task / spawn_child_task / events    │
   └──────────────┬─────────────────────────────────┬─────────────────────┘
                  │                                 │
   ┌──────────────▼───────────┐         ┌──────────▼──────────────┐
   │  scheduler  (M4)         │         │  spawn_manager (M5)     │
   │  policy_engine (M4)      │         │                         │
   └──────────────┬───────────┘         └─────────────────────────┘
                  │
   ┌──────────────▼─────────────────────────────────────────────────────┐
   │  runtime: notebook_runner (M2), materializer (M3),                 │
   │  parameter_injection (M3), template_registry (M3),                 │
   │  reconstruct (M1)                                                  │
   └──────────────┬─────────────────────────────────────────────────────┘
                  │
   ┌──────────────▼─────────────────────────────────────────────────────┐
   │  storage: JSONLEventStore (M1), AtomicJSONStore (M1),              │
   │  WorkspaceLayout                                                   │
   └────────────────────────────────────────────────────────────────────┘
   ┌────────────────────────────────────────────────────────────────────┐
   │  models: TaskSpec, Budget, ProvenanceEvent, PolicyProfile, …       │
   │  llm: StructuredLLM + providers (FakeProvider, LMStudioProvider)   │
   │  security: redaction (applied at JSONL append)                     │
   └────────────────────────────────────────────────────────────────────┘
```

## Public surfaces (in milestone order)

| Surface | Module | Milestone |
| --- | --- | --- |
| Pydantic models | `agent_kernel.models` | M1 |
| Append-only JSONL + atomic state | `agent_kernel.storage` | M1 |
| Notebook runner + CLI | `agent_kernel.runtime.notebook_runner`, `agent_kernel.cli` | M2 |
| Template + parameter injection | `agent_kernel.runtime.materializer`, `parameter_injection`, `template_registry` | M3 |
| Public Python API | `agent_kernel.api.AgentKernel` | M4 |
| Spawn manager | `agent_kernel.runtime.spawn_manager` | M5 |
| Control kernel + `%agent` magics | `agent_kernel.kernel`, `agent_kernel.magics`, `agent_kernel.install` | M6 |
| LLM adapter + providers | `agent_kernel.llm` | M7 |
| Jupyter Server REST extension | `agent_kernel_server.app`, `agent_kernel_server.handlers` | M8 |

## Key design decisions

- **Pure policy engine.** `agent_kernel.runtime.policy_engine` is the
  single source of truth for budget / quota / lineage decisions and
  contains no I/O. All adapters call into it and apply the returned
  `Decision`. This module is the natural seam where a fixture-driven
  differential test suite (ACL2-shaped or hand-built golden vectors)
  can be added post-MVP without refactor.
- **Integration tests are the milestone gate.** Each milestone closes
  only when an end-to-end integration test exercises the new
  capability through its public surface (CLI, kernel cell, or REST
  call) against real `nbclient` execution and real on-disk persistence.
- **No SQLite for MVP.** All durable state is JSONL events +
  write-temp-then-rename JSON snapshots. Reconstruction is an event
  fold (`agent_kernel.runtime.reconstruct.reconstruct_tasks`).
- **Pattern-based redaction at the JSONL boundary.** Every event passes
  through `agent_kernel.security.redact_payload` before serialization
  so OpenAI/Anthropic/GitHub/AWS/JWT-shaped secrets and sensitive field
  names (`api_key`, `password`, `token`, …) never reach the durable
  ledger. This is defense-in-depth, not a replacement for proper
  secret handling at the call site.
- **LLM provider abstraction.** `StructuredLLM` consumes a `Provider`
  Protocol; ships with `FakeProvider` (hermetic CI) and
  `LMStudioProvider` (local OpenAI-compatible HTTP). Adapter handles
  retry-on-validation, integer-precise micro-USD budget debits, and
  per-call provenance events (`llm.call.started`, `llm.call.completed`,
  `budget.debited`).

## Workspace layout

```
<workspace>/
├── notebooks/                       # materialized task notebooks
├── runs/<run_id>/
│   ├── executed.ipynb               # nbclient output
│   ├── stdout.log
│   └── stderr.log
└── .agent_kernel/
    ├── events/YYYY-MM-DD.jsonl      # provenance ledger (redacted)
    ├── tasks/<task_id>.json         # atomic state snapshots
    └── artifacts/                   # executable-cell artifacts
```

## Recovery and replay

`scripts/replay.py <workspace>` reads all JSONL files for a workspace,
reconstructs current task state via the event fold, and emits a stable
sha256 digest of the canonical (line-sorted, key-sorted) serialization.
The M9 integration test asserts bit-exact reproduction across a fresh
rebuild of the workspace by re-appending the same events into a new
`JSONLEventStore`.
