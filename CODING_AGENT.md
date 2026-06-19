# CODING_AGENT.md

## Purpose

This repository implements a Python-first JupyterLab server-side Agent Kernel.
The Python package is named `agent_kernel`; the sibling Jupyter Server extension
is `agent_kernel_server`. The repo itself is named `wiki3-ai/agent-client` for
historical reasons.

The first production target is:
- JupyterLab
- Jupyter Server
- filesystem-backed workspace state
- notebook execution through nbclient
- typed schemas via Pydantic
- structured LLM calls through LiteLLM + Instructor, tested locally against
  **LM Studio** (OpenAI-compatible endpoint, default `http://localhost:1234/v1`)

This repo does **not** require a frontend for MVP.

## Architectural rules

1. Keep policy/budget/quota logic pure and typed.
2. Keep notebook execution in `runtime/notebook_runner.py`.
3. Never let provider-specific code leak into task models.
4. All durable data structures must be Pydantic models with `extra="forbid"`.
5. All provenance must be append-only JSONL.
6. Child task spawn must reserve parent budget before materialization.
7. Notebook paths are first-class identifiers; do not invent hidden surrogate path schemes.
8. Support any installed Jupyter kernel for execution; executable parameter
   injection is plugin-based by kernel type.

## Testing rules

- **Unit tests are for productivity, not deliverable status.**
- **Integration tests are the milestone gate.** A milestone is not complete
  until an end-to-end integration test exercises the new capability through
  its public surface against real `nbclient`, real filesystem, and (where
  applicable) a real Jupyter Server.
- Live LLM tests use LM Studio and are marked `@pytest.mark.llm`; they are
  skipped automatically when the endpoint is unreachable.

## PR checklist

- [ ] Does this change preserve append-only provenance?
- [ ] Are all new durable structures modeled with Pydantic (`extra="forbid"`)?
- [ ] Are notebook paths and cell IDs preserved?
- [ ] Is there at least one integration test exercising the new behavior?
- [ ] If policy logic changed, were Hypothesis property tests updated?
- [ ] Were docs/schema snapshots updated?
- [ ] Did you avoid adding frontend dependencies?
- [ ] Did you avoid leaking provider-specific data into task models?

## Non-goals for MVP

- No custom JupyterLab frontend
- No browser/WASM execution path
- No DB-only storage requirement
- No MCP abstraction for notebooks themselves
- No full ACL2 differential suite (deferred; Hypothesis property tests substitute)
