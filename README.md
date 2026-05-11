# agent-kernel

A Python-first, server-side Jupyter kernel and companion Jupyter Server extension that provides notebook-native agent authoring, orchestration, and provenance on top of standard JupyterLab and Jupyter Server.

> **Status:** Pre-alpha. Under active milestone-driven development. See [`CODING_AGENT.md`](./CODING_AGENT.md) and [`agent-client-spec.md`](./agent-client-spec.md).

> **Note on repository name:** This repository is named `wiki3-ai/agent-client` for historical reasons; the Python package is named `agent_kernel` per the design spec.

## Architecture

Three-part system:

1. **Control kernel** (`agent_kernel`) — `ipykernel.kernelbase.Kernel` subclass providing notebook-native authoring, `%agent` magics, and orchestration UX.
2. **Server extension** (`agent_kernel_server`) — Jupyter Server `ExtensionApp` providing the scheduler, task registry, provenance emission, and REST hooks.
3. **Notebook runner** — `nbformat` + `nbclient` driving child notebook execution against any installed Jupyter kernel.

LLM integration is **LiteLLM + Instructor + Pydantic**. Provenance is **append-only JSONL + atomic JSON state**.

## Install (development)

Requires **Python 3.13**.

```bash
pip install -e ".[dev,server,llm]"
```

## Run tests

```bash
pytest                            # unit + integration (no LLM, no slow)
pytest -m integration             # integration only
pytest -m "integration and llm"   # requires LM Studio at http://localhost:1234/v1
```

## Project structure

See [`agent-client-spec.md`](./agent-client-spec.md) for the full design spec.
