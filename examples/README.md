# Examples

Runnable notebooks demonstrating the current `agent_kernel` MVP.
Start with [`docs/getting-started.md`](../docs/getting-started.md) for
the narrative walkthrough; this directory is the hands-on companion.

| # | Notebook | Kernel | What it shows |
| --- | --- | --- | --- |
| 01 | [`01_hello_runner.ipynb`](01_hello_runner.ipynb) | `python3` | Execute a notebook with `NotebookRunner`; inspect the JSONL trace it produced |
| 02 | [`02_python_api_tasks.ipynb`](02_python_api_tasks.ipynb) | `python3` | Use `AgentKernel` to create / run / inspect tasks; see budget + refund |
| 03 | [`03_spawn_lineage.ipynb`](03_spawn_lineage.ipynb) | `python3` | Spawn a child task from the `python-analysis` template; lineage chain in the ledger |
| 04 | [`04_control_kernel_magics.ipynb`](04_control_kernel_magics.ipynb) | `agent-kernel` | Drive the system from a notebook running on the agent-kernel itself, using `%agent` magics |
| 05 | [`05_llm_fake_provider.ipynb`](05_llm_fake_provider.ipynb) | `python3` | `StructuredLLM` + `FakeProvider`: validated Pydantic responses, retry-on-validation, budget debits in the ledger |

## Prereqs

```bash
pip install -e ".[dev,llm]"
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

Notebook 04 additionally needs the agent-kernel kernelspec:

```bash
python -m agent_kernel install --user
```

## Running them

Notebooks 01, 02, 03, and 05 use only the standard `python3` kernel.
Each one provisions a temp workspace, demonstrates one capability,
and cleans up.

```bash
jupyter lab examples/
# or, headless:
python -m nbclient.cli --kernel python3 --timeout 60 examples/01_hello_runner.ipynb
```

## Rebuilding the notebooks

These `.ipynb` files are generated from in-line cell sources in
`_build_notebooks.py` so they stay in lockstep with the API. If you
change the build script, regenerate:

```bash
python examples/_build_notebooks.py
```

The notebooks are committed (so they render on GitHub and `jupyter lab`
can open them directly), and the build script is the source of truth.
