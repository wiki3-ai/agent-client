#!/usr/bin/env bash
# Provision the dev environment for agent-kernel.
set -euo pipefail

echo ">>> Upgrading pip toolchain"
python -m pip install --upgrade pip wheel setuptools

echo ">>> Installing agent-kernel in editable mode with dev/server/llm extras"
pip install -e ".[dev,server,llm]"

echo ">>> Installing JupyterLab for interactive use"
pip install "jupyterlab>=4.2"

echo ">>> Registering the agent-kernel kernelspec (if install module is available)"
if python -c "import agent_kernel.install" 2>/dev/null; then
    python -m agent_kernel.install --user || true
fi

echo ">>> Enabling agent_kernel_server extension"
jupyter server extension enable agent_kernel_server --user || true

echo ">>> Done. Try: 'jupyter lab --ip=0.0.0.0 --no-browser' or 'pytest'"
