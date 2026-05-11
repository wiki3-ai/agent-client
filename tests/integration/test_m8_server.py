"""M8 integration gate: Jupyter Server extension + REST surface.

This test launches Jupyter Server as a subprocess with the
``agent_kernel_server`` extension enabled. We then use ``httpx`` with a
valid token to:

1. Create a task (``POST /api/agent-kernel/tasks``)
2. Run it (``POST /api/agent-kernel/tasks/{id}/run``)
3. Poll it (``GET /api/agent-kernel/tasks/{id}``)
4. List its events (``GET /api/agent-kernel/tasks/{id}/events``)

And in parallel we assert that unauthenticated requests are rejected with
HTTP 401/403.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from agent_kernel.models.event import EventType


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def jupyter_server(tmp_path: Path):
    """Launch Jupyter Server in a subprocess with the extension enabled."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    port = _free_port()
    token = "test-token-secret-123"

    nb_path = workspace / "tasks" / "ok.ipynb"
    nb_path.parent.mkdir(parents=True)
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [new_code_cell("print('hi')", id="c1")]
    nbformat.write(nb, nb_path)

    env = os.environ.copy()
    env["AGENT_KERNEL_WORKSPACE"] = str(workspace)
    env["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    env["JUPYTER_DATA_DIR"] = str(tmp_path / "jdata")
    config_dir = tmp_path / "jconfig"
    config_dir.mkdir()
    env["JUPYTER_CONFIG_DIR"] = str(config_dir)
    # Write a config file that enables our extension. This is the canonical
    # way Jupyter Server discovers extensions; CLI flags are flaky here.
    cfgd = config_dir / "jupyter_server_config.d"
    cfgd.mkdir(parents=True)
    (cfgd / "agent_kernel_server.json").write_text(
        '{"ServerApp": {"jpserver_extensions": {"agent_kernel_server": true}}}'
    )
    # Make the local checkout importable from the subprocess.
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jupyter_server",
            "--no-browser",
            "--port",
            str(port),
            "--ip",
            "127.0.0.1",
            f"--ServerApp.token={token}",
            "--ServerApp.password=",
            "--ServerApp.disable_check_xsrf=True",
            f"--ServerApp.root_dir={workspace}",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"token {token}"}

    # Poll for readiness via the unauthenticated health endpoint.
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"jupyter server exited early:\n{out}")
        try:
            r = httpx.get(f"{base_url}/api/agent-kernel/health", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except (httpx.HTTPError, httpx.ConnectError, OSError):
            pass
        time.sleep(0.25)
    if not ready:
        proc.terminate()
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"jupyter server did not become ready:\n{out}")

    try:
        yield {
            "base_url": base_url,
            "headers": headers,
            "workspace": workspace,
            "nb_path": nb_path,
            "proc": proc,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.slow
def test_rest_create_run_poll_list_events(jupyter_server) -> None:
    base = jupyter_server["base_url"]
    headers = jupyter_server["headers"]
    nb_path = jupyter_server["nb_path"]

    with httpx.Client(base_url=base, headers=headers, timeout=30.0) as client:
        # 1. Create
        r = client.post(
            "/api/agent-kernel/tasks",
            json={"notebook_path": str(nb_path), "kernel_name": "python3"},
        )
        assert r.status_code == 201, r.text
        task = r.json()["task"]
        tid = task["task_id"]
        assert task["status"] == "draft"

        # 2. Run
        r = client.post(f"/api/agent-kernel/tasks/{tid}/run", timeout=60.0)
        assert r.status_code == 200, r.text
        final = r.json()["task"]
        assert final["status"] == "completed"

        # 3. Poll
        r = client.get(f"/api/agent-kernel/tasks/{tid}")
        assert r.status_code == 200
        assert r.json()["task"]["status"] == "completed"

        # 4. Events
        r = client.get(f"/api/agent-kernel/tasks/{tid}/events")
        assert r.status_code == 200
        events = r.json()["events"]
        types = {e["event_type"] for e in events}
        assert EventType.task_created.value in types
        assert EventType.task_completed.value in types
        assert EventType.notebook_execution_started.value in types

        # 5. 404 for unknown task
        r = client.get("/api/agent-kernel/tasks/task_DOES_NOT_EXIST")
        assert r.status_code == 404


@pytest.mark.integration
@pytest.mark.slow
def test_rest_unauthenticated_requests_are_rejected(jupyter_server) -> None:
    base = jupyter_server["base_url"]
    nb_path = jupyter_server["nb_path"]

    with httpx.Client(base_url=base, timeout=10.0) as client:
        # Health is intentionally open.
        r = client.get("/api/agent-kernel/health")
        assert r.status_code == 200

        # Tasks endpoints require auth.
        r = client.post(
            "/api/agent-kernel/tasks",
            json={"notebook_path": str(nb_path)},
        )
        assert r.status_code in (401, 403), (r.status_code, r.text)

        r = client.get("/api/agent-kernel/tasks/anything")
        assert r.status_code in (401, 403), r.status_code

        r = client.post("/api/agent-kernel/tasks/anything/run")
        assert r.status_code in (401, 403), r.status_code

        r = client.get("/api/agent-kernel/tasks/anything/events")
        assert r.status_code in (401, 403), r.status_code


@pytest.mark.integration
@pytest.mark.slow
def test_rest_bad_request_body(jupyter_server) -> None:
    base = jupyter_server["base_url"]
    headers = jupyter_server["headers"]
    with httpx.Client(base_url=base, headers=headers, timeout=10.0) as client:
        # Missing notebook_path
        r = client.post("/api/agent-kernel/tasks", json={})
        assert r.status_code == 400

        # Invalid JSON
        r = client.post(
            "/api/agent-kernel/tasks",
            content=b"not json",
            headers={**headers, "Content-Type": "application/json"},
        )
        assert r.status_code == 400
