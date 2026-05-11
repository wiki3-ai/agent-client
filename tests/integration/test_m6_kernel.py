"""M6 integration gate: control kernel + ``%agent`` magics in a real Jupyter env.

This test installs the agent-kernel kernelspec into a temporary
``JUPYTER_DATA_DIR``, then drives a fixture notebook through ``nbclient``
**using the agent-kernel itself as the executing kernel**, asserting that
magic invocations produce the expected task records, child spawns, and
JSONL events.

This is the first test that proves the kernel works in a real Jupyter
environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_notebook

from agent_kernel.api import AgentKernel
from agent_kernel.install import install_kernelspec
from agent_kernel.magics import dispatch
from agent_kernel.models.event import EventType
from agent_kernel.models.task import TaskStatus
from agent_kernel.storage import WorkspaceLayout

# ----------------------------- magic dispatcher ----------------------------


@pytest.mark.integration
def test_magic_help_and_policy_and_quota(tmp_path: Path) -> None:
    ak = AgentKernel(tmp_path)
    h = json.loads(dispatch("%agent help", ak))
    assert h["ok"] and "task" in h["magics"]
    p = json.loads(dispatch("%agent policy show", ak))
    assert p["ok"] and p["profile"]["name"] == "local-dev"
    q = json.loads(dispatch("%agent quota", ak))
    assert q["ok"] and q["quota"]["kernel_slots_total"] >= 1


@pytest.mark.integration
def test_magic_task_new_and_status_and_list(tmp_path: Path) -> None:
    ak = AgentKernel(tmp_path)
    (tmp_path / "x.ipynb").write_text("{}")
    out = json.loads(dispatch("%agent task new x.ipynb --kernel python3", ak))
    assert out["ok"]
    tid = out["task_id"]
    s = json.loads(dispatch(f"%agent task status {tid}", ak))
    assert s["ok"] and s["task"]["task_id"] == tid
    ls = json.loads(dispatch("%agent task list", ak))
    assert tid in ls["task_ids"]


@pytest.mark.integration
def test_magic_spawn_and_ledger_tail(tmp_path: Path) -> None:
    ws = WorkspaceLayout(tmp_path)
    ws.ensure()
    ak = AgentKernel(tmp_path)
    parent = ak.create_task(notebook_path=str(tmp_path / "parent.ipynb"), kernel_name="python3")
    out = json.loads(
        dispatch(
            f'%agent spawn {parent.task_id} python-analysis --param query="hello" --param limit=5',
            ak,
        )
    )
    assert out["ok"], out
    assert out["child_task_id"]

    tail = json.loads(dispatch("%agent ledger tail 5", ak))
    assert tail["ok"]
    assert tail["events"]
    # ledger tail with --task filter
    tail_p = json.loads(dispatch(f"%agent ledger tail 100 --task {parent.task_id}", ak))
    assert all(e["task_id"] == parent.task_id for e in tail_p["events"])


@pytest.mark.integration
def test_magic_unknown_raises(tmp_path: Path) -> None:
    from agent_kernel.magics import MagicError

    ak = AgentKernel(tmp_path)
    with pytest.raises(MagicError):
        dispatch("%agent nonexistent", ak)


# --------------------------- kernelspec installation -----------------------


@pytest.mark.integration
def test_kernelspec_install_into_temp_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(tmp_path / "jupyter_data"))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path / "jupyter_data"))
    dest = install_kernelspec(name="agent-kernel-test", user=True)
    assert dest.exists()
    assert (dest / "kernel.json").exists()
    spec = json.loads((dest / "kernel.json").read_text())
    assert "agent_kernel.kernel" in spec["argv"][2]
    assert spec["language"] == "python"

    # Visible to KernelSpecManager
    ksm = KernelSpecManager()
    assert "agent-kernel-test" in ksm.find_kernel_specs()


# ----------------------- control kernel in real Jupyter -------------------


@pytest.mark.integration
@pytest.mark.slow
def test_agent_kernel_drives_a_notebook_in_real_jupyter_env(tmp_path: Path, monkeypatch) -> None:
    """The M6 gate: install the kernelspec into a temp JUPYTER_DATA_DIR,
    drive a fixture notebook through nbclient using ``agent-kernel`` as the
    executing kernel, and assert that the magics produced the expected
    task records and provenance events.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Isolate the kernelspec install path.
    jupyter_data = tmp_path / "jupyter_data"
    jupyter_data.mkdir()
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(jupyter_data))
    monkeypatch.setenv("JUPYTER_PATH", str(jupyter_data))
    # The kernel process picks up its workspace from this env var.
    monkeypatch.setenv("AGENT_KERNEL_WORKSPACE", str(workspace))

    install_kernelspec(name="agent-kernel", user=True)

    # Confirm KernelSpecManager finds it.
    assert "agent-kernel" in KernelSpecManager().find_kernel_specs()

    # Build a notebook whose cells drive the agent surface entirely via magics
    # plus regular Python (which the kernel still supports).
    target_nb = workspace / "tasks" / "first.ipynb"
    target_nb.parent.mkdir(parents=True)
    target_payload = new_notebook()
    target_payload.cells = [new_code_cell("print('hello from target')", id="only")]
    target_payload.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    target_payload.metadata["language_info"] = {"name": "python"}
    nbformat.write(target_payload, target_nb)

    driver = new_notebook()
    driver.metadata["kernelspec"] = {"name": "agent-kernel", "display_name": "agent-kernel"}
    driver.metadata["language_info"] = {"name": "python"}
    driver.cells = [
        new_code_cell("%agent policy show", id="c_policy"),
        new_code_cell(f"%agent task new {target_nb!s} --kernel python3", id="c_new"),
        new_code_cell(
            # Use direct API access (the kernel is still a Python kernel) to
            # extract the latest task id, run it, and report status. Magics
            # write to stream; they don't set _. This is the supported pattern.
            "import json, os\n"
            "from agent_kernel.api import AgentKernel\n"
            "ak = AgentKernel(os.environ['AGENT_KERNEL_WORKSPACE'])\n"
            "tid = ak.scheduler.tasks_store.list_ids()[-1]\n"
            "final = ak.run_task(tid)\n"
            "print('RUN', json.dumps({'task_id': final.task_id, 'status': final.status.value}))\n",
            id="c_run",
        ),
        new_code_cell("%agent ledger tail 5", id="c_tail"),
    ]
    driver_path = workspace / "driver.ipynb"
    nbformat.write(driver, driver_path)

    # Execute via nbclient using the agent-kernel kernel.
    nb = nbformat.read(driver_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=60,
        kernel_name="agent-kernel",
        allow_errors=False,
        resources={"metadata": {"path": str(workspace)}},
    )
    # Make sure the subprocess kernel can find this checkout's package.
    # We add the repo root to PYTHONPATH for the kernel launch.
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    # nbclient inherits env from the current process by default; we set it
    # via the test process env so the spawned kernel sees it.
    for k, v in env.items():
        os.environ[k] = v

    client.execute()

    executed = workspace / "driver-executed.ipynb"
    nbformat.write(nb, executed)

    # The policy cell printed JSON containing the profile.
    policy_text = _stream(nb.cells[0])
    assert json.loads(policy_text.strip())["profile"]["name"] == "local-dev"

    # The new-task cell returned a task_id.
    new_text = _stream(nb.cells[1])
    new_payload = json.loads(new_text.strip())
    assert new_payload["ok"] and new_payload["task_id"]
    tid = new_payload["task_id"]

    # The run cell (regular Python, calling the API directly) reports completion.
    run_text = _stream(nb.cells[2])
    assert "RUN" in run_text
    assert tid in run_text
    assert '"status": "completed"' in run_text

    # The ledger tail cell shows events for this task.
    tail_text = _stream(nb.cells[3])
    tail_payload = json.loads(tail_text.strip())
    assert tail_payload["ok"]
    types_in_tail = {e["type"] for e in tail_payload["events"]}
    assert "task.completed" in types_in_tail or "notebook.execution.completed" in types_in_tail

    # And the workspace JSONL has the expected end-state for the task.
    ak_outer = AgentKernel(workspace)
    persisted = ak_outer.get_task(tid)
    assert persisted is not None
    assert persisted.status == TaskStatus.completed
    types = {e.event_type for e in ak_outer.list_events(tid)}
    assert EventType.task_completed in types
    assert EventType.notebook_execution_started in types


def _stream(cell: nbformat.NotebookNode) -> str:
    return "".join(
        o.get("text", "") for o in (cell.get("outputs") or []) if o.get("output_type") == "stream"
    )


# ------------------------- CLI install subcommand -------------------------


@pytest.mark.integration
def test_cli_install_subcommand(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("JUPYTER_DATA_DIR", str(tmp_path / "jd"))
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path / "jd"))
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_kernel",
            "install",
            "--user",
            "--name",
            "agent-kernel-cli-test",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert r.returncode == 0, r.stderr
    assert "Installed kernelspec" in r.stdout
