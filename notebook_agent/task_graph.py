"""Filesystem-backed task graph (Sections 7 & 8 of the spec)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from slugify import slugify

from ._clock import iso_now, make_task_id, utcnow
from .budget import Budget
from .events import EventLog

# Valid task statuses (Section 19).
VALID_STATUSES = frozenset(
    {"pending", "running", "success", "failed", "budget_exhausted", "cancelled", "skipped"}
)


def _slugify(title: str, fallback: str = "task") -> str:
    s = slugify(title, max_length=60) or fallback
    return s


@dataclass
class Task:
    """Filesystem-backed task record."""

    task_id: str
    title: str
    slug: str
    request: str
    directory: Path
    parent_task_id: str | None = None
    status: str = "pending"
    budget: Budget = field(default_factory=Budget)
    created_at: str = field(default_factory=iso_now)
    children: list[Task] = field(default_factory=list)
    stage_used: str | None = None

    # ------------ filesystem layout helpers ------------

    @property
    def task_json(self) -> Path:
        return self.directory / "task.json"

    @property
    def manifest_json(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def readme(self) -> Path:
        return self.directory / "README.md"

    @property
    def inputs_dir(self) -> Path:
        return self.directory / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.directory / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.directory / "logs"

    @property
    def artifacts_dir(self) -> Path:
        return self.directory / "artifacts"

    @property
    def children_dir(self) -> Path:
        return self.directory / "children"

    @property
    def task_notebook(self) -> Path:
        return self.directory / "task.ipynb"

    @property
    def executed_notebook(self) -> Path:
        return self.directory / "executed.ipynb"

    @property
    def request_md(self) -> Path:
        return self.inputs_dir / "request.md"

    @property
    def parameters_json(self) -> Path:
        return self.inputs_dir / "parameters.json"

    @property
    def events_log(self) -> Path:
        return self.logs_dir / "events.jsonl"

    @property
    def lm_calls_log(self) -> Path:
        return self.logs_dir / "lm_calls.jsonl"

    @property
    def result_json(self) -> Path:
        return self.outputs_dir / "result.json"

    @property
    def answer_md(self) -> Path:
        return self.outputs_dir / "answer.md"

    # ------------ event log ------------

    def event_log(self) -> EventLog:
        return EventLog(self.events_log)

    # ------------ serialization ------------

    def to_task_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "title": self.title,
            "slug": self.slug,
            "request": self.request,
            "status": self.status,
            "budget": self.budget.to_dict(),
            "created_at": self.created_at,
        }

    def write_task_json(self) -> None:
        self.task_json.write_text(json.dumps(self.to_task_dict(), indent=2), encoding="utf-8")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def read_manifest(self) -> dict[str, Any]:
        if not self.manifest_json.exists():
            return {}
        return json.loads(self.manifest_json.read_text(encoding="utf-8"))

    def update_status(self, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid task status: {status!r}; valid: {sorted(VALID_STATUSES)}")
        self.status = status
        self.write_task_json()
        # Mirror status into manifest if present.
        if self.manifest_json.exists():
            m = self.read_manifest()
            m["status"] = status
            self.write_manifest(m)

    # ------------ README ------------

    def write_readme(self) -> None:
        budget_snapshot = self.budget.to_dict()
        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"- **Task ID:** `{self.task_id}`")
        lines.append(f"- **Status:** `{self.status}`")
        if self.parent_task_id:
            lines.append(f"- **Parent task:** `{self.parent_task_id}`")
        else:
            lines.append("- **Parent task:** _(root)_")
        if self.stage_used:
            lines.append(f"- **Stage used:** `{self.stage_used}`")
        lines.append(f"- **Created at:** {self.created_at}")
        lines.append("")
        lines.append("## Request")
        lines.append("")
        lines.append(self.request.strip() or "_(none)_")
        lines.append("")
        lines.append("## Budget")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(budget_snapshot, indent=2))
        lines.append("```")
        lines.append("")
        if self.children:
            lines.append("## Child tasks")
            lines.append("")
            for c in self.children:
                rel = c.directory.relative_to(self.directory)
                lines.append(f"- `{rel}` — {c.title} ({c.status})")
            lines.append("")
        if self.result_json.exists():
            lines.append("## Important outputs")
            lines.append("")
            lines.append(f"- `{self.result_json.relative_to(self.directory)}`")
            if self.answer_md.exists():
                lines.append(f"- `{self.answer_md.relative_to(self.directory)}`")
            lines.append("")
        lines.append("## Reproduction")
        lines.append("")
        lines.append("```bash")
        lines.append("papermill task.ipynb executed.ipynb -f inputs/parameters.json")
        lines.append("```")
        lines.append("")
        self.readme.write_text("\n".join(lines), encoding="utf-8")

    # ------------ children ------------

    def list_children(self) -> list[Task]:
        if not self.children_dir.exists():
            return []
        out: list[Task] = []
        for d in sorted(p for p in self.children_dir.iterdir() if p.is_dir()):
            t = Task.load(d)
            if t is not None:
                out.append(t)
        return out

    def create_child(
        self,
        title: str,
        request: str,
        *,
        parameters: dict[str, Any] | None = None,
        budget: Budget | None = None,
    ) -> Task:
        """Create a child task in this task's children/ directory.

        Child directories are numerically prefixed (001-, 002-, ...).
        """
        self.children_dir.mkdir(parents=True, exist_ok=True)
        # Determine next ordinal from existing children.
        existing = [p.name for p in self.children_dir.iterdir() if p.is_dir()]
        ordinal = len(existing) + 1
        slug = _slugify(title, fallback="child")
        dir_name = f"{ordinal:03d}-{slug}"
        child_dir = self.children_dir / dir_name
        child = _initialize_task(
            directory=child_dir,
            title=title,
            request=request,
            parameters=parameters or {},
            budget=budget or Budget(),
            parent_task_id=self.task_id,
        )
        self.children.append(child)
        # Persist parent's child list into its manifest.
        manifest = self.read_manifest() or _new_manifest(self)
        manifest.setdefault("children", [])
        manifest["children"].append(
            {
                "task_id": child.task_id,
                "title": child.title,
                "directory": str(child.directory.relative_to(self.directory)),
                "status": child.status,
            }
        )
        self.write_manifest(manifest)
        self.event_log().append(
            "child_task_created",
            child_task_id=child.task_id,
            directory=str(child.directory.relative_to(self.directory)),
        )
        # Refresh README to include the new child.
        self.write_readme()
        return child

    # ------------ loading ------------

    @classmethod
    def load(cls, directory: Path | str) -> Task | None:
        d = Path(directory)
        tj = d / "task.json"
        if not tj.exists():
            return None
        data = json.loads(tj.read_text(encoding="utf-8"))
        task = cls(
            task_id=data["task_id"],
            title=data.get("title", "task"),
            slug=data.get("slug", _slugify(data.get("title", "task"))),
            request=data.get("request", ""),
            directory=d,
            parent_task_id=data.get("parent_task_id"),
            status=data.get("status", "pending"),
            budget=Budget.from_dict(data.get("budget") or {}),
            created_at=data.get("created_at", iso_now()),
        )
        task.children = task.list_children()
        return task


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _new_manifest(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "parent_task_id": task.parent_task_id,
        "status": task.status,
        "stage_used": None,
        "started_at": None,
        "finished_at": None,
        "budget_initial": task.budget.to_dict(),
        "budget_used": {},
        "budget_remaining": {},
        "children": [],
        "outputs": {},
        "tests": {},
        "stage_decision": {
            "chosen": None,
            "retrieve": {"attempted": False, "result": None},
            "compose": {"attempted": False, "result": None},
            "transform": {"attempted": False, "result": None},
            "generate": {"attempted": False, "result": None},
        },
    }


def _initialize_task(
    *,
    directory: Path,
    title: str,
    request: str,
    parameters: dict[str, Any],
    budget: Budget,
    parent_task_id: str | None,
) -> Task:
    directory.mkdir(parents=True, exist_ok=False)
    task = Task(
        task_id=make_task_id(),
        title=title,
        slug=_slugify(title),
        request=request,
        directory=directory,
        parent_task_id=parent_task_id,
        status="pending",
        budget=budget,
    )
    # Create the required directory structure (Section 8).
    for d in (task.inputs_dir, task.outputs_dir, task.logs_dir, task.artifacts_dir, task.children_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Write inputs.
    task.request_md.write_text(request, encoding="utf-8")
    task.parameters_json.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    # Write task.json + manifest.json + README.md.
    task.write_task_json()
    task.write_manifest(_new_manifest(task))
    task.write_readme()
    # Touch events log + emit task_created.
    task.event_log().append("task_created", task_id=task.task_id, title=title)
    if budget.to_dict() != Budget().to_dict():
        task.event_log().append("budget_allocated", budget=budget.to_dict())
    return task


def create_root_task(
    runs_root: Path | str,
    *,
    title: str,
    request: str,
    parameters: dict[str, Any] | None = None,
    budget: Budget | dict[str, Any] | None = None,
    when: datetime | None = None,
) -> Task:
    """Create a brand-new top-level task under ``runs_root``.

    The task directory follows ``runs/YYYY/MM/DD/HHMMSS-task-slug`` (Section 7).
    """
    when = when or utcnow()
    if isinstance(budget, dict):
        budget_obj = Budget.from_dict(budget)
    elif budget is None:
        budget_obj = Budget()
    else:
        budget_obj = budget
    runs_root = Path(runs_root)
    slug = _slugify(title)
    date_dir = runs_root / f"{when.year:04d}" / f"{when.month:02d}" / f"{when.day:02d}"
    base_name = when.strftime("%H%M%S") + f"-{slug}"
    candidate = date_dir / base_name
    # Disambiguate if the exact directory already exists (e.g. two tasks in the same second).
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = date_dir / f"{base_name}-{suffix}"
    return _initialize_task(
        directory=candidate,
        title=title,
        request=request,
        parameters=parameters or {},
        budget=budget_obj,
        parent_task_id=None,
    )


# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------


@dataclass
class TaskGraph:
    """In-memory representation of a task and (recursively) all its descendants."""

    root: Task

    @classmethod
    def load(cls, directory: Path | str) -> TaskGraph:
        root = Task.load(directory)
        if root is None:
            raise FileNotFoundError(f"No task.json found at {directory}")
        cls._load_children(root)
        return cls(root=root)

    @classmethod
    def _load_children(cls, task: Task) -> None:
        task.children = task.list_children()
        for c in task.children:
            cls._load_children(c)

    def walk(self) -> list[Task]:
        out: list[Task] = []

        def _w(t: Task) -> None:
            out.append(t)
            for c in t.children:
                _w(c)

        _w(self.root)
        return out

    def to_dict(self) -> dict[str, Any]:
        def _node(t: Task) -> dict[str, Any]:
            return {
                "task_id": t.task_id,
                "title": t.title,
                "status": t.status,
                "directory": str(t.directory),
                "children": [_node(c) for c in t.children],
            }

        return _node(self.root)
