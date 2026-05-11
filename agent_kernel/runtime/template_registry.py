"""Template registry: notebook templates that materialize into runnable notebooks.

For MVP we ship one template (``python-analysis``) and accept either:
- a builtin name, resolved to a path under ``agent_kernel/templates/``
- an arbitrary filesystem path to a ``.ipynb`` template
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import nbformat
from nbformat import NotebookNode


class TemplateNotFound(KeyError):
    pass


_BUILTIN_TEMPLATES = {"python-analysis"}


def list_templates() -> list[str]:
    return sorted(_BUILTIN_TEMPLATES)


def load_template(name_or_path: str) -> NotebookNode:
    """Load a template notebook by builtin name or filesystem path."""
    if name_or_path in _BUILTIN_TEMPLATES:
        return _load_builtin(name_or_path)
    p = Path(name_or_path)
    if p.exists():
        return nbformat.read(p, as_version=4)
    raise TemplateNotFound(name_or_path)


def _load_builtin(name: str) -> NotebookNode:
    try:
        ref = resources.files("agent_kernel.templates").joinpath(f"{name}.ipynb")
        with resources.as_file(ref) as path:
            return nbformat.read(path, as_version=4)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise TemplateNotFound(name) from exc
