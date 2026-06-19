"""Build fixture notebooks programmatically.

Importable so tests can construct notebooks at runtime if needed; also
runnable to write the canonical ``.ipynb`` files into
``tests/fixtures/notebooks/``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_notebook


def passing_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [
        new_code_cell("x = 1\nx", id="cell_one"),
        new_code_cell("y = x + 41\ny", id="cell_two"),
        new_code_cell("print('hello from fixture')", id="cell_three"),
    ]
    return nb


def erroring_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [
        new_code_cell("good = 'before'\ngood", id="cell_good"),
        new_code_cell("raise RuntimeError('boom from fixture')", id="cell_bad"),
        new_code_cell("never_runs = True", id="cell_skipped"),
    ]
    return nb


def timeout_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [
        new_code_cell("ok = 1", id="cell_fast"),
        new_code_cell("import time\ntime.sleep(60)", id="cell_slow"),
    ]
    return nb


FIXTURES = {
    "passing.ipynb": passing_notebook,
    "erroring.ipynb": erroring_notebook,
    "timeout.ipynb": timeout_notebook,
}


def write_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in FIXTURES.items():
        nbformat.write(builder(), out_dir / name)


if __name__ == "__main__":
    write_all(Path(__file__).parent / "notebooks")
