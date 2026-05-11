"""Atomic JSON state store.

Writes go to a sibling ``.tmp`` file, are fsynced, then renamed over the
target path via ``os.replace`` for atomicity on POSIX. Reads use Pydantic
model validation; missing files return ``None``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AtomicJSONStore:
    """Atomic write-then-rename JSON state store, indexed by object id."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, object_id: str) -> Path:
        return self.directory / f"{object_id}.json"

    def write(self, object_id: str, model: BaseModel, *, fsync: bool = True) -> None:
        path = self._path(object_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = model.model_dump_json(exclude_none=False, indent=2)
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(tmp, path)

    def read(self, object_id: str, model_cls: type[T]) -> T | None:
        path = self._path(object_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return model_cls.model_validate(obj)

    def exists(self, object_id: str) -> bool:
        return self._path(object_id).exists()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json") if p.is_file())

    def delete(self, object_id: str) -> bool:
        path = self._path(object_id)
        if path.exists():
            path.unlink()
            return True
        return False
