from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.file_storage.base import FileStorage


class LocalFileStorage(FileStorage):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        normalized = str(key).replace("\\", "/").lstrip("/")
        candidate = (self.root / normalized).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Storage key escapes the configured root")
        return candidate

    def put_file(self, key: str, source: Path) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def put_json(self, key: str, value: Any) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(destination)

    def put_bytes(self, key: str, value: bytes) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(value)
        temporary.replace(destination)

    def read_json(self, key: str) -> Any:
        return json.loads(self._path(key).read_text(encoding="utf-8"))

    @contextmanager
    def materialize(self, key: str) -> Iterator[Path]:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        yield path

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
