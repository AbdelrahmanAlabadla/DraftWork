from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
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

    def delete_tree(self, prefix: str) -> int:
        path = self._path(prefix)
        if not path.exists():
            return 0
        if path.is_file():
            path.unlink()
            return 1
        count = sum(1 for item in path.rglob("*") if item.is_file())
        shutil.rmtree(path)
        return count

    def delete_older_than(
        self, prefix: str, cutoff: datetime, *, path_component: str | None = None
    ) -> int:
        root = self._path(prefix)
        if not root.exists():
            return 0
        cutoff_utc = cutoff.astimezone(timezone.utc)
        deleted = 0
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            relative_parts = path.relative_to(self.root).parts
            if path_component and path_component not in relative_parts:
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff_utc:
                path.unlink(missing_ok=True)
                deleted += 1
        for directory in sorted(
            (item for item in root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        return deleted

    def list_prefixes(self, prefix: str, *, levels: int) -> list[str]:
        root = self._path(prefix)
        if levels < 1 or not root.is_dir():
            return []
        root_depth = len(root.parts)
        results = []
        for path in root.rglob("*"):
            if path.is_dir() and len(path.parts) - root_depth == levels:
                results.append(path.relative_to(self.root).as_posix())
        return sorted(results)
