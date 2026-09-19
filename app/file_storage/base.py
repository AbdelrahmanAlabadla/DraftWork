from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from datetime import datetime
from typing import Any


class FileStorage(ABC):
    @abstractmethod
    def put_file(self, key: str, source: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def put_json(self, key: str, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def put_bytes(self, key: str, value: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, key: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def materialize(self, key: str) -> AbstractContextManager[Path]:
        """Return a local path for a key for the duration of the context."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_tree(self, prefix: str) -> int:
        """Delete every object below a storage prefix and return the count."""
        raise NotImplementedError

    @abstractmethod
    def delete_older_than(
        self, prefix: str, cutoff: datetime, *, path_component: str | None = None
    ) -> int:
        """Delete old objects below a prefix, optionally restricting a path part."""
        raise NotImplementedError

    @abstractmethod
    def list_prefixes(self, prefix: str, *, levels: int) -> list[str]:
        """Return directory-like keys exactly ``levels`` below a prefix."""
        raise NotImplementedError
