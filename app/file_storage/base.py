from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
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
