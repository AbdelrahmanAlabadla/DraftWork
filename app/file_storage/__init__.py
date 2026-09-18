from app.file_storage.base import FileStorage
from app.file_storage.local import LocalFileStorage

__all__ = ["FileStorage", "LocalFileStorage", "get_file_storage"]


def get_file_storage() -> FileStorage:
    from app import config

    if config.STORAGE_BACKEND == "local":
        return LocalFileStorage(config.LOCAL_STORAGE_ROOT)
    raise RuntimeError(f"Unsupported STORAGE_BACKEND: {config.STORAGE_BACKEND}")
