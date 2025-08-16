from pathlib import Path
from typing import Optional
import anyio
from prefect.logging.loggers import get_logger
from prefect.types._datetime import DateTime, now, parse_datetime
from .protocol import LockManager

class _LockInfo(TypedDict):
    holder: str
    expiration: Optional[DateTime]
    path: Path

class FileSystemLockManager(LockManager):
    lock_files_directory: Path

    def __init__(self, lock_files_directory: Path):
        self.lock_files_directory = lock_files_directory.expanduser().resolve()
        self._locks: dict[str, _LockInfo] = {}

    def _ensure_lock_files_directory_exists(self) -> None:
        ...

    def _lock_path_for_key(self, key: str) -> Path:
        ...

    def _get_lock_info(self, key: str, use_cache: bool = True) -> Optional[_LockInfo]:
        ...

    async def _aget_lock_info(self, key: str, use_cache: bool = True) -> Optional[_LockInfo]:
        ...

    def acquire_lock(self, key: str, holder: str, acquire_timeout: Optional[int] = None, hold_timeout: Optional[int] = None) -> bool:
        ...

    async def aacquire_lock(self, key: str, holder: str, acquire_timeout: Optional[int] = None, hold_timeout: Optional[int] = None) -> bool:
        ...

    def release_lock(self, key: str, holder: str) -> None:
        ...

    def is_locked(self, key: str, use_cache: bool = False) -> bool:
        ...

    def is_lock_holder(self, key: str, holder: str) -> bool:
        ...

    def wait_for_lock(self, key: str, timeout: Optional[int] = None) -> bool:
        ...

    async def await_for_lock(self, key: str, timeout: Optional[int] = None) -> bool:
        ...
