import asyncio
import threading
from typing import Any, Optional, TypedDict
from typing_extensions import Self
from .protocol import LockManager

class _LockInfo(TypedDict):
    holder: Any
    lock: threading.Lock
    expiration_timer: Optional[threading.Timer]

class MemoryLockManager(LockManager):
    _instance: Optional['MemoryLockManager'] = None

    def __new__(cls, *args, **kwargs) -> 'MemoryLockManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._locks_dict_lock: threading.Lock = threading.Lock()
        self._locks: dict[str, _LockInfo] = {}

    def _expire_lock(self, key: str) -> None:
        ...

    def acquire_lock(self, key: str, holder: Any, acquire_timeout: Optional[float] = None, hold_timeout: Optional[float] = None) -> bool:
        ...

    async def aacquire_lock(self, key: str, holder: Any, acquire_timeout: Optional[float] = None, hold_timeout: Optional[float] = None) -> bool:
        ...

    def release_lock(self, key: str, holder: Any) -> None:
        ...

    def is_locked(self, key: str) -> bool:
        ...

    def is_lock_holder(self, key: str, holder: Any) -> bool:
        ...

    def wait_for_lock(self, key: str, timeout: Optional[float] = None) -> bool:
        ...

    async def await_for_lock(self, key: str, timeout: Optional[float] = None) -> bool:
        ...
