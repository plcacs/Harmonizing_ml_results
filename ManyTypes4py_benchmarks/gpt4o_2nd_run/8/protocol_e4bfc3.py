from typing import Optional, Protocol, runtime_checkable

@runtime_checkable
class LockManager(Protocol):

    def acquire_lock(self, key: str, holder: str, acquire_timeout: Optional[int] = None, hold_timeout: Optional[int] = None) -> bool:
        ...

    async def aacquire_lock(self, key: str, holder: str, acquire_timeout: Optional[int] = None, hold_timeout: Optional[int] = None) -> bool:
        ...

    def release_lock(self, key: str, holder: str) -> None:
        ...

    def is_locked(self, key: str) -> bool:
        ...

    def is_lock_holder(self, key: str, holder: str) -> bool:
        ...

    def wait_for_lock(self, key: str, timeout: Optional[int] = None) -> bool:
        ...

    async def await_for_lock(self, key: str, timeout: Optional[int] = None) -> bool:
        ...
