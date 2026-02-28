from typing import Optional, Protocol, runtime_checkable
from typing import AsyncContextManager, Awaitable

@runtime_checkable
class LockManager(Protocol):
    def acquire_lock(
        self, 
        key: object, 
        holder: object, 
        acquire_timeout: Optional[float] = None, 
        hold_timeout: Optional[float] = None
    ) -> Awaitable[bool]:
        ...

    async def aacquire_lock(
        self, 
        key: object, 
        holder: object, 
        acquire_timeout: Optional[float] = None, 
        hold_timeout: Optional[float] = None
    ) -> bool:
        ...

    def release_lock(self, key: object, holder: object) -> None:
        ...

    def is_locked(self, key: object) -> bool:
        ...

    def is_lock_holder(self, key: object, holder: object) -> bool:
        ...

    def wait_for_lock(self, key: object, timeout: Optional[float] = None) -> bool:
        ...

    async def await_for_lock(self, key: object, timeout: Optional[float] = None) -> bool:
        ...
