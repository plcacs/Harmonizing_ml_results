    def acquire_lock(self, key: str, holder: str, acquire_timeout: Optional[float] = None, hold_timeout: Optional[float] = None) -> bool:

    async def aacquire_lock(self, key: str, holder: str, acquire_timeout: Optional[float] = None, hold_timeout: Optional[float] = None) -> bool:

    def release_lock(self, key: str, holder: str) -> None:

    def wait_for_lock(self, key: str, timeout: Optional[float] = None) -> bool:

    async def await_for_lock(self, key: str, timeout: Optional[float] = None) -> bool:

    def is_locked(self, key: str) -> bool:

    def is_lock_holder(self, key: str, holder: str) -> bool:
