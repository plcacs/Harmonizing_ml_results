from typing import Optional, Union, Dict
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.lock import Lock as AsyncLock
from redis.lock import Lock
from prefect.locking.protocol import LockManager


class RedisLockManager(LockManager):
    """
    A lock manager that uses Redis as a backend.

    Attributes:
        host: The host of the Redis server
        port: The port the Redis server is running on
        db: The database to write to and read from
        username: The username to use when connecting to the Redis server
        password: The password to use when connecting to the Redis server
        ssl: Whether to use SSL when connecting to the Redis server
        client: The Redis client used to communicate with the Redis server
        async_client: The asynchronous Redis client used to communicate with the Redis server
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssl: bool = False,
    ) -> None:
        self.host: str = host
        self.port: int = port
        self.db: int = db
        self.username: Optional[str] = username
        self.password: Optional[str] = password
        self.ssl: bool = ssl
        self.client: Redis = Redis(
            host=self.host, port=self.port, db=self.db, username=self.username, password=self.password
        )
        self.async_client: AsyncRedis = AsyncRedis(
            host=self.host, port=self.port, db=self.db, username=self.username, password=self.password
        )
        self._locks: Dict[str, Union[Lock, AsyncLock]] = {}

    @staticmethod
    def _lock_name_for_key(key: str) -> str:
        return f'lock:{key}'

    def acquire_lock(
        self, key: str, holder: str, acquire_timeout: Optional[float] = None, hold_timeout: Optional[float] = None
    ) -> bool:
        lock_name: str = self._lock_name_for_key(key)
        lock: Optional[Union[Lock, AsyncLock]] = self._locks.get(lock_name)
        if lock is not None and self.is_lock_holder(key, holder):
            return True
        else:
            lock = Lock(self.client, lock_name, timeout=hold_timeout, thread_local=False)
        lock_acquired: bool = lock.acquire(token=holder, blocking_timeout=acquire_timeout)
        if lock_acquired:
            self._locks[lock_name] = lock
        return lock_acquired

    async def aacquire_lock(
        self, key: str, holder: str, acquire_timeout: Optional[float] = None, hold_timeout: Optional[float] = None
    ) -> bool:
        lock_name: str = self._lock_name_for_key(key)
        lock: Optional[Union[Lock, AsyncLock]] = self._locks.get(lock_name)
        if lock is not None and self.is_lock_holder(key, holder):
            return True
        else:
            lock = AsyncLock(self.async_client, lock_name, timeout=hold_timeout, thread_local=False)
        lock_acquired: bool = await lock.acquire(token=holder, blocking_timeout=acquire_timeout)
        if lock_acquired:
            self._locks[lock_name] = lock
        return lock_acquired

    def release_lock(self, key: str, holder: str) -> None:
        lock_name: str = self._lock_name_for_key(key)
        lock: Optional[Union[Lock, AsyncLock]] = self._locks.get(lock_name)
        if lock is None or not self.is_lock_holder(key, holder):
            raise ValueError(f'No lock held by {holder} for transaction with key {key}')
        lock.release()
        del self._locks[lock_name]

    def wait_for_lock(self, key: str, timeout: Optional[float] = None) -> bool:
        lock_name: str = self._lock_name_for_key(key)
        lock: Lock = Lock(self.client, lock_name)
        lock_freed: bool = lock.acquire(blocking_timeout=timeout)
        if lock_freed:
            lock.release()
        return lock_freed

    async def await_for_lock(self, key: str, timeout: Optional[float] = None) -> bool:
        lock_name: str = self._lock_name_for_key(key)
        lock: AsyncLock = AsyncLock(self.async_client, lock_name)
        lock_freed: bool = await lock.acquire(blocking_timeout=timeout)
        if lock_freed:
            await lock.release()
        return lock_freed

    def is_locked(self, key: str) -> bool:
        lock_name: str = self._lock_name_for_key(key)
        lock: Lock = Lock(self.client, lock_name)
        return lock.locked()

    def is_lock_holder(self, key: str, holder: str) -> bool:
        lock_name: str = self._lock_name_for_key(key)
        lock: Optional[Union[Lock, AsyncLock]] = self._locks.get(lock_name)
        if lock is None:
            return False
        token: Optional[bytes] = getattr(lock.local, 'token', None)
        if token is None:
            return False
        return token.decode() == holder