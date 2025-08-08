from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator, Hashable
from contextlib import AbstractAsyncContextManager
from typing import Any, Optional

from sqlalchemy import AdaptedConnection
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, AsyncSessionTransaction
from sqlalchemy.pool import ConnectionPoolEntry

SQLITE_BEGIN_MODE: ContextVar[Optional[str]] = ContextVar('SQLITE_BEGIN_MODE', default=None)
_EngineCacheKey: TypeAlias = tuple[AbstractEventLoop, str, bool, Optional[float]]
ENGINES: dict[_EngineCacheKey, AsyncEngine] = {}

class ConnectionTracker:
    def __init__(self):
        self.active: bool = False
        self.all_connections: dict[AdaptedConnection, str] = {}
        self.open_connections: dict[AdaptedConnection, str] = {}
        self.left_field_closes: dict[AdaptedConnection, str] = {}
        self.connects: int = 0
        self.closes: int = 0

    def track_pool(self, pool: ConnectionPoolEntry) -> None:
        ...

    def on_connect(self, adapted_connection: AdaptedConnection, connection_record: Any) -> None:
        ...

    def on_close(self, adapted_connection: AdaptedConnection, connection_record: Any) -> None:
        ...

    def on_close_detached(self, adapted_connection: AdaptedConnection) -> None:
        ...

    def clear(self) -> None:
        ...

TRACKER: ConnectionTracker = ConnectionTracker()

class BaseDatabaseConfiguration(ABC):
    def __init__(self, connection_url: str, echo: Optional[bool] = None, timeout: Optional[float] = None, connection_timeout: Optional[float] = None, sqlalchemy_pool_size: Optional[int] = None, sqlalchemy_max_overflow: Optional[int] = None, connection_app_name: Optional[str] = None):
        ...

    def unique_key(self) -> tuple[type, str]:
        ...

    @abstractmethod
    async def engine(self) -> AsyncEngine:
        ...

    @abstractmethod
    async def session(self, engine: AsyncEngine) -> AsyncSession:
        ...

    @abstractmethod
    async def create_db(self, connection: AdaptedConnection, base_metadata: Any) -> None:
        ...

    @abstractmethod
    async def drop_db(self, connection: AdaptedConnection, base_metadata: Any) -> None:
        ...

    @abstractmethod
    def is_inmemory(self) -> bool:
        ...

    @abstractmethod
    def begin_transaction(self, session: AsyncSession, with_for_update: bool = False) -> None:
        ...

class AsyncPostgresConfiguration(BaseDatabaseConfiguration):
    async def engine(self) -> AsyncEngine:
        ...

    async def schedule_engine_disposal(self, cache_key: _EngineCacheKey) -> None:
        ...

    async def session(self, engine: AsyncEngine) -> AsyncSession:
        ...

    @asynccontextmanager
    async def begin_transaction(self, session: AsyncSession, with_for_update: bool = False) -> AsyncSessionTransaction:
        ...

    async def create_db(self, connection: AdaptedConnection, base_metadata: Any) -> None:
        ...

    async def drop_db(self, connection: AdaptedConnection, base_metadata: Any) -> None:
        ...

    def is_inmemory(self) -> bool:
        ...

class AioSqliteConfiguration(BaseDatabaseConfiguration):
    MIN_SQLITE_VERSION: tuple[int, int, int] = (3, 24, 0)

    async def engine(self) -> AsyncEngine:
        ...

    async def schedule_engine_disposal(self, cache_key: _EngineCacheKey) -> None:
        ...

    def setup_sqlite(self, conn: aiosqlite.AsyncAdapt_aiosqlite_connection, record: Any) -> None:
        ...

    def begin_sqlite_conn(self, conn: aiosqlite.AsyncAdapt_aiosqlite_connection) -> None:
        ...

    def begin_sqlite_stmt(self, conn: aiosqlite.AsyncAdapt_aiosqlite_connection) -> None:
        ...

    @asynccontextmanager
    async def begin_transaction(self, session: AsyncSession, with_for_update: bool = False) -> AsyncSessionTransaction:
        ...

    async def session(self, engine: AsyncEngine) -> AsyncSession:
        ...

    async def create_db(self, connection: AdaptedConnection, base_metadata: Any) -> None:
        ...

    async def drop_db(self, connection: AdaptedConnection, base_metadata: Any) -> None:
        ...

    def is_inmemory(self) -> bool:
        ...
