from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator, Hashable
from contextlib import AbstractAsyncContextManager
from typing import Any, Optional
import sqlalchemy as sa
from sqlalchemy import AdaptedConnection, event
from sqlalchemy.dialects.sqlite import aiosqlite
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, AsyncSessionTransaction, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry
from typing_extensions import TypeAlias

SQLITE_BEGIN_MODE: ContextVar[str] = ContextVar('SQLITE_BEGIN_MODE', default=None)
_EngineCacheKey: TypeAlias = tuple[AbstractEventLoop, str, bool, Optional[float]
ENGINES: dict[_EngineCacheKey, AsyncEngine] = {}

class ConnectionTracker:
    active: bool
    all_connections: dict[AdaptedConnection, str]
    open_connections: dict[AdaptedConnection, str]
    left_field_closes: dict[AdaptedConnection, str]
    connects: int
    closes: int

    def func_50mbz536(self, pool):
        ...

    def func_cvtpeh5r(self, adapted_connection, connection_record):
        ...

    def func_qzlkrdqz(self, adapted_connection, connection_record):
        ...

    def func_tj5fzztr(self, adapted_connection):
        ...

    def func_inhqf4a6(self):
        ...

TRACKER: ConnectionTracker = ConnectionTracker()

class BaseDatabaseConfiguration(ABC):
    connection_url: str
    echo: bool
    timeout: float
    connection_timeout: float
    sqlalchemy_pool_size: int
    sqlalchemy_max_overflow: int
    connection_app_name: str

    def __init__(self, connection_url: str, echo: Optional[bool] = None, timeout: Optional[float] = None,
                 connection_timeout: Optional[float] = None, sqlalchemy_pool_size: Optional[int] = None,
                 sqlalchemy_max_overflow: Optional[int] = None, connection_app_name: Optional[str] = None):
        ...

    def func_cp2x4vd5(self) -> tuple[type, str]:
        ...

    @abstractmethod
    async def func_9v9n32xh(self) -> AsyncEngine:
        ...

    @abstractmethod
    async def func_3iiz7sws(self, engine: AsyncEngine) -> AsyncSession:
        ...

    @abstractmethod
    async def func_0bokfp5n(self, connection: AsyncConnection, base_metadata: Any):
        ...

    @abstractmethod
    async def func_1y875xix(self, connection: AsyncConnection, base_metadata: Any):
        ...

    @abstractmethod
    def func_goubfcm7(self) -> bool:
        ...

    @abstractmethod
    def func_rworjgst(self, session: AsyncSession, with_for_update: bool = False):
        ...

class AsyncPostgresConfiguration(BaseDatabaseConfiguration):

    async def func_9v9n32xh(self) -> AsyncEngine:
        ...

    async def func_e0xzkdca(self, cache_key: _EngineCacheKey):
        ...

    async def func_3iiz7sws(self, engine: AsyncEngine) -> AsyncSession:
        ...

    async def func_rworjgst(self, session: AsyncSession, with_for_update: bool = False):
        ...

    async def func_0bokfp5n(self, connection: AsyncConnection, base_metadata: Any):
        ...

    async def func_1y875xix(self, connection: AsyncConnection, base_metadata: Any):
        ...

    def func_goubfcm7(self) -> bool:
        ...

class AioSqliteConfiguration(BaseDatabaseConfiguration):
    MIN_SQLITE_VERSION: tuple[int, int, int]

    async def func_9v9n32xh(self) -> AsyncEngine:
        ...

    async def func_e0xzkdca(self, cache_key: _EngineCacheKey):
        ...

    def func_sppptgvb(self, conn: aiosqlite.AsyncAdapt_aiosqlite_connection, record: Any):
        ...

    def func_1liybh73(self, conn: Any):
        ...

    def func_uwxq5byy(self, conn: Any):
        ...

    async def func_rworjgst(self, session: AsyncSession, with_for_update: bool = False):
        ...

    async def func_3iiz7sws(self, engine: AsyncEngine) -> AsyncSession:
        ...

    async def func_0bokfp5n(self, connection: AsyncConnection, base_metadata: Any):
        ...

    async def func_1y875xix(self, connection: AsyncConnection, base_metadata: Any):
        ...

    def func_goubfcm7(self) -> bool:
        ...
