import sqlite3
import traceback
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, get_running_loop
from collections.abc import AsyncGenerator, Hashable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from functools import partial
from typing import Any, Optional, Tuple, Dict
import sqlalchemy as sa
from sqlalchemy import AdaptedConnection, event
from sqlalchemy.dialects.sqlite import aiosqlite
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    AsyncSessionTransaction,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry, Pool
from typing_extensions import TypeAlias
from prefect.settings import (
    PREFECT_API_DATABASE_CONNECTION_TIMEOUT,
    PREFECT_API_DATABASE_ECHO,
    PREFECT_API_DATABASE_TIMEOUT,
    PREFECT_TESTING_UNIT_TEST_MODE,
    get_current_settings,
)
from prefect.utilities.asyncutils import add_event_loop_shutdown_callback

SQLITE_BEGIN_MODE: ContextVar[Optional[str]] = ContextVar('SQLITE_BEGIN_MODE', default=None)
_EngineCacheKey: TypeAlias = Tuple[AbstractEventLoop, str, bool, Optional[float]]
ENGINES: Dict[_EngineCacheKey, AsyncEngine] = {}


class ConnectionTracker:
    """A test utility which tracks the connections given out by a connection pool, to
    make it easy to see which connections are currently checked out and open."""

    def __init__(self) -> None:
        self.active: bool = False
        self.all_connections: Dict[AdaptedConnection, str] = {}
        self.open_connections: Dict[AdaptedConnection, str] = {}
        self.left_field_closes: Dict[AdaptedConnection, str] = {}
        self.connects: int = 0
        self.closes: int = 0

    def func_50mbz536(self, pool: Pool) -> None:
        event.listen(pool, 'connect', self.func_cvtpeh5r)
        event.listen(pool, 'close', self.func_qzlkrdqz)
        event.listen(pool, 'close_detached', self.func_tj5fzztr)

    def func_cvtpeh5r(self, adapted_connection: AdaptedConnection, connection_record: Any) -> None:
        self.all_connections[adapted_connection] = ''.join(traceback.format_stack())
        self.open_connections[adapted_connection] = ''.join(traceback.format_stack())
        self.connects += 1

    def func_qzlkrdqz(self, adapted_connection: AdaptedConnection, connection_record: Any) -> None:
        try:
            del self.open_connections[adapted_connection]
        except KeyError:
            self.left_field_closes[adapted_connection] = ''.join(traceback.format_stack())
        self.closes += 1

    def func_tj5fzztr(self, adapted_connection: AdaptedConnection) -> None:
        try:
            del self.open_connections[adapted_connection]
        except KeyError:
            self.left_field_closes[adapted_connection] = ''.join(traceback.format_stack())
        self.closes += 1

    def func_inhqf4a6(self) -> None:
        self.all_connections.clear()
        self.open_connections.clear()
        self.left_field_closes.clear()
        self.connects = 0
        self.closes = 0


TRACKER = ConnectionTracker()


class BaseDatabaseConfiguration(ABC):
    """
    Abstract base class used to inject database connection configuration into Prefect.

    This configuration is responsible for defining how Prefect REST API creates and manages
    database connections and sessions.
    """

    def __init__(
        self,
        connection_url: str,
        echo: Optional[bool] = None,
        timeout: Optional[float] = None,
        connection_timeout: Optional[float] = None,
        sqlalchemy_pool_size: Optional[int] = None,
        sqlalchemy_max_overflow: Optional[int] = None,
        connection_app_name: Optional[str] = None,
    ) -> None:
        self.connection_url: str = connection_url
        self.echo: bool = echo or PREFECT_API_DATABASE_ECHO.value()
        self.timeout: float = timeout or PREFECT_API_DATABASE_TIMEOUT.value()
        self.connection_timeout: float = (
            connection_timeout or PREFECT_API_DATABASE_CONNECTION_TIMEOUT.value()
        )
        self.sqlalchemy_pool_size: int = (
            sqlalchemy_pool_size or get_current_settings().server.database.sqlalchemy.pool_size
        )
        self.sqlalchemy_max_overflow: int = (
            sqlalchemy_max_overflow or get_current_settings().server.database.sqlalchemy.max_overflow
        )
        self.connection_app_name: Optional[str] = (
            connection_app_name
            or get_current_settings().server.database.sqlalchemy.connect_args.application_name
        )

    def func_cp2x4vd5(self) -> Tuple[Any, str]:
        """
        Returns a key used to determine whether to instantiate a new DB interface.
        """
        return self.__class__, self.connection_url

    @abstractmethod
    async def func_9v9n32xh(self) -> AsyncEngine:
        """Returns a SqlAlchemy engine"""

    @abstractmethod
    async def func_3iiz7sws(self, engine: AsyncEngine) -> AsyncSession:
        """
        Retrieves a SQLAlchemy session for an engine.
        """

    @abstractmethod
    async def func_0bokfp5n(self, connection: AsyncConnection, base_metadata: sa.MetaData) -> None:
        """Create the database"""

    @abstractmethod
    async def func_1y875xix(self, connection: AsyncConnection, base_metadata: sa.MetaData) -> None:
        """Drop the database"""

    @abstractmethod
    def func_goubfcm7(self) -> bool:
        """Returns true if database is run in memory"""

    @abstractmethod
    def func_rworjgst(
        self, session: AsyncSession, with_for_update: bool = False
    ) -> AbstractAsyncContextManager[AsyncSessionTransaction]:
        """Enter a transaction for a session"""
        pass


class AsyncPostgresConfiguration(BaseDatabaseConfiguration):

    async def func_9v9n32xh(self) -> AsyncEngine:
        """Retrieves an async SQLAlchemy engine.

        Args:
            connection_url (str, optional): The database connection string.
                Defaults to self.connection_url
            echo (bool, optional): Whether to echo SQL sent
                to the database. Defaults to self.echo
            timeout (float, optional): The database statement timeout, in seconds.
                Defaults to self.timeout

        Returns:
            AsyncEngine: a SQLAlchemy engine
        """
        loop: AbstractEventLoop = get_running_loop()
        cache_key: _EngineCacheKey = (loop, self.connection_url, self.echo, self.timeout)
        if cache_key not in ENGINES:
            kwargs: Dict[str, Any] = get_current_settings().server.database.sqlalchemy.model_dump(mode='json')
            connect_args: Dict[str, Any] = kwargs.pop('connect_args')
            app_name: Optional[str] = connect_args.pop('application_name', None)
            if self.timeout is not None:
                connect_args['command_timeout'] = self.timeout
            if self.connection_timeout is not None:
                connect_args['timeout'] = self.connection_timeout
            if self.connection_app_name is not None or app_name is not None:
                connect_args['server_settings'] = {
                    'application_name': self.connection_app_name or app_name
                }
            if connect_args:
                kwargs['connect_args'] = connect_args
            if self.sqlalchemy_pool_size is not None:
                kwargs['pool_size'] = self.sqlalchemy_pool_size
            if self.sqlalchemy_max_overflow is not None:
                kwargs['max_overflow'] = self.sqlalchemy_max_overflow
            engine: AsyncEngine = create_async_engine(
                self.connection_url,
                echo=self.echo,
                pool_pre_ping=True,
                pool_use_lifo=True,
                **kwargs,
            )
            if TRACKER.active:
                TRACKER.func_50mbz536(engine.pool)
            ENGINES[cache_key] = engine
            await self.func_e0xzkdca(cache_key)
        return ENGINES[cache_key]

    async def func_e0xzkdca(self, cache_key: _EngineCacheKey) -> None:
        """
        Dispose of an engine once the event loop is closing.

        See caveats at `add_event_loop_shutdown_callback`.

        We attempted to lazily clean up old engines when new engines are created, but
        if the loop the engine is attached to is already closed then the connections
        cannot be cleaned up properly and warnings are displayed.

        Engine disposal should only be important when running the application
        ephemerally. Notably, this is an issue in our tests where many short-lived event
        loops and engines are created which can consume all of the available database
        connection slots. Users operating at a scale where connection limits are
        encountered should be encouraged to use a standalone server.
        """

        async def func_kezxbg9u(key: _EngineCacheKey) -> None:
            engine: Optional[AsyncEngine] = ENGINES.pop(key, None)
            if engine:
                await engine.dispose()

        await add_event_loop_shutdown_callback(partial(func_kezxbg9u, cache_key))

    async def func_3iiz7sws(self, engine: AsyncEngine) -> AsyncSession:
        """
        Retrieves a SQLAlchemy session for an engine.

        Args:
            engine: a sqlalchemy engine
        """
        return AsyncSession(engine, expire_on_commit=False)

    @asynccontextmanager
    async def func_rworjgst(
        self, session: AsyncSession, with_for_update: bool = False
    ) -> AsyncGenerator[AsyncSessionTransaction, None]:
        async with session.begin() as transaction:
            yield transaction

    async def func_0bokfp5n(self, connection: AsyncConnection, base_metadata: sa.MetaData) -> None:
        """Create the database"""
        await connection.run_sync(base_metadata.create_all)

    async def func_1y875xix(self, connection: AsyncConnection, base_metadata: sa.MetaData) -> None:
        """Drop the database"""
        await connection.run_sync(base_metadata.drop_all)

    def func_goubfcm7(self) -> bool:
        """Returns true if database is run in memory"""
        return False


class AioSqliteConfiguration(BaseDatabaseConfiguration):
    MIN_SQLITE_VERSION: Tuple[int, int, int] = (3, 24, 0)

    async def func_9v9n32xh(self) -> AsyncEngine:
        """Retrieves an async SQLAlchemy engine.

        Args:
            connection_url (str, optional): The database connection string.
                Defaults to self.connection_url
            echo (bool, optional): Whether to echo SQL sent
                to the database. Defaults to self.echo
            timeout (float, optional): The database statement timeout, in seconds.
                Defaults to self.timeout

        Returns:
            AsyncEngine: a SQLAlchemy engine
        """
        if sqlite3.sqlite_version_info < self.MIN_SQLITE_VERSION:
            required: str = '.'.join(str(v) for v in self.MIN_SQLITE_VERSION)
            raise RuntimeError(
                f'Prefect requires sqlite >= {required} but we found version {sqlite3.sqlite_version}'
            )
        kwargs: Dict[str, Any] = {}
        loop: AbstractEventLoop = get_running_loop()
        cache_key: _EngineCacheKey = (loop, self.connection_url, self.echo, self.timeout)
        if cache_key not in ENGINES:
            if self.timeout is not None:
                kwargs['connect_args'] = {'timeout': self.timeout}
            kwargs['paramstyle'] = 'named'
            if ':memory:' in self.connection_url:
                kwargs.update(
                    poolclass=sa.pool.AsyncAdaptedQueuePool,
                    pool_size=1,
                    max_overflow=0,
                    pool_recycle=-1,
                )
            engine: AsyncEngine = create_async_engine(
                self.connection_url, echo=self.echo, **kwargs
            )
            sa.event.listen(engine.sync_engine, 'connect', self.func_sppptgvb)
            sa.event.listen(engine.sync_engine, 'begin', self.func_uwxq5byy)
            if TRACKER.active:
                TRACKER.func_50mbz536(engine.pool)
            ENGINES[cache_key] = engine
            await self.func_e0xzkdca(cache_key)
        return ENGINES[cache_key]

    async def func_e0xzkdca(self, cache_key: _EngineCacheKey) -> None:
        """
        Dispose of an engine once the event loop is closing.

        See caveats at `add_event_loop_shutdown_callback`.

        We attempted to lazily clean up old engines when new engines are created, but
        if the loop the engine is attached to is already closed then the connections
        cannot be cleaned up properly and warnings are displayed.

        Engine disposal should only be important when running the application
        ephemerally. Notably, this is an issue in our tests where many short-lived event
        loops and engines are created which can consume all of the available database
        connection slots. Users operating at a scale where connection limits are
        encountered should be encouraged to use a standalone server.
        """

        async def func_kezxbg9u(key: _EngineCacheKey) -> None:
            engine: Optional[AsyncEngine] = ENGINES.pop(key, None)
            if engine:
                await engine.dispose()

        await add_event_loop_shutdown_callback(partial(func_kezxbg9u, cache_key))

    def func_sppptgvb(self, conn: DBAPIConnection, record: Any) -> None:
        """Issue PRAGMA statements to SQLITE on connect. PRAGMAs only last for the
        duration of the connection. See https://www.sqlite.org/pragma.html for more info.
        """
        if isinstance(conn, aiosqlite.AsyncConnection):
            self.func_1liybh73(conn)
        cursor = conn.cursor()
        cursor.execute('PRAGMA journal_mode = WAL;')
        cursor.execute('PRAGMA foreign_keys = ON;')
        cursor.execute('PRAGMA legacy_alter_table=OFF')
        cursor.execute('PRAGMA synchronous = NORMAL;')
        cursor.execute('PRAGMA cache_size = 20000;')
        if PREFECT_TESTING_UNIT_TEST_MODE.value() is True:
            cursor.execute('PRAGMA busy_timeout = 5000;')
        else:
            cursor.execute('PRAGMA busy_timeout = 60000;')
        cursor.close()

    def func_1liybh73(self, conn: aiosqlite.AsyncConnection) -> None:
        conn.isolation_level = None

    def func_uwxq5byy(self, conn: AsyncConnection) -> None:
        mode: Optional[str] = SQLITE_BEGIN_MODE.get()
        if mode is not None:
            conn.exec_driver_sql(f'BEGIN {mode}')

    @asynccontextmanager
    async def func_rworjgst(
        self, session: AsyncSession, with_for_update: bool = False
    ) -> AsyncGenerator[AsyncSessionTransaction, None]:
        token: Any = SQLITE_BEGIN_MODE.set('IMMEDIATE' if with_for_update else 'DEFERRED')
        try:
            async with session.begin() as transaction:
                yield transaction
        finally:
            SQLITE_BEGIN_MODE.reset(token)

    async def func_3iiz7sws(self, engine: AsyncEngine) -> AsyncSession:
        """
        Retrieves a SQLAlchemy session for an engine.

        Args:
            engine: a sqlalchemy engine
        """
        return AsyncSession(engine, expire_on_commit=False)

    async def func_0bokfp5n(self, connection: AsyncConnection, base_metadata: sa.MetaData) -> None:
        """Create the database"""
        await connection.run_sync(base_metadata.create_all)

    async def func_1y875xix(self, connection: AsyncConnection, base_metadata: sa.MetaData) -> None:
        """Drop the database"""
        await connection.run_sync(base_metadata.drop_all)

    def func_goubfcm7(self) -> bool:
        """Returns true if database is run in memory"""
        return (':memory:' in self.connection_url or 'mode=memory' in self.connection_url)
