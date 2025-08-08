import sqlite3
import traceback
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, get_running_loop
from collections.abc import AsyncGenerator, Hashable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from functools import partial
from typing import Any, Optional
import sqlalchemy as sa
from sqlalchemy import AdaptedConnection, event
from sqlalchemy.dialects.sqlite import aiosqlite
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, AsyncSessionTransaction, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry
from typing_extensions import TypeAlias
from prefect.settings import PREFECT_API_DATABASE_CONNECTION_TIMEOUT, PREFECT_API_DATABASE_ECHO, PREFECT_API_DATABASE_TIMEOUT, PREFECT_TESTING_UNIT_TEST_MODE, get_current_settings
from prefect.utilities.asyncutils import add_event_loop_shutdown_callback
SQLITE_BEGIN_MODE = ContextVar('SQLITE_BEGIN_MODE', default=None)
_EngineCacheKey = tuple[AbstractEventLoop, str, bool, Optional[float]]
ENGINES = {}

class ConnectionTracker:
    """A test utility which tracks the connections given out by a connection pool, to
    make it easy to see which connections are currently checked out and open."""

    def __init__(self):
        self.active = False
        self.all_connections = {}
        self.open_connections = {}
        self.left_field_closes = {}
        self.connects = 0
        self.closes = 0

    def track_pool(self, pool):
        event.listen(pool, 'connect', self.on_connect)
        event.listen(pool, 'close', self.on_close)
        event.listen(pool, 'close_detached', self.on_close_detached)

    def on_connect(self, adapted_connection, connection_record):
        self.all_connections[adapted_connection] = traceback.format_stack()
        self.open_connections[adapted_connection] = traceback.format_stack()
        self.connects += 1

    def on_close(self, adapted_connection, connection_record):
        try:
            del self.open_connections[adapted_connection]
        except KeyError:
            self.left_field_closes[adapted_connection] = traceback.format_stack()
        self.closes += 1

    def on_close_detached(self, adapted_connection):
        try:
            del self.open_connections[adapted_connection]
        except KeyError:
            self.left_field_closes[adapted_connection] = traceback.format_stack()
        self.closes += 1

    def clear(self):
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

    def __init__(self, connection_url, echo=None, timeout=None, connection_timeout=None, sqlalchemy_pool_size=None, sqlalchemy_max_overflow=None, connection_app_name=None):
        self.connection_url = connection_url
        self.echo = echo or PREFECT_API_DATABASE_ECHO.value()
        self.timeout = timeout or PREFECT_API_DATABASE_TIMEOUT.value()
        self.connection_timeout = connection_timeout or PREFECT_API_DATABASE_CONNECTION_TIMEOUT.value()
        self.sqlalchemy_pool_size = sqlalchemy_pool_size or get_current_settings().server.database.sqlalchemy.pool_size
        self.sqlalchemy_max_overflow = sqlalchemy_max_overflow or get_current_settings().server.database.sqlalchemy.max_overflow
        self.connection_app_name = connection_app_name or get_current_settings().server.database.sqlalchemy.connect_args.application_name

    def unique_key(self):
        """
        Returns a key used to determine whether to instantiate a new DB interface.
        """
        return (self.__class__, self.connection_url)

    @abstractmethod
    async def engine(self):
        """Returns a SqlAlchemy engine"""

    @abstractmethod
    async def session(self, engine):
        """
        Retrieves a SQLAlchemy session for an engine.
        """

    @abstractmethod
    async def create_db(self, connection, base_metadata):
        """Create the database"""

    @abstractmethod
    async def drop_db(self, connection, base_metadata):
        """Drop the database"""

    @abstractmethod
    def is_inmemory(self):
        """Returns true if database is run in memory"""

    @abstractmethod
    def begin_transaction(self, session, with_for_update=False):
        """Enter a transaction for a session"""
        pass

class AsyncPostgresConfiguration(BaseDatabaseConfiguration):

    async def engine(self):
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
        loop = get_running_loop()
        cache_key = (loop, self.connection_url, self.echo, self.timeout)
        if cache_key not in ENGINES:
            kwargs = get_current_settings().server.database.sqlalchemy.model_dump(mode='json')
            connect_args = kwargs.pop('connect_args')
            app_name = connect_args.pop('application_name', None)
            if self.timeout is not None:
                connect_args['command_timeout'] = self.timeout
            if self.connection_timeout is not None:
                connect_args['timeout'] = self.connection_timeout
            if self.connection_app_name is not None or app_name is not None:
                connect_args['server_settings'] = dict(application_name=self.connection_app_name or app_name)
            if connect_args:
                kwargs['connect_args'] = connect_args
            if self.sqlalchemy_pool_size is not None:
                kwargs['pool_size'] = self.sqlalchemy_pool_size
            if self.sqlalchemy_max_overflow is not None:
                kwargs['max_overflow'] = self.sqlalchemy_max_overflow
            engine = create_async_engine(self.connection_url, echo=self.echo, pool_pre_ping=True, pool_use_lifo=True, **kwargs)
            if TRACKER.active:
                TRACKER.track_pool(engine.pool)
            ENGINES[cache_key] = engine
            await self.schedule_engine_disposal(cache_key)
        return ENGINES[cache_key]

    async def schedule_engine_disposal(self, cache_key):
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

        async def dispose_engine(cache_key):
            engine = ENGINES.pop(cache_key, None)
            if engine:
                await engine.dispose()
        await add_event_loop_shutdown_callback(partial(dispose_engine, cache_key))

    async def session(self, engine):
        """
        Retrieves a SQLAlchemy session for an engine.

        Args:
            engine: a sqlalchemy engine
        """
        return AsyncSession(engine, expire_on_commit=False)

    @asynccontextmanager
    async def begin_transaction(self, session, with_for_update=False):
        async with session.begin() as transaction:
            yield transaction

    async def create_db(self, connection, base_metadata):
        """Create the database"""
        await connection.run_sync(base_metadata.create_all)

    async def drop_db(self, connection, base_metadata):
        """Drop the database"""
        await connection.run_sync(base_metadata.drop_all)

    def is_inmemory(self):
        """Returns true if database is run in memory"""
        return False

class AioSqliteConfiguration(BaseDatabaseConfiguration):
    MIN_SQLITE_VERSION = (3, 24, 0)

    async def engine(self):
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
            required = '.'.join((str(v) for v in self.MIN_SQLITE_VERSION))
            raise RuntimeError(f'Prefect requires sqlite >= {required} but we found version {sqlite3.sqlite_version}')
        kwargs = dict()
        loop = get_running_loop()
        cache_key = (loop, self.connection_url, self.echo, self.timeout)
        if cache_key not in ENGINES:
            if self.timeout is not None:
                kwargs['connect_args'] = dict(timeout=self.timeout)
            kwargs['paramstyle'] = 'named'
            if ':memory:' in self.connection_url:
                kwargs.update(poolclass=sa.pool.AsyncAdaptedQueuePool, pool_size=1, max_overflow=0, pool_recycle=-1)
            engine = create_async_engine(self.connection_url, echo=self.echo, **kwargs)
            sa.event.listen(engine.sync_engine, 'connect', self.setup_sqlite)
            sa.event.listen(engine.sync_engine, 'begin', self.begin_sqlite_stmt)
            if TRACKER.active:
                TRACKER.track_pool(engine.pool)
            ENGINES[cache_key] = engine
            await self.schedule_engine_disposal(cache_key)
        return ENGINES[cache_key]

    async def schedule_engine_disposal(self, cache_key):
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

        async def dispose_engine(cache_key):
            engine = ENGINES.pop(cache_key, None)
            if engine:
                await engine.dispose()
        await add_event_loop_shutdown_callback(partial(dispose_engine, cache_key))

    def setup_sqlite(self, conn, record):
        """Issue PRAGMA statements to SQLITE on connect. PRAGMAs only last for the
        duration of the connection. See https://www.sqlite.org/pragma.html for more info.
        """
        if isinstance(conn, aiosqlite.AsyncAdapt_aiosqlite_connection):
            self.begin_sqlite_conn(conn)
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

    def begin_sqlite_conn(self, conn):
        conn.isolation_level = None

    def begin_sqlite_stmt(self, conn):
        mode = SQLITE_BEGIN_MODE.get()
        if mode is not None:
            conn.exec_driver_sql(f'BEGIN {mode}')

    @asynccontextmanager
    async def begin_transaction(self, session, with_for_update=False):
        token = SQLITE_BEGIN_MODE.set('IMMEDIATE' if with_for_update else 'DEFERRED')
        try:
            async with session.begin() as transaction:
                yield transaction
        finally:
            SQLITE_BEGIN_MODE.reset(token)

    async def session(self, engine):
        """
        Retrieves a SQLAlchemy session for an engine.

        Args:
            engine: a sqlalchemy engine
        """
        return AsyncSession(engine, expire_on_commit=False)

    async def create_db(self, connection, base_metadata):
        """Create the database"""
        await connection.run_sync(base_metadata.create_all)

    async def drop_db(self, connection, base_metadata):
        """Drop the database"""
        await connection.run_sync(base_metadata.drop_all)

    def is_inmemory(self):
        """Returns true if database is run in memory"""
        return ':memory:' in self.connection_url or 'mode=memory' in self.connection_url