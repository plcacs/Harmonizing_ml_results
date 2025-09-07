import asyncio
import contextlib
import functools
import logging
import typing
import weakref
from contextvars import ContextVar
from types import TracebackType
from urllib.parse import SplitResult, parse_qsl, unquote, urlsplit

from sqlalchemy import text
from sqlalchemy.sql import ClauseElement
from databases.importer import import_from_string
from databases.interfaces import DatabaseBackend, Record, TransactionBackend

try:
    import click
    LOG_EXTRA: dict[str, str] = {'color_message': 'Query: ' + click.style('%s', bold=True) + ' Args: %s'}
    CONNECT_EXTRA: dict[str, str] = {'color_message': 'Connected to database ' + click.style('%s', bold=True)}
    DISCONNECT_EXTRA: dict[str, str] = {'color_message': 'Disconnected from database ' + click.style('%s', bold=True)}
except ImportError:
    LOG_EXTRA = {}
    CONNECT_EXTRA = {}
    DISCONNECT_EXTRA = {}
logger = logging.getLogger('databases')
_ACTIVE_TRANSACTIONS: ContextVar[typing.Optional[weakref.WeakKeyDictionary]] = ContextVar('databases:active_transactions', default=None)

class Database:
    SUPPORTED_BACKENDS: dict[str, str] = {
        'postgresql': 'databases.backends.postgres:PostgresBackend',
        'postgresql+aiopg': 'databases.backends.aiopg:AiopgBackend',
        'postgres': 'databases.backends.postgres:PostgresBackend',
        'mysql': 'databases.backends.mysql:MySQLBackend',
        'mysql+asyncmy': 'databases.backends.asyncmy:AsyncMyBackend',
        'sqlite': 'databases.backends.sqlite:SQLiteBackend'
    }

    def __init__(self, url: str, *, force_rollback: bool = False, **options: typing.Any) -> None:
        self.url: DatabaseURL = DatabaseURL(url)
        self.options: dict[str, typing.Any] = options
        self.is_connected: bool = False
        self._connection_map: weakref.WeakKeyDictionary[asyncio.Task, Connection] = weakref.WeakKeyDictionary()
        self._force_rollback: bool = force_rollback
        backend_str: str = self._get_backend()
        backend_cls: type[DatabaseBackend] = import_from_string(backend_str)
        assert issubclass(backend_cls, DatabaseBackend)
        self._backend: DatabaseBackend = backend_cls(self.url, **self.options)
        self._global_connection: typing.Optional[Connection] = None
        self._global_transaction: typing.Optional[Transaction] = None

    @property
    def _current_task(self) -> asyncio.Task:
        task: typing.Optional[asyncio.Task] = asyncio.current_task()
        if not task:
            raise RuntimeError('No currently active asyncio.Task found')
        return task

    @property
    def _connection(self) -> typing.Optional["Connection"]:
        return self._connection_map.get(self._current_task)

    @_connection.setter
    def _connection(self, connection: typing.Optional["Connection"]) -> typing.Optional["Connection"]:
        task: asyncio.Task = self._current_task
        if connection is None:
            self._connection_map.pop(task, None)
        else:
            self._connection_map[task] = connection
        return self._connection

    async def connect(self) -> None:
        """
        Establish the connection pool.
        """
        if self.is_connected:
            logger.debug('Already connected, skipping connection')
            return
        await self._backend.connect()
        logger.info('Connected to database %s', self.url.obscure_password, extra=CONNECT_EXTRA)
        self.is_connected = True
        if self._force_rollback:
            assert self._global_connection is None
            assert self._global_transaction is None
            self._global_connection = Connection(self, self._backend)
            self._global_transaction = self._global_connection.transaction(force_rollback=True)
            await self._global_transaction.__aenter__()

    async def disconnect(self) -> None:
        """
        Close all connections in the connection pool.
        """
        if not self.is_connected:
            logger.debug('Already disconnected, skipping disconnection')
            return
        if self._force_rollback:
            assert self._global_connection is not None
            assert self._global_transaction is not None
            await self._global_transaction.__aexit__(None, None, None)
            self._global_transaction = None
            self._global_connection = None
        else:
            self._connection = None
        await self._backend.disconnect()
        logger.info('Disconnected from database %s', self.url.obscure_password, extra=DISCONNECT_EXTRA)
        self.is_connected = False

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: typing.Optional[type[BaseException]], exc_value: typing.Optional[BaseException], traceback: typing.Optional[TracebackType]) -> None:
        await self.disconnect()

    async def fetch_all(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> list[Record]:
        async with self.connection() as connection:
            return await connection.fetch_all(query, values)

    async def fetch_one(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> typing.Optional[Record]:
        async with self.connection() as connection:
            return await connection.fetch_one(query, values)

    async def fetch_val(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None, column: int = 0) -> typing.Any:
        async with self.connection() as connection:
            return await connection.fetch_val(query, values, column=column)

    async def execute(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> typing.Any:
        async with self.connection() as connection:
            return await connection.execute(query, values)

    async def execute_many(self, query: typing.Union[str, ClauseElement], values: list[dict[str, typing.Any]]) -> typing.Any:
        async with self.connection() as connection:
            return await connection.execute_many(query, values)

    async def iterate(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> typing.AsyncGenerator[Record, None]:
        async with self.connection() as connection:
            async for record in connection.iterate(query, values):
                yield record

    def connection(self) -> "Connection":
        if self._global_connection is not None:
            return self._global_connection
        if not self._connection:
            self._connection = Connection(self, self._backend)
        return self._connection

    def transaction(self, *, force_rollback: bool = False, **kwargs: typing.Any) -> "Transaction":
        return Transaction(self.connection, force_rollback=force_rollback, **kwargs)

    @contextlib.contextmanager
    def force_rollback(self) -> typing.Generator[None, None, None]:
        initial: bool = self._force_rollback
        self._force_rollback = True
        try:
            yield
        finally:
            self._force_rollback = initial

    def _get_backend(self) -> str:
        return self.SUPPORTED_BACKENDS.get(self.url.scheme, self.SUPPORTED_BACKENDS[self.url.dialect])


class Connection:
    def __init__(self, database: Database, backend: DatabaseBackend) -> None:
        self._database: Database = database
        self._backend: DatabaseBackend = backend
        self._connection_lock: asyncio.Lock = asyncio.Lock()
        self._connection: typing.Any = self._backend.connection()
        self._connection_counter: int = 0
        self._transaction_lock: asyncio.Lock = asyncio.Lock()
        self._transaction_stack: list["Transaction"] = []
        self._query_lock: asyncio.Lock = asyncio.Lock()

    async def __aenter__(self) -> "Connection":
        async with self._connection_lock:
            self._connection_counter += 1
            try:
                if self._connection_counter == 1:
                    await self._connection.acquire()
            except BaseException as e:
                self._connection_counter -= 1
                raise e
        return self

    async def __aexit__(self, exc_type: typing.Optional[type[BaseException]], exc_value: typing.Optional[BaseException], traceback: typing.Optional[TracebackType]) -> None:
        async with self._connection_lock:
            assert self._connection is not None
            self._connection_counter -= 1
            if self._connection_counter == 0:
                await self._connection.release()
                self._database._connection = None

    async def fetch_all(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> list[Record]:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_all(built_query)

    async def fetch_one(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> typing.Optional[Record]:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_one(built_query)

    async def fetch_val(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None, column: int = 0) -> typing.Any:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_val(built_query, column)

    async def execute(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> typing.Any:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.execute(built_query)

    async def execute_many(self, query: typing.Union[str, ClauseElement], values: list[dict[str, typing.Any]]) -> None:
        queries: list[typing.Any] = [self._build_query(query, values_set) for values_set in values]
        async with self._query_lock:
            await self._connection.execute_many(queries)

    async def iterate(self, query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> typing.AsyncGenerator[Record, None]:
        built_query = self._build_query(query, values)
        async with self.transaction():
            async with self._query_lock:
                async for record in self._connection.iterate(built_query):
                    yield record

    def transaction(self, *, force_rollback: bool = False, **kwargs: typing.Any) -> "Transaction":
        def connection_callable() -> "Connection":
            return self
        return Transaction(connection_callable, force_rollback, **kwargs)

    @property
    def raw_connection(self) -> typing.Any:
        return self._connection.raw_connection

    @staticmethod
    def _build_query(query: typing.Union[str, ClauseElement], values: typing.Optional[dict[str, typing.Any]] = None) -> ClauseElement:
        if isinstance(query, str):
            query = text(query)
            return query.bindparams(**values) if values is not None else query
        elif values:
            return query.values(**values)
        return query


_CallableType = typing.TypeVar('_CallableType', bound=typing.Callable[..., typing.Any])

class Transaction:
    def __init__(self, connection_callable: typing.Callable[[], Connection], force_rollback: bool, **kwargs: typing.Any) -> None:
        self._connection_callable: typing.Callable[[], Connection] = connection_callable
        self._force_rollback: bool = force_rollback
        self._extra_options: dict[str, typing.Any] = kwargs

    @property
    def _connection(self) -> Connection:
        return self._connection_callable()

    @property
    def _transaction(self) -> typing.Optional[TransactionBackend]:
        transactions = _ACTIVE_TRANSACTIONS.get()
        if transactions is None:
            return None
        return transactions.get(self, None)

    @_transaction.setter
    def _transaction(self, transaction: typing.Optional[TransactionBackend]) -> typing.Optional[TransactionBackend]:
        transactions = _ACTIVE_TRANSACTIONS.get()
        if transactions is None:
            transactions = weakref.WeakKeyDictionary()
        else:
            transactions = transactions.copy()
        if transaction is None:
            transactions.pop(self, None)
        else:
            transactions[self] = transaction
        _ACTIVE_TRANSACTIONS.set(transactions)
        return transactions.get(self, None)

    async def __aenter__(self) -> "Transaction":
        """
        Called when entering `async with database.transaction()`
        """
        await self.start()
        return self

    async def __aexit__(self, exc_type: typing.Optional[type[BaseException]], exc_value: typing.Optional[BaseException], traceback: typing.Optional[TracebackType]) -> None:
        """
        Called when exiting `async with database.transaction()`
        """
        if exc_type is not None or self._force_rollback:
            await self.rollback()
        else:
            await self.commit()

    def __await__(self) -> typing.Generator[typing.Any, None, "Transaction"]:
        """
        Called if using the low-level `transaction = await database.transaction()`
        """
        return self.start().__await__()

    def __call__(self, func: _CallableType) -> _CallableType:
        """
        Called if using `@database.transaction()` as a decorator.
        """
        @functools.wraps(func)
        async def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            async with self:
                return await func(*args, **kwargs)
        return typing.cast(_CallableType, wrapper)

    async def start(self) -> "Transaction":
        self._transaction = self._connection._connection.transaction()
        async with self._connection._transaction_lock:
            is_root: bool = not self._connection._transaction_stack
            await self._connection.__aenter__()
            await self._transaction.start(is_root=is_root, extra_options=self._extra_options)
            self._connection._transaction_stack.append(self)
        return self

    async def commit(self) -> None:
        async with self._connection._transaction_lock:
            assert self._connection._transaction_stack[-1] is self
            self._connection._transaction_stack.pop()
            assert self._transaction is not None
            await self._transaction.commit()
            await self._connection.__aexit__(None, None, None)
            self._transaction = None

    async def rollback(self) -> None:
        async with self._connection._transaction_lock:
            assert self._connection._transaction_stack[-1] is self
            self._connection._transaction_stack.pop()
            assert self._transaction is not None
            await self._transaction.rollback()
            await self._connection.__aexit__(None, None, None)
            self._transaction = None


class _EmptyNetloc(str):
    def __bool__(self) -> bool:
        return True


class DatabaseURL:
    def __init__(self, url: typing.Union[str, "DatabaseURL"]) -> None:
        if isinstance(url, DatabaseURL):
            self._url: str = url._url
        elif isinstance(url, str):
            self._url = url
        else:
            raise TypeError(f'Invalid type for DatabaseURL. Expected str or DatabaseURL, got {type(url)}')

    @property
    def components(self) -> SplitResult:
        if not hasattr(self, '_components'):
            self._components = urlsplit(self._url)
        return self._components

    @property
    def scheme(self) -> str:
        return self.components.scheme

    @property
    def dialect(self) -> str:
        return self.components.scheme.split('+')[0]

    @property
    def driver(self) -> str:
        if '+' not in self.components.scheme:
            return ''
        return self.components.scheme.split('+', 1)[1]

    @property
    def userinfo(self) -> typing.Optional[bytes]:
        if self.components.username:
            info: str = self.components.username
            if self.components.password:
                info += ':' + self.components.password
            return info.encode('utf-8')
        return None

    @property
    def username(self) -> typing.Optional[str]:
        if self.components.username is None:
            return None
        return unquote(self.components.username)

    @property
    def password(self) -> typing.Optional[str]:
        if self.components.password is None:
            return None
        return unquote(self.components.password)

    @property
    def hostname(self) -> typing.Optional[str]:
        return self.components.hostname or self.options.get('host') or self.options.get('unix_sock')

    @property
    def port(self) -> typing.Optional[int]:
        return self.components.port

    @property
    def netloc(self) -> str:
        return self.components.netloc

    @property
    def database(self) -> str:
        path: str = self.components.path
        if path.startswith('/'):
            path = path[1:]
        return unquote(path)

    @property
    def options(self) -> dict[str, str]:
        if not hasattr(self, '_options'):
            self._options = dict(parse_qsl(self.components.query))
        return self._options

    def replace(self, **kwargs: typing.Any) -> "DatabaseURL":
        if 'username' in kwargs or 'password' in kwargs or 'hostname' in kwargs or ('port' in kwargs):
            hostname: typing.Any = kwargs.pop('hostname', self.hostname)
            port: typing.Any = kwargs.pop('port', self.port)
            username: typing.Any = kwargs.pop('username', self.components.username)
            password: typing.Any = kwargs.pop('password', self.components.password)
            netloc: str = hostname
            if port is not None:
                netloc += f':{port}'
            if username is not None:
                userpass: str = username
                if password is not None:
                    userpass += f':{password}'
                netloc = f'{userpass}@{netloc}'
            kwargs['netloc'] = netloc
        if 'database' in kwargs:
            kwargs['path'] = '/' + kwargs.pop('database')
        if 'dialect' in kwargs or 'driver' in kwargs:
            dialect: str = kwargs.pop('dialect', self.dialect)
            driver: str = kwargs.pop('driver', self.driver)
            kwargs['scheme'] = f'{dialect}+{driver}' if driver else dialect
        if not kwargs.get('netloc', self.netloc):
            kwargs['netloc'] = _EmptyNetloc()
        components = self.components._replace(**kwargs)
        return self.__class__(components.geturl())

    @property
    def obscure_password(self) -> str:
        if self.password:
            return self.replace(password='********')._url
        return self._url

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.obscure_password)})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DatabaseURL):
            return str(self) == str(other)
        return False