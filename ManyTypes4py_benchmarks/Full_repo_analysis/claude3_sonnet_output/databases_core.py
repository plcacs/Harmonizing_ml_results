import asyncio
import contextlib
import functools
import logging
import typing
import weakref
from contextvars import ContextVar
from types import TracebackType
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, cast, overload
from urllib.parse import SplitResult, parse_qsl, unquote, urlsplit
from sqlalchemy import text
from sqlalchemy.sql import ClauseElement
from databases.importer import import_from_string
from databases.interfaces import DatabaseBackend, Record, TransactionBackend

try:
    import click
    LOG_EXTRA: Dict[str, str] = {'color_message': 'Query: ' + click.style('%s', bold=True) + ' Args: %s'}
    CONNECT_EXTRA: Dict[str, str] = {'color_message': 'Connected to database ' + click.style('%s', bold=True)}
    DISCONNECT_EXTRA: Dict[str, str] = {'color_message': 'Disconnected from database ' + click.style('%s', bold=True)}
except ImportError:
    LOG_EXTRA: Dict[str, Any] = {}
    CONNECT_EXTRA: Dict[str, Any] = {}
    DISCONNECT_EXTRA: Dict[str, Any] = {}

logger: logging.Logger = logging.getLogger('databases')
_ACTIVE_TRANSACTIONS: ContextVar[Optional[weakref.WeakKeyDictionary]] = ContextVar('databases:active_transactions', default=None)

class Database:
    SUPPORTED_BACKENDS: Dict[str, str] = {
        'postgresql': 'databases.backends.postgres:PostgresBackend',
        'postgresql+aiopg': 'databases.backends.aiopg:AiopgBackend',
        'postgres': 'databases.backends.postgres:PostgresBackend',
        'mysql': 'databases.backends.mysql:MySQLBackend',
        'mysql+asyncmy': 'databases.backends.asyncmy:AsyncMyBackend',
        'sqlite': 'databases.backends.sqlite:SQLiteBackend'
    }

    def __init__(self, url: Union[str, 'DatabaseURL'], *, force_rollback: bool = False, **options: Any) -> None:
        self.url: DatabaseURL = DatabaseURL(url)
        self.options: Dict[str, Any] = options
        self.is_connected: bool = False
        self._connection_map: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._force_rollback: bool = force_rollback
        backend_str: str = self._get_backend()
        backend_cls: Type[DatabaseBackend] = import_from_string(backend_str)
        assert issubclass(backend_cls, DatabaseBackend)
        self._backend: DatabaseBackend = backend_cls(self.url, **self.options)
        self._global_connection: Optional[Connection] = None
        self._global_transaction: Optional[Transaction] = None

    @property
    def _current_task(self) -> asyncio.Task:
        task = asyncio.current_task()
        if not task:
            raise RuntimeError('No currently active asyncio.Task found')
        return task

    @property
    def _connection(self) -> Optional[Connection]:
        return self._connection_map.get(self._current_task)

    @_connection.setter
    def _connection(self, connection: Optional[Connection]) -> Optional[Connection]:
        task = self._current_task
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
            return None
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
            return None
        if self._force_rollback:
            assert self._global_connection is not None
            assert self._global_transaction is not None
            await self._global_transaction.__aexit__()
            self._global_transaction = None
            self._global_connection = None
        else:
            self._connection = None
        await self._backend.disconnect()
        logger.info('Disconnected from database %s', self.url.obscure_password, extra=DISCONNECT_EXTRA)
        self.is_connected = False

    async def __aenter__(self) -> 'Database':
        await self.connect()
        return self

    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]] = None, 
        exc_value: Optional[BaseException] = None, 
        traceback: Optional[TracebackType] = None
    ) -> None:
        await self.disconnect()

    async def fetch_all(self, query: Union[str, ClauseElement], values: Optional[Dict[str, Any]] = None) -> List[Record]:
        async with self.connection() as connection:
            return await connection.fetch_all(query, values)

    async def fetch_one(self, query: Union[str, ClauseElement], values: Optional[Dict[str, Any]] = None) -> Optional[Record]:
        async with self.connection() as connection:
            return await connection.fetch_one(query, values)

    async def fetch_val(
        self, 
        query: Union[str, ClauseElement], 
        values: Optional[Dict[str, Any]] = None, 
        column: Union[int, str] = 0
    ) -> Any:
        async with self.connection() as connection:
            return await connection.fetch_val(query, values, column=column)

    async def execute(self, query: Union[str, ClauseElement], values: Optional[Dict[str, Any]] = None) -> Any:
        async with self.connection() as connection:
            return await connection.execute(query, values)

    async def execute_many(self, query: Union[str, ClauseElement], values: List[Dict[str, Any]]) -> None:
        async with self.connection() as connection:
            return await connection.execute_many(query, values)

    async def iterate(
        self, 
        query: Union[str, ClauseElement], 
        values: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Record, None]:
        async with self.connection() as connection:
            async for record in connection.iterate(query, values):
                yield record

    def connection(self) -> Connection:
        if self._global_connection is not None:
            return self._global_connection
        if not self._connection:
            self._connection = Connection(self, self._backend)
        return self._connection

    def transaction(self, *, force_rollback: bool = False, **kwargs: Any) -> Transaction:
        return Transaction(self.connection, force_rollback=force_rollback, **kwargs)

    @contextlib.contextmanager
    def force_rollback(self) -> typing.Iterator[None]:
        initial = self._force_rollback
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
        self._connection: Any = self._backend.connection()
        self._connection_counter: int = 0
        self._transaction_lock: asyncio.Lock = asyncio.Lock()
        self._transaction_stack: List[Transaction] = []
        self._query_lock: asyncio.Lock = asyncio.Lock()

    async def __aenter__(self) -> 'Connection':
        async with self._connection_lock:
            self._connection_counter += 1
            try:
                if self._connection_counter == 1:
                    await self._connection.acquire()
            except BaseException as e:
                self._connection_counter -= 1
                raise e
        return self

    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]] = None, 
        exc_value: Optional[BaseException] = None, 
        traceback: Optional[TracebackType] = None
    ) -> None:
        async with self._connection_lock:
            assert self._connection is not None
            self._connection_counter -= 1
            if self._connection_counter == 0:
                await self._connection.release()
                self._database._connection = None

    async def fetch_all(self, query: Union[str, ClauseElement], values: Optional[Dict[str, Any]] = None) -> List[Record]:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_all(built_query)

    async def fetch_one(self, query: Union[str, ClauseElement], values: Optional[Dict[str, Any]] = None) -> Optional[Record]:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_one(built_query)

    async def fetch_val(
        self, 
        query: Union[str, ClauseElement], 
        values: Optional[Dict[str, Any]] = None, 
        column: Union[int, str] = 0
    ) -> Any:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_val(built_query, column)

    async def execute(self, query: Union[str, ClauseElement], values: Optional[Dict[str, Any]] = None) -> Any:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.execute(built_query)

    async def execute_many(self, query: Union[str, ClauseElement], values: List[Dict[str, Any]]) -> None:
        queries = [self._build_query(query, values_set) for values_set in values]
        async with self._query_lock:
            await self._connection.execute_many(queries)

    async def iterate(
        self, 
        query: Union[str, ClauseElement], 
        values: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Record, None]:
        built_query = self._build_query(query, values)
        async with self.transaction():
            async with self._query_lock:
                async for record in self._connection.iterate(built_query):
                    yield record

    def transaction(self, *, force_rollback: bool = False, **kwargs: Any) -> 'Transaction':
        def connection_callable() -> Connection:
            return self
        return Transaction(connection_callable, force_rollback, **kwargs)

    @property
    def raw_connection(self) -> Any:
        return self._connection.raw_connection

    @staticmethod
    def _build_query(query: Union[str, ClauseElement], values: Optional[Dict[str, Any]] = None) -> ClauseElement:
        if isinstance(query, str):
            query = text(query)
            return query.bindparams(**values) if values is not None else query
        elif values:
            return query.values(**values)
        return query

_CallableType = TypeVar('_CallableType', bound=Callable[..., Any])

class Transaction:
    def __init__(
        self, 
        connection_callable: Callable[[], Connection], 
        force_rollback: bool, 
        **kwargs: Any
    ) -> None:
        self._connection_callable: Callable[[], Connection] = connection_callable
        self._force_rollback: bool = force_rollback
        self._extra_options: Dict[str, Any] = kwargs

    @property
    def _connection(self) -> Connection:
        return self._connection_callable()

    @property
    def _transaction(self) -> Optional[TransactionBackend]:
        transactions = _ACTIVE_TRANSACTIONS.get()
        if transactions is None:
            return None
        return transactions.get(self, None)

    @_transaction.setter
    def _transaction(self, transaction: Optional[TransactionBackend]) -> Optional[TransactionBackend]:
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

    async def __aenter__(self) -> 'Transaction':
        """
        Called when entering `async with database.transaction()`
        """
        await self.start()
        return self

    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]] = None, 
        exc_value: Optional[BaseException] = None, 
        traceback: Optional[TracebackType] = None
    ) -> None:
        """
        Called when exiting `async with database.transaction()`
        """
        if exc_type is not None or self._force_rollback:
            await self.rollback()
        else:
            await self.commit()

    def __await__(self) -> typing.Generator[Any, None, 'Transaction']:
        """
        Called if using the low-level `transaction = await database.transaction()`
        """
        return self.start().__await__()

    def __call__(self, func: _CallableType) -> _CallableType:
        """
        Called if using `@database.transaction()` as a decorator.
        """
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)
        return cast(_CallableType, wrapper)

    async def start(self) -> 'Transaction':
        self._transaction = self._connection._connection.transaction()
        async with self._connection._transaction_lock:
            is_root = not self._connection._transaction_stack
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
            await self._connection.__aexit__()
            self._transaction = None

    async def rollback(self) -> None:
        async with self._connection._transaction_lock:
            assert self._connection._transaction_stack[-1] is self
            self._connection._transaction_stack.pop()
            assert self._transaction is not None
            await self._transaction.rollback()
            await self._connection.__aexit__()
            self._transaction = None

class _EmptyNetloc(str):
    def __bool__(self) -> bool:
        return True

class DatabaseURL:
    def __init__(self, url: Union[str, 'DatabaseURL']) -> None:
        if isinstance(url, DatabaseURL):
            self._url: str = url._url
        elif isinstance(url, str):
            self._url = url
        else:
            raise TypeError(f'Invalid type for DatabaseURL. Expected str or DatabaseURL, got {type(url)}')

    @property
    def components(self) -> SplitResult:
        if not hasattr(self, '_components'):
            self._components: SplitResult = urlsplit(self._url)
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
    def userinfo(self) -> Optional[bytes]:
        if self.components.username:
            info = self.components.username
            if self.components.password:
                info += ':' + self.components.password
            return info.encode('utf-8')
        return None

    @property
    def username(self) -> Optional[str]:
        if self.components.username is None:
            return None
        return unquote(self.components.username)

    @property
    def password(self) -> Optional[str]:
        if self.components.password is None:
            return None
        return unquote(self.components.password)

    @property
    def hostname(self) -> Optional[str]:
        return self.components.hostname or self.options.get('host') or self.options.get('unix_sock')

    @property
    def port(self) -> Optional[int]:
        return self.components.port

    @property
    def netloc(self) -> str:
        return self.components.netloc

    @property
    def database(self) -> str:
        path = self.components.path
        if path.startswith('/'):
            path = path[1:]
        return unquote(path)

    @property
    def options(self) -> Dict[str, str]:
        if not hasattr(self, '_options'):
            self._options: Dict[str, str] = dict(parse_qsl(self.components.query))
        return self._options

    def replace(self, **kwargs: Any) -> 'DatabaseURL':
        if 'username' in kwargs or 'password' in kwargs or 'hostname' in kwargs or ('port' in kwargs):
            hostname = kwargs.pop('hostname', self.hostname)
            port = kwargs.pop('port', self.port)
            username = kwargs.pop('username', self.components.username)
            password = kwargs.pop('password', self.components.password)
            netloc = hostname
            if port is not None:
                netloc += f':{port}'
            if username is not None:
                userpass = username
                if password is not None:
                    userpass += f':{password}'
                netloc = f'{userpass}@{netloc}'
            kwargs['netloc'] = netloc
        if 'database' in kwargs:
            kwargs['path'] = '/' + kwargs.pop('database')
        if 'dialect' in kwargs or 'driver' in kwargs:
            dialect = kwargs.pop('dialect', self.dialect)
            driver = kwargs.pop('driver', self.driver)
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
        if not isinstance(other, (str, DatabaseURL)):
            return NotImplemented
        return str(self) == str(other)
