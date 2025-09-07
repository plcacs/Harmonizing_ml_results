import getpass
import logging
import typing
import uuid
import asyncmy
from sqlalchemy.dialects.mysql import pymysql
from sqlalchemy.engine.cursor import CursorResultMetaData
from sqlalchemy.engine.interfaces import Dialect, ExecutionContext
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.ddl import DDLElement
from databases.backends.common.records import Record, Row, create_column_maps
from databases.core import LOG_EXTRA, DatabaseURL
from databases.interfaces import ConnectionBackend, DatabaseBackend, Record as RecordInterface, TransactionBackend
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
logger = logging.getLogger('databases')

class AsyncMyBackend(DatabaseBackend):

    def __init__(self, database_url: Union[str, DatabaseURL], **options: Any) -> None:
        self._database_url = DatabaseURL(database_url)
        self._options = options
        self._dialect: Dialect = pymysql.dialect(paramstyle='pyformat')
        self._dialect.supports_native_decimal = True
        self._pool: Optional[asyncmy.Pool] = None

    def _get_connection_kwargs(self) -> Dict[str, Any]:
        url_options = self._database_url.options
        kwargs: Dict[str, Any] = {}
        min_size = url_options.get('min_size')
        max_size = url_options.get('max_size')
        pool_recycle = url_options.get('pool_recycle')
        ssl = url_options.get('ssl')
        unix_socket = url_options.get('unix_socket')
        if min_size is not None:
            kwargs['minsize'] = int(min_size)
        if max_size is not None:
            kwargs['maxsize'] = int(max_size)
        if pool_recycle is not None:
            kwargs['pool_recycle'] = int(pool_recycle)
        if ssl is not None:
            kwargs['ssl'] = {'true': True, 'false': False}[ssl.lower()]
        if unix_socket is not None:
            kwargs['unix_socket'] = unix_socket
        for (key, value) in self._options.items():
            if key == 'min_size':
                key = 'minsize'
            elif key == 'max_size':
                key = 'maxsize'
            kwargs[key] = value
        return kwargs

    async def connect(self) -> None:
        assert self._pool is None, 'DatabaseBackend is already running'
        kwargs = self._get_connection_kwargs()
        self._pool = await asyncmy.create_pool(host=self._database_url.hostname, port=self._database_url.port or 3306, user=self._database_url.username or getpass.getuser(), password=self._database_url.password, db=self._database_url.database, autocommit=True, **kwargs)

    async def disconnect(self) -> None:
        assert self._pool is not None, 'DatabaseBackend is not running'
        self._pool.close()
        await self._pool.wait_closed()
        self._pool = None

    def connection(self) -> "AsyncMyConnection":
        return AsyncMyConnection(self, self._dialect)

class CompilationContext:

    def __init__(self, context: ExecutionContext) -> None:
        self.context = context

class AsyncMyConnection(ConnectionBackend):

    def __init__(self, database: AsyncMyBackend, dialect: Dialect) -> None:
        self._database = database
        self._dialect = dialect
        self._connection: Optional[asyncmy.Connection] = None

    async def acquire(self) -> None:
        assert self._connection is None, 'Connection is already acquired'
        assert self._database._pool is not None, 'DatabaseBackend is not running'
        self._connection = await self._database._pool.acquire()

    async def release(self) -> None:
        assert self._connection is not None, 'Connection is not acquired'
        assert self._database._pool is not None, 'DatabaseBackend is not running'
        await self._database._pool.release(self._connection)
        self._connection = None

    async def fetch_all(self, query: ClauseElement) -> List[RecordInterface]:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, result_columns, context) = self._compile(query)
        column_maps = create_column_maps(result_columns)
        dialect = self._dialect
        async with self._connection.cursor() as cursor:
            try:
                await cursor.execute(query_str, args)
                rows = await cursor.fetchall()
                metadata = CursorResultMetaData(context, cursor.description)
                rows = [Row(metadata, metadata._processors, metadata._keymap, row) for row in rows]
                return [Record(row, result_columns, dialect, column_maps) for row in rows]
            finally:
                await cursor.close()

    async def fetch_one(self, query: ClauseElement) -> Optional[RecordInterface]:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, result_columns, context) = self._compile(query)
        column_maps = create_column_maps(result_columns)
        dialect = self._dialect
        async with self._connection.cursor() as cursor:
            try:
                await cursor.execute(query_str, args)
                row = await cursor.fetchone()
                if row is None:
                    return None
                metadata = CursorResultMetaData(context, cursor.description)
                row = Row(metadata, metadata._processors, metadata._keymap, row)
                return Record(row, result_columns, dialect, column_maps)
            finally:
                await cursor.close()

    async def execute(self, query: ClauseElement) -> Union[int, Any]:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, _, _) = self._compile(query)
        async with self._connection.cursor() as cursor:
            try:
                await cursor.execute(query_str, args)
                if cursor.lastrowid == 0:
                    return cursor.rowcount
                return cursor.lastrowid
            finally:
                await cursor.close()

    async def execute_many(self, queries: List[ClauseElement]) -> None:
        assert self._connection is not None, 'Connection is not acquired'
        async with self._connection.cursor() as cursor:
            try:
                for single_query in queries:
                    (single_query, args, _, _) = self._compile(single_query)
                    await cursor.execute(single_query, args)
            finally:
                await cursor.close()

    async def iterate(self, query: ClauseElement) -> AsyncGenerator[RecordInterface, None]:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, result_columns, context) = self._compile(query)
        column_maps = create_column_maps(result_columns)
        dialect = self._dialect
        async with self._connection.cursor() as cursor:
            try:
                await cursor.execute(query_str, args)
                metadata = CursorResultMetaData(context, cursor.description)
                async for row in cursor:
                    record = Row(metadata, metadata._processors, metadata._keymap, row)
                    yield Record(record, result_columns, dialect, column_maps)
            finally:
                await cursor.close()

    def transaction(self) -> "AsyncMyTransaction":
        return AsyncMyTransaction(self)

    def _compile(self, query: ClauseElement) -> Tuple[str, Dict[str, Any], Optional[Any], CompilationContext]:
        compiled = query.compile(dialect=self._dialect, compile_kwargs={'render_postcompile': True})
        execution_context = self._dialect.execution_ctx_cls()
        execution_context.dialect = self._dialect
        if not isinstance(query, DDLElement):
            compiled_params = sorted(compiled.params.items())
            args = compiled.construct_params()
            for (key, val) in args.items():
                if key in compiled._bind_processors:
                    args[key] = compiled._bind_processors[key](val)
            execution_context.result_column_struct = (compiled._result_columns, compiled._ordered_columns, compiled._textual_ordered_columns, compiled._ad_hoc_textual, compiled._loose_column_name_matching)
            mapping = {key: '$' + str(i) for (i, (key, _)) in enumerate(compiled_params, start=1)}
            compiled_query = compiled.string % mapping
            result_map = compiled._result_columns
        else:
            args = {}
            result_map = None
            compiled_query = compiled.string
        query_message = compiled_query.replace(' \n', ' ').replace('\n', ' ')
        logger.debug('Query: %s Args: %s', query_message, repr(tuple(args)), extra=LOG_EXTRA)
        return (compiled.string, args, result_map, CompilationContext(execution_context))

    @property
    def raw_connection(self) -> asyncmy.Connection:
        assert self._connection is not None, 'Connection is not acquired'
        return self._connection

class AsyncMyTransaction(TransactionBackend):

    def __init__(self, connection: AsyncMyConnection) -> None:
        self._connection = connection
        self._is_root = False
        self._savepoint_name = ''

    async def start(self, is_root: bool, extra_options: Dict[str, Any]) -> None:
        assert self._connection._connection is not None, 'Connection is not acquired'
        self._is_root = is_root
        if self._is_root:
            await self._connection._connection.begin()
        else:
            id = str(uuid.uuid4()).replace('-', '_')
            self._savepoint_name = f'STARLETTE_SAVEPOINT_{id}'
            async with self._connection._connection.cursor() as cursor:
                try:
                    await cursor.execute(f'SAVEPOINT {self._savepoint_name}')
                finally:
                    await cursor.close()

    async def commit(self) -> None:
        assert self._connection._connection is not None, 'Connection is not acquired'
        if self._is_root:
            await self._connection._connection.commit()
        else:
            async with self._connection._connection.cursor() as cursor:
                try:
                    await cursor.execute(f'RELEASE SAVEPOINT {self._savepoint_name}')
                finally:
                    await cursor.close()

    async def rollback(self) -> None:
        assert self._connection._connection is not None, 'Connection is not acquired'
        if self._is_root:
            await self._connection._connection.rollback()
        else:
            async with self._connection._connection.cursor() as cursor:
                try:
                    await cursor.execute(f'ROLLBACK TO SAVEPOINT {self._savepoint_name}')
                finally:
                    await cursor.close()
