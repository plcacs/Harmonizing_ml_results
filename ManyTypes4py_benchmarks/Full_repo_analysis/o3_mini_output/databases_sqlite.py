import logging
import sqlite3
import typing as t
import uuid
from urllib.parse import urlencode
import aiosqlite
from sqlalchemy.dialects.sqlite import pysqlite
from sqlalchemy.engine.cursor import CursorResultMetaData
from sqlalchemy.engine.interfaces import Dialect, ExecutionContext
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.ddl import DDLElement
from databases.backends.common.records import Record, Row, create_column_maps
from databases.core import LOG_EXTRA, DatabaseURL
from databases.interfaces import ConnectionBackend, DatabaseBackend, TransactionBackend

logger = logging.getLogger('databases')


class SQLiteBackend(DatabaseBackend):
    def __init__(self, database_url: str, **options: t.Any) -> None:
        self._database_url: DatabaseURL = DatabaseURL(database_url)
        self._options: t.Any = options
        self._dialect: Dialect = pysqlite.dialect(paramstyle='qmark')
        self._dialect.supports_native_decimal = False
        self._pool: SQLitePool = SQLitePool(self._database_url, **self._options)

    async def connect(self) -> None:
        ...

    async def disconnect(self) -> None:
        if self._pool._memref:
            self._pool._memref = None

    def connection(self) -> "SQLiteConnection":
        return SQLiteConnection(self._pool, self._dialect)


class SQLitePool:
    def __init__(self, url: DatabaseURL, **options: t.Any) -> None:
        self._database: str = url.database
        self._memref: t.Optional[sqlite3.Connection] = None
        if url.options:
            self._database += '?' + urlencode(url.options)
        self._options: t.Any = options
        if url.options and 'cache' in url.options:
            self._memref = sqlite3.connect(self._database, **self._options)

    async def acquire(self) -> aiosqlite.Connection:
        connection: aiosqlite.Connection = await aiosqlite.connect(
            database=self._database, isolation_level=None, **self._options
        )
        await connection.__aenter__()
        return connection

    async def release(self, connection: aiosqlite.Connection) -> None:
        await connection.__aexit__(None, None, None)


class CompilationContext:
    def __init__(self, context: ExecutionContext) -> None:
        self.context: ExecutionContext = context


class SQLiteConnection(ConnectionBackend):
    def __init__(self, pool: SQLitePool, dialect: Dialect) -> None:
        self._pool: SQLitePool = pool
        self._dialect: Dialect = dialect
        self._connection: t.Optional[aiosqlite.Connection] = None

    async def acquire(self) -> None:
        assert self._connection is None, 'Connection is already acquired'
        self._connection = await self._pool.acquire()

    async def release(self) -> None:
        assert self._connection is not None, 'Connection is not acquired'
        await self._pool.release(self._connection)
        self._connection = None

    async def fetch_all(self, query: ClauseElement) -> t.List[Record]:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, result_columns, context) = self._compile(query)
        column_maps = create_column_maps(result_columns)
        dialect = self._dialect
        async with self._connection.execute(query_str, args) as cursor:
            rows = await cursor.fetchall()
            metadata = CursorResultMetaData(context, cursor.description)
            rows = [Row(metadata, metadata._processors, metadata._keymap, row) for row in rows]
            return [Record(row, result_columns, dialect, column_maps) for row in rows]

    async def fetch_one(self, query: ClauseElement) -> t.Optional[Record]:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, result_columns, context) = self._compile(query)
        column_maps = create_column_maps(result_columns)
        dialect = self._dialect
        async with self._connection.execute(query_str, args) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            metadata = CursorResultMetaData(context, cursor.description)
            row = Row(metadata, metadata._processors, metadata._keymap, row)
            return Record(row, result_columns, dialect, column_maps)

    async def execute(self, query: ClauseElement) -> int:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, result_columns, context) = self._compile(query)
        async with self._connection.cursor() as cursor:
            await cursor.execute(query_str, args)
            if cursor.lastrowid == 0:
                return cursor.rowcount
            return cursor.lastrowid

    async def execute_many(self, queries: t.Iterable[ClauseElement]) -> None:
        assert self._connection is not None, 'Connection is not acquired'
        for single_query in queries:
            await self.execute(single_query)

    async def iterate(self, query: ClauseElement) -> t.AsyncGenerator[Record, None]:
        assert self._connection is not None, 'Connection is not acquired'
        (query_str, args, result_columns, context) = self._compile(query)
        column_maps = create_column_maps(result_columns)
        dialect = self._dialect
        async with self._connection.execute(query_str, args) as cursor:
            metadata = CursorResultMetaData(context, cursor.description)
            async for row in cursor:
                record_row = Row(metadata, metadata._processors, metadata._keymap, row)
                yield Record(record_row, result_columns, dialect, column_maps)

    def transaction(self) -> "SQLiteTransaction":
        return SQLiteTransaction(self)

    def _compile(self, query: ClauseElement) -> t.Tuple[str, t.List[t.Any], t.Optional[t.Any], CompilationContext]:
        compiled = query.compile(dialect=self._dialect, compile_kwargs={'render_postcompile': True})
        execution_context: ExecutionContext = self._dialect.execution_ctx_cls()
        execution_context.dialect = self._dialect
        args: t.List[t.Any] = []
        result_map: t.Optional[t.Any] = None
        if not isinstance(query, DDLElement):
            compiled_params = sorted(compiled.params.items())
            params = compiled.construct_params()
            for key in compiled.positiontup:
                raw_val = params[key]
                if key in compiled._bind_processors:
                    val = compiled._bind_processors[key](raw_val)
                else:
                    val = raw_val
                args.append(val)
            execution_context.result_column_struct = (
                compiled._result_columns,
                compiled._ordered_columns,
                compiled._textual_ordered_columns,
                compiled._ad_hoc_textual,
                compiled._loose_column_name_matching,
            )
            mapping = {key: '$' + str(i) for (i, (key, _)) in enumerate(compiled_params, start=1)}
            compiled_query = compiled.string % mapping
            result_map = compiled._result_columns
        else:
            compiled_query = compiled.string
        query_message: str = compiled_query.replace(' \n', ' ').replace('\n', ' ')
        logger.debug('Query: %s Args: %s', query_message, repr(tuple(args)), extra=LOG_EXTRA)
        return (compiled.string, args, result_map, CompilationContext(execution_context))

    @property
    def raw_connection(self) -> aiosqlite.Connection:
        assert self._connection is not None, 'Connection is not acquired'
        return self._connection


class SQLiteTransaction(TransactionBackend):
    def __init__(self, connection: SQLiteConnection) -> None:
        self._connection: SQLiteConnection = connection
        self._is_root: bool = False
        self._savepoint_name: str = ''

    async def start(self, is_root: bool, extra_options: t.Any) -> None:
        assert self._connection._connection is not None, 'Connection is not acquired'
        self._is_root = is_root
        if self._is_root:
            async with self._connection._connection.execute('BEGIN') as cursor:
                await cursor.close()
        else:
            uid: str = str(uuid.uuid4()).replace('-', '_')
            self._savepoint_name = f'STARLETTE_SAVEPOINT_{uid}'
            async with self._connection._connection.execute(f'SAVEPOINT {self._savepoint_name}') as cursor:
                await cursor.close()

    async def commit(self) -> None:
        assert self._connection._connection is not None, 'Connection is not acquired'
        if self._is_root:
            async with self._connection._connection.execute('COMMIT') as cursor:
                await cursor.close()
        else:
            async with self._connection._connection.execute(f'RELEASE SAVEPOINT {self._savepoint_name}') as cursor:
                await cursor.close()

    async def rollback(self) -> None:
        assert self._connection._connection is not None, 'Connection is not acquired'
        if self._is_root:
            async with self._connection._connection.execute('ROLLBACK') as cursor:
                await cursor.close()
        else:
            async with self._connection._connection.execute(f'ROLLBACK TO SAVEPOINT {self._savepoint_name}') as cursor:
                await cursor.close()