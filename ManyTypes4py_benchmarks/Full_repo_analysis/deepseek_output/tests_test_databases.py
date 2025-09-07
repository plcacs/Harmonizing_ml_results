import asyncio
import datetime
import decimal
import enum
import functools
import gc
import itertools
import os
import sqlite3
from typing import Any, AsyncGenerator, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union, cast
from unittest.mock import MagicMock, patch
import pytest
import sqlalchemy
from databases import Database, DatabaseURL
from sqlalchemy import Table, Column, Integer, String, Boolean, DateTime, Date, Time, Enum, JSON, Numeric, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select
from sqlalchemy.sql.schema import DropTable, CreateTable

assert 'TEST_DATABASE_URLS' in os.environ, 'TEST_DATABASE_URLS is not set.'
DATABASE_URLS: List[str] = [url.strip() for url in os.environ['TEST_DATABASE_URLS'].split(',')]

class AsyncMock(MagicMock):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super(AsyncMock, self).__call__(*args, **kwargs)

class MyEpochType(sqlalchemy.types.TypeDecorator):
    impl = sqlalchemy.Integer
    epoch = datetime.date(1970, 1, 1)

    def process_bind_param(self, value: Optional[datetime.date], dialect: Any) -> Optional[int]:
        if value is None:
            return None
        return (value - self.epoch).days

    def process_result_value(self, value: Optional[int], dialect: Any) -> Optional[datetime.date]:
        if value is None:
            return None
        return self.epoch + datetime.timedelta(days=value)

metadata: MetaData = sqlalchemy.MetaData()
notes: Table = sqlalchemy.Table('notes', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('text', sqlalchemy.String(length=100)), sqlalchemy.Column('completed', sqlalchemy.Boolean))
articles: Table = sqlalchemy.Table('articles', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('title', sqlalchemy.String(length=100)), sqlalchemy.Column('published', sqlalchemy.DateTime))
events: Table = sqlalchemy.Table('events', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('date', sqlalchemy.Date))
daily_schedule: Table = sqlalchemy.Table('daily_schedule', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('time', sqlalchemy.Time))

class TshirtSize(enum.Enum):
    SMALL = 'SMALL'
    MEDIUM = 'MEDIUM'
    LARGE = 'LARGE'
    XL = 'XL'

class TshirtColor(enum.Enum):
    BLUE = 0
    GREEN = 1
    YELLOW = 2
    RED = 3

tshirt_size: Table = sqlalchemy.Table('tshirt_size', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('size', sqlalchemy.Enum(TshirtSize)), sqlalchemy.Column('color', sqlalchemy.Enum(TshirtColor)))
session: Table = sqlalchemy.Table('session', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('data', sqlalchemy.JSON))
custom_date: Table = sqlalchemy.Table('custom_date', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('title', sqlalchemy.String(length=100)), sqlalchemy.Column('published', MyEpochType))
prices: Table = sqlalchemy.Table('prices', metadata, sqlalchemy.Column('id', sqlalchemy.Integer, primary_key=True), sqlalchemy.Column('price', sqlalchemy.Numeric(precision=30, scale=20)))

@pytest.fixture(autouse=True, scope='function')
def create_test_database() -> Iterator[None]:
    for url in DATABASE_URLS:
        database_url: DatabaseURL = DatabaseURL(url)
        if database_url.scheme in ['mysql', 'mysql+aiomysql', 'mysql+asyncmy']:
            url = str(database_url.replace(driver='pymysql'))
        elif database_url.scheme in ['postgresql+aiopg', 'sqlite+aiosqlite', 'postgresql+asyncpg']:
            url = str(database_url.replace(driver=None))
        engine: Engine = sqlalchemy.create_engine(url)
        metadata.create_all(engine)
    yield
    for url in DATABASE_URLS:
        database_url = DatabaseURL(url)
        if database_url.scheme in ['mysql', 'mysql+aiomysql', 'mysql+asyncmy']:
            url = str(database_url.replace(driver='pymysql'))
        elif database_url.scheme in ['postgresql+aiopg', 'sqlite+aiosqlite', 'postgresql+asyncpg']:
            url = str(database_url.replace(driver=None))
        engine = sqlalchemy.create_engine(url)
        metadata.drop_all(engine)
    gc.collect()

def async_adapter(wrapped_func: Any) -> Any:
    """
    Decorator used to run async test cases.
    """

    @functools.wraps(wrapped_func)
    def run_sync(*args: Any, **kwargs: Any) -> Any:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        task: Any = wrapped_func(*args, **kwargs)
        return loop.run_until_complete(task)
    return run_sync

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_queries(database_url: str) -> None:
    """
    Test that the basic `execute()`, `execute_many()`, `fetch_all()``, and
    `fetch_one()` interfaces are all supported (using SQLAlchemy core).
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: Any = notes.insert()
            values: Dict[str, Any] = {'text': 'example1', 'completed': True}
            await database.execute(query, values)
            query = notes.insert()
            values_list: List[Dict[str, Any]] = [{'text': 'example2', 'completed': False}, {'text': 'example3', 'completed': True}]
            await database.execute_many(query, values_list)
            query = notes.select()
            results: List[Any] = await database.fetch_all(query=query)
            assert len(results) == 3
            assert results[0]['text'] == 'example1'
            assert results[0]['completed'] == True
            assert results[1]['text'] == 'example2'
            assert results[1]['completed'] == False
            assert results[2]['text'] == 'example3'
            assert results[2]['completed'] == True
            query = notes.select()
            result: Optional[Any] = await database.fetch_one(query=query)
            assert result is not None
            assert result['text'] == 'example1'
            assert result['completed'] == True
            query = sqlalchemy.sql.select(*[notes.c.text])
            result_val: Any = await database.fetch_val(query=query)
            assert result_val == 'example1'
            query = sqlalchemy.sql.select(*[notes.c.text]).where(notes.c.text == 'impossible')
            result_val = await database.fetch_val(query=query)
            assert result_val is None
            query = sqlalchemy.sql.select(*[notes.c.id, notes.c.text])
            result_val = await database.fetch_val(query=query, column=1)
            assert result_val == 'example1'
            query = sqlalchemy.sql.select(*[notes.c.text])
            result = await database.fetch_one(query=query)
            assert result is not None
            assert result['text'] == 'example1'
            assert result[0] == 'example1'
            query = notes.select()
            iterate_results: List[Any] = []
            async for result_iter in database.iterate(query=query):
                iterate_results.append(result_iter)
            assert len(iterate_results) == 3
            assert iterate_results[0]['text'] == 'example1'
            assert iterate_results[0]['completed'] == True
            assert iterate_results[1]['text'] == 'example2'
            assert iterate_results[1]['completed'] == False
            assert iterate_results[2]['text'] == 'example3'
            assert iterate_results[2]['completed'] == True

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_queries_raw(database_url: str) -> None:
    """
    Test that the basic `execute()`, `execute_many()`, `fetch_all()``, and
    `fetch_one()` interfaces are all supported (raw queries).
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: str = 'INSERT INTO notes(text, completed) VALUES (:text, :completed)'
            values: Dict[str, Any] = {'text': 'example1', 'completed': True}
            await database.execute(query, values)
            query = 'INSERT INTO notes(text, completed) VALUES (:text, :completed)'
            values_list: List[Dict[str, Any]] = [{'text': 'example2', 'completed': False}, {'text': 'example3', 'completed': True}]
            await database.execute_many(query, values_list)
            query = 'SELECT * FROM notes WHERE completed = :completed'
            results: List[Any] = await database.fetch_all(query=query, values={'completed': True})
            assert len(results) == 2
            assert results[0]['text'] == 'example1'
            assert results[0]['completed'] == True
            assert results[1]['text'] == 'example3'
            assert results[1]['completed'] == True
            query = 'SELECT * FROM notes WHERE completed = :completed'
            result: Optional[Any] = await database.fetch_one(query=query, values={'completed': False})
            assert result is not None
            assert result['text'] == 'example2'
            assert result['completed'] == False
            query = 'SELECT completed FROM notes WHERE text = :text'
            result_val: Any = await database.fetch_val(query=query, values={'text': 'example1'})
            assert result_val == True
            query = 'SELECT * FROM notes WHERE text = :text'
            result_val = await database.fetch_val(query=query, values={'text': 'example1'}, column='completed')
            assert result_val == True
            query = 'SELECT * FROM notes'
            iterate_results: List[Any] = []
            async for result_iter in database.iterate(query=query):
                iterate_results.append(result_iter)
            assert len(iterate_results) == 3
            assert iterate_results[0]['text'] == 'example1'
            assert iterate_results[0]['completed'] == True
            assert iterate_results[1]['text'] == 'example2'
            assert iterate_results[1]['completed'] == False
            assert iterate_results[2]['text'] == 'example3'
            assert iterate_results[2]['completed'] == True

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_ddl_queries(database_url: str) -> None:
    """
    Test that the built-in DDL elements such as `DropTable()`,
    `CreateTable()` are supported (using SQLAlchemy core).
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: DropTable = sqlalchemy.schema.DropTable(notes)
            await database.execute(query)
            query = sqlalchemy.schema.CreateTable(notes)
            await database.execute(query)

@pytest.mark.parametrize('exception', [Exception, asyncio.CancelledError])
@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_queries_after_error(database_url: str, exception: Type[BaseException]) -> None:
    """
    Test that the basic `execute()` works after a previous error.
    """
    async with Database(database_url) as database:
        with patch.object(database.connection()._connection, 'acquire', new=AsyncMock(side_effect=exception)):
            with pytest.raises(exception):
                query: Select = notes.select()
                await database.fetch_all(query)
        query = notes.select()
        await database.fetch_all(query)

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_results_support_mapping_interface(database_url: str) -> None:
    """
    Casting results to a dict should work, since the interface defines them
    as supporting the mapping interface.
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: Any = notes.insert()
            values: Dict[str, Any] = {'text': 'example1', 'completed': True}
            await database.execute(query, values)
            query = notes.select()
            results: List[Any] = await database.fetch_all(query=query)
            results_as_dicts: List[Dict[str, Any]] = [dict(item) for item in results]
            assert len(results[0]) == 3
            assert len(results_as_dicts[0]) == 3
            assert isinstance(results_as_dicts[0]['id'], int)
            assert results_as_dicts[0]['text'] == 'example1'
            assert results_as_dicts[0]['completed'] == True

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_results_support_column_reference(database_url: str) -> None:
    """
    Casting results to a dict should work, since the interface defines them
    as supporting the mapping interface.
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            now: datetime.datetime = datetime.datetime.now().replace(microsecond=0)
            today: datetime.date = datetime.date.today()
            query: Any = articles.insert()
            values: Dict[str, Any] = {'title': 'Hello, world Article', 'published': now}
            await database.execute(query, values)
            query = custom_date.insert()
            values = {'title': 'Hello, world Custom', 'published': today}
            await database.execute(query, values)
            query = sqlalchemy.select(*[articles, custom_date])
            results: List[Any] = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0][articles.c.title] == 'Hello, world Article'
            assert results[0][articles.c.published] == now
            assert results[0][custom_date.c.title] == 'Hello, world Custom'
            assert results[0][custom_date.c.published] == today

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_result_values_allow_duplicate_names(database_url: str) -> None:
    """
    The values of a result should respect when two columns are selected
    with the same name.
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: str = 'SELECT 1 AS id, 2 AS id'
            row: Any = await database.fetch_one(query=query)
            assert list(row._mapping.keys()) == ['id', 'id']
            assert list(row._mapping.values()) == [1, 2]

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_fetch_one_returning_no_results(database_url: str) -> None:
    """
    fetch_one should return `None` when no results match.
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: Select = notes.select()
            result: Optional[Any] = await database.fetch_one(query=query)
            assert result is None

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_execute_return_val(database_url: str) -> None:
    """
    Test using return value from `execute()` to get an inserted primary key.
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: Any = notes.insert()
            values: Dict[str, Any] = {'text': 'example1', 'completed': True}
            pk: Any = await database.execute(query, values)
            assert isinstance(pk, int)
            if database.url.scheme == 'postgresql+aiopg':
                assert pk == 0
            else:
                query = notes.select().where(notes.c.id == pk)
                result: Optional[Any] = await database.fetch_one(query)
                assert result is not None
                assert result['text'] == 'example1'
                assert result['completed'] == True

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_rollback_isolation(database_url: str) -> None:
    """
    Ensure that `database.transaction(force_rollback=True)` provides strict isolation.
    """
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query: Any = notes.insert().values(text='example1', completed=True)
            await database.execute(query)
        query = notes.select()
        results: List[Any] = await database.fetch_all(query=query)
        assert len(results) == 0

@pytest.mark.parametrize('database_url', DATABASE_URLS)
@async_adapter
async def test_rollback_isolation_with_contextmanager(database_url: str) -> None:
    """
    Ensure that `database.force_rollback()` provides strict isolation.
    """
    database: Database = Database(database_url)
    with database.force_rollback():
        async with database:
            query: Any = notes.insert().values(text='example1', completed=True)
            await database.execute(query)
        async with database:
            query = notes.select()
            results: List[Any] = await database.fetch_all(query