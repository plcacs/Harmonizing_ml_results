#!/usr/bin/env python3
import asyncio
import datetime
import decimal
import enum
import functools
import gc
import itertools
import os
import sqlite3
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Generator, Iterable, List, MutableMapping, Sequence
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy
from databases import Database, DatabaseURL

assert "TEST_DATABASE_URLS" in os.environ, "TEST_DATABASE_URLS is not set."
DATABASE_URLS: List[str] = [url.strip() for url in os.environ["TEST_DATABASE_URLS"].split(",")]


class AsyncMock(MagicMock):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super(AsyncMock, self).__call__(*args, **kwargs)


class MyEpochType(sqlalchemy.types.TypeDecorator):
    impl = sqlalchemy.Integer
    epoch = datetime.date(1970, 1, 1)

    def process_bind_param(self, value: datetime.date, dialect: Any) -> int:
        return (value - self.epoch).days

    def process_result_value(self, value: int, dialect: Any) -> datetime.date:
        return self.epoch + datetime.timedelta(days=value)


metadata = sqlalchemy.MetaData()
notes = sqlalchemy.Table(
    "notes",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("text", sqlalchemy.String(length=100)),
    sqlalchemy.Column("completed", sqlalchemy.Boolean),
)
articles = sqlalchemy.Table(
    "articles",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("title", sqlalchemy.String(length=100)),
    sqlalchemy.Column("published", sqlalchemy.DateTime),
)
events = sqlalchemy.Table(
    "events",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("date", sqlalchemy.Date),
)
daily_schedule = sqlalchemy.Table(
    "daily_schedule",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("time", sqlalchemy.Time),
)


class TshirtSize(enum.Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    XL = "XL"


class TshirtColor(enum.Enum):
    BLUE = 0
    GREEN = 1
    YELLOW = 2
    RED = 3


tshirt_size = sqlalchemy.Table(
    "tshirt_size",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("size", sqlalchemy.Enum(TshirtSize)),
    sqlalchemy.Column("color", sqlalchemy.Enum(TshirtColor)),
)
session = sqlalchemy.Table(
    "session",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("data", sqlalchemy.JSON),
)
custom_date = sqlalchemy.Table(
    "custom_date",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("title", sqlalchemy.String(length=100)),
    sqlalchemy.Column("published", MyEpochType),
)
prices = sqlalchemy.Table(
    "prices",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("price", sqlalchemy.Numeric(precision=30, scale=20)),
)


@pytest.fixture(autouse=True, scope="function")
def create_test_database() -> Generator[None, None, None]:
    for url in DATABASE_URLS:
        database_url = DatabaseURL(url)
        if database_url.scheme in ["mysql", "mysql+aiomysql", "mysql+asyncmy"]:
            url = str(database_url.replace(driver="pymysql"))
        elif database_url.scheme in ["postgresql+aiopg", "sqlite+aiosqlite", "postgresql+asyncpg"]:
            url = str(database_url.replace(driver=None))
        engine = sqlalchemy.create_engine(url)
        metadata.create_all(engine)
    yield
    for url in DATABASE_URLS:
        database_url = DatabaseURL(url)
        if database_url.scheme in ["mysql", "mysql+aiomysql", "mysql+asyncmy"]:
            url = str(database_url.replace(driver="pymysql"))
        elif database_url.scheme in ["postgresql+aiopg", "sqlite+aiosqlite", "postgresql+asyncpg"]:
            url = str(database_url.replace(driver=None))
        engine = sqlalchemy.create_engine(url)
        metadata.drop_all(engine)
    gc.collect()


def async_adapter(
    wrapped_func: Callable[..., Awaitable[Any]]
) -> Callable[..., Any]:
    """
    Decorator used to run async test cases.
    """
    @functools.wraps(wrapped_func)
    def run_sync(*args: Any, **kwargs: Any) -> Any:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        task = wrapped_func(*args, **kwargs)
        return loop.run_until_complete(task)
    return run_sync


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_queries(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = notes.insert()
            values: Dict[str, Any] = {"text": "example1", "completed": True}
            await database.execute(query, values)
            query = notes.insert()
            values_list: List[Dict[str, Any]] = [
                {"text": "example2", "completed": False},
                {"text": "example3", "completed": True},
            ]
            await database.execute_many(query, values_list)
            query = notes.select()
            results: List[Dict[str, Any]] = await database.fetch_all(query=query)
            assert len(results) == 3
            assert results[0]["text"] == "example1"
            assert results[0]["completed"] is True
            assert results[1]["text"] == "example2"
            assert results[1]["completed"] is False
            assert results[2]["text"] == "example3"
            assert results[2]["completed"] is True
            query = notes.select()
            result: Dict[str, Any] = await database.fetch_one(query=query)
            assert result["text"] == "example1"
            assert result["completed"] is True
            query = sqlalchemy.sql.select(*[notes.c.text])
            result_val: Any = await database.fetch_val(query=query)
            assert result_val == "example1"
            query = sqlalchemy.sql.select(*[notes.c.text]).where(notes.c.text == "impossible")
            result_val = await database.fetch_val(query=query)
            assert result_val is None
            query = sqlalchemy.sql.select(*[notes.c.id, notes.c.text])
            result_val = await database.fetch_val(query=query, column=1)
            assert result_val == "example1"
            query = sqlalchemy.sql.select(*[notes.c.text])
            result = await database.fetch_one(query=query)
            assert result["text"] == "example1"
            assert result[0] == "example1"
            query = notes.select()
            iterate_results: List[Dict[str, Any]] = []
            async for result in database.iterate(query=query):
                iterate_results.append(result)
            assert len(iterate_results) == 3
            assert iterate_results[0]["text"] == "example1"
            assert iterate_results[0]["completed"] is True
            assert iterate_results[1]["text"] == "example2"
            assert iterate_results[1]["completed"] is False
            assert iterate_results[2]["text"] == "example3"
            assert iterate_results[2]["completed"] is True


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_queries_raw(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query_raw: str = "INSERT INTO notes(text, completed) VALUES (:text, :completed)"
            values: Dict[str, Any] = {"text": "example1", "completed": True}
            await database.execute(query_raw, values)
            query_raw = "INSERT INTO notes(text, completed) VALUES (:text, :completed)"
            values_list: List[Dict[str, Any]] = [
                {"text": "example2", "completed": False},
                {"text": "example3", "completed": True},
            ]
            await database.execute_many(query_raw, values_list)
            query_raw = "SELECT * FROM notes WHERE completed = :completed"
            results = await database.fetch_all(query=query_raw, values={"completed": True})
            assert len(results) == 2
            assert results[0]["text"] == "example1"
            assert results[0]["completed"] is True
            assert results[1]["text"] == "example3"
            assert results[1]["completed"] is True
            query_raw = "SELECT * FROM notes WHERE completed = :completed"
            result = await database.fetch_one(query=query_raw, values={"completed": False})
            assert result["text"] == "example2"
            assert result["completed"] is False
            query_raw = "SELECT completed FROM notes WHERE text = :text"
            result = await database.fetch_val(query=query_raw, values={"text": "example1"})
            assert result is True
            query_raw = "SELECT * FROM notes WHERE text = :text"
            result = await database.fetch_val(query=query_raw, values={"text": "example1"}, column="completed")
            assert result is True
            query_raw = "SELECT * FROM notes"
            iterate_results: List[Dict[str, Any]] = []
            async for result in database.iterate(query=query_raw):
                iterate_results.append(result)
            assert len(iterate_results) == 3
            assert iterate_results[0]["text"] == "example1"
            assert iterate_results[0]["completed"] is True
            assert iterate_results[1]["text"] == "example2"
            assert iterate_results[1]["completed"] is False
            assert iterate_results[2]["text"] == "example3"
            assert iterate_results[2]["completed"] is True


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_ddl_queries(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = sqlalchemy.schema.DropTable(notes)
            await database.execute(query)
            query = sqlalchemy.schema.CreateTable(notes)
            await database.execute(query)


@pytest.mark.parametrize("exception", [Exception, asyncio.CancelledError])
@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_queries_after_error(database_url: str, exception: Exception) -> None:
    async with Database(database_url) as database:
        with patch.object(database.connection()._connection, "acquire", new=AsyncMock(side_effect=exception)):
            with pytest.raises(exception):
                query = notes.select()
                await database.fetch_all(query)
        query = notes.select()
        await database.fetch_all(query)


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_results_support_mapping_interface(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = notes.insert()
            values: Dict[str, Any] = {"text": "example1", "completed": True}
            await database.execute(query, values)
            query = notes.select()
            results = await database.fetch_all(query=query)
            results_as_dicts: List[Dict[str, Any]] = [dict(item) for item in results]
            assert len(results[0]) == 3
            assert len(results_as_dicts[0]) == 3
            assert isinstance(results_as_dicts[0]["id"], int)
            assert results_as_dicts[0]["text"] == "example1"
            assert results_as_dicts[0]["completed"] is True


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_results_support_column_reference(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            now: datetime.datetime = datetime.datetime.now().replace(microsecond=0)
            today: datetime.date = datetime.date.today()
            query = articles.insert()
            values: Dict[str, Any] = {"title": "Hello, world Article", "published": now}
            await database.execute(query, values)
            query = custom_date.insert()
            values = {"title": "Hello, world Custom", "published": today}
            await database.execute(query, values)
            query = sqlalchemy.select(*[articles, custom_date])
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0][articles.c.title] == "Hello, world Article"
            assert results[0][articles.c.published] == now
            assert results[0][custom_date.c.title] == "Hello, world Custom"
            assert results[0][custom_date.c.published] == today


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_result_values_allow_duplicate_names(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = "SELECT 1 AS id, 2 AS id"
            row = await database.fetch_one(query=query)
            assert list(row._mapping.keys()) == ["id", "id"]
            assert list(row._mapping.values()) == [1, 2]


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_fetch_one_returning_no_results(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = notes.select()
            result = await database.fetch_one(query=query)
            assert result is None


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_execute_return_val(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = notes.insert()
            values: Dict[str, Any] = {"text": "example1", "completed": True}
            pk: int = await database.execute(query, values)
            assert isinstance(pk, int)
            if database.url.scheme == "postgresql+aiopg":
                assert pk == 0
            else:
                query = notes.select().where(notes.c.id == pk)
                result = await database.fetch_one(query)
                assert result["text"] == "example1"
                assert result["completed"] is True


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_rollback_isolation(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = notes.insert().values(text="example1", completed=True)
            await database.execute(query)
        query = notes.select()
        results = await database.fetch_all(query=query)
        assert len(results) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_rollback_isolation_with_contextmanager(database_url: str) -> None:
    database = Database(database_url)
    with database.force_rollback():
        async with database:
            query = notes.insert().values(text="example1", completed=True)
            await database.execute(query)
        async with database:
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_commit(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            async with database.transaction():
                query = notes.insert().values(text="example1", completed=True)
                await database.execute(query)
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_context_child_task_inheritance(database_url: str) -> None:
    async with Database(database_url) as database:

        async def check_transaction(transaction: Any, active_transaction: Any) -> None:
            assert transaction._transaction is active_transaction

        async with database.transaction() as transaction:
            await asyncio.create_task(check_transaction(transaction, transaction._transaction))


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_context_child_task_inheritance_example(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction():
            await database.execute(notes.insert().values(id=1, text="setup", completed=True))
            await database.execute(notes.update().where(notes.c.id == 1).values(text="prior"))
            result = await database.fetch_one(notes.select().where(notes.c.id == 1))
            assert result.text == "prior"

            async def run_update_from_child_task(connection: Any) -> None:
                await connection.execute(notes.update().where(notes.c.id == 1).values(text="test"))
            await asyncio.create_task(run_update_from_child_task(database.connection()))
            result = await database.fetch_one(notes.select().where(notes.c.id == 1))
            assert result.text == "test"


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_context_sibling_task_isolation(database_url: str) -> None:
    start: asyncio.Event = asyncio.Event()
    end: asyncio.Event = asyncio.Event()
    async with Database(database_url) as database:

        async def check_transaction(transaction: Any) -> None:
            await start.wait()
            assert transaction._transaction is None
            end.set()
        transaction = database.transaction()
        assert transaction._transaction is None
        task = asyncio.create_task(check_transaction(transaction))
        async with transaction:
            start.set()
            assert transaction._transaction is not None
            await end.wait()
        await task


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_context_sibling_task_isolation_example(database_url: str) -> None:
    setup: asyncio.Event = asyncio.Event()
    done: asyncio.Event = asyncio.Event()

    async def tx1(connection: Database) -> None:
        async with connection.transaction():
            await connection.execute(notes.insert(), values={"id": 1, "text": "tx1", "completed": False})
            setup.set()
            await done.wait()

    async def tx2(connection: Database) -> None:
        async with connection.transaction():
            await setup.wait()
            result = await connection.fetch_all(notes.select())
            assert result == [], result
            done.set()
    async with Database(database_url) as db:
        await asyncio.gather(tx1(db), tx2(db))


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connection_cleanup_contextmanager(database_url: str) -> None:
    ready: asyncio.Event = asyncio.Event()
    done: asyncio.Event = asyncio.Event()

    async def check_child_connection(database: Database) -> None:
        async with database.connection():
            ready.set()
            await done.wait()
    async with Database(database_url) as database:
        connection = database.connection()
        assert isinstance(database._connection_map, MutableMapping)
        assert database._connection_map.get(asyncio.current_task()) is connection
        task = asyncio.create_task(check_child_connection(database))
        await ready.wait()
        assert database._connection_map.get(task) is not None
        assert database._connection_map.get(task) is not connection
        done.set()
        await task
    assert isinstance(database._connection_map, MutableMapping)
    assert len(database._connection_map) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connection_cleanup_garbagecollector(database_url: str) -> None:
    database = Database(database_url)
    await database.connect()
    created: asyncio.Event = asyncio.Event()

    async def check_child_connection(database: Database) -> None:
        database.connection()
        created.set()
    task = asyncio.create_task(check_child_connection(database))
    await created.wait()
    assert task in database._connection_map
    await task
    del task
    gc.collect()
    assert len(database._connection_map) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_context_cleanup_contextmanager(database_url: str) -> None:
    from databases.core import _ACTIVE_TRANSACTIONS
    assert _ACTIVE_TRANSACTIONS.get() is None
    async with Database(database_url) as database:
        async with database.transaction() as transaction:
            open_transactions: MutableMapping = _ACTIVE_TRANSACTIONS.get()  # type: ignore
            assert isinstance(open_transactions, MutableMapping)
            assert open_transactions.get(transaction) is transaction._transaction
        open_transactions = _ACTIVE_TRANSACTIONS.get()  # type: ignore
        assert isinstance(open_transactions, MutableMapping)
        assert open_transactions.get(transaction, None) is None


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_context_cleanup_garbagecollector(database_url: str) -> None:
    from databases.core import _ACTIVE_TRANSACTIONS
    assert _ACTIVE_TRANSACTIONS.get() is None
    async with Database(database_url) as database:
        transaction = database.transaction()
        await transaction.start()
        open_transactions: MutableMapping = _ACTIVE_TRANSACTIONS.get()  # type: ignore
        assert isinstance(open_transactions, MutableMapping)
        assert open_transactions.get(transaction) is transaction._transaction
        del transaction
        gc.collect()
        assert len(open_transactions) == 1
        transaction = database.connection()._transaction_stack[-1]
        await transaction.rollback()
        del transaction
        assert len(open_transactions) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_commit_serializable(database_url: str) -> None:
    database_url_obj: DatabaseURL = DatabaseURL(database_url)
    if database_url_obj.scheme not in ["postgresql", "postgresql+asyncpg"]:
        pytest.skip("Test (currently) only supports asyncpg")
    if database_url_obj.scheme == "postgresql+asyncpg":
        database_url_obj = database_url_obj.replace(driver=None)

    def insert_independently() -> None:
        engine = sqlalchemy.create_engine(str(database_url_obj))
        conn = engine.connect()
        query = notes.insert().values(text="example1", completed=True)
        conn.execute(query)
        conn.close()

    def delete_independently() -> None:
        engine = sqlalchemy.create_engine(str(database_url_obj))
        conn = engine.connect()
        query = notes.delete()
        conn.execute(query)
        conn.close()

    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True, isolation="serializable"):
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 0
            insert_independently()
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 0
            delete_independently()


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_rollback(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            try:
                async with database.transaction():
                    query = notes.insert().values(text="example1", completed=True)
                    await database.execute(query)
                    raise RuntimeError()
            except RuntimeError:
                pass
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_commit_low_level(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            transaction = await database.transaction()
            try:
                query = notes.insert().values(text="example1", completed=True)
                await database.execute(query)
            except Exception:
                await transaction.rollback()
            else:
                await transaction.commit()
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_rollback_low_level(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            transaction = await database.transaction()
            try:
                query = notes.insert().values(text="example1", completed=True)
                await database.execute(query)
                raise RuntimeError()
            except Exception:
                await transaction.rollback()
            else:
                await transaction.commit()
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_decorator(database_url: str) -> None:
    database = Database(database_url, force_rollback=True)

    @database.transaction()
    async def insert_data(raise_exception: bool) -> None:
        query = notes.insert().values(text="example", completed=True)
        await database.execute(query)
        if raise_exception:
            raise RuntimeError()

    async with database:
        with pytest.raises(RuntimeError):
            await insert_data(raise_exception=True)
        results = await database.fetch_all(query=notes.select())
        assert len(results) == 0
        await insert_data(raise_exception=False)
        results = await database.fetch_all(query=notes.select())
        assert len(results) == 1


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_transaction_decorator_concurrent(database_url: str) -> None:
    database = Database(database_url)

    @database.transaction()
    async def insert_data() -> None:
        await database.execute(query=notes.insert().values(text="example", completed=True))

    async with database:
        await asyncio.gather(insert_data(), insert_data(), insert_data(), insert_data(), insert_data(), insert_data())
        results = await database.fetch_all(query=notes.select())
        assert len(results) == 6


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_datetime_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            now: datetime.datetime = datetime.datetime.now().replace(microsecond=0)
            query = articles.insert()
            values: Dict[str, Any] = {"title": "Hello, world", "published": now}
            await database.execute(query, values)
            query = articles.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0]["title"] == "Hello, world"
            assert results[0]["published"] == now


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_date_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            now: datetime.date = datetime.date.today()
            query = events.insert()
            values: Dict[str, Any] = {"date": now}
            await database.execute(query, values)
            query = events.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0]["date"] == now


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_time_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            now: datetime.time = datetime.datetime.now().time().replace(microsecond=0)
            query = daily_schedule.insert()
            values: Dict[str, Any] = {"time": now}
            await database.execute(query, values)
            query = daily_schedule.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0]["time"] == now


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_decimal_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            price: decimal.Decimal = decimal.Decimal("0.700000000000001")
            query = prices.insert()
            values: Dict[str, Any] = {"price": price}
            await database.execute(query, values)
            query = prices.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            if database_url.startswith("sqlite"):
                assert results[0]["price"] == pytest.approx(price)
            else:
                assert results[0]["price"] == price


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_enum_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            size: TshirtSize = TshirtSize.SMALL
            color: TshirtColor = TshirtColor.GREEN
            values: Dict[str, Any] = {"size": size, "color": color}
            query = tshirt_size.insert()
            await database.execute(query, values)
            query = tshirt_size.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0]["size"] == size
            assert results[0]["color"] == color


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_json_dict_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            data: Dict[str, Any] = {"text": "hello", "boolean": True, "int": 1}
            values: Dict[str, Any] = {"data": data}
            query = session.insert()
            await database.execute(query, values)
            query = session.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0]["data"] == {"text": "hello", "boolean": True, "int": 1}


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_json_list_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            data: List[str] = ["lemon", "raspberry", "lime", "pumice"]
            values: Dict[str, Any] = {"data": data}
            query = session.insert()
            await database.execute(query, values)
            query = session.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0]["data"] == ["lemon", "raspberry", "lime", "pumice"]


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_custom_field(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            today: datetime.date = datetime.date.today()
            query = custom_date.insert()
            values: Dict[str, Any] = {"title": "Hello, world", "published": today}
            await database.execute(query, values)
            query = custom_date.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
            assert results[0]["title"] == "Hello, world"
            assert results[0]["published"] == today


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connections_isolation(database_url: str) -> None:
    async with Database(database_url) as database:
        try:
            query = notes.insert().values(text="example1", completed=True)
            await database.execute(query)
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
        finally:
            query = notes.delete()
            await database.execute(query)


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_commit_on_root_transaction(database_url: str) -> None:
    async with Database(database_url) as database:
        try:
            async with database.transaction():
                query = notes.insert().values(text="example1", completed=True)
                await database.execute(query)
            query = notes.select()
            results = await database.fetch_all(query=query)
            assert len(results) == 1
        finally:
            query = notes.delete()
            await database.execute(query)


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connect_and_disconnect(database_url: str) -> None:
    database = Database(database_url)
    assert not database.is_connected
    await database.connect()
    assert database.is_connected
    await database.disconnect()
    assert not database.is_connected
    await database.connect()
    await database.connect()
    assert database.is_connected
    await database.disconnect()
    await database.disconnect()
    assert not database.is_connected


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connection_context_same_task(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.connection() as connection_1:
            async with database.connection() as connection_2:
                assert connection_1 is connection_2


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connection_context_multiple_sibling_tasks(database_url: str) -> None:
    async with Database(database_url) as database:
        connection_1: Any = None
        connection_2: Any = None
        test_complete: asyncio.Event = asyncio.Event()

        async def get_connection_1() -> None:
            nonlocal connection_1
            async with database.connection() as connection:
                connection_1 = connection
                await test_complete.wait()

        async def get_connection_2() -> None:
            nonlocal connection_2
            async with database.connection() as connection:
                connection_2 = connection
                await test_complete.wait()

        task_1 = asyncio.create_task(get_connection_1())
        task_2 = asyncio.create_task(get_connection_2())
        while connection_1 is None or connection_2 is None:
            await asyncio.sleep(1e-06)
        assert connection_1 is not connection_2
        test_complete.set()
        await task_1
        await task_2


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connection_context_multiple_tasks(database_url: str) -> None:
    async with Database(database_url) as database:
        parent_connection = database.connection()
        connection_1: Any = None
        connection_2: Any = None
        task_1_ready: asyncio.Event = asyncio.Event()
        task_2_ready: asyncio.Event = asyncio.Event()
        test_complete: asyncio.Event = asyncio.Event()

        async def get_connection_1() -> None:
            nonlocal connection_1
            async with database.connection() as connection:
                connection_1 = connection
                task_1_ready.set()
                await test_complete.wait()

        async def get_connection_2() -> None:
            nonlocal connection_2
            async with database.connection() as connection:
                connection_2 = connection
                task_2_ready.set()
                await test_complete.wait()

        task_1 = asyncio.create_task(get_connection_1())
        task_2 = asyncio.create_task(get_connection_2())
        await task_1_ready.wait()
        await task_2_ready.wait()
        assert connection_1 is not parent_connection
        assert connection_2 is not parent_connection
        assert connection_1 is not connection_2
        test_complete.set()
        await task_1
        await task_2


@pytest.mark.parametrize(
    "database_url1,database_url2",
    (
        pytest.param(db1, db2, id=f"{db1} | {db2}")
        for (db1, db2) in itertools.combinations(DATABASE_URLS, 2)
    ),
)
@async_adapter
async def test_connection_context_multiple_databases(database_url1: str, database_url2: str) -> None:
    async with Database(database_url1) as database1:
        async with Database(database_url2) as database2:
            assert database1.connection() is not database2.connection()


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_connection_context_with_raw_connection(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.connection() as connection_1:
            async with database.connection() as connection_2:
                assert connection_1 is connection_2
                assert connection_1.raw_connection is connection_2.raw_connection


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_queries_with_expose_backend_connection(database_url: str) -> None:
    async with Database(database_url) as database:
        async with database.connection() as connection:
            async with connection.transaction(force_rollback=True):
                raw_connection = connection.raw_connection
                if database.url.scheme in ["mysql", "mysql+asyncmy", "mysql+aiomysql", "postgresql+aiopg"]:
                    insert_query: str = "INSERT INTO notes (text, completed) VALUES (%s, %s)"
                else:
                    insert_query = "INSERT INTO notes (text, completed) VALUES ($1, $2)"
                values: Sequence[Any] = ("example1", True)
                if database.url.scheme in ["mysql", "mysql+aiomysql", "postgresql+aiopg"]:
                    cursor = await raw_connection.cursor()
                    await cursor.execute(insert_query, values)
                elif database.url.scheme == "mysql+asyncmy":
                    async with raw_connection.cursor() as cursor:
                        await cursor.execute(insert_query, values)
                elif database.url.scheme in ["postgresql", "postgresql+asyncpg"]:
                    await raw_connection.execute(insert_query, *values)
                elif database.url.scheme in ["sqlite", "sqlite+aiosqlite"]:
                    await raw_connection.execute(insert_query, values)
                values_list = [("example2", False), ("example3", True)]
                if database.url.scheme in ["mysql", "mysql+aiomysql"]:
                    cursor = await raw_connection.cursor()
                    await cursor.executemany(insert_query, values_list)
                elif database.url.scheme == "mysql+asyncmy":
                    async with raw_connection.cursor() as cursor:
                        await cursor.executemany(insert_query, values_list)
                elif database.url.scheme == "postgresql+aiopg":
                    cursor = await raw_connection.cursor()
                    for value in values_list:
                        await cursor.execute(insert_query, value)
                else:
                    await raw_connection.executemany(insert_query, values_list)
                select_query: str = "SELECT notes.id, notes.text, notes.completed FROM notes"
                if database.url.scheme in ["mysql", "mysql+aiomysql", "postgresql+aiopg"]:
                    cursor = await raw_connection.cursor()
                    await cursor.execute(select_query)
                    results = await cursor.fetchall()
                elif database.url.scheme == "mysql+asyncmy":
                    async with raw_connection.cursor() as cursor:
                        await cursor.execute(select_query)
                        results = await cursor.fetchall()
                elif database.url.scheme in ["postgresql", "postgresql+asyncpg"]:
                    results = await raw_connection.fetch(select_query)
                elif database.url.scheme in ["sqlite", "sqlite+aiosqlite"]:
                    results = await raw_connection.execute_fetchall(select_query)
                assert len(results) == 3
                assert results[0][1] == "example1"
                assert results[0][2] is True
                assert results[1][1] == "example2"
                assert results[1][2] is False
                assert results[2][1] == "example3"
                assert results[2][2] is True
                if database.url.scheme in ["postgresql", "postgresql+asyncpg"]:
                    result = await raw_connection.fetchrow(select_query)
                elif database.url.scheme == "mysql+asyncmy":
                    async with raw_connection.cursor() as cursor:
                        await cursor.execute(select_query)
                        result = await cursor.fetchone()
                else:
                    cursor = await raw_connection.cursor()
                    await cursor.execute(select_query)
                    result = await cursor.fetchone()
                assert result[1] == "example1"
                assert result[2] is True


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_database_url_interface(database_url: str) -> None:
    async with Database(database_url) as database:
        assert isinstance(database.url, DatabaseURL)
        assert database.url == database_url


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_concurrent_access_on_single_connection(database_url: str) -> None:
    async with Database(database_url, force_rollback=True) as database:

        async def db_lookup() -> None:
            await database.fetch_one("SELECT 1 AS value")
        await asyncio.gather(db_lookup(), db_lookup())


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_concurrent_transactions_on_single_connection(database_url: str) -> None:
    async with Database(database_url) as database:

        @database.transaction()
        async def db_lookup() -> None:
            await database.fetch_one(query="SELECT 1 AS value")
        await asyncio.gather(db_lookup(), db_lookup())


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_concurrent_tasks_on_single_connection(database_url: str) -> None:
    async with Database(database_url) as database:

        async def db_lookup() -> None:
            await database.fetch_one(query="SELECT 1 AS value")
        await asyncio.gather(asyncio.create_task(db_lookup()), asyncio.create_task(db_lookup()))


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_concurrent_task_transactions_on_single_connection(database_url: str) -> None:
    async with Database(database_url) as database:

        @database.transaction()
        async def db_lookup() -> None:
            await database.fetch_one(query="SELECT 1 AS value")
        await asyncio.gather(asyncio.create_task(db_lookup()), asyncio.create_task(db_lookup()))


@pytest.mark.parametrize("database_url", DATABASE_URLS)
def test_global_connection_is_initialized_lazily(database_url: str) -> None:
    database_url_obj = DatabaseURL(database_url)
    if database_url_obj.dialect != "postgresql":
        pytest.skip("Test requires `pg_sleep()`")
    database = Database(database_url, force_rollback=True)

    @async_adapter
    async def run_database_queries() -> None:
        async with database:

            async def db_lookup() -> None:
                await database.fetch_one("SELECT pg_sleep(1)")
            await asyncio.gather(db_lookup(), db_lookup())
    run_database_queries()


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_iterate_outside_transaction_with_values(database_url: str) -> None:
    database_url_obj = DatabaseURL(database_url)
    if database_url_obj.dialect == "mysql":
        pytest.skip("MySQL does not support `FROM (VALUES ...)` (F641)")
    async with Database(database_url) as database:
        query = "SELECT * FROM (VALUES (1), (2), (3), (4), (5)) as t"
        iterate_results: List[Dict[str, Any]] = []
        async for result in database.iterate(query=query):
            iterate_results.append(result)
        assert len(iterate_results) == 5


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_iterate_outside_transaction_with_temp_table(database_url: str) -> None:
    database_url_obj = DatabaseURL(database_url)
    if database_url_obj.dialect == "sqlite":
        pytest.skip("SQLite interface does not work with temporary tables.")
    async with Database(database_url) as database:
        query = "CREATE TEMPORARY TABLE no_transac(num INTEGER)"
        await database.execute(query)
        query = "INSERT INTO no_transac(num) VALUES (1), (2), (3), (4), (5)"
        await database.execute(query)
        query = "SELECT * FROM no_transac"
        iterate_results: List[Dict[str, Any]] = []
        async for result in database.iterate(query=query):
            iterate_results.append(result)
        assert len(iterate_results) == 5


@pytest.mark.parametrize("select_query", [notes.select(), "SELECT * FROM notes"])
@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_column_names(database_url: str, select_query: Any) -> None:
    async with Database(database_url) as database:
        async with database.transaction(force_rollback=True):
            query = notes.insert()
            values: Dict[str, Any] = {"text": "example1", "completed": True}
            await database.execute(query, values)
            results = await database.fetch_all(query=select_query)
            assert len(results) == 1
            assert sorted(results[0]._mapping.keys()) == ["completed", "id", "text"]
            assert results[0]["text"] == "example1"
            assert results[0]["completed"] is True


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_postcompile_queries(database_url: str) -> None:
    async with Database(database_url) as database:
        query = notes.insert()
        values: Dict[str, Any] = {"text": "example1", "completed": True}
        await database.execute(query, values)
        query = notes.select().where(notes.c.id.in_([2, 3]))
        results = await database.fetch_all(query=query)
        assert len(results) == 0


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_result_named_access(database_url: str) -> None:
    async with Database(database_url) as database:
        query = notes.insert()
        values: Dict[str, Any] = {"text": "example1", "completed": True}
        await database.execute(query, values)
        query = notes.select().where(notes.c.text == "example1")
        result = await database.fetch_one(query=query)
        assert result.text == "example1"
        assert result.completed is True


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_mapping_property_interface(database_url: str) -> None:
    async with Database(database_url) as database:
        query = notes.select()
        single_result = await database.fetch_one(query=query)
        assert single_result._mapping["text"] == "example1"
        assert single_result._mapping["completed"] is True
        list_result = await database.fetch_all(query=query)
        assert list_result[0]._mapping["text"] == "example1"
        assert list_result[0]._mapping["completed"] is True


@async_adapter
async def test_should_not_maintain_ref_when_no_cache_param() -> None:
    async with Database("sqlite:///file::memory:", uri=True) as database:
        query = sqlalchemy.schema.CreateTable(notes)
        await database.execute(query)
        query = notes.insert()
        values: Dict[str, Any] = {"text": "example1", "completed": True}
        with pytest.raises(sqlite3.OperationalError):
            await database.execute(query, values)


@async_adapter
async def test_should_maintain_ref_when_cache_param() -> None:
    async with Database("sqlite:///file::memory:?cache=shared", uri=True) as database:
        query = sqlalchemy.schema.CreateTable(notes)
        await database.execute(query)
        query = notes.insert()
        values: Dict[str, Any] = {"text": "example1", "completed": True}
        await database.execute(query, values)
        query = notes.select().where(notes.c.text == "example1")
        result = await database.fetch_one(query=query)
        assert result.text == "example1"
        assert result.completed is True


@async_adapter
async def test_should_remove_ref_on_disconnect() -> None:
    async with Database("sqlite:///file::memory:?cache=shared", uri=True) as database:
        query = sqlalchemy.schema.CreateTable(notes)
        await database.execute(query)
        query = notes.insert()
        values: Dict[str, Any] = {"text": "example1", "completed": True}
        await database.execute(query, values)
    gc.collect()
    async with Database("sqlite:///file::memory:?cache=shared", uri=True) as database:
        query = notes.select()
        with pytest.raises(sqlite3.OperationalError):
            await database.fetch_all(query=query)


@pytest.mark.parametrize("database_url", DATABASE_URLS)
@async_adapter
async def test_mapping_property_interface_duplicate(database_url: str) -> None:
    async with Database(database_url) as database:
        query = notes.insert()
        values: Dict[str, Any] = {"text": "example1", "completed": True}
        await database.execute(query, values)
        query = notes.select()
        single_result = await database.fetch_one(query=query)
        assert single_result._mapping["text"] == "example1"
        assert single_result._mapping["completed"] is True
        list_result = await database.fetch_all(query=query)
        assert list_result[0]._mapping["text"] == "example1"
        assert list_result[0]._mapping["completed"] is True
