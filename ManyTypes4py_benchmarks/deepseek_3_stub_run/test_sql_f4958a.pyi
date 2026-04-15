from __future__ import annotations
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
import contextlib
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
import sqlite3
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)
import uuid
import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series, Timestamp
from pandas.io import sql

if TYPE_CHECKING:
    import sqlalchemy
    import sqlalchemy.engine
    import sqlalchemy.sql.elements
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text
    import adbc_driver_manager.dbapi
    import pyarrow as pa

_T = TypeVar("_T")

pytestmark: list[pytest.MarkDecorator] = ...

@pytest.fixture
def sql_strings() -> dict[str, dict[str, str]]: ...

def iris_table_metadata() -> sqlalchemy.Table: ...

def create_and_load_iris_sqlite3(
    conn: Union[sqlite3.Connection, adbc_driver_manager.dbapi.Connection],
    iris_file: Path,
) -> None: ...

def create_and_load_iris_postgresql(
    conn: Union[sqlalchemy.engine.Connection, adbc_driver_manager.dbapi.Connection],
    iris_file: Path,
) -> None: ...

def create_and_load_iris(
    conn: Union[str, sqlalchemy.engine.Engine, sqlalchemy.engine.Connection],
    iris_file: Path,
) -> None: ...

def create_and_load_iris_view(
    conn: Union[
        sqlite3.Connection,
        sqlalchemy.engine.Engine,
        sqlalchemy.engine.Connection,
        adbc_driver_manager.dbapi.Connection,
    ]
) -> None: ...

def types_table_metadata(dialect: str) -> sqlalchemy.Table: ...

def create_and_load_types_sqlite3(
    conn: Union[sqlite3.Connection, adbc_driver_manager.dbapi.Connection],
    types_data: list[tuple[Any, ...]],
) -> None: ...

def create_and_load_types_postgresql(
    conn: Union[sqlalchemy.engine.Connection, adbc_driver_manager.dbapi.Connection],
    types_data: list[tuple[Any, ...]],
) -> None: ...

def create_and_load_types(
    conn: Union[str, sqlalchemy.engine.Engine, sqlalchemy.engine.Connection],
    types_data: list[dict[str, Any]],
    dialect: str,
) -> None: ...

def create_and_load_postgres_datetz(
    conn: Union[str, sqlalchemy.engine.Engine, sqlalchemy.engine.Connection],
) -> Series: ...

def check_iris_frame(frame: DataFrame) -> None: ...

def count_rows(
    conn: Union[
        str,
        sqlite3.Connection,
        sqlalchemy.engine.Engine,
        sqlalchemy.engine.Connection,
        adbc_driver_manager.dbapi.Connection,
    ],
    table_name: str,
) -> int: ...

@pytest.fixture
def iris_path(datapath: Callable[..., str]) -> Path: ...

@pytest.fixture
def types_data() -> list[dict[str, Any]]: ...

@pytest.fixture
def types_data_frame(types_data: list[dict[str, Any]]) -> DataFrame: ...

@pytest.fixture
def test_frame1() -> DataFrame: ...

@pytest.fixture
def test_frame3() -> DataFrame: ...

def get_all_views(
    conn: Union[
        sqlite3.Connection,
        sqlalchemy.engine.Engine,
        sqlalchemy.engine.Connection,
        adbc_driver_manager.dbapi.Connection,
    ]
) -> list[str]: ...

def get_all_tables(
    conn: Union[
        sqlite3.Connection,
        sqlalchemy.engine.Engine,
        sqlalchemy.engine.Connection,
        adbc_driver_manager.dbapi.Connection,
    ]
) -> list[str]: ...

def drop_table(
    table_name: str,
    conn: Union[
        sqlite3.Connection,
        sqlalchemy.engine.Engine,
        sqlalchemy.engine.Connection,
        adbc_driver_manager.dbapi.Connection,
    ],
) -> None: ...

def drop_view(
    view_name: str,
    conn: Union[
        sqlite3.Connection,
        sqlalchemy.engine.Engine,
        sqlalchemy.engine.Connection,
        adbc_driver_manager.dbapi.Connection,
    ],
) -> None: ...

@pytest.fixture
def mysql_pymysql_engine() -> Iterator[sqlalchemy.engine.Engine]: ...

@pytest.fixture
def mysql_pymysql_engine_iris(
    mysql_pymysql_engine: sqlalchemy.engine.Engine,
    iris_path: Path,
) -> sqlalchemy.engine.Engine: ...

@pytest.fixture
def mysql_pymysql_engine_types(
    mysql_pymysql_engine: sqlalchemy.engine.Engine,
    types_data: list[dict[str, Any]],
) -> sqlalchemy.engine.Engine: ...

@pytest.fixture
def mysql_pymysql_conn(
    mysql_pymysql_engine: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def mysql_pymysql_conn_iris(
    mysql_pymysql_engine_iris: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def mysql_pymysql_conn_types(
    mysql_pymysql_engine_types: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def postgresql_psycopg2_engine() -> Iterator[sqlalchemy.engine.Engine]: ...

@pytest.fixture
def postgresql_psycopg2_engine_iris(
    postgresql_psycopg2_engine: sqlalchemy.engine.Engine,
    iris_path: Path,
) -> sqlalchemy.engine.Engine: ...

@pytest.fixture
def postgresql_psycopg2_engine_types(
    postgresql_psycopg2_engine: sqlalchemy.engine.Engine,
    types_data: list[dict[str, Any]],
) -> sqlalchemy.engine.Engine: ...

@pytest.fixture
def postgresql_psycopg2_conn(
    postgresql_psycopg2_engine: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def postgresql_adbc_conn() -> Iterator[adbc_driver_manager.dbapi.Connection]: ...

@pytest.fixture
def postgresql_adbc_iris(
    postgresql_adbc_conn: adbc_driver_manager.dbapi.Connection,
    iris_path: Path,
) -> adbc_driver_manager.dbapi.Connection: ...

@pytest.fixture
def postgresql_adbc_types(
    postgresql_adbc_conn: adbc_driver_manager.dbapi.Connection,
    types_data: list[dict[str, Any]],
) -> adbc_driver_manager.dbapi.Connection: ...

@pytest.fixture
def postgresql_psycopg2_conn_iris(
    postgresql_psycopg2_engine_iris: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def postgresql_psycopg2_conn_types(
    postgresql_psycopg2_engine_types: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def sqlite_str() -> Iterator[str]: ...

@pytest.fixture
def sqlite_engine(sqlite_str: str) -> Iterator[sqlalchemy.engine.Engine]: ...

@pytest.fixture
def sqlite_conn(
    sqlite_engine: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def sqlite_str_iris(sqlite_str: str, iris_path: Path) -> str: ...

@pytest.fixture
def sqlite_engine_iris(
    sqlite_engine: sqlalchemy.engine.Engine,
    iris_path: Path,
) -> sqlalchemy.engine.Engine: ...

@pytest.fixture
def sqlite_conn_iris(
    sqlite_engine_iris: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def sqlite_str_types(
    sqlite_str: str,
    types_data: list[dict[str, Any]],
) -> str: ...

@pytest.fixture
def sqlite_engine_types(
    sqlite_engine: sqlalchemy.engine.Engine,
    types_data: list[dict[str, Any]],
) -> sqlalchemy.engine.Engine: ...

@pytest.fixture
def sqlite_conn_types(
    sqlite_engine_types: sqlalchemy.engine.Engine,
) -> Iterator[sqlalchemy.engine.Connection]: ...

@pytest.fixture
def sqlite_adbc_conn() -> Iterator[adbc_driver_manager.dbapi.Connection]: ...

@pytest.fixture
def sqlite_adbc_iris(
    sqlite_adbc_conn: adbc_driver_manager.dbapi.Connection,
    iris_path: Path,
) -> adbc_driver_manager.dbapi.Connection: ...

@pytest.fixture
def sqlite_adbc_types(
    sqlite_adbc_conn: adbc_driver_manager.dbapi.Connection,
    types_data: list[dict[str, Any]],
) -> adbc_driver_manager.dbapi.Connection: ...

@pytest.fixture
def sqlite_buildin() -> Iterator[sqlite3.Connection]: ...

@pytest.fixture
def sqlite_buildin_iris(
    sqlite_buildin: sqlite3.Connection,
    iris_path: Path,
) -> sqlite3.Connection: ...

@pytest.fixture
def sqlite_buildin_types(
    sqlite_buildin: sqlite3.Connection,
    types_data: list[dict[str, Any]],
) -> sqlite3.Connection: ...

mysql_connectable: list[Union[str, pytest.param]] = ...
mysql_connectable_iris: list[Union[str, pytest.param]] = ...
mysql_connectable_types: list[Union[str, pytest.param]] = ...
postgresql_connectable: list[Union[str, pytest.param]] = ...
postgresql_connectable_iris: list[Union[str, pytest.param]] = ...
postgresql_connectable_types: list[Union[str, pytest.param]] = ...
sqlite_connectable: list[str] = ...
sqlite_connectable_iris: list[str] = ...
sqlite_connectable_types: list[str] = ...
sqlalchemy_connectable: list[Union[str, pytest.param]] = ...
sqlalchemy_connectable_iris: list[Union[str, pytest.param]] = ...
sqlalchemy_connectable_types: list[Union[str, pytest.param]] = ...
adbc_connectable: list[Union[str, pytest.param]] = ...
adbc_connectable_iris: list[Union[str, pytest.param]] = ...
adbc_connectable_types: list[Union[str, pytest.param]] = ...
all_connectable: list[Union[str, pytest.param]] = ...
all_connectable_iris: list[Union[str, pytest.param]] = ...
all_connectable_types: list[Union[str, pytest.param]] = ...

@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql(
    conn: str,
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_empty(
    conn: str,
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes_missing(
    conn: str,
    request: pytest.FixtureRequest,
    nulls_fixture: Any,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("method", [None, "multi"])
def test_to_sql(
    conn: str,
    method: Optional[str],
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("mode, num_row_coef", [("replace", 1), ("append", 2)])
def test_to_sql_exist(
    conn: str,
    mode: str,
    num_row_coef: int,
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_exist_fail(
    conn: str,
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_chunksize(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_query_expression_with_parameter(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_string_with_parameter(
    conn: str,
    request: pytest.FixtureRequest,
    sql_strings: dict[str, dict[str, str]],
) -> None: ...

@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table_chunksize(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_to_sql_callable(
    conn: str,
    test_frame1: DataFrame,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_types)
def test_default_type_conversion(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", mysql_connectable)
def test_read_procedure(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("expected_count", [2, "Success!"])
def test_copy_from_callable_insertion_method(
    conn: str,
    expected_count: Union[int, str],
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", postgresql_connectable)
def test_insertion_method_on_conflict_do_nothing(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_on_public_schema(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", mysql_connectable)
def test_insertion_method_on_conflict_update(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", postgresql_connectable)
def test_read_view_postgres(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

def test_read_view_sqlite(sqlite_buildin: sqlite3.Connection) -> None: ...

def flavor(conn_name: str) -> str: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_parameter(
    conn: str,
    request: pytest.FixtureRequest,
    sql_strings: dict[str, dict[str, str]],
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_named_parameter(
    conn: str,
    request: pytest.FixtureRequest,
    sql_strings: dict[str, dict[str, str]],
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_no_parameter_with_percent(
    conn: str,
    request: pytest.FixtureRequest,
    sql_strings: dict[str, dict[str, str]],
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_api_read_sql_view(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable_iris)
def test_api_read_sql_with_chunksize_no_result(
    conn: str,
    request: pytest.FixtureRequest,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql(
    conn: str,
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_fail(
    conn: str,
    request: pytest.FixtureRequest,
    test_frame1: DataFrame,
) -> None: ...

@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_replace(