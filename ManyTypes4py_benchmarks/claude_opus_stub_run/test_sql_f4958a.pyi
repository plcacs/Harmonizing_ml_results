from __future__ import annotations

import sqlite3
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame, Series, Timestamp

from pandas.io import sql


pytestmark: list[pytest.MarkDecorator]


@pytest.fixture
def sql_strings() -> dict[str, dict[str, str]]: ...


def iris_table_metadata() -> Any: ...


def create_and_load_iris_sqlite3(conn: sqlite3.Connection, iris_file: Path) -> None: ...


def create_and_load_iris_postgresql(conn: Any, iris_file: Path) -> None: ...


def create_and_load_iris(conn: Any, iris_file: Path) -> None: ...


def create_and_load_iris_view(conn: Any) -> None: ...


def types_table_metadata(dialect: str) -> Any: ...


def create_and_load_types_sqlite3(conn: Any, types_data: list[tuple[Any, ...]]) -> None: ...


def create_and_load_types_postgresql(conn: Any, types_data: list[tuple[Any, ...]]) -> None: ...


def create_and_load_types(conn: Any, types_data: list[dict[str, Any]], dialect: str) -> None: ...


def create_and_load_postgres_datetz(conn: Any) -> Series: ...


def check_iris_frame(frame: DataFrame) -> None: ...


def count_rows(conn: Any, table_name: str) -> int: ...


@pytest.fixture
def iris_path(datapath: Any) -> Path: ...


@pytest.fixture
def types_data() -> list[dict[str, Any]]: ...


@pytest.fixture
def types_data_frame(types_data: list[dict[str, Any]]) -> DataFrame: ...


@pytest.fixture
def test_frame1() -> DataFrame: ...


@pytest.fixture
def test_frame3() -> DataFrame: ...


def get_all_views(conn: Any) -> list[str]: ...


def get_all_tables(conn: Any) -> list[str]: ...


def drop_table(table_name: str, conn: Any) -> None: ...


def drop_view(view_name: str, conn: Any) -> None: ...


@pytest.fixture
def mysql_pymysql_engine() -> Generator[Any, None, None]: ...


@pytest.fixture
def mysql_pymysql_engine_iris(mysql_pymysql_engine: Any, iris_path: Path) -> Any: ...


@pytest.fixture
def mysql_pymysql_engine_types(mysql_pymysql_engine: Any, types_data: list[dict[str, Any]]) -> Any: ...


@pytest.fixture
def mysql_pymysql_conn(mysql_pymysql_engine: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def mysql_pymysql_conn_iris(mysql_pymysql_engine_iris: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def mysql_pymysql_conn_types(mysql_pymysql_engine_types: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def postgresql_psycopg2_engine() -> Generator[Any, None, None]: ...


@pytest.fixture
def postgresql_psycopg2_engine_iris(postgresql_psycopg2_engine: Any, iris_path: Path) -> Any: ...


@pytest.fixture
def postgresql_psycopg2_engine_types(postgresql_psycopg2_engine: Any, types_data: list[dict[str, Any]]) -> Any: ...


@pytest.fixture
def postgresql_psycopg2_conn(postgresql_psycopg2_engine: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def postgresql_adbc_conn() -> Generator[Any, None, None]: ...


@pytest.fixture
def postgresql_adbc_iris(postgresql_adbc_conn: Any, iris_path: Path) -> Any: ...


@pytest.fixture
def postgresql_adbc_types(postgresql_adbc_conn: Any, types_data: list[dict[str, Any]]) -> Any: ...


@pytest.fixture
def postgresql_psycopg2_conn_iris(postgresql_psycopg2_engine_iris: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def postgresql_psycopg2_conn_types(postgresql_psycopg2_engine_types: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def sqlite_str() -> Generator[str, None, None]: ...


@pytest.fixture
def sqlite_engine(sqlite_str: str) -> Generator[Any, None, None]: ...


@pytest.fixture
def sqlite_conn(sqlite_engine: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def sqlite_str_iris(sqlite_str: str, iris_path: Path) -> str: ...


@pytest.fixture
def sqlite_engine_iris(sqlite_engine: Any, iris_path: Path) -> Any: ...


@pytest.fixture
def sqlite_conn_iris(sqlite_engine_iris: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def sqlite_str_types(sqlite_str: str, types_data: list[dict[str, Any]]) -> str: ...


@pytest.fixture
def sqlite_engine_types(sqlite_engine: Any, types_data: list[dict[str, Any]]) -> Any: ...


@pytest.fixture
def sqlite_conn_types(sqlite_engine_types: Any) -> Generator[Any, None, None]: ...


@pytest.fixture
def sqlite_adbc_conn() -> Generator[Any, None, None]: ...


@pytest.fixture
def sqlite_adbc_iris(sqlite_adbc_conn: Any, iris_path: Path) -> Any: ...


@pytest.fixture
def sqlite_adbc_types(sqlite_adbc_conn: Any, types_data: list[dict[str, Any]]) -> Any: ...


@pytest.fixture
def sqlite_buildin() -> Generator[sqlite3.Connection, None, None]: ...


@pytest.fixture
def sqlite_buildin_iris(sqlite_buildin: sqlite3.Connection, iris_path: Path) -> sqlite3.Connection: ...


@pytest.fixture
def sqlite_buildin_types(sqlite_buildin: sqlite3.Connection, types_data: list[dict[str, Any]]) -> sqlite3.Connection: ...


mysql_connectable: list[pytest.ParameterSet]
mysql_connectable_iris: list[pytest.ParameterSet]
mysql_connectable_types: list[pytest.ParameterSet]
postgresql_connectable: list[pytest.ParameterSet]
postgresql_connectable_iris: list[pytest.ParameterSet]
postgresql_connectable_types: list[pytest.ParameterSet]
sqlite_connectable: list[str]
sqlite_connectable_iris: list[str]
sqlite_connectable_types: list[str]
sqlalchemy_connectable: list[Any]
sqlalchemy_connectable_iris: list[Any]
sqlalchemy_connectable_types: list[Any]
adbc_connectable: list[Any]
adbc_connectable_iris: list[Any]
adbc_connectable_types: list[Any]
all_connectable: list[Any]
all_connectable_iris: list[Any]
all_connectable_types: list[Any]


def test_dataframe_to_sql(conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None: ...


def test_dataframe_to_sql_empty(conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None: ...


def test_dataframe_to_sql_arrow_dtypes(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_dataframe_to_sql_arrow_dtypes_missing(conn: str, request: pytest.FixtureRequest, nulls_fixture: Any) -> None: ...


def test_to_sql(conn: str, method: str | None, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None: ...


def test_to_sql_exist(conn: str, mode: str, num_row_coef: int, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None: ...


def test_to_sql_exist_fail(conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None: ...


def test_read_iris_query(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_iris_query_chunksize(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_iris_query_expression_with_parameter(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_iris_query_string_with_parameter(conn: str, request: pytest.FixtureRequest, sql_strings: dict[str, dict[str, str]]) -> None: ...


def test_read_iris_table(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_iris_table_chunksize(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_to_sql_callable(conn: str, test_frame1: DataFrame, request: pytest.FixtureRequest) -> None: ...


def test_default_type_conversion(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_procedure(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_copy_from_callable_insertion_method(conn: str, expected_count: int | str, request: pytest.FixtureRequest) -> None: ...


def test_insertion_method_on_conflict_do_nothing(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_to_sql_on_public_schema(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_insertion_method_on_conflict_update(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_view_postgres(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_view_sqlite(sqlite_buildin: sqlite3.Connection) -> None: ...


def flavor(conn_name: str) -> str: ...


def test_read_sql_iris_parameter(conn: str, request: pytest.FixtureRequest, sql_strings: dict[str, dict[str, str]]) -> None: ...


def test_read_sql_iris_named_parameter(conn: str, request: pytest.FixtureRequest, sql_strings: dict[str, dict[str, str]]) -> None: ...


def test_read_sql_iris_no_parameter_with_percent(conn: str, request: pytest.FixtureRequest, sql_strings: dict[str, dict[str, str]]) -> None: ...


def test_api_read_sql_view(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_read_sql_with_chunksize_no_result(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_to_sql(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_to_sql_fail(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_to_sql_replace(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_to_sql_append(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_to_sql_type_mapping(conn: str, request: pytest.FixtureRequest, test_frame3: DataFrame) -> None: ...


def test_api_to_sql_series(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_roundtrip(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_roundtrip_chunksize(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_execute_sql(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_date_parsing(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_custom_dateparsing_error(conn: str, request: pytest.FixtureRequest, read_sql: Any, text: str, mode: str | tuple[str, ...], error: str, types_data_frame: DataFrame) -> None: ...


def test_api_date_and_index(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_timedelta(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_complex_raises(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_to_sql_index_label(conn: str, request: pytest.FixtureRequest, index_name: str | int | None, index_label: str | int | None, expected: str) -> None: ...


def test_api_to_sql_index_label_multiindex(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_multiindex_roundtrip(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_dtype_argument(conn: str, request: pytest.FixtureRequest, dtype: Any) -> None: ...


def test_api_integer_col_names(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_get_schema(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_get_schema_with_schema(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_get_schema_dtypes(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_get_schema_keys(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_api_chunksize_read(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_categorical(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_unicode_column_name(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_escaped_table_name(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_api_read_sql_duplicate_columns(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_table_columns(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_read_table_index_col(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_read_sql_delegate(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_not_reflect_all_tables(sqlite_conn: Any) -> None: ...


def test_warning_case_insensitive_table_name(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_sqlalchemy_type_mapping(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_sqlalchemy_integer_mapping(conn: str, request: pytest.FixtureRequest, integer: str | type, expected: str) -> None: ...


def test_sqlalchemy_integer_overload_mapping(conn: str, request: pytest.FixtureRequest, integer: str) -> None: ...


def test_database_uri_string(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_pg8000_sqlalchemy_passthrough_error(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_query_by_text_obj(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_query_by_select_obj(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_column_with_percentage(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_sql_open_close(test_frame3: DataFrame) -> None: ...


def test_con_string_import_error() -> None: ...


def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed() -> None: ...


def test_sqlite_read_sql_delegate(sqlite_buildin_iris: sqlite3.Connection) -> None: ...


def test_get_schema2(test_frame1: DataFrame) -> None: ...


def test_sqlite_type_mapping(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_create_table(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_drop_table(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_roundtrip(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_execute_sql(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_sqlalchemy_read_table(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_sqlalchemy_read_table_columns(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_table_absent_raises(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_sqlalchemy_default_type_conversion(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_bigint(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_default_date_load(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_datetime_with_timezone_query(conn: str, request: pytest.FixtureRequest, parse_dates: list[str] | None) -> None: ...


def test_datetime_with_timezone_query_chunksize(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_datetime_with_timezone_table(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_datetime_with_timezone_roundtrip(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_out_of_bounds_datetime(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_naive_datetimeindex_roundtrip(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_date_parsing(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_datetime(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_datetime_NaT(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_datetime_date(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_datetime_time(conn: str, request: pytest.FixtureRequest, sqlite_buildin: sqlite3.Connection) -> None: ...


def test_mixed_dtype_insert(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_nan_numeric(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_nan_fullcolumn(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_nan_string(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_to_sql_save_index(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_transactions(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_transaction_rollback(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_get_schema_create_table(conn: str, request: pytest.FixtureRequest, test_frame3: DataFrame) -> None: ...


def test_dtype(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_notna_dtype(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_double_precision(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_connectable_issue_example(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_to_sql_with_negative_npinf(conn: str, request: pytest.FixtureRequest, input: dict[str, list[Any]]) -> None: ...


def test_temporary_table(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_invalid_engine(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_to_sql_with_sql_engine(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_options_sqlalchemy(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_options_auto(conn: str, request: pytest.FixtureRequest, test_frame1: DataFrame) -> None: ...


def test_options_get_engine() -> None: ...


def test_get_engine_auto_error_message() -> None: ...


def test_read_sql_dtype_backend(conn: str, request: pytest.FixtureRequest, string_storage: str, func: str, dtype_backend: str, dtype_backend_data: DataFrame, dtype_backend_expected: Any) -> None: ...


def test_read_sql_dtype_backend_table(conn: str, request: pytest.FixtureRequest, string_storage: str, func: str, dtype_backend: str, dtype_backend_data: DataFrame, dtype_backend_expected: Any) -> None: ...


def test_read_sql_invalid_dtype_backend_table(conn: str, request: pytest.FixtureRequest, func: str, dtype_backend_data: DataFrame) -> None: ...


@pytest.fixture
def dtype_backend_data() -> DataFrame: ...


@pytest.fixture
def dtype_backend_expected() -> Any: ...


def test_chunksize_empty_dtypes(conn: str, request: pytest.FixtureRequest) -> None: ...


def test_read_sql_dtype(conn: str, request: pytest.FixtureRequest, func: str, dtype_backend: Any) -> None: ...


def test_bigint_warning(sqlite_engine: Any) -> None: ...


def test_valueerror_exception(sqlite_engine: Any) -> None: ...


def test_row_object_is_named_tuple(sqlite_engine: Any) -> None: ...


def test_read_sql_string_inference(sqlite_engine: Any) -> None: ...


def test_roundtripping_datetimes(sqlite_engine: Any) -> None: ...


@pytest.fixture
def sqlite_builtin_detect_types() -> Generator[sqlite3.Connection, None, None]: ...


def test_roundtripping_datetimes_detect_types(sqlite_builtin_detect_types: sqlite3.Connection) -> None: ...


def test_psycopg2_schema_support(postgresql_psycopg2_engine: Any) -> None: ...


def test_self_join_date_columns(postgresql_psycopg2_engine: Any) -> None: ...


def test_create_and_drop_table(sqlite_engine: Any) -> None: ...


def test_sqlite_datetime_date(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_sqlite_datetime_time(tz_aware: bool, sqlite_buildin: sqlite3.Connection) -> None: ...


def get_sqlite_column_type(conn: sqlite3.Connection, table: str, column: str) -> str: ...


def test_sqlite_test_dtype(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_sqlite_notna_dtype(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_sqlite_illegal_names(sqlite_buildin: sqlite3.Connection) -> None: ...


def format_query(sql: str, *args: Any) -> str: ...


def tquery(query: str, con: Any = ...) -> list[Any] | None: ...


def test_xsqlite_basic(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_xsqlite_write_row_by_row(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_xsqlite_execute(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_xsqlite_schema(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_xsqlite_execute_fail(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_xsqlite_execute_closed_connection() -> None: ...


def test_xsqlite_keyword_as_column_names(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_xsqlite_onecolumn_of_integer(sqlite_buildin: sqlite3.Connection) -> None: ...


def test_xsqlite_if_exists(sqlite_buildin: sqlite3.Connection) -> None: ...