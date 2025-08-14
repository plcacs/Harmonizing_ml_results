from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, time
from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np
import pandas as pd
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError, DatabaseError
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

from pandas import get_option
from pandas.core.api import DataFrame, Series
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime

from pandas.io._util import arrow_table_to_pandas

if TYPE_CHECKING:
    from collections.abc import Callable
    from sqlalchemy import Table
    from sqlalchemy.sql.expression import Select, TextClause
    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        IndexLabel,
        Self,
    )
    from pandas import Index

# Type aliases
SQLAlchemySelectable = Union[str, Select, TextClause]
SQLConnection = Any  # Can be sqlite3.Connection, SQLAlchemy connectable, or str
SQLTableName = str
SQLSchema = Optional[str]
SQLIfExists = Literal["fail", "replace", "append"]
SQLMethod = Optional[Union[Literal["multi"], Callable]]
SQLEngine = Literal["auto", "sqlalchemy"]
SQLChunksize = Optional[int]
SQLIndexCol = Optional[Union[str, List[str]]]
SQLParseDates = Optional[Union[List[str], Dict[str, str], Dict[str, Dict[str, Any]]]
SQLParams = Optional[Union[List[Any], Mapping[str, Any]]]
SQLColumns = Optional[List[str]]
SQLDtype = Optional[DtypeArg]
SQLDtypeBackend = Union[DtypeBackend, Literal["numpy"]]

def _process_parse_dates_argument(
    parse_dates: SQLParseDates
) -> List[Union[str, Dict[str, Any]]]:
    if parse_dates is True or parse_dates is None or parse_dates is False:
        return []
    elif not hasattr(parse_dates, "__iter__"):
        return [parse_dates]
    return list(parse_dates)

def _handle_date_column(
    col: Series,
    utc: bool = False,
    format: Optional[Union[str, Dict[str, Any]]] = None
) -> Series:
    if isinstance(format, dict):
        return to_datetime(col, **format)
    else:
        if format is None and (
            issubclass(col.dtype.type, np.floating)
            or issubclass(col.dtype.type, np.integer)
        ):
            format = "s"
        if format in ["D", "d", "h", "m", "s", "ms", "us", "ns"]:
            return to_datetime(col, errors="coerce", unit=format, utc=utc)
        elif isinstance(col.dtype, DatetimeTZDtype):
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors="coerce", format=format, utc=utc)

def _parse_date_columns(
    data_frame: DataFrame,
    parse_dates: SQLParseDates
) -> DataFrame:
    parse_dates = _process_parse_dates_argument(parse_dates)
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        if isinstance(df_col.dtype, DatetimeTZDtype) or col_name in parse_dates:
            try:
                fmt = parse_dates[col_name]
            except (KeyError, TypeError):
                fmt = None
            data_frame.isetitem(i, _handle_date_column(df_col, format=fmt))
    return data_frame

def _convert_arrays_to_dataframe(
    data: List[Tuple[Any, ...]],
    columns: List[str],
    coerce_float: bool = True,
    dtype_backend: SQLDtypeBackend = "numpy",
) -> DataFrame:
    content = lib.to_object_array_tuples(data)
    idx_len = content.shape[0]
    arrays = convert_object_array(
        list(content.T),
        dtype=None,
        coerce_float=coerce_float,
        dtype_backend=dtype_backend,
    )
    if dtype_backend == "pyarrow":
        pa = import_optional_dependency("pyarrow")
        result_arrays = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == "string":
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        arrays = result_arrays
    if arrays:
        return DataFrame._from_arrays(
            arrays, columns=columns, index=range(idx_len), verify_integrity=False
        )
    else:
        return DataFrame(columns=columns)

def _wrap_result(
    data: List[Tuple[Any, ...]],
    columns: List[str],
    index_col: SQLIndexCol = None,
    coerce_float: bool = True,
    parse_dates: SQLParseDates = None,
    dtype: SQLDtype = None,
    dtype_backend: SQLDtypeBackend = "numpy",
) -> DataFrame:
    frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)
    if dtype:
        frame = frame.astype(dtype)
    frame = _parse_date_columns(frame, parse_dates)
    if index_col is not None:
        frame = frame.set_index(index_col)
    return frame

def _wrap_result_adbc(
    df: DataFrame,
    *,
    index_col: SQLIndexCol = None,
    parse_dates: SQLParseDates = None,
    dtype: SQLDtype = None,
    dtype_backend: SQLDtypeBackend = "numpy",
) -> DataFrame:
    if dtype:
        df = df.astype(dtype)
    df = _parse_date_columns(df, parse_dates)
    if index_col is not None:
        df = df.set_index(index_col)
    return df

@overload
def read_sql_table(
    table_name: str,
    con: SQLConnection,
    schema: SQLSchema = ...,
    index_col: SQLIndexCol = ...,
    coerce_float: bool = ...,
    parse_dates: SQLParseDates = ...,
    columns: SQLColumns = ...,
    chunksize: None = ...,
    dtype_backend: SQLDtypeBackend = ...,
) -> DataFrame: ...

@overload
def read_sql_table(
    table_name: str,
    con: SQLConnection,
    schema: SQLSchema = ...,
    index_col: SQLIndexCol = ...,
    coerce_float: bool = ...,
    parse_dates: SQLParseDates = ...,
    columns: SQLColumns = ...,
    chunksize: int = ...,
    dtype_backend: SQLDtypeBackend = ...,
) -> Iterator[DataFrame]: ...

def read_sql_table(
    table_name: str,
    con: SQLConnection,
    schema: SQLSchema = None,
    index_col: SQLIndexCol = None,
    coerce_float: bool = True,
    parse_dates: SQLParseDates = None,
    columns: SQLColumns = None,
    chunksize: SQLChunksize = None,
    dtype_backend: SQLDtypeBackend = lib.no_default,
) -> Union[DataFrame, Iterator[DataFrame]]:
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        if not pandas_sql.has_table(table_name):
            raise ValueError(f"Table {table_name} not found")
        table = pandas_sql.read_table(
            table_name,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )
    if table is not None:
        return table
    else:
        raise ValueError(f"Table {table_name} not found", con)

@overload
def read_sql_query(
    sql: SQLAlchemySelectable,
    con: SQLConnection,
    index_col: SQLIndexCol = ...,
    coerce_float: bool = ...,
    params: SQLParams = ...,
    parse_dates: SQLParseDates = ...,
    chunksize: None = ...,
    dtype: SQLDtype = ...,
    dtype_backend: SQLDtypeBackend = ...,
) -> DataFrame: ...

@overload
def read_sql_query(
    sql: SQLAlchemySelectable,
    con: SQLConnection,
    index_col: SQLIndexCol = ...,
    coerce_float: bool = ...,
    params: SQLParams = ...,
    parse_dates: SQLParseDates = ...,
    chunksize: int = ...,
    dtype: SQLDtype = ...,
    dtype_backend: SQLDtypeBackend = ...,
) -> Iterator[DataFrame]: ...

def read_sql_query(
    sql: SQLAlchemySelectable,
    con: SQLConnection,
    index_col: SQLIndexCol = None,
    coerce_float: bool = True,
    params: SQLParams = None,
    parse_dates: SQLParseDates = None,
    chunksize: SQLChunksize = None,
    dtype: SQLDtype = None,
    dtype_backend: SQLDtypeBackend = lib.no_default,
) -> Union[DataFrame, Iterator[DataFrame]]:
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        return pandas_sql.read_query(
            sql,
            index_col=index_col,
            params=params,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            chunksize=chunksize,
            dtype=dtype,
            dtype_backend=dtype_backend,
        )

@overload
def read_sql(
    sql: SQLAlchemySelectable,
    con: SQLConnection,
    index_col: SQLIndexCol = ...,
    coerce_float: bool = ...,
    params: SQLParams = ...,
    parse_dates: SQLParseDates = ...,
    columns: List[str] = ...,
    chunksize: None = ...,
    dtype_backend: SQLDtypeBackend = ...,
    dtype: SQLDtype = None,
) -> DataFrame: ...

@overload
def read_sql(
    sql: SQLAlchemySelectable,
    con: SQLConnection,
    index_col: SQLIndexCol = ...,
    coerce_float: bool = ...,
    params: SQLParams = ...,
    parse_dates: SQLParseDates = ...,
    columns: List[str] = ...,
    chunksize: int = ...,
    dtype_backend: SQLDtypeBackend = ...,
    dtype: SQLDtype = None,
) -> Iterator[DataFrame]: ...

def read_sql(
    sql: SQLAlchemySelectable,
    con: SQLConnection,
    index_col: SQLIndexCol = None,
    coerce_float: bool = True,
    params: SQLParams = None,
    parse_dates: SQLParseDates = None,
    columns: SQLColumns = None,
    chunksize: SQLChunksize = None,
    dtype_backend: SQLDtypeBackend = lib.no_default,
    dtype: SQLDtype = None,
) -> Union[DataFrame, Iterator[DataFrame]]:
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        if isinstance(pandas_sql, SQLiteDatabase):
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )

        try:
            _is_table_name = pandas_sql.has_table(sql)
        except Exception:
            _is_table_name = False

        if _is_table_name:
            return pandas_sql.read_table(
                sql,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
            )
        else:
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )

def to_sql(
    frame: Union[DataFrame, Series],
    name: str,
    con: SQLConnection,
    schema: SQLSchema = None,
    if_exists: SQLIfExists = "fail",
    index: bool = True,
    index_label: IndexLabel = None,
    chunksize: SQLChunksize = None,
    dtype: SQLDtype = None,
    method: SQLMethod = None,
    engine: SQLEngine = "auto",
    **engine_kwargs: Any,
) -> Optional[int]:
    if if_exists not in ("fail", "replace", "append"):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError(
            "'frame' argument should be either a Series or a DataFrame"
        )

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        return pandas_sql.to_sql(
            frame,
            name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
            engine=engine,
            **engine_kwargs,
        )

def has_table(table_name: str, con: SQLConnection, schema: SQLSchema = None) -> bool:
    with pandasSQL_builder(con, schema=schema) as pandas_sql:
        return pandas_sql.has_table(table_name)

table_exists = has_table

def pandasSQL_builder(
    con: SQLConnection,
    schema: SQLSchema = None,
    need_transaction: bool = False,
) -> PandasSQL:
    import sqlite3
    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)

    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")
    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError("Using URI string without sqlalchemy installed.")
    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Connectable)):
        return SQLDatabase(con, schema, need_transaction)

    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    if adbc and isinstance(con, adbc.Connection):
        return ADBCDatabase(con)

    warnings.warn(
        "pandas only supports SQLAlchemy connectable (engine/connection) or "
        "database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 "
        "objects are not tested. Please consider using SQLAlchemy.",
        UserWarning,
        stacklevel=find_stack_level(),
    )
    return SQLiteDatabase(con)

class SQLTable(PandasObject):
    def __init__(
        self,
        name: str,
        pandas_sql_engine: PandasSQL,
        frame: Optional[DataFrame] = None,
        index: Union[bool, str, List[str], None] = True,
        if_exists: SQLIfExists = "fail",
        prefix: str = "pandas",
        index_label: IndexLabel = None,
        schema: SQLSchema = None,
        keys: Optional[List[str]] = None,
        dtype: SQLDtype = None,
    ) -> None:
        self.name = name
        self.pd_sql = pandas_sql_engine
        self.prefix = prefix
        self.frame = frame
        self.index = self._index_name(index, index_label)
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype

        if frame is not None:
            self.table = self._create_table_setup()
        else:
            self.table = self.pd_sql.get_table(self.name, self.schema)

        if self.table is None:
            raise ValueError(f"Could not init table '{name}'")
        if not len(self.name):
            raise ValueError("Empty table name specified")

    def exists(self) -> bool:
        return self.pd_sql.has_table(self.name, self.schema)

    def sql_schema(self) -> str:
        from sqlalchemy.schema import CreateTable
        return str(CreateTable(self.table).compile(self.pd_sql.con))

    def _execute_create(self) -> None:
        self.table = self.table.to_metadata(self.pd_sql.meta)
        with self.pd_sql.run_transaction():
            self.table.create(bind=self.pd_sql.con)

    def create(self) -> None:
        if self.exists():
            if