"""
Collection of query wrappers / abstractions to both facilitate data
retrieval and to reduce dependency on DB-specific API.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, time
from functools import partial
import re
from typing import (
    TYPE_CHECKING, Any, Literal, cast, overload, Optional, Union, List, Dict, 
    Tuple, Sequence, Generator, Iterator, Mapping, Callable, TypeVar, Type
)
import warnings
import numpy as np
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError, DatabaseError
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
    is_dict_like, is_list_like, is_object_dtype, is_string_dtype
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
    from collections.abc import Callable as CallableABC, Generator as GeneratorABC
    from sqlalchemy import Table
    from sqlalchemy.sql.expression import Select, TextClause
    from pandas._typing import DtypeArg, DtypeBackend, IndexLabel, Self
    from pandas import Index
    from sqlite3 import Connection as SQLite3Connection
    from adbc_driver_manager.dbapi import Connection as ADBCConnection
    from sqlalchemy.engine import Connection as SQLAlchemyConnection, Engine as SQLAlchemyEngine
    from pyarrow import Table as PyArrowTable

T = TypeVar('T')
DtypeBackendT = Literal['numpy', 'numpy_nullable', 'pyarrow']
IfExistsT = Literal['fail', 'replace', 'append']
EngineT = Literal['auto', 'sqlalchemy']
MethodT = Optional[Union[Literal['multi'], Callable[..., Optional[int]]]]
ParseDatesT = Union[
    bool, List[str], Dict[str, Union[str, Dict[str, Any]]]
]
IndexLabelT = Optional[Union[str, Sequence[str]]]
SQLQueryT = Union[str, Any]  # Any for SQLAlchemy selectable

def _process_parse_dates_argument(
    parse_dates: ParseDatesT
) -> List[Union[str, Dict[str, Any]]:
    """Process parse_dates argument for read_sql functions"""
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []
    elif not hasattr(parse_dates, '__iter__'):
        parse_dates = [parse_dates]
    return parse_dates

def _handle_date_column(
    col: Series, 
    utc: bool = False, 
    format: Optional[Union[str, Dict[str, Any]]] = None
) -> Series:
    if isinstance(format, dict):
        return to_datetime(col, **format)
    else:
        if format is None and (
            issubclass(col.dtype.type, np.floating) or 
            issubclass(col.dtype.type, np.integer)
        ):
            format = 's'
        if format in ['D', 'd', 'h', 'm', 's', 'ms', 'us', 'ns']:
            return to_datetime(col, errors='coerce', unit=format, utc=utc)
        elif isinstance(col.dtype, DatetimeTZDtype):
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors='coerce', format=format, utc=utc)

def _parse_date_columns(
    data_frame: DataFrame, 
    parse_dates: ParseDatesT
) -> DataFrame:
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
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
    dtype_backend: DtypeBackendT = 'numpy'
) -> DataFrame:
    content = lib.to_object_array_tuples(data)
    idx_len = content.shape[0]
    arrays = convert_object_array(
        list(content.T), dtype=None, coerce_float=coerce_float, 
        dtype_backend=dtype_backend
    )
    if dtype_backend == 'pyarrow':
        pa = import_optional_dependency('pyarrow')
        result_arrays = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == 'string':
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
    index_col: Optional[Union[str, List[str]]] = None, 
    coerce_float: bool = True, 
    parse_dates: Optional[ParseDatesT] = None, 
    dtype: Optional[DtypeArg] = None, 
    dtype_backend: DtypeBackendT = 'numpy'
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
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
    index_col: Optional[Union[str, List[str]]] = None, 
    parse_dates: Optional[ParseDatesT] = None, 
    dtype: Optional[DtypeArg] = None, 
    dtype_backend: DtypeBackendT = 'numpy'
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    if dtype:
        df = df.astype(dtype)
    df = _parse_date_columns(df, parse_dates)
    if index_col is not None:
        df = df.set_index(index_col)
    return df

@overload
def read_sql_table(
    table_name: str,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine],
    schema: Optional[str] = ...,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    parse_dates: Optional[ParseDatesT] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: Optional[int] = ...,
    dtype_backend: DtypeBackendT = ...,
) -> DataFrame: ...

@overload
def read_sql_table(
    table_name: str,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine],
    schema: Optional[str] = ...,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    parse_dates: Optional[ParseDatesT] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: Optional[int] = ...,
    dtype_backend: DtypeBackendT = ...,
) -> Iterator[DataFrame]: ...

def read_sql_table(
    table_name: str,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine],
    schema: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    parse_dates: Optional[ParseDatesT] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    dtype_backend: Union[DtypeBackendT, Literal[lib.no_default]] = lib.no_default,
) -> Union[DataFrame, Iterator[DataFrame]]:
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = 'numpy'
    assert dtype_backend is not lib.no_default
    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        if not pandas_sql.has_table(table_name):
            raise ValueError(f'Table {table_name} not found')
        table = pandas_sql.read_table(
            table_name, index_col=index_col, coerce_float=coerce_float,
            parse_dates=parse_dates, columns=columns, chunksize=chunksize,
            dtype_backend=dtype_backend
        )
    if table is not None:
        return table
    else:
        raise ValueError(f'Table {table_name} not found', con)

@overload
def read_sql_query(
    sql: SQLQueryT,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List, Tuple, Dict]] = ...,
    parse_dates: Optional[ParseDatesT] = ...,
    chunksize: Optional[int] = ...,
    dtype: Optional[DtypeArg] = ...,
    dtype_backend: DtypeBackendT = ...,
) -> DataFrame: ...

@overload
def read_sql_query(
    sql: SQLQueryT,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List, Tuple, Dict]] = ...,
    parse_dates: Optional[ParseDatesT] = ...,
    chunksize: Optional[int] = ...,
    dtype: Optional[DtypeArg] = ...,
    dtype_backend: DtypeBackendT = ...,
) -> Iterator[DataFrame]: ...

def read_sql_query(
    sql: SQLQueryT,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    params: Optional[Union[List, Tuple, Dict]] = None,
    parse_dates: Optional[ParseDatesT] = None,
    chunksize: Optional[int] = None,
    dtype: Optional[DtypeArg] = None,
    dtype_backend: Union[DtypeBackendT, Literal[lib.no_default]] = lib.no_default,
) -> Union[DataFrame, Iterator[DataFrame]]:
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = 'numpy'
    assert dtype_backend is not lib.no_default
    with pandasSQL_builder(con) as pandas_sql:
        return pandas_sql.read_query(
            sql, index_col=index_col, params=params, coerce_float=coerce_float,
            parse_dates=parse_dates, chunksize=chunksize, dtype=dtype,
            dtype_backend=dtype_backend
        )

@overload
def read_sql(
    sql: SQLQueryT,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List, Tuple, Dict]] = ...,
    parse_dates: Optional[ParseDatesT] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: Optional[int] = ...,
    dtype_backend: DtypeBackendT = ...,
    dtype: Optional[DtypeArg] = None,
) -> DataFrame: ...

@overload
def read_sql(
    sql: SQLQueryT,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List, Tuple, Dict]] = ...,
    parse_dates: Optional[ParseDatesT] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: Optional[int] = ...,
    dtype_backend: DtypeBackendT = ...,
    dtype: Optional[DtypeArg] = None,
) -> Iterator[DataFrame]: ...

def read_sql(
    sql: SQLQueryT,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    params: Optional[Union[List, Tuple, Dict]] = None,
    parse_dates: Optional[ParseDatesT] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    dtype_backend: Union[DtypeBackendT, Literal[lib.no_default]] = lib.no_default,
    dtype: Optional[DtypeArg] = None,
) -> Union[DataFrame, Iterator[DataFrame]]:
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = 'numpy'
    assert dtype_backend is not lib.no_default
    with pandasSQL_builder(con) as pandas_sql:
        if isinstance(pandas_sql, SQLiteDatabase):
            return pandas_sql.read_query(
                sql, index_col=index_col, params=params, coerce_float=coerce_float,
                parse_dates=parse_dates, chunksize=chunksize, dtype_backend=dtype_backend,
                dtype=dtype
            )
        try:
            _is_table_name = pandas_sql.has_table(sql)
        except Exception:
            _is_table_name = False
        if _is_table_name:
            return pandas_sql.read_table(
                sql, index_col=index_col, coerce_float=coerce_float,
                parse_dates=parse_dates, columns=columns, chunksize=chunksize,
                dtype_backend=dtype_backend
            )
        else:
            return pandas_sql.read_query(
                sql, index_col=index_col, params=params, coerce_float=coerce_float,
                parse_dates=parse_dates, chunksize=chunksize, dtype_backend=dtype_backend,
                dtype=dtype
            )

def to_sql(
    frame: Union[DataFrame, Series],
    name: str,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    schema: Optional[str] = None,
    if_exists: IfExistsT = 'fail',
    index: bool = True,
    index_label: Optional[Union[str, List[str]]] = None,
    chunksize: Optional[int] = None,
    dtype: Optional[Union[Dict[str, Any], Any]] = None,
    method: MethodT = None,
    engine: EngineT = 'auto',
    **engine_kwargs: Any,
) -> Optional[int]:
    if if_exists not in ('fail', 'replace', 'append'):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")
    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError("'frame' argument should be either a Series or a DataFrame")
    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        return pandas_sql.to_sql(
            frame, name, if_exists=if_exists, index=index, index_label=index_label,
            schema=schema, chunksize=chunksize, dtype=dtype, method=method,
            engine=engine, **engine_kwargs
        )

def has_table(
    table_name: str,
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    schema: Optional[str] = None,
) -> bool:
    with pandasSQL_builder(con, schema=schema) as pandas_sql:
        return pandas_sql.has_table(table_name)

table_exists = has_table

def pandasSQL_builder(
    con: Union[str, SQLAlchemyConnection, SQLAlchemyEngine, SQLite3Connection, ADBCConnection],
    schema: Optional[str] = None,
    need_transaction: bool = False,
) -> Union[SQLDatabase, SQLiteDatabase, ADBCDatabase]:
    import sqlite3
    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)
    sqlalchemy = import_optional_dependency('sqlalchemy', errors='ignore')
    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError('Using URI string without sqlalchemy installed.')
    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Connectable)):
        return SQLDatabase(con, schema, need_transaction)
    adbc = import_optional_dependency('adbc_driver_manager.dbapi', errors='ignore')
    if