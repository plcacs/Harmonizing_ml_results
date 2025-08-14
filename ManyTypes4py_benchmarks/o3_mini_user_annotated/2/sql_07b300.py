#!/usr/bin/env python3
"""
A module for SQL query wrappers and DataFrame I/O.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, time
from functools import partial
import re
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    overload,
    Union,
)

import numpy as np

from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError, DatabaseError
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import is_dict_like, is_list_like, is_object_dtype, is_string_dtype
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

if False:  # TYPE_CHECKING
    from collections.abc import Callable, Generator, Iterator, Mapping
    from sqlalchemy import Table
    from sqlalchemy.sql.expression import Select, TextClause
    from pandas._typing import DtypeArg, DtypeBackend, IndexLabel, Self
    from pandas import Index

# Define a type alias for parse_dates parameter.
ParseDatesArg = Union[
    bool,
    None,
    str,
    List[str],
    Dict[str, str],
    Dict[str, Dict[str, Any]]
]


def _process_parse_dates_argument(parse_dates: ParseDatesArg) -> List[Any]:
    """Process parse_dates argument for read_sql functions."""
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []
    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
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


def _parse_date_columns(data_frame: DataFrame, parse_dates: ParseDatesArg) -> DataFrame:
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
    parse_dates = _process_parse_dates_argument(parse_dates)
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        if isinstance(df_col.dtype, DatetimeTZDtype) or col_name in parse_dates:
            try:
                fmt = parse_dates[col_name]  # type: ignore
            except (KeyError, TypeError):
                fmt = None
            data_frame.isetitem(i, _handle_date_column(df_col, format=fmt))
    return data_frame


def _convert_arrays_to_dataframe(
    data: Any,
    columns: List[str],
    coerce_float: bool = True,
    dtype_backend: Union[str, Any] = "numpy"
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
        result_arrays: List[ArrowExtensionArray] = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == "string":
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        arrays = result_arrays  # type: ignore[assignment]
    if arrays:
        return DataFrame._from_arrays(
            arrays, columns=columns, index=list(range(idx_len)), verify_integrity=False
        )
    else:
        return DataFrame(columns=columns)


def _wrap_result(
    data: Any,
    columns: List[str],
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    parse_dates: ParseDatesArg = None,
    dtype: Optional[Any] = None,
    dtype_backend: Union[Any, Literal["numpy"]] = "numpy"
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
    parse_dates: ParseDatesArg = None,
    dtype: Optional[Any] = None,
    dtype_backend: Union[Any, Literal["numpy"]] = "numpy"
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
    con: Any,
    schema: Optional[str] = ...,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    parse_dates: Union[List[str], Dict[str, str], Dict[str, Dict[str, Any]], None] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: None = ...,
    dtype_backend: Union[Any, Literal["numpy"]] = ...,
) -> DataFrame: ...
    
@overload
def read_sql_table(
    table_name: str,
    con: Any,
    schema: Optional[str] = ...,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    parse_dates: Union[List[str], Dict[str, str], Dict[str, Dict[str, Any]], None] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: int = ...,
    dtype_backend: Union[Any, Literal["numpy"]] = ...,
) -> Iterator[DataFrame]: ...


def read_sql_table(
    table_name: str,
    con: Any,
    schema: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    parse_dates: Union[List[str], Dict[str, str], Dict[str, Dict[str, Any]], None] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    dtype_backend: Union[Any, Literal["numpy"]] = lib.no_default,
) -> Union[DataFrame, Iterator[DataFrame]]:
    """
    Read SQL database table into a DataFrame.
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        if not pandas_sql.has_table(table_name, schema):
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
    sql: Union[str, Any],
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = ...,
    parse_dates: Union[List[str], Dict[str, str], Dict[str, Dict[str, Any]], None] = ...,
    chunksize: None = ...,
    dtype: Optional[Any] = ...,
    dtype_backend: Union[Any, Literal["numpy"]] = ...,
) -> DataFrame: ...

@overload
def read_sql_query(
    sql: Union[str, Any],
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = ...,
    parse_dates: Union[List[str], Dict[str, str], Dict[str, Dict[str, Any]], None] = ...,
    chunksize: int = ...,
    dtype: Optional[Any] = ...,
    dtype_backend: Union[Any, Literal["numpy"]] = ...,
) -> Iterator[DataFrame]: ...


def read_sql_query(
    sql: Union[str, Any],
    con: Any,
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = None,
    parse_dates: Union[List[str], Dict[str, str], Dict[str, Dict[str, Any]], None] = None,
    chunksize: Optional[int] = None,
    dtype: Optional[Any] = None,
    dtype_backend: Union[Any, Literal["numpy"]] = lib.no_default,
) -> Union[DataFrame, Iterator[DataFrame]]:
    """
    Read SQL query into a DataFrame.
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
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
    sql: Union[str, Any],
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Any = ...,
    parse_dates: Any = ...,
    columns: List[str] = ...,
    chunksize: None = ...,
    dtype_backend: Union[Any, Literal["numpy"]] = ...,
    dtype: Optional[Any] = None,
) -> DataFrame: ...
    
@overload
def read_sql(
    sql: Union[str, Any],
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Any = ...,
    parse_dates: Any = ...,
    columns: List[str] = ...,
    chunksize: int = ...,
    dtype_backend: Union[Any, Literal["numpy"]] = ...,
    dtype: Optional[Any] = None,
) -> Iterator[DataFrame]: ...


def read_sql(
    sql: Union[str, Any],
    con: Any,
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    params: Any = None,
    parse_dates: Any = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    dtype_backend: Union[Any, Literal["numpy"]] = lib.no_default,
    dtype: Optional[Any] = None,
) -> Union[DataFrame, Iterator[DataFrame]]:
    """
    Read SQL query or database table into a DataFrame.
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
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
                dtype_backend=dtype_backend,
                dtype=dtype,
            )


def to_sql(
    frame: Union[DataFrame, Series],
    name: str,
    con: Any,
    schema: Optional[str] = None,
    if_exists: Literal["fail", "replace", "append"] = "fail",
    index: bool = True,
    index_label: Optional[Any] = None,
    chunksize: Optional[int] = None,
    dtype: Optional[Any] = None,
    method: Optional[Union[Literal["multi"], Callable]] = None,
    engine: str = "auto",
    **engine_kwargs: Any,
) -> Optional[int]:
    """
    Write records stored in a DataFrame to a SQL database.
    """
    if if_exists not in ("fail", "replace", "append"):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")
    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError("'frame' argument should be either a Series or a DataFrame")
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


def has_table(table_name: str, con: Any, schema: Optional[str] = None) -> bool:
    """
    Check if DataBase has named table.
    """
    with pandasSQL_builder(con, schema=schema) as pandas_sql:
        return pandas_sql.has_table(table_name)


table_exists = has_table


def pandasSQL_builder(
    con: Any,
    schema: Optional[str] = None,
    need_transaction: bool = False,
) -> PandasSQL:
    """
    Convenience function to return the correct PandasSQL subclass.
    """
    import sqlite3
    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)
    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")
    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError("Using URI string without sqlalchemy installed.")
    if sqlalchemy is not None and isinstance(con, (str, getattr(sqlalchemy.engine, "Connectable", object))):
        return SQLDatabase(con, schema, need_transaction)
    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    if adbc and hasattr(con, "cursor"):
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
    """
    For mapping Pandas tables to SQL tables.
    """

    def __init__(
        self,
        name: str,
        pandas_sql_engine: PandasSQL,
        frame: Optional[DataFrame] = None,
        index: Union[bool, str, List[str], None] = True,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        prefix: str = "pandas",
        index_label: Optional[Any] = None,
        schema: Any = None,
        keys: Any = None,
        dtype: Optional[Any] = None,
    ) -> None:
        self.name: str = name
        self.pd_sql: PandasSQL = pandas_sql_engine
        self.prefix: str = prefix
        self.frame: Optional[DataFrame] = frame
        self.index: Optional[List[str]] = self._index_name(index, index_label)
        self.schema: Any = schema
        self.if_exists: Literal["fail", "replace", "append"] = if_exists
        self.keys: Any = keys
        self.dtype: Optional[Any] = dtype
        if frame is not None:
            self.table: Any = self._create_table_setup()
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
        with self.pd_sql.run_transaction() as conn:
            self.table.create(bind=self.pd_sql.con)

    def create(self) -> None:
        if self.exists():
            if self.if_exists == "fail":
                raise ValueError(f"Table '{self.name}' already exists.")
            if self.if_exists == "replace":
                self.pd_sql.drop_table(self.name, self.schema)
                self._execute_create()
            elif self.if_exists == "append":
                pass
            else:
                raise ValueError(f"'{self.if_exists}' is not valid for if_exists")
        else:
            self._execute_create()

    def _execute_insert(self, conn: Any, keys: List[str], data_iter: Iterator[List[Any]]) -> int:
        data = [dict(zip(keys, row)) for row in data_iter]
        result = conn.execute(self.table.insert(), data)
        return result.rowcount

    def _execute_insert_multi(self, conn: Any, keys: List[str], data_iter: Iterator[List[Any]]) -> int:
        from sqlalchemy import insert
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(self.table).values(data)
        result = conn.execute(stmt)
        return result.rowcount

    def insert_data(self) -> tuple[List[str], List[np.ndarray]]:
        if self.index is not None:
            temp = self.frame.copy(deep=False)  # type: ignore
            temp.index.names = self.index
            try:
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f"duplicate name in index/columns: {err}") from err
        else:
            temp = self.frame  # type: ignore
        column_names: List[str] = list(map(str, temp.columns))
        ncols: int = len(column_names)
        data_list: List[np.ndarray] = [None] * ncols  # type: ignore
        for i, (_, ser) in enumerate(temp.items()):
            if ser.dtype.kind == "M":
                if isinstance(ser._values, ArrowExtensionArray):
                    import pyarrow as pa
                    if pa.types.is_date(ser.dtype.pyarrow_dtype):
                        d = ser._values.to_numpy(dtype=object)
                    else:
                        d = ser.dt.to_pydatetime()._values
                else:
                    d = ser._values.to_pydatetime()
            elif ser.dtype.kind == "m":
                vals = ser._values
                if isinstance(vals, ArrowExtensionArray):
                    vals = vals.to_numpy(dtype=np.dtype("m8[ns]"))
                d = vals.view("i8").astype(object)
            else:
                d = ser._values.astype(object)
            assert isinstance(d, np.ndarray), type(d)
            if ser._can_hold_na:
                mask = isna(d)
                d[mask] = None
            data_list[i] = d
        return column_names, data_list

    def insert(
        self,
        chunksize: Optional[int] = None,
        method: Optional[Union[Literal["multi"], Callable]] = None,
    ) -> Optional[int]:
        if method is None:
            exec_insert: Callable = self._execute_insert
        elif method == "multi":
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f"Invalid parameter `method`: {method}")
        keys, data_list = self.insert_data()
        nrows: int = len(self.frame)  # type: ignore
        if nrows == 0:
            return 0
        if chunksize is None:
            chunksize = nrows
        elif chunksize == 0:
            raise ValueError("chunksize argument should be non-zero")
        chunks: int = (nrows // chunksize) + 1
        total_inserted: Optional[int] = None
        with self.pd_sql.run_transaction() as conn:
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break
                chunk_iter = zip(*(arr[start_i:end_i] for arr in data_list))
                num_inserted = exec_insert(conn, keys, chunk_iter)
                if num_inserted is not None:
                    if total_inserted is None:
                        total_inserted = num_inserted
                    else:
                        total_inserted += num_inserted
        return total_inserted

    def _query_iterator(
        self,
        result: Any,
        exit_stack: ExitStack,
        chunksize: int,
        columns: List[str],
        coerce_float: bool = True,
        parse_dates: ParseDatesArg = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Generator[DataFrame, None, None]:
        has_read_data: bool = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield DataFrame.from_records([], columns=columns, coerce_float=coerce_float)
                    break
                has_read_data = True
                self.frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)
                self._harmonize_columns(parse_dates=parse_dates, dtype_backend=dtype_backend)
                if self.index is not None:
                    self.frame.set_index(self.index, inplace=True)
                yield self.frame

    def read(
        self,
        exit_stack: ExitStack,
        coerce_float: bool = True,
        parse_dates: ParseDatesArg = None,
        columns: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        from sqlalchemy import select
        if columns is not None and len(columns) > 0:
            cols = [self.table.c[n] for n in columns]
            if self.index is not None:
                for idx in self.index[::-1]:
                    cols.insert(0, self.table.c[idx])
            sql_select = select(*cols)
        else:
            sql_select = select(self.table)
        result = self.pd_sql.execute(sql_select)
        column_names: List[str] = result.keys()
        if chunksize is not None:
            return self._query_iterator(
                result,
                exit_stack,
                chunksize,
                column_names,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            self.frame = _convert_arrays_to_dataframe(data, column_names, coerce_float, dtype_backend)
            self._harmonize_columns(parse_dates=parse_dates, dtype_backend=dtype_backend)
            if self.index is not None:
                self.frame.set_index(self.index, inplace=True)
            return self.frame

    def _index_name(self, index: Union[bool, str, List[str], None], index_label: Any) -> Optional[List[str]]:
        if index is True:
            nlevels = self.frame.index.nlevels  # type: ignore
            if index_label is not None:
                if not isinstance(index_label, list):
                    index_label = [index_label]
                if len(index_label) != nlevels:
                    raise ValueError("Length of 'index_label' should match number of levels, which is {nlevels}")
                return index_label
            if nlevels == 1 and "index" not in self.frame.columns and self.frame.index.name is None:  # type: ignore
                return ["index"]
            else:
                return com.fill_missing_names(self.frame.index.names)  # type: ignore
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None

    def _get_column_names_and_types(self, dtype_mapper: Callable[[Any], Any]) -> List[tuple[str, Any, bool]]:
        column_names_and_types: List[tuple[str, Any, bool]] = []
        if self.index is not None:
            for i, idx_label in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))  # type: ignore
                column_names_and_types.append((str(idx_label), idx_type, True))
        column_names_and_types += [
            (str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False)
            for i in range(len(self.frame.columns))
        ]
        return column_names_and_types

    def _create_table_setup(self) -> Any:
        from sqlalchemy import Column, PrimaryKeyConstraint, Table
        from sqlalchemy.schema import MetaData
        column_names_and_types = self._get_column_names_and_types(self._sqlalchemy_type)
        columns: List[Any] = [Column(name, typ, index=is_index) for name, typ, is_index in column_names_and_types]
        if self.keys is not None:
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=self.name + "_pk")
            columns.append(pkc)
        schema = self.schema or self.pd_sql.meta.schema
        meta = MetaData()
        return Table(self.name, meta, *columns, schema=schema)

    def _harmonize_columns(
        self,
        parse_dates: ParseDatesArg = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> None:
        parse_dates = _process_parse_dates_argument(parse_dates)
        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                df_col = self.frame[col_name]
                if col_name in parse_dates:
                    try:
                        fmt = parse_dates[col_name]  # type: ignore
                    except TypeError:
                        fmt = None
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue
                col_type = self._get_dtype(sql_col.type)
                if col_type is datetime or col_type is date or col_type is DatetimeTZDtype:
                    utc = col_type is DatetimeTZDtype
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                elif dtype_backend == "numpy" and col_type is float:
                    self.frame[col_name] = df_col.astype(col_type)
                elif (using_string_dtype() and is_string_dtype(col_type) and is_object_dtype(self.frame[col_name])):
                    self.frame[col_name] = df_col.astype(col_type)
                elif dtype_backend == "numpy" and len(df_col) == df_col.count():
                    if col_type is np.dtype("int64") or col_type is bool:
                        self.frame[col_name] = df_col.astype(col_type)
            except KeyError:
                pass

    def _sqlalchemy_type(self, col: Union[Series, Any]) -> Any:
        dtype: Any = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]
        col_type = lib.infer_dtype(col, skipna=True)
        from sqlalchemy.types import TIMESTAMP, BigInteger, Boolean, Date, DateTime, Float, Integer, SmallInteger, Text, Time
        if col_type in ("datetime64", "datetime"):
            try:
                if col.dt.tz is not None:  # type: ignore
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                if getattr(col, "tz", None) is not None:
                    return TIMESTAMP(timezone=True)
            return DateTime
        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            return BigInteger
        elif col_type == "floating":
            if col.dtype == "float32":
                return Float(precision=23)
            else:
                return Float(precision=53)
        elif col_type == "integer":
            if col.dtype.name.lower() in ("int8", "uint8", "int16"):
                return SmallInteger
            elif col.dtype.name.lower() in ("uint16", "int32"):
                return Integer
            elif col.dtype.name.lower() == "uint64":
                raise ValueError("Unsigned 64 bit integer datatype is not supported")
            else:
                return BigInteger
        elif col_type == "boolean":
            return Boolean
        elif col_type == "date":
            return Date
        elif col_type == "time":
            return Time
        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")
        return Text

    def _get_dtype(self, sqltype: Any) -> Any:
        from sqlalchemy.types import TIMESTAMP, Boolean, Date, DateTime, Float, Integer, String
        if isinstance(sqltype, Float):
            return float
        elif isinstance(sqltype, Integer):
            return np.dtype("int64")
        elif isinstance(sqltype, TIMESTAMP):
            if not sqltype.timezone:
                return datetime
            return DatetimeTZDtype
        elif isinstance(sqltype, DateTime):
            return datetime
        elif isinstance(sqltype, Date):
            return date
        elif isinstance(sqltype, Boolean):
            return bool
        elif isinstance(sqltype, String):
            if using_string_dtype():
                return StringDtype(na_value=np.nan)
        return object


class PandasSQL(PandasObject, ABC):
    """
    Abstract base class for SQL conversions.
    """

    def __enter__(self) -> PandasSQL:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def read_table(
        self,
        table_name: str,
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        raise NotImplementedError

    @abstractmethod
    def read_query(
        self,
        sql: Union[str, Any],
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        params: Any = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        pass

    @abstractmethod
    def to_sql(
        self,
        frame: DataFrame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label: Any = None,
        schema: Any = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        method: Optional[Union[Literal["multi"], Callable]] = None,
        engine: str = "auto",
        **engine_kwargs: Any,
    ) -> Optional[int]:
        pass

    @abstractmethod
    def execute(self, sql: Union[str, Any], params: Any = None) -> Any:
        pass

    @abstractmethod
    def has_table(self, name: str, schema: Optional[str] = None) -> bool:
        pass

    @abstractmethod
    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: Optional[List[str]] = None,
        dtype: Optional[Any] = None,
        schema: Optional[str] = None,
    ) -> str:
        pass


class BaseEngine:
    def insert_records(
        self,
        table: SQLTable,
        con: Any,
        frame: DataFrame,
        name: str,
        index: Union[bool, str, List[str], None] = True,
        schema: Any = None,
        chunksize: Optional[int] = None,
        method: Optional[Union[Literal["multi"], Callable]] = None,
        **engine_kwargs: Any,
    ) -> Optional[int]:
        raise AbstractMethodError(self)


class SQLAlchemyEngine(BaseEngine):
    def __init__(self) -> None:
        import_optional_dependency("sqlalchemy", extra="sqlalchemy is required for SQL support.")

    def insert_records(
        self,
        table: SQLTable,
        con: Any,
        frame: DataFrame,
        name: str,
        index: Union[bool, str, List[str], None] = True,
        schema: Any = None,
        chunksize: Optional[int] = None,
        method: Optional[Union[Literal["multi"], Callable]] = None,
        **engine_kwargs: Any,
    ) -> Optional[int]:
        from sqlalchemy import exc
        try:
            return table.insert(chunksize=chunksize, method=method)
        except exc.StatementError as err:
            msg = r"""(\(1054, "Unknown column 'inf(e0)?' in 'field list'"\))(?#
            )|inf can not be used with MySQL"""
            err_text = str(err.orig)
            if re.search(msg, err_text):
                raise ValueError("inf cannot be used with MySQL") from err
            raise err


def get_engine(engine: str) -> BaseEngine:
    """return our implementation"""
    if engine == "auto":
        engine = get_option("io.sql.engine")
    if engine == "auto":
        engine_classes = [SQLAlchemyEngine]
        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)
        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'sqlalchemy'.\n"
            "A suitable version of sqlalchemy is required for sql I/O support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )
    if engine == "sqlalchemy":
        return SQLAlchemyEngine()
    raise ValueError("engine must be one of 'auto', 'sqlalchemy'")


class SQLDatabase(PandasSQL):
    def __init__(self, con: Any, schema: Optional[str] = None, need_transaction: bool = False) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        from sqlalchemy.schema import MetaData
        self.exit_stack: ExitStack = ExitStack()
        if isinstance(con, str):
            con = create_engine(con)
            self.exit_stack.callback(con.dispose)
        if isinstance(con, Engine):
            con = self.exit_stack.enter_context(con.connect())
        if need_transaction and not con.in_transaction():
            self.exit_stack.enter_context(con.begin())
        self.con: Any = con
        self.meta = MetaData(schema=schema)
        self.returns_generator: bool = False

    def __exit__(self, *args: Any) -> None:
        if not self.returns_generator:
            self.exit_stack.close()

    @contextmanager
    def run_transaction(self) -> Generator[Any, None, None]:
        if not self.con.in_transaction():
            with self.con.begin():
                yield self.con
        else:
            yield self.con

    def execute(self, sql: Union[str, Any], params: Any = None) -> Any:
        from sqlalchemy.exc import SQLAlchemyError
        args: List[Any] = [] if params is None else [params]
        if isinstance(sql, str):
            execute_function = self.con.exec_driver_sql
        else:
            execute_function = self.con.execute
        try:
            return execute_function(sql, *args)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Execution failed on sql '{sql}': {exc}") from exc

    def read_table(
        self,
        table_name: str,
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        self.meta.reflect(bind=self.con, only=[table_name], views=True)
        table = SQLTable(table_name, self, index=index_col, schema=schema)
        if chunksize is not None:
            self.returns_generator = True
        return table.read(
            self.exit_stack,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    @staticmethod
    def _query_iterator(
        result: Any,
        exit_stack: ExitStack,
        chunksize: int,
        columns: List[str],
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        dtype: Optional[Any] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Generator[DataFrame, None, None]:
        has_read_data: bool = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    self_result = _wrap_result(
                        [],
                        columns,
                        index_col=index_col,
                        coerce_float=coerce_float,
                        parse_dates=parse_dates,
                        dtype=dtype,
                        dtype_backend=dtype_backend,
                    )
                    yield self_result
                    break
                has_read_data = True
                yield _wrap_result(
                    data,
                    columns,
                    index_col=index_col,
                    coerce_float=coerce_float,
                    parse_dates=parse_dates,
                    dtype=dtype,
                    dtype_backend=dtype_backend,
                )

    def read_query(
        self,
        sql: str,
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        params: Any = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        result = self.execute(sql, params)
        columns: List[str] = list(result.keys())
        if chunksize is not None:
            self.returns_generator = True
            return SQLDatabase._query_iterator(
                result,
                self.exit_stack,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    read_sql = read_query

    def prep_table(
        self,
        frame: DataFrame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: Union[bool, str, List[str], None] = True,
        index_label: Any = None,
        schema: Any = None,
        dtype: Optional[Any] = None,
    ) -> SQLTable:
        if dtype:
            if not is_dict_like(dtype):
                dtype = {col_name: dtype for col_name in frame}  # type: ignore[misc]
            else:
                dtype = cast(dict, dtype)
            from sqlalchemy.types import TypeEngine
            for col, my_type in dtype.items():
                if isinstance(my_type, type) and issubclass(my_type, TypeEngine):
                    pass
                elif isinstance(my_type, TypeEngine):
                    pass
                else:
                    raise ValueError(f"The type of {col} is not a SQLAlchemy type")
        table = SQLTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )
        table.create()
        return table

    def check_case_sensitive(
        self,
        name: str,
        schema: Optional[str],
    ) -> None:
        if not name.isdigit() and not name.islower():
            from sqlalchemy import inspect as sqlalchemy_inspect
            insp = sqlalchemy_inspect(self.con)
            table_names = insp.get_table_names(schema=schema or self.meta.schema)
            if name not in table_names:
                msg = (
                    f"The provided table name '{name}' is not found exactly as "
                    "such in the database after writing the table, possibly "
                    "due to case sensitivity issues. Consider using lower "
                    "case table names."
                )
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )

    def to_sql(
        self,
        frame: DataFrame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label: Any = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        method: Optional[Union[Literal["multi"], Callable]] = None,
        engine: str = "auto",
        **engine_kwargs: Any,
    ) -> Optional[int]:
        sql_engine = get_engine(engine)
        table = self.prep_table(
            frame=frame,
            name=name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )
        total_inserted = sql_engine.insert_records(
            table=table,
            con=self.con,
            frame=frame,
            name=name,
            index=index,
            schema=schema,
            chunksize=chunksize,
            method=method,
            **engine_kwargs,
        )
        self.check_case_sensitive(name=name, schema=schema)
        return total_inserted

    @property
    def tables(self) -> Any:
        return self.meta.tables

    def has_table(self, name: str, schema: Optional[str] = None) -> bool:
        from sqlalchemy import inspect as sqlalchemy_inspect
        insp = sqlalchemy_inspect(self.con)
        return insp.has_table(name, schema or self.meta.schema)

    def get_table(self, table_name: str, schema: Optional[str] = None) -> Any:
        from sqlalchemy import Numeric, Table
        schema = schema or self.meta.schema
        tbl = Table(table_name, self.meta, autoload_with=self.con, schema=schema)
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False
        return tbl

    def drop_table(self, table_name: str, schema: Optional[str] = None) -> None:
        schema = schema or self.meta.schema
        if self.has_table(table_name, schema):
            self.meta.reflect(bind=self.con, only=[table_name], schema=schema, views=True)
            with self.run_transaction():
                self.get_table(table_name, schema).drop(bind=self.con)
            self.meta.clear()

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: Optional[List[str]] = None,
        dtype: Optional[Any] = None,
        schema: Optional[str] = None,
    ) -> str:
        table = SQLTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


class ADBCDatabase(PandasSQL):
    def __init__(self, con: Any) -> None:
        self.con: Any = con

    @contextmanager
    def run_transaction(self) -> Generator[Any, None, None]:
        with self.con.cursor() as cur:
            try:
                yield cur
            except Exception:
                self.con.rollback()
                raise
            self.con.commit()

    def execute(self, sql: Union[str, Any], params: Any = None) -> Any:
        from adbc_driver_manager import Error
        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args: List[Any] = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Error as exc:
            try:
                self.con.rollback()
            except Error as inner_exc:
                ex = DatabaseError(f"Execution failed on sql: {sql}\n{exc}\nunable to rollback")
                raise ex from inner_exc
            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    def read_table(
        self,
        table_name: str,
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> DataFrame:
        if coerce_float is not True:
            raise NotImplementedError("'coerce_float' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        if columns:
            index_select = maybe_make_list(index_col) if index_col else []
            to_select = index_select + columns
            select_list = ", ".join(f'"{x}"' for x in to_select)
        else:
            select_list = "*"
        if schema:
            stmt = f"SELECT {select_list} FROM {schema}.{table_name}"
        else:
            stmt = f"SELECT {select_list} FROM {table_name}"
        with self.execute(stmt) as cur:
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)
        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
        )

    def read_query(
        self,
        sql: str,
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        params: Any = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> DataFrame:
        if coerce_float is not True:
            raise NotImplementedError("'coerce_float' is not implemented for ADBC drivers")
        if params:
            raise NotImplementedError("'params' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        with self.execute(sql) as cur:
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)
        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
            dtype=dtype,
        )
    read_sql = read_query

    def to_sql(
        self,
        frame: DataFrame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label: Any = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        method: Optional[Union[Literal["multi"], Callable]] = None,
        engine: str = "auto",
        **engine_kwargs: Any,
    ) -> Optional[int]:
        import pyarrow as pa
        from adbc_driver_manager import Error
        if index_label:
            raise NotImplementedError("'index_label' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        if dtype:
            raise NotImplementedError("'dtype' is not implemented for ADBC drivers")
        if method:
            raise NotImplementedError("'method' is not implemented for ADBC drivers")
        if engine != "auto":
            raise NotImplementedError("engine != 'auto' not implemented for ADBC drivers")
        table_name_full = f"{schema}.{name}" if schema else name
        mode: str = "create"
        if self.has_table(name, schema):
            if if_exists == "fail":
                raise ValueError(f"Table '{table_name_full}' already exists.")
            elif if_exists == "replace":
                sql_statement = f"DROP TABLE {table_name_full}"
                self.execute(sql_statement).close()
            elif if_exists == "append":
                mode = "append"
        try:
            tbl = pa.Table.from_pandas(frame, preserve_index=index)
        except pa.ArrowNotImplementedError as exc:
            raise ValueError("datatypes not supported") from exc
        with self.con.cursor() as cur:
            try:
                total_inserted = cur.adbc_ingest(
                    table_name=name, data=tbl, mode=mode, db_schema_name=schema
                )
            except Error as exc:
                raise DatabaseError(f"Failed to insert records on table={name} with {mode=}") from exc
        self.con.commit()
        return total_inserted

    def has_table(self, name: str, schema: Optional[str] = None) -> bool:
        meta = self.con.adbc_get_objects(
            db_schema_filter=schema, table_name_filter=name
        ).read_all()
        for catalog_schema in meta["catalog_db_schemas"].to_pylist():
            if not catalog_schema:
                continue
            for schema_record in catalog_schema:
                if not schema_record:
                    continue
                for table_record in schema_record["db_schema_tables"]:
                    if table_record["table_name"] == name:
                        return True
        return False

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: Optional[List[str]] = None,
        dtype: Optional[Any] = None,
        schema: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("not implemented for adbc")


_SQL_TYPES: Dict[str, str] = {
    "string": "TEXT",
    "floating": "REAL",
    "integer": "INTEGER",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "time": "TIME",
    "boolean": "INTEGER",
}


def _get_unicode_name(name: Any) -> str:
    try:
        uname = str(name).encode("utf-8", "strict").decode("utf-8")
    except UnicodeError as err:
        raise ValueError(f"Cannot convert identifier to UTF-8: '{name}'") from err
    return uname


def _get_valid_sqlite_name(name: Any) -> str:
    uname = _get_unicode_name(name)
    if not len(uname):
        raise ValueError("Empty table or column name specified")
    if "\x00" in uname:
        raise ValueError("SQLite identifier cannot contain NULs")
    return '"' + uname.replace('"', '""') + '"'


class SQLiteTable(SQLTable):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._register_date_adapters()

    def _register_date_adapters(self) -> None:
        import sqlite3
        def _adapt_time(t: time) -> str:
            return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}"
        adapt_date_iso = lambda val: val.isoformat()
        adapt_datetime_iso = lambda val: val.isoformat(" ")
        sqlite3.register_adapter(time, _adapt_time)
        sqlite3.register_adapter(date, adapt_date_iso)
        sqlite3.register_adapter(datetime, adapt_datetime_iso)
        convert_date = lambda val: date.fromisoformat(val.decode())
        convert_timestamp = lambda val: datetime.fromisoformat(val.decode())
        sqlite3.register_converter("date", convert_date)
        sqlite3.register_converter("timestamp", convert_timestamp)

    def sql_schema(self) -> str:
        return str(";\n".join(self.table))  # type: ignore

    def _execute_create(self) -> None:
        with self.pd_sql.run_transaction() as cur:
            for stmt in self.table:  # type: ignore
                cur.execute(stmt)

    def insert_statement(self, *, num_rows: int) -> str:
        names = list(map(str, self.frame.columns))  # type: ignore
        wld = "?"
        escape = _get_valid_sqlite_name
        if self.index is not None:
            for idx in self.index[::-1]:
                names.insert(0, idx)
        bracketed_names = [escape(column) for column in names]
        col_names = ",".join(bracketed_names)
        row_wildcards = ",".join([wld] * len(names))
        wildcards = ",".join([f"({row_wildcards})" for _ in range(num_rows)])
        insert_statement = f"INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}"
        return insert_statement

    def _execute_insert(self, conn: Any, keys: List[str], data_iter: Iterator[List[Any]]) -> int:
        from sqlite3 import Error
        data_list = list(data_iter)
        try:
            conn.executemany(self.insert_statement(num_rows=1), data_list)
        except Error as exc:
            raise DatabaseError("Execution failed") from exc
        return conn.rowcount

    def _execute_insert_multi(self, conn: Any, keys: List[str], data_iter: Iterator[List[Any]]) -> int:
        data_list = list(data_iter)
        flattened_data = [x for row in data_list for x in row]
        conn.execute(self.insert_statement(num_rows=len(data_list)), flattened_data)
        return conn.rowcount

    def _create_table_setup(self) -> List[str]:
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        escape = _get_valid_sqlite_name
        create_tbl_stmts = [escape(cname) + " " + ctype for cname, ctype, _ in column_names_and_types]
        if self.keys is not None and len(self.keys):
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ", ".join([escape(c) for c in keys])
            create_tbl_stmts.append(f"CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})")
        schema_name = (self.schema + ".") if self.schema else ""
        create_stmts = [
            "CREATE TABLE " + schema_name + escape(self.name) + " (\n" + ",\n  ".join(create_tbl_stmts) + "\n)"
        ]
        ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
        if len(ix_cols):
            cnames = "_".join(ix_cols)
            cnames_br = ",".join([escape(c) for c in ix_cols])
            create_stmts.append(
                "CREATE INDEX " + escape("ix_" + self.name + "_" + cnames) + "ON " + escape(self.name) + " (" + cnames_br + ")"
            )
        return create_stmts

    def _sql_type_name(self, col: Any) -> str:
        dtype: Any = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]
        col_type = lib.infer_dtype(col, skipna=True)
        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            col_type = "integer"
        elif col_type == "datetime64":
            col_type = "datetime"
        elif col_type == "empty":
            col_type = "string"
        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")
        if col_type not in _SQL_TYPES:
            col_type = "string"
        return _SQL_TYPES[col_type]


class SQLiteDatabase(PandasSQL):
    def __init__(self, con: Any) -> None:
        self.con: Any = con

    @contextmanager
    def run_transaction(self) -> Generator[Any, None, None]:
        cur = self.con.cursor()
        try:
            yield cur
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        finally:
            cur.close()

    @staticmethod
    def _query_iterator(
        cursor: Any,
        chunksize: int,
        columns: List[str],
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        dtype: Optional[Any] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Generator[DataFrame, None, None]:
        has_read_data: bool = False
        while True:
            data = cursor.fetchmany(chunksize)
            if isinstance(data, tuple):
                data = list(data)
            if not data:
                cursor.close()
                if not has_read_data:
                    result = DataFrame.from_records([], columns=columns, coerce_float=coerce_float)
                    if dtype:
                        result = result.astype(dtype)
                    yield result
                break
            has_read_data = True
            yield _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )

    def read_query(
        self,
        sql: Union[str, Any],
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        params: Any = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        dtype_backend: Union[Any, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        cursor = self.execute(sql, params)
        columns: List[str] = [col_desc[0] for col_desc in cursor.description]
        if chunksize is not None:
            return self._query_iterator(
                cursor,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = self._fetchall_as_list(cursor)
            cursor.close()
            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    def _fetchall_as_list(self, cur: Any) -> List[Any]:
        result = cur.fetchall()
        if not isinstance(result, list):
            result = list(result)
        return result

    def to_sql(
        self,
        frame: DataFrame,
        name: str,
        if_exists: str = "fail",
        index: bool = True,
        index_label: Any = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Any] = None,
        method: Optional[Union[Literal["multi"], Callable]] = None,
        engine: str = "auto",
        **engine_kwargs: Any,
    ) -> Optional[int]:
        if dtype:
            if not is_dict_like(dtype):
                dtype = {col_name: dtype for col_name in frame}  # type: ignore[misc]
            else:
                dtype = cast(dict, dtype)
            for col, my_type in dtype.items():
                if not isinstance(my_type, str):
                    raise ValueError(f"{col} ({my_type}) not a string")
        table = SQLiteTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            dtype=dtype,
        )
        table.create()
        return table.insert(chunksize, method)

    def has_table(self, name: str, schema: Optional[str] = None) -> bool:
        wld = "?"
        query = f"""
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type IN ('table', 'view')
            AND name={wld};
        """
        return len(self.execute(query, [name]).fetchall()) > 0

    def get_table(self, table_name: str, schema: Optional[str] = None) -> None:
        return None

    def drop_table(self, name: str, schema: Optional[str] = None) -> None:
        drop_sql = f"DROP TABLE {_get_valid_sqlite_name(name)}"
        self.execute(drop_sql)

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: Any = None,
        dtype: Optional[Any] = None,
        schema: Optional[str] = None,
    ) -> str:
        table = SQLiteTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


def get_schema(
    frame: DataFrame,
    name: str,
    keys: Any = None,
    con: Any = None,
    dtype: Optional[Any] = None,
    schema: Optional[str] = None,
) -> str:
    with pandasSQL_builder(con=con) as pandas_sql:
        return pandas_sql._create_sql_schema(
            frame, name, keys=keys, dtype=dtype, schema=schema
        )