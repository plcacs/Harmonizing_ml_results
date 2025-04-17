from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, time
from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
    from collections.abc import (
        Callable,
        Generator,
        Iterator,
        Mapping,
    )
    from sqlalchemy import Table
    from sqlalchemy.sql.expression import Select, TextClause
    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        IndexLabel,
        Self,
    )
    from pandas import Index

# -----------------------------------------------------------------------------
# -- Helper functions

def _process_parse_dates_argument(
    parse_dates: Union[bool, List[str], Dict[str, Any],
) -> List[Union[str, Dict[str, Any]]]:
    """Process parse_dates argument for read_sql functions"""
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []
    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
    return parse_dates

def _handle_date_column(
    col: pd.Series,
    utc: bool = False,
    format: Optional[Union[str, Dict[str, Any]]] = None,
) -> pd.Series:
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
    parse_dates: Union[List[str], Dict[str, Any]],
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
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
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
        arrays = result_arrays  # type: ignore[assignment]
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
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
    dtype: Optional[DtypeArg] = None,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
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
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
    dtype: Optional[DtypeArg] = None,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    if dtype:
        df = df.astype(dtype)

    df = _parse_date_columns(df, parse_dates)

    if index_col is not None:
        df = df.set_index(index_col)

    return df

# -----------------------------------------------------------------------------
# -- Read and write to DataFrames

@overload
def read_sql_table(
    table_name: str,
    con: Any,
    schema: Optional[str] = ...,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: None = ...,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = ...,
) -> DataFrame: ...

@overload
def read_sql_table(
    table_name: str,
    con: Any,
    schema: Optional[str] = ...,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: int = ...,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = ...,
) -> Iterator[DataFrame]: ...

def read_sql_table(
    table_name: str,
    con: Any,
    schema: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
) -> Union[DataFrame, Iterator[DataFrame]]:
    """
    Read SQL database table into a DataFrame.
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
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
    sql: str,
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ...,
    chunksize: None = ...,
    dtype: Optional[DtypeArg] = ...,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = ...,
) -> DataFrame: ...

@overload
def read_sql_query(
    sql: str,
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ...,
    chunksize: int = ...,
    dtype: Optional[DtypeArg] = ...,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = ...,
) -> Iterator[DataFrame]: ...

def read_sql_query(
    sql: str,
    con: Any,
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = None,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
    chunksize: Optional[int] = None,
    dtype: Optional[DtypeArg] = None,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
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
    sql: str,
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: None = ...,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = ...,
    dtype: Optional[DtypeArg] = None,
) -> DataFrame: ...

@overload
def read_sql(
    sql: str,
    con: Any,
    index_col: Optional[Union[str, List[str]]] = ...,
    coerce_float: bool = ...,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ...,
    columns: Optional[List[str]] = ...,
    chunksize: int = ...,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = ...,
    dtype: Optional[DtypeArg] = None,
) -> Iterator[DataFrame]: ...

def read_sql(
    sql: str,
    con: Any,
    index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True,
    params: Optional[Union[List[Any], Mapping[str, Any]]] = None,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
    dtype: Optional[DtypeArg] = None,
) -> Union[DataFrame, Iterator[DataFrame]]:
    """
    Read SQL query or database table into a DataFrame.
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
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
                dtype_backend=dtype_backend,
                dtype=dtype,
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
    index_label: Optional[Union[str, List[str]]] = None,
    chunksize: Optional[int] = None,
    dtype: Optional[DtypeArg] = None,
    method: Optional[Union[Literal["multi"], Callable]] = None,
    engine: str = "auto",
    **engine_kwargs,
) -> Optional[int]:
    """
    Write records stored in a DataFrame to a SQL database.
    """
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
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.
    """
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
    """
    For mapping Pandas tables to SQL tables.
    """
    def __init__(
        self,
        name: str,
        pandas_sql_engine: PandasSQL,
        frame: Optional[DataFrame] = None,
        index: Optional[Union[bool, str, List[str]]] = True,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        prefix: str = "pandas",
        index_label: Optional[Union[str, List[str]]] = None,
        schema: Optional[str] = None,
        keys: Optional[List[str]] = None,
        dtype: Optional[DtypeArg] = None,
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

    def _execute_insert(self, conn: Any, keys: List[str], data_iter: Iterator[Any]) -> int:
        data = [dict(zip(keys, row)) for row in data_iter]
        result = conn.execute(self.table.insert(), data)
        return result.rowcount

    def _execute_insert_multi(self, conn: Any, keys: List[str], data_iter: Iterator[Any]) -> int:
        data = [dict(zip(keys, row)) for row in data_iter]
        from sqlalchemy import insert
        stmt = insert(self.table).values(data)
        result = conn.execute(stmt)
        return result.rowcount

    def insert_data(self) -> Tuple[List[str], List[np.ndarray]]:
        if self.index is not None:
            temp = self.frame.copy(deep=False)
            temp.index.names = self.index
            try:
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f"duplicate name in index/columns: {err}") from err
        else:
            temp = self.frame

        column_names = list(map(str, temp.columns))
        ncols = len(column_names)
        data_list: List[np.ndarray] = [None] * ncols  # type: ignore[list-item]

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
            exec_insert = self._execute_insert
        elif method == "multi":
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f"Invalid parameter `method`: {method}")

        keys, data_list = self.insert_data()

        nrows = len(self.frame)

        if nrows == 0:
            return 0

        if chunksize is None:
            chunksize = nrows
        elif chunksize == 0:
            raise ValueError("chunksize argument should be non-zero")

        chunks = (nrows // chunksize) + 1
        total_inserted = None
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
        chunksize: Optional[int],
        columns: List[str],
        coerce_float: bool = True,
        parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
        dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
    ) -> Generator[DataFrame]:
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield DataFrame.from_records(
                            [], columns=columns, coerce_float=coerce_float
                        )
                    break

                has_read_data = True
                self.frame = _convert_arrays_to_dataframe(
                    data, columns, coerce_float, dtype_backend
                )

                self._harmonize_columns(
                    parse_dates=parse_dates, dtype_backend=dtype_backend
                )

                if self.index is not None:
                    self.frame.set_index(self.index, inplace=True)

                yield self.frame

    def read(
        self,
        exit_stack: ExitStack,
        coerce_float: bool = True,
        parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
        columns: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
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
        column_names = result.keys()

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
            self.frame = _convert_arrays_to_dataframe(
                data, column_names, coerce_float, dtype_backend
            )

            self._harmonize_columns(
                parse_dates=parse_dates, dtype_backend=dtype_backend
            )

            if self.index is not None:
                self.frame.set_index(self.index, inplace=True)

            return self.frame

    def _index_name(
        self,
        index: Optional[Union[bool, str, List[str]]],
        index_label: Optional[Union[str, List[str]]],
    ) -> Optional[List[str]]:
        if index is True:
            nlevels = self.frame.index.nlevels
            if index_label is not None:
                if not isinstance(index_label, list):
                    index_label = [index_label]
                if len(index_label) != nlevels:
                    raise ValueError(
                        "Length of 'index_label' should match number of "
                        f"levels, which is {nlevels}"
                    )
                return index_label
            if (
                nlevels == 1
                and "index" not in self.frame.columns
                and self.frame.index.name is None
            ):
                return ["index"]
            else:
                return com.fill_missing_names(self.frame.index.names)
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None

    def _get_column_names_and_types(
        self, dtype_mapper: Callable[[Any], Any]
    ) -> List[Tuple[str, Any, bool]]:
        column_names_and_types = []
        if self.index is not None:
            for i, idx_label in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))
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

        columns: List[Any] = [
            Column(name, typ, index=is_index)
            for name, typ, is_index in column_names_and_types
        ]

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
        parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
        dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
    ) -> None:
        parse_dates = _process_parse_dates_argument(parse_dates)

        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                df_col = self.frame[col_name]

                if col_name in parse_dates:
                    try:
                        fmt = parse_dates[col_name]
                    except TypeError:
                        fmt = None
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue

                col_type = self._get_dtype(sql_col.type)

                if (
                    col_type is datetime
                    or col_type is date
                    or col_type is DatetimeTZDtype
                ):
                    utc = col_type is DatetimeTZDtype
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                elif dtype_backend == "numpy" and col_type is float:
                    self.frame[col_name] = df_col.astype(col_type)
                elif (
                    using_string_dtype()
                    and is_string_dtype(col_type)
                    and is_object_dtype(self.frame[col_name])
                ):
                    self.frame[col_name] = df_col.astype(col_type)
                elif dtype_backend == "numpy" and len(df_col) == df_col.count():
                    if col_type is np.dtype("int64") or col_type is bool:
                        self.frame[col_name] = df_col.astype(col_type)
            except KeyError:
                pass

    def _sqlalchemy_type(self, col: Union[Index, Series]) -> Any:
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]

        col_type = lib.infer_dtype(col, skipna=True)

        from sqlalchemy.types import (
            TIMESTAMP,
            BigInteger,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            SmallInteger,
            Text,
            Time,
        )

        if col_type in ("datetime64", "datetime"):
            try:
                if col.dt.tz is not None:  # type: ignore[union-attr]
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                if getattr(col, "tz", None) is not None:
                    return TIMESTAMP(timezone=True)
            return DateTime
        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
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
        from sqlalchemy.types import (
            TIMESTAMP,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            String,
        )

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
    Subclasses Should define read_query and to_sql.
    """
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def read_table(
        self,
        table_name: str,
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
        columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        raise NotImplementedError

    @abstractmethod
    def read_query(
        self,
        sql: str,
        index_col: Optional[Union[str, List[str]]] = None,
        coerce_float: bool = True,
        parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None,
        params: Optional[Union[List[Any], Mapping[str, Any]]] = None,
        chunksize: Optional[int] = None,
        dtype: Optional[DtypeArg] = None,
        dtype_backend: Union[DtypeBackend, Literal["numpy"]] = "numpy",
    ) -> Union[DataFrame, Iterator[DataFrame]]:
        pass

    @abstractmethod
    def to_sql(
        self,
        frame: Union[DataFrame, Series],
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label: Optional