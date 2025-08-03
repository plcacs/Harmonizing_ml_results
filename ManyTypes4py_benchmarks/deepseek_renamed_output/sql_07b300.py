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
from typing import TYPE_CHECKING, Any, Literal, cast, overload, Optional, Union, List, Dict, Tuple, Generator, Sequence, Mapping, Callable, Iterator
import warnings
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
if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping
    from sqlalchemy import Table
    from sqlalchemy.sql.expression import Select, TextClause
    from pandas._typing import DtypeArg, DtypeBackend, IndexLabel, Self
    from pandas import Index


def func_bivw3y5u(parse_dates: Union[bool, List[Union[str, Dict[str, Any]], None]) -> List[Union[str, Dict[str, Any]]:
    """Process parse_dates argument for read_sql functions"""
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []
    elif not hasattr(parse_dates, '__iter__'):
        parse_dates = [parse_dates]
    return parse_dates


def func_z3puhg0s(col: Series, utc: bool = False, format: Optional[Union[str, Dict[str, Any]]] = None) -> Series:
    if isinstance(format, dict):
        return to_datetime(col, **format)
    else:
        if format is None and (issubclass(col.dtype.type, np.floating) or
            issubclass(col.dtype.type, np.integer)):
            format = 's'
        if format in ['D', 'd', 'h', 'm', 's', 'ms', 'us', 'ns']:
            return to_datetime(col, errors='coerce', unit=format, utc=utc)
        elif isinstance(col.dtype, DatetimeTZDtype):
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors='coerce', format=format, utc=utc)


def func_h9k1ygc4(data_frame: DataFrame, parse_dates: Union[bool, List[Union[str, Dict[str, Any]], None]) -> DataFrame:
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
    parse_dates = func_bivw3y5u(parse_dates)
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        if isinstance(df_col.dtype, DatetimeTZDtype
            ) or col_name in parse_dates:
            try:
                fmt = parse_dates[col_name]
            except (KeyError, TypeError):
                fmt = None
            data_frame.isetitem(i, func_z3puhg0s(df_col, format=fmt))
    return data_frame


def func_6cnete1o(data: List[Tuple[Any, ...]], columns: List[str], coerce_float: bool = True, dtype_backend: str = 'numpy') -> DataFrame:
    content = lib.to_object_array_tuples(data)
    idx_len = content.shape[0]
    arrays = convert_object_array(list(content.T), dtype=None, coerce_float
        =coerce_float, dtype_backend=dtype_backend)
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
        return DataFrame._from_arrays(arrays, columns=columns, index=range(
            idx_len), verify_integrity=False)
    else:
        return DataFrame(columns=columns)


def func_wnix7iby(data: List[Tuple[Any, ...]], columns: List[str], index_col: Optional[Union[str, List[str]]] = None, coerce_float: bool = True,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None, dtype: Optional[Union[str, Dict[str, Any]]] = None, dtype_backend: str = 'numpy') -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    frame = func_6cnete1o(data, columns, coerce_float, dtype_backend)
    if dtype:
        frame = frame.astype(dtype)
    frame = func_h9k1ygc4(frame, parse_dates)
    if index_col is not None:
        frame = frame.set_index(index_col)
    return frame


def func_cpxz0p3s(df: DataFrame, *, index_col: Optional[Union[str, List[str]]] = None, parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None, dtype: Optional[Union[str, Dict[str, Any]]] = None,
    dtype_backend: str = 'numpy') -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    if dtype:
        df = df.astype(dtype)
    df = func_h9k1ygc4(df, parse_dates)
    if index_col is not None:
        df = df.set_index(index_col)
    return df


@overload
def func_ykzak4i0(table_name: str, con: Union[str, Any], schema: Optional[str] = ..., index_col: Optional[Union[str, List[str]]] = ..., coerce_float: bool = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ..., columns: Optional[List[str]] = ..., chunksize: Optional[int] = ..., dtype_backend: str = ...) -> DataFrame:
    ...


@overload
def func_ykzak4i0(table_name: str, con: Union[str, Any], schema: Optional[str] = ..., index_col: Optional[Union[str, List[str]]] = ..., coerce_float: bool = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ..., columns: Optional[List[str]] = ..., chunksize: Optional[int] = ..., dtype_backend: str = ...) -> Generator[DataFrame, None, None]:
    ...


def func_ykzak4i0(table_name: str, con: Union[str, Any], schema: Optional[str] = None, index_col: Optional[Union[str, List[str]]] = None,
    coerce_float: bool = True, parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None, columns: Optional[List[str]] = None, chunksize: Optional[int] = None,
    dtype_backend: str = lib.no_default) -> Union[DataFrame, Generator[DataFrame, None, None]]:
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> pd.read_sql_table("table_name", "postgres:///db_name")  # doctest:+SKIP
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = 'numpy'
    assert dtype_backend is not lib.no_default
    with pandasSQL_builder(con, schema=schema, need_transaction=True
        ) as pandas_sql:
        if not pandas_sql.has_table(table_name):
            raise ValueError(f'Table {table_name} not found')
        table = pandas_sql.read_table(table_name, index_col=index_col,
            coerce_float=coerce_float, parse_dates=parse_dates, columns=
            columns, chunksize=chunksize, dtype_backend=dtype_backend)
    if table is not None:
        return table
    else:
        raise ValueError(f'Table {table_name} not found', con)


@overload
def func_sl4dt0t7(sql: str, con: Union[str, Any], index_col: Optional[Union[str, List[str]]] = ..., coerce_float: bool = ..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]] = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ..., chunksize: Optional[int] = ..., dtype: Optional[Union[str, Dict[str, Any]]] = ..., dtype_backend: str = ...) -> DataFrame:
    ...


@overload
def func_sl4dt0t7(sql: str, con: Union[str, Any], index_col: Optional[Union[str, List[str]]] = ..., coerce_float: bool = ..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]] = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ..., chunksize: Optional[int] = ..., dtype: Optional[Union[str, Dict[str, Any]]] = ..., dtype_backend: str = ...) -> Generator[DataFrame, None, None]:
    ...


def func_sl4dt0t7(sql: str, con: Union[str, Any], index_col: Optional[Union[str, List[str]]] = None, coerce_float: bool = True, params: Optional[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]] = None,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = None, chunksize: Optional[int] = None, dtype: Optional[Union[str, Dict[str, Any]]] = None, dtype_backend: str = lib.no_default
    ) -> Union[DataFrame, Generator[DataFrame, None, None]]:
    """
    Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    params : list, tuple or mapping, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.

        .. versionadded:: 1.3.0
    dtype_backend : {'numpy_nullable', 'pyarrow'}
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). If not specified, the default behavior
        is to not use nullable data types. If specified, the behavior
        is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
        * ``"pyarrow"``: returns pyarrow-backed nullable
          :class:`ArrowDtype` :class:`DataFrame`

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        Returns a DataFrame object that contains the result set of the
        executed SQL query, in relation to the specified database connection.

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.

    Examples
    --------
    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("sqlite:///database.db")  # doctest: +SKIP
    >>> sql_query = "SELECT int_column FROM test_data"  # doctest: +SKIP
    >>> with engine.connect() as conn, conn.begin():  # doctest: +SKIP
    ...     data = pd.read_sql_query(sql_query, conn)  # doctest: +SKIP
    """
    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = 'numpy'
    assert dtype_backend is not lib.no_default
    with pandasSQL_builder(con) as pandas_sql:
        return pandas_sql.read_query(sql, index_col=index_col, params=
            params, coerce_float=coerce_float, parse_dates=parse_dates,
            chunksize=chunksize, dtype=dtype, dtype_backend=dtype_backend)


@overload
def func_81sduy86(sql: str, con: Union[str, Any], index_col: Optional[Union[str, List[str]]] = ..., coerce_float: bool = ..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[str, Any]]] = ...,
    parse_dates: Optional[Union[List[str], Dict[str, Any]]] = ..., columns: Optional[List[str]] = ..., chunksize: Optional[int] = ..., dtype_backend: str = ..., dtype: Optional[Union[str, Dict[str, Any]]] = ...