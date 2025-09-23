"""
Wrappers around spark that correspond to common pandas functions.
"""
from typing import Any, Optional, Union, List, Tuple, Sized, cast, Dict, Sequence, IO, OrderedDict as OrderedDictType
from collections import OrderedDict
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from io import BytesIO
import json
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype, is_list_like
import pyarrow as pa
import pyarrow.parquet as pq
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, BooleanType, TimestampType, DecimalType, StringType, DateType, StructType
from databricks import koalas as ks
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.utils import align_diff_frames, default_session, is_name_like_tuple, name_like_string, same_anchor, scol_for, validate_axis
from databricks.koalas.frame import DataFrame, _reduce_spark_multi
from databricks.koalas.internal import InternalFrame, DEFAULT_SERIES_NAME, HIDDEN_COLUMNS
from databricks.koalas.series import Series, first_series
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.indexes import Index, DatetimeIndex

__all__ = ['from_pandas', 'range', 'read_csv', 'read_delta', 'read_table', 'read_spark_io', 'read_parquet', 'read_clipboard', 'read_excel', 'read_html', 'to_datetime', 'date_range', 'get_dummies', 'concat', 'melt', 'isna', 'isnull', 'notna', 'notnull', 'read_sql_table', 'read_sql_query', 'read_sql', 'read_json', 'merge', 'to_numeric', 'broadcast', 'read_orc']

def from_pandas(pobj: Union[pd.Series, pd.DataFrame, pd.Index]) -> Union[Series, DataFrame, Index]:
    """Create a Koalas DataFrame, Series or Index from a pandas DataFrame, Series or Index.

    This is similar to Spark's `SparkSession.createDataFrame()` with pandas DataFrame,
    but this also works with pandas Series and picks the index.

    Parameters
    ----------
    pobj : pandas.DataFrame or pandas.Series
        pandas DataFrame or Series to read.

    Returns
    -------
    Series or DataFrame
        If a pandas Series is passed in, this function returns a Koalas Series.
        If a pandas DataFrame is passed in, this function returns a Koalas DataFrame.
    """
    if isinstance(pobj, pd.Series):
        return Series(pobj)
    elif isinstance(pobj, pd.DataFrame):
        return DataFrame(pobj)
    elif isinstance(pobj, pd.Index):
        return DataFrame(pd.DataFrame(index=pobj)).index
    else:
        raise ValueError('Unknown data type: {}'.format(type(pobj).__name__))

_range = range

def range(start: int, end: Optional[int] = None, step: int = 1, num_partitions: Optional[int] = None) -> DataFrame:
    """
    Create a DataFrame with some range of numbers.

    The resulting DataFrame has a single int64 column named `id`, containing elements in a range
    from ``start`` to ``end`` (exclusive) with step value ``step``. If only the first parameter
    (i.e. start) is specified, we treat it as the end value with the start value being 0.

    This is similar to the range function in SparkSession and is used primarily for testing.

    Parameters
    ----------
    start : int
        the start value (inclusive)
    end : int, optional
        the end value (exclusive)
    step : int, optional, default 1
        the incremental step
    num_partitions : int, optional
        the number of partitions of the DataFrame

    Returns
    -------
    DataFrame

    Examples
    --------
    When the first parameter is specified, we generate a range of values up till that number.

    >>> ks.range(5)
       id
    0   0
    1   1
    2   2
    3   3
    4   4

    When start, end, and step are specified:

    >>> ks.range(start = 100, end = 200, step = 20)
        id
    0  100
    1  120
    2  140
    3  160
    4  180
    """
    sdf = default_session().range(start=start, end=end, step=step, numPartitions=num_partitions)
    return DataFrame(sdf)

def read_csv(path: str, sep: str = ',', header: Union[str, int, List[int]] = 'infer', names: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, usecols: Optional[Union[List[Union[int, str]], callable]] = None, squeeze: bool = False, mangle_dupe_cols: bool = True, dtype: Optional[Union[str, Dict[str, Any]]] = None, nrows: Optional[int] = None, parse_dates: bool = False, quotechar: Optional[str] = None, escapechar: Optional[str] = None, comment: Optional[str] = None, **options: Any) -> Union[DataFrame, Series]:
    """Read CSV (comma-separated) file into DataFrame or Series.

    Parameters
    ----------
    path : str
        The path string storing the CSV file to be read.
    sep : str, default ‘,’
        Delimiter to use. Must be a single character.
    header : int, list of int, default ‘infer’
        Whether to to use as the column names, and the start of the data.
        Default behavior is to infer the column names: if no names are passed
        the behavior is identical to `header=0` and column names are inferred from
        the first line of the file, if column names are passed explicitly then
        the behavior is identical to `header=None`. Explicitly pass `header=0` to be
        able to replace existing names
    names : str or array-like, optional
        List of column names to use. If file contains no header row, then you should
        explicitly pass `header=None`. Duplicates in this list will cause an error to be issued.
        If a string is given, it should be a DDL-formatted string in Spark SQL, which is
        preferred to avoid schema inference for better performance.
    index_col: str or list of str, optional, default: None
        Index column of table in Spark.
    usecols : list-like or callable, optional
        Return a subset of the columns. If list-like, all elements must either be
        positional (i.e. integer indices into the document columns) or strings that
        correspond to column names provided either by the user in names or inferred
        from the document header row(s).
        If callable, the callable function will be evaluated against the column names,
        returning names where the callable function evaluates to `True`.
    squeeze : bool, default False
        If the parsed data only contains one column then return a Series.
    mangle_dupe_cols : bool, default True
        Duplicate columns will be specified as 'X0', 'X1', ... 'XN', rather
        than 'X' ... 'X'. Passing in False will cause data to be overwritten if
        there are duplicate names in the columns.
        Currently only `True` is allowed.
    dtype : Type name or dict of column -> type, default None
        Data type for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32} Use str or object
        together with suitable na_values settings to preserve and not interpret dtype.
    nrows : int, default None
        Number of rows to read from the CSV file.
    parse_dates : boolean or list of ints or names or list of lists or dict, default `False`.
        Currently only `False` is allowed.
    quotechar : str (length 1), optional
        The character used to denote the start and end of a quoted item. Quoted items can include
        the delimiter and it will be ignored.
    escapechar : str (length 1), default None
        One-character string used to escape delimiter
    comment: str, optional
        Indicates the line should not be parsed.
    options : dict
        All other options passed directly into Spark's data source.

    Returns
    -------
    DataFrame or Series

    See Also
    --------
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.

    Examples
    --------
    >>> ks.read_csv('data.csv')  # doctest: +SKIP
    """
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    if mangle_dupe_cols is not True:
        raise ValueError('mangle_dupe_cols can only be `True`: %s' % mangle_dupe_cols)
    if parse_dates is not False:
        raise ValueError('parse_dates can only be `False`: %s' % parse_dates)
    if usecols is not None and (not callable(usecols)):
        usecols = list(usecols)
    if usecols is None or callable(usecols) or len(usecols) > 0:
        reader = default_session().read
        reader.option('inferSchema', True)
        reader.option('sep', sep)
        if header == 'infer':
            header = 0 if names is None else None
        if header == 0:
            reader.option('header', True)
        elif header is None:
            reader.option('header', False)
        else:
            raise ValueError('Unknown header argument {}'.format(header))
        if quotechar is not None:
            reader.option('quote', quotechar)
        if escapechar is not None:
            reader.option('escape', escapechar)
        if comment is not None:
            if not isinstance(comment, str) or len(comment) != 1:
                raise ValueError('Only length-1 comment characters supported')
            reader.option('comment', comment)
        reader.options(**options)
        if isinstance(names, str):
            sdf = reader.schema(names).csv(path)
            column_labels = OrderedDict(((col, col) for col in sdf.columns))
        else:
            sdf = reader.csv(path)
            if is_list_like(names):
                names = list(names)
                if len(set(names)) != len(names):
                    raise ValueError('Found non-unique column index')
                if len(names) != len(sdf.columns):
                    raise ValueError('The number of names [%s] does not match the number of columns [%d]. Try names by a Spark SQL DDL-formatted string.' % (len(sdf.schema), len(names)))
                column_labels = OrderedDict(zip(names, sdf.columns))
            elif header is None:
                column_labels = OrderedDict(enumerate(sdf.columns))
            else:
                column_labels = OrderedDict(((col, col) for col in sdf.columns))
        if usecols is not None:
            if callable(usecols):
                column_labels = OrderedDict(((label, col) for (label, col) in column_labels.items() if usecols(label)))
                missing = []
            elif all((isinstance(col, int) for col in usecols)):
                new_column_labels = OrderedDict(((label, col) for (i, (label, col)) in enumerate(column_labels.items()) if i in usecols))
                missing = [col for col in usecols if col >= len(column_labels) or list(column_labels)[col] not in new_column_labels]
                column_labels = new_column_labels
            elif all((isinstance(col, str) for col in usecols)):
                new_column_labels = OrderedDict(((label, col) for (label, col) in column_labels.items() if label in usecols))
                missing = [col for col in usecols if col not in new_column_labels]
                column_labels = new_column_labels
            else:
                raise ValueError("'usecols' must either be list-like of all strings, all unicode, all integers or a callable.")
            if len(missing) > 0:
                raise ValueError('Usecols do not match columns, columns expected but not found: %s' % missing)
            if len(column_labels) > 0:
                sdf = sdf.select([scol_for(sdf, col) for col in column_labels.values()])
            else:
                sdf = default_session().createDataFrame([], schema=StructType())
    else:
        sdf = default_session().createDataFrame([], schema=StructType())
        column_labels = OrderedDict()
    if nrows is not None:
        sdf = sdf.limit(nrows)
    if index_col is not None:
        if isinstance(index_col, (str, int)):
            index_col = [index_col]
        for col in index_col:
            if col not in column_labels:
                raise KeyError(col)
        index_spark_column_names = [column_labels[col] for col in index_col]
        index_names = [(col,) for col in index_col]
        column_labels = OrderedDict(((label, col) for (label, col) in column_labels.items() if label not in index_col))
    else:
        index_spark_column_names = []
        index_names = []
    kdf = DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in index_spark_column_names], index_names=index_names, column_labels=[label if is_name_like_tuple(label) else (label,) for label in column_labels], data_spark_columns=[scol_for(sdf, col) for col in column_labels.values()]))
    if dtype is not None:
        if isinstance(dtype, dict):
            for (col, tpe) in dtype.items():
                kdf[col] = kdf[col].astype(tpe)
        else:
            for col in kdf.columns:
                kdf[col] = kdf[col].astype(dtype)
    if squeeze and len(kdf.columns) == 1:
        return first_series(kdf)
    else:
        return kdf

def read_json(path: str, lines: bool = True, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> DataFrame:
    """
    Convert a JSON string to DataFrame.

    Parameters
    ----------
    path : string
        File path
    lines : bool, default True
        Read the file as a json object per line. It should be always True for now.
    index_col : str or list of str, optional, default: None
        Index column of table in Spark.
    options : dict
        All other options passed directly into Spark's data source.

    Examples
    --------
    >>> df = ks.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   columns=['col 1', 'col 2'])

    >>> df.to_json(path=r'%s/read_json/foo.json' % path, num_files=1)
    >>> ks.read_json(
    ...     path=r'%s/read_json/foo.json' % path
    ... ).sort_values(by="col 1")
      col 1 col 2
    0     a     b
    1     c     d

    >>> df.to_json(path=r'%s/read_json/foo.json' % path, num_files=1, lineSep='___')
    >>> ks.read_json(
    ...     path=r'%s/read_json/foo.json' % path, lineSep='___'
    ... ).sort_values(by="col 1")
      col 1 col 2
    0     a     b
    1     c     d

    You can preserve the index in the roundtrip as below.

    >>> df.to_json(path=r'%s/read_json/bar.json' % path, num_files=1, index_col="index")
    >>> ks.read_json(
    ...     path=r'%s/read_json/bar.json' % path, index_col="index"
    ... ).sort_values(by="col 1")  # doctest: +NORMALIZE_WHITESPACE
          col 1 col 2
    index
    0         a     b
    1         c     d
    """
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    if not lines:
        raise NotImplementedError('lines=False is not implemented yet.')
    return read_spark_io(path, format='json', index_col=index_col, **options)

def read_delta(path: str, version: Optional[str] = None, timestamp: Optional[str] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> DataFrame:
    """
    Read a Delta Lake table on some file system and return a DataFrame.

    If the Delta Lake table is already stored in the catalog (aka the metastore), use 'read_table'.

    Parameters
    ----------
    path : string
        Path to the Delta Lake table.
    version : string, optional
        Specifies the table version (based on Delta's internal transaction version) to read from,
        using Delta's time travel feature. This sets Delta's 'versionAsOf' option.
    timestamp : string, optional
        Specifies the table version (based on timestamp) to read from,
        using Delta's time travel feature. This must be a valid date or timestamp string in Spark,
        and sets Delta's 'timestampAsOf' option.
    index_col : str or list of str, optional, default: None
        Index column of table in Spark.
    options
        Additional options that can be passed onto Delta.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.to_delta
    read_table
    read_spark_io
    read_parquet

    Examples
    --------
    >>> ks.range(1).to_delta('%s