from typing import Any, Optional, Union, List, Tuple, Dict, Set, Callable, Iterable, cast, OrderedDict as OrderedDictType
from collections import OrderedDict
from collections.abc import Sized, Iterable as IterableType
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
from pyspark.sql.types import (
    ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType,
    BooleanType, TimestampType, DecimalType, StringType, DateType, StructType
)
from databricks import koalas as ks
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.utils import (
    align_diff_frames, default_session, is_name_like_tuple,
    name_like_string, same_anchor, scol_for, validate_axis
)
from databricks.koalas.frame import DataFrame, _reduce_spark_multi
from databricks.koalas.internal import (
    InternalFrame, DEFAULT_SERIES_NAME, HIDDEN_COLUMNS
)
from databricks.koalas.series import Series, first_series
from databricks.koalas.spark.utils import (
    as_nullable_spark_type, force_decimal_precision_scale
)
from databricks.koalas.indexes import Index, DatetimeIndex

__all__ = [
    'from_pandas', 'range', 'read_csv', 'read_delta', 'read_table',
    'read_spark_io', 'read_parquet', 'read_clipboard', 'read_excel',
    'read_html', 'to_datetime', 'date_range', 'get_dummies', 'concat',
    'melt', 'isna', 'isnull', 'notna', 'notnull', 'read_sql_table',
    'read_sql_query', 'read_sql', 'read_json', 'merge', 'to_numeric',
    'broadcast', 'read_orc'
]

def from_pandas(pobj: Union[pd.Series, pd.DataFrame, pd.Index]) -> Union[Series, DataFrame, Index]:
    """Create a Koalas DataFrame, Series or Index from a pandas DataFrame, Series or Index."""
    if isinstance(pobj, pd.Series):
        return Series(pobj)
    elif isinstance(pobj, pd.DataFrame):
        return DataFrame(pobj)
    elif isinstance(pobj, pd.Index):
        return DataFrame(pd.DataFrame(index=pobj)).index
    else:
        raise ValueError('Unknown data type: {}'.format(type(pobj).__name__))

_range = range

def range(
    start: int,
    end: Optional[int] = None,
    step: int = 1,
    num_partitions: Optional[int] = None
) -> DataFrame:
    """Create a DataFrame with some range of numbers."""
    sdf = default_session().range(start=start, end=end, step=step, numPartitions=num_partitions)
    return DataFrame(sdf)

def read_csv(
    path: str,
    sep: str = ',',
    header: Union[str, int, List[int]] = 'infer',
    names: Optional[Union[str, List[str]]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    usecols: Optional[Union[List[Union[int, str]], Callable[[str], bool]]] = None,
    squeeze: bool = False,
    mangle_dupe_cols: bool = True,
    dtype: Optional[Union[str, Dict[str, Any]]] = None,
    nrows: Optional[int] = None,
    parse_dates: bool = False,
    quotechar: Optional[str] = None,
    escapechar: Optional[str] = None,
    comment: Optional[str] = None,
    **options: Any
) -> Union[DataFrame, Series]:
    """Read CSV (comma-separated) file into DataFrame or Series."""
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
                column_labels = OrderedDict(((label, col) for label, col in column_labels.items() if usecols(label)))
                missing = []
            elif all((isinstance(col, int) for col in usecols)):
                new_column_labels = OrderedDict(((label, col) for i, (label, col) in enumerate(column_labels.items()) if i in usecols))
                missing = [col for col in usecols if col >= len(column_labels) or list(column_labels)[col] not in new_column_labels]
                column_labels = new_column_labels
            elif all((isinstance(col, str) for col in usecols)):
                new_column_labels = OrderedDict(((label, col) for label, col in column_labels.items() if label in usecols))
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
        column_labels = OrderedDict(((label, col) for label, col in column_labels.items() if label not in index_col))
    else:
        index_spark_column_names = []
        index_names = []
    kdf = DataFrame(InternalFrame(
        spark_frame=sdf,
        index_spark_columns=[scol_for(sdf, col) for col in index_spark_column_names],
        index_names=index_names,
        column_labels=[label if is_name_like_tuple(label) else (label,) for label in column_labels],
        data_spark_columns=[scol_for(sdf, col) for col in column_labels.values()]
    ))
    if dtype is not None:
        if isinstance(dtype, dict):
            for col, tpe in dtype.items():
                kdf[col] = kdf[col].astype(tpe)
        else:
            for col in kdf.columns:
                kdf[col] = kdf[col].astype(dtype)
    if squeeze and len(kdf.columns) == 1:
        return first_series(kdf)
    else:
        return kdf

# [Rest of the functions with similar type annotations...]

def _get_index_map(
    sdf: spark.DataFrame,
    index_col: Optional[Union[str, List[str]]] = None
) -> Tuple[Optional[List[spark.Column]], Optional[List[Tuple[str]]]]:
    if index_col is not None:
        if isinstance(index_col, str):
            index_col = [index_col]
        sdf_columns = set(sdf.columns)
        for col in index_col:
            if col not in sdf_columns:
                raise KeyError(col)
        index_spark_columns = [scol_for(sdf, col) for col in index_col]
        index_names = [(col,) for col in index_col]
    else:
        index_spark_columns = None
        index_names = None
    return (index_spark_columns, index_names)

_get_dummies_default_accept_types = (DecimalType, StringType, DateType)
_get_dummies_acceptable_types = _get_dummies_default_accept_types + (
    ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, BooleanType, TimestampType
)
