from typing import Any, Optional, Union, List, Tuple, Dict, Set, Callable, Iterable, Sized, cast, OrderedDict
from collections import OrderedDict as OrderedDictType
from collections.abc import Iterable as IterableABC
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
from databricks.koalas.internal import InternalFrame, DEFAULT_SERIES_NAME, HIDDEN_COLUMNS
from databricks.koalas.series import Series, first_series
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.indexes import Index, DatetimeIndex

__all__ = [
    'from_pandas', 'range', 'read_csv', 'read_delta', 'read_table',
    'read_spark_io', 'read_parquet', 'read_clipboard', 'read_excel',
    'read_html', 'to_datetime', 'date_range', 'get_dummies', 'concat',
    'melt', 'isna', 'isnull', 'notna', 'notnull', 'read_sql_table',
    'read_sql_query', 'read_sql', 'read_json', 'merge', 'to_numeric',
    'broadcast', 'read_orc'
]

def func_6knvxuc5(pobj: Union[pd.DataFrame, pd.Series, pd.Index]) -> Union[Series, DataFrame]:
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
    sdf = default_session().range(start=start, end=end, step=step, numPartitions=num_partitions)
    return DataFrame(sdf)

def func_krnpxuf1(
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
    **options: Dict[str, Any]
) -> Union[DataFrame, Series]:
    if 'options' in options and isinstance(options.get('options'), dict) and len(options) == 1:
        options = options.get('options')
    if mangle_dupe_cols is not True:
        raise ValueError('mangle_dupe_cols can only be `True`: %s' % mangle_dupe_cols)
    if parse_dates is not False:
        raise ValueError('parse_dates can only be `False`: %s' % parse_dates)
    if usecols is not None and not callable(usecols):
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
            column_labels = OrderedDict((col, col) for col in sdf.columns)
        else:
            sdf = reader.csv(path)
            if is_list_like(names):
                names = list(names)
                if len(set(names)) != len(names):
                    raise ValueError('Found non-unique column index')
                if len(names) != len(sdf.columns):
                    raise ValueError(
                        'The number of names [%s] does not match the number of columns [%d]. Try names by a Spark SQL DDL-formatted string.'
                         % (len(sdf.schema), len(names)))
                column_labels = OrderedDict(zip(names, sdf.columns))
            elif header is None:
                column_labels = OrderedDict(enumerate(sdf.columns))
            else:
                column_labels = OrderedDict((col, col) for col in sdf.columns)
        if usecols is not None:
            if callable(usecols):
                column_labels = OrderedDict((label, col) for label, col in
                    column_labels.items() if usecols(label))
                missing = []
            elif all(isinstance(col, int) for col in usecols):
                new_column_labels = OrderedDict((label, col) for i, (label,
                    col) in enumerate(column_labels.items()) if i in usecols)
                missing = [col for col in usecols if col >= len(
                    column_labels) or list(column_labels)[col] not in
                    new_column_labels]
                column_labels = new_column_labels
            elif all(isinstance(col, str) for col in usecols):
                new_column_labels = OrderedDict((label, col) for label, col in
                    column_labels.items() if label in usecols)
                missing = [col for col in usecols if col not in
                    new_column_labels]
                column_labels = new_column_labels
            else:
                raise ValueError(
                    "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
                    )
            if len(missing) > 0:
                raise ValueError(
                    'Usecols do not match columns, columns expected but not found: %s'
                     % missing)
            if len(column_labels) > 0:
                sdf = sdf.select([scol_for(sdf, col) for col in
                    column_labels.values()])
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
    else:
        index_spark_column_names = []
        index_names = []
    kdf = DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=[
        scol_for(sdf, col) for col in index_spark_column_names],
        index_names=index_names, column_labels=[(label if
        is_name_like_tuple(label) else (label,)) for label in column_labels
        ], data_spark_columns=[scol_for(sdf, col) for col in column_labels.
        values()]))
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

def func_99w02szi(
    path: str, 
    lines: bool = True, 
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Dict[str, Any]
) -> DataFrame:
    if 'options' in options and isinstance(options.get('options'), dict) and len(options) == 1:
        options = options.get('options')
    if not lines:
        raise NotImplementedError('lines=False is not implemented yet.')
    return read_spark_io(path, format='json', index_col=index_col, **options)

def func_c71e96xh(
    path: str, 
    version: Optional[str] = None, 
    timestamp: Optional[str] = None, 
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Dict[str, Any]
) -> DataFrame:
    if 'options' in options and isinstance(options.get('options'), dict) and len(options) == 1:
        options = options.get('options')
    if version is not None:
        options['versionAsOf'] = version
    if timestamp is not None:
        options['timestampAsOf'] = timestamp
    return read_spark_io(path, format='delta', index_col=index_col, **options)

def func_00vi5vk7(name: str, index_col: Optional[Union[str, List[str]]] = None) -> DataFrame:
    sdf = default_session().read.table(name)
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    return DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=
        index_spark_columns, index_names=index_names))

def func_kpb88wgg(
    path: Optional[str] = None, 
    format: Optional[str] = None, 
    schema: Optional[Union[str, StructType]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Dict[str, Any]
) -> DataFrame:
    if 'options' in options and isinstance(options.get('options'), dict) and len(options) == 1:
        options = options.get('options')
    sdf = default_session().read.load(path=path, format=format, schema=schema, **options)
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    return DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=
        index_spark_columns, index_names=index_names))

def func_vd8i07uq(
    path: str, 
    columns: Optional[List[str]] = None, 
    index_col: Optional[Union[str, List[str]]] = None,
    pandas_metadata: bool = False,
    **options: Dict[str, Any]
) -> DataFrame:
    if 'options' in options and isinstance(options.get('options'), dict) and len(options) == 1:
        options = options.get('options')
    if columns is not None:
        columns = list(columns)
    index_names = None
    if index_col is None and pandas_metadata:
        if LooseVersion(pyspark.__version__) < LooseVersion('3.0.0'):
            raise ValueError('pandas_metadata is not supported with Spark < 3.0.')

        @pandas_udf('index_col array<string>, index_names array<string>', PandasUDFType.SCALAR)
        def func_ig04lkw3(pser: pd.Series) -> pd.DataFrame:
            binary = pser.iloc[0]
            metadata = pq.ParquetFile(pa.BufferReader(binary)).metadata.metadata
            if b'pandas' in metadata:
                pandas_metadata = json.loads(metadata[b'pandas'].decode('utf8'))
                if all(isinstance(col, str) for col in pandas_metadata['index_columns']):
                    index_col = []
                    index_names = []
                    for col in pandas_metadata['index_columns']:
                        index_col.append(col)
                        for column in pandas_metadata['columns']:
                            if column['field_name'] == col:
                                index_names.append(column['name'])
                                break
                        else:
                            index_names.append(None)
                    return pd.DataFrame({'index_col': [index_col],
                        'index_names': [index_names]})
            return pd.DataFrame({'index_col': [None], 'index_names': [None]})
        index_col, index_names = default_session().read.format('binaryFile'
            ).load(path).limit(1).select(func_ig04lkw3('content').alias(
            'index_metadata')).select('index_metadata.*').head()
    kdf = func_kpb88wgg(path=path, format='parquet', options=options, index_col=index_col)
    if columns is not None:
        new_columns = [c for c in columns if c in kdf.columns]
        if len(new_columns) > 0:
            kdf = kdf[new_columns]
        else:
            sdf = default_session().createDataFrame([], schema=StructType())
            index_spark_columns, index_names = _get_index_map(sdf, index_col)
            kdf = DataFrame(InternalFrame(spark_frame=sdf,
                index_spark_columns=index_spark_columns, index_names=
                index_names))
    if index_names is not None:
        kdf.index.names = index_names
    return kdf

def func_0h4zz165(sep: str = '\\s+', **kwargs: Any) -> DataFrame:
    return cast(DataFrame, func_6knvxuc5(pd.read_clipboard(sep, **kwargs)))

def func_wziszf6g(
    io: Union[str, bytes, BytesIO], 
    sheet_name: Union[str, int, List[Union[str, int]]] = 0, 
    header: Union[int, List[int]] = 0, 
    names: Optional[List[str]] = None,
    index_col: Optional[Union[int, List[int]]] = None,
    usecols: Optional[Union[List[Union[int, str]], Callable[[str], bool]]] = None,
    squeeze: bool = False,
    dtype: Optional[Union[str, Dict[str, Any]]] = None,
    engine: Optional[str] = None,
    converters: Optional[Dict[Union[int, str], Callable]] = None,
    true_values: Optional[List[str]] = None,
    false_values: Optional[List[str]] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    nrows: Optional[int] = None,
    na_values: Optional[Union[str, List[str], Dict[str, str]]] = None,
    keep_default_na: bool = True,
    verbose: bool = False,
    parse_dates: Union[bool, List[Union[int, str]], List[List[Union[int, str]]], Dict[str, List[Union[int, str]]]] = False,
    date_parser: Optional[Callable] = None,
    thousands: Optional[str] = None,
    comment: Optional[str] = None,
    skipfooter: int = 0,
    convert_float: bool = True,
    mangle_dupe_cols: bool = True,
    **kwds: Any
) -> Union[DataFrame, Series, Dict[str, DataFrame]]:
    def func_bexfekkj(io_or_bin: Union[bytes, bytearray, BytesIO], sn: Union[str, int, List[Union[str, int]]], sq: bool) -> Union[pd.DataFrame, pd.Series, Dict[str, pd.DataFrame]]:
        return pd.read_excel(io=BytesIO(io_or_bin) if isinstance(io_or_bin,
            (bytes, bytearray)) else io_or_bin, sheet_name=sn, header=
            header, names=names, index_col=index_col, usecols=usecols,
            squeeze=sq, dtype=dtype, engine=engine, converters=converters,
            true_values=true_values, false_values=false_values, skiprows=
            skiprows, nrows=nrows, na_values=na_values, keep_default_na=
            keep_default_na, verbose=verbose, parse_dates=parse_dates,
            date_parser=date_parser, thousands=thousands, comment=comment,
            skipfooter=skipfooter, convert_float=convert_float,
            mangle_dupe_cols=mangle_dupe_cols, **kwds)
    if isinstance(io, str):
        if LooseVersion(pyspark.__version__) < LooseVersion('3.0.0'):
            raise ValueError(
                'The `io` parameter as a string is not supported if the underlying Spark is below 3.0. You can use `ks.from_pandas(pd.read_excel(...))` as a workaround'
                )
        binaries = default_session().read.format('binaryFile').load(io).select(
            'content').head(2)
        io_or_bin = binaries[0][0