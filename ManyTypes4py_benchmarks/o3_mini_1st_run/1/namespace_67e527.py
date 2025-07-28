from typing import Any, Optional, Union, List, Tuple, Sized, cast, Callable, Dict
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
from re import Pattern

__all__ = ['from_pandas', 'range', 'read_csv', 'read_delta', 'read_table', 'read_spark_io', 'read_parquet', 'read_clipboard', 'read_excel', 'read_html', 'to_datetime', 'date_range', 'get_dummies', 'concat', 'melt', 'isna', 'isnull', 'notna', 'notnull', 'read_sql_table', 'read_sql_query', 'read_sql', 'read_json', 'merge', 'to_numeric', 'broadcast', 'read_orc']

def from_pandas(pobj: Any) -> Union[Series, DataFrame, Index]:
    """Create a Koalas DataFrame, Series or Index from a pandas DataFrame, Series or Index.

    This is similar to Spark's `SparkSession.createDataFrame()` with pandas DataFrame,
    but this also works with pandas Series and picks the index.
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
    """
    sdf = default_session().range(start=start, end=end, step=step, numPartitions=num_partitions)
    return DataFrame(sdf)

def read_csv(
    path: str,
    sep: str = ',',
    header: Union[str, int, None] = 'infer',
    names: Optional[Union[str, List[str]]] = None,
    index_col: Optional[Union[str, int, List[Union[str, int]]]] = None,
    usecols: Optional[Union[List[Any], Callable[[Any], bool]]] = None,
    squeeze: bool = False,
    mangle_dupe_cols: bool = True,
    dtype: Optional[Union[Any, Dict[Any, Any]]] = None,
    nrows: Optional[int] = None,
    parse_dates: Union[bool, List[Any]] = False,
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
                missing: List[Any] = []
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

def read_json(path: str, lines: bool = True, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> DataFrame:
    """
    Convert a JSON string to DataFrame.
    """
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    if not lines:
        raise NotImplementedError('lines=False is not implemented yet.')
    return read_spark_io(path, format='json', index_col=index_col, **options)

def read_delta(path: str, version: Optional[Any] = None, timestamp: Optional[Any] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> DataFrame:
    """
    Read a Delta Lake table on some file system and return a DataFrame.
    """
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    if version is not None:
        options['versionAsOf'] = version
    if timestamp is not None:
        options['timestampAsOf'] = timestamp
    return read_spark_io(path, format='delta', index_col=index_col, **options)

def read_table(name: str, index_col: Optional[Union[str, List[str]]] = None) -> DataFrame:
    """
    Read a Spark table and return a DataFrame.
    """
    sdf = default_session().read.table(name)
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    return DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names))

def read_spark_io(
    path: Optional[str] = None,
    format: Optional[str] = None,
    schema: Optional[Union[str, StructType]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    """Load a DataFrame from a Spark data source."""
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    sdf = default_session().read.load(path=path, format=format, schema=schema, **options)
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    return DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names))

def read_parquet(
    path: str,
    columns: Optional[List[Any]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    pandas_metadata: bool = False,
    **options: Any
) -> DataFrame:
    """Load a parquet object from the file path, returning a DataFrame."""
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    if columns is not None:
        columns = list(columns)
    index_names: Optional[List[Any]] = None
    if index_col is None and pandas_metadata:
        if LooseVersion(pyspark.__version__) < LooseVersion('3.0.0'):
            raise ValueError('pandas_metadata is not supported with Spark < 3.0.')

        @pandas_udf('index_col array<string>, index_names array<string>', PandasUDFType.SCALAR)
        def read_index_metadata(pser: pd.Series) -> pd.DataFrame:
            binary = pser.iloc[0]
            metadata = pq.ParquetFile(pa.BufferReader(binary)).metadata.metadata
            if b'pandas' in metadata:
                pandas_metadata_inner = json.loads(metadata[b'pandas'].decode('utf8'))
                if all((isinstance(col, str) for col in pandas_metadata_inner['index_columns'])):
                    index_col_list: List[str] = []
                    index_names_list: List[Optional[str]] = []
                    for col in pandas_metadata_inner['index_columns']:
                        index_col_list.append(col)
                        for column in pandas_metadata_inner['columns']:
                            if column['field_name'] == col:
                                index_names_list.append(column['name'])
                                break
                        else:
                            index_names_list.append(None)
                    return pd.DataFrame({'index_col': [index_col_list], 'index_names': [index_names_list]})
            return pd.DataFrame({'index_col': [None], 'index_names': [None]})
        result = default_session().read.format('binaryFile').load(path).limit(1).select(read_index_metadata('content').alias('index_metadata')).select('index_metadata.*').head()
        index_col, index_names = result['index_col'], result['index_names']
    kdf = read_spark_io(path=path, format='parquet', options=options, index_col=index_col)
    if columns is not None:
        new_columns = [c for c in columns if c in kdf.columns]
        if len(new_columns) > 0:
            kdf = kdf[new_columns]
        else:
            sdf = default_session().createDataFrame([], schema=StructType())
            index_spark_columns, index_names = _get_index_map(sdf, index_col)
            kdf = DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names))
    if index_names is not None:
        kdf.index.names = index_names
    return kdf

def read_clipboard(sep: str = '\\s+', **kwargs: Any) -> DataFrame:
    """
    Read text from clipboard and pass to read_csv.
    """
    return cast(DataFrame, from_pandas(pd.read_clipboard(sep, **kwargs)))

def read_excel(
    io: Any,
    sheet_name: Union[str, int, List[Any]] = 0,
    header: Union[int, List[int], None] = 0,
    names: Optional[List[str]] = None,
    index_col: Optional[Union[int, List[int]]] = None,
    usecols: Optional[Union[int, str, List[Any], Callable[[Any], bool]]] = None,
    squeeze: bool = False,
    dtype: Optional[Union[Any, Dict[Any, Any]]] = None,
    engine: Optional[str] = None,
    converters: Optional[Dict[Any, Callable[[Any], Any]]] = None,
    true_values: Optional[List[Any]] = None,
    false_values: Optional[List[Any]] = None,
    skiprows: Optional[Union[List[int], int]] = None,
    nrows: Optional[int] = None,
    na_values: Optional[Union[str, List[str], Dict[Any, Any]]] = None,
    keep_default_na: bool = True,
    verbose: bool = False,
    parse_dates: Union[bool, List[Any]] = False,
    date_parser: Optional[Callable[..., Any]] = None,
    thousands: Optional[str] = None,
    comment: Optional[str] = None,
    skipfooter: int = 0,
    convert_float: bool = True,
    mangle_dupe_cols: bool = True,
    **kwds: Any
) -> Union[DataFrame, Series, Dict[Any, Union[DataFrame, Series]]]:
    """
    Read an Excel file into a Koalas DataFrame or Series.
    """
    def pd_read_excel(io_or_bin: Any, sn: Any, sq: bool) -> pd.DataFrame:
        return pd.read_excel(
            io=BytesIO(io_or_bin) if isinstance(io_or_bin, (bytes, bytearray)) else io_or_bin,
            sheet_name=sn,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            squeeze=sq,
            dtype=dtype,
            engine=engine,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            verbose=verbose,
            parse_dates=parse_dates,
            date_parser=date_parser,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            convert_float=convert_float,
            mangle_dupe_cols=mangle_dupe_cols,
            **kwds
        )
    if isinstance(io, str):
        if LooseVersion(pyspark.__version__) < LooseVersion('3.0.0'):
            raise ValueError('The `io` parameter as a string is not supported if the underlying Spark is below 3.0. You can use `ks.from_pandas(pd.read_excel(...))` as a workaround')
        binaries = default_session().read.format('binaryFile').load(io).select('content').head(2)
        io_or_bin = binaries[0][0]
        single_file = len(binaries) == 1
    else:
        io_or_bin = io
        single_file = True
    pdf_or_psers = pd_read_excel(io_or_bin, sn=sheet_name, sq=squeeze)
    if single_file:
        if isinstance(pdf_or_psers, dict):
            return OrderedDict([(sn, from_pandas(pdf_or_pser)) for sn, pdf_or_pser in pdf_or_psers.items()])
        else:
            return cast(Union[DataFrame, Series], from_pandas(pdf_or_psers))
    else:
        def read_excel_on_spark(pdf_or_pser: Union[pd.DataFrame, pd.Series], sn: Any) -> DataFrame:
            if isinstance(pdf_or_pser, pd.Series):
                pdf = pdf_or_pser.to_frame()
            else:
                pdf = pdf_or_pser
            kdf = from_pandas(pdf)
            return_schema = force_decimal_precision_scale(as_nullable_spark_type(kdf._internal.spark_frame.drop(*HIDDEN_COLUMNS).schema))
            def output_func(pdf_inner: pd.DataFrame) -> pd.DataFrame:
                pdf_inner = pd.concat([pd_read_excel(bin, sn=sn, sq=False) for bin in pdf_inner[pdf_inner.columns[0]]])
                reset_index = pdf_inner.reset_index()
                for name, col in reset_index.iteritems():
                    dt = col.dtype
                    if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                        continue
                    reset_index[name] = col.replace({np.nan: None})
                pdf_inner = reset_index
                return pdf_inner.rename(columns=dict(zip(pdf_inner.columns, return_schema.names)))
            sdf = default_session().read.format('binaryFile').load(io).select('content').mapInPandas(
                lambda iterator: map(output_func, iterator),
                schema=return_schema
            )
            kdf = DataFrame(kdf._internal.with_new_sdf(sdf))
            if squeeze and len(kdf.columns) == 1:
                return first_series(kdf)
            else:
                return kdf
        if isinstance(pdf_or_psers, dict):
            return OrderedDict([(sn, read_excel_on_spark(pdf_or_pser, sn)) for sn, pdf_or_pser in pdf_or_psers.items()])
        else:
            return read_excel_on_spark(pdf_or_psers, sheet_name)

def read_html(
    io: Union[str, Any],
    match: Union[str, Pattern[str]] = '.+',
    flavor: Optional[str] = None,
    header: Optional[Union[int, List[int]]] = None,
    index_col: Optional[Union[int, List[int]]] = None,
    skiprows: Optional[Union[int, List[int], slice]] = None,
    attrs: Optional[Dict[Any, Any]] = None,
    parse_dates: bool = False,
    thousands: str = ',',
    encoding: Optional[str] = None,
    decimal: str = '.',
    converters: Optional[Dict[Any, Callable[[Any], Any]]] = None,
    na_values: Optional[Any] = None,
    keep_default_na: bool = True,
    displayed_only: bool = True
) -> List[DataFrame]:
    """Read HTML tables into a list of DataFrame objects."""
    pdfs: List[pd.DataFrame] = pd.read_html(
        io=io,
        match=match,
        flavor=flavor,
        header=header,
        index_col=index_col,
        skiprows=skiprows,
        attrs=attrs,
        parse_dates=parse_dates,
        thousands=thousands,
        encoding=encoding,
        decimal=decimal,
        converters=converters,
        na_values=na_values,
        keep_default_na=keep_default_na,
        displayed_only=displayed_only
    )
    return [from_pandas(pdf) for pdf in pdfs]

def read_sql_table(
    table_name: str,
    con: str,
    schema: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    columns: Optional[List[Any]] = None,
    **options: Any
) -> DataFrame:
    """
    Read SQL database table into a DataFrame.
    """
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    reader = default_session().read
    reader.option('dbtable', table_name)
    reader.option('url', con)
    if schema is not None:
        reader.schema(schema)
    reader.options(**options)
    sdf = reader.format('jdbc').load()
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    kdf = DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names))
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        kdf = kdf[columns]
    return kdf

def read_sql_query(
    sql: str,
    con: str,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    """Read SQL query into a DataFrame."""
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    reader = default_session().read
    reader.option('query', sql)
    reader.option('url', con)
    reader.options(**options)
    sdf = reader.format('jdbc').load()
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    return DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names))

def read_sql(
    sql: str,
    con: str,
    index_col: Optional[Union[str, List[str]]] = None,
    columns: Optional[List[Any]] = None,
    **options: Any
) -> DataFrame:
    """
    Read SQL query or database table into a DataFrame.
    """
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    striped: str = sql.strip()
    if ' ' not in striped:
        return read_sql_table(sql, con, index_col=index_col, columns=columns, **options)
    else:
        return read_sql_query(sql, con, index_col=index_col, **options)

def to_datetime(
    arg: Any,
    errors: str = 'raise',
    format: Optional[str] = None,
    unit: Optional[str] = None,
    infer_datetime_format: bool = False,
    origin: Any = 'unix'
) -> Any:
    """
    Convert argument to datetime.
    """
    def pandas_to_datetime(pser_or_pdf: Any) -> Any:
        if isinstance(pser_or_pdf, pd.DataFrame):
            pser_or_pdf = pser_or_pdf[['year', 'month', 'day']]
        return pd.to_datetime(pser_or_pdf, errors=errors, format=format, unit=unit, infer_datetime_format=infer_datetime_format, origin=origin)
    if isinstance(arg, Series):
        return arg.koalas.transform_batch(pandas_to_datetime)
    if isinstance(arg, DataFrame):
        kdf = arg[['year', 'month', 'day']]
        return kdf.koalas.transform_batch(pandas_to_datetime)
    return pd.to_datetime(arg, errors=errors, format=format, unit=unit, infer_datetime_format=infer_datetime_format, origin=origin)

def date_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: Optional[Any] = None,
    tz: Optional[Any] = None,
    normalize: bool = False,
    name: Optional[str] = None,
    closed: Optional[str] = None,
    **kwargs: Any
) -> DatetimeIndex:
    """
    Return a fixed frequency DatetimeIndex.
    """
    return cast(DatetimeIndex, ks.from_pandas(pd.date_range(start=start, end=end, periods=periods, freq=freq, tz=tz, normalize=normalize, name=name, closed=closed, **kwargs)))

def get_dummies(
    data: Union[DataFrame, Series],
    prefix: Optional[Union[str, List[str], Dict[Any, str]]] = None,
    prefix_sep: str = '_',
    dummy_na: bool = False,
    columns: Optional[Any] = None,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: Optional[Any] = None
) -> DataFrame:
    """
    Convert categorical variable into dummy/indicator variables.
    """
    if sparse is not False:
        raise NotImplementedError('get_dummies currently does not support sparse')
    if columns is not None:
        if not is_list_like(columns):
            raise TypeError('Input must be a list-like for parameter `columns`')
    if dtype is None:
        dtype = 'byte'
    if isinstance(data, Series):
        if prefix is not None:
            prefix = [str(prefix)]
        kdf = data.to_frame()
        column_labels = kdf._internal.column_labels
        remaining_columns: List[Any] = []
    else:
        if isinstance(prefix, str):
            raise NotImplementedError('get_dummies currently does not support prefix as string types')
        kdf = data.copy()
        if columns is None:
            column_labels = [label for label in kdf._internal.column_labels if isinstance(kdf._internal.spark_type_for(label), _get_dummies_default_accept_types)]
        elif is_name_like_tuple(columns):
            column_labels = [label for label in kdf._internal.column_labels if label[:len(columns)] == columns]
            if len(column_labels) == 0:
                raise KeyError(name_like_string(columns))
            if prefix is None:
                prefix = [str(label[len(columns):]) if len(label) > len(columns) + 1 else label[len(columns)] if len(label) == len(columns) + 1 else '' for label in column_labels]
        elif any((isinstance(col, tuple) for col in columns)) and any((not is_name_like_tuple(col) for col in columns)):
            raise ValueError('Expected tuple, got {}'.format(type(set((col for col in columns if not is_name_like_tuple(col))).pop())))
        else:
            column_labels = [label for key in columns for label in kdf._internal.column_labels if label == key or label[0] == key]
        if len(column_labels) == 0:
            if columns is None:
                return kdf
            raise KeyError('{} not in index'.format(columns))
        if prefix is None:
            prefix = [str(label) if len(label) > 1 else label[0] for label in column_labels]
        column_labels_set = set(column_labels)
        remaining_columns = [kdf[label] if kdf._internal.column_labels_level == 1 else kdf[label].rename(name_like_string(label)) for label in kdf._internal.column_labels if label not in column_labels_set]
    if any((not isinstance(kdf._internal.spark_type_for(label), _get_dummies_acceptable_types) for label in column_labels)):
        raise NotImplementedError('get_dummies currently only accept {} values'.format(', '.join([t.typeName() for t in _get_dummies_acceptable_types])))
    if prefix is not None and len(column_labels) != len(prefix):
        raise ValueError("Length of 'prefix' ({}) did not match the length of the columns being encoded ({}).".format(len(prefix), len(column_labels)))
    elif isinstance(prefix, dict):
        prefix = [prefix[column_label[0]] for column_label in column_labels]
    all_values = _reduce_spark_multi(kdf._internal.spark_frame, [F.collect_set(kdf._internal.spark_column_for(label)) for label in column_labels])
    for i, label in enumerate(column_labels):
        values = all_values[i]
        if isinstance(values, np.ndarray):
            values = values.tolist()
        values = sorted(values)
        if drop_first:
            values = values[1:]
        def column_name(value: Any) -> str:
            if prefix is None or prefix[i] == '':
                return str(value)
            else:
                return '{}{}{}'.format(prefix[i], prefix_sep, value)
        for value in values:
            remaining_columns.append((kdf[label].notnull() & (kdf[label] == value)).astype(dtype).rename(column_name(value)))
        if dummy_na:
            remaining_columns.append(kdf[label].isnull().astype(dtype).rename(column_name(np.nan)))
    return kdf[remaining_columns]

def concat(
    objs: Iterable[Any],
    axis: Union[int, str] = 0,
    join: str = 'outer',
    ignore_index: bool = False,
    sort: bool = False
) -> Union[DataFrame, Series]:
    """
    Concatenate Koalas objects along a particular axis with optional set logic along the other axes.
    """
    if isinstance(objs, (DataFrame, IndexOpsMixin)) or not isinstance(objs, Iterable):
        raise TypeError('first argument must be an iterable of Koalas objects, you passed an object of type "{name}"'.format(name=type(objs).__name__))
    if len(cast(Sized, objs)) == 0:
        raise ValueError('No objects to concatenate')
    objs = list(filter(lambda obj: obj is not None, objs))
    if len(objs) == 0:
        raise ValueError('All objects passed were None')
    for obj in objs:
        if not isinstance(obj, (Series, DataFrame)):
            raise TypeError("cannot concatenate object of type '{name}; only ks.Series and ks.DataFrame are valid".format(name=type(obj).__name__))
    if join not in ['inner', 'outer']:
        raise ValueError('Only can inner (intersect) or outer (union) join the other axis.')
    axis = validate_axis(axis)
    if axis == 1:
        kdfs = [obj.to_frame() if isinstance(obj, Series) else obj for obj in objs]
        level = min((kdf._internal.column_labels_level for kdf in kdfs))
        kdfs = [DataFrame._index_normalized_frame(level, kdf) if kdf._internal.column_labels_level > level else kdf for kdf in kdfs]
        concat_kdf = kdfs[0]
        column_labels = concat_kdf._internal.column_labels.copy()
        kdfs_not_same_anchor = []
        for kdf in kdfs[1:]:
            duplicated = [label for label in kdf._internal.column_labels if label in column_labels]
            if len(duplicated) > 0:
                pretty_names = [name_like_string(label) for label in duplicated]
                raise ValueError('Labels have to be unique; however, got duplicated labels %s.' % pretty_names)
            column_labels.extend(kdf._internal.column_labels)
            if same_anchor(concat_kdf, kdf):
                concat_kdf = DataFrame(concat_kdf._internal.with_new_columns([concat_kdf._kser_for(label) for label in concat_kdf._internal.column_labels] + [kdf._kser_for(label) for label in kdf._internal.column_labels]))
            else:
                kdfs_not_same_anchor.append(kdf)
        if len(kdfs_not_same_anchor) > 0:
            def resolve_func(kdf_inner: DataFrame, this_column_labels: List[Any], that_column_labels: List[Any]) -> Any:
                raise AssertionError('This should not happen.')
            for kdf in kdfs_not_same_anchor:
                if join == 'inner':
                    concat_kdf = align_diff_frames(resolve_func, concat_kdf, kdf, fillna=False, how='inner')
                elif join == 'outer':
                    concat_kdf = align_diff_frames(resolve_func, concat_kdf, kdf, fillna=False, how='full')
            concat_kdf = concat_kdf[column_labels]
        if ignore_index:
            concat_kdf.columns = list(map(str, _range(len(concat_kdf.columns))))
        if sort:
            concat_kdf = concat_kdf.sort_index()
        return concat_kdf
    should_return_series = all(map(lambda obj: isinstance(obj, Series), objs))
    new_objs = []
    num_series = 0
    series_names = set()
    for obj in objs:
        if isinstance(obj, Series):
            num_series += 1
            series_names.add(obj.name)
            obj = obj.to_frame(DEFAULT_SERIES_NAME)
        new_objs.append(obj)
    objs = new_objs
    column_labels_levels = set((obj._internal.column_labels_level for obj in objs))
    if len(column_labels_levels) != 1:
        raise ValueError('MultiIndex columns should have the same levels')
    if not ignore_index:
        indices_of_kdfs = [kdf.index for kdf in objs]
        index_of_first_kdf = indices_of_kdfs[0]
        for index_of_kdf in indices_of_kdfs:
            if index_of_first_kdf.names != index_of_kdf.names:
                raise ValueError('Index type and names should be same in the objects to concatenate. You passed different indices {index_of_first_kdf} and {index_of_kdf}'.format(index_of_first_kdf=index_of_first_kdf.names, index_of_kdf=index_of_kdf.names))
    column_labels_of_kdfs = [kdf._internal.column_labels for kdf in objs]
    if ignore_index:
        index_names_of_kdfs = [[] for _ in objs]
    else:
        index_names_of_kdfs = [kdf._internal.index_names for kdf in objs]
    if all((name == index_names_of_kdfs[0] for name in index_names_of_kdfs)) and all((idx == column_labels_of_kdfs[0] for idx in column_labels_of_kdfs)):
        kdfs = objs
    elif join == 'inner':
        interested_columns = set.intersection(*map(set, column_labels_of_kdfs))
        merged_columns = [label for label in column_labels_of_kdfs[0] if label in interested_columns]
        if len(merged_columns) > 0 and len(merged_columns[0]) > 1 or sort:
            merged_columns = sorted(merged_columns, key=name_like_string)
        kdfs = [kdf[merged_columns] for kdf in objs]
    elif join == 'outer':
        merged_columns = []
        for labels in column_labels_of_kdfs:
            merged_columns.extend((label for label in labels if label not in merged_columns))
        assert len(merged_columns) > 0
        if LooseVersion(pd.__version__) < LooseVersion('0.24'):
            sort = len(merged_columns[0]) > 1 or (num_series == 0 and sort)
        else:
            sort = len(merged_columns[0]) > 1 or num_series > 1 or (num_series != 1 and sort)
        if sort:
            merged_columns = sorted(merged_columns, key=name_like_string)
        kdfs = []
        for kdf in objs:
            columns_to_add = list(set(merged_columns) - set(kdf._internal.column_labels))
            sdf = kdf._internal.resolved_copy.spark_frame
            for label in columns_to_add:
                sdf = sdf.withColumn(name_like_string(label), F.lit(None))
            data_columns = kdf._internal.data_spark_column_names + [name_like_string(label) for label in columns_to_add]
            kdf = DataFrame(kdf._internal.copy(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in kdf._internal.index_spark_column_names], column_labels=kdf._internal.column_labels + columns_to_add, data_spark_columns=[scol_for(sdf, col) for col in data_columns], data_dtypes=kdf._internal.data_dtypes + [None] * len(columns_to_add)))
            kdfs.append(kdf[merged_columns])
    if ignore_index:
        sdfs = [kdf._internal.spark_frame.select(kdf._internal.data_spark_columns) for kdf in kdfs]
    else:
        sdfs = [kdf._internal.spark_frame.select(kdf._internal.index_spark_columns + kdf._internal.data_spark_columns) for kdf in kdfs]
    concatenated = reduce(lambda x, y: x.union(y), sdfs)
    if ignore_index:
        index_spark_column_names = []
        index_names = []
        index_dtypes = []
    else:
        index_spark_column_names = kdfs[0]._internal.index_spark_column_names
        index_names = kdfs[0]._internal.index_names
        index_dtypes = kdfs[0]._internal.index_dtypes
    result_kdf = DataFrame(kdfs[0]._internal.copy(spark_frame=concatenated, index_spark_columns=[scol_for(concatenated, col) for col in index_spark_column_names], index_names=index_names, index_dtypes=index_dtypes, data_spark_columns=[scol_for(concatenated, col) for col in kdfs[0]._internal.data_spark_column_names], data_dtypes=None))
    if should_return_series:
        if len(series_names) == 1:
            name_ret = series_names.pop()
        else:
            name_ret = None
        return first_series(result_kdf).rename(name_ret)
    else:
        return result_kdf

def melt(frame: DataFrame, id_vars: Optional[Union[str, List[str]]] = None, value_vars: Optional[Union[str, List[str]]] = None, var_name: Optional[Union[str, List[str]]] = None, value_name: str = 'value') -> DataFrame:
    return DataFrame.melt(frame, id_vars, value_vars, var_name, value_name)

def isna(obj: Any) -> Any:
    """
    Detect missing values for an array-like object.
    """
    if isinstance(obj, (DataFrame, Series)):
        return obj.isnull()
    else:
        return pd.isnull(obj)
isnull = isna

def notna(obj: Any) -> Any:
    """
    Detect existing (non-missing) values.
    """
    if isinstance(obj, (DataFrame, Series)):
        return obj.notna()
    else:
        return pd.notna(obj)
notnull = notna

def merge(
    obj: DataFrame,
    right: DataFrame,
    how: str = 'inner',
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    left_index: bool = False,
    right_index: bool = False,
    suffixes: Tuple[str, str] = ('_x', '_y')
) -> DataFrame:
    """
    Merge DataFrame objects with a database-style join.
    """
    return obj.merge(right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, suffixes=suffixes)

def to_numeric(arg: Any) -> Any:
    """
    Convert argument to a numeric type.
    """
    if isinstance(arg, Series):
        return arg._with_new_scol(arg.spark.column.cast('float'))
    else:
        return pd.to_numeric(arg)

def broadcast(obj: DataFrame) -> DataFrame:
    """
    Marks a DataFrame as small enough for use in broadcast joins.
    """
    if not isinstance(obj, DataFrame):
        raise ValueError('Invalid type : expected DataFrame got {}'.format(type(obj).__name__))
    return DataFrame(obj._internal.with_new_sdf(F.broadcast(obj._internal.resolved_copy.spark_frame)))

def read_orc(
    path: str,
    columns: Optional[List[Any]] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options: Any
) -> DataFrame:
    """
    Load an ORC object from the file path, returning a DataFrame.
    """
    if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
        options = options.get('options')
    kdf = read_spark_io(path, format='orc', index_col=index_col, **options)
    if columns is not None:
        kdf_columns = kdf.columns
        new_columns: List[Any] = []
        for column in list(columns):
            if column in kdf_columns:
                new_columns.append(column)
            else:
                raise ValueError("Unknown column name '{}'".format(column))
        kdf = kdf[new_columns]
    return kdf

def _get_index_map(sdf: Any, index_col: Optional[Union[str, List[str]]] = None) -> Tuple[Optional[List[Any]], Optional[List[Tuple[str, ...]]]]:
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
_get_dummies_acceptable_types = _get_dummies_default_accept_types + (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, BooleanType, TimestampType)