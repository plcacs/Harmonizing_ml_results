#!/usr/bin/env python
# type: ignore
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Iterable, Mapping
from functools import partial, reduce
import datetime
import re
import sys
import warnings
import inspect
import types
import json
from itertools import zip_longest
from distutils.version import LooseVersion
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, cast

import numpy as np
import pandas as pd

from pyspark import StorageLevel
from pyspark.sql import DataFrame as SparkDataFrame, Window, functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType, StringType, StructField, StructType, ArrayType

# Assume the following helper objects and functions are defined elsewhere:
# _MissingPandasLikeDataFrame, InternalFrame, CachedAccessor, Series, Index,
# SeriesType, DataFrameType, ScalarType, as_nullable_spark_type, spark_type_to_pandas_dtype,
# infer_dtype_from_object, is_name_like_value, is_name_like_tuple, is_scalar, 
# align_diff_frames, verify_temp_column_name, get_option, force_decimal_precision_scale,
# _create_tuple_for_frame_type, is_testing
# For brevity, those definitions are omitted.

def _reduce_spark_multi(sdf: SparkDataFrame, aggs: List[Any]) -> List[Any]:
    """
    Performs a reduction on a spark DataFrame, the functions being known sql aggregate functions.
    """
    assert isinstance(sdf, SparkDataFrame)
    sdf0 = sdf.agg(*aggs)
    l = sdf0.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row = l.iloc[0]
    l2 = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2

class CachedDataFrame(DataFrame):
    """
    Cached Koalas DataFrame, which corresponds to pandas DataFrame logically, but internally
    it caches the corresponding Spark DataFrame.
    """
    def __init__(self, internal: InternalFrame, storage_level: Optional[StorageLevel] = None) -> None:
        if storage_level is None:
            object.__setattr__(self, '_cached', internal.spark_frame.cache())
        elif isinstance(storage_level, StorageLevel):
            object.__setattr__(self, '_cached', internal.spark_frame.persist(storage_level))
        else:
            raise TypeError('Only a valid pyspark.StorageLevel type is acceptable for the `storage_level`')
        super().__init__(internal)

    def __enter__(self) -> "CachedDataFrame":
        return self

    def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
        self.spark.unpersist()

    @property
    def storage_level(self) -> Any:
        warnings.warn('DataFrame.storage_level is deprecated as of DataFrame.spark.storage_level. Please use the API instead.', FutureWarning)
        return self.spark.storage_level

    def unpersist(self) -> Any:
        warnings.warn('DataFrame.unpersist is deprecated as of DataFrame.spark.unpersist. Please use the API instead.', FutureWarning)
        return self.spark.unpersist()

    def hint(self, name: str, *parameters: Any) -> Any:
        warnings.warn('DataFrame.hint is deprecated as of DataFrame.spark.hint. Please use the API instead.', FutureWarning)
        return self.spark.hint(name, *parameters)

    def to_table(self, name: str, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> Any:
        return self.spark.to_table(name, format, mode, partition_cols, index_col, **options)

    def to_delta(self, path: str, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        self.spark.to_spark_io(path=path, mode=mode, format='delta', partition_cols=partition_cols, index_col=index_col, **options)

    def to_parquet(self, path: str, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, compression: Optional[str] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        builder = self.to_spark(index_col=index_col).write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format('parquet').save(path)

    def to_orc(self, path: str, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='orc', partition_cols=partition_cols, index_col=index_col, **options)

    def to_spark_io(self, path: Optional[str] = None, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> Any:
        return self.spark.to_spark_io(path, format, mode, partition_cols, index_col, **options)

    def to_spark(self, index_col: Optional[Union[str, List[str]]] = None) -> SparkDataFrame:
        return self.spark.frame(index_col)

    def to_pandas(self) -> pd.DataFrame:
        return self._internal.to_pandas_frame.copy()

    def toPandas(self) -> pd.DataFrame:
        warnings.warn('DataFrame.toPandas is deprecated as of DataFrame.to_pandas. Please use the API instead.', FutureWarning)
        return self.to_pandas()

    def assign(self, **kwargs: Any) -> "DataFrame":
        return self._assign(kwargs)

    def apply_batch(self, func: Any, args: Tuple[Any, ...] = (), **kwds: Any) -> Any:
        warnings.warn('DataFrame.apply_batch is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.', FutureWarning)
        return self.koalas.apply_batch(func, args=args, **kwds)

    def map_in_pandas(self, func: Any) -> Any:
        warnings.warn('DataFrame.map_in_pandas is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.', FutureWarning)
        return self.koalas.apply_batch(func)

    def apply(self, func: Any, axis: Union[int, str] = 0, args: Tuple[Any, ...] = (), **kwds: Any) -> Union["DataFrame", "Series"]:
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.series import first_series
        if not isinstance(func, types.FunctionType):
            assert callable(func), 'the first argument should be a callable function.'
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = validate_axis(axis)
        should_return_series: bool = False
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get('return', None)
        should_infer_schema: bool = return_sig is None
        should_use_map_in_pandas: bool = LooseVersion(pyspark.__version__) >= LooseVersion('3.0')

        def apply_func(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf_or_pser = pdf.apply(func, axis=axis, args=args, **kwds)
            if isinstance(pdf_or_pser, pd.Series):
                return pdf_or_pser.to_frame()
            else:
                return pdf_or_pser

        self_applied = DataFrame(self._internal.resolved_copy)
        column_labels: Optional[List[Tuple[Any, ...]]] = None
        if should_infer_schema:
            limit = get_option('compute.shortcut_limit')
            pdf: pd.DataFrame = self_applied.head(limit + 1)._to_internal_pandas()
            applied = pdf.apply(func, axis=axis, args=args, **kwds)
            kser_or_kdf = ks.from_pandas(applied)
            if len(pdf) <= limit:
                return kser_or_kdf
            kdf = kser_or_kdf
            if isinstance(kser_or_kdf, ks.Series):
                should_return_series = True
                kdf = kser_or_kdf._kdf
            return_schema = force_decimal_precision_scale(as_nullable_spark_type(kdf._internal.to_internal_spark_frame.schema))
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(self_applied, apply_func, return_schema, retain_index=True)
                sdf = self_applied._internal.to_internal_spark_frame.mapInPandas(lambda iterator: map(output_func, iterator), schema=return_schema)
            else:
                sdf = GroupBy._spark_group_map_apply(self_applied, apply_func, (F.spark_partition_id(),), return_schema, retain_index=True)
            internal = kdf._internal.with_new_sdf(sdf)
        else:
            return_type = infer_return_type(func)
            require_index_axis = isinstance(return_type, SeriesType)
            require_column_axis = isinstance(return_type, DataFrameType)
            if require_index_axis:
                if axis != 0:
                    raise TypeError("The given function should specify a scalar or a series as its type hints when axis is 0 or 'index'; however, the return type was %s" % return_sig)
                return_schema = cast(SeriesType, return_type).spark_type
                fields_types = zip(self_applied.columns, [return_schema] * len(self_applied.columns))
                return_schema = StructType([StructField(c, t) for c, t in fields_types])
                data_dtypes = [cast(SeriesType, return_type).dtype] * len(self_applied.columns)
            elif require_column_axis:
                if axis != 1:
                    raise TypeError("The given function should specify a scalar or a frame as its type hints when axis is 1 or 'column'; however, the return type was %s" % return_sig)
                return_schema = cast(DataFrameType, return_type).spark_type
                data_dtypes = cast(DataFrameType, return_type).dtypes
            else:
                should_return_series = True
                return_schema = cast(ScalarType, return_type).spark_type
                return_schema = StructType([StructField(SPARK_DEFAULT_SERIES_NAME, return_schema)])
                data_dtypes = [cast(ScalarType, return_type).dtype]
                column_labels = [None]
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(self_applied, apply_func, return_schema, retain_index=False)
                sdf = self_applied._internal.to_internal_spark_frame.mapInPandas(lambda iterator: map(output_func, iterator), schema=return_schema)
            else:
                sdf = GroupBy._spark_group_map_apply(self_applied, apply_func, (F.spark_partition_id(),), return_schema, retain_index=False)
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=None, column_labels=column_labels, data_dtypes=data_dtypes)
        result = DataFrame(internal)
        if should_return_series:
            return first_series(result)
        else:
            return result

    def transform(self, func: Any, axis: Union[int, str] = 0, *args: Any, **kwargs: Any) -> "DataFrame":
        if not isinstance(func, types.FunctionType):
            assert callable(func), 'the first argument should be a callable function.'
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get('return', None)
        should_infer_schema: bool = return_sig is None
        if should_infer_schema:
            limit = get_option('compute.shortcut_limit')
            pdf: pd.DataFrame = self.head(limit + 1)._to_internal_pandas()
            transformed = pdf.transform(func, axis, *args, **kwargs)
            kdf = DataFrame(transformed)
            if len(pdf) <= limit:
                return kdf
            applied = []
            for input_label, output_label in zip(self._internal.column_labels, kdf._internal.column_labels):
                kser = self._kser_for(input_label)
                dtype = kdf._internal.dtype_for(output_label)
                return_schema = force_decimal_precision_scale(as_nullable_spark_type(kdf._internal.spark_type_for(output_label)))
                applied.append(kser.koalas._transform_batch(func=lambda c: func(c, *args, **kwargs), return_type=SeriesType(dtype, return_schema)))
            internal = self._internal.with_new_columns(applied, data_dtypes=kdf._internal.data_dtypes)
            return DataFrame(internal)
        else:
            return self._apply_series_op(lambda kser: kser.koalas.transform_batch(func, *args, **kwargs))

    def transform_batch(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        warnings.warn('DataFrame.transform_batch is deprecated as of DataFrame.koalas.transform_batch. Please use the API instead.', FutureWarning)
        return self.koalas.transform_batch(func, *args, **kwargs)

    def pop(self, item: Any) -> "Series":
        result = self[item]
        self._update_internal_frame(self.drop(item)._internal)
        return result

    def xs(self, key: Union[Any, Tuple[Any, ...]], axis: Union[int, str] = 0, level: Any = None) -> Union["DataFrame", "Series"]:
        from databricks.koalas.series import first_series
        if not is_name_like_value(key):
            raise ValueError("'key' should be a scalar value or tuple that contains scalar values")
        if level is not None and is_name_like_tuple(key):
            raise KeyError(key)
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if not is_name_like_tuple(key):
            key = (key,)
        if len(key) > self._internal.index_level:
            raise KeyError('Key length ({}) exceeds index depth ({})'.format(len(key), self._internal.index_level))
        if level is None:
            level = 0
        rows = [self._internal.index_spark_columns[lvl] == index for lvl, index in enumerate(key, level)]
        internal = self._internal.with_filter(reduce(lambda x, y: x & y, rows))
        if len(key) == self._internal.index_level:
            kdf = DataFrame(internal)
            pdf = kdf.head(2)._to_internal_pandas()
            if len(pdf) == 0:
                raise KeyError(key)
            elif len(pdf) > 1:
                return kdf
            else:
                return first_series(DataFrame(pdf.transpose()))
        else:
            index_spark_columns = internal.index_spark_columns[:level] + internal.index_spark_columns[level + len(key):]
            index_names = internal.index_names[:level] + internal.index_names[level + len(key):]
            index_dtypes = internal.index_dtypes[:level] + internal.index_dtypes[level + len(key):]
            internal = internal.copy(index_spark_columns=index_spark_columns, index_names=index_names, index_dtypes=index_dtypes).resolved_copy
            return DataFrame(internal)

    def between_time(self, start_time: Union[datetime.time, str], end_time: Union[datetime.time, str], include_start: bool = True, include_end: bool = True, axis: Union[int, str] = 0) -> "DataFrame":
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('between_time currently only works for axis=0')
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError('Index must be DatetimeIndex')
        kdf = self.copy()
        kdf.index.name = verify_temp_column_name(kdf, '__index_name__')
        return_types: List[Any] = [kdf.index.dtype] + list(kdf.dtypes)
        def pandas_between_time(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.between_time(start_time, end_time, include_start, include_end).reset_index()
        with option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_between_time)
        return DataFrame(self._internal.copy(spark_frame=kdf._internal.spark_frame, index_spark_columns=kdf._internal.data_spark_columns[:1], data_spark_columns=kdf._internal.data_spark_columns[1:]))

    def at_time(self, time: Union[datetime.time, str], asof: bool = False, axis: Union[int, str] = 0) -> "DataFrame":
        if asof:
            raise NotImplementedError("'asof' argument is not supported")
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('at_time currently only works for axis=0')
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError('Index must be DatetimeIndex')
        kdf = self.copy()
        kdf.index.name = verify_temp_column_name(kdf, '__index_name__')
        return_types: List[Any] = [kdf.index.dtype] + list(kdf.dtypes)
        if LooseVersion(pd.__version__) < LooseVersion('0.24'):
            def pandas_at_time(pdf: pd.DataFrame) -> pd.DataFrame:
                return pdf.at_time(time, asof).reset_index()
        else:
            def pandas_at_time(pdf: pd.DataFrame) -> pd.DataFrame:
                return pdf.at_time(time, asof, axis).reset_index()
        with option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_at_time)
        return DataFrame(self._internal.copy(spark_frame=kdf._internal.spark_frame, index_spark_columns=kdf._internal.data_spark_columns[:1], data_spark_columns=kdf._internal.data_spark_columns[1:]))

    def where(self, cond: Union["DataFrame", "Series"], other: Any = np.nan) -> "DataFrame":
        from databricks.koalas.series import Series
        tmp_cond_col_name = '__tmp_cond_col_{}__'.format
        tmp_other_col_name = '__tmp_other_col_{}__'.format
        kdf = self.copy()
        tmp_cond_col_names: List[str] = [tmp_cond_col_name(name_like_string(label)) for label in self._internal.column_labels]
        if isinstance(cond, DataFrame):
            cond = cond[[(cond._internal.spark_column_for(label) if label in cond._internal.column_labels else F.lit(False)).alias(name) for label, name in zip(self._internal.column_labels, tmp_cond_col_names)]]
            kdf[tmp_cond_col_names] = cond
        elif isinstance(cond, Series):
            cond = cond.to_frame()
            cond = cond[[cond._internal.data_spark_columns[0].alias(name) for name in tmp_cond_col_names]]
            kdf[tmp_cond_col_names] = cond
        else:
            raise ValueError('type of cond must be a DataFrame or Series')
        tmp_other_col_names: List[str] = [tmp_other_col_name(name_like_string(label)) for label in self._internal.column_labels]
        if isinstance(other, DataFrame):
            other = other[[(other._internal.spark_column_for(label) if label in other._internal.column_labels else F.lit(np.nan)).alias(name) for label, name in zip(self._internal.column_labels, tmp_other_col_names)]]
            kdf[tmp_other_col_names] = other
        elif isinstance(other, Series):
            other = other.to_frame()
            other = other[[other._internal.data_spark_columns[0].alias(name) for name in tmp_other_col_names]]
            kdf[tmp_other_col_names] = other
        else:
            for label in self._internal.column_labels:
                kdf[tmp_other_col_name(name_like_string(label))] = other
        data_spark_columns = []
        for label in self._internal.column_labels:
            data_spark_columns.append(F.when(kdf[tmp_cond_col_name(name_like_string(label))].spark.column, kdf._internal.spark_column_for(label)).otherwise(kdf[tmp_other_col_name(name_like_string(label))].spark.column).alias(kdf._internal.spark_column_name_for(label)))
        return DataFrame(kdf._internal.with_new_columns(data_spark_columns, column_labels=self._internal.column_labels))

    def mask(self, cond: Union["DataFrame", "Series"], other: Any = np.nan) -> "DataFrame":
        if not isinstance(cond, (DataFrame, Series)):
            raise ValueError('type of cond must be a DataFrame or Series')
        cond_inversed = cond._apply_series_op(lambda kser: ~kser)
        return self.where(cond_inversed, other)

    @property
    def index(self) -> "Index":
        from databricks.koalas.indexes.base import Index
        return Index._new_instance(self)

    @property
    def empty(self) -> bool:
        return len(self._internal.column_labels) == 0 or self._internal.resolved_copy.spark_frame.rdd.isEmpty()

    @property
    def style(self) -> Any:
        max_results = get_option('compute.max_rows')
        pdf = self.head(max_results + 1)._to_internal_pandas()
        if len(pdf) > max_results:
            warnings.warn("'style' property will only use top %s rows." % max_results, UserWarning)
        return pdf.head(max_results).style

    def set_index(self, keys: Union[Any, List[Any]], drop: bool = True, append: bool = False, inplace: bool = False) -> Optional["DataFrame"]:
        if is_name_like_tuple(keys):
            keys = [keys]
        elif is_name_like_value(keys):
            keys = [(keys,)]
        else:
            keys = [key if is_name_like_tuple(key) else (key,) for key in keys]
        columns = set(self._internal.column_labels)
        for key in keys:
            if key not in columns:
                raise KeyError(name_like_string(key))
        if drop:
            column_labels = [label for label in self._internal.column_labels if label not in keys]
        else:
            column_labels = self._internal.column_labels
        if append:
            index_spark_columns = self._internal.index_spark_columns + [self._internal.spark_column_for(label) for label in keys]
            index_names = self._internal.index_names + keys
            index_dtypes = self._internal.index_dtypes + [self._internal.dtype_for(label) for label in keys]
        else:
            index_spark_columns = [self._internal.spark_column_for(label) for label in keys]
            index_names = keys
            index_dtypes = [self._internal.dtype_for(label) for label in keys]
        internal = self._internal.copy(index_spark_columns=index_spark_columns, index_names=index_names, index_dtypes=index_dtypes, column_labels=column_labels, data_spark_columns=[self._internal.spark_column_for(label) for label in column_labels], data_dtypes=[self._internal.dtype_for(label) for label in column_labels])
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def reset_index(self, level: Optional[Union[int, str, Iterable[Union[int,str]]]] = None, drop: bool = False, inplace: bool = False, col_level: Union[int, str] = 0, col_fill: Any = '') -> Optional["DataFrame"]:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        multi_index = self._internal.index_level > 1
        def rename(index: int) -> Tuple[str, ...]:
            if multi_index:
                return ('level_{}'.format(index),)
            elif ('index',) not in self._internal.column_labels:
                return ('index',)
            else:
                return ('level_{}'.format(index),)
        if level is None:
            new_column_labels = [name if name is not None else rename(i) for i, name in enumerate(self._internal.index_names)]
            new_data_spark_columns = [scol.alias(name_like_string(label)) for scol, label in zip(self._internal.index_spark_columns, new_column_labels)]
            new_data_dtypes = self._internal.index_dtypes
            index_spark_columns: List[Any] = []
            index_names: List[Any] = []
            index_dtypes: List[Any] = []
        else:
            if is_list_like(level):
                level = list(level)
            if isinstance(level, int) or is_name_like_tuple(level):
                level = [level]
            elif is_name_like_value(level):
                level = [(level,)]
            else:
                level = [lvl if isinstance(lvl, int) or is_name_like_tuple(lvl) else (lvl,) for lvl in level]
            if all((isinstance(l, int) for l in level)):
                for lev in level:
                    if lev >= self._internal.index_level:
                        raise IndexError('Too many levels: Index has only {} levels, not {}'.format(self._internal.index_level, lev + 1))
                idx = level
            elif all((is_name_like_tuple(lev) for lev in level)):
                idx = []
                for l in level:
                    try:
                        i = self._internal.index_names.index(l)
                        idx.append(i)
                    except ValueError:
                        if multi_index:
                            raise KeyError('Level unknown not found')
                        else:
                            raise KeyError('Level unknown must be same as name ({})'.format(name_like_string(self._internal.index_names[0])))
            else:
                raise ValueError('Level should be all int or all string.')
            idx.sort()
            new_column_labels = []
            new_data_spark_columns = []
            new_data_dtypes = []
            index_spark_columns = self._internal.index_spark_columns.copy()
            index_names = self._internal.index_names.copy()
            index_dtypes = self._internal.index_dtypes.copy()
            for i in idx[::-1]:
                name = index_names.pop(i)
                new_column_labels.insert(0, name if name is not None else rename(i))
                scol = index_spark_columns.pop(i)
                new_data_spark_columns.insert(0, scol.alias(name_like_string(name)))
                new_data_dtypes.insert(0, index_dtypes.pop(i))
        if drop:
            new_data_spark_columns = []
            new_column_labels = []
            new_data_dtypes = []
        for label in new_column_labels:
            if label in self._internal.column_labels:
                raise ValueError('cannot insert {}, already exists'.format(name_like_string(label)))
        if self._internal.column_labels_level > 1:
            column_depth = len(self._internal.column_labels[0])
            if col_level >= column_depth:
                raise IndexError('Too many levels: Index has only {} levels, not {}'.format(column_depth, col_level + 1))
            if any((col_level + len(label) > column_depth for label in new_column_labels)):
                raise ValueError('Item must have length equal to number of levels.')
            new_column_labels = [tuple(list(label) + [''] * (column_depth - len(label) - col_level)) for label in new_column_labels]
        internal = self._internal.copy(index_spark_columns=index_spark_columns, index_names=index_names, index_dtypes=index_dtypes, column_labels=new_column_labels + self._internal.column_labels, data_spark_columns=new_data_spark_columns + self._internal.data_spark_columns, data_dtypes=new_data_dtypes + self._internal.data_dtypes)
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def isnull(self) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser.isnull())
    isna = isnull

    def notnull(self) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser.notnull())
    notna = notnull

    def insert(self, loc: int, column: Any, value: Any, allow_duplicates: bool = False) -> None:
        if not isinstance(loc, int):
            raise TypeError('loc must be int')
        assert 0 <= loc <= len(self.columns)
        assert allow_duplicates is False
        if not is_name_like_value(column):
            raise ValueError('"column" should be a scalar value or tuple that contains scalar values')
        if is_name_like_tuple(column):
            if len(column) != len(self.columns.levels):
                raise ValueError('"column" must have length equal to number of column levels.')
        if column in self.columns:
            raise ValueError('cannot insert %s, already exists' % column)
        kdf = self.copy()
        kdf[column] = value
        cols = kdf.columns[:-1].insert(loc, kdf.columns[-1])
        kdf = kdf[cols]
        self._update_internal_frame(kdf._internal)

    def shift(self, periods: int = 1, fill_value: Any = None) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser._shift(periods, fill_value), should_resolve=True)

    def diff(self, periods: int = 1, axis: Union[int, str] = 0) -> "DataFrame":
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        return self._apply_series_op(lambda kser: kser._diff(periods), should_resolve=True)

    def nunique(self, axis: Union[int, str] = 0, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> "Series":
        from databricks.koalas.series import first_series
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        sdf = self._internal.spark_frame.select([F.lit(None).cast(StringType()).alias(SPARK_DEFAULT_INDEX_NAME)] + [self._kser_for(label)._nunique(dropna, approx, rsd) for label in self._internal.column_labels])
        with ks.option_context('compute.max_rows', 1):
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, SPARK_DEFAULT_INDEX_NAME)], index_names=[None], index_dtypes=[None], column_labels=self._internal.column_labels, column_label_names=self._internal.column_label_names)
            return first_series(DataFrame(internal).transpose())

    def round(self, decimals: Union[int, Dict[Any, int], "Series"]) -> "DataFrame":
        if isinstance(decimals, ks.Series):
            decimals = {k if isinstance(k, tuple) else (k,): v for k, v in decimals._to_internal_pandas().items()}
        elif isinstance(decimals, dict):
            decimals = {k if is_name_like_tuple(k) else (k,): v for k, v in decimals.items()}
        elif isinstance(decimals, int):
            decimals = {k: decimals for k in self._internal.column_labels}
        else:
            raise ValueError('decimals must be an integer, a dict-like or a Series')
        def op(kser: "Series") -> Any:
            label = kser._column_label
            if label in decimals:
                return F.round(kser.spark.column, decimals[label]).alias(kser._internal.data_spark_column_names[0])
            else:
                return kser
        return self._apply_series_op(op)

    def _mark_duplicates(self, subset: Optional[Any] = None, keep: Union[str, bool] = 'first') -> Tuple[SparkDataFrame, str]:
        if subset is None:
            subset = self._internal.column_labels
        else:
            if is_name_like_tuple(subset):
                subset = [subset]
            elif is_name_like_value(subset):
                subset = [(subset,)]
            else:
                subset = [sub if is_name_like_tuple(sub) else (sub,) for sub in subset]
            diff = set(subset).difference(set(self._internal.column_labels))
            if len(diff) > 0:
                raise KeyError(', '.join([name_like_string(d) for d in diff]))
        group_cols = [self._internal.spark_column_name_for(label) for label in subset]
        sdf = self._internal.resolved_copy.spark_frame
        column = verify_temp_column_name(sdf, '__duplicated__')
        if keep == 'first' or keep == 'last':
            if keep == 'first':
                ord_func = F.asc
            else:
                ord_func = F.desc
            window = Window.partitionBy(group_cols).orderBy(ord_func(NATURAL_ORDER_COLUMN_NAME)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            sdf = sdf.withColumn(column, F.row_number().over(window) > 1)
        elif not keep:
            window = Window.partitionBy(group_cols).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            sdf = sdf.withColumn(column, F.count('*').over(window) > 1)
        else:
            raise ValueError("'keep' only supports 'first', 'last' and False")
        return (sdf, column)

    def duplicated(self, subset: Optional[Any] = None, keep: Union[str, bool] = 'first') -> "Series":
        from databricks.koalas.series import first_series
        sdf, column = self._mark_duplicates(subset, keep)
        sdf = sdf.select(self._internal.index_spark_columns + [scol_for(sdf, column).alias(SPARK_DEFAULT_SERIES_NAME)])
        return first_series(DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names], index_names=self._internal.index_names, index_dtypes=self._internal.index_dtypes, column_labels=[None], data_spark_columns=[scol_for(sdf, SPARK_DEFAULT_SERIES_NAME)])))

    def dot(self, other: "Series") -> "Series":
        if not isinstance(other, ks.Series):
            raise TypeError('Unsupported type {}'.format(type(other).__name__))
        else:
            return cast(ks.Series, other.dot(self.transpose())).rename(None)

    def __matmul__(self, other: "Series") -> "Series":
        return self.dot(other)

    def to_koalas(self, index_col: Optional[Union[str, List[str]]] = None) -> "DataFrame":
        if isinstance(self, DataFrame):
            return self
        else:
            assert isinstance(self, SparkDataFrame), type(self)
            from databricks.koalas.namespace import _get_index_map
            index_spark_columns, index_names = _get_index_map(self, index_col)
            internal = InternalFrame(spark_frame=self, index_spark_columns=index_spark_columns, index_names=index_names)
            return DataFrame(internal)

    def cache(self) -> "DataFrame":
        warnings.warn('DataFrame.cache is deprecated as of DataFrame.spark.cache. Please use the API instead.', FutureWarning)
        return self.spark.cache()

    def persist(self, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> "DataFrame":
        warnings.warn('DataFrame.persist is deprecated as of DataFrame.spark.persist. Please use the API instead.', FutureWarning)
        return self.spark.persist(storage_level)

    def hint(self, name: str, *parameters: Any) -> "DataFrame":
        warnings.warn('DataFrame.hint is deprecated as of DataFrame.spark.hint. Please use the API instead.', FutureWarning)
        return self.spark.hint(name, *parameters)

    def to_table(self, name: str, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> "DataFrame":
        return self.spark.to_table(name, format, mode, partition_cols, index_col, **options)

    def to_delta(self, path: str, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        warnings.warn('DataFrame.to_delta is deprecated in favor of DataFrame.to_table with format "delta".', FutureWarning)
        self.spark.to_spark_io(path=path, mode=mode, format='delta', partition_cols=partition_cols, index_col=index_col, **options)

    def to_parquet(self, path: str, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, compression: Optional[str] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        builder = self.to_spark(index_col=index_col).write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format('parquet').save(path)

    def to_orc(self, path: str, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='orc', partition_cols=partition_cols, index_col=index_col, **options)

    def to_spark_io(self, path: Optional[str] = None, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> Any:
        return self.spark.to_spark_io(path, format, mode, partition_cols, index_col, **options)

    def to_spark(self, index_col: Optional[Union[str, List[str]]] = None) -> SparkDataFrame:
        return self.spark.frame(index_col)

    def to_pandas(self) -> pd.DataFrame:
        return self._internal.to_pandas_frame.copy()

    def toPandas(self) -> pd.DataFrame:
        warnings.warn('DataFrame.toPandas is deprecated as of DataFrame.to_pandas. Please use the API instead.', FutureWarning)
        return self.to_pandas()

    def assign(self, **kwargs: Any) -> "DataFrame":
        return self._assign(kwargs)

    def _assign(self, kwargs: Dict[str, Any]) -> "DataFrame":
        from databricks.koalas.indexes import MultiIndex
        from databricks.koalas.series import IndexOpsMixin
        for k, v in kwargs.items():
            is_invalid_assignee = not (isinstance(v, (IndexOpsMixin, spark.Column)) or callable(v) or is_scalar(v)) or isinstance(v, MultiIndex)
            if is_invalid_assignee:
                raise TypeError("Column assignment doesn't support type {0}".format(type(v).__name__))
            if callable(v):
                kwargs[k] = v(self)
        pairs: Dict[Tuple[Any, ...], Tuple[Any, Optional[Any]]] = {
            k if is_name_like_tuple(k) else (k,): (v.spark.column, v.dtype) if isinstance(v, IndexOpsMixin) and (not isinstance(v, MultiIndex)) else (v, None) if isinstance(v, spark.Column) else (F.lit(v), None)
            for k, v in kwargs.items()
        }
        scols = []
        data_dtypes = []
        for label in self._internal.column_labels:
            for i in range(len(label)):
                if label[:len(label) - i] in pairs:
                    scols.append(pairs[label[:len(label) - i]][0].alias(self._internal.spark_column_name_for(label)))
                    data_dtypes.append(pairs[label[:len(label) - i]][1])
                    break
            else:
                scols.append(self._internal.spark_column_for(label))
                data_dtypes.append(self._internal.dtype_for(label))
        for label, (scol, dtype) in pairs.items():
            if label not in set((i[:len(label)] for i in self._internal.column_labels)):
                scols.append(scol.alias(name_like_string(label)))
                self._internal.column_labels.append(label)
                data_dtypes.append(dtype)
        level = self._internal.column_labels_level
        column_labels = [tuple(list(label) + [''] * (level - len(label))) for label in self._internal.column_labels]
        internal = self._internal.with_new_columns(scols, column_labels=column_labels, data_dtypes=data_dtypes)
        return DataFrame(internal)

    @staticmethod
    def from_records(data: Any, index: Any = None, exclude: Any = None, columns: Any = None, coerce_float: bool = False, nrows: Optional[int] = None) -> "DataFrame":
        return DataFrame(pd.DataFrame.from_records(data=data, index=index, exclude=exclude, columns=columns, coerce_float=coerce_float, nrows=nrows))

    def _to_internal_pandas(self) -> pd.DataFrame:
        return self._internal.to_pandas_frame

    def _get_or_create_repr_pandas_cache(self, n: int) -> pd.DataFrame:
        if not hasattr(self, '_repr_pandas_cache') or n not in self._repr_pandas_cache:
            object.__setattr__(self, '_repr_pandas_cache', {n: self.head(n + 1)._to_internal_pandas()})
        return self._repr_pandas_cache[n]

    def __repr__(self) -> str:
        max_display_count = get_option('display.max_rows')
        if max_display_count is None:
            return self._to_internal_pandas().to_string()
        pdf = self._get_or_create_repr_pandas_cache(max_display_count)
        pdf_length = len(pdf)
        pdf = pdf.iloc[:max_display_count]
        if pdf_length > max_display_count:
            repr_string = pdf.to_string(show_dimensions=True)
            match = REPR_PATTERN.search(repr_string)
            if match is not None:
                nrows = match.group('rows')
                ncols = match.group('columns')
                footer = '\n\n[Showing only the first {nrows} rows x {ncols} columns]'.format(nrows=nrows, ncols=ncols)
                return REPR_PATTERN.sub(footer, repr_string)
        return pdf.to_string()

    def _repr_html_(self) -> str:
        max_display_count = get_option('display.max_rows')
        bold_rows = not LooseVersion('0.25.1') == LooseVersion(pd.__version__)
        if max_display_count is None:
            return self._to_internal_pandas().to_html(notebook=True, bold_rows=bold_rows)
        pdf = self._get_or_create_repr_pandas_cache(max_display_count)
        pdf_length = len(pdf)
        pdf = pdf.iloc[:max_display_count]
        if pdf_length > max_display_count:
            repr_html = pdf.to_html(show_dimensions=True, notebook=True, bold_rows=bold_rows)
            match = REPR_HTML_PATTERN.search(repr_html)
            if match is not None:
                nrows = match.group('rows')
                ncols = match.group('columns')
                by = chr(215)
                footer = '\n<p>Showing only the first {rows} rows {by} {cols} columns</p>\n</div>'.format(rows=nrows, by=by, cols=ncols)
                return REPR_HTML_PATTERN.sub(footer, repr_html)
        return pdf.to_html(notebook=True, bold_rows=bold_rows)

    def __getitem__(self, key: Any) -> Any:
        from databricks.koalas.series import Series
        if key is None:
            raise KeyError('none key')
        elif isinstance(key, Series):
            return self.loc[key.astype(bool)]
        elif isinstance(key, slice):
            if any((type(n) == int or n is None for n in [key.start, key.stop])):
                return self.iloc[key]
            return self.loc[key]
        elif is_name_like_value(key):
            return self.loc[:, key]
        elif is_list_like(key):
            return self.loc[:, list(key)]
        raise NotImplementedError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        from databricks.koalas.series import Series
        if isinstance(value, (DataFrame, Series)) and (not same_anchor(value, self)):
            level = self._internal.column_labels_level
            key = DataFrame._index_normalized_label(level, key)
            value = DataFrame._index_normalized_frame(level, value)
            def assign_columns(kdf: "DataFrame", this_column_labels: List[Tuple[Any, ...]], that_column_labels: List[Tuple[Any, ...]]) -> Iterator[Tuple["Series", Tuple[Any, ...]]]:
                assert len(key) == len(that_column_labels)
                for k, this_label, that_label in zip_longest(key, this_column_labels, that_column_labels):
                    yield (kdf._kser_for(that_label), tuple(['that', *k]))
                    if this_label is not None and this_label[1:] != k:
                        yield (kdf._kser_for(this_label), this_label)
            kdf = align_diff_frames(assign_columns, self, value, fillna=True, how='full')
        elif isinstance(value, list):
            if len(self) != len(value):
                raise ValueError('Length of values does not match length of index')
            with option_context('compute.default_index_type', 'distributed-sequence', 'compute.ops_on_diff_frames', True):
                kdf = self.reset_index()
                kdf[key] = ks.DataFrame(value)
                kdf = kdf.set_index(kdf.columns[:self._internal.index_level])
                kdf.index.names = self.index.names
        elif isinstance(key, list):
            assert isinstance(value, DataFrame)
            field_names = value.columns
            kdf = self._assign({k: value[c] for k, c in zip(key, field_names)})
        else:
            kdf = self._assign({key: value})
        self._update_internal_frame(kdf._internal)

    def __getattr__(self, key: str) -> Any:
        if key.startswith('__'):
            raise AttributeError(key)
        if hasattr(_MissingPandasLikeDataFrame, key):
            property_or_func = getattr(_MissingPandasLikeDataFrame, key)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        try:
            return self.loc[:, key]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, key))

    def __setattr__(self, key: str, value: Any) -> None:
        try:
            object.__getattribute__(self, key)
            return object.__setattr__(self, key, value)
        except AttributeError:
            pass
        if (key,) in self._internal.column_labels:
            self[key] = value
        else:
            msg = "Koalas doesn't allow columns to be created via a new attribute name"
            if is_testing():
                raise AssertionError(msg)
            else:
                warnings.warn(msg, UserWarning)

    def __len__(self) -> int:
        return self._internal.resolved_copy.spark_frame.count()

    def __iter__(self) -> Iterator[Any]:
        return iter(self.columns)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Union["DataFrame", "Series"]:
        if all((isinstance(inp, DataFrame) for inp in inputs)) and any((not same_anchor(inp, inputs[0]) for inp in inputs)):
            assert len(inputs) == 2
            this = inputs[0]
            that = inputs[1]
            if this._internal.column_labels_level != that._internal.column_labels_level:
                raise ValueError('cannot join with no overlapping index names')
            def apply_op(kdf: "DataFrame", this_column_labels: List[Tuple[Any, ...]], that_column_labels: List[Tuple[Any, ...]]) -> Iterator[Tuple["Series", Tuple[Any, ...]]]:
                for this_label, that_label in zip(this_column_labels, that_column_labels):
                    yield (getattr(kdf._kser_for(this_label), ufunc.__name__)(kdf._kser_for(that_label), **kwargs).rename(this_label), this_label)
            return align_diff_frames(apply_op, this, that, fillna=True, how='full')
        else:
            applied = []
            this = inputs[0]
            assert all((inp is this for inp in inputs if isinstance(inp, DataFrame)))
            for label in this._internal.column_labels:
                arguments = []
                for inp in inputs:
                    arguments.append(inp[label] if isinstance(inp, DataFrame) else inp)
                applied.append(ufunc(*arguments, **kwargs).rename(label))
            internal = this._internal.with_new_columns(applied)
            return DataFrame(internal)

    if sys.version_info >= (3, 7):
        def __class_getitem__(cls, params: Any) -> Any:
            return _create_tuple_for_frame_type(params)
    elif (3, 5) <= sys.version_info < (3, 7):
        is_dataframe = None

# End of annotated Python code.
