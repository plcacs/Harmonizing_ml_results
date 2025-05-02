from functools import partial
from typing import Any, List, Optional, Tuple, Union, Set, Dict, cast
import warnings
import pandas as pd
import numpy as np
from pandas.api.types import is_list_like, is_interval_dtype, is_bool_dtype, is_categorical_dtype, is_integer_dtype, is_float_dtype, is_numeric_dtype, is_object_dtype
from pandas.core.accessor import CachedAccessor
from pandas.io.formats.printing import pprint_thing
from pandas.api.types import CategoricalDtype, is_hashable
from pandas._libs import lib
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, FractionalType, IntegralType, TimestampType
from databricks import koalas as ks
from databricks.koalas.config import get_option, option_context
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.missing.indexes import MissingPandasLikeIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.spark.accessors import SparkIndexMethods
from databricks.koalas.utils import is_name_like_tuple, is_name_like_value, name_like_string, same_anchor, scol_for, verify_temp_column_name, validate_bool_kwarg, ERROR_MESSAGE_CANNOT_COMBINE
from databricks.koalas.internal import InternalFrame, DEFAULT_SERIES_NAME, SPARK_DEFAULT_INDEX_NAME, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.typedef import Scalar

class Index(IndexOpsMixin):
    def __new__(
        cls,
        data: Optional[Union[Series, Index, List, Tuple, np.ndarray, pd.Index]] = None,
        dtype: Optional[Union[str, np.dtype]] = None,
        copy: bool = False,
        name: Optional[Union[str, Tuple[str, ...]] = None,
        tupleize_cols: bool = True,
        **kwargs: Any
    ) -> 'Index':
        if not is_hashable(name):
            raise TypeError('Index.name must be a hashable type')
        if isinstance(data, Series):
            if dtype is not None:
                data = data.astype(dtype)
            if name is not None:
                data = data.rename(name)
            internal = InternalFrame(
                spark_frame=data._internal.spark_frame,
                index_spark_columns=data._internal.data_spark_columns,
                index_names=data._internal.column_labels,
                index_dtypes=data._internal.data_dtypes,
                column_labels=[],
                data_spark_columns=[],
                data_dtypes=[]
            )
            return DataFrame(internal).index
        elif isinstance(data, Index):
            if copy:
                data = data.copy()
            if dtype is not None:
                data = data.astype(dtype)
            if name is not None:
                data = data.rename(name)
            return data
        return ks.from_pandas(pd.Index(
            data=data, dtype=dtype, copy=copy, name=name,
            tupleize_cols=tupleize_cols, **kwargs))

    @staticmethod
    def _new_instance(anchor: DataFrame) -> 'Index':
        from databricks.koalas.indexes.category import CategoricalIndex
        from databricks.koalas.indexes.datetimes import DatetimeIndex
        from databricks.koalas.indexes.multi import MultiIndex
        from databricks.koalas.indexes.numeric import Float64Index, Int64Index
        if anchor._internal.index_level > 1:
            instance = object.__new__(MultiIndex)
        elif isinstance(anchor._internal.index_dtypes[0], CategoricalDtype):
            instance = object.__new__(CategoricalIndex)
        elif isinstance(anchor._internal.spark_type_for(anchor._internal.index_spark_columns[0]), IntegralType):
            instance = object.__new__(Int64Index)
        elif isinstance(anchor._internal.spark_type_for(anchor._internal.index_spark_columns[0]), FractionalType):
            instance = object.__new__(Float64Index)
        elif isinstance(anchor._internal.spark_type_for(anchor._internal.index_spark_columns[0]), TimestampType):
            instance = object.__new__(DatetimeIndex)
        else:
            instance = object.__new__(Index)
        instance._anchor = anchor
        return instance

    @property
    def _kdf(self) -> DataFrame:
        return self._anchor

    @property
    def _internal(self) -> InternalFrame:
        internal = self._kdf._internal
        return internal.copy(
            column_labels=internal.index_names,
            data_spark_columns=internal.index_spark_columns,
            data_dtypes=internal.index_dtypes,
            column_label_names=None
        )

    @property
    def _column_label(self) -> Union[str, Tuple[str, ...]]:
        return self._kdf._internal.index_names[0]

    def _with_new_scol(self, scol: F.Column, *, dtype: Optional[Union[str, np.dtype]] = None) -> 'Index':
        internal = self._internal.copy(
            index_spark_columns=[scol.alias(SPARK_DEFAULT_INDEX_NAME)],
            index_dtypes=[dtype],
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[]
        )
        return DataFrame(internal).index

    spark = CachedAccessor('spark', SparkIndexMethods)

    def _summary(self, name: Optional[str] = None) -> str:
        head, tail, total_count = tuple(
            self._internal.spark_frame.select(
                F.first(self.spark.column),
                F.last(self.spark.column),
                F.count(F.expr('*'))
            .toPandas().iloc[0])
        if total_count > 0:
            index_summary = ', %s to %s' % (pprint_thing(head), pprint_thing(tail))
        else:
            index_summary = ''
        if name is None:
            name = type(self).__name__
        return '%s: %s entries%s' % (name, total_count, index_summary)

    @property
    def size(self) -> int:
        return len(self)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._kdf),)

    def identical(self, other: Any) -> bool:
        from databricks.koalas.indexes.multi import MultiIndex
        self_name = self.names if isinstance(self, MultiIndex) else self.name
        other_name = other.names if isinstance(other, MultiIndex) else other.name
        return self_name == other_name and self.equals(other)

    def equals(self, other: Any) -> bool:
        if same_anchor(self, other):
            return True
        elif type(self) == type(other):
            if get_option('compute.ops_on_diff_frames'):
                with option_context('compute.default_index_type', 'distributed-sequence'):
                    return (self.to_series('self').reset_index(drop=True) == 
                            other.to_series('other').reset_index(drop=True)).all()
            else:
                raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)
        else:
            return False

    def transpose(self) -> 'Index':
        return self

    T = property(transpose)

    def _to_internal_pandas(self) -> pd.Index:
        return self._kdf._internal.to_pandas_frame.index

    def to_pandas(self) -> pd.Index:
        return self._to_internal_pandas().copy()

    def toPandas(self) -> pd.Index:
        warnings.warn('Index.toPandas is deprecated as of Index.to_pandas. Please use the API instead.', FutureWarning)
        return self.to_pandas()

    def to_numpy(self, dtype: Optional[Union[str, np.dtype]] = None, copy: bool = False) -> np.ndarray:
        result = np.asarray(self._to_internal_pandas()._values, dtype=dtype)
        if copy:
            result = result.copy()
        return result

    @property
    def values(self) -> np.ndarray:
        warnings.warn(f'We recommend using `{type(self).__name__}.to_numpy()` instead.')
        return self.to_numpy()

    @property
    def asi8(self) -> Optional[np.ndarray]:
        warnings.warn(f'We recommend using `{type(self).__name__}.to_numpy()` instead.')
        if isinstance(self.spark.data_type, IntegralType):
            return self.to_numpy()
        elif isinstance(self.spark.data_type, TimestampType):
            return np.array(list(map(lambda x: x.astype(np.int64), self.to_numpy())))
        else:
            return None

    @property
    def spark_type(self) -> DataType:
        warnings.warn('Index.spark_type is deprecated as of Index.spark.data_type. Please use the API instead.', FutureWarning)
        return self.spark.data_type

    @property
    def has_duplicates(self) -> bool:
        sdf = self._internal.spark_frame.select(self.spark.column)
        scol = scol_for(sdf, sdf.columns[0])
        return sdf.select(F.count(scol) != F.countDistinct(scol)).first()[0]

    @property
    def is_unique(self) -> bool:
        return not self.has_duplicates

    @property
    def name(self) -> Optional[Union[str, Tuple[str, ...]]]:
        return self.names[0]

    @name.setter
    def name(self, name: Optional[Union[str, Tuple[str, ...]]]) -> None:
        self.names = [name]

    @property
    def names(self) -> List[Optional[Union[str, Tuple[str, ...]]]]:
        return [name if name is None or len(name) > 1 else name[0] 
                for name in self._internal.index_names]

    @names.setter
    def names(self, names: List[Optional[Union[str, Tuple[str, ...]]]]) -> None:
        if not is_list_like(names):
            raise ValueError('Names must be a list-like')
        if self._internal.index_level != len(names):
            raise ValueError(f'Length of new names must be {self._internal.index_level}, got {len(names)}')
        if self._internal.index_level == 1:
            self.rename(names[0], inplace=True)
        else:
            self.rename(names, inplace=True)

    @property
    def nlevels(self) -> int:
        return self._internal.index_level

    def rename(
        self, 
        name: Union[str, List[str], Tuple[str, ...]], 
        inplace: bool = False
    ) -> Optional['Index']:
        names = self._verify_for_rename(name)
        internal = self._kdf._internal.copy(index_names=names)
        if inplace:
            self._kdf._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal).index

    def _verify_for_rename(self, name: Union[str, List[str], Tuple[str, ...]]) -> List[Optional[Tuple[str, ...]]]:
        if is_hashable(name):
            if is_name_like_tuple(name):
                return [name]
            elif is_name_like_value(name):
                return [(name,)]
        raise TypeError('Index.name must be a hashable type')

    def fillna(self, value: Union[float, int, str, bool]) -> 'Index':
        if not isinstance(value, (float, int, str, bool)):
            raise TypeError(f'Unsupported type {type(value).__name__}')
        sdf = self._internal.spark_frame.fillna(value)
        result = DataFrame(self._kdf._internal.with_new_sdf(sdf)).index
        return result

    def drop_duplicates(self) -> 'Index':
        sdf = self._internal.spark_frame.select(self._internal.index_spark_columns).drop_duplicates()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes
        )
        return DataFrame(internal).index

    def to_series(self, name: Optional[Union[str, Tuple[str, ...]]] = None) -> Series:
        if not is_hashable(name):
            raise TypeError('Series.name must be a hashable type')
        scol = self.spark.column
        if name is not None:
            scol = scol.alias(name_like_string(name))
        elif self._internal.index_level == 1:
            name = self.name
        column_labels = [name if is_name_like_tuple(name) else (name,)]
        internal = self._internal.copy(
            column_labels=column_labels,
            data_spark_columns=[scol],
            column_label_names=None
        )
        return first_series(DataFrame(internal))

    def to_frame(
        self, 
        index: bool = True, 
        name: Optional[Union[str, Tuple[str, ...]]] = None
    ) -> DataFrame:
        if name is None:
            if self._internal.index_names[0] is None:
                name = (DEFAULT_SERIES_NAME,)
            else:
                name = self._internal.index_names[0]
        elif not is_name_like_tuple(name):
            if is_name_like_value(name):
                name = (name,)
            else:
                raise TypeError(f"unhashable type: '{type(name).__name__}'")
        return self._to_frame(index=index, names=[name])

    def _to_frame(
        self, 
        index: bool, 
        names: List[Optional[Tuple[str, ...]]]
    ) -> DataFrame:
        if index:
            index_spark_columns = self._internal.index_spark_columns
            index_names = self._internal.index_names
            index_dtypes = self._internal.index_dtypes
        else:
            index_spark_columns = []
            index_names = []
            index_dtypes = []
        internal = InternalFrame(
            spark_frame=self._internal.spark_frame,
            index_spark_columns=index_spark_columns,
            index_names=index_names,
            index_dtypes=index_dtypes,
            column_labels=names,
            data_spark_columns=self._internal.index_spark_columns,
            data_dtypes=self._internal.index_dtypes
        )
        return DataFrame(internal)

    def is_boolean(self) -> bool:
        return is_bool_dtype(self.dtype)

    def is_categorical(self) -> bool:
        return is_categorical_dtype(self.dtype)

    def is_floating(self) -> bool:
        return is_float_dtype(self.dtype)

    def is_integer(self) -> bool:
        return is_integer_dtype(self.dtype)

    def is_interval(self) -> bool:
        return is_interval_dtype(self.dtype)

    def is_numeric(self) -> bool:
        return is_numeric_dtype(self.dtype)

    def is_object(self) -> bool:
        return is_object_dtype(self.dtype)

    def is_type_compatible(self, kind: str) -> bool:
        return kind == self.inferred_type

    def dropna(self) -> 'Index':
        sdf = self._internal.spark_frame.select(self._internal.index_spark_columns).dropna()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes
        )
        return DataFrame(internal).index

    def unique(self, level: Optional[Union[int, str]] = None) -> 'Index':
        if level is not None:
            self._validate_index_level(level)
        scols = self._internal.index_spark_columns
        sdf = self._kdf._internal.spark_frame.select(scols).distinct()
        return DataFrame(
            InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
                index_names=self._internal.index_names,
                index_dtypes=self._internal.index_dtypes
            )
        ).index

    def drop(self, labels: Union[Any, List[Any]]) -> 'Index':
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame[~internal.index_spark_columns[0].isin(labels)]
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[]
        )
        return DataFrame(internal).index

    def _validate_index_level(self, level: Union[int, str]) -> None:
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError(f'Too many levels: Index has only 1 level, {level} is not a valid level number')
            elif level > 0:
                raise IndexError(f'Too many levels: Index has only 1 level, not {level + 1}')
        elif level != self.name:
            raise KeyError(f'Requested level ({level}) does not match index name ({self.name})')

    def get_level_values(self, level: Union[int, str]) -> 'Index':
        self._validate_index_level(level)
        return self

    def copy(
        self, 
        name: Optional[Union[str, Tuple[str, ...]]] = None, 
        deep: Optional[bool] = None
    ) -> 'Index':
        result = self._kdf.copy().index
        if name:
            result.name = name
        return result

    def droplevel(self, level: Union[int, str, List[Union[int, str]]]) -> 'Index':
        names = self.names
        nlevels = self.nlevels
        if not is_list_like(level):
            level = [level]
        int_level = set()
        for n in level:
            if isinstance(n, int):
                if n < 0:
                    n = n + nlevels
                    if n < 0:
                        raise IndexError(f'Too