from functools import partial
from typing import Any, List, Optional, Tuple, Union, Iterator, Type
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
from databricks.koalas.utils import (
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    same_anchor,
    scol_for,
    verify_temp_column_name,
    validate_bool_kwarg,
    ERROR_MESSAGE_CANNOT_COMBINE,
)
from databricks.koalas.internal import InternalFrame, DEFAULT_SERIES_NAME, SPARK_DEFAULT_INDEX_NAME, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.typedef import Scalar


class Index(IndexOpsMixin):
    """
    Koalas Index that corresponds to pandas Index logically. This might hold Spark Column
    internally.
    
    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : dtype, default None
        If dtype is None, we find the dtype that best fits the data.
        If an actual dtype is provided, we coerce to that dtype if it's safe.
        Otherwise, an error will be raised.
    copy : bool
        Make a copy of input ndarray.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.
    
    See Also
    --------
    MultiIndex : A multi-level, or hierarchical, Index.
    DatetimeIndex : Index of datetime64 data.
    Int64Index : A special case of :class:`Index` with purely integer labels.
    Float64Index : A special case of :class:`Index` with purely float labels.
    
    Examples
    --------
    >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, 3]).index
    Int64Index([1, 2, 3], dtype='int64')
    
    >>> ks.DataFrame({'a': [1, 2, 3]}, index=list('abc')).index
    Index(['a', 'b', 'c'], dtype='object')
    
    >>> ks.Index([1, 2, 3])
    Int64Index([1, 2, 3], dtype='int64')
    
    >>> ks.Index(list('abc'))
    Index(['a', 'b', 'c'], dtype='object')
    
    From a Series:
    
    >>> s = ks.Series([1, 2, 3], index=[10, 20, 30])
    >>> ks.Index(s)
    Int64Index([1, 2, 3], dtype='int64')
    
    From an Index:
    
    >>> idx = ks.Index([1, 2, 3])
    >>> ks.Index(idx)
    Int64Index([1, 2, 3], dtype='int64')
    """

    def __new__(
        cls: Type["Index"],
        data: Any = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Any = None,
        tupleize_cols: bool = True,
        **kwargs: Any,
    ) -> "Index":
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
                data_dtypes=[],
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
        return ks.from_pandas(pd.Index(data=data, dtype=dtype, copy=copy, name=name, tupleize_cols=tupleize_cols, **kwargs))

    @staticmethod
    def _new_instance(anchor: DataFrame) -> "Index":
        from databricks.koalas.indexes.category import CategoricalIndex
        from databricks.koalas.indexes.datetimes import DatetimeIndex
        from databricks.koalas.indexes.multi import MultiIndex
        from databricks.koalas.indexes.numeric import Float64Index, Int64Index
        if anchor._internal.index_level > 1:
            instance: "Index" = object.__new__(MultiIndex)
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
            column_label_names=None,
        )

    @property
    def _column_label(self) -> Any:
        return self._kdf._internal.index_names[0]

    def _with_new_scol(self, scol: spark.Column, *, dtype: Optional[Any] = None) -> "Index":
        """
        Copy Koalas Index with the new Spark Column.
    
        :param scol: the new Spark Column
        :return: the copied Index
        """
        internal = self._internal.copy(
            index_spark_columns=[scol.alias(SPARK_DEFAULT_INDEX_NAME)],
            index_dtypes=[dtype],
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return DataFrame(internal).index

    spark = CachedAccessor("spark", SparkIndexMethods)

    def _summary(self, name: Optional[str] = None) -> str:
        """
        Return a summarized representation.
    
        Parameters
        ----------
        name : str
            name to use in the summary representation
    
        Returns
        -------
        String with a summarized representation of the index
        """
        head, tail, total_count = tuple(
            self._internal.spark_frame.select(
                F.first(self.spark.column), F.last(self.spark.column), F.count(F.expr("*"))
            ).toPandas().iloc[0]
        )
        if total_count > 0:
            index_summary = ", %s to %s" % (pprint_thing(head), pprint_thing(tail))
        else:
            index_summary = ""
        if name is None:
            name = type(self).__name__
        return "%s: %s entries%s" % (name, total_count, index_summary)

    @property
    def size(self) -> int:
        """
        Return an int representing the number of elements in this object.
        """
        return len(self)

    @property
    def shape(self) -> Tuple[int]:
        """
        Return a tuple of the shape of the underlying data.
        """
        return (len(self._kdf),)

    def identical(self, other: Any) -> bool:
        """
        Similar to equals, but check that other comparable attributes are
        also equal.
        """
        from databricks.koalas.indexes.multi import MultiIndex
        self_name = self.names if isinstance(self, MultiIndex) else self.name
        other_name = other.names if isinstance(other, MultiIndex) else other.name
        return self_name == other_name and self.equals(other)

    def equals(self, other: Any) -> bool:
        """
        Determine if two Index objects contain the same elements.
        """
        if same_anchor(self, other):
            return True
        elif type(self) == type(other):
            if get_option("compute.ops_on_diff_frames"):
                with option_context("compute.default_index_type", "distributed-sequence"):
                    return (self.to_series("self").reset_index(drop=True) == other.to_series("other").reset_index(drop=True)).all()
            else:
                raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)
        else:
            return False

    def transpose(self) -> "Index":
        """
        Return the transpose, For index, It will be index itself.
        """
        return self

    T = property(transpose)

    def _to_internal_pandas(self) -> pd.Index:
        """
        Return a pandas Index directly from _internal to avoid overhead of copy.
    
        This method is for internal use only.
        """
        return self._kdf._internal.to_pandas_frame.index

    def to_pandas(self) -> pd.Index:
        """
        Return a pandas Index.
        """
        return self._to_internal_pandas().copy()

    def toPandas(self) -> pd.Index:
        warnings.warn("Index.toPandas is deprecated as of Index.to_pandas. Please use the API instead.", FutureWarning)
        return self.to_pandas()
    toPandas.__doc__ = to_pandas.__doc__

    def to_numpy(self, dtype: Optional[Any] = None, copy: bool = False) -> np.ndarray:
        """
        A NumPy ndarray representing the values in this Index or MultiIndex.
        """
        result = np.asarray(self._to_internal_pandas()._values, dtype=dtype)
        if copy:
            result = result.copy()
        return result

    @property
    def values(self) -> np.ndarray:
        """
        Return an array representing the data in the Index.
        """
        warnings.warn("We recommend using `{}.to_numpy()` instead.".format(type(self).__name__))
        return self.to_numpy()

    @property
    def asi8(self) -> Optional[np.ndarray]:
        """
        Integer representation of the values.
        """
        warnings.warn("We recommend using `{}.to_numpy()` instead.".format(type(self).__name__))
        if isinstance(self.spark.data_type, IntegralType):
            return self.to_numpy()
        elif isinstance(self.spark.data_type, TimestampType):
            return np.array(list(map(lambda x: x.astype(np.int64), self.to_numpy())))
        else:
            return None

    @property
    def spark_type(self) -> DataType:
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        warnings.warn("Index.spark_type is deprecated as of Index.spark.data_type. Please use the API instead.", FutureWarning)
        return self.spark.data_type

    @property
    def has_duplicates(self) -> bool:
        """
        If index has duplicates, return True, otherwise False.
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        scol = scol_for(sdf, sdf.columns[0])
        return sdf.select(F.count(scol) != F.countDistinct(scol)).first()[0]

    @property
    def is_unique(self) -> bool:
        """
        Return if the index has unique values.
        """
        return not self.has_duplicates

    @property
    def name(self) -> Any:
        """Return name of the Index."""
        return self.names[0]

    @name.setter
    def name(self, name: Any) -> None:
        self.names = [name]

    @property
    def names(self) -> List[Any]:
        """Return names of the Index."""
        return [name if name is None or len(name) > 1 else name[0] for name in self._internal.index_names]

    @names.setter
    def names(self, names: Any) -> None:
        if not is_list_like(names):
            raise ValueError("Names must be a list-like")
        if self._internal.index_level != len(names):
            raise ValueError("Length of new names must be {}, got {}".format(self._internal.index_level, len(names)))
        if self._internal.index_level == 1:
            self.rename(names[0], inplace=True)
        else:
            self.rename(names, inplace=True)

    @property
    def nlevels(self) -> int:
        """
        Number of levels in Index & MultiIndex.
        """
        return self._internal.index_level

    def rename(self, name: Any, inplace: bool = False) -> Optional["Index"]:
        """
        Alter Index or MultiIndex name.
        """
        names = self._verify_for_rename(name)
        internal = self._kdf._internal.copy(index_names=names)
        if inplace:
            self._kdf._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal).index

    def _verify_for_rename(self, name: Any) -> List[Any]:
        if is_hashable(name):
            if is_name_like_tuple(name):
                return [name]
            elif is_name_like_value(name):
                return [(name,)]
        raise TypeError("Index.name must be a hashable type")

    def fillna(self, value: Any) -> "Index":
        """
        Fill NA/NaN values with the specified value.
        """
        if not isinstance(value, (float, int, str, bool)):
            raise TypeError("Unsupported type %s" % type(value).__name__)
        sdf = self._internal.spark_frame.fillna(value)
        result = DataFrame(self._kdf._internal.with_new_sdf(sdf)).index
        return result

    def drop_duplicates(self) -> "Index":
        """
        Return Index with duplicate values removed.
        """
        sdf = self._internal.spark_frame.select(self._internal.index_spark_columns).drop_duplicates()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def to_series(self, name: Optional[Any] = None) -> Series:
        """
        Create a Series with both index and values equal to the index keys.
        """
        if not is_hashable(name):
            raise TypeError("Series.name must be a hashable type")
        scol = self.spark.column
        if name is not None:
            scol = scol.alias(name_like_string(name))
        elif self._internal.index_level == 1:
            name = self.name
        column_labels = [name if is_name_like_tuple(name) else (name,)]
        internal = self._internal.copy(column_labels=column_labels, data_spark_columns=[scol], column_label_names=None)
        return first_series(DataFrame(internal))

    def to_frame(self, index: bool = True, name: Optional[Any] = None) -> DataFrame:
        """
        Create a DataFrame with a column containing the Index.
        """
        if name is None:
            if self._internal.index_names[0] is None:
                name = (DEFAULT_SERIES_NAME,)
            else:
                name = self._internal.index_names[0]
        elif not is_name_like_tuple(name):
            if is_name_like_value(name):
                name = (name,)
            else:
                raise TypeError("unhashable type: '{}'".format(type(name).__name__))
        return self._to_frame(index=index, names=[name])

    def _to_frame(self, index: bool, names: List[Tuple[Any, ...]]) -> DataFrame:
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
            data_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal)

    def is_boolean(self) -> bool:
        """
        Return if the current index type is a boolean type.
        """
        return is_bool_dtype(self.dtype)

    def is_categorical(self) -> bool:
        """
        Return if the current index type is a categorical type.
        """
        return is_categorical_dtype(self.dtype)

    def is_floating(self) -> bool:
        """
        Return if the current index type is a floating type.
        """
        return is_float_dtype(self.dtype)

    def is_integer(self) -> bool:
        """
        Return if the current index type is an integer type.
        """
        return is_integer_dtype(self.dtype)

    def is_interval(self) -> bool:
        """
        Return if the current index type is an interval type.
        """
        return is_interval_dtype(self.dtype)

    def is_numeric(self) -> bool:
        """
        Return if the current index type is a numeric type.
        """
        return is_numeric_dtype(self.dtype)

    def is_object(self) -> bool:
        """
        Return if the current index type is an object type.
        """
        return is_object_dtype(self.dtype)

    def is_type_compatible(self, kind: str) -> bool:
        """
        Whether the index type is compatible with the provided type.
        """
        return kind == self.inferred_type

    def dropna(self) -> "Index":
        """
        Return Index or MultiIndex without NA/NaN values
        """
        sdf = self._internal.spark_frame.select(self._internal.index_spark_columns).dropna()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def unique(self, level: Optional[Any] = None) -> "Index":
        """
        Return unique values in the index.
        """
        if level is not None:
            self._validate_index_level(level)
        scols = self._internal.index_spark_columns
        sdf = self._kdf._internal.spark_frame.select(scols).distinct()
        return DataFrame(
            InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
                index_names=self._internal.index_names,
                index_dtypes=self._internal.index_dtypes,
            )
        ).index

    def drop(self, labels: Any) -> "Index":
        """
        Make new Index with passed list of labels deleted.
        """
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame[~internal.index_spark_columns[0].isin(labels)]
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return DataFrame(internal).index

    def _validate_index_level(self, level: Any) -> None:
        """
        Validate index level.
        """
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError("Too many levels: Index has only 1 level, %d is not a valid level number" % (level,))
            elif level > 0:
                raise IndexError("Too many levels: Index has only 1 level, not %d" % (level + 1))
        elif level != self.name:
            raise KeyError("Requested level ({}) does not match index name ({})".format(level, self.name))

    def get_level_values(self, level: Any) -> "Index":
        """
        Return Index if a valid level is given.
        """
        self._validate_index_level(level)
        return self

    def copy(self, name: Optional[Any] = None, deep: Optional[Any] = None) -> "Index":
        """
        Make a copy of this object.
        """
        result = self._kdf.copy().index
        if name:
            result.name = name
        return result

    def droplevel(self, level: Any) -> "Index":
        """
        Return index with requested level(s) removed.
        """
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
                        raise IndexError("Too many levels: Index has only {} levels, {} is not a valid level number".format(nlevels, n - nlevels))
                if n >= nlevels:
                    raise IndexError("Too many levels: Index has only {} levels, not {}".format(nlevels, n + 1))
            else:
                if n not in names:
                    raise KeyError("Level {} not found".format(n))
                n = names.index(n)
            int_level.add(n)
        if len(level) >= nlevels:
            raise ValueError("Cannot remove {} levels from an index with {} levels: at least one level must be left.".format(len(level), nlevels))
        index_spark_columns, index_names, index_dtypes = zip(
            *[item for i, item in enumerate(zip(self._internal.index_spark_columns, self._internal.index_names, self._internal.index_dtypes)) if i not in int_level]
        )
        internal = self._internal.copy(
            index_spark_columns=list(index_spark_columns),
            index_names=list(index_names),
            index_dtypes=list(index_dtypes),
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return DataFrame(internal).index

    def symmetric_difference(self, other: "Index", result_name: Optional[str] = None, sort: Optional[bool] = None) -> "Index":
        """
        Compute the symmetric difference of two Index objects.
        """
        if type(self) != type(other):
            raise NotImplementedError("Doesn't support symmetric_difference between Index & MultiIndex for now")
        sdf_self = self._kdf._internal.spark_frame.select(self._internal.index_spark_columns)
        sdf_other = other._kdf._internal.spark_frame.select(other._internal.index_spark_columns)
        sdf_symdiff = sdf_self.union(sdf_other).subtract(sdf_self.intersect(sdf_other))
        if sort:
            sdf_symdiff = sdf_symdiff.sort(self._internal.index_spark_column_names)
        internal = InternalFrame(
            spark_frame=sdf_symdiff,
            index_spark_columns=[scol_for(sdf_symdiff, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        result = DataFrame(internal).index
        if result_name:
            result.name = result_name
        return result

    def sort_values(self, ascending: bool = True) -> "Index":
        """
        Return a sorted copy of the index.
        """
        sdf = self._internal.spark_frame
        sdf = sdf.orderBy(self._internal.index_spark_columns, ascending=ascending).select(self._internal.index_spark_columns)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def sort(self, *args: Any, **kwargs: Any) -> None:
        """
        Use sort_values instead.
        """
        raise TypeError("cannot sort an Index object in-place, use sort_values instead")

    def min(self) -> Any:
        """
        Return the minimum value of the Index.
        """
        sdf = self._internal.spark_frame
        min_row = sdf.select(F.min(F.struct(self._internal.index_spark_columns)).alias("min_row")).select("min_row.*").toPandas()
        result = tuple(min_row.iloc[0])
        return result if len(result) > 1 else result[0]

    def max(self) -> Any:
        """
        Return the maximum value of the Index.
        """
        sdf = self._internal.spark_frame
        max_row = sdf.select(F.max(F.struct(self._internal.index_spark_columns)).alias("max_row")).select("max_row.*").toPandas()
        result = tuple(max_row.iloc[0])
        return result if len(result) > 1 else result[0]

    def delete(self, loc: Union[int, List[int]]) -> "Index":
        """
        Make new Index with passed location(-s) deleted.
        """
        length = len(self)

        def is_len_exceeded(index: int) -> bool:
            return index >= length if index >= 0 else abs(index) > length

        if not is_list_like(loc):
            if is_len_exceeded(loc):
                raise IndexError("index {} is out of bounds for axis 0 with size {}".format(loc, length))
            loc = [loc]
        else:
            for index in loc:
                if is_len_exceeded(index):
                    raise IndexError("index {} is out of bounds for axis 0 with size {}".format(index, length))
        loc = [int(item) for item in loc]
        loc = [item if item >= 0 else length + item for item in loc]
        index_value_column_format = "__index_value_{}__"
        sdf = self._internal._sdf
        index_value_column_names = [verify_temp_column_name(sdf, index_value_column_format.format(i)) for i in range(self._internal.index_level)]
        index_value_columns = [index_scol.alias(index_vcol_name) for index_scol, index_vcol_name in zip(self._internal.index_spark_columns, index_value_column_names)]
        sdf = sdf.select(index_value_columns)
        sdf = InternalFrame.attach_default_index(sdf, default_index_type="distributed-sequence")
        sdf = sdf.where(~F.col(SPARK_INDEX_NAME_FORMAT(0)).isin(loc))
        sdf = sdf.select(index_value_column_names)
        index_origin_columns = [F.col(index_vcol_name).alias(index_scol_name) for index_vcol_name, index_scol_name in zip(index_value_column_names, self._internal.index_spark_column_names)]
        sdf = sdf.select(index_origin_columns)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def append(self, other: "Index") -> "Index":
        """
        Append a collection of Index options together.
        """
        from databricks.koalas.indexes.multi import MultiIndex
        if type(self) is not type(other):
            raise NotImplementedError("append() between Index & MultiIndex currently is not supported")
        sdf_self = self._internal.spark_frame.select(self._internal.index_spark_columns)
        sdf_other = other._internal.spark_frame.select(other._internal.index_spark_columns)
        sdf_appended = sdf_self.union(sdf_other)
        if isinstance(self, MultiIndex):
            index_names = self._internal.index_names
        else:
            index_names = None
        internal = InternalFrame(
            spark_frame=sdf_appended,
            index_spark_columns=[scol_for(sdf_appended, col) for col in self._internal.index_spark_column_names],
            index_names=index_names,
        )
        return DataFrame(internal).index

    def argmax(self) -> int:
        """
        Return a maximum argument indexer.
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        sequence_col = verify_temp_column_name(sdf, "__distributed_sequence_column__")
        sdf = InternalFrame.attach_distributed_sequence_column(sdf, column_name=sequence_col)
        return sdf.orderBy(scol_for(sdf, self._internal.data_spark_column_names[0]).desc(), F.col(sequence_col).asc()).select(sequence_col).first()[0]

    def argmin(self) -> int:
        """
        Return a minimum argument indexer.
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        sequence_col = verify_temp_column_name(sdf, "__distributed_sequence_column__")
        sdf = InternalFrame.attach_distributed_sequence_column(sdf, column_name=sequence_col)
        return sdf.orderBy(scol_for(sdf, self._internal.data_spark_column_names[0]).asc(), F.col(sequence_col).asc()).select(sequence_col).first()[0]

    def set_names(self, names: Any, level: Optional[Any] = None, inplace: bool = False) -> Optional["Index"]:
        """
        Set Index or MultiIndex name.
        """
        from databricks.koalas.indexes.multi import MultiIndex
        if isinstance(self, MultiIndex):
            if level is not None:
                self_names = self.names
                self_names[level] = names
                names = self_names
        return self.rename(name=names, inplace=inplace)

    def difference(self, other: Any, sort: Optional[bool] = None) -> "Index":
        """
        Return a new Index with elements from the index that are not in `other`.
        """
        from databricks.koalas.indexes.multi import MultiIndex
        is_index_types_different = isinstance(other, Index) and (not isinstance(self, type(other)))
        if is_index_types_different:
            if isinstance(self, MultiIndex):
                return self.rename([None] * len(self))
            elif isinstance(self, Index):
                return self.rename(None)
        if not isinstance(other, (Index, Series, tuple, list, set, dict)):
            raise TypeError("Input must be Index or array-like")
        if not isinstance(sort, (type(None), bool)):
            raise ValueError("The 'sort' keyword only takes the values of None or True; {} was passed.".format(sort))
        if isinstance(self, MultiIndex) and (not isinstance(other, MultiIndex)):
            is_other_list_of_tuples = isinstance(other, (list, set, dict)) and all([isinstance(item, tuple) for item in other])
            if is_other_list_of_tuples:
                other = MultiIndex.from_tuples(other)
            elif isinstance(other, Series):
                other = Index(other)
            else:
                raise TypeError("other must be a MultiIndex or a list of tuples")
        if not isinstance(other, Index):
            other = Index(other)
        sdf_self = self._internal.spark_frame
        sdf_other = other._internal.spark_frame
        idx_self = self._internal.index_spark_columns
        idx_other = other._internal.index_spark_columns
        sdf_diff = sdf_self.select(idx_self).subtract(sdf_other.select(idx_other))
        internal = InternalFrame(
            spark_frame=sdf_diff,
            index_spark_columns=[scol_for(sdf_diff, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        result = DataFrame(internal).index
        if isinstance(self, type(other)) and isinstance(self, MultiIndex):
            if self.names == other.names:
                result.names = self.names
        elif isinstance(self, type(other)) and (not isinstance(self, MultiIndex)):
            if self.name == other.name:
                result.name = self.name
        return result if sort is None else result.sort_values()

    @property
    def is_all_dates(self) -> bool:
        """
        Return if all data types of the index are datetime.
        """
        return isinstance(self.spark.data_type, TimestampType)

    def repeat(self, repeats: int) -> "Index":
        """
        Repeat elements of a Index/MultiIndex.
        """
        if not isinstance(repeats, int):
            raise ValueError("`repeats` argument must be integer, but got {}".format(type(repeats).__name__))
        elif repeats < 0:
            raise ValueError("negative dimensions are not allowed")
        kdf = DataFrame(self._internal.resolved_copy)
        if repeats == 0:
            return DataFrame(kdf._internal.with_filter(F.lit(False))).index
        else:
            return ks.concat([kdf] * repeats).index

    def asof(self, label: Any) -> Any:
        """
        Return the label from the index, or, if not present, the previous one.
        """
        sdf = self._internal.spark_frame
        if self.is_monotonic_increasing:
            sdf = sdf.where(self.spark.column <= F.lit(label).cast(self.spark.data_type)).select(F.max(self.spark.column))
        elif self.is_monotonic_decreasing:
            sdf = sdf.where(self.spark.column >= F.lit(label).cast(self.spark.data_type)).select(F.min(self.spark.column))
        else:
            raise ValueError("index must be monotonic increasing or decreasing")
        result = sdf.toPandas().iloc[0, 0]
        return result if result is not None else np.nan

    def union(self, other: Any, sort: Optional[bool] = None) -> "Index":
        """
        Form the union of two Index objects.
        """
        from databricks.koalas.indexes.multi import MultiIndex
        sort = True if sort is None else sort
        sort = validate_bool_kwarg(sort, "sort")
        if type(self) is not type(other):
            if isinstance(self, MultiIndex):
                if not isinstance(other, list) or not all([isinstance(item, tuple) for item in other]):
                    raise TypeError("other must be a MultiIndex or a list of tuples")
                other = MultiIndex.from_tuples(other)
            elif isinstance(other, MultiIndex):
                raise NotImplementedError("Union between Index and MultiIndex is not yet supported")
            elif isinstance(other, Series):
                other = other.to_frame()
                other = other.set_index(other.columns[0]).index
            elif isinstance(other, DataFrame):
                raise ValueError("Index data must be 1-dimensional")
            else:
                other = Index(other)
        sdf_self = self._internal.spark_frame.select(self._internal.index_spark_columns)
        sdf_other = other._internal.spark_frame.select(other._internal.index_spark_columns)
        sdf = sdf_self.union(sdf_other.subtract(sdf_self))
        if isinstance(self, MultiIndex):
            sdf = sdf.drop_duplicates()
        if sort:
            sdf = sdf.sort(self._internal.index_spark_column_names)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names],
            index_names=self._internal.index_names,
        )
        return DataFrame(internal).index

    def holds_integer(self) -> bool:
        """
        Whether the type is an integer type.
        """
        return isinstance(self.spark.data_type, IntegralType)

    def intersection(self, other: Any) -> "Index":
        """
        Form the intersection of two Index objects.
        """
        from databricks.koalas.indexes.multi import MultiIndex
        if isinstance(other, DataFrame):
            raise ValueError("Index data must be 1-dimensional")
        elif isinstance(other, MultiIndex):
            return self._kdf.head(0).index.rename(None)
        elif isinstance(other, Index):
            spark_frame_other = other.to_frame().to_spark()
            keep_name = self.name == other.name
        elif isinstance(other, Series):
            spark_frame_other = other.to_frame().to_spark()
            keep_name = True
        elif is_list_like(other):
            other = Index(other)
            if isinstance(other, MultiIndex):
                return other.to_frame().head(0).index
            spark_frame_other = other.to_frame().to_spark()
            keep_name = True
        else:
            raise TypeError("Input must be Index or array-like")
        spark_frame_self = self.to_frame(name=SPARK_DEFAULT_INDEX_NAME).to_spark()
        spark_frame_intersected = spark_frame_self.intersect(spark_frame_other)
        if keep_name:
            index_names = self._internal.index_names
        else:
            index_names = None
        internal = InternalFrame(
            spark_frame=spark_frame_intersected,
            index_spark_columns=[scol_for(spark_frame_intersected, SPARK_DEFAULT_INDEX_NAME)],
            index_names=index_names,
        )
        return DataFrame(internal).index

    def item(self) -> Any:
        """
        Return the first element of the underlying data as a python scalar.
        """
        return self.to_series().item()

    def insert(self, loc: int, item: Any) -> "Index":
        """
        Make new Index inserting new item at location.
        """
        if loc < 0:
            length = len(self)
            loc = loc + length
            loc = 0 if loc < 0 else loc
        index_name = self._internal.index_spark_column_names[0]
        sdf_before = self.to_frame(name=index_name)[:loc].to_spark()
        sdf_middle = Index([item]).to_frame(name=index_name).to_spark()
        sdf_after = self.to_frame(name=index_name)[loc:].to_spark()
        sdf = sdf_before.union(sdf_middle).union(sdf_after)
        internal = self._internal.with_new_sdf(sdf)
        return DataFrame(internal).index

    def view(self) -> "Index":
        """
        this is defined as a copy with the same identity
        """
        return self.copy()

    def to_list(self) -> List[Any]:
        """
        Return a list of the values.
        """
        return self._to_internal_pandas().tolist()

    tolist = to_list

    @property
    def inferred_type(self) -> str:
        """
        Return a string of the type inferred from the values.
        """
        return lib.infer_dtype([self.to_series().head(1).item()])

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeIndex, item):
            property_or_func = getattr(MissingPandasLikeIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def __repr__(self) -> str:
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            return repr(self._to_internal_pandas())
        pindex = self._kdf._get_or_create_repr_pandas_cache(max_display_count).index
        pindex_length = len(pindex)
        repr_string = repr(pindex[:max_display_count])
        if pindex_length > max_display_count:
            footer = "\nShowing only the first {}".format(max_display_count)
            return repr_string + footer
        return repr_string

    def __iter__(self) -> Iterator[Any]:
        return MissingPandasLikeIndex.__iter__(self)

    def __xor__(self, other: Any) -> "Index":
        return self.symmetric_difference(other)

    def __bool__(self) -> bool:
        raise ValueError(
            "The truth value of a {0} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(self.__class__.__name__)
        )