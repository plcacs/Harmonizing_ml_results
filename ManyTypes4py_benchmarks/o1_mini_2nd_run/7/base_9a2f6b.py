from functools import partial
from typing import Any, List, Optional, Tuple, Union, Type, TypeVar, Iterable
import warnings
import pandas as pd
import numpy as np
from pandas.api.types import (
    is_list_like,
    is_interval_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
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
from databricks.koalas.internal import (
    InternalFrame,
    DEFAULT_SERIES_NAME,
    SPARK_DEFAULT_INDEX_NAME,
    SPARK_INDEX_NAME_FORMAT,
)
from databricks.koalas.typedef import Scalar

Self = TypeVar("Self", bound="Index")

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
        cls: Type[Self],
        data: Optional[Any] = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Optional[Any] = None,
        tupleize_cols: bool = True,
        **kwargs: Any,
    ) -> Self:
        if not is_hashable(name):
            raise TypeError("Index.name must be a hashable type")
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
            return DataFrame(internal).index  # type: ignore
        elif isinstance(data, Index):
            if copy:
                data = data.copy()
            if dtype is not None:
                data = data.astype(dtype)
            if name is not None:
                data = data.rename(name)
            return data  # type: ignore
        return ks.from_pandas(
            pd.Index(
                data=data,
                dtype=dtype,
                copy=copy,
                name=name,
                tupleize_cols=tupleize_cols,
                **kwargs,
            )
        )

    @staticmethod
    def _new_instance(anchor: "DataFrame") -> "Index":
        from databricks.koalas.indexes.category import CategoricalIndex
        from databricks.koalas.indexes.datetimes import DatetimeIndex
        from databricks.koalas.indexes.multi import MultiIndex
        from databricks.koalas.indexes.numeric import Float64Index, Int64Index

        if anchor._internal.index_level > 1:
            instance: "Index" = object.__new__(MultiIndex)
        elif isinstance(anchor._internal.index_dtypes[0], CategoricalDtype):
            instance = object.__new__(CategoricalIndex)
        elif isinstance(
            anchor._internal.spark_type_for(anchor._internal.index_spark_columns[0]),
            IntegralType,
        ):
            instance = object.__new__(Int64Index)
        elif isinstance(
            anchor._internal.spark_type_for(anchor._internal.index_spark_columns[0]),
            FractionalType,
        ):
            instance = object.__new__(Float64Index)
        elif isinstance(
            anchor._internal.spark_type_for(anchor._internal.index_spark_columns[0]),
            TimestampType,
        ):
            instance = object.__new__(DatetimeIndex)
        else:
            instance = object.__new__(Index)
        instance._anchor = anchor
        return instance

    @property
    def _kdf(self) -> "DataFrame":
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
    def _column_label(self) -> str:
        return self._kdf._internal.index_names[0]

    def _with_new_scol(self, scol: Any, *, dtype: Optional[Any] = None) -> "Index":
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
        return DataFrame(internal).index  # type: ignore

    spark: SparkIndexMethods = CachedAccessor("spark", SparkIndexMethods)

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
                F.first(self.spark.column),
                F.last(self.spark.column),
                F.count(F.expr("*")),
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

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df.index.size
        4

        >>> df.set_index('dogs', append=True).index.size
        4
        """
        return len(self)

    @property
    def shape(self) -> Tuple[int]:
        """
        Return a tuple of the shape of the underlying data.

        Examples
        --------
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')
        >>> idx.shape
        (3,)

        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )
        >>> midx.shape
        (3,)
        """
        return (len(self._kdf),)

    def identical(self, other: "Index") -> bool:
        """
        Similar to equals, but check that other comparable attributes are
        also equal.

        Returns
        -------
        bool
            If two Index objects have equal elements and same type True,
            otherwise False.

        Examples
        --------

        >>> from databricks.koalas.config import option_context
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])

        For Index

        >>> idx.identical(idx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.identical(ks.Index(['a', 'b', 'c']))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.identical(ks.Index(['b', 'b', 'a']))
        False
        >>> idx.identical(midx)
        False

        For MultiIndex

        >>> midx.identical(midx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.identical(ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')]))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.identical(ks.MultiIndex.from_tuples([('c', 'z'), ('b', 'y'), ('a', 'x')]))
        False
        >>> midx.identical(idx)
        False
        """
        from databricks.koalas.indexes.multi import MultiIndex

        self_name: Union[Tuple[Any, ...], Any] = (
            self.names if isinstance(self, MultiIndex) else self.name
        )
        other_name: Union[Tuple[Any, ...], Any] = (
            other.names if isinstance(other, MultiIndex) else other.name
        )
        return self_name == other_name and self.equals(other)

    def equals(self, other: Any) -> bool:
        """
        Determine if two Index objects contain the same elements.

        Returns
        -------
        bool
            True if "other" is an Index and it has the same elements as calling
            index; False otherwise.

        Examples
        --------

        >>> from databricks.koalas.config import option_context
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx.name = "name"
        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx.names = ("nameA", "nameB")

        For Index

        >>> idx.equals(idx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.equals(ks.Index(['a', 'b', 'c']))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.equals(ks.Index(['b', 'b', 'a']))
        False
        >>> idx.equals(midx)
        False

        For MultiIndex

        >>> midx.equals(midx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.equals(ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')]))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.equals(ks.MultiIndex.from_tuples([('c', 'z'), ('b', 'y'), ('a', 'x')]))
        False
        >>> midx.equals(idx)
        False
        """
        if same_anchor(self, other):
            return True
        elif type(self) == type(other):
            if get_option("compute.ops_on_diff_frames"):
                with option_context("compute.default_index_type", "distributed-sequence"):
                    return (
                        self.to_series("self")
                        .reset_index(drop=True)
                        == other.to_series("other")
                        .reset_index(drop=True)
                    ).all()
            else:
                raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)
        else:
            return False

    def transpose(self: Self) -> Self:
        """
        Return the transpose, For index, It will be index itself.

        Examples
        --------
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')

        >>> idx.transpose()
        Index(['a', 'b', 'c'], dtype='object')

        For MultiIndex

        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> midx.transpose()  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )
        """
        return self

    T: "Index" = property(transpose)

    def _to_internal_pandas(self) -> pd.Index:
        """
        Return a pandas Index directly from _internal to avoid overhead of copy.

        This method is for internal use only.
        """
        return self._kdf._internal.to_pandas_frame.index

    def to_pandas(self) -> pd.Index:
        """
        Return a pandas Index.

        .. note:: This method should only be used if the resulting pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df['dogs'].index.to_pandas()
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
        return self._to_internal_pandas().copy()

    def toPandas(self) -> pd.Index:
        warnings.warn(
            "Index.toPandas is deprecated as of Index.to_pandas. Please use the API instead.",
            FutureWarning,
        )
        return self.to_pandas()

    toPandas.__doc__ = to_pandas.__doc__

    def to_numpy(
        self, dtype: Optional[Union[str, np.dtype]] = None, copy: bool = False
    ) -> np.ndarray:
        """
        A NumPy ndarray representing the values in this Index or MultiIndex.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> ks.Series([1, 2, 3, 4]).index.to_numpy()
        array([0, 1, 2, 3])
        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]]).index.to_numpy()
        array([(1, 4), (2, 5), (3, 6)], dtype=object)
        """
        result = np.asarray(self._to_internal_pandas()._values, dtype=dtype)
        if copy:
            result = result.copy()
        return result

    @property
    def values(self) -> np.ndarray:
        """
        Return an array representing the data in the Index.

        .. warning:: We recommend using `Index.to_numpy()` instead.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> ks.Series([1, 2, 3, 4]).index.values
        array([0, 1, 2, 3])
        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]]).index.values
        array([(1, 4), (2, 5), (3, 6)], dtype=object)
        """
        warnings.warn(
            "We recommend using `{}.to_numpy()` instead.".format(type(self).__name__),
            FutureWarning,
        )
        return self.to_numpy()

    @property
    def asi8(self) -> Optional[np.ndarray]:
        """
        Integer representation of the values.

        .. warning:: We recommend using `Index.to_numpy()` instead.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray
            An ndarray with int64 dtype.

        Examples
        --------
        >>> ks.Index([1, 2, 3]).asi8
        array([1, 2, 3])

        Returns None for non-int64 dtype

        >>> ks.Index(['a', 'b', 'c']).asi8 is None
        True
        """
        warnings.warn(
            "We recommend using `{}.to_numpy()` instead.".format(type(self).__name__),
            FutureWarning,
        )
        if isinstance(self.spark.data_type, IntegralType):
            return self.to_numpy()
        elif isinstance(self.spark.data_type, TimestampType):
            return np.array(
                list(map(lambda x: x.astype(np.int64), self.to_numpy()))
            )
        else:
            return None

    @property
    def spark_type(self) -> DataType:
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        warnings.warn(
            "Index.spark_type is deprecated as of Index.spark.data_type. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.data_type

    @property
    def has_duplicates(self) -> bool:
        """
        If index has duplicates, return True, otherwise False.

        Examples
        --------
        >>> idx = ks.Index([1, 5, 7, 7])
        >>> idx.has_duplicates
        True

        >>> idx = ks.Index([1, 5, 7])
        >>> idx.has_duplicates
        False

        >>> idx = ks.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"])
        >>> idx.has_duplicates
        True

        >>> idx = ks.Index(["Orange", "Apple",
        ...                 "Watermelon"])
        >>> idx.has_duplicates
        False
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        scol = scol_for(sdf, sdf.columns[0])
        return sdf.select(F.count(scol) != F.countDistinct(scol)).first()[0]

    @property
    def is_unique(self) -> bool:
        """
        Return if the index has unique values.

        Examples
        --------
        >>> idx = ks.Index([1, 5, 7, 7])
        >>> idx.is_unique
        False

        >>> idx = ks.Index([1, 5, 7])
        >>> idx.is_unique
        True

        >>> idx = ks.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"])
        >>> idx.is_unique
        False

        >>> idx = ks.Index(["Orange", "Apple",
        ...                 "Watermelon"])
        >>> idx.is_unique
        True
        """
        return not self.has_duplicates

    @property
    def name(self) -> Optional[Any]:
        """Return name of the Index."""
        return self.names[0]

    @name.setter
    def name(self, name: Optional[Any]) -> None:
        self.names = [name]

    @property
    def names(self) -> List[Optional[Any]]:
        """Return names of the Index."""
        return [
            name if name is None or len(name) > 1 else name[0]
            for name in self._internal.index_names
        ]

    @names.setter
    def names(self, names: Iterable[Optional[Any]]) -> None:
        if not is_list_like(names):
            raise ValueError("Names must be a list-like")
        if self._internal.index_level != len(names):
            raise ValueError(
                "Length of new names must be {}, got {}".format(
                    self._internal.index_level, len(names)
                )
            )
        if self._internal.index_level == 1:
            self.rename(names[0], inplace=True)
        else:
            self.rename(list(names), inplace=True)

    @property
    def nlevels(self) -> int:
        """
        Number of levels in Index & MultiIndex.

        Examples
        --------
        >>> kdf = ks.DataFrame({"a": [1, 2, 3]}, index=pd.Index(['a', 'b', 'c'], name="idx"))
        >>> kdf.index.nlevels
        1

        >>> kdf = ks.DataFrame({'a': [1, 2, 3]}, index=[list('abc'), list('def')])
        >>> kdf.index.nlevels
        2
        """
        return self._internal.index_level

    def rename(
        self,
        name: Optional[Union[Any, Iterable[Any]]],
        inplace: bool = False,
    ) -> Optional["Index"]:
        """
        Alter Index or MultiIndex name.
        Able to set new names without level. Defaults to returning new index.

        Parameters
        ----------
        name : label or list of labels
            Name(s) to set.
        inplace : boolean, default False
            Modifies the object directly, instead of creating a new Index or MultiIndex.

        Returns
        -------
        Index or MultiIndex
            The same type as the caller or None if inplace is True.

        Examples
        --------
        >>> df = ks.DataFrame({'a': ['A', 'C'], 'b': ['A', 'B']}, columns=['a', 'b'])
        >>> df.index.rename("c")
        Int64Index([0, 1], dtype='int64', name='c')

        >>> df.set_index("a", inplace=True)
        >>> df.index.rename("d")
        Index(['A', 'C'], dtype='object', name='d')

        You can also change the index name in place.

        >>> df.index.rename("e", inplace=True)
        >>> df.index
        Index(['A', 'C'], dtype='object', name='e')

        >>> df  # doctest: +NORMALIZE_WHITESPACE
           b
        e
        A  A
        C  B

        Support for MultiIndex

        >>> kidx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        >>> kidx.names = ['hello', 'koalas']
        >>> kidx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['hello', 'koalas'])

        >>> kidx.rename(['aloha', 'databricks'])  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['aloha', 'databricks'])
        """
        names = self._verify_for_rename(name)
        internal = self._kdf._internal.copy(index_names=names)
        if inplace:
            self._kdf._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal).index

    def _verify_for_rename(
        self, name: Optional[Union[Any, Iterable[Any]]]
    ) -> List[Optional[Any]]:
        if is_hashable(name):
            if is_name_like_tuple(name):
                return [name]
            elif is_name_like_value(name):
                return [(name,)]
        raise TypeError("Index.name must be a hashable type")

    def fillna(self, value: Union[float, int, str, bool]) -> "Index":
        """
        Fill NA/NaN values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill holes (e.g. 0). This value cannot be a list-likes.

        Returns
        -------
        Index :
            filled with value

        Examples
        --------
        >>> ki = ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, None]).index
        >>> ki
        Float64Index([1.0, 2.0, nan], dtype='float64')

        >>> ki.fillna(0)
        Float64Index([1.0, 2.0, 0.0], dtype='float64')
        """
        if not isinstance(value, (float, int, str, bool)):
            raise TypeError("Unsupported type %s" % type(value).__name__)
        sdf = self._internal.spark_frame.fillna(value)
        result = DataFrame(self._kdf._internal.with_new_sdf(sdf)).index
        return result

    def drop_duplicates(self) -> "Index":
        """
        Return Index with duplicate values removed.

        Returns
        -------
        deduplicated : Index

        See Also
        --------
        Series.drop_duplicates : Equivalent method on Series.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.

        Examples
        --------
        Generate an pandas.Index with duplicate values.

        >>> idx = ks.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

        >>> idx.drop_duplicates().sort_values()
        Index(['beetle', 'cow', 'hippo', 'lama'], dtype='object')
        """
        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns
        ).drop_duplicates()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def to_series(self, name: Optional[str] = None) -> Series:
        """
        Create a Series with both index and values equal to the index keys
        useful with map for returning an indexer based on an index.

        Parameters
        ----------
        name : string, optional
            name of resulting Series. If None, defaults to name of original
            index

        Returns
        -------
        Series : dtype will be based on the type of the Index values.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df['dogs'].index.to_series()
        a    a
        b    b
        c    c
        d    d
        dtype: object
        """
        if not is_hashable(name):
            raise TypeError("Series.name must be a hashable type")
        scol = self.spark.column
        if name is not None:
            scol = scol.alias(name_like_string(name))
        elif self._internal.index_level == 1:
            name = self.name
        column_labels: List[Tuple[Any, ...]] = [
            name if is_name_like_tuple(name) else (name,)
        ]
        internal = self._internal.copy(
            column_labels=column_labels,
            data_spark_columns=[scol],
            column_label_names=None,
        )
        return first_series(DataFrame(internal))

    def to_frame(
        self, index: bool = True, name: Optional[Any] = None
    ) -> DataFrame:
        """
        Create a DataFrame with a column containing the Index.

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original Index.
        name : object, default None
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = ks.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()  # doctest: +NORMALIZE_WHITESPACE
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
          animal
        0    Ant
        1   Bear
        2    Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(name='zoo')  # doctest: +NORMALIZE_WHITESPACE
                 zoo
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
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
                raise TypeError(
                    "unhashable type: '{}'".format(type(name).__name__)
                )
        return self._to_frame(index=index, names=[name])

    def _to_frame(
        self, index: bool, names: List[Optional[Any]]
    ) -> DataFrame:
        if index:
            index_spark_columns = self._internal.index_spark_columns
            index_names = self._internal.index_names
            index_dtypes = self._internal.index_dtypes
        else:
            index_spark_columns: List[Any] = []
            index_names: List[Optional[Any]] = []
            index_dtypes: List[Any] = []
        internal = InternalFrame(
            spark_frame=self._internal.spark_frame,
            index_spark_columns=index_spark_columns,
            index_names=index_names,
            index_dtypes=index_dtypes,
            column_labels=names,
            data_spark_columns=self._internal.index_spark_columns if not index else [],
            data_dtypes=self._internal.index_dtypes if not index else [],
        )
        return DataFrame(internal)

    def is_boolean(self) -> bool:
        """
        Return if the current index type is a boolean type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[True]).index.is_boolean()
        True
        """
        return is_bool_dtype(self.dtype)

    def is_categorical(self) -> bool:
        """
        Return if the current index type is a categorical type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_categorical()
        False
        """
        return is_categorical_dtype(self.dtype)

    def is_floating(self) -> bool:
        """
        Return if the current index type is a floating type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_floating()
        False
        """
        return is_float_dtype(self.dtype)

    def is_integer(self) -> bool:
        """
        Return if the current index type is a integer type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_integer()
        True
        """
        return is_integer_dtype(self.dtype)

    def is_interval(self) -> bool:
        """
        Return if the current index type is an interval type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_interval()
        False
        """
        return is_interval_dtype(self.dtype)

    def is_numeric(self) -> bool:
        """
        Return if the current index type is a numeric type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_numeric()
        True
        """
        return is_numeric_dtype(self.dtype)

    def is_object(self) -> bool:
        """
        Return if the current index type is a object type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=["a"]).index.is_object()
        True
        """
        return is_object_dtype(self.dtype)

    def is_type_compatible(self, kind: str) -> bool:
        """
        Whether the index type is compatible with the provided type.

        Examples
        --------
        >>> kidx = ks.Index([1, 2, 3])
        >>> kidx.is_type_compatible('integer')
        True

        >>> kidx = ks.Index([1.0, 2.0, 3.0])
        >>> kidx.is_type_compatible('integer')
        False
        >>> kidx.is_type_compatible('floating')
        True
        """
        return kind == self.inferred_type

    def dropna(self) -> "Index":
        """
        Return Index or MultiIndex without NA/NaN values

        Examples
        --------

        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', None],
        ...                   columns=['max_speed', 'shield'])
        >>> df
               max_speed  shield
        cobra          1       2
        viper          4       5
        NaN            7       8

        >>> df.index.dropna()
        Index(['cobra', 'viper'], dtype='object')

        Also support for MultiIndex

        >>> midx = pd.MultiIndex(
        ...     [['lama', 'cow', 'falcon'], [None, 'weight', 'length']],
        ...     [[0, 1, 1, 1, 1, 1, 2, 2, 2], [0, 1, 1, 0, 1, 2, 1, 1, 2]],
        ... )
        >>> s = ks.Series(
        ...     [45, 200, 1.2, 30, 250, 1.5, 320, 1, None],
        ...     index=midx,
        ... )
        >>> s
        lama    NaN        45.0
        cow     weight    200.0
                weight      1.2
                NaN        30.0
                weight    250.0
                length      1.5
        falcon  weight    320.0
                weight      1.0
                length      NaN
        dtype: float64

        >>> s.index.dropna()  # doctest: +SKIP
        MultiIndex([(   'cow', 'weight'),
                    (   'cow', 'weight'),
                    (   'cow', 'weight'),
                    (   'cow', 'length'),
                    ('falcon', 'weight'),
                    ('falcon', 'weight'),
                    ('falcon', 'length')],
                   )
        """
        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns
        ).dropna()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def unique(self, level: Optional[int] = None) -> "Index":
        """
        Return unique values in the index.

        Be aware the order of unique values might be different than pandas.Index.unique

        Parameters
        ----------
        level : int or str, optional, default is None

        Returns
        -------
        Index without duplicates

        See Also
        --------
        Series.unique
        groupby.SeriesGroupBy.unique

        Examples
        --------
        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 1, 3]).index.unique().sort_values()
        Int64Index([1, 3], dtype='int64')

        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=['d', 'e', 'e']).index.unique().sort_values()
        Index(['d', 'e'], dtype='object')

        MultiIndex

        >>> ks.MultiIndex.from_tuples([("A", "X"), ("A", "Y"), ("A", "X")]).unique()
        ... # doctest: +SKIP
        MultiIndex([('A', 'X'),
                    ('A', 'Y')],
                   )
        """
        if level is not None:
            self._validate_index_level(level)
        scols = self._internal.index_spark_columns
        sdf = self._kdf._internal.spark_frame.select(scols).distinct()
        return DataFrame(
            InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[
                    scol_for(sdf, col) for col in self._internal.index_spark_column_names
                ],
                index_names=self._internal.index_names,
                index_dtypes=self._internal.index_dtypes,
            )
        ).index

    def drop(self, labels: Union[Any, Iterable[Any]]) -> "Index":
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like

        Returns
        -------
        dropped : Index

        Examples
        --------
        >>> index = ks.Index([1, 2, 3])
        >>> index
        Int64Index([1, 2, 3], dtype='int64')

        >>> index.drop([1])
        Int64Index([2, 3], dtype='int64')
        """
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame[~internal.index_spark_columns[0].isin(labels)]
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return DataFrame(internal).index

    def _validate_index_level(self, level: Union[int, str]) -> None:
        """
        Validate index level.
        For single-level Index getting level number is a no-op, but some
        verification must be done like in MultiIndex.
        """
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError(
                    "Too many levels: Index has only 1 level, {} is not a valid level number".format(
                        level
                    )
                )
            elif level > 0:
                raise IndexError(
                    "Too many levels: Index has only 1 level, not {}".format(level + 1)
                )
        elif level != self.name:
            raise KeyError(
                "Requested level ({}) does not match index name ({})".format(
                    level, self.name
                )
            )

    def get_level_values(self, level: Union[int, str]) -> "Index":
        """
        Return Index if a valid level is given.

        Examples:
        --------
        >>> kidx = ks.Index(['a', 'b', 'c'], name='ks')
        >>> kidx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object', name='ks')

        >>> kidx.get_level_values('ks')
        Index(['a', 'b', 'c'], dtype='object', name='ks')
        """
        self._validate_index_level(level)
        return self

    def copy(
        self, name: Optional[str] = None, deep: Optional[Any] = None
    ) -> "Index":
        """
        Make a copy of this object. name sets those attributes on the new object.

        Parameters
        ----------
        name : string, optional
            to set name of index
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', 'sidewinder'],
        ...                   columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8
        >>> df.index
        Index(['cobra', 'viper', 'sidewinder'], dtype='object')

        Copy index

        >>> df.index.copy()
        Index(['cobra', 'viper', 'sidewinder'], dtype='object')

        Copy index with name

        >>> df.index.copy(name='snake')
        Index(['cobra', 'viper', 'sidewinder'], dtype='object', name='snake')
        """
        result = self._kdf.copy().index
        if name:
            result.name = name
        return result

    def droplevel(self, level: Union[int, str, Iterable[Union[int, str]]]) -> "Index":
        """
        Return index with requested level(s) removed.
        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex.

        Parameters
        ----------
        level : int, str, tuple, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        Index or MultiIndex

        Examples
        --------
        >>> midx = ks.DataFrame({'a': ['a', 'b']}, index=[['a', 'x'], ['b', 'y'], [1, 2]]).index
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'b', 1),
                    ('x', 'y', 2)],
                   )
        >>> midx.droplevel([0, 1])  # doctest: +SKIP
        Int64Index([1, 2], dtype='int64')
        >>> midx.droplevel(0)  # doctest: +SKIP
        MultiIndex([('b', 1),
                    ('y', 2)],
                   )
        >>> midx.names = [("a", "b"), "b", "c"]
        >>> midx.droplevel([('a', 'b')])  # doctest: +SKIP
        MultiIndex([('b', 1),
                    ('y', 2)],
                   names=['b', 'c'])
        """
        names = self.names
        nlevels = self.nlevels
        if not is_list_like(level):
            level = [level]
        int_level: set = set()
        for n in level:
            if isinstance(n, int):
                if n < 0:
                    n = n + nlevels
                    if n < 0:
                        raise IndexError(
                            "Too many levels: Index has only {} levels, {} is not a valid level number".format(
                                nlevels, n - nlevels
                            )
                        )
                if n >= nlevels:
                    raise IndexError(
                        "Too many levels: Index has only {} levels, not {}".format(
                            nlevels, n + 1
                        )
                    )
            else:
                if n not in names:
                    raise KeyError("Level {} not found".format(n))
                n = names.index(n)
            int_level.add(n)
        if len(level) >= nlevels:
            raise ValueError(
                "Cannot remove {} levels from an index with {} levels: at least one level must be left.".format(
                    len(level), nlevels
                )
            )
        index_spark_columns, index_names, index_dtypes = zip(
            *[
                item
                for i, item in enumerate(
                    zip(
                        self._internal.index_spark_columns,
                        self._internal.index_names,
                        self._internal.index_dtypes,
                    )
                )
                if i not in int_level
            ]
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

    def symmetric_difference(
        self,
        other: "Index",
        result_name: Optional[str] = None,
        sort: Optional[bool] = None,
    ) -> "Index":
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : str
        sort : True or None, default None
            Whether to sort the resulting index.
            * True : Attempt to sort the result.
            * None : Do not sort the result.

        Returns
        -------
        symmetric_difference : Index

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> s1 = ks.Series([1, 2, 3, 4], index=[1, 2, 3, 4])
        >>> s2 = ks.Series([1, 2, 3, 4], index=[2, 3, 4, 5])

        >>> s1.index.symmetric_difference(s2.index)  # doctest: +SKIP
        Int64Index([5, 1], dtype='int64')

        You can set name of result Index.

        >>> s1.index.symmetric_difference(s2.index, result_name='koalas')  # doctest: +SKIP
        Int64Index([5, 1], dtype='int64', name='koalas')

        You can set sort to `True`, if you want to sort the resulting index.

        >>> s1.index.symmetric_difference(s2.index, sort=True)
        Int64Index([1, 5], dtype='int64')

        You can also use the ``^`` operator:

        >>> s1.index ^ s2.index  # doctest: +SKIP
        Int64Index([5, 1], dtype='int64')
        """
        if type(self) != type(other):
            raise NotImplementedError(
                "Doesn't support symmetric_difference between Index & MultiIndex for now"
            )
        sdf_self = self._kdf._internal.spark_frame.select(
            self._internal.index_spark_columns
        )
        sdf_other = other._kdf._internal.spark_frame.select(
            other._internal.index_spark_columns
        )
        sdf_symdiff = sdf_self.union(sdf_other).subtract(sdf_self.intersect(sdf_other))
        if sort:
            sdf_symdiff = sdf_symdiff.sort(self._internal.index_spark_columns)
        internal = InternalFrame(
            spark_frame=sdf_symdiff,
            index_spark_columns=[
                scol_for(sdf_symdiff, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        result = DataFrame(internal).index
        if result_name:
            result.name = result_name
        return result

    def sort_values(
        self, ascending: bool = True
    ) -> "Index":
        """
        Return a sorted copy of the index.

        .. note:: This method is not supported for pandas when index has NaN value.
                  pandas raises unexpected TypeError, but we support treating NaN
                  as the smallest value.

        Parameters
        ----------
        ascending : bool, default True
            Should the index values be sorted in an ascending order.

        Returns
        -------
        sorted_index : ks.Index or ks.MultiIndex
            Sorted copy of the index.

        See Also
        --------
        Series.sort_values : Sort values of a Series.
        DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = ks.Index([10, 100, 1, 1000])
        >>> idx
        Int64Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Int64Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order.

        >>> idx.sort_values(ascending=False)
        Int64Index([1000, 100, 10, 1], dtype='int64')

        Support for MultiIndex.

        >>> kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('c', 'y', 2), ('b', 'z', 3)])
        >>> kidx  # doctest: +SKIP
        MultiIndex([('a', 'x', 1),
                    ('c', 'y', 2),
                    ('b', 'z', 3)],
                   )

        >>> kidx.sort_values()  # doctest: +SKIP
        MultiIndex([('a', 'x', 1),
                    ('b', 'z', 3),
                    ('c', 'y', 2)],
                   )

        >>> kidx.sort_values(ascending=False)  # doctest: +SKIP
        MultiIndex([('c', 'y', 2),
                    ('b', 'z', 3),
                    ('a', 'x', 1)],
                   )
        """
        sdf = self._internal.spark_frame
        sdf = sdf.orderBy(
            self._internal.index_spark_columns, ascending=ascending
        ).select(self._internal.index_spark_columns)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def sort(self, *args: Any, **kwargs: Any) -> None:
        """
        Use sort_values instead.
        """
        raise TypeError(
            "cannot sort an Index object in-place, use sort_values instead"
        )

    def min(self) -> Any:
        """
        Return the minimum value of the Index.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = ks.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = ks.Index(['c', 'b', 'a'])
        >>> idx.min()
        'a'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2)])
        >>> idx.min()
        ('a', 'x', 1)
        """
        sdf = self._internal.spark_frame
        min_row = sdf.select(
            F.min(F.struct(*self._internal.index_spark_columns)).alias("min_row")
        ).select("min_row.*").toPandas()
        result = tuple(min_row.iloc[0]) if len(min_row.columns) > 1 else min_row.iloc[0, 0]
        return result

    def max(self) -> Any:
        """
        Return the maximum value of the Index.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        Series.max : Return the maximum value in a Series.
        DataFrame.max : Return the maximum values in a DataFrame.

        Examples
        --------
        >>> idx = ks.Index([3, 2, 1])
        >>> idx.max()
        3

        >>> idx = ks.Index(['c', 'b', 'a'])
        >>> idx.max()
        'c'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2)])
        >>> idx.max()
        ('b', 'y', 2)
        """
        sdf = self._internal.spark_frame
        max_row = sdf.select(
            F.max(F.struct(*self._internal.index_spark_columns)).alias("max_row")
        ).select("max_row.*").toPandas()
        result = tuple(max_row.iloc[0]) if len(max_row.columns) > 1 else max_row.iloc[0, 0]
        return result

    def delete(self, loc: Union[int, Iterable[int]]) -> "Index":
        """
        Make new Index with passed location(-s) deleted.

        .. note:: this API can be pretty expensive since it is based on
             a global sequence internally.

        Returns
        -------
        new_index : Index

        Examples
        --------
        >>> kidx = ks.Index([10, 10, 9, 8, 4, 2, 4, 4, 2, 2, 10, 10])
        >>> kidx
        Int64Index([10, 10, 9, 8, 4, 2, 4, 4, 2, 2, 10, 10], dtype='int64')

        >>> kidx.delete(0).sort_values()
        Int64Index([2, 2, 2, 4, 4, 4, 8, 9, 10, 10, 10], dtype='int64')

        >>> kidx.delete([0, 1, 2, 3, 10, 11]).sort_values()
        Int64Index([2, 2, 2, 4, 4, 4], dtype='int64')

        MultiIndex

        >>> kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        >>> kidx  # doctest: +SKIP
        MultiIndex([('a', 'x', 1),
                    ('b', 'y', 2),
                    ('c', 'z', 3)],
                   )

        >>> kidx.delete([0, 2]).sort_values()  # doctest: +SKIP
        MultiIndex([('b', 'y', 2)],
                   )
        """
        length = len(self)

        def is_len_exceeded(index: int) -> bool:
            """Check if the given index is exceeded the length or not"""
            return index >= length if index >= 0 else abs(index) > length

        if not is_list_like(loc):
            if is_len_exceeded(loc):
                raise IndexError(
                    "index {} is out of bounds for axis 0 with size {}".format(
                        loc, length
                    )
                )
            loc = [loc]
        else:
            for index in loc:
                if is_len_exceeded(index):
                    raise IndexError(
                        "index {} is out of bounds for axis 0 with size {}".format(
                            index, length
                        )
                    )
        loc = [int(item) for item in loc]
        loc = [item if item >= 0 else length + item for item in loc]
        index_value_column_format = "__index_value_{}__"
        sdf = self._internal._sdf
        index_value_column_names = [
            verify_temp_column_name(sdf, index_value_column_format.format(i))
            for i in range(self._internal.index_level)
        ]
        index_value_columns = [
            index_scol.alias(index_vcol_name)
            for index_scol, index_vcol_name in zip(
                self._internal.index_spark_columns, index_value_column_names
            )
        ]
        sdf = sdf.select(index_value_columns)
        sdf = InternalFrame.attach_default_index(
            sdf, default_index_type="distributed-sequence"
        )
        sdf = sdf.where(~F.col(SPARK_INDEX_NAME_FORMAT(0)).isin(loc))
        sdf = sdf.select(index_value_column_names)
        index_origin_columns = [
            F.col(index_vcol_name).alias(index_scol_name)
            for index_vcol_name, index_scol_name in zip(
                index_value_column_names, self._internal.index_spark_column_names
            )
        ]
        sdf = sdf.select(index_origin_columns)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )
        return DataFrame(internal).index

    def append(self, other: "Index") -> "Index":
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index

        Returns
        -------
        appended : Index

        Examples
        --------
        >>> kidx = ks.Index([10, 5, 0, 5, 10, 5, 0, 10])
        >>> kidx
        Int64Index([10, 5, 0, 5, 10, 5, 0, 10], dtype='int64')

        >>> kidx.append(kidx)
        Int64Index([10, 5, 0, 5, 10, 5, 0, 10, 10, 5, 0, 5, 10, 5, 0, 10], dtype='int64')

        Support for MiltiIndex

        >>> kidx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        >>> kidx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   )

        >>> kidx.append(kidx)  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('a', 'x'),
                    ('b', 'y')],
                   )
        """
        from databricks.koalas.indexes.multi import MultiIndex

        if type(self) is not type(other):
            raise NotImplementedError(
                "append() between Index & MultiIndex currently is not supported"
            )
        sdf_self = self._kdf._internal.spark_frame.select(
            self._internal.index_spark_columns
        )
        sdf_other = other._kdf._internal.spark_frame.select(
            other._internal.index_spark_columns
        )
        sdf_appended = sdf_self.union(sdf_other)
        if isinstance(self, MultiIndex):
            index_names = self._internal.index_names
        else:
            index_names = None
        internal = InternalFrame(
            spark_frame=sdf_appended,
            index_spark_columns=[
                scol_for(sdf_appended, col)
                for col in self._internal.index_spark_column_names
            ],
            index_names=index_names,
        )
        return DataFrame(internal).index

    def argmax(self) -> int:
        """
        Return a maximum argument indexer.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        maximum argument indexer

        Examples
        --------
        >>> kidx = ks.Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3])
        >>> kidx
        Int64Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3], dtype='int64')

        >>> kidx.argmax()
        4
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        sequence_col = verify_temp_column_name(
            sdf, "__distributed_sequence_column__"
        )
        sdf = InternalFrame.attach_distributed_sequence_column(
            sdf, column_name=sequence_col
        )
        return (
            sdf.orderBy(
                scol_for(sdf, self._internal.data_spark_column_names[0]).desc(),
                F.col(sequence_col).asc(),
            )
            .select(sequence_col)
            .first()[0]
        )

    def argmin(self) -> int:
        """
        Return a minimum argument indexer.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        minimum argument indexer

        Examples
        --------
        >>> kidx = ks.Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3])
        >>> kidx
        Int64Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3], dtype='int64')

        >>> kidx.argmin()
        7
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        sequence_col = verify_temp_column_name(
            sdf, "__distributed_sequence_column__"
        )
        sdf = InternalFrame.attach_distributed_sequence_column(
            sdf, column_name=sequence_col
        )
        return (
            sdf.orderBy(
                scol_for(sdf, self._internal.data_spark_column_names[0]).asc(),
                F.col(sequence_col).asc(),
            )
            .select(sequence_col)
            .first()[0]
        )

    def set_names(
        self,
        names: Union[Any, Iterable[Any]],
        level: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
        inplace: bool = False,
    ) -> Optional["Index"]:
        """
        Set Index or MultiIndex name.
        Able to set new names partially and by level.

        Parameters
        ----------
        names : label or list of label
            Name(s) to set.
        level : int, label or list of int or label, optional
            If the index is a MultiIndex, level(s) to set (None for all
            levels). Otherwise level must be None.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index
            The same type as the caller or None if inplace is True.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = ks.Index([1, 2, 3, 4])
        >>> idx
        Int64Index([1, 2, 3, 4], dtype='int64')

        >>> idx.set_names('quarter')
        Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')

        For MultiIndex

        >>> idx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        >>> idx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   )

        >>> idx.set_names(['kind', 'year'], inplace=True)
        >>> idx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['kind', 'year'])

        >>> idx.set_names('species', level=0)  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['species', 'year'])
        """
        from databricks.koalas.indexes.multi import MultiIndex

        if isinstance(self, MultiIndex):
            if level is not None:
                self_names = self.names
                if isinstance(level, Iterable) and not isinstance(level, (str, tuple)):
                    for lvl, nm in zip(level, names):
                        self_names[lvl] = nm
                else:
                    self_names[level] = names
                names = self_names
        return self.rename(name=names, inplace=inplace)

    def difference(
        self, other: Any, sort: Optional[bool] = None
    ) -> "Index":
        """
        Return a new Index with elements from the index that are not in
        `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : True or None, default None
            Whether to sort the resulting index.
            * True : Attempt to sort the result.
            * None : Do not sort the result.

        Returns
        -------
        difference : Index

        Examples
        --------

        >>> idx1 = ks.Index([2, 1, 3, 4])
        >>> idx2 = ks.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2, sort=True)
        Int64Index([1, 2], dtype='int64')

        MultiIndex

        >>> midx1 = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        >>> midx2 = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'z', 2), ('k', 'z', 3)])
        >>> midx1.difference(midx2)  # doctest: +SKIP
        MultiIndex([('b', 'y', 2),
                    ('c', 'z', 3)],
                   )
        """
        from databricks.koalas.indexes.multi import MultiIndex

        is_index_types_different = isinstance(other, Index) and (
            not isinstance(self, type(other))
        )
        if is_index_types_different:
            if isinstance(self, MultiIndex):
                return self.rename([None] * len(self))
            elif isinstance(self, Index):
                return self.rename(None)
        if not isinstance(other, (Index, Series, tuple, list, set, dict)):
            raise TypeError("Input must be Index or array-like")
        if not isinstance(sort, (type(None), type(True))):
            raise ValueError(
                "The 'sort' keyword only takes the values of None or True; {} was passed.".format(
                    sort
                )
            )
        if isinstance(self, MultiIndex) and (not isinstance(other, MultiIndex)):
            is_other_list_of_tuples = isinstance(other, (list, set, dict)) and all(
                [isinstance(item, tuple) for item in other]
            )
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
        sdf_diff = sdf_self.select(
            self._internal.index_spark_columns
        ).subtract(sdf_other.select(other._internal.index_spark_columns))
        internal = InternalFrame(
            spark_frame=sdf_diff,
            index_spark_columns=[
                scol_for(sdf_diff, col) for col in self._internal.index_spark_column_names
            ],
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
        remember that since Koalas does not support multiple data types in an index,
        so it returns True if any type of data is datetime.

        Examples
        --------
        >>> from datetime import datetime

        >>> idx = ks.Index([datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 2, 3, 0, 0, 0)])
        >>> idx
        DatetimeIndex(['2019-01-01', '2019-02-03'], dtype='datetime64[ns]', freq=None)

        >>> idx.is_all_dates
        True

        >>> idx = ks.Index([datetime(2019, 1, 1, 0, 0, 0), None])
        >>> idx
        DatetimeIndex(['2019-01-01', 'NaT'], dtype='datetime64[ns]', freq=None)

        >>> idx.is_all_dates
        True

        >>> idx = ks.Index([0, 1, 2])
        >>> idx
        Int64Index([0, 1, 2], dtype='int64')

        >>> idx.is_all_dates
        False
        """
        return isinstance(self.spark.data_type, TimestampType)

    def repeat(
        self, repeats: int
    ) -> "Index":
        """
        Repeat elements of a Index/MultiIndex.

        Returns a new Index/MultiIndex where each element of the current Index/MultiIndex
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Index.

        Returns
        -------
        repeated_index : Index/MultiIndex
            Newly created Index/MultiIndex with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.

        Examples
        --------
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')
        >>> idx.repeat(2)
        Index(['a', 'b', 'c', 'a', 'b', 'c'], dtype='object')

        For MultiIndex,

        >>> midx = ks.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c')],
                   )
        >>> midx.repeat(2)  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c'),
                    ('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c')],
                   )
        >>> midx.repeat(0)  # doctest: +SKIP
        MultiIndex([], )
        """
        if not isinstance(repeats, int):
            raise ValueError(
                "`repeats` argument must be integer, but got {}".format(
                    type(repeats).__name__
                )
            )
        elif repeats < 0:
            raise ValueError("negative dimensions are not allowed")
        kdf = DataFrame(self._internal.resolved_copy)
        if repeats == 0:
            return DataFrame(kdf._internal.with_filter(F.lit(False))).index
        else:
            return ks.concat([kdf] * repeats).index

    def argmax(self) -> int:
        """
        Return a maximum argument indexer.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        maximum argument indexer

        Examples
        --------
        >>> kidx = ks.Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3])
        >>> kidx
        Int64Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3], dtype='int64')

        >>> kidx.argmax()
        4
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        sequence_col = verify_temp_column_name(
            sdf, "__distributed_sequence_column__"
        )
        sdf = InternalFrame.attach_distributed_sequence_column(
            sdf, column_name=sequence_col
        )
        return (
            sdf.orderBy(
                scol_for(sdf, self._internal.data_spark_column_names[0]).desc(),
                F.col(sequence_col).asc(),
            )
            .select(sequence_col)
            .first()[0]
        )

    def argmin(self) -> int:
        """
        Return a minimum argument indexer.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        minimum argument indexer

        Examples
        --------
        >>> kidx = ks.Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3])
        >>> kidx
        Int64Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3], dtype='int64')

        >>> kidx.argmin()
        7
        """
        sdf = self._internal.spark_frame.select(self.spark.column)
        sequence_col = verify_temp_column_name(
            sdf, "__distributed_sequence_column__"
        )
        sdf = InternalFrame.attach_distributed_sequence_column(
            sdf, column_name=sequence_col
        )
        return (
            sdf.orderBy(
                scol_for(sdf, self._internal.data_spark_column_names[0]).asc(),
                F.col(sequence_col).asc(),
            )
            .select(sequence_col)
            .first()[0]
        )

    def difference(
        self, other: Any, sort: Optional[bool] = None
    ) -> "Index":
        """
        Return a new Index with elements from the index that are not in
        `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : True or None, default None
            Whether to sort the resulting index.
            * True : Attempt to sort the result.
            * None : Do not sort the result.

        Returns
        -------
        difference : Index

        Examples
        --------

        >>> idx1 = ks.Index([2, 1, 3, 4])
        >>> idx2 = ks.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2, sort=True)
        Int64Index([1, 2], dtype='int64')

        MultiIndex

        >>> midx1 = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        >>> midx2 = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'z', 2), ('k', 'z', 3)])
        >>> midx1.difference(midx2)  # doctest: +SKIP
        MultiIndex([('b', 'y', 2),
                    ('c', 'z', 3)],
                   )
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
        sdf_self = self._internal.spark_frame
        sdf_other = other._internal.spark_frame
        sdf_diff = sdf_self.select(
            self._internal.index_spark_columns
        ).subtract(sdf_other.select(other._internal.index_spark_columns))
        internal = InternalFrame(
            spark_frame=sdf_diff,
            index_spark_columns=[
                scol_for(sdf_diff, col) for col in self._internal.index_spark_column_names
            ],
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
    def inferred_type(self) -> str:
        """
        Return a string of the type inferred from the values.

        Examples
        --------
        >>> from datetime import datetime
        >>> ks.Index([1, 2, 3]).inferred_type
        'integer'

        >>> ks.Index([1.0, 2.0, 3.0]).inferred_type
        'floating'

        >>> ks.Index(['a', 'b', 'c']).inferred_type
        'string'

        >>> ks.Index([True, False, True, False]).inferred_type
        'boolean'
        """
        return lib.infer_dtype([self.to_series().head(1).item()])

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeIndex, item):
            property_or_func = getattr(MissingPandasLikeIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, item)
        )

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

    def __iter__(self) -> Iterable[Any]:
        return MissingPandasLikeIndex.__iter__(self)

    def __xor__(self, other: "Index") -> "Index":
        return self.symmetric_difference(other)

    def __bool__(self) -> bool:
        raise ValueError(
            "The truth value of a {0} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(
                self.__class__.__name__
            )
        )
