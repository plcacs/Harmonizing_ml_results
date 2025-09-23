from distutils.version import LooseVersion
from functools import partial
from typing import Any, Optional, Tuple, Union, cast, List, Sequence
import warnings
import pandas as pd
from pandas.api.types import is_list_like
from pandas.api.types import is_hashable
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F, Window
from databricks import koalas as ks
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeMultiIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.utils import compare_disallow_null, default_session, is_name_like_tuple, name_like_string, scol_for, verify_temp_column_name
from databricks.koalas.internal import InternalFrame, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.typedef import Scalar

class MultiIndex(Index):
    """
    Koalas MultiIndex that corresponds to pandas MultiIndex logically. This might hold Spark Column
    internally.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : A single-level Index.

    Examples
    --------
    >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]]).index  # doctest: +SKIP
    MultiIndex([(1, 4),
                (2, 5),
                (3, 6)],
               )

    >>> ks.DataFrame({'a': [1, 2, 3]}, index=[list('abc'), list('def')]).index  # doctest: +SKIP
    MultiIndex([('a', 'd'),
                ('b', 'e'),
                ('c', 'f')],
               )
    """

    def __new__(cls, levels: Optional[Sequence] = None, codes: Optional[Sequence] = None, sortorder: Optional[int] = None, names: Optional[Sequence] = None, dtype: Any = None, copy: bool = False, name: Optional[str] = None, verify_integrity: bool = True) -> 'MultiIndex':
        if LooseVersion(pd.__version__) < LooseVersion('0.24'):
            if levels is None or codes is None:
                raise TypeError('Must pass both levels and codes')
            pidx = pd.MultiIndex(levels=levels, labels=codes, sortorder=sortorder, names=names, dtype=dtype, copy=copy, name=name, verify_integrity=verify_integrity)
        else:
            pidx = pd.MultiIndex(levels=levels, codes=codes, sortorder=sortorder, names=names, dtype=dtype, copy=copy, name=name, verify_integrity=verify_integrity)
        return ks.from_pandas(pidx)

    @property
    def _internal(self) -> InternalFrame:
        internal = self._kdf._internal
        scol = F.struct(internal.index_spark_columns)
        return internal.copy(column_labels=[None], data_spark_columns=[scol], data_dtypes=[None], column_label_names=None)

    @property
    def _column_label(self) -> None:
        return None

    def __abs__(self) -> None:
        raise TypeError('TypeError: cannot perform __abs__ with this index type: MultiIndex')

    def _with_new_scol(self, scol: spark.Column, *, dtype: Optional[Any] = None) -> 'MultiIndex':
        raise NotImplementedError('Not supported for type MultiIndex')

    def _align_and_column_op(self, f: Any, *args: Any) -> Any:
        raise NotImplementedError('Not supported for type MultiIndex')

    def any(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError('cannot perform any with this index type: MultiIndex')

    def all(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError('cannot perform all with this index type: MultiIndex')

    @staticmethod
    def from_tuples(tuples: List[Tuple], sortorder: Optional[int] = None, names: Optional[Sequence] = None) -> 'MultiIndex':
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        Examples
        --------

        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> ks.MultiIndex.from_tuples(tuples, names=('number', 'color'))  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        return cast(MultiIndex, ks.from_pandas(pd.MultiIndex.from_tuples(tuples=tuples, sortorder=sortorder, names=names)))

    @staticmethod
    def from_arrays(arrays: Sequence, sortorder: Optional[int] = None, names: Optional[Sequence] = None) -> 'MultiIndex':
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays: list / sequence of array-likes
            Each array-like gives one level’s value for each data point. len(arrays)
            is the number of levels.
        sortorder: int or None
            Level of sortedness (must be lexicographically sorted by that level).
        names: list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index: MultiIndex

        Examples
        --------

        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> ks.MultiIndex.from_arrays(arrays, names=('number', 'color'))  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        return cast(MultiIndex, ks.from_pandas(pd.MultiIndex.from_arrays(arrays=arrays, sortorder=sortorder, names=names)))

    @staticmethod
    def from_product(iterables: Sequence, sortorder: Optional[int] = None, names: Optional[Sequence] = None) -> 'MultiIndex':
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> ks.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])  # doctest: +SKIP
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        return cast(MultiIndex, ks.from_pandas(pd.MultiIndex.from_product(iterables=iterables, sortorder=sortorder, names=names)))

    @staticmethod
    def from_frame(df: DataFrame, names: Optional[Sequence] = None) -> 'MultiIndex':
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = ks.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df  # doctest: +SKIP
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> ks.MultiIndex.from_frame(df)  # doctest: +SKIP
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> ks.MultiIndex.from_frame(df, names=['state', 'observation'])  # doctest: +SKIP
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        if not isinstance(df, DataFrame):
            raise TypeError('Input must be a DataFrame')
        sdf = df.to_spark()
        if names is None:
            names = df._internal.column_labels
        elif not is_list_like(names):
            raise ValueError('Names should be list-like for a MultiIndex')
        else:
            names = [name if is_name_like_tuple(name) else (name,) for name in names]
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in sdf.columns], index_names=names)
        return cast(MultiIndex, DataFrame(internal).index)

    @property
    def name(self) -> str:
        raise PandasNotImplementedError(class_name='pd.MultiIndex', property_name='name')

    @name.setter
    def name(self, name: str) -> None:
        raise PandasNotImplementedError(class_name='pd.MultiIndex', property_name='name')

    def _verify_for_rename(self, name: Any) -> List[Tuple]:
        if is_list_like(name):
            if self._internal.index_level != len(name):
                raise ValueError('Length of new names must be {}, got {}'.format(self._internal.index_level, len(name)))
            if any((not is_hashable(n) for n in name)):
                raise TypeError('MultiIndex.name must be a hashable type')
            return [n if is_name_like_tuple(n) else (n,) for n in name]
        else:
            raise TypeError('Must pass list-like as `names`.')

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1) -> 'MultiIndex':
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        Examples
        --------
        >>> midx = ks.MultiIndex.from_arrays([['a', 'b'], [1, 2]], names = ['word', 'number'])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 1),
                    ('b', 2)],
                   names=['word', 'number'])

        >>> midx.swaplevel(0, 1)  # doctest: +SKIP
        MultiIndex([(1, 'a'),
                    (2, 'b')],
                   names=['number', 'word'])

        >>> midx.swaplevel('number', 'word')  # doctest: +SKIP
        MultiIndex([(1, 'a'),
                    (2, 'b')],
                   names=['number', 'word'])
        """
        for index in (i, j):
            if not isinstance(index, int) and index not in self.names:
                raise KeyError('Level %s not found' % index)
        i = i if isinstance(i, int) else self.names.index(i)
        j = j if isinstance(j, int) else self.names.index(j)
        for index in (i, j):
            if index >= len(self.names) or index < -len(self.names):
                raise IndexError('Too many levels: Index has only %s levels, %s is not a valid level number' % (len(self.names), index))
        index_map = list(zip(self._internal.index_spark_columns, self._internal.index_names, self._internal.index_dtypes))
        (index_map[i], index_map[j]) = (index_map[j], index_map[i])
        (index_spark_columns, index_names, index_dtypes) = zip(*index_map)
        internal = self._internal.copy(index_spark_columns=list(index_spark_columns), index_names=list(index_names), index_dtypes=list(index_dtypes), column_labels=[], data_spark_columns=[], data_dtypes=[])
        return cast(MultiIndex, DataFrame(internal).index)

    @property
    def levshape(self) -> Tuple[int, ...]:
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> midx.levshape
        (3, 3)
        """
        result = self._internal.spark_frame.agg(*(F.countDistinct(c) for c in self._internal.index_spark_columns)).collect()[0]
        return tuple(result)

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type: Any) -> Any:
        return compare_disallow_null

    def _is_monotonic(self, order: str) -> bool:
        if order == 'increasing':
            return self._is_monotonic_increasing().all()
        else:
            return self._is_monotonic_decreasing().all()

    def _is_monotonic_increasing(self) -> Series:
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)
        cond = F.lit(True)
        has_not_null = F.lit(True)
        for scol in self._internal.index_spark_columns[::-1]:
            data_type = self._internal.spark_type_for(scol)
            prev = F.lag(scol, 1).over(window)
            compare = MultiIndex._comparator_for_monotonic_increasing(data_type)
            has_not_null = has_not_null & scol.isNotNull()
            cond = F.when(scol.eqNullSafe(prev), cond).otherwise(compare(scol, prev, spark.Column.__gt__))
        cond = has_not_null & (prev.isNull() | cond)
        cond_name = verify_temp_column_name(self._internal.spark_frame.select(self._internal.index_spark_columns), '__is_monotonic_increasing_cond__')
        sdf = self._internal.spark_frame.select(self._internal.index_spark_columns + [cond.alias(cond_name)])
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names], index_names=self._internal.index_names, index_dtypes=self._internal.index_dtypes)
        return first_series(DataFrame(internal))

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type: Any) -> Any:
        return compare_disallow_null

    def _is_monotonic_decreasing(self) -> Series:
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)
        cond = F.lit(True)
        has_not_null = F.lit(True)
        for scol in self._internal.index_spark_columns[::-1]:
            data_type = self._internal.spark_type_for(scol)
            prev