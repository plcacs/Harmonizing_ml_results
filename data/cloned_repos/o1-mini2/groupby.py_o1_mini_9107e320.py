#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A wrapper for GroupedData to behave similar to pandas GroupBy.
"""

from abc import ABCMeta, abstractmethod
import sys
import inspect
from collections import OrderedDict, namedtuple
from collections.abc import Callable
from distutils.version import LooseVersion
from functools import partial
from itertools import product
from typing import (
    Any,
    Callable as TypingCallable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import pandas as pd
from pandas.api.types import is_hashable, is_list_like

from pyspark.sql import Window, functions as F
from pyspark.sql.types import (
    FloatType,
    DoubleType,
    NumericType,
    StructField,
    StructType,
    StringType,
)
from pyspark.sql.functions import PandasUDFType, pandas_udf, Column

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.typedef import (
    DataFrameType,
    ScalarType,
    SeriesType,
    infer_return_type,
)
from databricks.koalas.frame import DataFrame
from databricks.koalas.internal import (
    InternalFrame,
    HIDDEN_COLUMNS,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_INDEX_NAME_FORMAT,
    SPARK_DEFAULT_SERIES_NAME,
)
from databricks.koalas.missing.groupby import (
    MissingPandasLikeDataFrameGroupBy,
    MissingPandasLikeSeriesGroupBy,
)
from databricks.koalas.series import Series, first_series
from databricks.koalas.config import get_option
from databricks.koalas.utils import (
    align_diff_frames,
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    same_anchor,
    scol_for,
    verify_temp_column_name,
)
from databricks.koalas.spark.utils import (
    as_nullable_spark_type,
    force_decimal_precision_scale,
)
from databricks.koalas.window import RollingGroupby, ExpandingGroupby
from databricks.koalas.exceptions import DataError
from databricks.koalas.spark import functions as SF

# to keep it the same as pandas
NamedAgg = namedtuple("NamedAgg", ["column", "aggfunc"])


class GroupBy(object, metaclass=ABCMeta):
    """
    :ivar _kdf: The parent dataframe that is used to perform the groupby
    :type _kdf: DataFrame
    :ivar _groupkeys: The list of keys that will be used to perform the grouping
    :type _groupkeys: List[Series]
    """

    def __init__(
        self,
        kdf: DataFrame,
        groupkeys: List[Series],
        as_index: bool,
        dropna: bool,
        column_labels_to_exlcude: Set[Tuple],
        agg_columns_selected: bool,
        agg_columns: List[Series],
    ) -> None:
        self._kdf = kdf
        self._groupkeys = groupkeys
        self._as_index = as_index
        self._dropna = dropna
        self._column_labels_to_exlcude = column_labels_to_exlcude
        self._agg_columns_selected = agg_columns_selected
        self._agg_columns = agg_columns

    @property
    def _groupkeys_scols(self) -> List[Column]:
        return [s.spark.column for s in self._groupkeys]

    @property
    def _agg_columns_scols(self) -> List[Column]:
        return [s.spark.column for s in self._agg_columns]

    @abstractmethod
    def _apply_series_op(
        self,
        op: TypingCallable[[SeriesGroupBy], Union[Series, DataFrame]],
        should_resolve: bool = False,
        numeric_only: bool = False,
    ) -> Union[Series, DataFrame]:
        pass

    # TODO: Series support is not implemented yet.
    # TODO: not all arguments are implemented comparing to pandas' for now.
    def aggregate(
        self,
        func_or_funcs: Optional[Union[Dict[Union[str, Tuple[Any, ...]], Union[str, List[str]]], str, List[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame:
        """Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func_or_funcs : dict, str or list
             a dict mapping from column name (string) to
             aggregate functions (string or list of strings).

        Returns
        -------
        Series or DataFrame

            The return can be:

            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

            Return Series or DataFrame.

        Notes
        -----
        `agg` is an alias for `aggregate`. Use the alias.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 2],
        ...                    'B': [1, 2, 3, 4],
        ...                    'C': [0.362, 0.227, 1.267, -0.562]},
        ...                   columns=['A', 'B', 'C'])

        >>> df
           A  B      C
        0  1  1  0.362
        1  1  2  0.227
        2  2  3  1.267
        3  2  4 -0.562

        Different aggregations per column

        >>> aggregated = df.groupby('A').agg({'B': 'min', 'C': 'sum'})
        >>> aggregated[['B', 'C']].sort_index()  # doctest: +NORMALIZE_WHITESPACE
           B      C
        A
        1  1  0.589
        2  3  0.705

        >>> aggregated = df.groupby('A').agg({'B': ['min', 'max']})
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B
           min  max
        A
        1    1    2
        2    3    4

        >>> aggregated = df.groupby('A').agg('min')
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B      C
        A
        1    1  0.227
        2    3 -0.562

        >>> aggregated = df.groupby('A').agg(['min', 'max'])
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B           C
           min  max    min    max
        A
        1    1    2  0.227  0.362
        2    3    4 -0.562  1.267

        To control the output names with different aggregations per column, Koalas
        also supports 'named aggregation' or nested renaming in .agg. It can also be
        used when applying multiple aggregation functions to specific columns.

        >>> aggregated = df.groupby('A').agg(b_max=ks.NamedAgg(column='B', aggfunc='max'))
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             b_max
        A
        1        2
        2        4

        >>> aggregated = df.groupby('A').agg(b_max=('B', 'max'), b_min=('B', 'min'))
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             b_max   b_min
        A
        1        2       1
        2        4       3

        >>> aggregated = df.groupby('A').agg(b_max=('B', 'max'), c_min=('C', 'min'))
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             b_max   c_min
        A
        1        2   0.227
        2        4  -0.562
        """
        # I think current implementation of func and arguments in Koalas for aggregate is different
        # than pandas, later once arguments are added, this could be removed.
        if func_or_funcs is None and not kwargs:
            raise ValueError("No aggregation argument or function specified.")

        relabeling = func_or_funcs is None and is_multi_agg_with_relabel(**kwargs)
        if relabeling:
            (
                func_or_funcs,
                columns,
                order,
            ) = normalize_keyword_aggregation(kwargs)

        if not isinstance(func_or_funcs, (str, list)):
            if not isinstance(func_or_funcs, dict) or not all(
                is_name_like_value(key)
                and (
                    isinstance(value, str)
                    or (
                        isinstance(value, list)
                        and all(isinstance(v, str) for v in value)
                    )
                )
                for key, value in func_or_funcs.items()
            ):
                raise ValueError(
                    "aggs must be a dict mapping from column name "
                    "to aggregate functions (string or list of strings)."
                )
        else:
            agg_cols = [col.name for col in self._agg_columns]
            func_or_funcs = OrderedDict([(col, func_or_funcs) for col in agg_cols])

        kdf = DataFrame(
            GroupBy._spark_groupby(self._kdf, func_or_funcs, self._groupkeys)
        )  # type: DataFrame

        if self._dropna:
            kdf = DataFrame(
                kdf._internal.with_new_sdf(
                    kdf._internal.spark_frame.dropna(
                        subset=kdf._internal.index_spark_column_names
                    )
                )
            )

        if not self._as_index:
            should_drop_index = set(
                i for i, gkey in enumerate(self._groupkeys) if gkey._kdf is not self._kdf
            )
            if len(should_drop_index) > 0:
                kdf = kdf.reset_index(level=should_drop_index, drop=True)
            if len(should_drop_index) < len(self._groupkeys):
                kdf = kdf.reset_index()

        if relabeling:
            kdf = kdf[order]
            kdf.columns = columns
        return kdf

    agg = aggregate

    @staticmethod
    def _spark_groupby(
        kdf: DataFrame,
        func: OrderedDict[str, Union[str, List[str]]],
        groupkeys: List[Series],
    ) -> InternalFrame:
        groupkey_names = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_scols = [
            s.spark.column.alias(name)
            for s, name in zip(groupkeys, groupkey_names)
        ]

        multi_aggs = any(isinstance(v, list) for v in func.values())
        reordered = []
        data_columns = []
        column_labels = []
        for key, value in func.items():
            label = key if is_name_like_tuple(key) else (key,)
            if len(label) != kdf._internal.column_labels_level:
                raise TypeError(
                    "The length of the key must be the same as the column label level."
                )
            for aggfunc in [value] if isinstance(value, str) else value:
                column_label = (
                    tuple(list(label) + [aggfunc]) if multi_aggs else label
                )
                column_labels.append(column_label)

                data_col = name_like_string(column_label)
                data_columns.append(data_col)

                col_name = kdf._internal.spark_column_name_for(label)
                if aggfunc == "nunique":
                    reordered.append(
                        F.expr(
                            "count(DISTINCT `{0}`) as `{1}`".format(
                                col_name, data_col
                            )
                        )
                    )

                # Implement "quartiles" aggregate function for ``describe``.
                elif aggfunc == "quartiles":
                    reordered.append(
                        F.expr(
                            "percentile_approx(`{0}`, array(0.25, 0.5, 0.75)) as `{1}`".format(
                                col_name, data_col
                            )
                        )
                    )

                else:
                    reordered.append(
                        F.expr(
                            "{1}(`{0}`) as `{2}`".format(col_name, aggfunc, data_col)
                        )
                    )

        sdf = kdf._internal.spark_frame.select(
            groupkey_scols + kdf._internal.data_spark_columns
        )
        sdf = sdf.groupby(*groupkey_names).agg(*reordered)
        if len(groupkeys) > 0:
            index_spark_column_names = groupkey_names
            index_names = [kser._column_label for kser in groupkeys]
            index_dtypes = [kser.dtype for kser in groupkeys]
        else:
            index_spark_column_names = []
            index_names = []
            index_dtypes = []

        return InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in index_spark_column_names
            ],
            index_names=index_names,
            index_dtypes=index_dtypes,
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
        )

    def count(self) -> Union[DataFrame, Series]:
        """
        Compute count of group, excluding missing values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])
        >>> df.groupby('A').count().sort_index()  # doctest: +NORMALIZE_WHITESPACE
            B  C
        A
        1  2  3
        2  2  2
        """
        return self._reduce_for_stat_function(F.count, only_numeric=False)

    # TODO: We should fix See Also when Series implementation is finished.
    def first(self) -> Union[DataFrame, Series]:
        """
        Compute first of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.first, only_numeric=False)

    def last(self) -> Union[DataFrame, Series]:
        """
        Compute last of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(
            lambda col: F.last(col, ignorenulls=True), only_numeric=False
        )

    def max(self) -> Union[DataFrame, Series]:
        """
        Compute max of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.max, only_numeric=False)

    # TODO: examples should be updated.
    def mean(self) -> Union[DataFrame, Series]:
        """
        Compute mean of groups, excluding missing values.

        Returns
        -------
        koalas.Series or koalas.DataFrame

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby('A').mean().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000
        """

        return self._reduce_for_stat_function(F.mean, only_numeric=True)

    def min(self) -> Union[DataFrame, Series]:
        """
        Compute min of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.min, only_numeric=False)

    # TODO: sync the doc.
    def std(self, ddof: int = 1) -> Union[DataFrame, Series]:
        """
        Compute standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        assert ddof in (0, 1)

        return self._reduce_for_stat_function(
            F.stddev_pop if ddof == 0 else F.stddev_samp, only_numeric=True
        )

    def sum(self) -> Union[DataFrame, Series]:
        """
        Compute sum of group values

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.sum, only_numeric=True)

    # TODO: sync the doc.
    def var(self, ddof: int = 1) -> Union[DataFrame, Series]:
        """
        Compute variance of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        assert ddof in (0, 1)

        return self._reduce_for_stat_function(
            F.var_pop if ddof == 0 else F.var_samp, only_numeric=True
        )

    # TODO: skipna should be implemented.
    def all(self) -> Union[DataFrame, Series]:
        """
        Returns True if all values in the group are truthful, else False.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ...                    'B': [True, True, True, False, False,
        ...                          False, None, True, None, False]},
        ...                   columns=['A', 'B'])
        >>> df
           A      B
        0  1   True
        1  1   True
        2  2   True
        3  2  False
        4  3  False
        5  3  False
        6  4   None
        7  4   True
        8  5   None
        9  5  False

        >>> df.groupby('A').all().sort_index()  # doctest: +NORMALIZE_WHITESPACE
               B
        A
        1   True
        2  False
        3  False
        4   True
        5  False
        """
        return self._reduce_for_stat_function(
            lambda col: F.min(
                F.coalesce(col.cast("boolean"), F.lit(True))
            ),
            only_numeric=False,
        )

    # TODO: skipna should be implemented.
    def any(self) -> Union[DataFrame, Series]:
        """
        Returns True if any value in the group is truthful, else False.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ...                    'B': [True, True, True, False, False,
        ...                          False, None, True, None, False]},
        ...                   columns=['A', 'B'])
        >>> df
           A      B
        0  1   True
        1  1   True
        2  2   True
        3  2  False
        4  3  False
        5  3  False
        6  4   None
        7  4   True
        8  5   None
        9  5  False

        >>> df.groupby('A').any().sort_index()  # doctest: +NORMALIZE_WHITESPACE
               B
        A
        1   True
        2   True
        3  False
        4   True
        5  False
        """
        return self._reduce_for_stat_function(
            lambda col: F.max(
                F.coalesce(col.cast("boolean"), F.lit(False))
            ),
            only_numeric=False,
        )

    # TODO: groupby multiply columns should be implemented.
    def size(self) -> Series:
        """
        Compute group sizes.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 2, 3, 3, 3],
        ...                    'B': [1, 1, 2, 3, 3, 3]},
        ...                   columns=['A', 'B'])
        >>> df
           A  B
        0  1  1
        1  2  1
        2  2  2
        3  3  3
        4  3  3
        5  3  3

        >>> df.groupby('A').size().sort_index()
        A
        1    1
        2    2
        3    3
        dtype: int64

        >>> df.groupby(['A', 'B']).size().sort_index()
        A  B
        1  1    1
        2  1    1
           2    1
        3  3    3
        dtype: int64

        For Series,

        >>> df.B.groupby(df.A).size().sort_index()
        A
        1    1
        2    2
        3    3
        Name: B, dtype: int64

        >>> df.groupby(df.A).B.size().sort_index()
        A
        1    1
        2    2
        3    3
        Name: B, dtype: int64
        """
        groupkeys = self._groupkeys
        groupkey_names = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_scols = [
            s.spark.column.alias(name) for s, name in zip(groupkeys, groupkey_names)
        ]
        sdf = self._kdf._internal.spark_frame.select(
            groupkey_scols + self._kdf._internal.data_spark_columns
        )
        sdf = sdf.groupby(*groupkey_names).count()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in groupkey_names
            ],
            index_names=[kser._column_label for kser in groupkeys],
            index_dtypes=[kser.dtype for kser in groupkeys],
            column_labels=[None],
            data_spark_columns=[scol_for(sdf, "count")],
        )
        return first_series(DataFrame(internal))

    def diff(self, periods: int = 1) -> Union[DataFrame, Series]:
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another element in the
        DataFrame group (default is the element in the same column of the previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.

        Returns
        -------
        diffed : DataFrame or Series

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                    'b': [1, 1, 2, 3, 5, 8],
        ...                    'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        >>> df
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36

        >>> df.groupby(['b']).diff().sort_index()
             a    c
        0  NaN  NaN
        1  1.0  3.0
        2  NaN  NaN
        3  NaN  NaN
        4  NaN  NaN
        5  NaN  NaN

        Difference with previous column in a group.

        >>> df.groupby(['b'])['a'].diff().sort_index()
        0    NaN
        1    1.0
        2    NaN
        3    NaN
        4    NaN
        5    NaN
        Name: a, dtype: float64
        """
        return self._apply_series_op(
            lambda sg: sg._kser._diff(periods, part_cols=sg._groupkeys_scols),
            should_resolve=True,
        )

    def cumcount(self, ascending: bool = True) -> Series:
        """
        Number each item in each group from 0 to the length of that group - 1.

        Essentially this is equivalent to

        .. code-block:: python

            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.

        Returns
        -------
        Series
            Sequence number of each element within each group.

        Examples
        --------

        >>> df = ks.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],
        ...                   columns=['A'])
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').cumcount().sort_index()
        0    0
        1    1
        2    2
        3    0
        4    1
        5    3
        dtype: int64
        >>> df.groupby('A').cumcount(ascending=False).sort_index()
        0    3
        1    2
        2    1
        3    1
        4    0
        5    0
        dtype: int64
        """
        ret = (
            self._groupkeys[0]
            .rename()
            .spark.transform(lambda _: F.lit(0))
            ._cum(
                F.count,
                True,
                part_cols=self._groupkeys_scols,
                ascending=ascending,
            )
            - 1
        )
        internal = ret._internal.resolved_copy
        return first_series(DataFrame(internal))

    def cummax(self) -> Union[DataFrame, Series]:
        """
        Cumulative max for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cummax
        DataFrame.cummax

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cummax().sort_index()
              B  C
        0   NaN  4
        1   0.1  4
        2  20.0  4
        3  10.0  1

        It works as below in Series.

        >>> df.C.groupby(df.A).cummax().sort_index()
        0    4
        1    4
        2    4
        3    1
        Name: C, dtype: int64
        """
        return self._apply_series_op(
            lambda sg: sg._kser._cum(F.max, True, part_cols=sg._groupkeys_scols),
            should_resolve=True,
            numeric_only=True,
        )

    def cummin(self) -> Union[DataFrame, Series]:
        """
        Cumulative min for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cummin
        DataFrame.cummin

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cummin().sort_index()
              B  C
        0   NaN  4
        1   0.1  3
        2   0.1  2
        3  10.0  1

        It works as below in Series.

        >>> df.B.groupby(df.A).cummin().sort_index()
        0     NaN
        1     0.1
        2     0.1
        3    10.0
        Name: B, dtype: float64
        """
        return self._apply_series_op(
            lambda sg: sg._kser._cum(F.min, True, part_cols=sg._groupkeys_scols),
            should_resolve=True,
            numeric_only=True,
        )

    def cumprod(self) -> Union[DataFrame, Series]:
        """
        Cumulative product for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cumprod
        DataFrame.cumprod

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cumprod().sort_index()
              B   C
        0   NaN   4
        1   0.1  12
        2   2.0  24
        3  10.0   1

        It works as below in Series.

        >>> df.B.groupby(df.A).cumprod().sort_index()
        0     NaN
        1     0.1
        2     2.0
        3    10.0
        Name: B, dtype: float64
        """
        return self._apply_series_op(
            lambda sg: sg._kser._cumprod(
                True, part_cols=sg._groupkeys_scols
            ),
            should_resolve=True,
            numeric_only=True,
        )

    def cumsum(self) -> Union[DataFrame, Series]:
        """
        Cumulative sum for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cumsum
        DataFrame.cumsum

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cumsum().sort_index()
              B  C
        0   NaN  4
        1   0.1  7
        2  20.1  9
        3  10.0  1

        It works as below in Series.

        >>> df.B.groupby(df.A).cumsum().sort_index()
        0     NaN
        1     0.1
        2    20.1
        3    10.0
        Name: B, dtype: float64
        """
        return self._apply_series_op(
            lambda sg: sg._kser._cumsum(
                True, part_cols=sg._groupkeys_scols
            ),
            should_resolve=True,
            numeric_only=True,
        )

    def apply(
        self,
        func: TypingCallable[[pd.DataFrame], Union[pd.DataFrame, pd.Series, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]:
        """
        Apply function `func` group-wise and combine the results together.

        The function passed to `apply` must take a DataFrame as its first
        argument and return a DataFrame. `apply` will
        then take care of combining the results back together into a single
        dataframe. `apply` is therefore a highly flexible
        grouping method.

        While `apply` is a very flexible method, its downside is that
        using it can be quite a bit slower than using more specific methods
        like `agg` or `transform`. Koalas offers a wide range of method that will
        be much faster than using `apply` for their specific purposes, so try to
        use them before reaching for `apply`.

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type in ``func``, for instance, as below:

            >>> def pandas_div(x) -> ks.DataFrame[float, float]:
            ...     return x[['B', 'C']] / x[['B', 'C']]

            If the return type is specified, the output column names become
            `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
            DataFrame in ``func``.

            To specify the column names, you can assign them in a pandas friendly style as below:

            >>> def pandas_div(x) -> ks.DataFrame["a": float, "b": float]:
            ...     return x[['B', 'C']] / x[['B', 'C']]

            >>> pdf = pd.DataFrame({'B': [1.], 'C': [3.]})
            >>> def plus_one(x) -> ks.DataFrame[zip(pdf.columns, pdf.dtypes)]:
            ...     return x[['B', 'C']] / x[['B', 'C']]

            When the given function has the return type annotated, the original index of the
            GroupBy object will be lost and a default index will be attached to the result.
            Please be careful about configuring the default index. See also `Default Index Type
            <https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type>`_.

        .. note:: the dataframe within ``func`` is actually a pandas dataframe. Therefore,
            any pandas APIs within this function is allowed.

        Parameters
        ----------
        func : callable
            A callable that takes a DataFrame as its first argument, and
            returns a dataframe.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        applied : DataFrame or Series

        See Also
        --------
        aggregate : Apply aggregate function to the GroupBy object.
        DataFrame.apply : Apply a function to a DataFrame.
        Series.apply : Apply a function to a Series.

        Examples
        --------
        >>> df = ks.DataFrame({'A': 'a a b'.split(),
        ...                    'B': [1, 2, 3],
        ...                    'C': [4, 6, 5]}, columns=['A', 'B', 'C'])
        >>> g = df.groupby('A')

        Notice that ``g`` has two groups, ``a`` and ``b``.
        Calling `apply` in various ways, we can get different grouping results:
        Below the functions passed to `apply` takes a DataFrame as
        its argument and returns a DataFrame. `apply` combines the result for
        each group together into a new DataFrame:

        >>> def plus_min(x):
        ...     return x + x.min()
        >>> g.apply(plus_min).sort_index()  # doctest: +NORMALIZE_WHITESPACE
            A  B   C
        0  aa  2   8
        1  aa  3  10
        2  bb  6  10

        >>> g.apply(sum).sort_index()  # doctest: +NORMALIZE_WHITESPACE
            A  B   C
        A
        a  aa  3  10
        b   b  3   5

        >>> g.apply(len).sort_index()  # doctest: +NORMALIZE_WHITESPACE
        A
        a    2
        b    1
        dtype: int64

        You can specify the type hint and prevent schema inference for better performance.

        >>> def pandas_div(x) -> ks.DataFrame[float, float]:
        ...     return x[['B', 'C']] / x[['B', 'C']]
        >>> g.apply(pandas_div).sort_index()  # doctest: +NORMALIZE_WHITESPACE
            c0   c1
        0  1.0  1.0
        1  1.0  1.0
        2  1.0  1.0

        In case of Series, it works as below.

        >>> def plus_max(x) -> ks.Series[np.int]:
        ...     return x + x.max()
        >>> df.B.groupby(df.A).apply(plus_max).sort_index()  # doctest: +SKIP
        0    6
        1    3
        2    4
        Name: B, dtype: int64

        >>> def plus_min(x):
        ...     return x + x.min()
        >>> df.B.groupby(df.A).apply(plus_min).sort_index()
        0    2
        1    3
        2    6
        Name: B, dtype: int64

        You can also return a scalar value as a aggregated value of the group:

        >>> def plus_length(x) -> np.int:
        ...     return len(x)
        >>> df.B.groupby(df.A).apply(plus_length).sort_index()  # doctest: +SKIP
        0    1
        1    2
        Name: B, dtype: int64

        The extra arguments to the function can be passed as below.

        >>> def calculation(x, y, z) -> np.int:
        ...     return len(x) + y * z
        >>> df.B.groupby(df.A).apply(calculation, 5, z=10).sort_index()  # doctest: +SKIP
        0    51
        1    52
        Name: B, dtype: int64
        """
        if LooseVersion(pd.__version__) >= LooseVersion("1.3.0"):
            from pandas.core.common import _builtin_table
        else:
            from pandas.core.base import SelectionMixin

            _builtin_table = SelectionMixin._builtin_table

        if not isinstance(func, Callable):
            raise TypeError("%s object is not callable" % type(func).__name__)

        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None

        is_series_groupby = isinstance(self, SeriesGroupBy)

        kdf = self._kdf

        if self._agg_columns_selected:
            agg_columns = self._agg_columns
        else:
            agg_columns = [
                kdf._kser_for(label)
                for label in kdf._internal.column_labels
                if label not in self._column_labels_to_exlcude
            ]

        kdf, groupkey_labels, groupkey_names = GroupBy._prepare_group_map_apply(
            kdf, self._groupkeys, agg_columns
        )

        if is_series_groupby:
            name = kdf.columns[-1]
            pandas_apply = _builtin_table.get(func, func)
        else:
            f = _builtin_table.get(func, func)

            def pandas_apply(pdf: pd.DataFrame, *a: Any, **k: Any) -> Union[pd.DataFrame, pd.Series]:
                return f(pdf.drop(groupkey_names, axis=1), *a, **k)

        should_return_series = False

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            limit = get_option("compute.shortcut_limit")
            pdf = kdf.head(limit + 1)._to_internal_pandas()
            groupkeys = [
                pdf[groupkey_name].rename(kser.name)
                for groupkey_name, kser in zip(groupkey_names, self._groupkeys)
            ]
            if is_series_groupby:
                pser_or_pdf = pdf.groupby(groupkeys)[name].apply(
                    pandas_apply, *args, **kwargs
                )
            else:
                pser_or_pdf = pdf.groupby(groupkeys).apply(
                    pandas_apply, *args, **kwargs
                )
            kser_or_kdf = ks.from_pandas(pser_or_pdf)

            if len(pdf) <= limit:
                if isinstance(kser_or_kdf, ks.Series) and is_series_groupby:
                    kser_or_kdf = kser_or_kdf.rename(cast(SeriesGroupBy, self)._kser.name)
                return cast(Union[Series, DataFrame], kser_or_kdf)

            if isinstance(kser_or_kdf, Series):
                should_return_series = True
                kdf_from_pandas = kser_or_kdf._kdf
            else:
                kdf_from_pandas = cast(DataFrame, kser_or_kdf)

            return_schema = force_decimal_precision_scale(
                as_nullable_spark_type(
                    kdf_from_pandas._internal.spark_frame.drop(*HIDDEN_COLUMNS).schema
                )
            )
        else:
            return_type = infer_return_type(func)
            if not is_series_groupby and isinstance(return_type, SeriesType):
                raise TypeError(
                    "Series as a return type hint at frame groupby is not supported "
                    "currently; however got [%s]. Use DataFrame type hint instead."
                    % return_sig
                )

            if isinstance(return_type, DataFrameType):
                return_schema = cast(DataFrameType, return_type).spark_type
                data_dtypes = cast(DataFrameType, return_type).dtypes
            else:
                should_return_series = True
                return_schema = cast(
                    Union[SeriesType, ScalarType], return_type
                ).spark_type
                if is_series_groupby:
                    return_schema = StructType([StructField(name, return_schema)])
                else:
                    return_schema = StructType(
                        [StructField(SPARK_DEFAULT_SERIES_NAME, return_schema)]
                    )
                data_dtypes = [
                    cast(Union[SeriesType, ScalarType], return_type).dtype
                ]

        def pandas_groupby_apply(pdf: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
            if not is_series_groupby and LooseVersion(pd.__version__) < LooseVersion("0.25"):
                # `groupby.apply` in pandas<0.25 runs the functions twice for the first group.
                # https://github.com/pandas-dev/pandas/pull/24748

                should_skip_first_call = True

                def wrapped_func(df: pd.DataFrame, *a: Any, **k: Any) -> Union[pd.DataFrame, pd.Series]:
                    nonlocal should_skip_first_call
                    if should_skip_first_call:
                        should_skip_first_call = False
                        if should_return_series:
                            return pd.Series()
                        else:
                            return pd.DataFrame()
                    else:
                        return pandas_apply(df, *a, **k)

            else:
                wrapped_func = pandas_apply

            if is_series_groupby:
                pdf_or_ser = pdf.groupby(groupkey_names)[name].apply(
                    wrapped_func, *args, **kwargs
                )
            else:
                pdf_or_ser = pdf.groupby(groupkey_names).apply(
                    wrapped_func, *args, **kwargs
                )

            if not isinstance(pdf_or_ser, pd.DataFrame):
                return pd.DataFrame(pdf_or_ser)
            else:
                return pdf_or_ser

        sdf = GroupBy._spark_group_map_apply(
            kdf,
            pandas_groupby_apply,
            [kdf._internal.spark_column_for(label) for label in groupkey_labels],
            return_schema,
            retain_index=should_infer_schema,
        )

        if should_infer_schema:
            # If schema is inferred, we can restore indexes too.
            internal = kdf_from_pandas._internal.with_new_sdf(sdf)
        else:
            # Otherwise, it loses index.
            internal = InternalFrame(
                spark_frame=sdf,
                index_spark_columns=None,
                data_dtypes=data_dtypes,
            )

        if should_return_series:
            kser = first_series(DataFrame(internal))
            if is_series_groupby:
                kser = kser.rename(cast(SeriesGroupBy, self)._kser.name)
            return kser
        else:
            return DataFrame(internal)

    # TODO: implement 'dropna' parameter
    def filter(
        self, func: TypingCallable[[Union[DataFrame, Series]], Union[bool, pd.Series, pd.DataFrame]]
    ) -> Union[DataFrame, Series]:
        """
        Return a copy of a DataFrame excluding elements from groups that
        do not satisfy the boolean criterion specified by func.

        Parameters
        ----------
        f : function
            Function to apply to each subframe. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            if False, groups that evaluate False are filled with NaNs.

        Returns
        -------
        filtered : DataFrame or Series

        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.

        Examples
        --------
        >>> df = ks.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]}, columns=['A', 'B', 'C'])
        >>> grouped = df.groupby('A')
        >>> grouped.filter(lambda x: x['B'].mean() > 3.)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0

        >>> df.B.groupby(df.A).filter(lambda x: x.mean() > 3.)
        1    2
        3    4
        5    6
        Name: B, dtype: int64
        """
        if LooseVersion(pd.__version__) >= LooseVersion("1.3.0"):
            from pandas.core.common import _builtin_table
        else:
            from pandas.core.base import SelectionMixin

            _builtin_table = SelectionMixin._builtin_table

        if not isinstance(func, Callable):
            raise TypeError("%s object is not callable" % type(func).__name__)

        is_series_groupby = isinstance(self, SeriesGroupBy)

        kdf = self._kdf

        if self._agg_columns_selected:
            agg_columns = self._agg_columns
        else:
            agg_columns = [
                kdf._kser_for(label)
                for label in kdf._internal.column_labels
                if label not in self._column_labels_to_exlcude
            ]

        data_schema = (
            kdf[agg_columns]._internal.resolved_copy.spark_frame.drop(*HIDDEN_COLUMNS).schema
        )

        kdf, groupkey_labels, groupkey_names = GroupBy._prepare_group_map_apply(
            kdf, self._groupkeys, agg_columns
        )

        if is_series_groupby:

            def pandas_filter(pdf: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    pdf.groupby(groupkey_names)[pdf.columns[-1]].filter(func)
                )

        else:
            f = _builtin_table.get(func, func)

            def wrapped_func(pdf: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
                return f(pdf.drop(groupkey_names, axis=1))

            def pandas_filter(pdf: pd.DataFrame) -> pd.DataFrame:
                return pdf.groupby(groupkey_names).filter(wrapped_func).drop(
                    groupkey_names, axis=1
                )

        sdf = GroupBy._spark_group_map_apply(
            kdf,
            pandas_filter,
            [kdf._internal.spark_column_for(label) for label in groupkey_labels],
            data_schema,
            retain_index=True,
        )

        kdf = DataFrame(self._kdf[agg_columns]._internal.with_new_sdf(sdf))
        if is_series_groupby:
            return first_series(kdf)
        else:
            return kdf

    @staticmethod
    def _prepare_group_map_apply(
        kdf: DataFrame,
        groupkeys: List[Series],
        agg_columns: List[Series],
    ) -> Tuple[DataFrame, List[Tuple[Any, ...]], List[str]]:
        groupkey_labels = [
            verify_temp_column_name(kdf, "__groupkey_{}__".format(i))
            for i in range(len(groupkeys))
        ]
        kdf = DataFrame(
            kdf[
                [s.rename(label) for s, label in zip(groupkeys, groupkey_labels)]
                + agg_columns
            ]
        )
        groupkey_names = [
            name if len(name) > 1 else name[0] for name in groupkey_labels
        ]
        return kdf, groupkey_labels, groupkey_names

    @staticmethod
    def _spark_group_map_apply(
        kdf: DataFrame,
        func: TypingCallable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]],
        groupkeys_scols: List[Column],
        return_schema: StructType,
        retain_index: bool,
    ) -> Any:
        output_func = GroupBy._make_pandas_df_builder_func(
            kdf, func, return_schema, retain_index
        )
        grouped_map_func = pandas_udf(
            return_schema, PandasUDFType.GROUPED_MAP
        )(output_func)
        sdf = kdf._internal.spark_frame.drop(*HIDDEN_COLUMNS)
        return sdf.groupby(*groupkeys_scols).apply(grouped_map_func)

    @staticmethod
    def _make_pandas_df_builder_func(
        kdf: DataFrame,
        func: TypingCallable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]],
        return_schema: StructType,
        retain_index: bool,
    ) -> TypingCallable[[pd.DataFrame], pd.DataFrame]:
        """
        Creates a function that can be used inside the pandas UDF. This function can construct
        the same pandas DataFrame as if the Koalas DataFrame is collected to driver side.
        The index, column labels, etc. are re-constructed within the function.
        """
        arguments_for_restore_index = kdf._internal.arguments_for_restore_index

        def rename_output(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = InternalFrame.restore_index(pdf.copy(), **arguments_for_restore_index)

            pdf = func(pdf)

            # If schema should be inferred, we don't restore index. pandas seems restoring
            # the index in some cases.
            # When Spark output type is specified, without executing it, we don't know
            # if we should restore the index or not. For instance, see the example in
            # https://github.com/databricks/koalas/issues/628.
            pdf, _, _, _, _ = InternalFrame.prepare_pandas_frame(
                pdf, retain_index=retain_index
            )

            # Just positionally map the column names to given schema's.
            pdf.columns = return_schema.names

            return pdf

        return rename_output

    def rank(
        self, method: str = "average", ascending: bool = True
    ) -> Union[DataFrame, Series]:
        """
        Provide the rank of values within each group.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            * average: average rank of group
            * min: lowest rank in group
            * max: highest rank in group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups
        ascending : boolean, default True
            False for ranks by high (1) to low (N)

        Returns
        -------
        DataFrame with ranking of values within each group

        Examples
        --------

        >>> df = ks.DataFrame({
        ...     'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...     'b': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b'])
        >>> df
           a  b
        0  1  1
        1  1  2
        2  1  2
        3  2  2
        4  2  3
        5  2  3
        6  3  3
        7  3  4
        8  3  4

        >>> df.groupby("a").rank().sort_index()
             b
        0  1.0
        1  2.5
        2  2.5
        3  1.0
        4  2.5
        5  2.5
        6  1.0
        7  2.5
        8  2.5

        >>> df.b.groupby(df.a).rank(method='max').sort_index()
        0    1.0
        1    3.0
        2    3.0
        3    1.0
        4    3.0
        5    3.0
        6    1.0
        7    3.0
        8    3.0
        Name: b, dtype: float64

        """
        return self._apply_series_op(
            lambda sg: sg._kser._rank(
                method, ascending, part_cols=sg._groupkeys_scols
            ),
            should_resolve=True,
        )

    # TODO: add axis parameter
    def idxmax(self, skipna: bool = True) -> Union[DataFrame, Series]:
        """
        Return index of first occurrence of maximum over requested axis in group.
        NA/null values are excluded.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        See Also
        --------
        Series.idxmax
        DataFrame.idxmax
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 2, 2, 3],
        ...                    'b': [1, 2, 3, 4, 5],
        ...                    'c': [5, 4, 3, 2, 1]},
        ...                   columns=['a', 'b', 'c'])

        >>> df.groupby(['a'])['b'].idxmax().sort_index() # doctest: +NORMALIZE_WHITESPACE
        a
        1  1
        2  3
        3  4
        Name: b, dtype: int64

        >>> df.groupby(['a']).idxmax().sort_index() # doctest: +NORMALIZE_WHITESPACE
           b  c
        a
        1  1  0
        2  3  2
        3  4  4
        """
        if self._kdf._internal.index_level != 1:
            raise ValueError("idxmax only support one-level index now")

        groupkey_names = ["__groupkey_{}__".format(i) for i in range(len(self._groupkeys))]

        sdf = self._kdf._internal.spark_frame
        for s, name in zip(self._groupkeys, groupkey_names):
            sdf = sdf.withColumn(name, s.spark.column)
        index = self._kdf._internal.index_spark_column_names[0]

        stat_exprs: List[Any] = []
        for kser, c in zip(self._agg_columns, self._agg_columns_scols):
            name = kser._internal.data_spark_column_names[0]

            if skipna:
                order_column = Column(c._jc.desc_nulls_last())
            else:
                order_column = Column(c._jc.desc_nulls_first())
            window = Window.partitionBy(groupkey_names).orderBy(
                order_column, NATURAL_ORDER_COLUMN_NAME
            )
            sdf = sdf.withColumn(
                name,
                F.when(F.row_number().over(window) == 1, scol_for(sdf, index)).otherwise(None),
            )
            stat_exprs.append(F.max(scol_for(sdf, name)).alias(name))

        sdf = sdf.groupby(*groupkey_names).agg(*stat_exprs)

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in groupkey_names
            ],
            index_names=[kser._column_label for kser in self._groupkeys],
            index_dtypes=[kser.dtype for kser in self._groupkeys],
            column_labels=[kser._column_label for kser in self._agg_columns],
            data_spark_columns=[
                scol_for(sdf, kser._internal.data_spark_column_names[0])
                for kser in self._agg_columns
            ],
        )
        return DataFrame(internal)

    # TODO: add axis parameter
    def idxmin(self, skipna: bool = True) -> Union[DataFrame, Series]:
        """
        Return index of first occurrence of minimum over requested axis in group.
        NA/null values are excluded.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        See Also
        --------
        Series.idxmin
        DataFrame.idxmin
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 2, 2, 3],
        ...                    'b': [1, 2, 3, 4, 5],
        ...                    'c': [5, 4, 3, 2, 1]},
        ...                   columns=['a', 'b', 'c'])

        >>> df.groupby(['a'])['b'].idxmin().sort_index() # doctest: +NORMALIZE_WHITESPACE
        a
        1    0
        2    2
        3    4
        Name: b, dtype: int64

        >>> df.groupby(['a']).idxmin().sort_index() # doctest: +NORMALIZE_WHITESPACE
           b  c
        a
        1  0  1
        2  2  3
        3  4  4
        """
        if self._kdf._internal.index_level != 1:
            raise ValueError("idxmin only support one-level index now")

        groupkey_names = ["__groupkey_{}__".format(i) for i in range(len(self._groupkeys))]

        sdf = self._kdf._internal.spark_frame
        for s, name in zip(self._groupkeys, groupkey_names):
            sdf = sdf.withColumn(name, s.spark.column)
        index = self._kdf._internal.index_spark_column_names[0]

        stat_exprs: List[Any] = []
        for kser, c in zip(self._agg_columns, self._agg_columns_scols):
            name = kser._internal.data_spark_column_names[0]

            if skipna:
                order_column = Column(c._jc.asc_nulls_last())
            else:
                order_column = Column(c._jc.asc_nulls_first())
            window = Window.partitionBy(groupkey_names).orderBy(
                order_column, NATURAL_ORDER_COLUMN_NAME
            )
            sdf = sdf.withColumn(
                name,
                F.when(F.row_number().over(window) == 1, scol_for(sdf, index)).otherwise(None),
            )
            stat_exprs.append(F.max(scol_for(sdf, name)).alias(name))

        sdf = sdf.groupby(*groupkey_names).agg(*stat_exprs)

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in groupkey_names
            ],
            index_names=[kser._column_label for kser in self._groupkeys],
            column_labels=[kser._column_label for kser in self._agg_columns],
            data_spark_columns=[
                scol_for(sdf, kser._internal.data_spark_column_names[0])
                for kser in self._agg_columns
            ],
        )
        return DataFrame(internal)

    def fillna(
        self,
        value: Optional[Union[Any, Dict[str, Any], Series]] = None,
        method: Optional[str] = None,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Union[DataFrame, Series]:
        """Fill NA/NaN values in group.

        Parameters
        ----------
        value : scalar, dict, Series
            Value to use to fill holes. alternately a dict/Series of values
            specifying which value to use for each column.
            DataFrame is not supported.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series pad / ffill: propagate last valid
            observation forward to next valid backfill / bfill:
            use NEXT valid observation to fill gap
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values to
            forward/backward fill. In other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None

        Returns
        -------
        DataFrame
            DataFrame with NA entries filled.

        Examples
        --------

        >>> df = ks.DataFrame({
        ...     'A': [1, 1, 2, 2],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> df
           A    B    C  D
        0  1  2.0  NaN  0
        1  1  4.0  NaN  1
        2  2  NaN  NaN  5
        3  2  3.0  1.0  4

        We can also propagate non-null values forward or backward in group.

        >>> df.groupby(['A'])['B'].fillna(method='ffill').sort_index()
        0    2.0
        1    4.0
        2    NaN
        3    3.0
        Name: B, dtype: float64

        >>> df.groupby(['A']).fillna(method='bfill').sort_index()
             B    C  D
        0  2.0  NaN  0
        1  4.0  NaN  1
        2  3.0  1.0  5
        3  3.0  1.0  4
        """
        return self._apply_series_op(
            lambda sg: sg._kser._fillna(
                value=value,
                method=method,
                axis=axis,
                limit=limit,
                part_cols=sg._groupkeys_scols,
            ),
            should_resolve=(method is not None),
        )

    def bfill(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        """
        Synonym for `DataFrame.fillna()` with ``method=`bfill