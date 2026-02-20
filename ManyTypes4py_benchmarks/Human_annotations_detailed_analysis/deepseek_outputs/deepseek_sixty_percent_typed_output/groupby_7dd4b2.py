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
from typing import Any, List, Set, Tuple, Union, cast, Optional, Dict

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
from databricks.koalas.typedef import infer_return_type, DataFrameType, ScalarType, SeriesType
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
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
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
    ):
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
    def _apply_series_op(self, op: Callable, should_resolve: bool = False, numeric_only: bool = False) -> Union[DataFrame, Series]:
        pass

    # TODO: Series support is not implemented yet.
    # TODO: not all arguments are implemented comparing to pandas' for now.
    def aggregate(self, func_or_funcs: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None, *args: Any, **kwargs: Any) -> DataFrame:
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
        if func_or_funcs is None and kwargs is None:
            raise ValueError("No aggregation argument or function specified.")

        relabeling = func_or_funcs is None and is_multi_agg_with_relabel(**kwargs)
        if relabeling:
            func_or_funcs, columns, order = normalize_keyword_aggregation(kwargs)

        if not isinstance(func_or_funcs, (str, list)):
            if not isinstance(func_or_funcs, dict) or not all(
                is_name_like_value(key)
                and (
                    isinstance(value, str)
                    or isinstance(value, list)
                    and all(isinstance(v, str) for v in value)
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
                    kdf._internal.spark_frame.dropna(subset=kdf._internal.index_spark_column_names)
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
    def _spark_groupby(kdf: DataFrame, func: Dict[str, Union[str, List[str]]], groupkeys: Tuple[Series, ...] = ()) -> InternalFrame:
        groupkey_names = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_scols = [s.spark.column.alias(name) for s, name in zip(groupkeys, groupkey_names)]

        multi_aggs = any(isinstance(v, list) for v in func.values())
        reordered = []
        data_columns = []
        column_labels = []
        for key, value in func.items():
            label = key if is_name_like_tuple(key) else (key,)
            if len(label) != kdf._internal.column_labels_level:
                raise TypeError("The length of the key must be the same as the column label level.")
            for aggfunc in [value] if isinstance(value, str) else value:
                column_label = tuple(list(label) + [aggfunc]) if multi_aggs else label
                column_labels.append(column_label)

                data_col = name_like_string(column_label)
                data_columns.append(data_col)

                col_name = kdf._internal.spark_column_name_for(label)
                if aggfunc == "nunique":
                    reordered.append(
                        F.expr("count(DISTINCT `{0}`) as `{1}`".format(col_name, data_col))
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
                        F.expr("{1}(`{0}`) as `{2}`".format(col_name, aggfunc, data_col))
                    )

        sdf = kdf._internal.spark_frame.select(groupkey_scols + kdf._internal.data_spark_columns)
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
            index_spark_columns=[scol_for(sdf, col) for col in index_spark_column_names],
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
        ddof : int,