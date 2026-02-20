#!/usr/bin/env python
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
from collections.abc import Callable, Iterable
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
        column_labels_to_exlcude: Set[Tuple[Any, ...]],
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
    def _apply_series_op(self, op: Callable[[Any], Union[DataFrame, Series]], should_resolve: bool = False, numeric_only: bool = False) -> Union[DataFrame, Series]:
        pass

    # TODO: Series support is not implemented yet.
    # TODO: not all arguments are implemented comparing to pandas' for now.
    def aggregate(self, 
                  func_or_funcs: Optional[Union[str, List[str], Dict[Any, Union[str, List[str]]]]] = None, 
                  *args: Any, **kwargs: Any) -> DataFrame:
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
        """
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
                    or (isinstance(value, list)
                        and all(isinstance(v, str) for v in value))
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
    def _spark_groupby(kdf: DataFrame, func: Any, groupkeys: Iterable[Series] = ()) -> InternalFrame:
        groupkey_names: List[str] = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_scols: List[Column] = [s.spark.column.alias(name) for s, name in zip(groupkeys, groupkey_names)]

        multi_aggs = any(isinstance(v, list) for v in func.values())
        reordered: List[Any] = []
        data_columns: List[str] = []
        column_labels: List[Tuple[Any, ...]] = []
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
            index_spark_column_names: List[str] = groupkey_names
            index_names: List[Any] = [kser._column_label for kser in groupkeys]
            index_dtypes: List[Any] = [kser.dtype for kser in groupkeys]
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
        >>> df.groupby('A').count().sort_index()
            B  C
        A
        1  2  3
        2  2  2
        """
        return self._reduce_for_stat_function(F.count, only_numeric=False)

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
        >>> df.groupby('A').mean().sort_index()
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

    def std(self, ddof: int = 1) -> Union[DataFrame, Series]:
        """
        Compute standard deviation of groups, excluding missing values.
        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.
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

    def var(self, ddof: int = 1) -> Union[DataFrame, Series]:
        """
        Compute variance of groups, excluding missing values.
        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.
        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        assert ddof in (0, 1)
        return self._reduce_for_stat_function(
            F.var_pop if ddof == 0 else F.var_samp, only_numeric=True
        )

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
        >>> df.groupby('A').all().sort_index()
               B
        A
        1   True
        2  False
        3  False
        4   True
        5  False
        """
        return self._reduce_for_stat_function(
            lambda col: F.min(F.coalesce(col.cast("boolean"), F.lit(True))), only_numeric=False
        )

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
        >>> df.groupby('A').any().sort_index()
               B
        A
        1   True
        2   True
        3  False
        4   True
        5  False
        """
        return self._reduce_for_stat_function(
            lambda col: F.max(F.coalesce(col.cast("boolean"), F.lit(False))), only_numeric=False
        )

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
        >>> df.groupby('A').size().sort_index()
        A
        1    1
        2    2
        3    3
        dtype: int64
        """
        groupkeys = self._groupkeys
        groupkey_names: List[str] = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_scols: List[Column] = [s.spark.column.alias(name) for s, name in zip(groupkeys, groupkey_names)]
        sdf = self._kdf._internal.spark_frame.select(
            groupkey_scols + self._kdf._internal.data_spark_columns
        )
        sdf = sdf.groupby(*groupkey_names).count()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in groupkey_names],
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
        """
        return self._apply_series_op(
            lambda sg: sg._kser._diff(periods, part_cols=sg._groupkeys_scols), should_resolve=True
        )

    def cumcount(self, ascending: bool = True) -> Series:
        """
        Number each item in each group from 0 to the length of that group - 1.
        Returns
        -------
        Series
        """
        ret = (
            self._groupkeys[0]
            .rename()
            .spark.transform(lambda _: F.lit(0))
            ._cum(F.count, True, part_cols= self._groupkeys_scols, ascending=ascending)
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
        """
        return self._apply_series_op(
            lambda sg: sg._kser._cumprod(True, part_cols=sg._groupkeys_scols),
            should_resolve=True,
            numeric_only=True,
        )

    def cumsum(self) -> Union[DataFrame, Series]:
        """
        Cumulative sum for each group.
        Returns
        -------
        Series or DataFrame
        """
        return self._apply_series_op(
            lambda sg: sg._kser._cumsum(True, part_cols=sg._groupkeys_scols),
            should_resolve=True,
            numeric_only=True,
        )

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> Union[DataFrame, Series]:
        """
        Apply function `func` group-wise and combine the results together.
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
        """
        if LooseVersion(pd.__version__) >= LooseVersion("1.3.0"):
            from pandas.core.common import _builtin_table
        else:
            from pandas.core.base import SelectionMixin
            _builtin_table = SelectionMixin._builtin_table

        if not isinstance(func, Callable):  # type: ignore
            raise TypeError("%s object is not callable" % type(func).__name__)

        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema: bool = return_sig is None

        is_series_groupby: bool = isinstance(self, SeriesGroupBy)

        kdf = self._kdf

        if self._agg_columns_selected:
            agg_columns: List[Series] = self._agg_columns
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
            def pandas_apply(pdf: pd.DataFrame, *a: Any, **k: Any) -> Any:
                return f(pdf.drop(groupkey_names, axis=1), *a, **k)

        should_return_series: bool = False

        if should_infer_schema:
            limit = get_option("compute.shortcut_limit")
            pdf: pd.DataFrame = kdf.head(limit + 1)._to_internal_pandas()
            groupkeys = [
                pdf[groupkey_name].rename(kser.name)
                for groupkey_name, kser in zip(groupkey_names, self._groupkeys)
            ]
            if is_series_groupby:
                pser_or_pdf = pdf.groupby(groupkeys)[name].apply(pandas_apply, *args, **kwargs)
            else:
                pser_or_pdf = pdf.groupby(groupkeys).apply(pandas_apply, *args, **kwargs)
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
                    "currently; however got [%s]. Use DataFrame type hint instead." % return_sig
                )
            if isinstance(return_type, DataFrameType):
                return_schema = cast(DataFrameType, return_type).spark_type
                data_dtypes = cast(DataFrameType, return_type).dtypes
            else:
                should_return_series = True
                return_schema = cast(Union[SeriesType, ScalarType], return_type).spark_type
                if is_series_groupby:
                    return_schema = StructType([StructField(name, return_schema)])
                else:
                    return_schema = StructType(
                        [StructField(SPARK_DEFAULT_SERIES_NAME, return_schema)]
                    )
                data_dtypes = [cast(Union[SeriesType, ScalarType], return_type).dtype]

        def pandas_groupby_apply(pdf: pd.DataFrame) -> pd.DataFrame:
            if not is_series_groupby and LooseVersion(pd.__version__) < LooseVersion("0.25"):
                should_skip_first_call = True
                def wrapped_func(df: pd.DataFrame, *a: Any, **k: Any) -> Any:
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
                pdf_or_ser = pdf.groupby(groupkey_names)[name].apply(wrapped_func, *args, **kwargs)
            else:
                pdf_or_ser = pdf.groupby(groupkey_names).apply(wrapped_func, *args, **kwargs)
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
            internal = kdf_from_pandas._internal.with_new_sdf(sdf)
        else:
            internal = InternalFrame(
                spark_frame=sdf, index_spark_columns=None, data_dtypes=data_dtypes
            )
        if should_return_series:
            kser = first_series(DataFrame(internal))
            if is_series_groupby:
                kser = kser.rename(cast(SeriesGroupBy, self)._kser.name)
            return kser
        else:
            return DataFrame(internal)

    def filter(self, func: Callable) -> Union[DataFrame, Series]:
        """
        Return a copy of a DataFrame excluding elements from groups that
        do not satisfy the boolean criterion specified by func.
        Parameters
        ----------
        func : function
            Function to apply to each subframe.
        Returns
        -------
        filtered : DataFrame or Series
        """
        if LooseVersion(pd.__version__) >= LooseVersion("1.3.0"):
            from pandas.core.common import _builtin_table
        else:
            from pandas.core.base import SelectionMixin
            _builtin_table = SelectionMixin._builtin_table

        if not isinstance(func, Callable):  # type: ignore
            raise TypeError("%s object is not callable" % type(func).__name__)

        is_series_groupby: bool = isinstance(self, SeriesGroupBy)
        kdf = self._kdf

        if self._agg_columns_selected:
            agg_columns: List[Series] = self._agg_columns
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
                return pd.DataFrame(pdf.groupby(groupkey_names)[pdf.columns[-1]].filter(func))
        else:
            f = _builtin_table.get(func, func)
            def wrapped_func(pdf: pd.DataFrame) -> Any:
                return f(pdf.drop(groupkey_names, axis=1))
            def pandas_filter(pdf: pd.DataFrame) -> pd.DataFrame:
                return pdf.groupby(groupkey_names).filter(wrapped_func).drop(groupkey_names, axis=1)

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
    def _prepare_group_map_apply(kdf: DataFrame, groupkeys: List[Series], agg_columns: List[Series]) -> Tuple[DataFrame, List[str], List[str]]:
        groupkey_labels: List[str] = [
            verify_temp_column_name(kdf, "__groupkey_{}__".format(i)) for i in range(len(groupkeys))
        ]
        kdf = kdf[[s.rename(label) for s, label in zip(groupkeys, groupkey_labels)] + agg_columns]
        groupkey_names: List[str] = [label if len(label) > 1 else label[0] for label in groupkey_labels]
        return DataFrame(kdf._internal.resolved_copy), groupkey_labels, groupkey_names

    @staticmethod
    def _spark_group_map_apply(kdf: DataFrame, func: Callable[[pd.DataFrame], pd.DataFrame], groupkeys_scols: List[Column], return_schema: StructType, retain_index: bool) -> Any:
        output_func: Callable[[pd.DataFrame], pd.DataFrame] = GroupBy._make_pandas_df_builder_func(kdf, func, return_schema, retain_index)
        grouped_map_func = pandas_udf(return_schema, PandasUDFType.GROUPED_MAP)(output_func)
        sdf = kdf._internal.spark_frame.drop(*HIDDEN_COLUMNS)
        return sdf.groupby(*groupkeys_scols).apply(grouped_map_func)

    @staticmethod
    def _make_pandas_df_builder_func(kdf: DataFrame, func: Callable[[pd.DataFrame], pd.DataFrame], return_schema: StructType, retain_index: bool) -> Callable[[pd.DataFrame], pd.DataFrame]:
        arguments_for_restore_index = kdf._internal.arguments_for_restore_index
        def rename_output(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = InternalFrame.restore_index(pdf.copy(), **arguments_for_restore_index)
            pdf = func(pdf)
            pdf, _, _, _, _ = InternalFrame.prepare_pandas_frame(pdf, retain_index=retain_index)
            pdf.columns = return_schema.names
            return pdf
        return rename_output

    def rank(self, method: str = "average", ascending: bool = True) -> Union[DataFrame, Series]:
        """
        Provide the rank of values within each group.
        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        ascending : boolean, default True
        Returns
        -------
        DataFrame with ranking of values within each group
        """
        return self._apply_series_op(
            lambda sg: sg._kser._rank(method, ascending, part_cols=sg._groupkeys_scols),
            should_resolve=True,
        )

    def idxmax(self, skipna: bool = True) -> Union[DataFrame, Series]:
        """
        Return index of first occurrence of maximum over requested axis in group.
        Parameters
        ----------
        skipna : boolean, default True
        See Also
        --------
        databricks.koalas.Series.idxmax
        databricks.koalas.DataFrame.idxmax
        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 2, 2, 3],
        ...                    'b': [1, 2, 3, 4, 5],
        ...                    'c': [5, 4, 3, 2, 1]}, columns=['a', 'b', 'c'])
        >>> df.groupby(['a'])['b'].idxmax().sort_index()
        a
        1  1
        2  3
        3  4
        Name: b, dtype: int64
        """
        if self._kdf._internal.index_level != 1:
            raise ValueError("idxmax only support one-level index now")
        groupkey_names: List[str] = ["__groupkey_{}__".format(i) for i in range(len(self._groupkeys))]
        sdf = self._kdf._internal.spark_frame
        for s, name in zip(self._groupkeys, groupkey_names):
            sdf = sdf.withColumn(name, s.spark.column)
        index: str = self._kdf._internal.index_spark_column_names[0]
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
                name, F.when(F.row_number().over(window) == 1, scol_for(sdf, index)).otherwise(None)
            )
            stat_exprs.append(F.max(scol_for(sdf, name)).alias(name))
        sdf = sdf.groupby(*groupkey_names).agg(*stat_exprs)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in groupkey_names],
            index_names=[kser._column_label for kser in self._groupkeys],
            index_dtypes=[kser.dtype for kser in self._groupkeys],
            column_labels=[kser._column_label for kser in self._agg_columns],
            data_spark_columns=[
                scol_for(sdf, kser._internal.data_spark_column_names[0])
                for kser in self._agg_columns
            ],
        )
        return DataFrame(internal)

    def idxmin(self, skipna: bool = True) -> Union[DataFrame, Series]:
        """
        Return index of first occurrence of minimum over requested axis in group.
        Parameters
        ----------
        skipna : boolean, default True
        See Also
        --------
        databricks.koalas.Series.idxmin
        databricks.koalas.DataFrame.idxmin
        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 2, 2, 3],
        ...                    'b': [1, 2, 3, 4, 5],
        ...                    'c': [5, 4, 3, 2, 1]}, columns=['a', 'b', 'c'])
        >>> df.groupby(['a'])['b'].idxmin().sort_index()
        a
        1    0
        2    2
        3    4
        Name: b, dtype: int64
        """
        if self._kdf._internal.index_level != 1:
            raise ValueError("idxmin only support one-level index now")
        groupkey_names: List[str] = ["__groupkey_{}__".format(i) for i in range(len(self._groupkeys))]
        sdf = self._kdf._internal.spark_frame
        for s, name in zip(self._groupkeys, groupkey_names):
            sdf = sdf.withColumn(name, s.spark.column)
        index: str = self._kdf._internal.index_spark_column_names[0]
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
                name, F.when(F.row_number().over(window) == 1, scol_for(sdf, index)).otherwise(None)
            )
            stat_exprs.append(F.max(scol_for(sdf, name)).alias(name))
        sdf = sdf.groupby(*groupkey_names).agg(*stat_exprs)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in groupkey_names],
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
        value: Any = None, 
        method: Optional[str] = None, 
        axis: Any = None, 
        inplace: bool = False, 
        limit: Optional[int] = None
    ) -> Union[DataFrame, Series]:
        """Fill NA/NaN values in group.
        Parameters
        ----------
        value : scalar, dict, Series
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        axis : {0 or `index`}
        inplace : boolean, default False
        limit : int, default None
        Returns
        -------
        DataFrame
        """
        return self._apply_series_op(
            lambda sg: sg._kser._fillna(
                value=value, method=method, axis=axis, limit=limit, part_cols=sg._groupkeys_scols
            ),
            should_resolve=(method is not None),
        )

    def bfill(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        """
        Synonym for `DataFrame.fillna()` with ``method='bfill'``.
        Parameters
        ----------
        limit : int, default None
        Returns
        -------
        DataFrame
        """
        return self.fillna(method="bfill", limit=limit)

    backfill = bfill

    def ffill(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        """
        Synonym for `DataFrame.fillna()` with ``method='ffill'``.
        Parameters
        ----------
        limit : int, default None
        Returns
        -------
        DataFrame
        """
        return self.fillna(method="ffill", limit=limit)

    pad = ffill

    def _limit(self, n: int, asc: bool) -> Union[DataFrame, Series]:
        """
        Private function for tail and head.
        """
        kdf = self._kdf
        if self._agg_columns_selected:
            agg_columns: List[Series] = self._agg_columns
        else:
            agg_columns = [
                kdf._kser_for(label)
                for label in kdf._internal.column_labels
                if label not in self._column_labels_to_exlcude
            ]
        kdf, groupkey_labels, _ = GroupBy._prepare_group_map_apply(
            kdf, self._groupkeys, agg_columns,
        )
        groupkey_scols: List[Column] = [kdf._internal.spark_column_for(label) for label in groupkey_labels]
        sdf = kdf._internal.spark_frame
        tmp_col: str = verify_temp_column_name(sdf, "__row_number__")
        window = (
            Window.partitionBy(groupkey_scols).orderBy(F.col(NATURAL_ORDER_COLUMN_NAME).asc())
            if asc
            else Window.partitionBy(groupkey_scols).orderBy(F.col(NATURAL_ORDER_COLUMN_NAME).desc())
        )
        sdf = (
            sdf.withColumn(tmp_col, F.row_number().over(window))
            .filter(F.col(tmp_col) <= n)
            .drop(tmp_col)
        )
        internal = kdf._internal.with_new_sdf(sdf)
        return DataFrame(internal).drop(groupkey_labels, axis=1)

    def head(self, n: int = 5) -> Union[DataFrame, Series]:
        """
        Return first n rows of each group.
        Returns
        -------
        DataFrame or Series
        """
        return self._limit(n, asc=True)

    def tail(self, n: int = 5) -> Union[DataFrame, Series]:
        """
        Return last n rows of each group.
        Returns
        -------
        DataFrame or Series
        """
        return self._limit(n, asc=False)

    def shift(self, periods: int = 1, fill_value: Any = None) -> Union[DataFrame, Series]:
        """
        Shift each group by periods observations.
        Parameters
        ----------
        periods : integer, default 1
        fill_value : optional
        Returns
        -------
        Series or DataFrame
        """
        return self._apply_series_op(
            lambda sg: sg._kser._shift(periods, fill_value, part_cols=sg._groupkeys_scols),
            should_resolve=True,
        )

    def transform(self, func: Callable, *args: Any, **kwargs: Any) -> Union[DataFrame, Series]:
        """
        Apply function column-by-column to the GroupBy object.
        Parameters
        ----------
        func : callable
            A callable that takes a Series as its first argument, and
            returns a Series.
        Returns
        -------
        DataFrame
        """
        if not isinstance(func, Callable):  # type: ignore
            raise TypeError("%s object is not callable" % type(func).__name__)
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        kdf, groupkey_labels, groupkey_names = GroupBy._prepare_group_map_apply(
            self._kdf, self._groupkeys, agg_columns=self._agg_columns
        )
        def pandas_transform(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.groupby(groupkey_names).transform(func, *args, **kwargs)
        should_infer_schema: bool = return_sig is None
        if should_infer_schema:
            limit = get_option("compute.shortcut_limit")
            pdf: pd.DataFrame = kdf.head(limit + 1)._to_internal_pandas()
            pdf = pdf.groupby(groupkey_names).transform(func, *args, **kwargs)
            kdf_from_pandas: DataFrame = DataFrame(pdf)
            return_schema = force_decimal_precision_scale(
                as_nullable_spark_type(
                    kdf_from_pandas._internal.spark_frame.drop(*HIDDEN_COLUMNS).schema
                )
            )
            if len(pdf) <= limit:
                return kdf_from_pandas
            sdf = GroupBy._spark_group_map_apply(
                kdf,
                pandas_transform,
                [kdf._internal.spark_column_for(label) for label in groupkey_labels],
                return_schema,
                retain_index=True,
            )
            internal = kdf_from_pandas._internal.with_new_sdf(sdf)
        else:
            return_type = infer_return_type(func)
            if not isinstance(return_type, SeriesType):
                raise TypeError(
                    "Expected the return type of this function to be of Series type, "
                    "but found type {}".format(return_type)
                )
            return_schema = cast(SeriesType, return_type).spark_type
            data_columns = kdf._internal.data_spark_column_names
            return_schema = StructType(
                [StructField(c, return_schema) for c in data_columns if c not in groupkey_names]
            )
            data_dtypes = [
                cast(SeriesType, return_type).dtype for c in data_columns if c not in groupkey_names
            ]
            sdf = GroupBy._spark_group_map_apply(
                kdf,
                pandas_transform,
                [kdf._internal.spark_column_for(label) for label in groupkey_labels],
                return_schema,
                retain_index=False,
            )
            internal = InternalFrame(
                spark_frame=sdf, index_spark_columns=None, data_dtypes=data_dtypes
            )
        return DataFrame(internal)

    def nunique(self, dropna: bool = True) -> Union[DataFrame, Series]:
        """
        Return DataFrame with number of distinct observations per group for each column.
        Parameters
        ----------
        dropna : boolean, default True
        Returns
        -------
        nunique : DataFrame or Series
        """
        if dropna:
            stat_function = lambda col: F.countDistinct(col)
        else:
            stat_function = lambda col: (
                F.countDistinct(col)
                + F.when(F.count(F.when(col.isNull(), 1).otherwise(None)) >= 1, 1).otherwise(0)
            )
        return self._reduce_for_stat_function(stat_function, only_numeric=False)

    def rolling(self, window: Any, min_periods: Optional[int] = None) -> RollingGroupby:
        """
        Return an rolling grouper, providing rolling functionality per group.
        Parameters
        ----------
        window : int, or offset
        min_periods : int, default 1
        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return RollingGroupby(self, window, min_periods=min_periods)

    def expanding(self, min_periods: int = 1) -> ExpandingGroupby:
        """
        Return an expanding grouper, providing expanding functionality per group.
        Parameters
        ----------
        min_periods : int, default 1
        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return ExpandingGroupby(self, min_periods=min_periods)

    def get_group(self, name: Any) -> Union[DataFrame, Series]:
        """
        Construct DataFrame from group with provided name.
        Parameters
        ----------
        name : object
        Returns
        -------
        group : same type as obj
        """
        groupkeys = self._groupkeys
        if not is_hashable(name):
            raise TypeError("unhashable type: '{}'".format(type(name).__name__))
        elif len(groupkeys) > 1:
            if not isinstance(name, tuple):
                raise ValueError("must supply a tuple to get_group with multiple grouping keys")
            if len(groupkeys) != len(name):
                raise ValueError(
                    "must supply a same-length tuple to get_group with multiple grouping keys"
                )
        if not is_list_like(name):
            name = [name]
        cond = F.lit(True)
        for groupkey, item in zip(groupkeys, name):
            scol = groupkey.spark.column
            cond = cond & (scol == item)
        if self._agg_columns_selected:
            internal = self._kdf._internal
            spark_frame = internal.spark_frame.select(
                internal.index_spark_columns + self._agg_columns_scols
            ).filter(cond)
            internal = internal.copy(
                spark_frame=spark_frame,
                index_spark_columns=[
                    scol_for(spark_frame, col) for col in internal.index_spark_column_names
                ],
                column_labels=[s._column_label for s in self._agg_columns],
                data_spark_columns=[
                    scol_for(spark_frame, s._internal.data_spark_column_names[0])
                    for s in self._agg_columns
                ],
                data_dtypes=[s.dtype for s in self._agg_columns],
            )
        else:
            internal = self._kdf._internal.with_filter(cond)
        if internal.spark_frame.head() is None:
            raise KeyError(name)
        return DataFrame(internal)

    def median(self, numeric_only: bool = True, accuracy: int = 10000) -> Union[DataFrame, Series]:
        """
        Compute median of groups, excluding missing values.
        Parameters
        ----------
        numeric_only : bool, default True
        accuracy : int, default 10000
        Returns
        -------
        Series or DataFrame
        """
        if not isinstance(accuracy, int):
            raise ValueError(
                "accuracy must be an integer; however, got [%s]" % type(accuracy).__name__
            )
        stat_function = lambda col: SF.percentile_approx(col, 0.5, accuracy)
        return self._reduce_for_stat_function(stat_function, only_numeric=numeric_only)

    def _reduce_for_stat_function(self, sfun: Callable, only_numeric: bool) -> DataFrame:
        agg_columns: List[Series] = self._agg_columns
        agg_columns_scols: List[Column] = self._agg_columns_scols
        groupkey_names: List[str] = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(self._groupkeys))]
        groupkey_scols: List[Column] = [s.alias(name) for s, name in zip(self._groupkeys_scols, groupkey_names)]
        sdf = self._kdf._internal.spark_frame.select(groupkey_scols + agg_columns_scols)
        data_columns: List[str] = []
        column_labels: List[Any] = []
        if len(agg_columns) > 0:
            stat_exprs: List[Any] = []
            for kser in agg_columns:
                spark_type = kser.spark.data_type
                name = kser._internal.data_spark_column_names[0]
                label = kser._column_label
                scol = scol_for(sdf, name)
                if isinstance(spark_type, DoubleType) or isinstance(spark_type, FloatType):
                    stat_exprs.append(sfun(F.nanvl(scol, F.lit(None))).alias(name))
                    data_columns.append(name)
                    column_labels.append(label)
                elif isinstance(spark_type, NumericType) or not only_numeric:
                    stat_exprs.append(sfun(scol).alias(name))
                    data_columns.append(name)
                    column_labels.append(label)
            sdf = sdf.groupby(*groupkey_names).agg(*stat_exprs)
        else:
            sdf = sdf.select(*groupkey_names).distinct()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in groupkey_names],
            index_names=[kser._column_label for kser in self._groupkeys],
            index_dtypes=[kser.dtype for kser in self._groupkeys],
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            column_label_names=self._kdf._internal.column_label_names,
        )
        kdf = DataFrame(internal)
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
        return kdf

    @staticmethod
    def _resolve_grouping_from_diff_dataframes(
        kdf: DataFrame, by: List[Any]
    ) -> Tuple[DataFrame, List[Series], Set[Tuple[Any, ...]]]:
        column_labels_level = kdf._internal.column_labels_level
        column_labels: List[Any] = []
        additional_ksers: List[Series] = []
        additional_column_labels: List[Any] = []
        tmp_column_labels: Set[Tuple[Any, ...]] = set()
        for i, col_or_s in enumerate(by):
            if isinstance(col_or_s, Series):
                if col_or_s._kdf is kdf:
                    column_labels.append(col_or_s._column_label)
                elif same_anchor(col_or_s, kdf):
                    temp_label = verify_temp_column_name(kdf, "__tmp_groupkey_{}__".format(i))
                    column_labels.append(temp_label)
                    additional_ksers.append(col_or_s.rename(temp_label))
                    additional_column_labels.append(temp_label)
                else:
                    temp_label = verify_temp_column_name(
                        kdf,
                        tuple(
                            ([""] * (column_labels_level - 1)) + ["__tmp_groupkey_{}__".format(i)]
                        ),
                    )
                    column_labels.append(temp_label)
                    tmp_column_labels.add(temp_label)
            elif isinstance(col_or_s, tuple):
                kser = kdf[col_or_s]
                if not isinstance(kser, Series):
                    raise ValueError(name_like_string(col_or_s))
                column_labels.append(col_or_s)
            else:
                raise ValueError(col_or_s)
        kdf = DataFrame(
            kdf._internal.with_new_columns(
                [kdf._kser_for(label) for label in kdf._internal.column_labels] + additional_ksers
            )
        )
        for col_or_s, label in zip(by, column_labels):
            if label in tmp_column_labels:
                kser = col_or_s
                kdf = align_diff_frames(
                    None,  # assign_columns function
                    kdf,
                    kser.rename(label),
                    fillna=False,
                    how="inner",
                    preserve_order_column=True,
                )
        tmp_column_labels |= set(additional_column_labels)
        new_by_series: List[Series] = []
        for col_or_s, label in zip(by, column_labels):
            if label in tmp_column_labels:
                kser = col_or_s
                new_by_series.append(kdf._kser_for(label).rename(kser.name))
            else:
                new_by_series.append(kdf._kser_for(label))
        return kdf, new_by_series, tmp_column_labels

    @staticmethod
    def _resolve_grouping(kdf: DataFrame, by: List[Any]) -> List[Series]:
        new_by_series: List[Series] = []
        for col_or_s in by:
            if isinstance(col_or_s, Series):
                new_by_series.append(col_or_s)
            elif isinstance(col_or_s, tuple):
                kser = kdf[col_or_s]
                if not isinstance(kser, Series):
                    raise ValueError(name_like_string(col_or_s))
                new_by_series.append(kser)
            else:
                raise ValueError(col_or_s)
        return new_by_series


class DataFrameGroupBy(GroupBy):
    @staticmethod
    def _build(
        kdf: DataFrame, by: List[Any], as_index: bool, dropna: bool
    ) -> "DataFrameGroupBy":
        if any(isinstance(col_or_s, Series) and not same_anchor(kdf, col_or_s) for col_or_s in by):
            (
                kdf,
                new_by_series,
                column_labels_to_exlcude,
            ) = GroupBy._resolve_grouping_from_diff_dataframes(kdf, by)
        else:
            new_by_series = GroupBy._resolve_grouping(kdf, by)
            column_labels_to_exlcude = set()
        return DataFrameGroupBy(
            kdf,
            new_by_series,
            as_index=as_index,
            dropna=dropna,
            column_labels_to_exlcude=column_labels_to_exlcude,
        )

    def __init__(
        self,
        kdf: DataFrame,
        by: List[Any],
        as_index: bool,
        dropna: bool,
        column_labels_to_exlcude: Set[Tuple[Any, ...]],
        agg_columns: Optional[List[Any]] = None,
    ) -> None:
        agg_columns_selected: bool = agg_columns is not None
        if agg_columns_selected:
            for label in agg_columns:  # type: ignore
                if label in column_labels_to_exlcude:
                    raise KeyError(label)
        else:
            agg_columns = [
                label
                for label in kdf._internal.column_labels
                if not any(label == key._column_label and key._kdf is kdf for key in by)
                and label not in column_labels_to_exlcude
            ]
        super().__init__(
            kdf=kdf,
            groupkeys=by,
            as_index=as_index,
            dropna=dropna,
            column_labels_to_exlcude=column_labels_to_exlcude,
            agg_columns_selected=agg_columns_selected,
            agg_columns=[kdf[label] for label in agg_columns],
        )

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeDataFrameGroupBy, item):
            property_or_func = getattr(MissingPandasLikeDataFrameGroupBy, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        return self.__getitem__(item)

    def __getitem__(self, item: Any) -> Union["DataFrameGroupBy", "SeriesGroupBy"]:
        if self._as_index and is_name_like_value(item):
            return SeriesGroupBy(
                self._kdf._kser_for(item if is_name_like_tuple(item) else (item,)),
                self._groupkeys,
                dropna=self._dropna,
            )
        else:
            if is_name_like_tuple(item):
                item = [item]
            elif is_name_like_value(item):
                item = [(item,)]
            else:
                item = [i if is_name_like_tuple(i) else (i,) for i in item]
            if not self._as_index:
                groupkey_names = set(key._column_label for key in self._groupkeys)
                for name in item:
                    if name in groupkey_names:
                        raise ValueError(
                            "cannot insert {}, already exists".format(name_like_string(name))
                        )
            return DataFrameGroupBy(
                self._kdf,
                self._groupkeys,
                as_index=self._as_index,
                dropna=self._dropna,
                column_labels_to_exlcude=self._column_labels_to_exlcude,
                agg_columns=item,
            )

    def _apply_series_op(self, op: Callable[[Any], Union[DataFrame, Series]], should_resolve: bool = False, numeric_only: bool = False) -> DataFrame:
        applied: List[Union[DataFrame, Series]] = []
        for column in self._agg_columns:
            applied.append(op(column.groupby(self._groupkeys)))
        if numeric_only:
            applied = [col for col in applied if isinstance(col.spark.data_type, NumericType)]
            if not applied:
                raise DataError("No numeric types to aggregate")
        internal = self._kdf._internal.with_new_columns(applied, keep_order=False)
        if should_resolve:
            internal = internal.resolved_copy
        return DataFrame(internal)

    def describe(self) -> DataFrame:
        """
        Generate descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding
        ``NaN`` values.
        Returns
        -------
        DataFrame
        """
        for col in self._agg_columns:
            if isinstance(col.spark.data_type, StringType):
                raise NotImplementedError(
                    "DataFrameGroupBy.describe() doesn't support for string type for now"
                )
        kdf = self.aggregate(["count", "mean", "std", "min", "quartiles", "max"])
        sdf = kdf._internal.spark_frame
        agg_column_labels: List[Any] = [col._column_label for col in self._agg_columns]
        formatted_percentiles: List[str] = ["25%", "50%", "75%"]
        for label in agg_column_labels:
            quartiles_col = name_like_string(tuple(list(label) + ["quartiles"]))
            for i, percentile in enumerate(formatted_percentiles):
                sdf = sdf.withColumn(
                    name_like_string(tuple(list(label) + [percentile])),
                    scol_for(sdf, quartiles_col)[i],
                )
            sdf = sdf.drop(quartiles_col)
        stats: List[str] = ["count", "mean", "std", "min"] + formatted_percentiles + ["max"]
        column_labels: List[Tuple[Any, ...]] = [tuple(list(label) + [s]) for label, s in product(agg_column_labels, stats)]
        data_columns = list(map(name_like_string, column_labels))
        internal = kdf._internal.copy(
            spark_frame=sdf,
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            data_dtypes=None,
        )
        return DataFrame(internal).astype("float64")


class SeriesGroupBy(GroupBy):
    @staticmethod
    def _build(
        kser: Series, by: List[Union[Series, Tuple[Any, ...]]], as_index: bool, dropna: bool
    ) -> "SeriesGroupBy":
        if any(isinstance(col_or_s, Series) and not same_anchor(kser, col_or_s) for col_or_s in by):
            kdf, new_by_series, _ = GroupBy._resolve_grouping_from_diff_dataframes(kser.to_frame(), by)
            return SeriesGroupBy(
                first_series(kdf).rename(kser.name), new_by_series, as_index=as_index, dropna=dropna
            )
        else:
            new_by_series = GroupBy._resolve_grouping(kser._kdf, by)
            return SeriesGroupBy(kser, new_by_series, as_index=as_index, dropna=dropna)

    def __init__(self, kser: Series, by: List[Any], as_index: bool = True, dropna: bool = True) -> None:
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        super().__init__(
            kdf=kser._kdf,
            groupkeys=by,
            as_index=True,
            dropna=dropna,
            column_labels_to_exlcude=set(),
            agg_columns_selected=True,
            agg_columns=[kser],
        )
        self._kser = kser

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeSeriesGroupBy, item):
            property_or_func = getattr(MissingPandasLikeSeriesGroupBy, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def _apply_series_op(self, op: Callable[[Any], Union[DataFrame, Series]], should_resolve: bool = False, numeric_only: bool = False) -> Union[DataFrame, Series]:
        if numeric_only and not isinstance(self._agg_columns[0].spark.data_type, NumericType):
            raise DataError("No numeric types to aggregate")
        kser = op(self)
        if should_resolve:
            internal = kser._internal.resolved_copy
            return first_series(DataFrame(internal))
        else:
            return kser

    def _reduce_for_stat_function(self, sfun: Callable, only_numeric: bool) -> Series:
        return first_series(super()._reduce_for_stat_function(sfun, only_numeric))

    def agg(self, *args: Any, **kwargs: Any) -> None:
        return MissingPandasLikeSeriesGroupBy.agg(self, *args, **kwargs)

    def aggregate(self, *args: Any, **kwargs: Any) -> None:
        return MissingPandasLikeSeriesGroupBy.aggregate(self, *args, **kwargs)

    def transform(self, func: Callable, *args: Any, **kwargs: Any) -> Series:
        return first_series(super().transform(func, *args, **kwargs)).rename(self._kser.name)

    transform.__doc__ = GroupBy.transform.__doc__

    def idxmin(self, skipna: bool = True) -> Series:
        return first_series(super().idxmin(skipna))

    idxmin.__doc__ = GroupBy.idxmin.__doc__

    def idxmax(self, skipna: bool = True) -> Series:
        return first_series(super().idxmax(skipna))

    idxmax.__doc__ = GroupBy.idxmax.__doc__

    def head(self, n: int = 5) -> Series:
        return first_series(super().head(n)).rename(self._kser.name)

    head.__doc__ = GroupBy.head.__doc__

    def tail(self, n: int = 5) -> Series:
        return first_series(super().tail(n)).rename(self._kser.name)

    tail.__doc__ = GroupBy.tail.__doc__

    def size(self) -> Series:
        return super().size().rename(self._kser.name)

    size.__doc__ = GroupBy.size.__doc__

    def get_group(self, name: Any) -> Series:
        return first_series(super().get_group(name))

    get_group.__doc__ = GroupBy.get_group.__doc__

    def nsmallest(self, n: int = 5) -> Series:
        if self._kser._internal.index_level > 1:
            raise ValueError("nsmallest do not support multi-index now")
        groupkey_col_names: List[str] = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(self._groupkeys))]
        sdf = self._kser._internal.spark_frame.select(
            [scol.alias(name) for scol, name in zip(self._groupkeys_scols, groupkey_col_names)]
            + [
                scol.alias(SPARK_INDEX_NAME_FORMAT(i + len(self._groupkeys)))
                for i, scol in enumerate(self._kser._internal.index_spark_columns)
            ]
            + [self._kser.spark.column]
            + [NATURAL_ORDER_COLUMN_NAME]
        )
        window = Window.partitionBy(groupkey_col_names).orderBy(
            scol_for(sdf, self._kser._internal.data_spark_column_names[0]).asc(),
            NATURAL_ORDER_COLUMN_NAME,
        )
        temp_rank_column: str = verify_temp_column_name(sdf, "__rank__")
        sdf = (
            sdf.withColumn(temp_rank_column, F.row_number().over(window))
            .filter(F.col(temp_rank_column) <= n)
            .drop(temp_rank_column)
        ).drop(NATURAL_ORDER_COLUMN_NAME)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=(
                [scol_for(sdf, col) for col in groupkey_col_names]
                + [
                    scol_for(sdf, SPARK_INDEX_NAME_FORMAT(i + len(self._groupkeys)))
                    for i in range(self._kdf._internal.index_level)
                ]
            ),
            index_names=(
                [kser._column_label for kser in self._groupkeys] + self._kdf._internal.index_names
            ),
            index_dtypes=(
                [kser.dtype for kser in self._groupkeys] + self._kdf._internal.index_dtypes
            ),
            column_labels=[self._kser._column_label],
            data_spark_columns=[scol_for(sdf, self._kser._internal.data_spark_column_names[0])],
            data_dtypes=[self._kser.dtype],
        )
        return first_series(DataFrame(internal))

    def nlargest(self, n: int = 5) -> Series:
        if self._kser._internal.index_level > 1:
            raise ValueError("nlargest do not support multi-index now")
        groupkey_col_names: List[str] = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(self._groupkeys))]
        sdf = self._kser._internal.spark_frame.select(
            [scol.alias(name) for scol, name in zip(self._groupkeys_scols, groupkey_col_names)]
            + [
                scol.alias(SPARK_INDEX_NAME_FORMAT(i + len(self._groupkeys)))
                for i, scol in enumerate(self._kser._internal.index_spark_columns)
            ]
            + [self._kser.spark.column]
            + [NATURAL_ORDER_COLUMN_NAME]
        )
        window = Window.partitionBy(groupkey_col_names).orderBy(
            scol_for(sdf, self._kser._internal.data_spark_column_names[0]).desc(),
            NATURAL_ORDER_COLUMN_NAME,
        )
        temp_rank_column: str = verify_temp_column_name(sdf, "__rank__")
        sdf = (
            sdf.withColumn(temp_rank_column, F.row_number().over(window))
            .filter(F.col(temp_rank_column) <= n)
            .drop(temp_rank_column)
        ).drop(NATURAL_ORDER_COLUMN_NAME)
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=(
                [scol_for(sdf, col) for col in groupkey_col_names]
                + [
                    scol_for(sdf, SPARK_INDEX_NAME_FORMAT(i + len(self._groupkeys)))
                    for i in range(self._kdf._internal.index_level)
                ]
            ),
            index_names=(
                [kser._column_label for kser in self._groupkeys] + self._kdf._internal.index_names
            ),
            index_dtypes=(
                [kser.dtype for kser in self._groupkeys] + self._kdf._internal.index_dtypes
            ),
            column_labels=[self._kser._column_label],
            data_spark_columns=[scol_for(sdf, self._kser._internal.data_spark_column_names[0])],
            data_dtypes=[self._kser.dtype],
        )
        return first_series(DataFrame(internal))

    def value_counts(self, sort: Optional[bool] = None, ascending: Optional[bool] = None, dropna: bool = True) -> Series:
        groupkeys = self._groupkeys + self._agg_columns
        groupkey_names: List[str] = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(groupkeys))]
        groupkey_cols: List[Column] = [s.spark.column.alias(name) for s, name in zip(groupkeys, groupkey_names)]
        sdf = self._kdf._internal.spark_frame
        agg_column: str = self._agg_columns[0]._internal.data_spark_column_names[0]
        sdf = sdf.groupby(*groupkey_cols).count().withColumnRenamed("count", agg_column)
        if sort:
            if ascending:
                sdf = sdf.orderBy(scol_for(sdf, agg_column).asc())
            else:
                sdf = sdf.orderBy(scol_for(sdf, agg_column).desc())
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in groupkey_names],
            index_names=[kser._column_label for kser in groupkeys],
            index_dtypes=[kser.dtype for kser in groupkeys],
            column_labels=[self._agg_columns[0]._column_label],
            data_spark_columns=[scol_for(sdf, agg_column)],
        )
        return first_series(DataFrame(internal))

    def unique(self) -> Series:
        return self._reduce_for_stat_function(F.collect_set, only_numeric=False)


def is_multi_agg_with_relabel(**kwargs: Any) -> bool:
    if not kwargs:
        return False
    return all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values())


def normalize_keyword_aggregation(kwargs: Dict[Any, Any]) -> Tuple[OrderedDict[Any, List[Any]], Tuple[Any, ...], List[Tuple[Any, Any]]]:
    PY36 = sys.version_info >= (3, 6)
    if not PY36:
        kwargs = OrderedDict(sorted(kwargs.items()))
    aggspec: OrderedDict[Any, List[Any]] = OrderedDict()
    order: List[Tuple[Any, Any]] = []
    columns, pairs = list(zip(*kwargs.items()))
    for column, aggfunc in pairs:
        if column in aggspec:
            aggspec[column].append(aggfunc)
        else:
            aggspec[column] = [aggfunc]
        order.append((column, aggfunc))
    if isinstance(order[0][0], tuple):
        order = [(*levs, method) for levs, method in order]
    return aggspec, columns, order