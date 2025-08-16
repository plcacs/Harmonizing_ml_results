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
Commonly used utils in Koalas.
"""

import functools
from collections import OrderedDict
from contextlib import contextmanager
from distutils.version import LooseVersion
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Iterator, Set, cast
import warnings

import pyarrow
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
from pandas.api.types import is_list_like

# For running doctests and reference resolution in PyCharm.
from databricks import koalas as ks  # noqa: F401
from databricks.koalas.typedef.typehints import (
    as_spark_type,
    extension_dtypes,
    spark_type_to_pandas_dtype,
)

if TYPE_CHECKING:
    # This is required in old Python 3.5 to prevent circular reference.
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import InternalFrame
    from pyspark.sql import Column


ERROR_MESSAGE_CANNOT_COMBINE = (
    "Cannot combine the series or dataframe because it comes from a different dataframe. "
    "In order to allow this operation, enable 'compute.ops_on_diff_frames' option."
)


if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
    SPARK_CONF_ARROW_ENABLED = "spark.sql.execution.arrow.enabled"
else:
    SPARK_CONF_ARROW_ENABLED = "spark.sql.execution.arrow.pyspark.enabled"


def same_anchor(
    this: Union["DataFrame", "IndexOpsMixin", "InternalFrame"],
    that: Union["DataFrame", "IndexOpsMixin", "InternalFrame"],
) -> bool:
    """
    Check if the anchors of the given DataFrame or Series are the same or not.
    """
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import InternalFrame

    if isinstance(this, InternalFrame):
        this_internal = this
    else:
        assert isinstance(this, (DataFrame, IndexOpsMixin)), type(this)
        this_internal = this._internal

    if isinstance(that, InternalFrame):
        that_internal = that
    else:
        assert isinstance(that, (DataFrame, IndexOpsMixin)), type(that)
        that_internal = that._internal

    return (
        this_internal.spark_frame is that_internal.spark_frame
        and this_internal.index_level == that_internal.index_level
        and all(
            this_scol._jc.equals(that_scol._jc)
            for this_scol, that_scol in zip(
                this_internal.index_spark_columns, that_internal.index_spark_columns
            )
        )
    )


def combine_frames(
    this: "DataFrame",
    *args: Union["DataFrame", "Series"],
    how: str = "full",
    preserve_order_column: bool = False
) -> "DataFrame":
    """
    This method combines `this` DataFrame with a different `that` DataFrame or
    Series from a different DataFrame.

    It returns a DataFrame that has prefix `this_` and `that_` to distinct
    the columns names from both DataFrames

    It internally performs a join operation which can be expensive in general.
    So, if `compute.ops_on_diff_frames` option is False,
    this method throws an exception.
    """
    from databricks.koalas.config import get_option
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.internal import (
        InternalFrame,
        HIDDEN_COLUMNS,
        NATURAL_ORDER_COLUMN_NAME,
        SPARK_INDEX_NAME_FORMAT,
    )
    from databricks.koalas.series import Series

    if all(isinstance(arg, Series) for arg in args):
        assert all(
            same_anchor(arg, args[0]) for arg in args
        ), "Currently only one different DataFrame (from given Series) is supported"
        assert not same_anchor(this, args[0]), "We don't need to combine. All series is in this."
        that = args[0]._kdf[list(args)]
    elif len(args) == 1 and isinstance(args[0], DataFrame):
        assert isinstance(args[0], DataFrame)
        assert not same_anchor(
            this, args[0]
        ), "We don't need to combine. `this` and `that` are same."
        that = args[0]
    else:
        raise AssertionError("args should be single DataFrame or " "single/multiple Series")

    if get_option("compute.ops_on_diff_frames"):

        def resolve(internal: InternalFrame, side: str) -> InternalFrame:
            rename = lambda col: "__{}_{}".format(side, col)
            internal = internal.resolved_copy
            sdf = internal.spark_frame
            sdf = internal.spark_frame.select(
                [
                    scol_for(sdf, col).alias(rename(col))
                    for col in sdf.columns
                    if col not in HIDDEN_COLUMNS
                ]
                + list(HIDDEN_COLUMNS)
            )
            return internal.copy(
                spark_frame=sdf,
                index_spark_columns=[
                    scol_for(sdf, rename(col)) for col in internal.index_spark_column_names
                ],
                data_spark_columns=[
                    scol_for(sdf, rename(col)) for col in internal.data_spark_column_names
                ],
            )

        this_internal = resolve(this._internal, "this")
        that_internal = resolve(that._internal, "that")

        this_index_map = list(
            zip(
                this_internal.index_spark_column_names,
                this_internal.index_names,
                this_internal.index_dtypes,
            )
        )
        that_index_map = list(
            zip(
                that_internal.index_spark_column_names,
                that_internal.index_names,
                that_internal.index_dtypes,
            )
        )
        assert len(this_index_map) == len(that_index_map)

        join_scols: List[Column] = []
        merged_index_scols: List[Column] = []

        # Note that the order of each element in index_map is guaranteed according to the index
        # level.
        this_and_that_index_map = list(zip(this_index_map, that_index_map))

        this_sdf = this_internal.spark_frame.alias("this")
        that_sdf = that_internal.spark_frame.alias("that")

        # If the same named index is found, that's used.
        index_column_names: List[str] = []
        index_use_extension_dtypes: List[bool] = []
        for (
            i,
            ((this_column, this_name, this_dtype), (that_column, that_name, that_dtype)),
        ) in enumerate(this_and_that_index_map):
            if this_name == that_name:
                # We should merge the Spark columns into one
                # to mimic pandas' behavior.
                this_scol = scol_for(this_sdf, this_column)
                that_scol = scol_for(that_sdf, that_column)
                join_scol = this_scol == that_scol
                join_scols.append(join_scol)

                column_name = SPARK_INDEX_NAME_FORMAT(i)
                index_column_names.append(column_name)
                index_use_extension_dtypes.append(
                    any(isinstance(dtype, extension_dtypes) for dtype in [this_dtype, that_dtype])
                merged_index_scols.append(
                    F.when(this_scol.isNotNull(), this_scol).otherwise(that_scol).alias(column_name)
            else:
                raise ValueError("Index names must be exactly matched currently.")

        assert len(join_scols) > 0, "cannot join with no overlapping index names"

        joined_df = this_sdf.join(that_sdf, on=join_scols, how=how)

        if preserve_order_column:
            order_column = [scol_for(this_sdf, NATURAL_ORDER_COLUMN_NAME)]
        else:
            order_column = []

        joined_df = joined_df.select(
            merged_index_scols
            + [
                scol_for(this_sdf, this_internal.spark_column_name_for(label))
                for label in this_internal.column_labels
            ]
            + [
                scol_for(that_sdf, that_internal.spark_column_name_for(label))
                for label in that_internal.column_labels
            ]
            + order_column
        )

        index_spark_columns = [scol_for(joined_df, col) for col in index_column_names]
        index_dtypes = [
            spark_type_to_pandas_dtype(field.dataType, use_extension_dtypes=use_extension_dtypes)
            for field, use_extension_dtypes in zip(
                joined_df.select(index_spark_columns).schema, index_use_extension_dtypes
            )
        ]

        index_columns = set(index_column_names)
        new_data_columns = [
            col
            for col in joined_df.columns
            if col not in index_columns and col != NATURAL_ORDER_COLUMN_NAME
        ]
        data_dtypes = this_internal.data_dtypes + that_internal.data_dtypes

        level = max(this_internal.column_labels_level, that_internal.column_labels_level)

        def fill_label(label: Optional[Tuple]) -> List[Optional[str]]:
            if label is None:
                return ([""] * (level - 1)) + [None]
            else:
                return ([""] * (level - len(label))) + list(label)

        column_labels = [
            tuple(["this"] + fill_label(label)) for label in this_internal.column_labels
        ] + [tuple(["that"] + fill_label(label)) for label in that_internal.column_labels]
        column_label_names = (
            [None] * (1 + level - this_internal.column_labels_level)
        ) + this_internal.column_label_names
        return DataFrame(
            InternalFrame(
                spark_frame=joined_df,
                index_spark_columns=index_spark_columns,
                index_names=this_internal.index_names,
                index_dtypes=index_dtypes,
                column_labels=column_labels,
                data_spark_columns=[scol_for(joined_df, col) for col in new_data_columns],
                data_dtypes=data_dtypes,
                column_label_names=column_label_names,
            )
        )
    else:
        raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)


def align_diff_frames(
    resolve_func: Callable[..., Iterator[Tuple["Series", Tuple]]],
    this: "DataFrame",
    that: "DataFrame",
    fillna: bool = True,
    how: str = "full",
    preserve_order_column: bool = False,
) -> "DataFrame":
    """
    This method aligns two different DataFrames with a given `func`. Columns are resolved and
    handled within the given `func`.
    To use this, `compute.ops_on_diff_frames` should be True, for now.

    :param resolve_func: Takes aligned (joined) DataFrame, the column of the current DataFrame, and
        the column of another DataFrame. It returns an iterable that produces Series.

        >>> from databricks.koalas.config import set_option, reset_option
        >>>
        >>> set_option("compute.ops_on_diff_frames", True)
        >>>
        >>> kdf1 = ks.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1]})
        >>> kdf2 = ks.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1]})
        >>>
        >>> def func(kdf, this_column_labels, that_column_labels):
        ...    kdf  # conceptually this is A + B.
        ...
        ...    # Within this function, Series from A or B can be performed against `kdf`.
        ...    this_label = this_column_labels[0]  # this is ('a',) from kdf1.
        ...    that_label = that_column_labels[0]  # this is ('a',) from kdf2.
        ...    new_series = (kdf[this_label] - kdf[that_label]).rename(str(this_label))
        ...
        ...    # This new series will be placed in new DataFrame.
        ...    yield (new_series, this_label)
        >>>
        >>>
        >>> align_diff_frames(func, kdf1, kdf2).sort_index()
           a
        0  0
        1  0
        2  0
        3  0
        4  0
        5  0
        6  0
        7  0
        8  0
        >>> reset_option("compute.ops_on_diff_frames")

    :param this: a DataFrame to align
    :param that: another DataFrame to align
    :param fillna: If True, it fills missing values in non-common columns in both `this` and `that`.
        Otherwise, it returns as are.
    :param how: join way. In addition, it affects how `resolve_func` resolves the column conflict.
        - full: `resolve_func` should resolve only common columns from 'this' and 'that' DataFrames.
            For instance, if 'this' has columns A, B, C and that has B, C, D, `this_columns` and
            'that_columns' in this function are B, C and B, C.
        - left: `resolve_func` should resolve columns including that columns.
            For instance, if 'this' has columns A, B, C and that has B, C, D, `this_columns` is
            B, C but `that_columns` are B, C, D.
        - inner: Same as 'full' mode; however, internally performs inner join instead.
    :return: Aligned DataFrame
    """
    from databricks.koalas.frame import DataFrame

    assert how == "full" or how == "left" or how == "inner"

    this_column_labels = this._internal.column_labels
    that_column_labels = that._internal.column_labels
    common_column_labels = set(this_column_labels).intersection(that_column_labels)

    # 1. Perform the join given two dataframes.
    combined = combine_frames(this, that, how=how, preserve_order_column=preserve_order_column)

    # 2. Apply the given function to transform the columns in a batch and keep the new columns.
    combined_column_labels = combined._internal.column_labels

    that_columns_to_apply: List[Tuple] = []
    this_columns_to_apply: List[Tuple] = []
    additional_that_columns: List[Tuple] = []
    columns_to_keep: List[Column] = []
    column_labels_to_keep: List[Tuple] = []

    for combined_label in combined_column_labels:
        for common_label in common_column_labels:
            if combined_label == tuple(["this", *common_label]):
                this_columns_to_apply.append(combined_label)
                break
            elif combined_label == tuple(["that", *common_label]):
                that_columns_to_apply.append(combined_label)
                break
        else:
            if how == "left" and combined_label in [
                tuple(["that", *label]) for label in that_column_labels
            ]:
                # In this case, we will drop `that_columns` in `columns_to_keep` but passes
                # it later to `func`. `func` should resolve it.
                # Note that adding this into a separate list (`additional_that_columns`)
                # is intentional so that `this_columns` and `that_columns` can be paired.
                additional_that_columns.append(combined_label)
            elif fillna:
                columns_to_keep.append(F.lit(None).cast(DoubleType()).alias(str(combined_label)))
                column_labels_to_keep.append(combined_label)
            else:
                columns_to_keep.append(combined._kser_for(combined_label))
                column_labels_to_keep.append(combined_label)

    that_columns_to_apply += additional_that_columns

    # Should extract columns to apply and do it in a batch in case
    # it adds new columns for example.
    if len(this_columns_to_apply) > 0 or len(that_columns_to_apply) > 0:
        kser_set, column_labels_applied = zip(
            *resolve_func(combined, this_columns_to_apply, that_columns_to_apply)
        )
        columns_applied = list(kser_set)
        column_labels_applied = list(column_labels_applied)
    else:
        columns_applied = []
        column_labels_applied = []

    applied = DataFrame(
        combined._internal.with_new_columns(
            columns_applied + columns_to_keep,
            column_labels=column_labels_applied + column_labels_to_keep,
        )
    )  # type: DataFrame

    # 3. Restore the names back and deduplicate columns.
    this_labels = OrderedDict()
    # Add columns in an order of its original frame.
    for this_label in this_column_labels:
        for new_label in applied._internal.column_labels:
            if new_label[1:] not in this_labels and this_label == new_label[1:]:
                this_labels[new_label[1:]] = new_label

    # After that, we will add the rest columns.
    other_labels = OrderedDict()
    for new_label in applied._internal.column_labels:
        if new_label[1:] not in this_labels:
            other_labels[new_label[1:]] = new_label

    kdf = applied[list(this_labels.values()) + list(other_labels.values())]
    kdf.columns = kdf.columns.droplevel()
    return kdf


def is_testing() -> bool