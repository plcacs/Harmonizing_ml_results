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
Base and utility classes for Koalas objects.
"""
from abc import ABCMeta, abstractmethod
import datetime
from functools import wraps, partial
from itertools import chain
from typing import Any, Callable, Optional, Tuple, Union, cast, TYPE_CHECKING, List, Dict, Set
import warnings

import numpy as np
import pandas as pd  # noqa: F401
from pandas.api.types import is_list_like, CategoricalDtype
from pyspark import sql as spark
from pyspark.sql import functions as F, Window, Column
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegralType,
    LongType,
    NumericType,
    StringType,
    TimestampType,
)

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas import numpy_compat
from databricks.koalas.config import get_option, option_context
from databricks.koalas.internal import (
    InternalFrame,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_DEFAULT_INDEX_NAME,
)
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkIndexOpsMethods
from databricks.koalas.typedef import (
    Dtype,
    as_spark_type,
    extension_dtypes,
    koalas_dtype,
    spark_type_to_pandas_dtype,
)
from databricks.koalas.utils import (
    combine_frames,
    same_anchor,
    scol_for,
    validate_axis,
    ERROR_MESSAGE_CANNOT_COMBINE,
)
from databricks.koalas.frame import DataFrame

if TYPE_CHECKING:
    from databricks.koalas.indexes import Index
    from databricks.koalas.series import Series


def should_alignment_for_column_op(self: "IndexOpsMixin", other: "IndexOpsMixin") -> bool:
    from databricks.koalas.series import Series

    if isinstance(self, Series) and isinstance(other, Series):
        return not same_anchor(self, other)
    else:
        return self._internal.spark_frame is not other._internal.spark_frame


def align_diff_index_ops(func: Callable, this_index_ops: "IndexOpsMixin", *args: Any) -> "IndexOpsMixin":
    """
    Align the `IndexOpsMixin` objects and apply the function.

    Parameters
    ----------
    func : The function to apply
    this_index_ops : IndexOpsMixin
        A base `IndexOpsMixin` object
    args : list of other arguments including other `IndexOpsMixin` objects

    Returns
    -------
    `Index` if all `this_index_ops` and arguments are `Index`; otherwise `Series`
    """
    from databricks.koalas.indexes import Index
    from databricks.koalas.series import Series, first_series

    cols = [arg for arg in args if isinstance(arg, IndexOpsMixin)]

    if isinstance(this_index_ops, Series) and all(isinstance(col, Series) for col in cols):
        combined = combine_frames(
            this_index_ops.to_frame(),
            *[cast(Series, col).rename(i) for i, col in enumerate(cols)],
            how="full"
        )

        return column_op(func)(
            combined["this"]._kser_for(combined["this"]._internal.column_labels[0]),
            *[
                combined["that"]._kser_for(label)
                for label in combined["that"]._internal.column_labels
            ]
        ).rename(this_index_ops.name)
    else:
        # This could cause as many counts, reset_index calls, joins for combining
        # as the number of `Index`s in `args`. So far it's fine since we can assume the ops
        # only work between at most two `Index`s. We might need to fix it in the future.

        self_len = len(this_index_ops)
        if any(len(col) != self_len for col in args if isinstance(col, IndexOpsMixin)):
            raise ValueError("operands could not be broadcast together with shapes")

        with option_context("compute.default_index_type", "distributed-sequence"):
            if isinstance(this_index_ops, Index) and all(isinstance(col, Index) for col in cols):
                return Index(
                    column_op(func)(
                        this_index_ops.to_series().reset_index(drop=True),
                        *[
                            arg.to_series().reset_index(drop=True)
                            if isinstance(arg, Index)
                            else arg
                            for arg in args
                        ]
                    ).sort_index(),
                    name=this_index_ops.name,
                )
            elif isinstance(this_index_ops, Series):
                this = this_index_ops.reset_index()
                that = [
                    cast(Series, col.to_series() if isinstance(col, Index) else col)
                    .rename(i)
                    .reset_index(drop=True)
                    for i, col in enumerate(cols)
                ]

                combined = combine_frames(this, *that, how="full").sort_index()
                combined = combined.set_index(
                    combined._internal.column_labels[: this_index_ops._internal.index_level]
                )
                combined.index.names = this_index_ops._internal.index_names

                return column_op(func)(
                    first_series(combined["this"]),
                    *[
                        combined["that"]._kser_for(label)
                        for label in combined["that"]._internal.column_labels
                    ]
                ).rename(this_index_ops.name)
            else:
                this = cast(Index, this_index_ops).to_frame().reset_index(drop=True)

                that_series = next(col for col in cols if isinstance(col, Series))
                that_frame = that_series._kdf[
                    [
                        cast(Series, col.to_series() if isinstance(col, Index) else col).rename(i)
                        for i, col in enumerate(cols)
                    ]
                ]

                combined = combine_frames(this, that_frame.reset_index()).sort_index()

                self_index = (
                    combined["this"].set_index(combined["this"]._internal.column_labels).index
                )

                other = combined["that"].set_index(
                    combined["that"]._internal.column_labels[: that_series._internal.index_level]
                )
                other.index.names = that_series._internal.index_names

                return column_op(func)(
                    self_index,
                    *[
                        other._kser_for(label)
                        for label, col in zip(other._internal.column_labels, cols)
                    ]
                ).rename(that_series.name)


def booleanize_null(scol: Column, f: Callable) -> Column:
    """
    Booleanize Null in Spark Column
    """
    comp_ops = [
        getattr(Column, "__{}__".format(comp_op))
        for comp_op in ["eq", "ne", "lt", "le", "ge", "gt"]
    ]

    if f in comp_ops:
        # if `f` is "!=", fill null with True otherwise False
        filler = f == Column.__ne__
        scol = F.when(scol.isNull(), filler).otherwise(scol)

    return scol


def column_op(f: Callable) -> Callable:
    """
    A decorator that wraps APIs taking/returning Spark Column so that Koalas Series can be
    supported too. If this decorator is used for the `f` function that takes Spark Column and
    returns Spark Column, decorated `f` takes Koalas Series as well and returns Koalas
    Series.

    :param f: a function that takes Spark Column and returns Spark Column.
    :param self: Koalas Series
    :param args: arguments that the function `f` takes.
    """

    @wraps(f)
    def wrapper(self: "IndexOpsMixin", *args: Any) -> "IndexOpsMixin":
        from databricks.koalas.series import Series

        # It is possible for the function `f` takes other arguments than Spark Column.
        # To cover this case, explicitly check if the argument is Koalas Series and
        # extract Spark Column. For other arguments, they are used as are.
        cols = [arg for arg in args if isinstance(arg, IndexOpsMixin)]

        if all(not should_alignment_for_column_op(self, col) for col in cols):
            # Same DataFrame anchors
            args = [arg.spark.column if isinstance(arg, IndexOpsMixin) else arg for arg in args]
            scol = f(self.spark.column, *args)

            spark_type = self._internal.spark_frame.select(scol).schema[0].dataType
            use_extension_dtypes = any(
                isinstance(col.dtype, extension_dtypes) for col in [self] + cols
            )
            dtype = spark_type_to_pandas_dtype(
                spark_type, use_extension_dtypes=use_extension_dtypes
            )

            if not isinstance(dtype, extension_dtypes):
                scol = booleanize_null(scol, f)

            if isinstance(self, Series) or not any(isinstance(col, Series) for col in cols):
                index_ops = self._with_new_scol(scol, dtype=dtype)
            else:
                kser = next(col for col in cols if isinstance(col, Series))
                index_ops = kser._with_new_scol(scol, dtype=dtype)
        elif get_option("compute.ops_on_diff_frames"):
            index_ops = align_diff_index_ops(f, self, *args)
        else:
            raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)

        if not all(self.name == col.name for col in cols):
            index_ops = index_ops.rename(None)

        return index_ops

    return wrapper


def numpy_column_op(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(self: "IndexOpsMixin", *args: Any) -> "IndexOpsMixin":
        # PySpark does not support NumPy type out of the box. For now, we convert NumPy types
        # into some primitive types understandable in PySpark.
        new_args = []
        for arg in args:
            # TODO: This is a quick hack to support NumPy type. We should revisit this.
            if isinstance(self.spark.data_type, LongType) and isinstance(arg, np.timedelta64):
                new_args.append(float(arg / np.timedelta64(1, "s")))
            else:
                new_args.append(arg)
        return column_op(f)(self, *new_args)

    return wrapper


class IndexOpsMixin(object, metaclass=ABCMeta):
    """common ops mixin to support a unified interface / docs for Series / Index

    Assuming there are following attributes or properties and function.
    """

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @property
    @abstractmethod
    def _kdf(self) -> DataFrame:
        pass

    @abstractmethod
    def _with_new_scol(self, scol: spark.Column, *, dtype: Optional[Dtype] = None) -> "IndexOpsMixin":
        pass

    @property
    @abstractmethod
    def _column_label(self) -> Tuple:
        pass

    @property
    @abstractmethod
    def spark(self) -> SparkIndexOpsMethods:
        pass

    @property
    def spark_column(self) -> Column:
        warnings.warn(
            "Series.spark_column is deprecated as of Series.spark.column. "
            "Please use the API instead.",
            FutureWarning,
        )
        return self.spark.column

    spark_column.__doc__ = SparkIndexOpsMethods.column.__doc__

    # arithmetic operators
    __neg__ = column_op(Column.__neg__)

    def __add__(self, other: Any) -> Union["Series", "Index"]:
        if not isinstance(self.spark.data_type, StringType) and (
            (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("string addition can only be applied to string series or literals.")

        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError("addition can not be applied to date times.")

        if isinstance(self.spark.data_type, StringType):
            # Concatenate string columns
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType):
                return column_op(F.concat)(self, other)
            # Handle df['col'] + 'literal'
            elif isinstance(other, str):
                return column_op(F.concat)(self, F.lit(other))
            else:
                raise TypeError("string addition can only be applied to string series or literals.")
        else:
            return column_op(Column.__add__)(self, other)

    def __sub__(self, other: Any) -> Union["Series", "Index"]:
        if (
            isinstance(self.spark.data_type, StringType)
            or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("substraction can not be applied to string series or literals.")

        if isinstance(self.spark.data_type, TimestampType):
            # Note that timestamp subtraction casts arguments to integer. This is to mimic pandas's
            # behaviors. pandas returns 'timedelta64[ns]' from 'datetime64[ns]'s subtraction.
            msg = (
                "Note that there is a behavior difference of timestamp subtraction. "
                "The timestamp subtraction returns an integer in seconds, "
                "whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, IndexOpsMixin) and isinstance(
                other.spark.data_type, TimestampType
            ):
                warnings.warn(msg, UserWarning)
                return self.astype("long") - other.astype("long")
            elif isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return self.astype("long") - F.lit(other).cast(as_spark_type("long"))
            else:
                raise TypeError("datetime subtraction can only be applied to datetime series.")
        elif isinstance(self.spark.data_type, DateType):
            # Note that date subtraction casts arguments to integer. This is to mimic pandas's
            # behaviors. pandas returns 'timedelta64[ns]' in days from date's subtraction.
            msg = (
                "Note that there is a behavior difference of date subtraction. "
                "The date subtraction returns an integer in days, "
                "whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, DateType):
                warnings.warn(msg, UserWarning)
                return column_op(F.datediff)(self, other).astype("long")
            elif isinstance(other, datetime.date) and not isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return column_op(F.datediff)(self, F.lit(other)).astype("long")
            else:
                raise TypeError("date subtraction can only be applied to date series.")
        return column_op(Column.__sub__)(self, other)

    def __mul__(self, other: Any) -> Union["Series", "Index"]:
        if isinstance(other, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError("multiplication can not be applied to date times.")

        if (
            isinstance(self.spark.data_type, IntegralType)
            and isinstance(other, IndexOpsMixin)
            and isinstance(other.spark.data_type, StringType)
        ):
            return column_op(SF.repeat)(other, self)

        if isinstance(self.spark.data_type, StringType):
            if (
                isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, IntegralType)
            ) or isinstance(other, int):
                return column_op(SF.repeat)(self, other)
            else:
                raise TypeError(
                    "a string series can only be multiplied to an int series or literal"
                )

        return column_op(Column.__mul__)(self, other)

    def __truediv__(self, other: Any) -> Union["Series", "Index"]:
        """
        __truediv__ has different behaviour between pandas and PySpark for several cases.
        1. When divide np.inf by zero, PySpark returns null whereas pandas returns np.inf
        2. When divide positive number by zero, PySpark returns null whereas pandas returns np.inf
        3. When divide -np.inf by zero, PySpark returns null whereas pandas returns -np.inf
        4. When divide negative number by zero, PySpark returns null whereas pandas returns -np.inf

        +-------------------------------------------+
        | dividend (divisor: 0) | PySpark |  pandas |
        |-----------------------|---------|---------|
        |         np.inf        |   null  |  np.inf |
        |        -np.inf        |   null  | -np.inf |
        |           10          |   null  |  np.inf |
        |          -10          |   null  | -np.inf |
        +-----------------------|---------|---------+
        """

        if (
            isinstance(self.spark.data_type, StringType)
            or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("division can not be applied on string series or literals.")

        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError("division can not be applied to date times.")

        def truediv(left: Column, right: Any) -> Column:
            return F.when(F.lit(right != 0) | F.lit(right).isNull(), left.__div__(right)).otherwise(
                F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left).otherwise(
                    F.lit(np.inf).__div