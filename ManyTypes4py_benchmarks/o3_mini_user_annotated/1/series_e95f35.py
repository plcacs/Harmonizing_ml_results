#!/usr/bin/env python3
# type: ignore
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union, overload
import datetime
import re
import inspect
import sys
import warnings
from collections.abc import Mapping
from distutils.version import LooseVersion
from functools import partial, wraps, reduce

import numpy as np
import pandas as pd
from pandas.core.accessor import CachedAccessor
from pandas.io.formats.printing import pprint_thing
from pandas.api.types import is_list_like, is_hashable
from pandas.api.extensions import ExtensionDtype
from pandas.tseries.frequencies import DateOffset
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F, Column
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    StructType,
    IntegralType,
    ArrayType,
)
from pyspark.sql.window import Window

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.accessors import KoalasSeriesMethods
from databricks.koalas.categorical import CategoricalAccessor
from databricks.koalas.config import get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.exceptions import SparkPandasIndexingError
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import Frame
from databricks.koalas.internal import (
    InternalFrame,
    DEFAULT_SERIES_NAME,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_DEFAULT_INDEX_NAME,
    SPARK_DEFAULT_SERIES_NAME,
)
from databricks.koalas.missing.series import MissingPandasLikeSeries
from databricks.koalas.plot import KoalasPlotAccessor
from databricks.koalas.ml import corr
from databricks.koalas.utils import (
    combine_frames,
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    same_anchor,
    scol_for,
    sql_conf,
    validate_arguments_and_invoke_function,
    validate_axis,
    validate_bool_kwarg,
    verify_temp_column_name,
    SPARK_CONF_ARROW_ENABLED,
)
from databricks.koalas.datetimes import DatetimeMethods
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkSeriesMethods
from databricks.koalas.strings import StringMethods
from databricks.koalas.typedef import (
    infer_return_type,
    spark_type_to_pandas_dtype,
    ScalarType,
    Scalar,
    SeriesType,
)

T = TypeVar("T")

REPR_PATTERN = re.compile(r"Length: (?P<length>\d+)")


def unpack_scalar(sdf: spark.DataFrame) -> Any:
    l = sdf.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row = l.iloc[0]
    l2 = list(row)
    assert len(l2) == 1, (row, l2)
    return l2[0]


def first_series(df: Union[DataFrame, pd.DataFrame]) -> Union["Series", pd.Series]:
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    if isinstance(df, DataFrame):
        return df._kser_for(df._internal.column_labels[0])
    else:
        return df[df.columns[0]]


class Series(Frame, IndexOpsMixin, Generic[T]):
    def __init__(
        self,
        data: Any,
        index: Optional[Any] = None,
        dtype: Optional[Any] = None,
        name: Optional[Any] = None,
        copy: bool = False,
        fastpath: bool = False,
    ) -> None:
        # initialization code here (omitted for brevity)
        ...

    @property
    def _kdf(self) -> DataFrame:
        ...

    @property
    def _internal(self) -> InternalFrame:
        ...

    @property
    def _column_label(self) -> Any:
        ...

    def _update_anchor(self, kdf: DataFrame) -> None:
        ...

    def _with_new_scol(self, scol: Column, *, dtype: Optional[Any] = None) -> "Series":
        ...

    spark = CachedAccessor("spark", SparkSeriesMethods)

    @property
    def dtypes(self) -> np.dtype:
        return self.dtype

    @property
    def axes(self) -> List[Any]:
        return [self.index]

    @property
    def spark_type(self) -> Any:
        warnings.warn(
            "Series.spark_type is deprecated as of Series.spark.data_type. "
            "Please use the API instead.",
            FutureWarning,
        )
        return self.spark.data_type

    spark_type.__doc__ = SparkSeriesMethods.data_type.__doc__

    def add(self, other: Any) -> "Series":
        return self + other

    add.__doc__ = "Addition of two Series."

    def radd(self, other: Any) -> "Series":
        return other + self

    radd.__doc__ = "Reverse addition of two Series."

    def div(self, other: Any) -> "Series":
        return self / other

    div.__doc__ = "Floating division of two Series."
    divide = div

    def rdiv(self, other: Any) -> "Series":
        return other / self

    rdiv.__doc__ = "Reverse floating division of two Series."

    def truediv(self, other: Any) -> "Series":
        return self / other

    truediv.__doc__ = "True division of two Series."

    def rtruediv(self, other: Any) -> "Series":
        return other / self

    rtruediv.__doc__ = "Reverse true division of two Series."

    def mul(self, other: Any) -> "Series":
        return self * other

    mul.__doc__ = "Multiplication of two Series."
    multiply = mul

    def rmul(self, other: Any) -> "Series":
        return other * self

    rmul.__doc__ = "Reverse multiplication of two Series."

    def sub(self, other: Any) -> "Series":
        return self - other

    sub.__doc__ = "Subtraction of two Series."
    subtract = sub

    def rsub(self, other: Any) -> "Series":
        return other - self

    rsub.__doc__ = "Reverse subtraction of two Series."

    def mod(self, other: Any) -> "Series":
        return self % other

    mod.__doc__ = "Modulo of two Series."

    def rmod(self, other: Any) -> "Series":
        return other % self

    rmod.__doc__ = "Reverse modulo of two Series."

    def pow(self, other: Any) -> "Series":
        return self ** other

    pow.__doc__ = "Exponential power of a Series."

    def rpow(self, other: Any) -> "Series":
        return other ** self

    rpow.__doc__ = "Reverse exponential power of a Series."

    def floordiv(self, other: Any) -> "Series":
        return self // other

    floordiv.__doc__ = "Integer division of two Series."

    def rfloordiv(self, other: Any) -> "Series":
        return other // self

    rfloordiv.__doc__ = "Reverse integer division of two Series."

    koalas = CachedAccessor("koalas", KoalasSeriesMethods)

    def eq(self, other: Any) -> Any:
        return self == other

    equals = eq

    def gt(self, other: Any) -> "Series":
        return self > other

    def ge(self, other: Any) -> "Series":
        return self >= other

    def lt(self, other: Any) -> "Series":
        return self < other

    def le(self, other: Any) -> "Series":
        return self <= other

    def ne(self, other: Any) -> "Series":
        return self != other

    def divmod(self, other: Any) -> Tuple["Series", "Series"]:
        return (self.floordiv(other), self.mod(other))

    def rdivmod(self, other: Any) -> Tuple["Series", "Series"]:
        return (self.rfloordiv(other), self.rmod(other))

    def between(self, left: Any, right: Any, inclusive: bool = True) -> "Series":
        if inclusive:
            lmask = self >= left
            rmask = self <= right
        else:
            lmask = self > left
            rmask = self < right

        return lmask & rmask

    def map(self, arg: Union[dict, Callable[[Any], Any]]) -> "Series":
        if isinstance(arg, dict):
            is_start = True
            current = F.when(F.lit(False), F.lit(None).cast(self.spark.data_type))
            for to_replace, value in arg.items():
                if is_start:
                    current = F.when(self.spark.column == F.lit(to_replace), value)
                    is_start = False
                else:
                    current = current.when(self.spark.column == F.lit(to_replace), value)
            if hasattr(arg, "__missing__"):
                tmp_val = arg[np._NoValue]
                del arg[np._NoValue]
                current = current.otherwise(F.lit(tmp_val))
            else:
                current = current.otherwise(F.lit(None).cast(self.spark.data_type))
            return self._with_new_scol(current)
        else:
            return self.apply(arg)

    def alias(self, name: Any) -> "Series":
        warnings.warn(
            "Series.alias is deprecated as of Series.rename. Please use the API instead.",
            FutureWarning,
        )
        return self.rename(name)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self),)

    @property
    def name(self) -> Union[Any, Tuple]:
        name = self._column_label
        if name is not None and len(name) == 1:
            return name[0]
        else:
            return name

    @name.setter
    def name(self, name: Union[Any, Tuple]) -> None:
        self.rename(name, inplace=True)

    def rename(self, index: Optional[Any] = None, **kwargs: Any) -> "Series":
        if index is None:
            pass
        elif not is_hashable(index):
            raise TypeError("Series.name must be a hashable type")
        elif not isinstance(index, tuple):
            index = (index,)
        scol = self.spark.column.alias(name_like_string(index))
        internal = self._internal.copy(
            column_labels=[index], data_spark_columns=[scol], column_label_names=None
        )
        kdf = DataFrame(internal)
        if kwargs.get("inplace", False):
            self._col_label = index
            self._update_anchor(kdf)
            return self
        else:
            return first_series(kdf)

    def rename_axis(
        self,
        mapper: Optional[Any] = None,
        index: Optional[Any] = None,
        inplace: bool = False,
    ) -> Optional["Series"]:
        kdf = self.to_frame().rename_axis(mapper=mapper, index=index, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    @property
    def index(self) -> "ks.Index":
        return self._kdf.index

    @property
    def is_unique(self) -> bool:
        scol = self.spark.column
        return self._internal.spark_frame.select(
            (F.count(scol) == F.countDistinct(scol))
            & (F.count(F.when(scol.isNull(), 1).otherwise(None)) <= 1)
        ).collect()[0][0]

    def reset_index(
        self,
        level: Optional[Any] = None,
        drop: bool = False,
        name: Optional[Any] = None,
        inplace: bool = False,
    ) -> Optional[Union["Series", DataFrame]]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace and not drop:
            raise TypeError("Cannot reset_index inplace on a Series to create a DataFrame")
        if drop:
            kdf = self._kdf[[self.name]]
        else:
            kser = self
            if name is not None:
                kser = kser.rename(name)
            kdf = kser.to_frame()
        kdf = kdf.reset_index(level=level, drop=drop)
        if drop:
            if inplace:
                self._update_anchor(kdf)
                return None
            else:
                return first_series(kdf)
        else:
            return kdf

    def to_frame(self, name: Optional[Union[Any, Tuple]] = None) -> DataFrame:
        if name is not None:
            renamed = self.rename(name)
        elif self._column_label is None:
            renamed = self.rename(DEFAULT_SERIES_NAME)
        else:
            renamed = self
        return DataFrame(renamed._internal)

    to_dataframe = to_frame

    def to_string(
        self,
        buf: Optional[Any] = None,
        na_rep: str = "NaN",
        float_format: Optional[Callable[[float], str]] = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: Optional[int] = None,
    ) -> str:
        args = locals()
        if max_rows is not None:
            kseries = self.head(max_rows)
        else:
            kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_string, pd.Series.to_string, args
        )

    def to_clipboard(self, excel: bool = True, sep: Optional[str] = None, **kwargs: Any) -> None:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_clipboard, pd.Series.to_clipboard, args
        )

    to_clipboard.__doc__ = DataFrame.to_clipboard.__doc__

    def to_dict(self, into: type = dict) -> Mapping[Any, Any]:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_dict, pd.Series.to_dict, args
        )

    def to_latex(
        self,
        buf: Optional[Any] = None,
        columns: Optional[Any] = None,
        col_space: Optional[Any] = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[Any] = None,
        float_format: Optional[Any] = None,
        sparsify: Optional[Any] = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: Optional[Any] = None,
        longtable: Optional[Any] = None,
        escape: Optional[Any] = None,
        encoding: Optional[Any] = None,
        decimal: str = ".",
        multicolumn: Optional[Any] = None,
        multicolumn_format: Optional[Any] = None,
        multirow: Optional[Any] = None,
    ) -> Optional[str]:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_latex, pd.Series.to_latex, args
        )

    to_latex.__doc__ = DataFrame.to_latex.__doc__

    def to_pandas(self) -> pd.Series:
        return self._to_internal_pandas().copy()

    def toPandas(self) -> pd.Series:
        warnings.warn(
            "Series.toPandas is deprecated as of Series.to_pandas. Please use the API instead.",
            FutureWarning,
        )
        return self.to_pandas()

    toPandas.__doc__ = to_pandas.__doc__

    def to_list(self) -> List[Any]:
        return self._to_internal_pandas().tolist()

    tolist = to_list

    def drop_duplicates(self, keep: Union[str, bool] = "first", inplace: bool = False) -> Optional["Series"]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].drop_duplicates(keep=keep)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def reindex(self, index: Optional[Any] = None, fill_value: Optional[Any] = None) -> "Series":
        return first_series(self.to_frame().reindex(index=index, fill_value=fill_value)).rename(self.name)

    def reindex_like(self, other: Union["Series", DataFrame]) -> "Series":
        if isinstance(other, (Series, DataFrame)):
            return self.reindex(index=other.index)
        else:
            raise TypeError("other must be a Koalas Series or DataFrame")

    def fillna(
        self, value: Optional[Any] = None, method: Optional[str] = None, axis: Optional[Any] = None, inplace: bool = False, limit: Optional[int] = None
    ) -> Optional["Series"]:
        kser = self._fillna(value=value, method=method, axis=axis, limit=limit)
        if method is not None:
            kser = DataFrame(kser._kdf._internal.resolved_copy)._kser_for(self._column_label)
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace:
            self._kdf._update_internal_frame(kser._kdf._internal, requires_same_anchor=False)
            return None
        else:
            return kser._with_new_scol(kser.spark.column)

    def _fillna(
        self, value: Optional[Any] = None, method: Optional[str] = None, axis: Optional[Any] = None, limit: Optional[int] = None, part_cols: Tuple[Any, ...] = ()
    ) -> "Series":
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError("fillna currently only works for axis=0 or axis='index'")
        if (value is None) and (method is None):
            raise ValueError("Must specify a fillna 'value' or 'method' parameter.")
        if (method is not None) and (method not in ["ffill", "pad", "backfill", "bfill"]):
            raise ValueError("Expecting 'pad', 'ffill', 'backfill' or 'bfill'.")
        scol = self.spark.column
        if isinstance(self.spark.data_type, (FloatType, DoubleType)):
            cond = scol.isNull() | F.isnan(scol)
        else:
            if not self.spark.nullable:
                return self.copy()
            cond = scol.isNull()
        if value is not None:
            if not isinstance(value, (float, int, str, bool)):
                raise TypeError("Unsupported type %s" % type(value).__name__)
            if limit is not None:
                raise ValueError("limit parameter for value is not support now")
            scol = F.when(cond, value).otherwise(scol)
        else:
            if method in ["ffill", "pad"]:
                func = F.last
                end = Window.currentRow - 1
                if limit is not None:
                    begin = Window.currentRow - limit
                else:
                    begin = Window.unboundedPreceding
            elif method in ["bfill", "backfill"]:
                func = F.first
                begin = Window.currentRow + 1
                if limit is not None:
                    end = Window.currentRow + limit
                else:
                    end = Window.unboundedFollowing
            window = (
                Window.partitionBy(*part_cols)
                .orderBy(NATURAL_ORDER_COLUMN_NAME)
                .rowsBetween(begin, end)
            )
            scol = F.when(cond, func(scol, True).over(window)).otherwise(scol)
        return DataFrame(
            self._kdf._internal.with_new_spark_column(
                self._column_label, scol.alias(name_like_string(self.name))
            )
        )._kser_for(self._column_label)

    def dropna(self, axis: int = 0, inplace: bool = False, **kwargs: Any) -> Optional["Series"]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].dropna(axis=axis, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def clip(self, lower: Optional[Union[float, int]] = None, upper: Optional[Union[float, int]] = None) -> "Series":
        if is_list_like(lower) or is_list_like(upper):
            raise ValueError("List-like values are not supported for 'lower' and 'upper' at the moment")
        if lower is None and upper is None:
            return self
        if isinstance(self.spark.data_type, NumericType):
            scol = self.spark.column
            if lower is not None:
                scol = F.when(scol < lower, lower).otherwise(scol)
            if upper is not None:
                scol = F.when(scol > upper, upper).otherwise(scol)
            return self._with_new_scol(scol, dtype=self.dtype)
        else:
            return self

    def drop(self, labels: Optional[Any] = None, index: Optional[Union[Any, Tuple, List[Any], List[Tuple]]] = None, level: Optional[Any] = None) -> "Series":
        return first_series(self._drop(labels=labels, index=index, level=level))

    def _drop(
        self, labels: Optional[Any] = None, index: Optional[Union[Any, Tuple, List[Any], List[Tuple]]] = None, level: Optional[Any] = None
    ) -> DataFrame:
        if labels is not None:
            if index is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'")
            return self._drop(index=labels, level=level)
        if index is not None:
            internal = self._internal
            if level is None:
                level = 0
            if level >= internal.index_level:
                raise ValueError("'level' should be less than the number of indexes")
            if is_name_like_tuple(index):
                index = [index]
            elif is_name_like_value(index):
                index = [(index,)]
            elif all(is_name_like_value(idxes, allow_tuple=False) for idxes in index):
                index = [(idex,) for idex in index]
            elif not all(is_name_like_tuple(idxes) for idxes in index):
                raise ValueError(
                    "If the given index is a list, it should only contains names as all tuples or all non tuples that contain index names"
                )
            drop_index_scols = []
            for idxes in index:
                try:
                    index_scols = [
                        internal.index_spark_columns[lvl] == idx
                        for lvl, idx in enumerate(idxes, level)
                    ]
                except IndexError:
                    raise KeyError(
                        "Key length ({}) exceeds index depth ({})".format(
                            internal.index_level, len(idxes)
                        )
                    )
                drop_index_scols.append(reduce(lambda x, y: x & y, index_scols))
            cond = ~reduce(lambda x, y: x | y, drop_index_scols)
            return DataFrame(internal.with_filter(cond))
        else:
            raise ValueError("Need to specify at least one of 'labels' or 'index'")

    def head(self, n: int = 5) -> "Series":
        return first_series(self.to_frame().head(n)).rename(self.name)

    def last(self, offset: Union[str, DateOffset]) -> "Series":
        return first_series(self.to_frame().last(offset)).rename(self.name)

    def first(self, offset: Union[str, DateOffset]) -> "Series":
        return first_series(self.to_frame().first(offset)).rename(self.name)

    def unique(self) -> "Series":
        sdf = self._internal.spark_frame.select(self.spark.column).distinct()
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=None,
            column_labels=[self._column_label],
            data_spark_columns=[scol_for(sdf, self._internal.data_spark_column_names[0])],
            data_dtypes=[self.dtype],
            column_label_names=self._internal.column_label_names,
        )
        return first_series(DataFrame(internal))

    def sort_values(
        self, ascending: bool = True, inplace: bool = False, na_position: str = "last"
    ) -> Optional["Series"]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]]._sort(
            by=[self.spark.column], ascending=ascending, inplace=False, na_position=na_position
        )
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def sort_index(
        self,
        axis: int = 0,
        level: Optional[Union[int, List[int]]] = None,
        ascending: bool = True,
        inplace: bool = False,
        kind: Optional[str] = None,
        na_position: str = "last",
    ) -> Optional["Series"]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].sort_index(
            axis=axis, level=level, ascending=ascending, kind=kind, na_position=na_position
        )
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def swaplevel(self, i: int = -2, j: int = -1, copy: bool = True) -> "Series":
        assert copy is True
        return first_series(self.to_frame().swaplevel(i, j, axis=0)).rename(self.name)

    def swapaxes(self, i: Union[str, int], j: Union[str, int], copy: bool = True) -> "Series":
        assert copy is True
        i = validate_axis(i)
        j = validate_axis(j)
        if not i == j == 0:
            raise ValueError("Axis must be 0 for Series")
        return self.copy()

    def add_prefix(self, prefix: str) -> "Series":
        assert isinstance(prefix, str)
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame.select(
            [
                F.concat(F.lit(prefix), index_spark_column).alias(index_spark_column_name)
                for index_spark_column, index_spark_column_name in zip(
                    internal.index_spark_columns, internal.index_spark_column_names
                )
            ]
            + internal.data_spark_columns
        )
        return first_series(
            DataFrame(internal.with_new_sdf(sdf, index_dtypes=([None] * internal.index_level)))
        )

    def add_suffix(self, suffix: str) -> "Series":
        assert isinstance(suffix, str)
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame.select(
            [
                F.concat(index_spark_column, F.lit(suffix)).alias(index_spark_column_name)
                for index_spark_column, index_spark_column_name in zip(
                    internal.index_spark_columns, internal.index_spark_column_names
                )
            ]
            + internal.data_spark_columns
        )
        return first_series(
            DataFrame(internal.with_new_sdf(sdf, index_dtypes=([None] * internal.index_level)))
        )

    def corr(self, other: "Series", method: str = "pearson") -> float:
        columns = ["__corr_arg1__", "__corr_arg2__"]
        kdf = self._kdf.assign(__corr_arg1__=self, __corr_arg2__=other)[columns]
        kdf.columns = columns
        c = corr(kdf, method=method)
        return c.loc[tuple(columns)]

    def nsmallest(self, n: int = 5) -> "Series":
        return self.sort_values(ascending=True).head(n)

    def nlargest(self, n: int = 5) -> "Series":
        return self.sort_values(ascending=False).head(n)

    def append(
        self, to_append: "Series", ignore_index: bool = False, verify_integrity: bool = False
    ) -> "Series":
        return first_series(
            self.to_frame().append(to_append.to_frame(), ignore_index, verify_integrity)
        ).rename(self.name)

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> "Series":
        return first_series(
            self.to_frame().sample(n=n, frac=frac, replace=replace, random_state=random_state)
        ).rename(self.name)

    sample.__doc__ = DataFrame.sample.__doc__

    def hist(self, bins: int = 10, **kwds: Any) -> Any:
        return self.plot.hist(bins, **kwds)

    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def apply(self, func: Callable[..., Any], args: Tuple[Any, ...] = (), **kwds: Any) -> "Series":
        assert callable(func), "the first argument should be a callable function."
        try:
            spec = inspect.getfullargspec(func)
            return_sig = spec.annotations.get("return", None)
            should_infer_schema = return_sig is None
        except TypeError:
            should_infer_schema = True
        apply_each = wraps(func)(lambda s: s.apply(func, args=args, **kwds))
        if should_infer_schema:
            return self.koalas._transform_batch(apply_each, None)
        else:
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, ScalarType):
                raise ValueError(
                    "Expected the return type of this function to be of scalar type, "
                    "but found type {}".format(sig_return)
                )
            return_type = cast(ScalarType, sig_return)
            return self.koalas._transform_batch(apply_each, return_type)

    def aggregate(self, func: Union[str, List[str]]) -> Union[Scalar, "Series"]:
        if isinstance(func, list):
            return first_series(self.to_frame().aggregate(func)).rename(self.name)
        elif isinstance(func, str):
            return getattr(self, func)()
        else:
            raise ValueError("func must be a string or list of strings")

    agg = aggregate

    def transpose(self, *args: Any, **kwargs: Any) -> "Series":
        return self.copy()

    T = property(transpose)

    def transform(self, func: Union[Callable[..., Any], List[Callable[..., Any]]], axis: Any = 0, *args: Any, **kwargs: Any) -> Union["Series", DataFrame]:
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if isinstance(func, list):
            applied = []
            for f in func:
                applied.append(self.apply(f, args=args, **kwargs).rename(f.__name__))
            internal = self._internal.with_new_columns(applied)
            return DataFrame(internal)
        else:
            return self.apply(func, args=args, **kwargs)

    def transform_batch(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> "ks.Series":
        warnings.warn(
            "Series.transform_batch is deprecated as of Series.koalas.transform_batch. "
            "Please use the API instead.",
            FutureWarning,
        )
        return self.koalas.transform_batch(func, *args, **kwargs)

    transform_batch.__doc__ = KoalasSeriesMethods.transform_batch.__doc__

    def round(self, decimals: int = 0) -> "Series":
        if not isinstance(decimals, int):
            raise ValueError("decimals must be an integer")
        scol = F.round(self.spark.column, decimals)
        return self._with_new_scol(scol)

    def quantile(
        self, q: Union[float, Iterable[float]] = 0.5, accuracy: int = 10000
    ) -> Union[Scalar, "Series"]:
        if isinstance(q, Iterable):
            return first_series(
                self.to_frame().quantile(q=q, axis=0, numeric_only=False, accuracy=accuracy)
            ).rename(self.name)
        else:
            if not isinstance(accuracy, int):
                raise ValueError(
                    "accuracy must be an integer; however, got [%s]" % type(accuracy).__name__
                )
            if not isinstance(q, float):
                raise ValueError(
                    "q must be a float or an array of floats; however, [%s] found." % type(q)
                )
            if q < 0.0 or q > 1.0:
                raise ValueError("percentiles should all be in the interval [0, 1].")
            def quantile(spark_column: Column, spark_type: Any) -> Any:
                if isinstance(spark_type, (BooleanType, NumericType)):
                    return SF.percentile_approx(spark_column.cast(DoubleType()), q, accuracy)
                else:
                    raise TypeError(
                        "Could not convert {} ({}) to numeric".format(
                            spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                        )
                    )
            return self._reduce_for_stat_function(quantile, name="quantile")

    def rank(self, method: str = "average", ascending: bool = True) -> "Series":
        return self._rank(method, ascending).spark.analyzed

    def _rank(self, method: str = "average", ascending: bool = True, *, part_cols: Tuple[Any, ...] = ()) -> "Series":
        if method not in ["average", "min", "max", "first", "dense"]:
            msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
            raise ValueError(msg)
        if self._internal.index_level > 1:
            raise ValueError("rank do not support index now")
        if ascending:
            asc_func = lambda scol: scol.asc()
        else:
            asc_func = lambda scol: scol.desc()
        if method == "first":
            window = (
                Window.orderBy(
                    asc_func(self.spark.column), asc_func(F.col(NATURAL_ORDER_COLUMN_NAME)),
                )
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            scol = F.row_number().over(window)
        elif method == "dense":
            window = (
                Window.orderBy(asc_func(self.spark.column))
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            scol = F.dense_rank().over(window)
        else:
            if method == "average":
                stat_func = F.mean
            elif method == "min":
                stat_func = F.min
            elif method == "max":
                stat_func = F.max
            window1 = (
                Window.orderBy(asc_func(self.spark.column))
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            window2 = Window.partitionBy([self.spark.column] + list(part_cols)).rowsBetween(
                Window.unboundedPreceding, Window.unboundedFollowing
            )
            scol = stat_func(F.row_number().over(window1)).over(window2)
        kser = self._with_new_scol(scol)
        return kser.astype(np.float64)

    def filter(self, items: Optional[Any] = None, like: Optional[Any] = None, regex: Optional[Any] = None, axis: Optional[Any] = None) -> "Series":
        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError("Series does not support columns axis.")
        return first_series(
            self.to_frame().filter(items=items, like=like, regex=regex, axis=axis)
        ).rename(self.name)

    filter.__doc__ = DataFrame.filter.__doc__

    def describe(self, percentiles: Optional[List[float]] = None) -> "Series":
        return first_series(self.to_frame().describe(percentiles)).rename(self.name)

    describe.__doc__ = DataFrame.describe.__doc__

    def diff(self, periods: int = 1) -> "Series":
        return self._diff(periods).spark.analyzed

    def _diff(self, periods: int, *, part_cols: Tuple[Any, ...] = ()) -> "Series":
        if not isinstance(periods, int):
            raise ValueError("periods should be an int; however, got [%s]" % type(periods).__name__)
        window = (
            Window.partitionBy(*part_cols)
            .orderBy(NATURAL_ORDER_COLUMN_NAME)
            .rowsBetween(-periods, -periods)
        )
        scol = self.spark.column - F.lag(self.spark.column, periods).over(window)
        return self._with_new_scol(scol, dtype=self.dtype)

    def idxmax(self, skipna: bool = True) -> Union[Tuple[Any, ...], Any]:
        sdf = self._internal.spark_frame
        scol = self.spark.column
        index_scols = self._internal.index_spark_columns
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_last()), NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_first()), NATURAL_ORDER_COLUMN_NAME)
        results = sdf.select([scol] + index_scols).take(1)
        if len(results) == 0:
            raise ValueError("attempt to get idxmin of an empty sequence")
        if results[0][0] is None:
            return np.nan
        values = list(results[0][1:])
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def idxmin(self, skipna: bool = True) -> Union[Tuple[Any, ...], Any]:
        sdf = self._internal.spark_frame
        scol = self.spark.column
        index_scols = self._internal.index_spark_columns
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_last()), NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_first()), NATURAL_ORDER_COLUMN_NAME)
        results = sdf.select([scol] + index_scols).take(1)
        if len(results) == 0:
            raise ValueError("attempt to get idxmin of an empty sequence")
        if results[0][0] is None:
            return np.nan
        values = list(results[0][1:])
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def pop(self, item: Any) -> Union["Series", Scalar]:
        if not is_name_like_value(item):
            raise ValueError("'key' should be string or tuple that contains strings")
        if not is_name_like_tuple(item):
            item = (item,)
        if self._internal.index_level < len(item):
            raise KeyError(
                "Key length ({}) exceeds index depth ({})".format(
                    len(item), self._internal.index_level
                )
            )
        internal = self._internal
        scols = internal.index_spark_columns[len(item) :] + [self.spark.column]
        rows = [internal.spark_columns[level] == index for level, index in enumerate(item)]
        sdf = internal.spark_frame.filter(reduce(lambda x, y: x & y, rows)).select(scols)
        kdf = self._drop(item)
        self._update_anchor(kdf)
        if self._internal.index_level == len(item):
            pdf = sdf.limit(2).toPandas()
            length = len(pdf)
            if length == 1:
                return pdf[internal.data_spark_column_names[0]].iloc[0]
            item_string = name_like_string(item)
            sdf = sdf.withColumn(SPARK_DEFAULT_INDEX_NAME, F.lit(str(item_string)))
            internal = InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[scol_for(sdf, SPARK_DEFAULT_INDEX_NAME)],
                column_labels=[self._column_label],
                data_dtypes=[self.dtype],
            )
            return first_series(DataFrame(internal))
        else:
            internal = internal.copy(
                spark_frame=sdf,
                index_spark_columns=[
                    scol_for(sdf, col) for col in internal.index_spark_column_names[len(item) :]
                ],
                index_dtypes=internal.index_dtypes[len(item) :],
                index_names=self._internal.index_names[len(item) :],
                data_spark_columns=[scol_for(sdf, internal.data_spark_column_names[0])],
            )
            return first_series(DataFrame(internal))

    def copy(self, deep: Optional[Any] = None) -> "Series":
        return self._kdf.copy()._kser_for(self._column_label)

    def mode(self, dropna: bool = True) -> "Series":
        ser_count = self.value_counts(dropna=dropna, sort=False)
        sdf_count = ser_count._internal.spark_frame
        most_value = ser_count.max()
        sdf_most_value = sdf_count.filter("count == {}".format(most_value))
        sdf = sdf_most_value.select(
            F.col(SPARK_DEFAULT_INDEX_NAME).alias(SPARK_DEFAULT_SERIES_NAME)
        )
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=None, column_labels=[None])
        return first_series(DataFrame(internal))

    def keys(self) -> "ks.Index":
        return self.index

    def replace(self, to_replace: Optional[Any] = None, value: Optional[Any] = None, regex: bool = False) -> "Series":
        if to_replace is None:
            return self.fillna(method="ffill")
        if not isinstance(to_replace, (str, list, tuple, dict, int, float)):
            raise ValueError("'to_replace' should be one of str, list, tuple, dict, int, float")
        if regex:
            raise NotImplementedError("replace currently not support for regex")
        to_replace = list(to_replace) if isinstance(to_replace, tuple) else to_replace
        value = list(value) if isinstance(value, tuple) else value
        if isinstance(to_replace, list) and isinstance(value, list):
            if not len(to_replace) == len(value):
                raise ValueError(
                    "Replacement lists must match in length. Expecting {} got {}".format(
                        len(to_replace), len(value)
                    )
                )
            to_replace = {k: v for k, v in zip(to_replace, value)}
        if isinstance(to_replace, dict):
            is_start = True
            if len(to_replace) == 0:
                current = self.spark.column
            else:
                for to_replace_, value in to_replace.items():
                    cond = (
                        (F.isnan(self.spark.column) | self.spark.column.isNull())
                        if pd.isna(to_replace_)
                        else (self.spark.column == F.lit(to_replace_))
                    )
                    if is_start:
                        current = F.when(cond, value)
                        is_start = False
                    else:
                        current = current.when(cond, value)
                current = current.otherwise(self.spark.column)
        else:
            cond = self.spark.column.isin(to_replace)
            if np.array(pd.isna(to_replace)).any():
                cond = cond | F.isnan(self.spark.column) | self.spark.column.isNull()
            current = F.when(cond, value).otherwise(self.spark.column)
        return self._with_new_scol(current)

    def update(self, other: "Series") -> None:
        if not isinstance(other, Series):
            raise ValueError("'other' must be a Series")
        combined = combine_frames(self._kdf, other._kdf, how="leftouter")
        this_scol = combined["this"]._internal.spark_column_for(self._column_label)
        that_scol = combined["that"]._internal.spark_column_for(other._column_label)
        scol = (
            F.when(that_scol.isNotNull(), that_scol)
            .otherwise(this_scol)
            .alias(self._kdf._internal.spark_column_name_for(self._column_label))
        )
        internal = combined["this"]._internal.with_new_spark_column(
            self._column_label, scol
        )
        self._kdf._update_internal_frame(internal.resolved_copy, requires_same_anchor=False)

    def where(self, cond: "Series", other: Any = np.nan) -> "Series":
        assert isinstance(cond, Series)
        should_try_ops_on_diff_frame = not same_anchor(cond, self) or (
            isinstance(other, Series) and not same_anchor(other, self)
        )
        if should_try_ops_on_diff_frame:
            kdf = self.to_frame()
            tmp_cond_col = verify_temp_column_name(kdf, "__tmp_cond_col__")
            tmp_other_col = verify_temp_column_name(kdf, "__tmp_other_col__")
            kdf[tmp_cond_col] = cond
            kdf[tmp_other_col] = other
            condition = (
                F.when(
                    kdf[tmp_cond_col].spark.column,
                    kdf._kser_for(kdf._internal.column_labels[0]).spark.column,
                )
                .otherwise(kdf[tmp_other_col].spark.column)
                .alias(kdf._internal.data_spark_column_names[0])
            )
            internal = kdf._internal.with_new_columns(
                [condition], column_labels=self._internal.column_labels
            )
            return first_series(DataFrame(internal))
        else:
            if isinstance(other, Series):
                other = other.spark.column
            condition = (
                F.when(cond.spark.column, self.spark.column)
                .otherwise(other)
                .alias(self._internal.data_spark_column_names[0])
            )
            return self._with_new_scol(condition)

    def mask(self, cond: "Series", other: Any = np.nan) -> "Series":
        return self.where(~cond, other)

    def xs(self, key: Any, level: Optional[Union[int, str, List[Any]]] = None) -> "Series":
        if not isinstance(key, tuple):
            key = (key,)
        if level is None:
            level = 0
        internal = self._internal
        scols = (
            internal.index_spark_columns[:level]
            + internal.index_spark_columns[level + len(key) :]
            + [self.spark.column]
        )
        rows = [internal.spark_columns[lvl] == index for lvl, index in enumerate(key, level)]
        sdf = internal.spark_frame.filter(reduce(lambda x, y: x & y, rows)).select(scols)
        if internal.index_level == len(key):
            pdf = sdf.limit(2).toPandas()
            length = len(pdf)
            if length == 1:
                return pdf[internal.data_spark_column_names[0]].iloc[0]
        index_spark_column_names = (
            internal.index_spark_column_names[:level]
            + internal.index_spark_column_names[level + len(key) :]
        )
        index_names = internal.index_names[:level] + internal.index_names[level + len(key) :]
        index_dtypes = internal.index_dtypes[:level] + internal.index_dtypes[level + len(key) :]
        internal = internal.copy(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in index_spark_column_names],
            index_names=index_names,
            index_dtypes=index_dtypes,
            data_spark_columns=[scol_for(sdf, internal.data_spark_column_names[0])],
        )
        return first_series(DataFrame(internal))

    def pct_change(self, periods: int = 1) -> "Series":
        scol = self.spark.column
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        prev_row = F.lag(scol, periods).over(window)
        return self._with_new_scol((scol - prev_row) / prev_row).spark.analyzed

    def combine_first(self, other: "Series") -> "Series":
        if not isinstance(other, ks.Series):
            raise ValueError("`combine_first` only allows `Series` for parameter `other`")
        if same_anchor(self, other):
            this = self.spark.column
            that = other.spark.column
            combined = self._kdf
        else:
            combined = combine_frames(self._kdf, other._kdf)
            this = combined["this"]._internal.spark_column_for(self._column_label)
            that = combined["that"]._internal.spark_column_for(other._column_label)
        cond = F.when(self.spark.column.isNull(), that).otherwise(this)
        if same_anchor(self, other):
            return self._with_new_scol(cond)
        index_scols = combined._internal.index_spark_columns
        sdf = combined._internal.spark_frame.select(
            *index_scols, cond.alias(self._internal.data_spark_column_names[0])
        ).distinct()
        internal = self._internal.with_new_sdf(sdf, data_dtypes=[None])
        return first_series(DataFrame(internal))

    def dot(self, other: Union["Series", DataFrame]) -> Union[Scalar, "Series"]:
        if isinstance(other, DataFrame):
            if not same_anchor(self, other):
                if not self.index.sort_values().equals(other.index.sort_values()):
                    raise ValueError("matrices are not aligned")
            other = other.copy()
            column_labels = other._internal.column_labels
            self_column_label = verify_temp_column_name(other, "__self_column__")
            other[self_column_label] = self
            self_kser = other._kser_for(self_column_label)
            product_ksers = [other._kser_for(label) * self_kser for label in column_labels]
            dot_product_kser = DataFrame(
                other._internal.with_new_columns(product_ksers, column_labels=column_labels)
            ).sum()
            return cast(Series, dot_product_kser).rename(self.name)
        else:
            assert isinstance(other, Series)
            if not same_anchor(self, other):
                if len(self.index) != len(other.index):
                    raise ValueError("matrices are not aligned")
            return (self * other).sum()

    def __matmul__(self, other: Any) -> Union[Scalar, "Series"]:
        return self.dot(other)

    def repeat(self, repeats: Union[int, "Series"]) -> "Series":
        if not isinstance(repeats, (int, Series)):
            raise ValueError(
                "`repeats` argument must be integer or Series, but got {}".format(type(repeats))
            )
        if isinstance(repeats, Series):
            if LooseVersion(pyspark.__version__) < LooseVersion("2.4"):
                raise ValueError(
                    "`repeats` argument must be integer with Spark<2.4, but got {}".format(
                        type(repeats)
                    )
                )
            if not same_anchor(self, repeats):
                kdf = self.to_frame()
                temp_repeats = verify_temp_column_name(kdf, "__temp_repeats__")
                kdf[temp_repeats] = repeats
                return (
                    kdf._kser_for(kdf._internal.column_labels[0])
                    .repeat(kdf[temp_repeats])
                    .rename(self.name)
                )
            else:
                scol = F.explode(
                    SF.array_repeat(self.spark.column, repeats.astype("int32").spark.column)
                ).alias(name_like_string(self._column_label))
                sdf = self._internal.spark_frame.select(self._internal.index_spark_columns + [scol])
                internal = self._internal.copy(
                    spark_frame=sdf,
                    index_spark_columns=[
                        scol_for(sdf, col) for col in self._internal.index_spark_column_names
                    ],
                    data_spark_columns=[scol_for(sdf, name_like_string(self._column_label))],
                )
                return first_series(DataFrame(internal))
        else:
            if repeats < 0:
                raise ValueError("negative dimensions are not allowed")
            kdf = self._kdf[[self.name]]
            if repeats == 0:
                return first_series(DataFrame(kdf._internal.with_filter(F.lit(False))))
            else:
                return first_series(ks.concat([kdf] * repeats))

    def asof(self, where: Any) -> Union[Scalar, "Series"]:
        should_return_series = True
        if isinstance(self.index, ks.MultiIndex):
            raise ValueError("asof is not supported for a MultiIndex")
        if isinstance(where, (ks.Index, ks.Series, DataFrame)):
            raise ValueError("where cannot be an Index, Series or a DataFrame")
        if not self.index.is_monotonic_increasing:
            raise ValueError("asof requires a sorted index")
        if not is_list_like(where):
            should_return_series = False
            where = [where]
        index_scol = self._internal.index_spark_columns[0]
        index_type = self._internal.spark_type_for(index_scol)
        cond = [
            F.max(F.when(index_scol <= F.lit(index).cast(index_type), self.spark.column))
            for index in where
        ]
        sdf = self._internal.spark_frame.select(cond)
        if not should_return_series:
            with sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
                result = sdf.limit(1).toPandas().iloc[0, 0]
            return result if result is not None else np.nan
        with ks.option_context("compute.default_index_type", "distributed", "compute.max_rows", 1):
            kdf = ks.DataFrame(sdf)
            kdf.columns = pd.Index(where)
            return first_series(kdf.transpose()).rename(self.name)

    def mad(self) -> float:
        sdf = self._internal.spark_frame
        spark_column = self.spark.column
        avg = unpack_scalar(sdf.select(F.avg(spark_column)))
        mad = unpack_scalar(sdf.select(F.avg(F.abs(spark_column - avg))))
        return mad

    def unstack(self, level: int = -1) -> DataFrame:
        if not isinstance(self.index, ks.MultiIndex):
            raise ValueError("Series.unstack only support for a MultiIndex")
        index_nlevels = self.index.nlevels
        if level > 0 and (level > index_nlevels - 1):
            raise IndexError(
                "Too many levels: Index has only {} levels, not {}".format(index_nlevels, level + 1)
            )
        elif level < 0 and (level < -index_nlevels):
            raise IndexError(
                "Too many levels: Index has only {} levels, {} is not a valid level number".format(
                    index_nlevels, level
                )
            )
        internal = self._internal.resolved_copy
        index_map = list(zip(internal.index_spark_column_names, internal.index_names))
        pivot_col, column_label_names = index_map.pop(level)
        index_scol_names, index_names = zip(*index_map)
        col = internal.data_spark_column_names[0]
        sdf = internal.spark_frame
        sdf = sdf.groupby(list(index_scol_names)).pivot(pivot_col).agg(F.first(scol_for(sdf, col)))
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in index_scol_names],
            index_names=list(index_names),
            column_label_names=[column_label_names],
        )
        return DataFrame(internal)

    def item(self) -> Scalar:
        return self.head(2)._to_internal_pandas().item()

    def iteritems(self) -> Iterable[Tuple[Any, Any]]:
        internal_index_columns = self._internal.index_spark_column_names
        internal_data_column = self._internal.data_spark_column_names[0]
        def extract_kv_from_spark_row(row: Any) -> Tuple[Any, Any]:
            k = (
                row[internal_index_columns[0]]
                if len(internal_index_columns) == 1
                else tuple(row[c] for c in internal_index_columns)
            )
            v = row[internal_data_column]
            return k, v
        for k, v in map(
            extract_kv_from_spark_row, self._internal.resolved_copy.spark_frame.toLocalIterator()
        ):
            yield k, v

    def items(self) -> Iterable[Tuple[Any, Any]]:
        return self.iteritems()

    def droplevel(self, level: Union[int, str, List[Union[int, str]]]) -> "Series":
        return first_series(self.to_frame().droplevel(level=level, axis=0)).rename(self.name)

    def tail(self, n: int = 5) -> "Series":
        return first_series(self.to_frame().tail(n=n)).rename(self.name)

    def explode(self) -> "Series":
        if not isinstance(self.spark.data_type, ArrayType):
            return self.copy()
        scol = F.explode_outer(self.spark.column).alias(name_like_string(self._column_label))
        internal = self._internal.with_new_columns([scol], keep_order=False)
        return first_series(DataFrame(internal))

    def argsort(self) -> "Series":
        notnull = self.loc[self.notnull()]
        sdf_for_index = notnull._internal.spark_frame.select(notnull._internal.index_spark_columns)
        tmp_join_key = verify_temp_column_name(sdf_for_index, "__tmp_join_key__")
        sdf_for_index = InternalFrame.attach_distributed_sequence_column(
            sdf_for_index, tmp_join_key
        )
        sdf_for_data = notnull._internal.spark_frame.select(
            self.spark.column.alias("values"), NATURAL_ORDER_COLUMN_NAME
        )
        sdf_for_data = InternalFrame.attach_distributed_sequence_column(
            sdf_for_data, SPARK_DEFAULT_SERIES_NAME
        )
        sdf_for_data = sdf_for_data.sort(
            scol_for(sdf_for_data, "values"), NATURAL_ORDER_COLUMN_NAME
        ).drop("values", NATURAL_ORDER_COLUMN_NAME)
        tmp_join_key = verify_temp_column_name(sdf_for_data, "__tmp_join_key__")
        sdf_for_data = InternalFrame.attach_distributed_sequence_column(sdf_for_data, tmp_join_key)
        sdf = sdf_for_index.join(sdf_for_data, on=tmp_join_key).drop(tmp_join_key)
        internal = self._internal.with_new_sdf(
            spark_frame=sdf, data_columns=[SPARK_DEFAULT_SERIES_NAME], data_dtypes=[None]
        )
        kser = first_series(DataFrame(internal))
        return cast(
            Series, ks.concat([kser, self.loc[self.isnull()].spark.transform(lambda _: F.lit(-1))])
        )

    def argmax(self) -> int:
        sdf = self._internal.spark_frame.select(self.spark.column, NATURAL_ORDER_COLUMN_NAME)
        max_value = sdf.select(
            F.max(scol_for(sdf, self._internal.data_spark_column_names[0])),
            F.first(NATURAL_ORDER_COLUMN_NAME),
        ).head()
        if max_value[1] is None:
            raise ValueError("attempt to get argmax of an empty sequence")
        elif max_value[0] is None:
            return -1
        seq_col_name = verify_temp_column_name(sdf, "__distributed_sequence_column__")
        sdf = InternalFrame.attach_distributed_sequence_column(
            sdf.drop(NATURAL_ORDER_COLUMN_NAME), seq_col_name
        )
        return sdf.filter(
            scol_for(sdf, self._internal.data_spark_column_names[0]) == max_value[0]
        ).head()[0]

    def argmin(self) -> int:
        sdf = self._internal.spark_frame.select(self.spark.column, NATURAL_ORDER_COLUMN_NAME)
        min_value = sdf.select(
            F.min(scol_for(sdf, self._internal.data_spark_column_names[0])),
            F.first(NATURAL_ORDER_COLUMN_NAME),
        ).head()
        if min_value[1] is None:
            raise ValueError("attempt to get argmin of an empty sequence")
        elif min_value[0] is None:
            return -1
        seq_col_name = verify_temp_column_name(sdf, "__distributed_sequence_column__")
        sdf = InternalFrame.attach_distributed_sequence_column(
            sdf.drop(NATURAL_ORDER_COLUMN_NAME), seq_col_name
        )
        return sdf.filter(
            scol_for(sdf, self._internal.data_spark_column_names[0]) == min_value[0]
        ).head()[0]

    def compare(
        self, other: "Series", keep_shape: bool = False, keep_equal: bool = False
    ) -> DataFrame:
        if same_anchor(self, other):
            self_column_label = verify_temp_column_name(other.to_frame(), "__self_column__")
            other_column_label = verify_temp_column_name(self.to_frame(), "__other_column__")
            combined = DataFrame(
                self._internal.with_new_columns(
                    [self.rename(self_column_label), other.rename(other_column_label)]
                )
            )
        else:
            if not self.index.equals(other.index):
                raise ValueError("Can only compare identically-labeled Series objects")
            combined = combine_frames(self.to_frame(), other.to_frame())
        this_column_label = "self"
        that_column_label = "other"
        if keep_equal and keep_shape:
            combined.columns = pd.Index([this_column_label, that_column_label])
            return combined
        this_data_scol = combined._internal.data_spark_columns[0]
        that_data_scol = combined._internal.data_spark_columns[1]
        index_scols = combined._internal.index_spark_columns
        sdf = combined._internal.spark_frame
        if keep_shape:
            this_scol = (
                F.when(this_data_scol == that_data_scol, None)
                .otherwise(this_data_scol)
                .alias(this_column_label)
            )
            that_scol = (
                F.when(this_data_scol == that_data_scol, None)
                .otherwise(that_data_scol)
                .alias(that_column_label)
            )
        else:
            sdf = sdf.filter(~this_data_scol.eqNullSafe(that_data_scol))
            this_scol = this_data_scol.alias(this_column_label)
            that_scol = that_data_scol.alias(that_column_label)
        sdf = sdf.select(index_scols + [this_scol, that_scol, NATURAL_ORDER_COLUMN_NAME])
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
            column_labels=[(this_column_label,), (that_column_label,)],
            data_spark_columns=[scol_for(sdf, this_column_label), scol_for(sdf, that_column_label)],
            column_label_names=[None],
        )
        return DataFrame(internal)

    def align(
        self,
        other: Union[DataFrame, "Series"],
        join: str = "outer",
        axis: Optional[Union[int, str]] = None,
        copy: bool = True,
    ) -> Tuple["Series", Union[DataFrame, "Series"]]:
        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError("Series does not support columns axis.")
        self_df = self.to_frame()
        left, right = self_df.align(other, join=join, axis=axis, copy=False)
        if left is self_df:
            left_ser = self
        else:
            left_ser = first_series(left).rename(self.name)
        return (left_ser.copy(), right.copy()) if copy else (left_ser, right)

    def between_time(
        self,
        start_time: Union[datetime.time, str],
        end_time: Union[datetime.time, str],
        include_start: bool = True,
        include_end: bool = True,
        axis: Union[int, str] = 0,
    ) -> "Series":
        return first_series(
            self.to_frame().between_time(start_time, end_time, include_start, include_end, axis)
        ).rename(self.name)

    def at_time(
        self, time: Union[datetime.time, str], asof: bool = False, axis: Union[int, str] = 0
    ) -> "Series":
        return first_series(self.to_frame().at_time(time, asof, axis)).rename(self.name)

    def _cum(self, func: Callable[[Column], Column], skipna: bool, part_cols: Tuple[Any, ...] = (), ascending: bool = True) -> "Series":
        if ascending:
            window = (
                Window.orderBy(F.asc(NATURAL_ORDER_COLUMN_NAME))
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        else:
            window = (
                Window.orderBy(F.desc(NATURAL_ORDER_COLUMN_NAME))
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        if skipna:
            scol = F.when(
                self.spark.column.isNull(),
                F.lit(None),
            ).otherwise(func(self.spark.column).over(window))
        else:
            scol = F.when(
                F.max(self.spark.column.isNull()).over(window),
                F.lit(None),
            ).otherwise(func(self.spark.column).over(window))
        return self._with_new_scol(scol)

    def _cumsum(self, skipna: bool, part_cols: Tuple[Any, ...] = ()) -> "Series":
        kser = self
        if isinstance(kser.spark.data_type, BooleanType):
            kser = kser.spark.transform(lambda scol: scol.cast(LongType()))
        elif not isinstance(kser.spark.data_type, NumericType):
            raise TypeError(
                "Could not convert {} ({}) to numeric".format(
                    spark_type_to_pandas_dtype(kser.spark.data_type),
                    kser.spark.data_type.simpleString(),
                )
            )
        return kser._cum(F.sum, skipna, part_cols)

    def _cumprod(self, skipna: bool, part_cols: Tuple[Any, ...] = ()) -> "Series":
        if isinstance(self.spark.data_type, BooleanType):
            scol = self._cum(
                lambda scol: F.min(F.coalesce(scol, F.lit(True))), skipna, part_cols
            ).spark.column.cast(LongType())
        elif isinstance(self.spark.data_type, NumericType):
            num_zeros = self._cum(
                lambda scol: F.sum(F.when(scol == 0, 1).otherwise(0)), skipna, part_cols
            ).spark.column
            num_negatives = self._cum(
                lambda scol: F.sum(F.when(scol < 0, 1).otherwise(0)), skipna, part_cols
            ).spark.column
            sign = F.when(num_negatives % 2 == 0, 1).otherwise(-1)
            abs_prod = F.exp(
                self._cum(lambda scol: F.sum(F.log(F.abs(scol))), skipna, part_cols).spark.column
            )
            scol = F.when(num_zeros > 0, 0).otherwise(sign * abs_prod)
            if isinstance(self.spark.data_type, IntegralType):
                scol = F.round(scol).cast(LongType())
        else:
            raise TypeError(
                "Could not convert {} ({}) to numeric".format(
                    spark_type_to_pandas_dtype(self.spark.data_type),
                    self.spark.data_type.simpleString(),
                )
            )
        return self._with_new_scol(scol)

    dt = CachedAccessor("dt", DatetimeMethods)
    str = CachedAccessor("str", StringMethods)
    cat = CachedAccessor("cat", CategoricalAccessor)
    plot = CachedAccessor("plot", KoalasPlotAccessor)

    def _apply_series_op(self, op: Callable[["Series"], "Series"], should_resolve: bool = False) -> "Series":
        kser = op(self)
        if should_resolve:
            internal = kser._internal.resolved_copy
            return first_series(DataFrame(internal))
        else:
            return kser

    def _reduce_for_stat_function(self, sfun: Callable[..., Any], name: str, axis: Optional[Any] = None, numeric_only: Optional[bool] = None, **kwargs: Any) -> Any:
        from inspect import signature
        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError("Series does not support columns axis.")
        num_args = len(signature(sfun).parameters)
        spark_column = self.spark.column
        spark_type = self.spark.data_type
        if num_args == 1:
            scol = sfun(spark_column)
        else:
            assert num_args == 2
            scol = sfun(spark_column, spark_type)
        min_count = kwargs.get("min_count", 0)
        if min_count > 0:
            scol = F.when(Frame._count_expr(spark_column, spark_type) >= min_count, scol)
        result = unpack_scalar(self._internal.spark_frame.select(scol))
        return result if result is not None else np.nan

    def __getitem__(self, key: Any) -> Any:
        try:
            if (isinstance(key, slice) and any(isinstance(n, int) for n in [key.start, key.stop])) or (
                isinstance(key, int)
                and not isinstance(self.index.spark.data_type, (IntegerType, LongType))
            ):
                return self.iloc[key]
            return self.loc[key]
        except SparkPandasIndexingError:
            raise KeyError(
                "Key length ({}) exceeds index depth ({})".format(
                    len(key), self._internal.index_level
                )
            )

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        if hasattr(MissingPandasLikeSeries, item):
            property_or_func = getattr(MissingPandasLikeSeries, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError("'Series' object has no attribute '{}'".format(item))

    def _to_internal_pandas(self) -> pd.Series:
        return self._kdf._internal.to_pandas_frame[self.name]

    def __repr__(self) -> str:
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            return self._to_internal_pandas().to_string(name=self.name, dtype=self.dtype)
        pser = self._kdf._get_or_create_repr_pandas_cache(max_display_count)[self.name]
        pser_length = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit("\n", 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group("length")
                dtype_name = str(self.dtype.name)
                if self.name is None:
                    footer = "\ndtype: {dtype}\nShowing only the first {length}".format(
                        length=length, dtype=pprint_thing(dtype_name)
                    )
                else:
                    footer = (
                        "\nName: {name}, dtype: {dtype}"
                        "\nShowing only the first {length}".format(
                            length=length, name=self.name, dtype=pprint_thing(dtype_name)
                        )
                    )
                return rest + footer
        return pser.to_string(name=self.name, dtype=self.dtype)

    def __dir__(self) -> List[str]:
        if not isinstance(self.spark.data_type, StructType):
            fields = []
        else:
            fields = [f for f in self.spark.data_type.fieldNames() if " " not in f]
        return super().__dir__() + fields

    def __iter__(self) -> Iterable[Any]:
        return MissingPandasLikeSeries.__iter__(self)

    if sys.version_info >= (3, 7):
        def __class_getitem__(cls, params: Any) -> Any:
            return _create_type_for_series_type(params)
    elif (3, 5) <= sys.version_info < (3, 7):
        is_series = None


# End of class Series


# End of module; the module provides unpack_scalar and first_series along with the Series class.
