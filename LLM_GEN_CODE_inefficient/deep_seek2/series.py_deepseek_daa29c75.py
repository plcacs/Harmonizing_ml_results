from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar, Union, cast, Mapping, Callable
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

def _create_type_for_series_type(param: Any) -> Type[SeriesType]:
    from databricks.koalas.typedef import NameTypeHolder

    if isinstance(param, ExtensionDtype):
        new_class = type("NameType", (NameTypeHolder,), {})
        new_class.tpe = param
    else:
        new_class = param.type if isinstance(param, np.dtype) else param

    return SeriesType[new_class]

class Series(Frame, IndexOpsMixin, Generic[T]):
    def __init__(self, data: Any = None, index: Any = None, dtype: Any = None, name: Any = None, copy: bool = False, fastpath: bool = False) -> None:
        assert data is not None

        if isinstance(data, DataFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath

            self._anchor = data
            self._col_label = index
        else:
            if isinstance(data, pd.Series):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert not fastpath
                s = data
            else:
                s = pd.Series(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
                )
            internal = InternalFrame.from_pandas(pd.DataFrame(s))
            if s.name is None:
                internal = internal.copy(column_labels=[None])
            anchor = DataFrame(internal)

            self._anchor = anchor
            self._col_label = anchor._internal.column_labels[0]
            object.__setattr__(anchor, "_kseries", {self._column_label: self})

    @property
    def _kdf(self) -> DataFrame:
        return self._anchor

    @property
    def _internal(self) -> InternalFrame:
        return self._kdf._internal.select_column(self._column_label)

    @property
    def _column_label(self) -> Tuple:
        return self._col_label

    def _update_anchor(self, kdf: DataFrame) -> None:
        assert kdf._internal.column_labels == [self._column_label], (
            kdf._internal.column_labels,
            [self._column_label],
        )
        self._anchor = kdf
        object.__setattr__(kdf, "_kseries", {self._column_label: self})

    def _with_new_scol(self, scol: spark.Column, *, dtype: Any = None) -> "Series":
        internal = self._internal.copy(
            data_spark_columns=[scol.alias(name_like_string(self._column_label))],
            data_dtypes=[dtype],
        )
        return first_series(DataFrame(internal))

    spark = CachedAccessor("spark", SparkSeriesMethods)

    @property
    def dtypes(self) -> np.dtype:
        return self.dtype

    @property
    def axes(self) -> List:
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

    add.__doc__ = _flex_doc_SERIES.format(
        desc="Addition",
        op_name="+",
        equiv="series + other",
        reverse="radd",
        series_examples=_add_example_SERIES,
    )

    def radd(self, other: Any) -> "Series":
        return other + self

    radd.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Addition",
        op_name="+",
        equiv="other + series",
        reverse="add",
        series_examples=_add_example_SERIES,
    )

    def div(self, other: Any) -> "Series":
        return self / other

    div.__doc__ = _flex_doc_SERIES.format(
        desc="Floating division",
        op_name="/",
        equiv="series / other",
        reverse="rdiv",
        series_examples=_div_example_SERIES,
    )

    divide = div

    def rdiv(self, other: Any) -> "Series":
        return other / self

    rdiv.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Floating division",
        op_name="/",
        equiv="other / series",
        reverse="div",
        series_examples=_div_example_SERIES,
    )

    def truediv(self, other: Any) -> "Series":
        return self / other

    truediv.__doc__ = _flex_doc_SERIES.format(
        desc="Floating division",
        op_name="/",
        equiv="series / other",
        reverse="rtruediv",
        series_examples=_div_example_SERIES,
    )

    def rtruediv(self, other: Any) -> "Series":
        return other / self

    rtruediv.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Floating division",
        op_name="/",
        equiv="other / series",
        reverse="truediv",
        series_examples=_div_example_SERIES,
    )

    def mul(self, other: Any) -> "Series":
        return self * other

    mul.__doc__ = _flex_doc_SERIES.format(
        desc="Multiplication",
        op_name="*",
        equiv="series * other",
        reverse="rmul",
        series_examples=_mul_example_SERIES,
    )

    multiply = mul

    def rmul(self, other: Any) -> "Series":
        return other * self

    rmul.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Multiplication",
        op_name="*",
        equiv="other * series",
        reverse="mul",
        series_examples=_mul_example_SERIES,
    )

    def sub(self, other: Any) -> "Series":
        return self - other

    sub.__doc__ = _flex_doc_SERIES.format(
        desc="Subtraction",
        op_name="-",
        equiv="series - other",
        reverse="rsub",
        series_examples=_sub_example_SERIES,
    )

    subtract = sub

    def rsub(self, other: Any) -> "Series":
        return other - self

    rsub.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Subtraction",
        op_name="-",
        equiv="other - series",
        reverse="sub",
        series_examples=_sub_example_SERIES,
    )

    def mod(self, other: Any) -> "Series":
        return self % other

    mod.__doc__ = _flex_doc_SERIES.format(
        desc="Modulo",
        op_name="%",
        equiv="series % other",
        reverse="rmod",
        series_examples=_mod_example_SERIES,
    )

    def rmod(self, other: Any) -> "Series":
        return other % self

    rmod.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Modulo",
        op_name="%",
        equiv="other % series",
        reverse="mod",
        series_examples=_mod_example_SERIES,
    )

    def pow(self, other: Any) -> "Series":
        return self ** other

    pow.__doc__ = _flex_doc_SERIES.format(
        desc="Exponential power of series",
        op_name="**",
        equiv="series ** other",
        reverse="rpow",
        series_examples=_pow_example_SERIES,
    )

    def rpow(self, other: Any) -> "Series":
        return other ** self

    rpow.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Exponential power",
        op_name="**",
        equiv="other ** series",
        reverse="pow",
        series_examples=_pow_example_SERIES,
    )

    def floordiv(self, other: Any) -> "Series":
        return self // other

    floordiv.__doc__ = _flex_doc_SERIES.format(
        desc="Integer division",
        op_name="//",
        equiv="series // other",
        reverse="rfloordiv",
        series_examples=_floordiv_example_SERIES,
    )

    def rfloordiv(self, other: Any) -> "Series":
        return other // self

    rfloordiv.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Integer division",
        op_name="//",
        equiv="other // series",
        reverse="floordiv",
        series_examples=_floordiv_example_SERIES,
    )

    koalas = CachedAccessor("koalas", KoalasSeriesMethods)

    def eq(self, other: Any) -> bool:
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

    def map(self, arg: Union[Callable, dict]) -> "Series":
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

    def alias(self, name: str) -> "Series":
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

    def rename(self, index: Any = None, **kwargs) -> "Series":
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
        self, mapper: Optional[Any] = None, index: Optional[Any] = None, inplace: bool = False
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
        self, level: Any = None, drop: bool = False, name: Any = None, inplace: bool = False
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

    def to_frame(self, name: Union[Any, Tuple] = None) -> DataFrame:
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
        buf: Any = None,
        na_rep: str = "NaN",
        float_format: Any = None,
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

    def to_clipboard(self, excel: bool = True, sep: Any = None, **kwargs) -> None:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_clipboard, pd.Series.to_clipboard, args
        )

    to_clipboard.__doc__ = DataFrame.to_clipboard.__doc__

    def to_dict(self, into: Any = dict) -> Mapping:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_dict, pd.Series.to_dict, args
        )

    def to_latex(
        self,
        buf: Any = None,
        columns: Any = None,
        col_space: Any = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Any = None,
        float_format: Any = None,
        sparsify: Any = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: Any = None,
        longtable: Any = None,
        escape: Any = None,
        encoding: Any = None,
        decimal: str = ".",
        multicolumn: Any = None,
        multicolumn_format: Any = None,
        multirow: Any = None,
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

    def to_list(self) -> List:
        return self._to_internal_pandas().tolist()

    tolist = to_list

    def drop_duplicates(self, keep: str = "first", inplace: bool = False) -> Optional["Series"]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].drop_duplicates(keep=keep)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def reindex(self, index: Optional[Any] = None, fill_value: Optional[Any] = None) -> "Series":
        return first_series(self.to_frame().reindex(index=index, fill_value=fill_value)).rename(
            self.name
        )

    def reindex_like(self, other: Union["Series", "DataFrame"]) -> "Series":
        if isinstance(other, (Series, DataFrame)):
            return self.reindex(index=other.index)
        else:
            raise TypeError("other must be a Koalas Series or DataFrame")

    def fillna(
        self, value: Any = None, method: Any = None, axis: Any = None, inplace: bool = False, limit: Any = None
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

    def _fillna(self, value: Any = None, method: Any = None, axis: Any = None, limit: Any = None, part_cols: Any = ()) -> Any:
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

    def dropna(self, axis: Any = 0, inplace: bool = False, **kwargs) -> Optional["Series"]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].dropna(axis=axis, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def clip(self, lower: Union[float, int] = None, upper: Union[float, int] = None) -> "Series":
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

    def drop(
        self, labels: Any = None, index: Union[Any, Tuple, List[Any], List[Tuple]] = None, level: Any = None
    ) -> "Series":
        return first_series(self._drop(labels=labels, index=index, level=level))

    def _drop(
        self, labels: Any = None, index: Union[Any, Tuple, List[Any], List[Tuple]] = None, level: Any = None
    ) -> Any:
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
                    "If the given index is a list, it "
                    "should only contains names as all tuples or all non tuples "
                    "that contain index names"
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
        kind: str = None,
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

    def hist(self, bins: int = 10, **kwds) -> Any:
        return self.plot.hist(bins, **kwds)

    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def apply(self, func: Callable, args: Any = (), **kwds) -> "Series":
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

    def transpose(self, *args, **kwargs) -> "Series":
        return self.copy()

    T = property(transpose)

    def transform(self, func: Callable, axis: int = 0, *args, **kwargs) -> Union["Series", DataFrame]:
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

    def transform_batch(self, func: Callable, *args, **kwargs) -> "ks.Series":
        warnings.warn(
            "Series.transform_batch is deprecated as of Series.koalas.transform_batch. "
            "Please use the API instead.",
            FutureWarning,
        )
        return self.koalas.transform_batch(func, *args, **kwargs)

    transform_batch.__doc__ = KoalasSeriesMethods.transform_batch