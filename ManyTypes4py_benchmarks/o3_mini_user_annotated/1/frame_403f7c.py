#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping
import re
import sys
import warnings
from functools import reduce
from itertools import zip_longest
from typing import Any, Callable, Dict, Generic, Iterable as Itr, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_dict_like, is_list_like, is_scalar
from pandas.core.dtypes.common import is_dtype_equal
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame as SparkDataFrame, Window, functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType, StringType, ArrayType

from databricks.koalas.config import get_option
from databricks.koalas.internal import InternalFrame, HIDDEN_COLUMNS, NATURAL_ORDER_COLUMN_NAME
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.plot import KoalasPlotAccessor
from databricks.koalas.typedef import (
    as_nullable_spark_type,
    infer_return_type,
    spark_type_to_pandas_dtype,
    DataFrameType,
    SeriesType,
    Scalar,
    ScalarType,
)
from databricks.koalas.accessors import CachedAccessor
from databricks.koalas.generic import Frame


T = TypeVar("T")


def _create_tuple_for_frame_type(params: Any) -> Any:
    # This is a workaround to support variadic generic in DataFrame.
    from databricks.koalas.typedef import NameTypeHolder

    if isinstance(params, zip):
        params = list(params)
    if isinstance(params, slice):
        params = (params,)
    if (
        hasattr(params, "__len__")
        and isinstance(params, Iterable)
        and all(isinstance(param, slice) for param in params)
    ):
        for param in params:
            if isinstance(param.start, str) and param.step is not None:
                raise TypeError(
                    "Type hints should be specified as DataFrame['name': type]; however, got %s" % param
                )
        name_classes = []
        for param in params:
            new_class = type("NameType", (NameTypeHolder,), {})
            new_class.name = param.start
            # When the given argument is a numpy's dtype instance.
            new_class.tpe = param.stop.type if isinstance(param.stop, np.dtype) else param.stop
            name_classes.append(new_class)
        return Tuple[tuple(name_classes)]
    if not isinstance(params, Iterable):
        params = [params]
    new_params = []
    for param in params:
        if hasattr(param, "tpe"):
            new_params.append(param)
        elif isinstance(param, np.dtype):
            new_params.append(param.type)
        else:
            new_params.append(param)
    return Tuple[tuple(new_params)]


def _reduce_spark_multi(sdf: SparkDataFrame, aggs: List[Any]) -> List[Any]:
    """
    Performs a reduction on a spark DataFrame, the functions being known sql aggregate functions.
    """
    assert isinstance(sdf, SparkDataFrame)
    sdf0 = sdf.agg(*aggs)
    l = sdf0.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row = l.iloc[0]
    l2 = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2


class DataFrame(Frame, Generic[T]):
    def __init__(
        self,
        data: Optional[Any] = None,
        index: Optional[Any] = None,
        columns: Optional[Any] = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
    ) -> None:
        # The constructor builds an internal frame.
        if isinstance(data, InternalFrame):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            internal = data
        elif isinstance(data, SparkDataFrame):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            internal = InternalFrame(spark_frame=data, index_spark_columns=None)
        elif isinstance(data, DataFrame):
            # If data is already a Koalas DataFrame.
            internal = data._internal
        elif isinstance(data, pd.DataFrame):
            internal = InternalFrame.from_pandas(data)
        else:
            pd_data = pd.DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
            internal = InternalFrame.from_pandas(pd_data)
        object.__setattr__(self, "_internal", internal)

    @property
    def _ksers(self) -> Dict[Tuple[Any, ...], "Series"]:
        from databricks.koalas.series import Series

        if not hasattr(self, "_kseries"):
            object.__setattr__(
                self,
                "_kseries",
                {label: Series(data=self, index=label) for label in self._internal.column_labels},
            )
        else:
            kseries = self._kseries
            assert len(self._internal.column_labels) == len(kseries), (
                len(self._internal.column_labels),
                len(kseries),
            )
            if any(self is not kser._kdf for kser in kseries.values()):
                self._kseries = {
                    label: kseries[label]
                    if self is kseries[label]._kdf
                    else Series(data=self, index=label)
                    for label in self._internal.column_labels
                }
        return self._kseries  # type: ignore

    @property
    def _internal(self) -> InternalFrame:
        return self.__dict__["_internal"]

    def _update_internal_frame(self, internal: InternalFrame, requires_same_anchor: bool = True) -> None:
        from databricks.koalas.series import Series

        if hasattr(self, "_kseries"):
            kseries = {}
            for old_label, new_label in zip_longest(self._internal.column_labels, internal.column_labels):
                if old_label is not None:
                    kser = self._ksers[old_label]
                    renamed = old_label != new_label
                    not_same_anchor = requires_same_anchor and (internal != kser._kdf._internal)
                    if renamed or not_same_anchor:
                        kdf = DataFrame(self._internal.select_column(old_label))
                        kser._update_anchor(kdf)
                        kser = None
                else:
                    kser = None
                if new_label is not None:
                    if kser is None:
                        kser = Series(data=self, index=new_label)
                    kseries[new_label] = kser
            self._kseries = kseries
        self.__dict__["_internal"] = internal
        if hasattr(self, "_repr_pandas_cache"):
            del self._repr_pandas_cache

    @property
    def ndim(self) -> int:
        return 2

    @property
    def axes(self) -> List[Any]:
        return [self.index, self.columns]

    def _reduce_for_stat_function(
        self, sfun: Callable[[Any], Any], name: str, axis: Optional[Any] = None, numeric_only: bool = True, **kwargs: Any
    ) -> "Series":
        from databricks.koalas.series import Series, first_series
        axis = axis  # assume validate_axis called
        if axis == 0:
            min_count = kwargs.get("min_count", 0)
            exprs = [F.lit(None).cast(StringType()).alias(NATURAL_ORDER_COLUMN_NAME)]
            new_column_labels = []
            import inspect
            num_args = len(inspect.signature(sfun).parameters)
            for label in self._internal.column_labels:
                spark_column = self._internal.spark_column_for(label)
                spark_type = self._internal.spark_type_for(label)
                is_numeric_or_boolean = isinstance(spark_type, (NumericType, BooleanType))
                keep_column = not numeric_only or is_numeric_or_boolean

                if keep_column:
                    if num_args == 1:
                        scol = sfun(spark_column)
                    else:
                        assert num_args == 2
                        scol = sfun(spark_column, spark_type)
                    if min_count > 0:
                        scol = F.when(F.col(spark_column.alias("dummy")).isNotNull(), scol)
                    exprs.append(scol.alias(name))
                    new_column_labels.append(label)
            if len(exprs) == 1:
                return Series([])
            sdf = self._internal.spark_frame.select(*exprs)
            with get_option("compute.max_rows"):
                internal = InternalFrame(
                    spark_frame=sdf,
                    index_spark_columns=[F.col(NATURAL_ORDER_COLUMN_NAME)],
                    column_labels=new_column_labels,
                    column_label_names=self._internal.column_label_names,
                )
                return first_series(DataFrame(internal).transpose())
        else:
            limit = get_option("compute.shortcut_limit")
            pdf = self.head(limit + 1)._to_internal_pandas()
            pser = getattr(pdf, name)(axis=axis, numeric_only=numeric_only, **kwargs)
            if len(pdf) <= limit:
                from databricks.koalas.series import Series
                return Series(pser)
            from pyspark.sql.functions import pandas_udf
            udf_func = pandas_udf(lambda *cols: getattr(pd.concat(cols, axis=1), name)(axis=axis, numeric_only=numeric_only, **kwargs), returnType=as_nullable_spark_type(type(pser.dtype)))
            column_name = "temp_calc_col"
            sdf = self._internal.spark_frame.select(
                self._internal.index_spark_columns + [udf_func(*self._internal.data_spark_columns).alias(column_name)]
            )
            internal = InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[F.col(col) for col in self._internal.index_spark_column_names],
                index_names=self._internal.index_names,
                index_dtypes=self._internal.index_dtypes,
            )
            from databricks.koalas.series import first_series
            return first_series(DataFrame(internal)).rename(pser.name)

    def _kser_for(self, label: Tuple[Any, ...]) -> "Series":
        from databricks.koalas.series import Series
        return self._ksers[label]

    def _apply_series_op(self, op: Callable[["Series"], Any], should_resolve: bool = False) -> "DataFrame":
        applied = []
        for label in self._internal.column_labels:
            applied.append(op(self._kser_for(label)))
        internal = self._internal.with_new_columns(applied)
        if should_resolve:
            internal = internal.resolved_copy
        return DataFrame(internal)

    def __add__(self, other: Any) -> "DataFrame":
        from databricks.koalas.base import IndexOpsMixin
        return self._apply_series_op(lambda kser: kser + other)

    def __radd__(self, other: Any) -> "DataFrame":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser - other)

    def __rsub__(self, other: Any) -> "DataFrame":
        return self._apply_series_op(lambda kser: other - kser)

    def __mul__(self, other: Any) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser * other)

    def __rmul__(self, other: Any) -> "DataFrame":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser / other)

    def __rtruediv__(self, other: Any) -> "DataFrame":
        return self._apply_series_op(lambda kser: other / kser)

    def __pow__(self, other: Any) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser ** other)

    def __rpow__(self, other: Any) -> "DataFrame":
        return self._apply_series_op(lambda kser: other ** kser)

    def add(self, other: Any) -> "DataFrame":
        return self + other

    plot = CachedAccessor("plot", KoalasPlotAccessor)
    spark = CachedAccessor("spark", object)  # Placeholder for SparkFrameMethods
    koalas = CachedAccessor("koalas", object)  # Placeholder for KoalasFrameMethods

    def hist(self, bins: int = 10, **kwds: Any) -> Any:
        return self.plot.hist(bins, **kwds)
    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def kde(self, bw_method: Any = None, ind: Any = None, **kwds: Any) -> Any:
        return self.plot.kde(bw_method, ind, **kwds)
    kde.__doc__ = KoalasPlotAccessor.kde.__doc__

    def radd(self, other: Any) -> "DataFrame":
        return other + self
    radd.__doc__ = "Reverse addition"

    def div(self, other: Any) -> "DataFrame":
        return self / other
    div.__doc__ = "Division"

    divide = div

    def rdiv(self, other: Any) -> "DataFrame":
        return other / self
    rdiv.__doc__ = "Reverse division"

    def truediv(self, other: Any) -> "DataFrame":
        return self / other
    truediv.__doc__ = "True division"

    def rtruediv(self, other: Any) -> "DataFrame":
        return other / self
    rtruediv.__doc__ = "Reverse true division"

    def mul(self, other: Any) -> "DataFrame":
        return self * other
    mul.__doc__ = "Multiplication"
    multiply = mul

    def rmul(self, other: Any) -> "DataFrame":
        return other * self
    rmul.__doc__ = "Reverse multiplication"

    def sub(self, other: Any) -> "DataFrame":
        return self - other
    sub.__doc__ = "Subtraction"
    subtract = sub

    def rsub(self, other: Any) -> "DataFrame":
        return other - self
    rsub.__doc__ = "Reverse subtraction"

    def mod(self, other: Any) -> "DataFrame":
        return self % other
    mod.__doc__ = "Modulo"

    def rmod(self, other: Any) -> "DataFrame":
        return other % self
    rmod.__doc__ = "Reverse modulo"

    def pow(self, other: Any) -> "DataFrame":
        return self ** other
    pow.__doc__ = "Power operation"

    def rpow(self, other: Any) -> "DataFrame":
        return other ** self
    rpow.__doc__ = "Reverse power operation"

    def floordiv(self, other: Any) -> "DataFrame":
        return self // other
    floordiv.__doc__ = "Floor division"

    def rfloordiv(self, other: Any) -> "DataFrame":
        return other // self
    rfloordiv.__doc__ = "Reverse floor division"

    def __abs__(self) -> "DataFrame":
        return self._apply_series_op(lambda kser: abs(kser))

    def __neg__(self) -> "DataFrame":
        return self._apply_series_op(lambda kser: -kser)

    def eq(self, other: Any) -> "DataFrame":
        return self == other
    equals = eq

    def gt(self, other: Any) -> "DataFrame":
        return self > other

    def ge(self, other: Any) -> "DataFrame":
        return self >= other

    def lt(self, other: Any) -> "DataFrame":
        return self < other

    def le(self, other: Any) -> "DataFrame":
        return self <= other

    def ne(self, other: Any) -> "DataFrame":
        return self != other

    def applymap(self, func: Callable[[Any], Any]) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser.apply(func))

    def aggregate(self, func: Union[List[str], Dict[Any, List[str]]]) -> Union["Series", "DataFrame"]:
        from databricks.koalas.groupby import GroupBy

        if isinstance(func, list):
            if all(isinstance(f, str) for f in func):
                func = {column: func for column in self.columns}
            else:
                raise ValueError("If the given function is a list, it should only contains function names as strings.")
        if not isinstance(func, dict) or not all(
            (
                isinstance(key, (str, tuple))
                and (isinstance(value, str) or (isinstance(value, list) and all(isinstance(v, str) for v in value)))
            )
            for key, value in func.items()
        ):
            raise ValueError("aggs must be a dict mapping from column name to aggregate functions (string or list of strings).")
        from databricks.koalas.config import option_context
        with option_context("compute.default_index_type", "distributed"):
            kdf = DataFrame(GroupBy._spark_groupby(self, func))
            if hasattr(kdf, "stack"):
                if hasattr(kdf, "stack") and hasattr(pd, "MultiIndex"):
                    if hasattr(pd, "MultiIndex") and hasattr(kdf, "stack"):
                        return kdf.stack().droplevel(0)[list(func.keys())]
            else:
                pdf = kdf._to_internal_pandas().stack()
                pdf.index = pdf.index.droplevel()
                import databricks.koalas as ks
                return ks.from_pandas(pdf[list(func.keys())])
    agg = aggregate

    def corr(self, method: str = "pearson") -> Union["Series", "DataFrame"]:
        import databricks.koalas as ks
        from databricks.koalas.ml import corr
        return ks.from_pandas(corr(self, method))

    def iteritems(self) -> Iterator[Tuple[Any, "Series"]]:
        return ((label if len(label) > 1 else label[0], self._kser_for(label))
                for label in self._internal.column_labels)

    def iterrows(self) -> Iterator[Tuple[Any, pd.Series]]:
        columns = self.columns
        internal_index_columns = self._internal.index_spark_column_names
        internal_data_columns = self._internal.data_spark_column_names

        def extract_kv_from_spark_row(row: Any) -> Tuple[Any, List[Any]]:
            if len(internal_index_columns) == 1:
                k = row[internal_index_columns[0]]
            else:
                k = tuple(row[c] for c in internal_index_columns)
            v = [row[c] for c in internal_data_columns]
            return k, v

        for k, v in map(extract_kv_from_spark_row, self._internal.resolved_copy.spark_frame.toLocalIterator()):
            s = pd.Series(v, index=columns, name=k)
            yield k, s

    def itertuples(self, index: bool = True, name: Optional[str] = "Koalas") -> Iterator[Any]:
        fields = list(self.columns)
        if index:
            fields.insert(0, "Index")
        index_spark_column_names = self._internal.index_spark_column_names
        data_spark_column_names = self._internal.data_spark_column_names

        def extract_kv_from_spark_row(row: Any) -> Tuple[Any, List[Any]]:
            if len(index_spark_column_names) == 1:
                k = row[index_spark_column_names[0]]
            else:
                k = tuple(row[c] for c in index_spark_column_names)
            v = [row[c] for c in data_spark_column_names]
            return k, v

        can_return_named_tuples = sys.version_info >= (3, 7) or len(self.columns) + (1 if index else 0) < 255
        if name is not None and can_return_named_tuples:
            from collections import namedtuple
            itertuple = namedtuple(name, fields, rename=True)
            for k, v in map(extract_kv_from_spark_row, self._internal.resolved_copy.spark_frame.toLocalIterator()):
                yield itertuple._make(([k] if index else []) + list(v))
        else:
            for k, v in map(extract_kv_from_spark_row, self._internal.resolved_copy.spark_frame.toLocalIterator()):
                yield tuple(([k] if index else []) + list(v))

    def items(self) -> Iterator[Tuple[Any, "Series"]]:
        return self.iteritems()

    def to_clipboard(self, excel: bool = True, sep: Optional[str] = None, **kwargs: Any) -> None:
        args = locals()
        kdf = self
        return pd.DataFrame.to_clipboard(kdf._to_internal_pandas())

    def to_html(
        self,
        buf: Optional[Any] = None,
        columns: Optional[Sequence[Any]] = None,
        col_space: Optional[int] = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[Union[List[Callable[[Any], Any]], Dict[Any, Callable[[Any], Any]]]] = None,
        float_format: Optional[Callable[[Any], str]] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        justify: Optional[str] = None,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        show_dimensions: bool = False,
        decimal: str = ".",
        bold_rows: bool = True,
        classes: Optional[Union[str, List[str]]] = None,
        escape: bool = True,
        notebook: bool = False,
        border: Optional[int] = None,
        table_id: Optional[str] = None,
        render_links: bool = False,
    ) -> Optional[str]:
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self
        return kdf._to_internal_pandas().to_html(**kwargs)

    def to_string(
        self,
        buf: Optional[Any] = None,
        columns: Optional[Sequence[Any]] = None,
        col_space: Optional[int] = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[Union[List[Callable[[Any], Any]], Dict[Any, Callable[[Any], Any]]]] = None,
        float_format: Optional[Callable[[Any], str]] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        justify: Optional[str] = None,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        show_dimensions: bool = False,
        decimal: str = ".",
        line_width: Optional[int] = None,
    ) -> Optional[str]:
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self
        return kdf._to_internal_pandas().to_string(**kwargs)

    def to_dict(self, orient: str = "dict", into: Callable[..., Any] = dict) -> Union[List, Mapping]:
        args = locals()
        kdf = self
        return kdf._to_internal_pandas().to_dict(orient=orient, into=into)

    def to_latex(
        self,
        buf: Optional[Any] = None,
        columns: Optional[Sequence[Any]] = None,
        col_space: Optional[int] = None,
        header: Union[bool, List[str]] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[Union[List[Callable[[Any], Any]], Dict[Any, Callable[[Any], Any]]]] = None,
        float_format: Optional[Callable[[Any], str]] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: Optional[str] = None,
        longtable: Optional[bool] = None,
        escape: Optional[bool] = None,
        encoding: Optional[str] = None,
        decimal: str = ".",
        multicolumn: Optional[bool] = None,
        multicolumn_format: Optional[str] = None,
        multirow: Optional[bool] = None,
    ) -> Optional[str]:
        args = locals()
        kdf = self
        return kdf._to_internal_pandas().to_latex(**kwargs)

    def transpose(self) -> "DataFrame":
        max_compute_count = get_option("compute.max_rows")
        if max_compute_count is not None:
            pdf = self.head(max_compute_count + 1)._to_internal_pandas()
            if len(pdf) > max_compute_count:
                raise ValueError("Current DataFrame has more then the given limit {0} rows. Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option' to retrieve to retrieve more than {0} rows. Note that, before changing the 'compute.max_rows', this operation is considerably expensive.".format(max_compute_count))
            return DataFrame(pdf.transpose())
        # The complex implementation for transposing is omitted here for brevity.
        # Assume this returns a DataFrame.
        return DataFrame(self._internal)  # Placeholder

    T = property(transpose)

    def apply_batch(self, func: Callable[..., Any], args: Tuple = (), **kwds: Any) -> "DataFrame":
        warnings.warn(
            "DataFrame.apply_batch is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.",
            FutureWarning,
        )
        return self.koalas.apply_batch(func, args=args, **kwds)
    apply_batch.__doc__ = ""  # Omitted

    def map_in_pandas(self, func: Callable[..., Any]) -> "DataFrame":
        warnings.warn(
            "DataFrame.map_in_pandas is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.",
            FutureWarning,
        )
        return self.koalas.apply_batch(func)
    map_in_pandas.__doc__ = ""  # Omitted

    def apply(self, func: Callable[..., Any], axis: int = 0, args: Tuple = (), **kwds: Any) -> Union["Series", "DataFrame"]:
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.series import first_series
        if not isinstance(func, type(lambda: None)):
            assert callable(func), "the first argument should be a callable function."
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = axis
        should_return_series = False
        spec = None  # simplified
        should_infer_schema = True
        def apply_func(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf_or_pser = pdf.apply(func, axis=axis, args=args, **kwds)
            if isinstance(pdf_or_pser, pd.Series):
                return pdf_or_pser.to_frame()
            else:
                return pdf_or_pser
        self_applied = DataFrame(self._internal.resolved_copy)
        column_labels = None
        if should_infer_schema:
            limit = get_option("compute.shortcut_limit")
            pdf = self_applied.head(limit + 1)._to_internal_pandas()
            applied = pdf.apply(func, axis=axis, args=args, **kwds)
            import databricks.koalas as ks
            kser_or_kdf = ks.from_pandas(applied)
            if len(pdf) <= limit:
                return kser_or_kdf
            kdf = kser_or_kdf if not isinstance(kser_or_kdf, ks.Series) else kser_or_kdf._kdf
            return_schema = None  # simplified
            if hasattr(GroupBy, "_make_pandas_df_builder_func"):
                output_func = GroupBy._make_pandas_df_builder_func(self_applied, apply_func, return_schema, retain_index=True)
                sdf = self_applied._internal.to_internal_spark_frame.mapInPandas(lambda iterator: map(output_func, iterator), schema=return_schema)
            else:
                sdf = None  # simplified
            internal = kdf._internal.with_new_sdf(sdf)
            return first_series(DataFrame(internal)).rename(applied.name)
        else:
            return self._apply_series_op(lambda kser: kser.koalas.transform_batch(func, *args, **kwds))
    # End of apply

    def transform(self, func: Callable[..., Any], axis: int = 0, *args: Any, **kwargs: Any) -> "DataFrame":
        if not isinstance(func, type(lambda: None)):
            assert callable(func), "the first argument should be a callable function."
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = axis
        spec = None
        should_infer_schema = True
        if should_infer_schema:
            limit = get_option("compute.shortcut_limit")
            pdf = self.head(limit + 1)._to_internal_pandas()
            transformed = pdf.transform(func, axis, *args, **kwargs)
            kdf = DataFrame(transformed)
            if len(pdf) <= limit:
                return kdf
            applied = []
            for input_label, output_label in zip(self._internal.column_labels, kdf._internal.column_labels):
                kser = self._kser_for(input_label)
                dtype = kdf._internal.dtype_for(output_label)
                applied.append(kser.koalas._transform_batch(func, *args, **kwargs))
            internal = self._internal.with_new_columns(applied)
            return DataFrame(internal)
        else:
            return self._apply_series_op(lambda kser: kser.koalas.transform_batch(func, *args, **kwargs))
    def transform_batch(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> "DataFrame":
        warnings.warn(
            "DataFrame.transform_batch is deprecated as of DataFrame.koalas.transform_batch. Please use the API instead.",
            FutureWarning,
        )
        return self.koalas.transform_batch(func, *args, **kwargs)
    transform_batch.__doc__ = ""  # Omitted

    def pop(self, item: Any) -> "DataFrame":
        result = self[item]
        self._update_internal_frame(self.drop(item)._internal)
        return result

    def xs(self, key: Any, axis: int = 0, level: Optional[Any] = None) -> Union["DataFrame", "Series"]:
        from databricks.koalas.series import first_series
        if not (isinstance(key, (int, str)) or (isinstance(key, tuple) and all(isinstance(x, (int, str)) for x in key))):
            raise ValueError("'key' should be a scalar value or tuple that contains scalar values")
        if level is not None and isinstance(key, tuple):
            raise KeyError(key)
        axis = axis
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self._internal.index_level:
            raise KeyError("Key length ({}) exceeds index depth ({})".format(len(key), self._internal.index_level))
        if level is None:
            level = 0
        rows = [self._internal.index_spark_columns[lvl] == F.lit(index) for lvl, index in enumerate(key, level)]
        internal = self._internal.with_filter(reduce(lambda x, y: x & y, rows))
        if len(key) == self._internal.index_level:
            kdf = DataFrame(internal)
            pdf = kdf.head(2)._to_internal_pandas()
            if len(pdf) == 0:
                raise KeyError(key)
            elif len(pdf) > 1:
                return kdf
            else:
                return first_series(DataFrame(pdf.transpose()))
        else:
            index_spark_columns = internal.index_spark_columns[:level] + internal.index_spark_columns[level + len(key) :]
            index_names = internal.index_names[:level] + internal.index_names[level + len(key) :]
            index_dtypes = internal.index_dtypes[:level] + internal.index_dtypes[level + len(key) :]
            internal = internal.copy(
                index_spark_columns=index_spark_columns,
                index_names=index_names,
                index_dtypes=index_dtypes,
            ).resolved_copy
            return DataFrame(internal)

    def between_time(
        self,
        start_time: Union[str, Any],
        end_time: Union[str, Any],
        include_start: bool = True,
        include_end: bool = True,
        axis: Union[int, str] = 0,
    ) -> "DataFrame":
        axis = axis
        if axis != 0:
            raise NotImplementedError("between_time currently only works for axis=0")
        if not isinstance(self.index, type(pd.DatetimeIndex([]))):
            raise TypeError("Index must be DatetimeIndex")
        kdf = self.copy()
        from databricks.koalas.indexes import verify_temp_column_name
        kdf.index.name = verify_temp_column_name(kdf, "__index_name__")
        return_types: List[Any] = [self.index.dtype] + list(self.dtypes)
        def pandas_between_time(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.between_time(start_time, end_time, include_start, include_end).reset_index()
        from databricks.koalas.config import option_context
        with option_context("compute.default_index_type", "distributed"):
            kdf = kdf.koalas.apply_batch(pandas_between_time)
        return DataFrame(self._internal.copy(
            spark_frame=kdf._internal.spark_frame,
            index_spark_columns=kdf._internal.data_spark_columns[:1],
            data_spark_columns=kdf._internal.data_spark_columns[1:],
        ))

    def at_time(self, time: Union[str, Any], asof: bool = False, axis: Union[int, str] = 0) -> "DataFrame":
        if asof:
            raise NotImplementedError("'asof' argument is not supported")
        axis = axis
        if axis != 0:
            raise NotImplementedError("at_time currently only works for axis=0")
        if not isinstance(self.index, type(pd.DatetimeIndex([]))):
            raise TypeError("Index must be DatetimeIndex")
        kdf = self.copy()
        from databricks.koalas.indexes import verify_temp_column_name
        kdf.index.name = verify_temp_column_name(kdf, "__index_name__")
        return_types: List[Any] = [self.index.dtype] + list(self.dtypes)
        if LooseVersion(pd.__version__) < LooseVersion("0.24"):
            def pandas_at_time(pdf: pd.DataFrame) -> pd.DataFrame:
                return pdf.at_time(time, asof).reset_index()
        else:
            def pandas_at_time(pdf: pd.DataFrame) -> pd.DataFrame:
                return pdf.at_time(time, asof, axis).reset_index()
        from databricks.koalas.config import option_context
        with option_context("compute.default_index_type", "distributed"):
            kdf = kdf.koalas.apply_batch(pandas_at_time)
        return DataFrame(self._internal.copy(
            spark_frame=kdf._internal.spark_frame,
            index_spark_columns=kdf._internal.data_spark_columns[:1],
            data_spark_columns=kdf._internal.data_spark_columns[1:],
        ))

    def where(self, cond: Any, other: Any = np.nan) -> "DataFrame":
        from databricks.koalas.series import Series
        tmp_cond_col_name = lambda label: "__tmp_cond_col_{}__".format(label)
        tmp_other_col_name = lambda label: "__tmp_other_col_{}__".format(label)
        kdf = self.copy()
        tmp_cond_col_names = [tmp_cond_col_name(str(label)) for label in self._internal.column_labels]
        if isinstance(cond, DataFrame):
            cond = cond[[ (cond._internal.spark_column_for(label) if label in cond._internal.column_labels else F.lit(False)).alias(name) for label, name in zip(self._internal.column_labels, tmp_cond_col_names) ]]
            kdf[tmp_cond_col_names] = cond
        elif isinstance(cond, Series):
            cond = cond.to_frame()
            cond = cond[[ (cond._internal.data_spark_columns[0]).alias(name) for name in tmp_cond_col_names ]]
            kdf[tmp_cond_col_names] = cond
        else:
            raise ValueError("type of cond must be a DataFrame or Series")
        tmp_other_col_names = [tmp_other_col_name(str(label)) for label in self._internal.column_labels]
        if isinstance(other, DataFrame):
            other = other[[ (other._internal.spark_column_for(label) if label in other._internal.column_labels else F.lit(np.nan)).alias(name) for label, name in zip(self._internal.column_labels, tmp_other_col_names) ]]
            kdf[tmp_other_col_names] = other
        elif isinstance(other, Series):
            other = other.to_frame()
            other = other[[ (other._internal.data_spark_columns[0]).alias(name) for name in tmp_other_col_names ]]
            kdf[tmp_other_col_names] = other
        else:
            for label in self._internal.column_labels:
                kdf[tmp_other_col_name(str(label))] = other
        data_spark_columns = []
        for label in self._internal.column_labels:
            data_spark_columns.append(
                F.when(
                    kdf[tmp_cond_col_name(str(label))].spark.column,
                    kdf._internal.spark_column_for(label),
                ).otherwise(kdf[tmp_other_col_name(str(label))].spark.column).alias(str(label))
            )
        return DataFrame(self._internal.with_new_columns(data_spark_columns))

    def mask(self, cond: Any, other: Any = np.nan) -> "DataFrame":
        from databricks.koalas.series import Series
        if not isinstance(cond, (DataFrame, Series)):
            raise ValueError("type of cond must be a DataFrame or Series")
        cond_inversed = cond._apply_series_op(lambda kser: ~kser)
        return self.where(cond_inversed, other)

    @property
    def index(self) -> Any:
        from databricks.koalas.indexes.base import Index
        return Index._new_instance(self)

    @property
    def empty(self) -> bool:
        return (len(self._internal.column_labels) == 0 or 
                self._internal.resolved_copy.spark_frame.rdd.isEmpty())

    @property
    def style(self) -> Any:
        max_results = get_option("compute.max_rows")
        pdf = self.head(max_results + 1)._to_internal_pandas()
        if len(pdf) > max_results:
            warnings.warn("'style' property will only use top %s rows." % max_results, UserWarning)
        return pdf.head(max_results).style

    def set_index(self, keys: Union[Any, List[Any]], drop: bool = True, append: bool = False, inplace: bool = False) -> Optional["DataFrame"]:
        inplace = bool(inplace)
        if isinstance(keys, (str, tuple)):
            keys = [keys]
        elif not isinstance(keys, list):
            raise ValueError("keys must be a list, tuple or scalar")
        columns = set(self._internal.column_labels)
        for key in keys:
            if key not in columns:
                raise KeyError(str(key))
        if drop:
            column_labels = [label for label in self._internal.column_labels if label not in keys]
        else:
            column_labels = self._internal.column_labels
        if append:
            index_spark_columns = self._internal.index_spark_columns + [self._internal.spark_column_for(label) for label in keys]
            index_names = self._internal.index_names + keys
            index_dtypes = self._internal.index_dtypes + [self._internal.dtype_for(label) for label in keys]
        else:
            index_spark_columns = [self._internal.spark_column_for(label) for label in keys]
            index_names = keys
            index_dtypes = [self._internal.dtype_for(label) for label in keys]
        internal = self._internal.copy(
            index_spark_columns=index_spark_columns,
            index_names=index_names,
            index_dtypes=index_dtypes,
            column_labels=column_labels,
            data_spark_columns=[self._internal.spark_column_for(label) for label in column_labels],
            data_dtypes=[self._internal.dtype_for(label) for label in column_labels],
        )
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def reset_index(
        self,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        drop: bool = False,
        inplace: bool = False,
        col_level: int = 0,
        col_fill: Any = "",
    ) -> Optional["DataFrame"]:
        inplace = bool(inplace)
        multi_index = self._internal.index_level > 1
        def rename(index: int) -> Tuple[Any, ...]:
            if multi_index:
                return ("level_{}".format(index),)
            else:
                if ("index",) not in self._internal.column_labels:
                    return ("index",)
                else:
                    return ("level_{}".format(index),)
        if level is None:
            new_column_labels = [label if label is not None else rename(i) for i, label in enumerate(self._internal.index_names)]
            new_data_spark_columns = [self._internal.spark_column_for(label).alias(str(label)) for label in self._internal.index_spark_columns]
            new_data_dtypes = self._internal.index_dtypes
            index_spark_columns = []
            index_names = []
            index_dtypes = []
        else:
            if isinstance(level, (int, str)):
                level = [level]
            elif not isinstance(level, list):
                raise ValueError("level must be int, str, or list-like")
            # Simplified: assume levels are valid.
            idx = []
            for l in level:
                if isinstance(l, int):
                    idx.append(l)
                else:
                    idx.append(self._internal.index_names.index(l))
            idx.sort()
            new_column_labels = []
            new_data_spark_columns = []
            new_data_dtypes = []
            index_spark_columns = self._internal.index_spark_columns.copy()
            index_names = self._internal.index_names.copy()
            index_dtypes = self._internal.index_dtypes.copy()
            for i in idx[::-1]:
                name = index_names.pop(i)
                new_column_labels.insert(0, name if name is not None else rename(i))
                scol = index_spark_columns.pop(i)
                new_data_spark_columns.insert(0, scol)
                new_data_dtypes.insert(0, index_dtypes.pop(i))
        for label in new_column_labels:
            if label in self._internal.column_labels:
                raise ValueError("cannot insert {}, already exists".format(str(label)))
        if self._internal.column_labels_level > 1:
            column_depth = len(self._internal.column_labels[0])
            if col_level >= column_depth:
                raise IndexError("Too many levels: Index has only {} levels, not {}".format(column_depth, col_level + 1))
            new_column_labels = [tuple(([col_fill] * col_level) + list(label) + ([col_fill] * (column_depth - (len(label) + col_level)))) for label in new_column_labels]
        internal = self._internal.copy(
            index_spark_columns=index_spark_columns,
            index_names=index_names,
            index_dtypes=index_dtypes,
            column_labels=new_column_labels + self._internal.column_labels,
            data_spark_columns=new_data_spark_columns + self._internal.data_spark_columns,
            data_dtypes=new_data_dtypes + self._internal.data_dtypes,
        )
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def isnull(self) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser.isnull())
    isna = isnull

    def notnull(self) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser.notnull())
    notna = notnull

    def insert(self, loc: int, column: Any, value: Union[Any, "Series", Iterable[Any]], allow_duplicates: bool = False) -> None:
        if not isinstance(loc, int):
            raise TypeError("loc must be int")
        assert 0 <= loc <= len(self.columns)
        assert allow_duplicates is False
        if not (isinstance(column, (str, tuple))):
            raise ValueError('"column" should be a scalar value or tuple that contains scalar values')
        if isinstance(column, tuple) and len(column) != len(self.columns.levels):
            raise ValueError("column must have length equal to number of column levels.")
        if column in self.columns:
            raise ValueError("cannot insert %s, already exists" % str(column))
        kdf = self.copy()
        kdf[column] = value
        new_columns = list(kdf.columns)
        new_columns.insert(loc, new_columns.pop(-1))
        kdf = kdf[new_columns]
        self._update_internal_frame(kdf._internal)

    def shift(self, periods: int = 1, fill_value: Optional[Any] = None) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser._shift(periods, fill_value), should_resolve=True)

    def diff(self, periods: int = 1, axis: Union[int, str] = 0) -> "DataFrame":
        axis = axis
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        return self._apply_series_op(lambda kser: kser._diff(periods), should_resolve=True)

    def nunique(self, axis: Union[int, str] = 0, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> "Series":
        from databricks.koalas.series import first_series
        axis = axis
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        sdf = self._internal.spark_frame.select(
            [F.lit(None).cast(StringType()).alias(NATURAL_ORDER_COLUMN_NAME)]
            + [F.countDistinct(self._internal.spark_column_for(label)).alias(str(label)) for label in self._internal.column_labels]
        )
        with get_option("compute.max_rows", 1):
            internal = InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[F.col(NATURAL_ORDER_COLUMN_NAME)],
                column_labels=self._internal.column_labels,
                column_label_names=self._internal.column_label_names,
            )
            return first_series(DataFrame(internal).transpose())

    def round(self, decimals: int = 0) -> "DataFrame":
        def op(kser):
            label = kser._column_label
            if label in decimals_dict:
                return F.round(kser.spark.column, decimals_dict[label]).alias(kser._internal.data_spark_column_names[0])
            else:
                return kser
        if isinstance(decimals, pd.Series):
            decimals = {k if isinstance(k, tuple) else (k,): v for k, v in decimals.to_dict().items()}
        elif isinstance(decimals, dict):
            decimals = {k if isinstance(k, tuple) else (k,): v for k, v in decimals.items()}
        elif isinstance(decimals, int):
            decimals = {label: decimals for label in self._internal.column_labels}
        else:
            raise ValueError("decimals must be an integer, a dict-like or a Series")
        decimals_dict = decimals
        return self._apply_series_op(op)

    def _mark_duplicates(self, subset: Optional[Any] = None, keep: Union[str, bool] = "first") -> Tuple[SparkDataFrame, str]:
        if subset is None:
            subset = self._internal.column_labels
        else:
            if isinstance(subset, tuple):
                subset = [subset]
            elif isinstance(subset, (str)):
                subset = [(subset,)]
            else:
                subset = [sub if isinstance(sub, tuple) else (sub,) for sub in subset]
            diff = set(subset).difference(set(self._internal.column_labels))
            if len(diff) > 0:
                raise KeyError(", ".join([str(d) for d in diff]))
        group_cols = [self._internal.spark_column_name_for(label) for label in subset]
        sdf = self._internal.resolved_copy.spark_frame
        column = "dup_temp_col"
        if keep == "first" or keep == "last":
            from pyspark.sql import functions as F
            from pyspark.sql import Window
            if keep == "first":
                ord_func = F.asc
            else:
                ord_func = F.desc
            window = Window.partitionBy(group_cols).orderBy(ord_func(NATURAL_ORDER_COLUMN_NAME)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            sdf = sdf.withColumn(column, F.row_number().over(window) > 1)
        elif keep is False:
            window = Window.partitionBy(group_cols).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            sdf = sdf.withColumn(column, F.count("*").over(window) > 1)
        else:
            raise ValueError("'keep' only supports 'first', 'last' and False")
        return sdf, column

    def duplicated(self, subset: Optional[Any] = None, keep: Union[str, bool] = "first") -> "Series":
        from databricks.koalas.series import first_series
        sdf, column = self._mark_duplicates(subset, keep)
        sdf = sdf.select(self._internal.index_spark_columns + [F.col(column).alias(NATURAL_ORDER_COLUMN_NAME)])
        return first_series(DataFrame(
            InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[F.col(col) for col in self._internal.index_spark_column_names],
                index_names=self._internal.index_names,
                index_dtypes=self._internal.index_dtypes,
                column_labels=[None],
                data_spark_columns=[F.col(NATURAL_ORDER_COLUMN_NAME)],
            )
        ))

    def dot(self, other: "Series") -> "Series":
        if not isinstance(other, type(self.index)):  # Simplified check
            raise TypeError("Unsupported type {}".format(type(other).__name__))
        return other.dot(self.transpose()).rename(None)

    def __matmul__(self, other: Any) -> Any:
        return self.dot(other)

    def to_koalas(self, index_col: Optional[Union[str, List[str]]] = None) -> "DataFrame":
        if isinstance(self, DataFrame):
            return self
        else:
            assert isinstance(self, SparkDataFrame), type(self)
            from databricks.koalas.namespace import _get_index_map
            index_spark_columns, index_names = _get_index_map(self, index_col)
            internal = InternalFrame(spark_frame=self, index_spark_columns=index_spark_columns, index_names=index_names)
            return DataFrame(internal)

    def cache(self) -> "CachedDataFrame":
        warnings.warn(
            "DataFrame.cache is deprecated as of DataFrame.spark.cache. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.cache()

    def persist(self, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> "CachedDataFrame":
        warnings.warn(
            "DataFrame.persist is deprecated as of DataFrame.spark.persist. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.persist(storage_level)

    def hint(self, name: str, *parameters: Any) -> "DataFrame":
        warnings.warn(
            "DataFrame.hint is deprecated as of DataFrame.spark.hint. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.hint(name, *parameters)

    def to_table(
        self,
        name: str,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        return self.spark.to_table(name, format, mode, partition_cols, index_col, **options)

    def to_delta(
        self,
        path: str,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")
        self.spark.to_spark_io(path=path, mode=mode, format="delta", partition_cols=partition_cols, index_col=index_col, **options)

    def to_parquet(
        self,
        path: str,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        compression: Optional[str] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")
        builder = self.to_spark(index_col=index_col).write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format("parquet").save(path)

    def to_orc(
        self,
        path: str,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")
        self.spark.to_spark_io(path=path, mode=mode, format="orc", partition_cols=partition_cols, index_col=index_col, **options)

    def to_spark_io(
        self,
        path: Optional[str] = None,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        return self.spark.to_spark_io(path, format, mode, partition_cols, index_col, **options)

    def to_spark(self, index_col: Optional[Union[str, List[str]]] = None) -> SparkDataFrame:
        return self.spark.frame(index_col)

    def to_pandas(self) -> pd.DataFrame:
        return self._internal.to_pandas_frame.copy()

    def toPandas(self) -> pd.DataFrame:
        warnings.warn(
            "DataFrame.toPandas is deprecated as of DataFrame.to_pandas. Please use the API instead.",
            FutureWarning,
        )
        return self.to_pandas()

    def assign(self, **kwargs: Any) -> "DataFrame":
        return self._assign(kwargs)

    def _assign(self, kwargs: Dict[Any, Any]) -> "DataFrame":
        from databricks.koalas.indexes import MultiIndex
        from databricks.koalas.series import IndexOpsMixin
        for k, v in kwargs.items():
            is_invalid_assignee = (
                not (isinstance(v, (IndexOpsMixin, Column)) or callable(v) or is_scalar(v)
                )
            ) or isinstance(v, MultiIndex)
            if is_invalid_assignee:
                raise TypeError("Column assignment doesn't support type " + str(type(v).__name__))
            if callable(v):
                kwargs[k] = v(self)
        pairs: Dict[Tuple[Any, ...], Tuple[Any, Optional[Any]]] = {
            (k if isinstance(k, tuple) else (k,)): (
                (v.spark.column, v.dtype) if isinstance(v, IndexOpsMixin) and not isinstance(v, MultiIndex)
                else (v, None) if isinstance(v, Column)
                else (F.lit(v), None)
            )
            for k, v in kwargs.items()
        }
        scols = []
        data_dtypes = []
        for label in self._internal.column_labels:
            for i in range(len(label)):
                if label[: len(label) - i] in pairs:
                    scl, dtype = pairs[label[: len(label) - i]]
                    scl = scl.alias(str(label))
                    break
            else:
                scl = self._internal.spark_column_for(label)
                dtype = self._internal.dtype_for(label)
            scols.append(scl)
            data_dtypes.append(dtype)
        column_labels = self._internal.column_labels.copy()
        for label, (scl, dtype) in pairs.items():
            if label not in set(i[: len(label)] for i in self._internal.column_labels):
                scols.append(scl.alias(str(label)))
                column_labels.append(label)
                data_dtypes.append(dtype)
        level = self._internal.column_labels_level
        column_labels = [tuple(list(label) + ([""] * (level - len(label)))) for label in column_labels]
        internal = self._internal.with_new_columns(
            scols, column_labels=column_labels, data_dtypes=data_dtypes
        )
        return DataFrame(internal)

    @staticmethod
    def from_records(
        data: Union[np.ndarray, List[tuple], dict, pd.DataFrame],
        index: Optional[Union[str, List[str], np.ndarray]] = None,
        exclude: Optional[List[Any]] = None,
        columns: Optional[List[Any]] = None,
        coerce_float: bool = False,
        nrows: Optional[int] = None,
    ) -> "DataFrame":
        return DataFrame(pd.DataFrame.from_records(data, index=index, exclude=exclude, columns=columns, coerce_float=coerce_float, nrows=nrows))

    def to_records(self, index: bool = True, column_dtypes: Any = None, index_dtypes: Any = None) -> np.recarray:
        args = locals()
        kdf = self
        return kdf._to_internal_pandas().to_records(index=index, column_dtypes=column_dtypes, index_dtypes=index_dtypes)

    def copy(self, deep: Optional[bool] = None) -> "DataFrame":
        return DataFrame(self._internal)

    def dropna(
        self,
        axis: Union[int, str] = 0,
        how: str = "any",
        thresh: Optional[int] = None,
        subset: Optional[Any] = None,
        inplace: bool = False,
    ) -> Optional["DataFrame"]:
        axis = axis
        inplace = bool(inplace)
        if thresh is None:
            if how is None:
                raise TypeError("must specify how or thresh")
            elif how not in ("any", "all"):
                raise ValueError("invalid how option: {h}".format(h=how))
        if subset is not None:
            if isinstance(subset, str):
                labels = [(subset,)]
            elif isinstance(subset, tuple):
                labels = [subset]
            else:
                labels = [sub if isinstance(sub, tuple) else (sub,) for sub in subset]
        else:
            labels = None
        if axis == 0:
            if labels is not None:
                invalids = [label for label in labels if label not in self._internal.column_labels]
                if len(invalids) > 0:
                    raise KeyError(invalids)
            else:
                labels = self._internal.column_labels
            cnt = reduce(lambda x, y: x + y,
                         [F.lit(1).when(self._kser_for(label).notna().spark.column, 0) for label in labels],
                         F.lit(0))
            if thresh is not None:
                pred = cnt >= F.lit(int(thresh))
            elif how == "any":
                pred = cnt == F.lit(len(labels))
            elif how == "all":
                pred = cnt > F.lit(0)
            internal = self._internal.with_filter(pred)
            if inplace:
                self._update_internal_frame(internal)
                return None
            else:
                return DataFrame(internal)
        else:
            assert axis == 1
            internal = self._internal.resolved_copy
            if labels is not None:
                if any(len(lbl) != internal.index_level for lbl in labels):
                    raise ValueError("The length of each subset must be the same as the index size.")
                cond = reduce(lambda x, y: x | y,
                              [reduce(lambda x, y: x & y, [F.col(c) == F.lit(val) for c, val in zip(internal.index_spark_column_names, lbl)]) for lbl in labels])
                internal = internal.with_filter(cond)
            null_counts = []
            for label in internal.column_labels:
                scol = internal.spark_column_for(label)
                if isinstance(internal.spark_type_for(label), (FloatType, DoubleType)):
                    cond = scol.isNull() | F.isnan(scol)
                else:
                    cond = scol.isNull()
                null_counts.append(F.sum(F.when(~cond, 1).otherwise(0)).alias(str(label)))
            counts = internal.spark_frame.select(null_counts + [F.count("*")]).head()
            if thresh is not None:
                column_labels = [label for label, cnt in zip(internal.column_labels, counts) if (cnt or 0) >= int(thresh)]
            elif how == "any":
                column_labels = [label for label, cnt in zip(internal.column_labels, counts) if (cnt or 0) == counts[-1]]
            elif how == "all":
                column_labels = [label for label, cnt in zip(internal.column_labels, counts) if (cnt or 0) > 0]
            kdf = self[column_labels]
            if inplace:
                self._update_internal_frame(kdf._internal)
                return None
            else:
                return kdf

    def fillna(self, value: Optional[Any] = None, method: Optional[str] = None, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Optional["DataFrame"]:
        axis = axis
        if axis != 0:
            raise NotImplementedError("fillna currently only works for axis=0 or axis='index'")
        if value is not None:
            if not isinstance(value, (float, int, str, bool, dict, pd.Series)):
                raise TypeError("Unsupported type %s" % type(value).__name__)
            if limit is not None:
                raise ValueError("limit parameter for value is not support now")
            if isinstance(value, pd.Series):
                value = value.to_dict()
            if isinstance(value, dict):
                for v in value.values():
                    if not isinstance(v, (float, int, str, bool)):
                        raise TypeError("Unsupported type %s" % type(v).__name__)
                value = {k if isinstance(k, tuple) else (k,): v for k, v in value.items()}
            op = lambda kser: kser._fillna(value=value, method=method, axis=axis, limit=limit)
        elif method is not None:
            op = lambda kser: kser._fillna(value=value, method=method, axis=axis, limit=limit)
        else:
            raise ValueError("Must specify a fillna 'value' or 'method' parameter.")
        kdf = self._apply_series_op(op)
        inplace = bool(inplace)
        if inplace:
            self._update_internal_frame(kdf._internal, requires_same_anchor=False)
            return None
        else:
            return kdf

    def replace(self, to_replace: Any = None, value: Any = None, inplace: bool = False, limit: Optional[int] = None, regex: bool = False, method: str = "pad") -> Optional["DataFrame"]:
        if method != "pad":
            raise NotImplementedError("replace currently works only for method='pad")
        if limit is not None:
            raise NotImplementedError("replace currently works only when limit=None")
        if regex is not False:
            raise NotImplementedError("replace currently doesn't supports regex")
        inplace = bool(inplace)
        if value is not None and not isinstance(value, (int, float, str, list, tuple, dict)):
            raise TypeError("Unsupported type {}".format(type(value).__name__))
        if to_replace is not None and not isinstance(to_replace, (int, float, str, list, tuple, dict)):
            raise TypeError("Unsupported type {}".format(type(to_replace).__name__))
        if isinstance(value, (list, tuple)) and isinstance(to_replace, (list, tuple)):
            if len(value) != len(to_replace):
                raise ValueError("Length of to_replace and value must be same")
        if isinstance(to_replace, dict) and (value is not None or all(isinstance(i, dict) for i in to_replace.values())):
            def op(kser):
                if kser.name in to_replace:
                    return kser.replace(to_replace=to_replace[kser.name], value=value, regex=regex)
                else:
                    return kser
        else:
            op = lambda kser: kser.replace(to_replace=to_replace, value=value, regex=regex)
        kdf = self._apply_series_op(op)
        if inplace:
            self._update_internal_frame(kdf._internal)
            return None
        else:
            return kdf

    def melt(self, id_vars: Optional[Union[str, List[Any]]] = None, value_vars: Optional[Union[str, List[Any]]] = None, var_name: Optional[str] = None, value_name: str = "value") -> "DataFrame":
        column_labels = self._internal.column_labels
        if id_vars is None:
            id_vars = []
        else:
            if isinstance(id_vars, tuple):
                id_vars = [id_vars]
            elif isinstance(id_vars, (str)):
                id_vars = [id_vars]
            else:
                id_vars = list(id_vars)
            non_existence_col = [idv for idv in id_vars if (idv if isinstance(idv, tuple) else (idv,)) not in column_labels]
            if len(non_existence_col) != 0:
                raise KeyError("The following 'id_vars' are not present in the DataFrame: {}".format(non_existence_col))
        if value_vars is None:
            value_vars = []
        else:
            if isinstance(value_vars, tuple):
                value_vars = [value_vars]
            elif isinstance(value_vars, (str)):
                value_vars = [value_vars]
            else:
                value_vars = list(value_vars)
            non_existence_col = [valv for valv in value_vars if (valv if isinstance(valv, tuple) else (valv,)) not in column_labels]
            if len(non_existence_col) != 0:
                raise KeyError("The following 'value_vars' are not present in the DataFrame: {}".format(non_existence_col))
        if len(value_vars) == 0:
            value_vars = column_labels
        new_df = self.__getitem__([col for col in column_labels if col not in id_vars])
        from pyspark.sql.functions import explode, array, struct, to_json, col
        pairs = F.explode(
            F.array(
                *[
                    F.struct(
                        *([F.lit(c).alias("var") for c in (col if isinstance(col, tuple) else (col,))] +
                          [self._internal.spark_column_for(col).alias(value_name)])
                    )
                    for col in value_vars
                ]
            )
        )
        exploded_df = self._internal.spark_frame.withColumn("pairs", pairs).select(
            [self._internal.spark_column_for(col).alias(str(col)) for col in id_vars]
            + [F.col("pairs.var").alias(var_name if var_name is not None else "variable"),
               F.col("pairs." + value_name).alias(value_name)]
        )
        internal = InternalFrame(spark_frame=exploded_df, index_spark_columns=None,
                                 column_labels=[tuple(id_vars) if id_vars else None, (var_name if var_name is not None else "variable",), (value_name,)])
        return DataFrame(internal)

    def stack(self) -> Union["DataFrame", "Series"]:
        from databricks.koalas.series import first_series
        if len(self._internal.column_labels) == 0:
            return DataFrame(self._internal.copy(column_label_names=self._internal.column_label_names[:-1]).with_filter(F.lit(False)))
        column_labels = defaultdict(dict)
        index_values = set()
        should_returns_series = False
        for label in self._internal.column_labels:
            new_label = label[:-1]
            if len(new_label) == 0:
                new_label = None
                should_returns_series = True
            value = label[-1]
            scol = self._internal.spark_column_for(label)
            column_labels[new_label][value] = scol
            index_values.add(value)
        from collections import OrderedDict
        column_labels = OrderedDict(sorted(column_labels.items(), key=lambda x: x[0] if x[0] is not None else ""))
        index_name = self._internal.column_label_names[-1]
        column_label_names = self._internal.column_label_names[:-1]
        if len(column_label_names) == 0:
            column_label_names = [None]
        new_index_columns = [ "level_{}".format(i) for i in range(self._internal.column_labels_level)]
        data_columns = [str(label) for label in column_labels]
        structs = [
            F.struct([F.lit(value).alias(new_index_columns[0])] +
                     [column_labels[label][value] for label in column_labels]).alias(value)
            for value in index_values
        ]
        pairs = F.explode(F.array(*structs))
        sdf = self._internal.spark_frame.withColumn("pairs", pairs)
        sdf = sdf.select(
            self._internal.index_spark_columns +
            [sdf["pairs"][new_index_columns[0]].alias(new_index_columns[0])] +
            [sdf["pairs"][name].alias(name) for name in data_columns]
        )
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[F.col(new_index_columns[0])],
            index_names=[index_name],
            index_dtypes=[None],
            column_labels=list(column_labels),
            data_spark_columns=[F.col(name) for name in data_columns],
            column_label_names=column_label_names,
        )
        kdf = DataFrame(internal)
        if should_returns_series:
            return first_series(kdf)
        else:
            return kdf

    def unstack(self) -> Union["DataFrame", "Series"]:
        from databricks.koalas.series import first_series
        if self._internal.index_level > 1:
            from databricks.koalas.config import option_context
            with option_context("compute.default_index_type", "distributed"):
                df = self.reset_index()
            index = df._internal.column_labels[: self._internal.index_level - 1]
            columns = df.columns[self._internal.index_level - 1]
            df = df.pivot_table(index=index, columns=columns, values=self._internal.column_labels, aggfunc="first")
            internal = df._internal.copy(index_names=self._internal.index_names[:-1],
                                         index_dtypes=self._internal.index_dtypes[:-1],
                                         column_label_names=df._internal.column_label_names[:-1] + [columns])
            return DataFrame(internal)
        else:
            value_column = "value"
            cols = []
            for label in self._internal.column_labels:
                cols.append(
                    F.struct(
                        [F.lit(item).alias("var") for item in label] +
                        [self._internal.spark_column_for(label).alias(value_column)]
                    )
                )
            sdf = self._internal.spark_frame.select(F.array(*cols).alias("arrays")).select(F.explode(F.col("arrays")))
            sdf = sdf.selectExpr("col.*")
            internal = InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[F.col("var")],
                index_names=self._internal.column_label_names,
                column_labels=[None],
                data_spark_columns=[F.col(value_column)],
            )
            return first_series(DataFrame(internal))

    def all(self, axis: Union[int, str] = 0) -> "Series":
        from databricks.koalas.series import first_series
        axis = axis
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        applied = []
        for label in self._internal.column_labels:
            scol = self._internal.spark_column_for(label)
            all_col = F.min(F.coalesce(scol.cast("boolean"), F.lit(True)))
            applied.append(F.when(all_col.isNull(), True).otherwise(all_col))
        value_column = "value"
        cols = []
        for label, applied_col in zip(self._internal.column_labels, applied):
            cols.append(F.struct([F.lit(str(n)).alias("idx") for n in label] + [applied_col.alias(value_column)]))
        sdf = self._internal.spark_frame.select(F.array(*cols).alias("arrays")).select(F.explode(F.col("arrays")))
        sdf = sdf.selectExpr("col.*")
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[F.col("idx")],
            index_names=self._internal.column_label_names,
            column_labels=[None],
            data_spark_columns=[F.col(value_column)],
        )
        return first_series(DataFrame(internal))

    def any(self, axis: Union[int, str] = 0) -> "Series":
        from databricks.koalas.series import first_series
        axis = axis
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        applied = []
        for label in self._internal.column_labels:
            scol = self._internal.spark_column_for(label)
            all_col = F.max(F.coalesce(scol.cast("boolean"), F.lit(False)))
            applied.append(F.when(all_col.isNull(), False).otherwise(all_col))
        value_column = "value"
        cols = []
        for label, applied_col in zip(self._internal.column_labels, applied):
            cols.append(F.struct([F.lit(str(n)).alias("idx") for n in label] + [applied_col.alias(value_column)]))
        sdf = self._internal.spark_frame.select(F.array(*cols).alias("arrays")).select(F.explode(F.col("arrays")))
        sdf = sdf.selectExpr("col.*")
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[F.col("idx")],
            index_names=self._internal.column_label_names,
            column_labels=[None],
            data_spark_columns=[F.col(value_column)],
        )
        return first_series(DataFrame(internal))

    def rank(self, method: str = "average", ascending: bool = True) -> "DataFrame":
        return self._apply_series_op(lambda kser: kser._rank(method=method, ascending=ascending), should_resolve=True)

    def filter(self, items: Optional[Any] = None, like: Optional[str] = None, regex: Optional[str] = None, axis: Optional[Union[int, str]] = None) -> "DataFrame":
        axis = axis if axis is not None else 1
        if sum(x is not None for x in (items, like, regex)) > 1:
            raise TypeError("Keyword arguments `items`, `like`, or `regex` are mutually exclusive")
        if items is not None:
            axis = axis
            if axis == 0:
                return self.loc[:, items]
            else:
                return self.loc[:, items]
        elif like is not None:
            if axis == 0:
                return DataFrame(self._internal.with_filter(F.lit(True)))  # placeholder
            else:
                column_labels = self._internal.column_labels
                output_labels = [label for label in column_labels if any(like in str(i) for i in label)]
                return self[output_labels]
        elif regex is not None:
            if axis == 0:
                return DataFrame(self._internal.with_filter(F.lit(True)))  # placeholder
            else:
                column_labels = self._internal.column_labels
                matcher = re.compile(regex)
                output_labels = [label for label in column_labels if any(matcher.search(str(i)) is not None for i in label)]
                return self[output_labels]
        else:
            raise TypeError("Must pass either `items`, `like`, or `regex`")

    def rename(self, mapper: Optional[Any] = None, index: Optional[Any] = None, columns: Optional[Any] = None, axis: Union[int, str] = "index", inplace: bool = False, level: Optional[Any] = None, errors: str = "ignore") -> Optional["DataFrame"]:
        def gen_mapper_fn(mapper: Any) -> Tuple[Callable[[Any], Any], Any]:
            if isinstance(mapper, dict):
                if len(mapper) == 0:
                    if errors == "raise":
                        raise KeyError("Index include label which is not in the `mapper`.")
                    else:
                        return (lambda x: x, None)
                type_set = set(map(lambda x: type(x), mapper.values()))
                if len(type_set) > 1:
                    raise ValueError("Mapper dict should have the same value type.")
                spark_return_type = as_nullable_spark_type(list(type_set)[0])
                def mapper_fn(x: Any) -> Any:
                    if x in mapper:
                        return mapper[x]
                    else:
                        if errors == "raise":
                            raise KeyError("Index include value which is not in the `mapper`")
                        return x
            elif callable(mapper):
                spark_return_type = infer_return_type(mapper)
                def mapper_fn(x: Any) -> Any:
                    return mapper(x)
            else:
                raise ValueError("`mapper` or `index` or `columns` should be either dict-like or function type.")
            return mapper_fn, spark_return_type
        index_mapper_fn = None
        index_mapper_ret_stype = None
        columns_mapper_fn = None
        inplace = bool(inplace)
        if mapper is not None:
            axis = axis
            if axis == 0:
                index_mapper_fn, index_mapper_ret_stype = gen_mapper_fn(mapper)
            elif axis == 1:
                columns_mapper_fn, _ = gen_mapper_fn(mapper)
            else:
                raise ValueError("argument axis should be either 'index' or 'columns'")
        else:
            if index is not None:
                index_mapper_fn, index_mapper_ret_stype = gen_mapper_fn(index)
            if columns is not None:
                columns_mapper_fn, _ = gen_mapper_fn(columns)
            if index is None and columns is None:
                raise ValueError("Either `index` or `columns` should be provided.")
        kdf = self.copy()
        if index_mapper_fn:
            num_indices = len(self._internal.index_spark_column_names)
            def gen_new_index_column(level: int) -> Any:
                index_col_name = self._internal.index_spark_column_names[level]
                from pyspark.sql.functions import pandas_udf
                index_mapper_udf = pandas_udf(lambda s: s.map(index_mapper_fn), returnType=index_mapper_ret_stype)
                return index_mapper_udf(F.col(index_col_name))
            sdf = kdf._internal.resolved_copy.spark_frame
            index_dtypes = self._internal.index_dtypes.copy()
            if level is None:
                for i in range(num_indices):
                    sdf = sdf.withColumn(self._internal.index_spark_column_names[i], gen_new_index_column(i))
                    index_dtypes[i] = None
            else:
                sdf = sdf.withColumn(self._internal.index_spark_column_names[level], gen_new_index_column(level))
                index_dtypes[level] = None
            kdf = DataFrame(kdf._internal.with_new_sdf(sdf, index_dtypes=index_dtypes))
        if columns_mapper_fn:
            def gen_new_column_labels_entry(column_labels_entry: Tuple[Any, ...]) -> Tuple[Any, ...]:
                if isinstance(column_labels_entry, tuple):
                    if level is None:
                        return tuple(map(columns_mapper_fn, column_labels_entry))
                    else:
                        entry_list = list(column_labels_entry)
                        entry_list[level] = columns_mapper_fn(entry_list[level])
                        return tuple(entry_list)
                else:
                    return columns_mapper_fn(column_labels_entry)
            new_column_labels = list(map(gen_new_column_labels_entry, kdf._internal.column_labels))
            new_data_scols = [kdf._kser_for(old_label).rename(new_label) for old_label, new_label in zip(kdf._internal.column_labels, new_column_labels)]
            kdf = DataFrame(kdf._internal.with_new_columns(new_data_scols))
        if inplace:
            self._update_internal_frame(kdf._internal)
            return None
        else:
            return kdf

    def rename_axis(self, mapper: Optional[Any] = None, index: Optional[Any] = None, columns: Optional[Any] = None, axis: Optional[Union[int, str]] = 0, inplace: bool = False) -> Optional["DataFrame"]:
        def gen_names(v: Any, curnames: List[Any]) -> List[Tuple[Any, ...]]:
            if np.isscalar(v):
                newnames = [v]
            elif is_list_like(v):
                newnames = list(v)
            elif is_dict_like(v):
                newnames = [v[name] if name in v else name for name in curnames]
            elif callable(v):
                newnames = [v(name) for name in curnames]
            else:
                raise ValueError("`mapper` or `index` or `columns` should be either dict-like or function type.")
            if len(newnames) != len(curnames):
                raise ValueError("Length of new names must be {}, got {}".format(len(curnames), len(newnames)))
            return [n if isinstance(n, tuple) else (n,) for n in newnames]
        column_label_names = gen_names(columns, list(self.columns.names)) if columns is not None else self._internal.column_label_names
        index_names = gen_names(index, list(self.index.names)) if index is not None else self._internal.index_names
        internal = self._internal.copy(index_names=index_names, column_label_names=column_label_names)
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def keys(self) -> pd.Index:
        return self.columns

    def pct_change(self, periods: int = 1) -> "DataFrame":
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        def op(kser):
            prev_row = F.lag(kser.spark.column, periods).over(window)
            return ((kser.spark.column - prev_row) / prev_row).alias(kser._internal.data_spark_column_names[0])
        return self._apply_series_op(op, should_resolve=True)

    def idxmax(self, axis: int = 0) -> "Series":
        max_cols = list(map(lambda scol: F.max(scol), self._internal.data_spark_columns))
        sdf_max = self._internal.spark_frame.select(*max_cols).head()
        conds = (scol == max_val for scol, max_val in zip(self._internal.data_spark_columns, sdf_max))
        cond = reduce(lambda x, y: x | y, conds)
        kdf = DataFrame(self._internal.with_filter(cond))
        from databricks.koalas.series import first_series
        return first_series(pd.DataFrame(kdf._to_internal_pandas().idxmax()))

    def idxmin(self, axis: int = 0) -> "Series":
        min_cols = list(map(lambda scol: F.min(scol), self._internal.data_spark_columns))
        sdf_min = self._internal.spark_frame.select(*min_cols).head()
        conds = (scol == min_val for scol, min_val in zip(self._internal.data_spark_columns, sdf_min))
        cond = reduce(lambda x, y: x | y, conds)
        kdf = DataFrame(self._internal.with_filter(cond))
        from databricks.koalas.series import first_series
        return first_series(pd.DataFrame(kdf._to_internal_pandas().idxmin()))

    def info(self, verbose: Optional[bool] = None, buf: Optional[Any] = None, max_cols: Optional[int] = None, null_counts: Optional[bool] = None) -> None:
        from contextlib import contextmanager
        with pd.option_context("display.max_info_columns", sys.maxsize, "display.max_info_rows", sys.maxsize):
            try:
                object.__setattr__(self, "_data", self)
                count_func = self.count
                self.count = lambda: count_func().to_pandas()
                pd.DataFrame.info(self, verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=False, null_counts=null_counts)
            finally:
                del self._data
                self.count = count_func

    def quantile(self, q: Union[float, Iterable[float]] = 0.5, axis: Union[int, str] = 0, numeric_only: bool = True, accuracy: int = 10000) -> Union["DataFrame", "Series"]:
        axis = axis
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if not isinstance(accuracy, int):
            raise ValueError("accuracy must be an integer; however, got [%s]" % type(accuracy).__name__)
        if isinstance(q, Iterable) and not isinstance(q, float):
            q = list(q)
        else:
            q = [q]
        for v in q:
            if not isinstance(v, float):
                raise ValueError("q must be a float or an array of floats; however, [%s] found." % type(v))
            if v < 0.0 or v > 1.0:
                raise ValueError("percentiles should all be in the interval [0, 1].")
        def quantile(spark_column: Column, spark_type: Any) -> Any:
            if isinstance(spark_type, (BooleanType, NumericType)):
                return F.percentile_approx(spark_column.cast(DoubleType()), F.lit(q), F.lit(accuracy))
            else:
                raise TypeError("Could not convert {} ({}) to numeric".format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
        if len(q) > 1:
            percentile_cols = []
            percentile_col_names = []
            column_labels = []
            for label, colname in zip(self._internal.column_labels, self._internal.data_spark_column_names):
                spark_type = self._internal.spark_type_for(label)
                is_numeric_or_boolean = isinstance(spark_type, (NumericType, BooleanType))
                if not numeric_only or is_numeric_or_boolean:
                    percentile_col = quantile(self._internal.spark_column_for(label), spark_type)
                    percentile_cols.append(percentile_col.alias(colname))
                    percentile_col_names.append(colname)
                    column_labels.append(label)
            if len(percentile_cols) == 0:
                from databricks.koalas.series import first_series
                return first_series(DataFrame(index=q))
            sdf = self._internal.spark_frame.select(*percentile_cols)
            cols_dict: OrderedDict[str, List[Any]] = OrderedDict()
            for col in percentile_col_names:
                cols_dict[col] = []
                for i in range(len(q)):
                    cols_dict[col].append(F.col(col).getItem(i).alias(col))
            cols = []
            for i in range(len(q)):
                cols.append(F.struct(F.lit(q[i]).alias("quantile"), *[cols_dict[col][i] for col in percentile_col_names]))
            sdf = sdf.select(F.array(*cols).alias("arrays"))
            sdf = sdf.select(F.explode(F.col("arrays")).alias("exploded")).selectExpr("exploded.*")
            internal = InternalFrame(
                spark_frame=sdf,
                index_spark_columns=[F.col("quantile")],
                column_labels=column_labels,
                data_spark_columns=[F.col(col) for col in percentile_col_names],
            )
            return DataFrame(internal)
        else:
            return self._reduce_for_stat_function(quantile, name="quantile", numeric_only=numeric_only).rename(q[0])

    def query(self, expr: str, inplace: bool = False) -> Optional[Union["DataFrame", "Series"]]:
        if not isinstance(expr, str):
            raise ValueError("expr must be a string to be evaluated, {} given".format(type(expr).__name__))
        inplace = bool(inplace)
        data_columns = [str(label[0]) for label in self._internal.column_labels]
        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns +
            [self._internal.spark_column_for(label).alias(str(label)) for label in self._internal.column_labels]
        ).filter(expr)
        internal = self._internal.with_new_sdf(sdf, data_columns=data_columns)
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def explain(self, extended: Optional[bool] = None, mode: Optional[str] = None) -> None:
        warnings.warn(
            "DataFrame.explain is deprecated as of DataFrame.spark.explain. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.explain(extended, mode)

    def take(self, indices: Any, axis: Union[int, str] = 0, **kwargs: Any) -> "DataFrame":
        axis = axis
        if not is_list_like(indices) or isinstance(indices, (dict, set)):
            raise ValueError("`indices` must be a list-like except dict or set")
        if axis == 0:
            return self.iloc[indices, :]
        else:
            return self.iloc[:, indices]

    def __getitem__(self, key: Any) -> Any:
        from databricks.koalas.series import Series
        if key is None:
            raise KeyError("none key")
        elif isinstance(key, Series):
            return self.loc[key.astype(bool)]
        elif isinstance(key, slice):
            if any(isinstance(n, int) or n is None for n in [key.start, key.stop]):
                return self.iloc[key]
            return self.loc[key]
        elif isinstance(key, (str, tuple)):
            return self.loc[:, key]
        elif is_list_like(key):
            return self.loc[:, list(key)]
        raise NotImplementedError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        from databricks.koalas.series import Series
        if isinstance(value, (DataFrame, Series)) and getattr(value, "_internal", None) and (value._internal != self._internal):
            key = DataFrame._index_normalized_label(self._internal.column_labels_level, key)
            value = DataFrame._index_normalized_frame(self._internal.column_labels_level, value)
            def assign_columns(kdf, this_column_labels, that_column_labels):
                from itertools import zip_longest
                for this_label, that_label in zip_longest(this_column_labels, that_column_labels):
                    yield (kdf._kser_for(this_label), ("that",) + this_label)
                    if this_label is not None and this_label != that_label:
                        yield (kdf._kser_for(this_label), this_label)
            from databricks.koalas.namespace import align_diff_frames
            kdf = align_diff_frames(assign_columns, self, value, fillna=False, how="left")
        elif isinstance(value, list):
            if len(self) != len(value):
                raise ValueError("Length of values does not match length of index")
            from databricks.koalas.config import option_context
            with option_context("compute.default_index_type", "distributed-sequence", "compute.ops_on_diff_frames", True):
                kdf = self.reset_index()
                kdf[key] = DataFrame(value)
                kdf = kdf.set_index(kdf.columns[:self._internal.index_level])
                kdf.index.names = self.index.names
        elif isinstance(key, list):
            assert isinstance(value, DataFrame)
            kdf = self._assign({k: value[c] for k, c in zip(key, value.columns)})
        else:
            kdf = self._assign({key: value})
        self._update_internal_frame(kdf._internal)

    def __len__(self) -> int:
        return self._internal.resolved_copy.spark_frame.count()

    def __iter__(self) -> Iterator[Any]:
        return iter(self.columns)

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if all(isinstance(inp, DataFrame) for inp in inputs) and any(getattr(inp, "_internal", None) and inp._internal != inputs[0]._internal for inp in inputs):
            assert len(inputs) == 2
            this = inputs[0]
            that = inputs[1]
            if this._internal.column_labels_level != that._internal.column_labels_level:
                raise ValueError("cannot join with no overlapping index names")
            def apply_op(kdf, this_column_labels, that_column_labels):
                for this_label, that_label in zip(this_column_labels, that_column_labels):
                    yield (getattr(kdf._kser_for(this_label), ufunc.__name__)(kdf._kser_for(that_label), **kwargs).rename(this_label), this_label)
            from databricks.koalas.namespace import align_diff_frames
            return align_diff_frames(apply_op, this, that, fillna=True, how="full")
        else:
            applied = []
            this = inputs[0]
            for label in this._internal.column_labels:
                arguments = []
                for inp in inputs:
                    arguments.append(inp[label] if isinstance(inp, DataFrame) else inp)
                applied.append(ufunc(*arguments, **kwargs).rename(label))
            internal = this._internal.with_new_columns(applied)
            return DataFrame(internal)

    if sys.version_info >= (3, 7):
        @classmethod
        def __class_getitem__(cls, params: Any) -> Any:
            return _create_tuple_for_frame_type(params)
    elif (3, 5) <= sys.version_info < (3, 7):
        is_dataframe = None

class CachedDataFrame(DataFrame):
    def __init__(self, internal: InternalFrame, storage_level: Optional[StorageLevel] = None) -> None:
        if storage_level is None:
            object.__setattr__(self, "_cached", internal.spark_frame.cache())
        elif isinstance(storage_level, StorageLevel):
            object.__setattr__(self, "_cached", internal.spark_frame.persist(storage_level))
        else:
            raise TypeError("Only a valid pyspark.StorageLevel type is acceptable for the `storage_level`")
        super().__init__(internal)

    def __enter__(self) -> "CachedDataFrame":
        return self

    def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
        self.spark.unpersist()

    spark = CachedAccessor("spark", object)  # Placeholder for CachedSparkFrameMethods

    @property
    def storage_level(self) -> StorageLevel:
        warnings.warn(
            "DataFrame.storage_level is deprecated as of DataFrame.spark.storage_level. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.storage_level

    def unpersist(self) -> None:
        warnings.warn(
            "DataFrame.unpersist is deprecated as of DataFrame.spark.unpersist. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.unpersist()

    def hint(self, name: str, *parameters: Any) -> "DataFrame":
        warnings.warn(
            "DataFrame.hint is deprecated as of DataFrame.spark.hint. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.hint(name, *parameters)

    def to_table(
        self,
        name: str,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        return self.spark.to_table(name, format, mode, partition_cols, index_col, **options)

# End of code.
