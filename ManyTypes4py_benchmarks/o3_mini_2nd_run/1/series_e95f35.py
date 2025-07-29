from collections.abc import Iterable
from functools import reduce
from typing import Any, Generic, Optional, Tuple, TypeVar, Union, Iterator

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import DataFrame as SparkDataFrame, Column, Window, functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, IntegerType, LongType, NumericType, ArrayType, StructType, IntegralType

from databricks.koalas.config import get_option, SPARK_CONF_ARROW_ENABLED
from databricks.koalas.frame import DataFrame
from databricks.koalas.internal import InternalFrame, NATURAL_ORDER_COLUMN_NAME, SPARK_DEFAULT_INDEX_NAME, SPARK_DEFAULT_SERIES_NAME
from databricks.koalas.missing.series import MissingPandasLikeSeries
from databricks.koalas.plot import KoalasPlotAccessor
from databricks.koalas.strings import StringMethods
from databricks.koalas.timedeltas import DatetimeMethods
from databricks.koalas.categorical import CategoricalAccessor
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.spark import functions as SF

T = TypeVar("T")


class Series(DataFrame, IndexOpsMixin, Generic[T]):
    dt: DatetimeMethods
    str: StringMethods
    cat: CategoricalAccessor
    plot: KoalasPlotAccessor

    def add(self, other: Any) -> "Series":
        return self + other

    def radd(self, other: Any) -> "Series":
        return other + self

    def div(self, other: Any) -> "Series":
        return self / other

    def rdiv(self, other: Any) -> "Series":
        return other / self

    def truediv(self, other: Any) -> "Series":
        return self / other

    def rtruediv(self, other: Any) -> "Series":
        return other / self

    def mul(self, other: Any) -> "Series":
        return self * other

    def rmul(self, other: Any) -> "Series":
        return other * self

    def sub(self, other: Any) -> "Series":
        return self - other

    def rsub(self, other: Any) -> "Series":
        return other - self

    def mod(self, other: Any) -> "Series":
        return self % other

    def rmod(self, other: Any) -> "Series":
        return other % self

    def pow(self, other: Any) -> "Series":
        return self ** other

    def rpow(self, other: Any) -> "Series":
        return other ** self

    def eq(self, other: Any) -> "Series":
        return self == other

    def equals(self, other: Any) -> "Series":
        return self == other

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

    def map(self, arg: Any) -> "Series":
        if isinstance(arg, dict):
            is_start: bool = True
            current: Any = F.when(F.lit(False), F.lit(None).cast(self.spark.data_type))
            for to_replace, value in arg.items():
                if is_start:
                    current = F.when(self.spark.column == F.lit(to_replace), value)
                    is_start = False
                else:
                    current = current.when(self.spark.column == F.lit(to_replace), value)
            if hasattr(arg, '__missing__'):
                tmp_val: Any = arg[np._NoValue]
                del arg[np._NoValue]
                current = current.otherwise(F.lit(tmp_val))
            else:
                current = current.otherwise(F.lit(None).cast(self.spark.data_type))
            return self._with_new_scol(current)
        else:
            return self.apply(arg)

    def alias(self, name: Any) -> "Series":
        import warnings
        warnings.warn('Series.alias is deprecated as of Series.rename. Please use the API instead.', FutureWarning)
        return self.rename(name)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self),)

    @property
    def name(self) -> Any:
        name_val = self._column_label
        if name_val is not None and len(name_val) == 1:
            return name_val[0]
        else:
            return name_val

    @name.setter
    def name(self, name: Any) -> None:
        self.rename(name, inplace=True)

    def rename(self, index: Optional[Any] = None, **kwargs: Any) -> Union["Series", None]:
        if index is None:
            pass
        elif not isinstance(index, (str, tuple)) and not hasattr(index, '__hash__'):
            raise TypeError('Series.name must be a hashable type')
        elif not isinstance(index, tuple):
            index = (index,)
        scol = self.spark.column.alias(str(index))
        internal = self._internal.copy(column_labels=[index], data_spark_columns=[scol], column_label_names=None)
        kdf: DataFrame = DataFrame(internal)
        if kwargs.get('inplace', False):
            self._col_label = index
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def rename_axis(self, mapper: Optional[Any] = None, index: Optional[Any] = None, inplace: bool = False) -> Union["Series", None]:
        kdf: DataFrame = self.to_frame().rename_axis(mapper=mapper, index=index, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    @property
    def index(self) -> Any:
        return self._kdf.index

    @property
    def is_unique(self) -> bool:
        scol = self.spark.column
        result: Any = self._internal.spark_frame.select((F.count(scol) == F.countDistinct(scol)) & (F.count(F.when(scol.isNull(), 1).otherwise(None)) <= 1)).collect()[0][0]
        return bool(result)

    def reset_index(self, level: Optional[Any] = None, drop: bool = False, name: Optional[Any] = None, inplace: bool = False) -> Union["Series", DataFrame, None]:
        from databricks.koalas.config import validate_bool_kwarg
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if inplace and (not drop):
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')
        if drop:
            kdf: DataFrame = self._kdf[[self.name]]
        else:
            kser: "Series" = self
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

    def to_frame(self, name: Optional[Any] = None) -> DataFrame:
        if name is not None:
            renamed: "Series" = self.rename(name)
        elif self._column_label is None:
            from databricks.koalas.internal import DEFAULT_SERIES_NAME
            renamed = self.rename(DEFAULT_SERIES_NAME)
        else:
            renamed = self
        return DataFrame(renamed._internal)
    to_dataframe = to_frame

    def to_string(self, buf: Optional[Any] = None, na_rep: str = 'NaN', float_format: Optional[Any] = None, header: bool = True,
                  index: bool = True, length: bool = False, dtype: bool = False, name: bool = False, max_rows: Optional[int] = None) -> str:
        args = locals()
        if max_rows is not None:
            kseries: "Series" = self.head(max_rows)
        else:
            kseries = self
        from databricks.koalas.utils import validate_arguments_and_invoke_function
        return validate_arguments_and_invoke_function(kseries._to_internal_pandas(), self.to_string, pd.Series.to_string, args)

    def to_clipboard(self, excel: bool = True, sep: Optional[str] = None, **kwargs: Any) -> Any:
        args = locals()
        kseries: "Series" = self
        from databricks.koalas.utils import validate_arguments_and_invoke_function
        return validate_arguments_and_invoke_function(kseries._to_internal_pandas(), self.to_clipboard, pd.Series.to_clipboard, args)
    to_clipboard.__doc__ = DataFrame.to_clipboard.__doc__

    def to_dict(self, into: Any = dict) -> Any:
        args = locals()
        kseries: "Series" = self
        from databricks.koalas.utils import validate_arguments_and_invoke_function
        return validate_arguments_and_invoke_function(kseries._to_internal_pandas(), self.to_dict, pd.Series.to_dict, args)

    def to_latex(self, buf: Optional[Any] = None, columns: Optional[Any] = None, col_space: Optional[Any] = None, header: bool = True,
                 index: bool = True, na_rep: str = 'NaN', formatters: Optional[Any] = None, float_format: Optional[Any] = None, sparsify: Optional[Any] = None,
                 index_names: bool = True, bold_rows: bool = False, column_format: Optional[Any] = None, longtable: Optional[Any] = None, escape: Optional[Any] = None,
                 encoding: Optional[Any] = None, decimal: str = '.', multicolumn: Optional[Any] = None, multicolumn_format: Optional[Any] = None, multirow: Optional[Any] = None) -> str:
        args = locals()
        kseries: "Series" = self
        from databricks.koalas.utils import validate_arguments_and_invoke_function
        return validate_arguments_and_invoke_function(kseries._to_internal_pandas(), self.to_latex, pd.Series.to_latex, args)
    to_latex.__doc__ = DataFrame.to_latex.__doc__

    def to_pandas(self) -> pd.Series:
        return self._to_internal_pandas().copy()

    def toPandas(self) -> pd.Series:
        import warnings
        warnings.warn('Series.toPandas is deprecated as of Series.to_pandas. Please use the API instead.', FutureWarning)
        return self.to_pandas()
    toPandas.__doc__ = to_pandas.__doc__

    def to_list(self) -> list:
        return self._to_internal_pandas().tolist()
    tolist = to_list

    def drop_duplicates(self, keep: Union[str, bool] = 'first', inplace: bool = False) -> Optional["Series"]:
        from databricks.koalas.utils import validate_bool_kwarg
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf: DataFrame = self._kdf[[self.name]].drop_duplicates(keep=keep)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def reindex(self, index: Optional[Any] = None, fill_value: Any = None) -> "Series":
        return first_series(self.to_frame().reindex(index=index, fill_value=fill_value)).rename(self.name)

    def reindex_like(self, other: Union["Series", DataFrame]) -> "Series":
        if isinstance(other, (Series, DataFrame)):
            return self.reindex(index=other.index)
        else:
            raise TypeError('other must be a Koalas Series or DataFrame')

    def fillna(self, value: Any = None, method: Optional[str] = None, axis: Optional[Any] = None, inplace: bool = False, limit: Optional[int] = None) -> Optional["Series"]:
        kser: "Series" = self._fillna(value=value, method=method, axis=axis, limit=limit)
        if method is not None:
            kser = DataFrame(kser._kdf._internal.resolved_copy)._kser_for(self._column_label)
        from databricks.koalas.utils import validate_bool_kwarg
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if inplace:
            self._kdf._update_internal_frame(kser._kdf._internal, requires_same_anchor=False)
            return None
        else:
            return kser._with_new_scol(kser.spark.column)

    def _fillna(self, value: Any = None, method: Optional[str] = None, axis: Optional[Any] = None, limit: Optional[int] = None, part_cols: Tuple[Any, ...] = ()) -> "Series":
        axis = axis  # assume validate_axis called externally
        if axis != 0:
            raise NotImplementedError("fillna currently only works for axis=0 or axis='index'")
        if value is None and method is None:
            raise ValueError("Must specify a fillna 'value' or 'method' parameter.")
        if method is not None and method not in ['ffill', 'pad', 'backfill', 'bfill']:
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
                raise TypeError('Unsupported type %s' % type(value).__name__)
            if limit is not None:
                raise ValueError('limit parameter for value is not support now')
            scol = F.when(cond, value).otherwise(scol)
        else:
            if method in ['ffill', 'pad']:
                func = F.last
                end = Window.currentRow - 1
                if limit is not None:
                    begin = Window.currentRow - limit
                else:
                    begin = Window.unboundedPreceding
            elif method in ['bfill', 'backfill']:
                func = F.first
                begin = Window.currentRow + 1
                if limit is not None:
                    end = Window.currentRow + limit
                else:
                    end = Window.unboundedFollowing
            window = Window.partitionBy(*part_cols).orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(begin, end)
            scol = F.when(cond, func(scol, True).over(window)).otherwise(scol)
        return DataFrame(self._kdf._internal.with_new_spark_column(self._column_label, scol.alias(str(self.name))))._kser_for(self._column_label)

    def dropna(self, axis: Union[int, str] = 0, inplace: bool = False, **kwargs: Any) -> Optional["Series"]:
        from databricks.koalas.utils import validate_bool_kwarg
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf: DataFrame = self._kdf[[self.name]].dropna(axis=axis, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def clip(self, lower: Optional[Union[float, int]] = None, upper: Optional[Union[float, int]] = None) -> "Series":
        if isinstance(lower, Iterable) or isinstance(upper, Iterable):
            raise ValueError("List-like value are not supported for 'lower' and 'upper' at the moment")
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

    def drop(self, labels: Optional[Any] = None, index: Optional[Any] = None, level: Optional[Any] = None) -> "Series":
        return first_series(self._drop(labels=labels, index=index, level=level))

    def _drop(self, labels: Optional[Any] = None, index: Optional[Any] = None, level: Optional[Any] = None) -> DataFrame:
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
            if isinstance(index, tuple) or not isinstance(index, list):
                index = [index]
            drop_index_scols = []
            for idxes in index:
                if not isinstance(idxes, tuple):
                    idxes = (idxes,)
                try:
                    index_scols = [internal.index_spark_columns[lvl] == idx for lvl, idx in enumerate(idxes, level)]
                except IndexError:
                    raise KeyError('Key length ({}) exceeds index depth ({})'.format(internal.index_level, len(idxes)))
                drop_index_scols.append(reduce(lambda x, y: x & y, index_scols))
            cond = ~reduce(lambda x, y: x | y, drop_index_scols)
            return DataFrame(internal.with_filter(cond))
        else:
            raise ValueError("Need to specify at least one of 'labels' or 'index'")

    def head(self, n: int = 5) -> "Series":
        return first_series(self.to_frame().head(n)).rename(self.name)

    def last(self, offset: Union[str, "DateOffset"]) -> "Series":
        return first_series(self.to_frame().last(offset)).rename(self.name)

    def first(self, offset: Union[str, "DateOffset"]) -> "Series":
        return first_series(self.to_frame().first(offset)).rename(self.name)

    def unique(self) -> "Series":
        sdf: SparkDataFrame = self._internal.spark_frame.select(self.spark.column).distinct()
        from databricks.koalas.internal import infer_return_type, spark_type_to_pandas_dtype, SeriesType, ScalarType
        internal = InternalFrame(spark_frame=sdf,
                                 index_spark_columns=None,
                                 column_labels=[self._column_label],
                                 data_spark_columns=[SF.scol_for(sdf, self._internal.data_spark_column_names[0])],
                                 data_dtypes=[self.dtype],
                                 column_label_names=self._internal.column_label_names)
        return first_series(DataFrame(internal))

    def sort_values(self, ascending: Union[bool, list] = True, inplace: bool = False, na_position: str = 'last') -> Optional["Series"]:
        from databricks.koalas.utils import validate_bool_kwarg
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf: DataFrame = self._kdf[[self.name]]._sort(by=[self.spark.column], ascending=ascending, inplace=False, na_position=na_position)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def sort_index(self, axis: Union[int, str] = 0, level: Optional[Union[int, str, list]] = None, ascending: bool = True, inplace: bool = False, kind: Optional[str] = None, na_position: str = 'last') -> Optional["Series"]:
        from databricks.koalas.utils import validate_bool_kwarg
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf: DataFrame = self._kdf[[self.name]].sort_index(axis=axis, level=level, ascending=ascending, kind=kind, na_position=na_position)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1, copy: bool = True) -> "Series":
        assert copy is True
        return first_series(self.to_frame().swaplevel(i, j, axis=0)).rename(self.name)

    def swapaxes(self, i: Union[int, str], j: Union[int, str], copy: bool = True) -> "Series":
        assert copy is True
        i = i  # assume validate_axis(i) done externally
        j = j
        if not i == j == 0:
            raise ValueError('Axis must be 0 for Series')
        return self.copy()

    def add_prefix(self, prefix: str) -> "Series":
        assert isinstance(prefix, str)
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame.select([F.concat(F.lit(prefix), index_spark_column).alias(index_spark_column_name)
                                            for index_spark_column, index_spark_column_name in zip(internal.index_spark_columns, internal.index_spark_column_names)] + internal.data_spark_columns)
        from databricks.koalas.internal import DEFAULT_SERIES_NAME
        return first_series(DataFrame(internal.with_new_sdf(sdf, index_dtypes=[None] * internal.index_level)))

    def add_suffix(self, suffix: str) -> "Series":
        assert isinstance(suffix, str)
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame.select([F.concat(index_spark_column, F.lit(suffix)).alias(index_spark_column_name)
                                            for index_spark_column, index_spark_column_name in zip(internal.index_spark_columns, internal.index_spark_column_names)] + internal.data_spark_columns)
        return first_series(DataFrame(internal.with_new_sdf(sdf, index_dtypes=[None] * internal.index_level)))

    def corr(self, other: "Series", method: str = 'pearson') -> float:
        columns = ['__corr_arg1__', '__corr_arg2__']
        kdf: DataFrame = self._kdf.assign(__corr_arg1__=self, __corr_arg2__=other)[columns]
        kdf.columns = columns
        from databricks.koalas.ml import corr
        c: pd.DataFrame = corr(kdf, method=method)
        return cast(float, c.loc[tuple(columns)])

    def nsmallest(self, n: int = 5) -> "Series":
        return self.sort_values(ascending=True).head(n)

    def nlargest(self, n: int = 5) -> "Series":
        return self.sort_values(ascending=False).head(n)

    def append(self, to_append: Union["Series", Tuple["Series", ...]], ignore_index: bool = False, verify_integrity: bool = False) -> "Series":
        return first_series(self.to_frame().append(to_append.to_frame(), ignore_index, verify_integrity)).rename(self.name)

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, replace: bool = False, random_state: Optional[int] = None) -> "Series":
        return first_series(self.to_frame().sample(n=n, frac=frac, replace=replace, random_state=random_state)).rename(self.name)
    sample.__doc__ = DataFrame.sample.__doc__

    def hist(self, bins: int = 10, **kwds: Any) -> Any:
        return self.plot.hist(bins, **kwds)
    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def apply(self, func: Any, args: Tuple = (), **kwds: Any) -> "Series":
        assert callable(func), 'the first argument should be a callable function.'
        import inspect
        try:
            spec = inspect.getfullargspec(func)
            return_sig = spec.annotations.get('return', None)
            should_infer_schema: bool = return_sig is None
        except TypeError:
            should_infer_schema = True

        from functools import wraps
        apply_each = wraps(func)(lambda s: s.apply(func, args=args, **kwds))
        from databricks.koalas.typedef import infer_return_type, ScalarType
        if should_infer_schema:
            return self.koalas._transform_batch(apply_each, None)
        else:
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, ScalarType):
                raise ValueError('Expected the return type of this function to be of scalar type, but found type {}'.format(sig_return))
            return_type = cast(ScalarType, sig_return)
            return self.koalas._transform_batch(apply_each, return_type)

    def aggregate(self, func: Union[str, list]) -> Union[Any, "Series"]:
        if isinstance(func, list):
            return first_series(self.to_frame().aggregate(func)).rename(self.name)
        elif isinstance(func, str):
            return getattr(self, func)()
        else:
            raise ValueError('func must be a string or list of strings')
    agg = aggregate

    def transpose(self, *args: Any, **kwargs: Any) -> "Series":
        return self.copy()
    T = property(transpose)

    def transform(self, func: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Union["Series", DataFrame]:
        axis = axis
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

    def transform_batch(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        import warnings
        warnings.warn('Series.transform_batch is deprecated as of Series.koalas.transform_batch. Please use the API instead.', FutureWarning)
        return self.koalas.transform_batch(func, *args, **kwargs)
    transform_batch.__doc__ = KoalasPlotAccessor.hist.__doc__

    def round(self, decimals: int = 0) -> "Series":
        if not isinstance(decimals, int):
            raise ValueError('decimals must be an integer')
        scol = F.round(self.spark.column, decimals)
        return self._with_new_scol(scol)

    def quantile(self, q: Union[float, Iterable[float]] = 0.5, accuracy: int = 10000) -> Union[float, "Series"]:
        if isinstance(q, Iterable):
            return first_series(self.to_frame().quantile(q=q, axis=0, numeric_only=False, accuracy=accuracy)).rename(self.name)
        else:
            if not isinstance(accuracy, int):
                raise ValueError('accuracy must be an integer; however, got [%s]' % type(accuracy).__name__)
            if not isinstance(q, float):
                raise ValueError('q must be a float or an array of floats; however, [%s] found.' % type(q))
            if q < 0.0 or q > 1.0:
                raise ValueError('percentiles should all be in the interval [0, 1].')

            def quantile_func(spark_column: Column, spark_type: Any) -> Column:
                if isinstance(spark_type, (BooleanType, NumericType)):
                    return SF.percentile_approx(spark_column.cast(DoubleType()), q, accuracy)
                else:
                    from databricks.koalas.typedef import spark_type_to_pandas_dtype
                    raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            return self._reduce_for_stat_function(quantile_func, name='quantile')

    def rank(self, method: str = 'average', ascending: bool = True) -> "Series":
        return self._rank(method, ascending).spark.analyzed

    def _rank(self, method: str = 'average', ascending: bool = True, *, part_cols: Tuple[Any, ...] = ()) -> "Series":
        if method not in ['average', 'min', 'max', 'first', 'dense']:
            msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
            raise ValueError(msg)
        if self._internal.index_level > 1:
            raise ValueError('rank do not support index now')
        asc_func = (lambda scol: scol.asc()) if ascending else (lambda scol: scol.desc())
        if method == 'first':
            window = Window.orderBy(asc_func(self.spark.column), asc_func(F.col(NATURAL_ORDER_COLUMN_NAME))).partitionBy(*part_cols).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            scol = F.row_number().over(window)
        elif method == 'dense':
            window = Window.orderBy(asc_func(self.spark.column)).partitionBy(*part_cols).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            scol = F.dense_rank().over(window)
        else:
            if method == 'average':
                stat_func = F.mean
            elif method == 'min':
                stat_func = F.min
            elif method == 'max':
                stat_func = F.max
            window1 = Window.orderBy(asc_func(self.spark.column)).partitionBy(*part_cols).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            window2 = Window.partitionBy([self.spark.column] + list(part_cols)).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            scol = stat_func(F.row_number().over(window1)).over(window2)
        kser: "Series" = self._with_new_scol(scol)
        return kser.astype(np.float64)

    def filter(self, items: Optional[Any] = None, like: Optional[Any] = None, regex: Optional[Any] = None, axis: Optional[Any] = None) -> "Series":
        axis = axis
        if axis == 1:
            raise ValueError('Series does not support columns axis.')
        return first_series(self.to_frame().filter(items=items, like=like, regex=regex, axis=axis)).rename(self.name)
    filter.__doc__ = DataFrame.filter.__doc__

    def describe(self, percentiles: Optional[Any] = None) -> "Series":
        return first_series(self.to_frame().describe(percentiles)).rename(self.name)
    describe.__doc__ = DataFrame.describe.__doc__

    def diff(self, periods: int = 1) -> "Series":
        return self._diff(periods).spark.analyzed

    def _diff(self, periods: int, *, part_cols: Tuple[Any, ...] = ()) -> "Series":
        if not isinstance(periods, int):
            raise ValueError('periods should be an int; however, got [%s]' % type(periods).__name__)
        window = Window.partitionBy(*part_cols).orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        scol = self.spark.column - F.lag(self.spark.column, periods).over(window)
        return self._with_new_scol(scol, dtype=self.dtype)

    def idxmax(self, skipna: bool = True) -> Any:
        sdf: SparkDataFrame = self._internal.spark_frame
        scol: Column = self.spark.column
        index_scols = self._internal.index_spark_columns
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_last()), NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_first()), NATURAL_ORDER_COLUMN_NAME)
        results = sdf.select([scol] + index_scols).take(1)
        if len(results) == 0:
            raise ValueError('attempt to get idxmin of an empty sequence')
        if results[0][0] is None:
            return np.nan
        values = list(results[0][1:])
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def idxmin(self, skipna: bool = True) -> Any:
        sdf: SparkDataFrame = self._internal.spark_frame
        scol: Column = self.spark.column
        index_scols = self._internal.index_spark_columns
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_last()), NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_first()), NATURAL_ORDER_COLUMN_NAME)
        results = sdf.select([scol] + index_scols).take(1)
        if len(results) == 0:
            raise ValueError('attempt to get idxmin of an empty sequence')
        if results[0][0] is None:
            return np.nan
        values = list(results[0][1:])
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def pop(self, item: Any) -> Any:
        from databricks.koalas.utils import is_name_like_value, is_name_like_tuple
        if not is_name_like_value(item):
            raise ValueError("'key' should be string or tuple that contains strings")
        if not is_name_like_tuple(item):
            item = (item,)
        if self._internal.index_level < len(item):
            raise KeyError('Key length ({}) exceeds index depth ({})'.format(len(item), self._internal.index_level))
        internal = self._internal
        scols = internal.index_spark_columns[len(item):] + [self.spark.column]
        rows = [internal.spark_columns[level] == index for level, index in enumerate(item)]
        sdf = internal.spark_frame.filter(reduce(lambda x, y: x & y, rows)).select(scols)
        kdf: DataFrame = self._drop(item)
        self._update_anchor(kdf)
        if self._internal.index_level == len(item):
            pdf: pd.DataFrame = sdf.limit(2).toPandas()
            length = len(pdf)
            if length == 1:
                return pdf[internal.data_spark_column_names[0]].iloc[0]
            item_string = str(item)
            sdf = sdf.withColumn(SPARK_DEFAULT_INDEX_NAME, F.lit(str(item_string)))
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=[F.col(SPARK_DEFAULT_INDEX_NAME)], column_labels=[self._column_label], data_dtypes=[self.dtype])
            return first_series(DataFrame(internal))
        else:
            internal = internal.copy(spark_frame=sdf, index_spark_columns=[F.col(c) for c in internal.index_spark_column_names[len(item):]], index_dtypes=internal.index_dtypes[len(item):], index_names=self._internal.index_names[len(item):], data_spark_columns=[F.col(internal.data_spark_column_names[0])])
            return first_series(DataFrame(internal))

    def copy(self, deep: Optional[bool] = None) -> "Series":
        return self._kdf.copy()._kser_for(self._column_label)

    def mode(self, dropna: bool = True) -> "Series":
        ser_count: "Series" = self.value_counts(dropna=dropna, sort=False)
        sdf_count: SparkDataFrame = ser_count._internal.spark_frame
        most_value = ser_count.max()
        sdf_most_value = sdf_count.filter('count == {}'.format(most_value))
        sdf = sdf_most_value.select(F.col(SPARK_DEFAULT_INDEX_NAME).alias(SPARK_DEFAULT_SERIES_NAME))
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=None, column_labels=[None])
        return first_series(DataFrame(internal))

    def keys(self) -> Any:
        return self.index

    def replace(self, to_replace: Optional[Any] = None, value: Any = None, regex: bool = False) -> "Series":
        if to_replace is None:
            return self.fillna(method='ffill')
        if not isinstance(to_replace, (str, list, tuple, dict, int, float)):
            raise ValueError("'to_replace' should be one of str, list, tuple, dict, int, float")
        if regex:
            raise NotImplementedError('replace currently not support for regex')
        to_replace = list(to_replace) if isinstance(to_replace, tuple) else to_replace
        value = list(value) if isinstance(value, tuple) else value
        if isinstance(to_replace, list) and isinstance(value, list):
            if not len(to_replace) == len(value):
                raise ValueError('Replacement lists must match in length. Expecting {} got {}'.format(len(to_replace), len(value)))
            to_replace = {k: v for k, v in zip(to_replace, value)}
        if isinstance(to_replace, dict):
            is_start: bool = True
            if len(to_replace) == 0:
                current = self.spark.column
            else:
                for to_replace_, val in to_replace.items():
                    cond = F.isnan(self.spark.column) | self.spark.column.isNull() if pd.isna(to_replace_) else self.spark.column == F.lit(to_replace_)
                    if is_start:
                        current = F.when(cond, val)
                        is_start = False
                    else:
                        current = current.when(cond, val)
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
        from databricks.koalas.utils import combine_frames
        combined = combine_frames(self._kdf, other._kdf, how='leftouter')
        this_scol = combined['this']._internal.spark_column_for(self._column_label)
        that_scol = combined['that']._internal.spark_column_for(other._column_label)
        scol = F.when(that_scol.isNotNull(), that_scol).otherwise(this_scol).alias(self._kdf._internal.spark_column_name_for(self._column_label))
        internal = combined['this']._internal.with_new_spark_column(self._column_label, scol)
        self._kdf._update_internal_frame(internal.resolved_copy, requires_same_anchor=False)

    def where(self, cond: "Series", other: Any = np.nan) -> "Series":
        should_try_ops_on_diff_frame: bool = (not self._kdf is cond._kdf) or (isinstance(other, Series) and (not self._kdf is other._kdf))
        if should_try_ops_on_diff_frame:
            kdf = self.to_frame()
            tmp_cond_col = "__tmp_cond_col__"
            tmp_other_col = "__tmp_other_col__"
            kdf[tmp_cond_col] = cond
            kdf[tmp_other_col] = other
            condition = F.when(kdf[tmp_cond_col].spark.column, kdf._kser_for(kdf._internal.column_labels[0]).spark.column).otherwise(kdf[tmp_other_col].spark.column).alias(kdf._internal.data_spark_column_names[0])
            internal = kdf._internal.with_new_columns([condition], column_labels=self._internal.column_labels)
            return first_series(DataFrame(internal))
        else:
            if isinstance(other, Series):
                other = other.spark.column
            condition = F.when(cond.spark.column, self.spark.column).otherwise(other).alias(self._internal.data_spark_column_names[0])
            return self._with_new_scol(condition)

    def mask(self, cond: "Series", other: Any = np.nan) -> "Series":
        return self.where(~cond, other)

    def xs(self, key: Union[Any, Tuple[Any, ...]], level: Optional[Union[int, str, list]] = None) -> "Series":
        if not isinstance(key, tuple):
            key = (key,)
        if level is None:
            level = 0
        internal = self._internal
        scols = internal.index_spark_columns[:level] + internal.index_spark_columns[level + len(key):] + [self.spark.column]
        rows = [internal.spark_columns[lvl] == index for lvl, index in enumerate(key, level)]
        sdf = internal.spark_frame.filter(reduce(lambda x, y: x & y, rows)).select(scols)
        if internal.index_level == len(key):
            pdf = sdf.limit(2).toPandas()
            if len(pdf) == 1:
                return pdf[internal.data_spark_column_names[0]].iloc[0]
        index_spark_column_names = internal.index_spark_column_names[:level] + internal.index_spark_column_names[level + len(key):]
        index_names = internal.index_names[:level] + internal.index_names[level + len(key):]
        index_dtypes = internal.index_dtypes[:level] + internal.index_dtypes[level + len(key):]
        internal = internal.copy(spark_frame=sdf, index_spark_columns=[F.col(c) for c in index_spark_column_names], index_names=index_names, index_dtypes=index_dtypes, data_spark_columns=[F.col(internal.data_spark_column_names[0])])
        return first_series(DataFrame(internal))

    def pct_change(self, periods: int = 1) -> "Series":
        scol = self.spark.column
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        prev_row = F.lag(scol, periods).over(window)
        return self._with_new_scol((scol - prev_row) / prev_row).spark.analyzed

    def _cum(self, func: Any, skipna: bool, part_cols: Tuple[Any, ...] = (), ascending: bool = True) -> "Series":
        if ascending:
            window = Window.orderBy(F.asc(NATURAL_ORDER_COLUMN_NAME)).partitionBy(*part_cols).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        else:
            window = Window.orderBy(F.desc(NATURAL_ORDER_COLUMN_NAME)).partitionBy(*part_cols).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        if skipna:
            scol = F.when(self.spark.column.isNull(), F.lit(None)).otherwise(func(self.spark.column).over(window))
        else:
            scol = F.when(F.max(self.spark.column.isNull()).over(window), F.lit(None)).otherwise(func(self.spark.column).over(window))
        return self._with_new_scol(scol)

    def _cumsum(self, skipna: bool, part_cols: Tuple[Any, ...] = ()) -> "Series":
        kser: "Series" = self
        if isinstance(kser.spark.data_type, BooleanType):
            kser = kser.spark.transform(lambda scol: scol.cast(LongType()))
        elif not isinstance(kser.spark.data_type, NumericType):
            raise TypeError('Could not convert {} ({}) to numeric'.format(kser.spark.data_type, kser.spark.data_type.simpleString()))
        return kser._cum(F.sum, skipna, part_cols)

    def _cumprod(self, skipna: bool, part_cols: Tuple[Any, ...] = ()) -> "Series":
        if isinstance(self.spark.data_type, BooleanType):
            scol = self._cum(lambda scol: F.min(F.coalesce(scol, F.lit(True))), skipna, part_cols).spark.column.cast(LongType())
        elif isinstance(self.spark.data_type, NumericType):
            num_zeros = self._cum(lambda scol: F.sum(F.when(scol == 0, 1).otherwise(0)), skipna, part_cols).spark.column
            num_negatives = self._cum(lambda scol: F.sum(F.when(scol < 0, 1).otherwise(0)), skipna, part_cols).spark.column
            sign = F.when(num_negatives % 2 == 0, 1).otherwise(-1)
            abs_prod = F.exp(self._cum(lambda scol: F.sum(F.log(F.abs(scol))), skipna, part_cols).spark.column)
            scol = F.when(num_zeros > 0, 0).otherwise(sign * abs_prod)
            if isinstance(self.spark.data_type, IntegralType):
                scol = F.round(scol).cast(LongType())
        else:
            raise TypeError('Could not convert {} ({}) to numeric'.format(self.spark.data_type, self.spark.data_type.simpleString()))
        return self._with_new_scol(scol)

    def __getitem__(self, key: Any) -> "Series":
        try:
            if isinstance(key, slice) and any((type(n) == int for n in [key.start, key.stop])) or (isinstance(key, int) and (not isinstance(self.index.spark.data_type, (IntegerType, LongType)))):
                return self.iloc[key]
            return self.loc[key]
        except KeyError:
            raise KeyError('Key length ({}) exceeds index depth ({})'.format(len(key), self._internal.index_level))

    def __getattr__(self, item: str) -> Any:
        if item.startswith('__'):
            raise AttributeError(item)
        if hasattr(MissingPandasLikeSeries, item):
            property_or_func = getattr(MissingPandasLikeSeries, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                from functools import partial
                return partial(property_or_func, self)
        raise AttributeError("'Series' object has no attribute '{}'".format(item))

    def _to_internal_pandas(self) -> pd.Series:
        return self._kdf._internal.to_pandas_frame[self.name]

    def __repr__(self) -> str:
        max_display_count = get_option('display.max_rows')
        if max_display_count is None:
            return self._to_internal_pandas().to_string(name=self.name, dtype=self.dtype)
        pser: pd.Series = self._kdf._get_or_create_repr_pandas_cache(max_display_count)[self.name]
        pser_length = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit('\n', 1)
            import re
            REPR_PATTERN = re.compile('Length: (?P<length>[0-9]+)')
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group('length')
                dtype_name = str(self.dtype.name)
                if self.name is None:
                    footer = '\ndtype: {dtype}\nShowing only the first {length}'.format(length=length, dtype=dtype_name)
                else:
                    footer = '\nName: {name}, dtype: {dtype}\nShowing only the first {length}'.format(length=length, name=self.name, dtype=dtype_name)
                return rest + footer
        return pser.to_string(name=self.name, dtype=self.dtype)

    def __dir__(self) -> list:
        if not isinstance(self.spark.data_type, StructType):
            fields: list = []
        else:
            fields = [f for f in self.spark.data_type.fieldNames() if ' ' not in f]
        return super().__dir__() + fields

    def __iter__(self) -> Iterator[Any]:
        return MissingPandasLikeSeries.__iter__(self)
    if sys.version_info >= (3, 7):

        @classmethod
        def __class_getitem__(cls, params: Any) -> Any:
            from databricks.koalas.typedef import _create_type_for_series_type
            return _create_type_for_series_type(params)
    elif (3, 5) <= sys.version_info < (3, 7):
        is_series = None


def unpack_scalar(sdf: SparkDataFrame) -> Any:
    l = sdf.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row = l.iloc[0]
    l2 = list(row)
    assert len(l2) == 1, (row, l2)
    return l2[0]


def first_series(df: Union[DataFrame, pd.DataFrame]) -> Series:
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    if isinstance(df, DataFrame):
        return df._kser_for(df._internal.column_labels[0])
    else:
        return df[df.columns[0]]
