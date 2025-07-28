#!/usr/bin/env python
"""
Base and utility classes for Koalas objects.
"""
from abc import ABCMeta, abstractmethod
import datetime
from functools import wraps, partial
from itertools import chain
from typing import Any, Callable, Optional, Tuple, Union, cast, TYPE_CHECKING, Sequence
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like, CategoricalDtype
from pyspark import sql as spark
from pyspark.sql import functions as F, Window, Column
from pyspark.sql.types import BooleanType, DateType, DoubleType, FloatType, IntegralType, LongType, NumericType, StringType, TimestampType
from databricks import koalas as ks
from databricks.koalas import numpy_compat
from databricks.koalas.config import get_option, option_context
from databricks.koalas.internal import InternalFrame, NATURAL_ORDER_COLUMN_NAME, SPARK_DEFAULT_INDEX_NAME
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkIndexOpsMethods
from databricks.koalas.typedef import Dtype, as_spark_type, extension_dtypes, koalas_dtype, spark_type_to_pandas_dtype
from databricks.koalas.utils import combine_frames, same_anchor, scol_for, validate_axis, ERROR_MESSAGE_CANNOT_COMBINE
from databricks.koalas.frame import DataFrame

if TYPE_CHECKING:
    from databricks.koalas.indexes import Index
    from databricks.koalas.series import Series

def should_alignment_for_column_op(self: Any, other: Any) -> bool:
    from databricks.koalas.series import Series
    if isinstance(self, Series) and isinstance(other, Series):
        return not same_anchor(self, other)
    else:
        return self._internal.spark_frame is not other._internal.spark_frame

def align_diff_index_ops(func: Callable, this_index_ops: Any, *args: Any) -> Union["Index", "Series"]:
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
    if isinstance(this_index_ops, Series) and all((isinstance(col, Series) for col in cols)):
        combined = combine_frames(this_index_ops.to_frame(), *[cast(Series, col).rename(i) for i, col in enumerate(cols)], how='full')
        return column_op(func)(
            combined['this']._kser_for(combined['this']._internal.column_labels[0]),
            *[combined['that']._kser_for(label) for label in combined['that']._internal.column_labels]
        ).rename(this_index_ops.name)
    else:
        self_len = len(this_index_ops)
        if any((len(col) != self_len for col in args if isinstance(col, IndexOpsMixin))):
            raise ValueError('operands could not be broadcast together with shapes')
        with option_context('compute.default_index_type', 'distributed-sequence'):
            if isinstance(this_index_ops, Index) and all((isinstance(col, Index) for col in cols)):
                return Index(
                    column_op(func)(
                        this_index_ops.to_series().reset_index(drop=True),
                        *[arg.to_series().reset_index(drop=True) if isinstance(arg, Index) else arg for arg in args]
                    ).sort_index(),
                    name=this_index_ops.name
                )
            elif isinstance(this_index_ops, Series):
                this = this_index_ops.reset_index()
                that = [cast(Series, col.to_series() if isinstance(col, Index) else col).rename(i).reset_index(drop=True) for i, col in enumerate(cols)]
                combined = combine_frames(this, *that, how='full').sort_index()
                combined = combined.set_index(combined._internal.column_labels[:this_index_ops._internal.index_level])
                combined.index.names = this_index_ops._internal.index_names
                return column_op(func)(
                    first_series(combined['this']),
                    *[combined['that']._kser_for(label) for label in combined['that']._internal.column_labels]
                ).rename(this_index_ops.name)
            else:
                this = cast("Index", this_index_ops).to_frame().reset_index(drop=True)
                that_series = next((col for col in cols if isinstance(col, Series)))
                that_frame = that_series._kdf[[cast(Series, col.to_series() if isinstance(col, Index) else col).rename(i) for i, col in enumerate(cols)]]
                combined = combine_frames(this, that_frame.reset_index()).sort_index()
                self_index = combined['this'].set_index(combined['this']._internal.column_labels).index
                other = combined['that'].set_index(combined['that']._internal.column_labels[:that_series._internal.index_level])
                other.index.names = that_series._internal.index_names
                return column_op(func)(
                    self_index,
                    *[other._kser_for(label) for label, col in zip(other._internal.column_labels, cols)]
                ).rename(that_series.name)

def booleanize_null(scol: Column, f: Callable) -> Column:
    """
    Booleanize Null in Spark Column
    """
    comp_ops = [getattr(Column, '__{}__'.format(comp_op)) for comp_op in ['eq', 'ne', 'lt', 'le', 'ge', 'gt']]
    if f in comp_ops:
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
        cols = [arg for arg in args if isinstance(arg, IndexOpsMixin)]
        if all((not should_alignment_for_column_op(self, col) for col in cols)):
            args = [arg.spark.column if isinstance(arg, IndexOpsMixin) else arg for arg in args]
            scol: Column = f(self.spark.column, *args)
            spark_type = self._internal.spark_frame.select(scol).schema[0].dataType
            use_extension_dtypes: bool = any((isinstance(col.dtype, extension_dtypes) for col in [self] + cols))
            dtype = spark_type_to_pandas_dtype(spark_type, use_extension_dtypes=use_extension_dtypes)
            if not isinstance(dtype, extension_dtypes):
                scol = booleanize_null(scol, f)
            if isinstance(self, Series) or not any((isinstance(col, Series) for col in cols)):
                index_ops = self._with_new_scol(scol, dtype=dtype)
            else:
                kser = next((col for col in cols if isinstance(col, Series)))
                index_ops = kser._with_new_scol(scol, dtype=dtype)
        elif get_option('compute.ops_on_diff_frames'):
            index_ops = align_diff_index_ops(f, self, *args)
        else:
            raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)
        if not all((self.name == col.name for col in cols)):
            index_ops = index_ops.rename(None)
        return index_ops
    return wrapper

def numpy_column_op(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(self: "IndexOpsMixin", *args: Any) -> "IndexOpsMixin":
        new_args = []
        for arg in args:
            if isinstance(self.spark.data_type, LongType) and isinstance(arg, np.timedelta64):
                new_args.append(float(arg / np.timedelta64(1, 's')))
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
    def _with_new_scol(self, scol: Column, *, dtype: Optional[Dtype] = None) -> "IndexOpsMixin":
        pass

    @property
    @abstractmethod
    def _column_label(self) -> Any:
        pass

    @property
    @abstractmethod
    def spark(self) -> Any:
        pass

    @property
    def spark_column(self) -> Any:
        warnings.warn('Series.spark_column is deprecated as of Series.spark.column. Please use the API instead.', FutureWarning)
        return self.spark.column
    spark_column.__doc__ = SparkIndexOpsMethods.column.__doc__

    __neg__ = column_op(Column.__neg__)

    def __add__(self, other: Any) -> "IndexOpsMixin":
        if not isinstance(self.spark.data_type, StringType) and (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType) or isinstance(other, str)):
            raise TypeError('string addition can only be applied to string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('addition can not be applied to date times.')
        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType):
                return column_op(F.concat)(self, other)
            elif isinstance(other, str):
                return column_op(F.concat)(self, F.lit(other))
            else:
                raise TypeError('string addition can only be applied to string series or literals.')
        else:
            return column_op(Column.__add__)(self, other)

    def __sub__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('substraction can not be applied to string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            msg: str = "Note that there is a behavior difference of timestamp subtraction. The timestamp subtraction returns an integer in seconds, whereas pandas returns 'timedelta64[ns]'."
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, TimestampType):
                warnings.warn(msg, UserWarning)
                return self.astype('long') - other.astype('long')
            elif isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return self.astype('long') - F.lit(other).cast(as_spark_type('long'))
            else:
                raise TypeError('datetime subtraction can only be applied to datetime series.')
        elif isinstance(self.spark.data_type, DateType):
            msg = "Note that there is a behavior difference of date subtraction. The date subtraction returns an integer in days, whereas pandas returns 'timedelta64[ns]'."
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, DateType):
                warnings.warn(msg, UserWarning)
                return column_op(F.datediff)(self, other).astype('long')
            elif isinstance(other, datetime.date) and (not isinstance(other, datetime.datetime)):
                warnings.warn(msg, UserWarning)
                return column_op(F.datediff)(self, F.lit(other)).astype('long')
            else:
                raise TypeError('date subtraction can only be applied to date series.')
        return column_op(Column.__sub__)(self, other)

    def __mul__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(other, str):
            raise TypeError('multiplication can not be applied to a string literal.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('multiplication can not be applied to date times.')
        if isinstance(self.spark.data_type, IntegralType) and isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType):
            return column_op(SF.repeat)(other, self)
        if isinstance(self.spark.data_type, StringType):
            if (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, IntegralType)) or isinstance(other, int):
                return column_op(SF.repeat)(self, other)
            else:
                raise TypeError('a string series can only be multiplied to an int series or literal')
        return column_op(Column.__mul__)(self, other)

    def __truediv__(self, other: Any) -> "IndexOpsMixin":
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
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('division can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def truediv(left: Column, right: Any) -> Column:
            return F.when(F.lit(right != 0) | F.lit(right).isNull(), left.__div__(right))\
                .otherwise(F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left)\
                .otherwise(F.lit(np.inf).__div__(left)))
        return numpy_column_op(truediv)(self, other)

    def __mod__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('modulo can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('modulo can not be applied to date times.')

        def mod(left: Column, right: Any) -> Column:
            return (left % right + right) % right
        return column_op(mod)(self, other)

    def __radd__(self, other: Any) -> "IndexOpsMixin":
        if not isinstance(self.spark.data_type, StringType) and isinstance(other, str):
            raise TypeError('string addition can only be applied to string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('addition can not be applied to date times.')
        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, str):
                return self._with_new_scol(F.concat(F.lit(other), self.spark.column))
            else:
                raise TypeError('string addition can only be applied to string series or literals.')
        else:
            return column_op(Column.__radd__)(self, other)

    def __rsub__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError('substraction can not be applied to string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            msg: str = "Note that there is a behavior difference of timestamp subtraction. The timestamp subtraction returns an integer in seconds, whereas pandas returns 'timedelta64[ns]'."
            if isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return -(self.astype('long') - F.lit(other).cast(as_spark_type('long')))
            else:
                raise TypeError('datetime subtraction can only be applied to datetime series.')
        elif isinstance(self.spark.data_type, DateType):
            msg = "Note that there is a behavior difference of date subtraction. The date subtraction returns an integer in days, whereas pandas returns 'timedelta64[ns]'."
            if isinstance(other, datetime.date) and (not isinstance(other, datetime.datetime)):
                warnings.warn(msg, UserWarning)
                return -column_op(F.datediff)(self, F.lit(other)).astype('long')
            else:
                raise TypeError('date subtraction can only be applied to date series.')
        return column_op(Column.__rsub__)(self, other)

    def __rmul__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(other, str):
            raise TypeError('multiplication can not be applied to a string literal.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('multiplication can not be applied to date times.')
        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, int):
                return column_op(SF.repeat)(self, other)
            else:
                raise TypeError('a string series can only be multiplied to an int series or literal')
        return column_op(Column.__rmul__)(self, other)

    def __rtruediv__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError('division can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def rtruediv(left: Column, right: Any) -> Column:
            return F.when(left == 0, F.lit(np.inf).__div__(right))\
                    .otherwise(F.lit(right).__truediv__(left))
        return numpy_column_op(rtruediv)(self, other)

    def __floordiv__(self, other: Any) -> "IndexOpsMixin":
        """
        __floordiv__ has different behaviour between pandas and PySpark for several cases.
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
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('division can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def floordiv(left: Column, right: Any) -> Column:
            return F.when(F.lit(right is np.nan), np.nan)\
                .otherwise(F.when(F.lit(right != 0) | F.lit(right).isNull(), F.floor(left.__div__(right)))\
                .otherwise(F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left)\
                .otherwise(F.lit(np.inf).__div__(left))))
        return numpy_column_op(floordiv)(self, other)

    def __rfloordiv__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError('division can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def rfloordiv(left: Column, right: Any) -> Column:
            return F.when(F.lit(left == 0), F.lit(np.inf).__div__(right))\
                    .otherwise(F.when(F.lit(left) == np.nan, np.nan).otherwise(F.floor(F.lit(right).__div__(left))))
        return numpy_column_op(rfloordiv)(self, other)

    def __rmod__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError('modulo can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('modulo can not be applied to date times.')

        def rmod(left: Column, right: Any) -> Column:
            return (right % left + left) % left
        return column_op(rmod)(self, other)

    def __pow__(self, other: Any) -> "IndexOpsMixin":
        def pow_func(left: Column, right: Any) -> Column:
            return F.when(left == 1, left).otherwise(Column.__pow__(left, right))
        return column_op(pow_func)(self, other)

    def __rpow__(self, other: Any) -> "IndexOpsMixin":
        def rpow_func(left: Column, right: Any) -> Column:
            return F.when(F.lit(right == 1), right).otherwise(Column.__rpow__(left, right))
        return column_op(rpow_func)(self, other)

    __abs__ = column_op(F.abs)
    __eq__ = column_op(Column.__eq__)
    __ne__ = column_op(Column.__ne__)
    __lt__ = column_op(Column.__lt__)
    __le__ = column_op(Column.__le__)
    __ge__ = column_op(Column.__ge__)
    __gt__ = column_op(Column.__gt__)

    def __and__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.dtype, extension_dtypes) or (isinstance(other, IndexOpsMixin) and isinstance(other.dtype, extension_dtypes)):

            def and_func(left: Column, right: Any) -> Column:
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                return left & right
        else:

            def and_func(left: Column, right: Any) -> Column:
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                scol = left & right
                return F.when(scol.isNull(), False).otherwise(scol)
        return column_op(and_func)(self, other)

    def __or__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.dtype, extension_dtypes) or (isinstance(other, IndexOpsMixin) and isinstance(other.dtype, extension_dtypes)):

            def or_func(left: Column, right: Any) -> Column:
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                return left | right
        else:

            def or_func(left: Column, right: Any) -> Column:
                if not isinstance(right, spark.Column) and pd.isna(right):
                    return F.lit(False)
                else:
                    scol = left | F.lit(right)
                    return F.when(left.isNull() | scol.isNull(), False).otherwise(scol)
        return column_op(or_func)(self, other)
    __invert__ = column_op(Column.__invert__)

    def __rand__(self, other: Any) -> "IndexOpsMixin":
        return self.__and__(other)

    def __ror__(self, other: Any) -> "IndexOpsMixin":
        return self.__or__(other)

    def __len__(self) -> int:
        return len(self._kdf)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        result = numpy_compat.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is NotImplemented:
            result = numpy_compat.maybe_dispatch_ufunc_to_spark_func(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        else:
            raise NotImplementedError('Koalas objects currently do not support %s.' % ufunc)

    @property
    def dtype(self) -> Any:
        """Return the dtype object of the underlying data.
        """
        return self._internal.data_dtypes[0]

    @property
    def empty(self) -> bool:
        """
        Returns true if the current object is empty. Otherwise, returns false.
        """
        return self._internal.resolved_copy.spark_frame.rdd.isEmpty()

    @property
    def hasnans(self) -> bool:
        """
        Return True if it has any missing values. Otherwise, it returns False.
        """
        sdf = self._internal.spark_frame
        scol = self.spark.column
        if isinstance(self.spark.data_type, (DoubleType, FloatType)):
            return sdf.select(F.max(scol.isNull() | F.isnan(scol))).collect()[0][0]
        else:
            return sdf.select(F.max(scol.isNull())).collect()[0][0]

    @property
    def is_monotonic(self) -> bool:
        """
        Return boolean if values in the object are monotonically increasing.
        """
        return self._is_monotonic('increasing')
    is_monotonic_increasing = is_monotonic

    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return boolean if values in the object are monotonically decreasing.
        """
        return self._is_monotonic('decreasing')

    def _is_locally_monotonic_spark_column(self, order: str) -> Column:
        window = Window.partitionBy(F.col('__partition_id')).orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)
        if order == 'increasing':
            return (F.col('__origin') >= F.lag(F.col('__origin'), 1).over(window)) & F.col('__origin').isNotNull()
        else:
            return (F.col('__origin') <= F.lag(F.col('__origin'), 1).over(window)) & F.col('__origin').isNotNull()

    def _is_monotonic(self, order: str) -> bool:
        assert order in ('increasing', 'decreasing')
        sdf = self._internal.spark_frame
        sdf = sdf.select(F.spark_partition_id().alias('__partition_id'),
                         F.col(NATURAL_ORDER_COLUMN_NAME),
                         self.spark.column.alias('__origin')).select(
                             F.col('__partition_id'),
                             F.col('__origin'),
                             self._is_locally_monotonic_spark_column(order).alias('__comparison_within_partition')
                         ).groupby(F.col('__partition_id')).agg(
                             F.min(F.col('__origin')).alias('__partition_min'),
                             F.max(F.col('__origin')).alias('__partition_max'),
                             F.min(F.coalesce(F.col('__comparison_within_partition'), F.lit(True))).alias('__comparison_within_partition')
                         )
        window = Window.orderBy(F.col('__partition_id')).rowsBetween(-1, -1)
        if order == 'increasing':
            comparison_col = F.col('__partition_min') >= F.lag(F.col('__partition_max'), 1).over(window)
        else:
            comparison_col = F.col('__partition_min') <= F.lag(F.col('__partition_max'), 1).over(window)
        sdf = sdf.select(comparison_col.alias('__comparison_between_partitions'),
                         F.col('__comparison_within_partition'))
        ret = sdf.select(F.min(F.coalesce(F.col('__comparison_between_partitions'), F.lit(True))) & F.min(F.coalesce(F.col('__comparison_within_partition'), F.lit(True)))).collect()[0][0]
        if ret is None:
            return True
        else:
            return ret

    @property
    def ndim(self) -> int:
        """
        Return an int representing the number of array dimensions.
        """
        return 1

    def astype(self, dtype: Any) -> "IndexOpsMixin":
        """
        Cast a Koalas object to a specified dtype ``dtype``.
        """
        dtype, spark_type = koalas_dtype(dtype)
        if not spark_type:
            raise ValueError('Type {} not understood'.format(dtype))
        if isinstance(self.dtype, CategoricalDtype):
            if isinstance(dtype, CategoricalDtype) and dtype.categories is None:
                return cast(Union["IndexOpsMixin", "Series"], self).copy()
            categories = self.dtype.categories
            if len(categories) == 0:
                scol = F.lit(None)
            else:
                kvs = list(chain(*[(F.lit(code), F.lit(category)) for code, category in enumerate(categories)]))
                map_scol = F.create_map(kvs)
                scol = map_scol.getItem(self.spark.column)
            return self._with_new_scol(scol.alias(self._internal.data_spark_column_names[0])).astype(dtype)
        elif isinstance(dtype, CategoricalDtype):
            if dtype.categories is None:
                codes, uniques = self.factorize()
                return codes._with_new_scol(codes.spark.column, dtype=CategoricalDtype(categories=uniques))
            else:
                categories = dtype.categories
                if len(categories) == 0:
                    scol = F.lit(-1)
                else:
                    kvs = list(chain(*[(F.lit(category), F.lit(code)) for code, category in enumerate(categories)]))
                    map_scol = F.create_map(kvs)
                    scol = F.coalesce(map_scol.getItem(self.spark.column), F.lit(-1))
                return self._with_new_scol(scol.alias(self._internal.data_spark_column_names[0]), dtype=dtype)
        if isinstance(spark_type, BooleanType):
            if isinstance(dtype, extension_dtypes):
                scol = self.spark.column.cast(spark_type)
            elif isinstance(self.spark.data_type, StringType):
                scol = F.when(self.spark.column.isNull(), F.lit(False)).otherwise(F.length(self.spark.column) > 0)
            elif isinstance(self.spark.data_type, (FloatType, DoubleType)):
                scol = F.when(self.spark.column.isNull() | F.isnan(self.spark.column), F.lit(True)).otherwise(self.spark.column.cast(spark_type))
            else:
                scol = F.when(self.spark.column.isNull(), F.lit(False)).otherwise(self.spark.column.cast(spark_type))
        elif isinstance(spark_type, StringType):
            if isinstance(dtype, extension_dtypes):
                if isinstance(self.spark.data_type, BooleanType):
                    scol = F.when(self.spark.column.isNotNull(), F.when(self.spark.column, 'True').otherwise('False'))
                elif isinstance(self.spark.data_type, TimestampType):
                    scol = F.when(self.spark.column.isNull(), str(pd.NaT)).otherwise(self.spark.column.cast(spark_type))
                else:
                    scol = self.spark.column.cast(spark_type)
            else:
                if isinstance(self.spark.data_type, NumericType):
                    null_str = str(np.nan)
                elif isinstance(self.spark.data_type, (DateType, TimestampType)):
                    null_str = str(pd.NaT)
                else:
                    null_str = str(None)
                if isinstance(self.spark.data_type, BooleanType):
                    casted = F.when(self.spark.column, 'True').otherwise('False')
                else:
                    casted = self.spark.column.cast(spark_type)
                scol = F.when(self.spark.column.isNull(), null_str).otherwise(casted)
        else:
            scol = self.spark.column.cast(spark_type)
        return self._with_new_scol(scol.alias(self._internal.data_spark_column_names[0]), dtype=dtype)

    def isin(self, values: Any) -> "IndexOpsMixin":
        """
        Check whether `values` are contained in Series or Index.
        """
        if not is_list_like(values):
            raise TypeError('only list-like objects are allowed to be passed to isin(), you passed a [{values_type}]'.format(values_type=type(values).__name__))
        values = values.tolist() if isinstance(values, np.ndarray) else list(values)
        return self._with_new_scol(self.spark.column.isin(values))

    def isnull(self) -> "IndexOpsMixin":
        """
        Detect existing (non-missing) values.
        """
        from databricks.koalas.indexes import MultiIndex
        if isinstance(self, MultiIndex):
            raise NotImplementedError('isna is not defined for MultiIndex')
        if isinstance(self.spark.data_type, (FloatType, DoubleType)):
            return self._with_new_scol(self.spark.column.isNull() | F.isnan(self.spark.column))
        else:
            return self._with_new_scol(self.spark.column.isNull())
    isna = isnull

    def notnull(self) -> "IndexOpsMixin":
        """
        Detect existing (non-missing) values.
        """
        from databricks.koalas.indexes import MultiIndex
        if isinstance(self, MultiIndex):
            raise NotImplementedError('notna is not defined for MultiIndex')
        return (~self.isnull()).rename(self.name)
    notna = notnull

    def all(self, axis: Union[int, str] = 0) -> bool:
        """
        Return whether all elements are True.
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        sdf = self._internal.spark_frame.select(self.spark.column)
        col = scol_for(sdf, sdf.columns[0])
        ret = sdf.select(F.min(F.coalesce(col.cast('boolean'), F.lit(True)))).collect()[0][0]
        if ret is None:
            return True
        else:
            return ret

    def any(self, axis: Union[int, str] = 0) -> bool:
        """
        Return whether any element is True.
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        sdf = self._internal.spark_frame.select(self.spark.column)
        col = scol_for(sdf, sdf.columns[0])
        ret = sdf.select(F.max(F.coalesce(col.cast('boolean'), F.lit(False)))).collect()[0][0]
        if ret is None:
            return False
        else:
            return ret

    def shift(self, periods: int = 1, fill_value: Any = None) -> Any:
        """
        Shift Series/Index by desired number of periods.
        """
        return self._shift(periods, fill_value).spark.analyzed

    def _shift(self, periods: int, fill_value: Any, *, part_cols: Tuple[Any, ...] = ()) -> "IndexOpsMixin":
        if not isinstance(periods, int):
            raise ValueError('periods should be an int; however, got [%s]' % type(periods).__name__)
        col = self.spark.column
        window = Window.partitionBy(*part_cols).orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        lag_col = F.lag(col, periods).over(window)
        col = F.when(lag_col.isNull() | F.isnan(lag_col), fill_value).otherwise(lag_col)
        return self._with_new_scol(col, dtype=self.dtype)

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, bins: Optional[Any] = None, dropna: bool = True) -> "Series":
        """
        Return a Series containing counts of unique values.
        """
        from databricks.koalas.series import first_series
        if bins is not None:
            raise NotImplementedError('value_counts currently does not support bins')
        if dropna:
            sdf_dropna = self._internal.spark_frame.select(self.spark.column).dropna()
        else:
            sdf_dropna = self._internal.spark_frame.select(self.spark.column)
        index_name = SPARK_DEFAULT_INDEX_NAME
        column_name = self._internal.data_spark_column_names[0]
        sdf = sdf_dropna.groupby(scol_for(sdf_dropna, column_name).alias(index_name)).count()
        if sort:
            if ascending:
                sdf = sdf.orderBy(F.col('count'))
            else:
                sdf = sdf.orderBy(F.col('count').desc())
        if normalize:
            sum_val = sdf_dropna.count()
            sdf = sdf.withColumn('count', F.col('count') / F.lit(sum_val))
        internal = InternalFrame(spark_frame=sdf,
                                 index_spark_columns=[scol_for(sdf, index_name)],
                                 column_labels=self._internal.column_labels,
                                 data_spark_columns=[scol_for(sdf, 'count')],
                                 column_label_names=self._internal.column_label_names)
        return first_series(DataFrame(internal))

    def nunique(self, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> int:
        """
        Return number of unique elements in the object.
        """
        res = self._internal.spark_frame.select([self._nunique(dropna, approx, rsd)])
        return res.collect()[0][0]

    def _nunique(self, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> Column:
        colname = self._internal.data_spark_column_names[0]
        count_fn: Callable = partial(F.approx_count_distinct, rsd=rsd) if approx else F.countDistinct
        if dropna:
            return count_fn(self.spark.column).alias(colname)
        else:
            return (count_fn(self.spark.column) + F.when(F.count(F.when(self.spark.column.isNull(), 1).otherwise(None)) >= 1, 1).otherwise(0)).alias(colname)

    def take(self, indices: Any) -> Union["Series", Any]:
        """
        Return the elements in the given *positional* indices along an axis.
        """
        if not is_list_like(indices) or isinstance(indices, (dict, set)):
            raise ValueError('`indices` must be a list-like except dict or set')
        if isinstance(self, ks.Series):
            return cast(ks.Series, self.iloc[indices])
        else:
            return self._kdf.iloc[indices].index

    def factorize(self, sort: bool = True, na_sentinel: Optional[int] = -1) -> Tuple["IndexOpsMixin", pd.Index]:
        """
        Encode the object as an enumerated type or categorical variable.
        """
        from databricks.koalas.series import first_series
        assert na_sentinel is None or isinstance(na_sentinel, int)
        assert sort is True
        if isinstance(self.dtype, CategoricalDtype):
            categories = self.dtype.categories
            if len(categories) == 0:
                scol = F.lit(None)
            else:
                kvs = list(chain(*[(F.lit(code), F.lit(category)) for code, category in enumerate(categories)]))
                map_scol = F.create_map(kvs)
                scol = map_scol.getItem(self.spark.column)
            codes, uniques = self._with_new_scol(scol.alias(self._internal.data_spark_column_names[0])).factorize(na_sentinel=na_sentinel)
            return (codes, uniques.astype(self.dtype))
        uniq_sdf = self._internal.spark_frame.select(self.spark.column).distinct()
        max_compute_count = get_option('compute.max_rows')
        if max_compute_count is not None:
            uniq_pdf = uniq_sdf.limit(max_compute_count + 1).toPandas()
            if len(uniq_pdf) > max_compute_count:
                raise ValueError("Current Series has more then {0} unique values. Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option' to more than {0} rows. Note that, before changing the 'compute.max_rows', this operation is considerably expensive.".format(max_compute_count))
        else:
            uniq_pdf = uniq_sdf.toPandas()
        uniq_series = first_series(uniq_pdf).drop_duplicates()
        uniques_list = uniq_series.tolist()
        uniques_list = sorted(uniques_list, key=lambda x: (pd.isna(x), x))
        unique_to_code = {}
        if na_sentinel is not None:
            na_sentinel_code = na_sentinel
        code = 0
        for unique in uniques_list:
            if pd.isna(unique):
                if na_sentinel is None:
                    na_sentinel_code = code
            else:
                unique_to_code[unique] = code
            code += 1
        kvs = list(chain(*[(F.lit(unique), F.lit(code)) for unique, code in unique_to_code.items()]))
        if len(kvs) == 0:
            new_scol = F.lit(na_sentinel_code)
        else:
            scol = self.spark.column
            if isinstance(self.spark.data_type, (FloatType, DoubleType)):
                cond = scol.isNull() | F.isnan(scol)
            else:
                cond = scol.isNull()
            map_scol = F.create_map(kvs)
            null_scol = F.when(cond, F.lit(na_sentinel_code))
            new_scol = null_scol.otherwise(map_scol.getItem(scol))
        codes = self._with_new_scol(new_scol.alias(self._internal.data_spark_column_names[0]))
        if na_sentinel is not None:
            uniques_list = [x for x in uniques_list if not pd.isna(x)]
        uniques = pd.Index(uniques_list)
        return (codes, uniques)
