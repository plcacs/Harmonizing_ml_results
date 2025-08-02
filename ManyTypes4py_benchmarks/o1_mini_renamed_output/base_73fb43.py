"""
Base and utility classes for Koalas objects.
"""
from abc import ABCMeta, abstractmethod
import datetime
from functools import wraps, partial
from itertools import chain
from typing import Any, Callable, Optional, Tuple, Union, cast, TYPE_CHECKING, List

import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like, CategoricalDtype
from pyspark import sql as spark
from pyspark.sql import functions as F, Window, Column
from pyspark.sql.types import (
    BooleanType, DateType, DoubleType, FloatType, IntegralType, LongType,
    NumericType, StringType, TimestampType
)
from databricks import koalas as ks
from databricks.koalas import numpy_compat
from databricks.koalas.config import get_option, option_context
from databricks.koalas.internal import InternalFrame, NATURAL_ORDER_COLUMN_NAME, SPARK_DEFAULT_INDEX_NAME
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkIndexOpsMethods
from databricks.koalas.typedef import (
    Dtype, as_spark_type, extension_dtypes, koalas_dtype, spark_type_to_pandas_dtype
)
from databricks.koalas.utils import combine_frames, same_anchor, scol_for, validate_axis, ERROR_MESSAGE_CANNOT_COMBINE
from databricks.koalas.frame import DataFrame
if TYPE_CHECKING:
    from databricks.koalas.indexes import Index
    from databricks.koalas.series import Series


def func_0zcp3wpt(self: "IndexOpsMixin", other: "IndexOpsMixin") -> bool:
    from databricks.koalas.series import Series
    if isinstance(self, Series) and isinstance(other, Series):
        return not same_anchor(self, other)
    else:
        return self._internal.spark_frame is not other._internal.spark_frame


def func_pbq1xee0(func: Callable, this_index_ops: "IndexOpsMixin", *args: Any) -> Union["Index", "Series"]:
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
    cols: List[IndexOpsMixin] = [arg for arg in args if isinstance(arg, IndexOpsMixin)]
    if isinstance(this_index_ops, Series) and all(isinstance(col, Series) for col in cols):
        combined = combine_frames(
            this_index_ops.to_frame(),
            *[cast(Series, col).rename(i) for i, col in enumerate(cols)],
            how='full'
        )
        return column_op(func)(
            combined['this']._kser_for(combined['this']._internal.column_labels[0]),
            *[combined['that']._kser_for(label) for label in combined['that']._internal.column_labels]
        ).rename(this_index_ops.name)
    else:
        self_len = len(this_index_ops)
        if any(len(col) != self_len for col in args if isinstance(col, IndexOpsMixin)):
            raise ValueError('operands could not be broadcast together with shapes')
        with option_context('compute.default_index_type', 'distributed-sequence'):
            if isinstance(this_index_ops, Index) and all(isinstance(col, Index) for col in cols):
                return Index(
                    column_op(func)(
                        this_index_ops.to_series().reset_index(drop=True),
                        *[(arg.to_series().reset_index(drop=True) if isinstance(arg, Index) else arg) for arg in args]
                    ).sort_index(),
                    name=this_index_ops.name
                )
            elif isinstance(this_index_ops, Series):
                this = this_index_ops.reset_index()
                that: List[Series] = [
                    cast(Series, col.to_series() if isinstance(col, Index) else col).rename(i).reset_index(drop=True)
                    for i, col in enumerate(cols)
                ]
                combined = combine_frames(this, *that, how='full').sort_index()
                combined = combined.set_index(combined._internal.column_labels[:this_index_ops._internal.index_level])
                combined.index.names = this_index_ops._internal.index_names
                return column_op(func)(
                    first_series(combined['this']),
                    *[combined['that']._kser_for(label) for label in combined['that']._internal.column_labels]
                ).rename(this_index_ops.name)
            else:
                this = cast(Index, this_index_ops).to_frame().reset_index(drop=True)
                that_series = next(col for col in cols if isinstance(col, Series))
                that_frame = that_series._kdf[
                    [cast(Series, col.to_series() if isinstance(col, Index) else col).rename(i) for i, col in enumerate(cols)]
                ]
                combined = combine_frames(this, that_frame.reset_index()).sort_index()
                self_index = combined['this'].set_index(combined['this']._internal.column_labels).index
                other = combined['that'].set_index(combined['that']._internal.column_labels[:that_series._internal.index_level])
                other.index.names = that_series._internal.index_names
                return column_op(func)(
                    self_index,
                    *[other._kser_for(label) for label, col in zip(other._internal.column_labels, cols)]
                ).rename(that_series.name)


def func_m12gtncb(scol: Column, f: Callable) -> Column:
    """
    Booleanize Null in Spark Column
    """
    comp_ops = [getattr(Column, f'__{comp_op}__') for comp_op in ['eq', 'ne', 'lt', 'le', 'ge', 'gt']]
    if f in comp_ops:
        filler = f == Column.__ne__
        scol = F.when(scol.isNull(), filler).otherwise(scol)
    return scol


def func_rx7wn3d2(f: Callable) -> Callable[..., "IndexOpsMixin"]:
    """
    A decorator that wraps APIs taking/returning Spark Column so that Koalas Series can be
    supported too. If this decorator is used for the `f` function that takes Spark Column and
    returns Spark Column, decorated `f` takes Koalas Series as well and returns Koalas
    Series.

    :param f: a function that takes Spark Column and returns Spark Column.
    """

    @wraps(f)
    def func_dwhuw3j4(self: "IndexOpsMixin", *args: Any) -> "IndexOpsMixin":
        from databricks.koalas.series import Series
        cols: List[IndexOpsMixin] = [arg for arg in args if isinstance(arg, IndexOpsMixin)]
        if all(not func_0zcp3wpt(self, col) for col in cols):
            args_converted = [(arg.spark.column if isinstance(arg, IndexOpsMixin) else arg) for arg in args]
            scol = f(self.spark.column, *args_converted)
            spark_type = self._internal.spark_frame.select(scol).schema[0].dataType
            use_extension_dtypes = any(isinstance(col.dtype, extension_dtypes) for col in [self] + cols)
            dtype = spark_type_to_pandas_dtype(spark_type, use_extension_dtypes=use_extension_dtypes)
            if not isinstance(dtype, extension_dtypes):
                scol = func_m12gtncb(scol, f)
            if isinstance(self, Series) or not any(isinstance(col, Series) for col in cols):
                index_ops = self._with_new_scol(scol, dtype=dtype)
            else:
                kser = next(col for col in cols if isinstance(col, Series))
                index_ops = kser._with_new_scol(scol, dtype=dtype)
        elif get_option('compute.ops_on_diff_frames'):
            index_ops = func_pbq1xee0(f, self, *args)
        else:
            raise ValueError(ERROR_MESSAGE_CANNOT_COMBINE)
        if not all(self.name == col.name for col in cols):
            index_ops = index_ops.rename(None)
        return index_ops

    return func_dwhuw3j4


def func_lqxh05t5(f: Callable) -> Callable[..., "IndexOpsMixin"]:
    
    @wraps(f)
    def func_dwhuw3j4(self: "IndexOpsMixin", *args: Any) -> "IndexOpsMixin":
        new_args: List[Any] = []
        for arg in args:
            if isinstance(self.spark.data_type, LongType) and isinstance(arg, np.timedelta64):
                new_args.append(float(arg / np.timedelta64(1, 's')))
            else:
                new_args.append(arg)
        return func_rx7wn3d2(f)(self, *new_args)
    return func_dwhuw3j4


class IndexOpsMixin(metaclass=ABCMeta):
    """common ops mixin to support a unified interface / docs for Series / Index

    Assuming there are following attributes or properties and function.
    """

    @property
    @abstractmethod
    def func_5n6ob5lk(self) -> Any:
        pass

    @property
    @abstractmethod
    def func_m5q4ias8(self) -> Any:
        pass

    @abstractmethod
    def func_92yjgrf6(self, scol: Column, *, dtype: Optional[Dtype] = None) -> Any:
        pass

    @property
    @abstractmethod
    def func_i1wdumc8(self) -> Any:
        pass

    @property
    @abstractmethod
    def func_av9jnpqm(self) -> Any:
        pass

    @property
    def func_fa9tn4qm(self) -> Column:
        warnings.warn(
            'Series.spark_column is deprecated as of Series.spark.column. Please use the API instead.',
            FutureWarning
        )
        return self.spark.column
    spark_column.__doc__ = SparkIndexOpsMethods.column.__doc__
    __neg__ = func_rx7wn3d2(Column.__neg__)

    def __add__(self, other: Any) -> "IndexOpsMixin":
        if not isinstance(self.spark.data_type, StringType) and (
            (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str)
        ):
            raise TypeError(
                'string addition can only be applied to string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('addition can not be applied to date times.')
        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType):
                return func_rx7wn3d2(F.concat)(self, other)
            elif isinstance(other, str):
                return func_rx7wn3d2(F.concat)(self, F.lit(other))
            else:
                raise TypeError(
                    'string addition can only be applied to string series or literals.'
                )
        else:
            return func_rx7wn3d2(Column.__add__)(self, other)

    def __sub__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or (
            (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError(
                'substraction can not be applied to string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            msg = (
                "Note that there is a behavior difference of timestamp subtraction. The timestamp subtraction returns an integer in seconds, whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, TimestampType):
                warnings.warn(msg, UserWarning)
                return self.astype('long') - other.astype('long')
            elif isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return self.astype('long') - F.lit(other).cast(as_spark_type('long'))
            else:
                raise TypeError(
                    'datetime subtraction can only be applied to datetime series.'
                )
        elif isinstance(self.spark.data_type, DateType):
            msg = (
                "Note that there is a behavior difference of date subtraction. The date subtraction returns an integer in days, whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, DateType):
                warnings.warn(msg, UserWarning)
                return func_rx7wn3d2(F.datediff)(self, other).astype('long')
            elif isinstance(other, datetime.date) and not isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return func_rx7wn3d2(F.datediff)(self, F.lit(other)).astype('long')
            else:
                raise TypeError(
                    'date subtraction can only be applied to date series.'
                )
        return func_rx7wn3d2(Column.__sub__)(self, other)

    def __mul__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(other, str):
            raise TypeError(
                'multiplication can not be applied to a string literal.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('multiplication can not be applied to date times.')
        if isinstance(self.spark.data_type, IntegralType) and (
            isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)
            or isinstance(other, int)
        ):
            return func_rx7wn3d2(SF.repeat)(other, self)
        if isinstance(self.spark.data_type, StringType):
            if (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, IntegralType)) or isinstance(other, int):
                return func_rx7wn3d2(SF.repeat)(self, other)
            else:
                raise TypeError(
                    'a string series can only be multiplied to an int series or literal'
                )
        return func_rx7wn3d2(Column.__mul__)(self, other)

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
        if isinstance(self.spark.data_type, StringType) or (
            (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError(
                'division can not be applied on string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def func_bweyflu8(left: Any, right: Any) -> Column:
            return F.when(F.lit(right != 0) | F.lit(right).isNull(), left.__div__(right))\
                .otherwise(F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left)
                .otherwise(F.lit(np.inf).__div__(left)))

        return func_lqxh05t5(truediv)(self, other)

    def __mod__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or (
            (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError(
                'modulo can not be applied on string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('modulo can not be applied to date times.')

        def func_ybwyt5gd(left: Any, right: Any) -> Column:
            return (left % right + right) % right

        return func_rx7wn3d2(mod)(self, other)

    def __radd__(self, other: Any) -> "IndexOpsMixin":
        if not isinstance(self.spark.data_type, StringType) and isinstance(other, str):
            raise TypeError(
                'string addition can only be applied to string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('addition can not be applied to date times.')
        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, str):
                return self._with_new_scol(F.concat(F.lit(other), self.spark.column))
            else:
                raise TypeError(
                    'string addition can only be applied to string series or literals.'
                )
        else:
            return func_rx7wn3d2(Column.__radd__)(self, other)

    def __rsub__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError(
                'substraction can not be applied to string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            msg = (
                "Note that there is a behavior difference of timestamp subtraction. The timestamp subtraction returns an integer in seconds, whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return -(self.astype('long') - F.lit(other).cast(as_spark_type('long')))
            else:
                raise TypeError(
                    'datetime subtraction can only be applied to datetime series.'
                )
        elif isinstance(self.spark.data_type, DateType):
            msg = (
                "Note that there is a behavior difference of date subtraction. The date subtraction returns an integer in days, whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, datetime.date) and not isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return -func_rx7wn3d2(F.datediff)(self, F.lit(other)).astype('long')
            else:
                raise TypeError(
                    'date subtraction can only be applied to date series.'
                )
        return func_rx7wn3d2(Column.__rsub__)(self, other)

    def __rmul__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(other, str):
            raise TypeError(
                'multiplication can not be applied to a string literal.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('multiplication can not be applied to date times.')
        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, int):
                return func_rx7wn3d2(SF.repeat)(self, other)
            else:
                raise TypeError(
                    'a string series can only be multiplied to an int series or literal'
                )
        return func_rx7wn3d2(Column.__rmul__)(self, other)

    def __rtruediv__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError(
                'division can not be applied on string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def func_bl45woce(left: Any, right: Any) -> Column:
            return F.when(left == 0, F.lit(np.inf).__div__(right))\
                .otherwise(F.lit(right).__truediv__(left))

        return func_lqxh05t5(rtruediv)(self, other)

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
        if isinstance(self.spark.data_type, StringType) or (
            (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError(
                'division can not be applied on string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def func_v329drfv(left: Any, right: Any) -> Column:
            return F.when(F.lit(right is np.nan), np.nan)\
                .otherwise(
                    F.when(F.lit(right != 0) | F.lit(right).isNull(), F.floor(left.__div__(right)))
                    .otherwise(
                        F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left)
                        .otherwise(F.lit(np.inf).__div__(left))
                    )
                )

        return func_lqxh05t5(floordiv)(self, other)

    def __rfloordiv__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError(
                'division can not be applied on string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def func_fej9sg0q(left: Any, right: Any) -> Column:
            return F.when(F.lit(left == 0), F.lit(np.inf).__div__(right))\
                .otherwise(
                    F.when(F.lit(left) == np.nan, np.nan)
                    .otherwise(F.floor(F.lit(right).__div__(left)))
                )

        return func_lqxh05t5(rfloordiv)(self, other)

    def __rmod__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError(
                'modulo can not be applied on string series or literals.'
            )
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('modulo can not be applied to date times.')

        def func_f11s20pm(left: Any, right: Any) -> Column:
            return (right % left + left) % left

        return func_rx7wn3d2(rmod)(self, other)

    def __pow__(self, other: Any) -> "IndexOpsMixin":

        def pow_func(left: Any, right: Any) -> Column:
            return F.when(left == 1, left).__pow__(right)

        return func_rx7wn3d2(pow_func)(self, other)

    def __rpow__(self, other: Any) -> "IndexOpsMixin":

        def rpow_func(left: Any, right: Any) -> Column:
            return F.when(F.lit(right == 1), right).__pow__(left)

        return func_rx7wn3d2(rpow_func)(self, other)

    __abs__ = func_rx7wn3d2(F.abs)
    __eq__ = func_rx7wn3d2(Column.__eq__)
    __ne__ = func_rx7wn3d2(Column.__ne__)
    __lt__ = func_rx7wn3d2(Column.__lt__)
    __le__ = func_rx7wn3d2(Column.__le__)
    __ge__ = func_rx7wn3d2(Column.__ge__)
    __gt__ = func_rx7wn3d2(Column.__gt__)

    def __and__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.dtype, extension_dtypes) or (
            isinstance(other, IndexOpsMixin) and isinstance(other.dtype, extension_dtypes)
        ):

            def func_n44rof1h(left: Any, right: Any) -> Column:
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                return left & right
        else:

            def func_n44rof1h(left: Any, right: Any) -> Column:
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                scol = left & right
                return F.when(scol.isNull(), False).otherwise(scol)
        return func_rx7wn3d2(and_func)(self, other)

    def __or__(self, other: Any) -> "IndexOpsMixin":
        if isinstance(self.dtype, extension_dtypes) or (
            isinstance(other, IndexOpsMixin) and isinstance(other.dtype, extension_dtypes)
        ):

            def func_r9o9xbmi(left: Any, right: Any) -> Column:
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                return left | right
        else:

            def func_r9o9xbmi(left: Any, right: Any) -> Column:
                if not isinstance(right, spark.Column) and pd.isna(right):
                    return F.lit(False)
                else:
                    scol = left | F.lit(right)
                    return F.when(left.isNull() | scol.isNull(), False).otherwise(scol)
        return func_rx7wn3d2(or_func)(self, other)
    __invert__ = func_rx7wn3d2(Column.__invert__)

    def __rand__(self, other: Any) -> "IndexOpsMixin":
        return self.__and__(other)

    def __ror__(self, other: Any) -> "IndexOpsMixin":
        return self.__or__(other)

    def __len__(self) -> int:
        return len(self._kdf)

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any) -> Any:
        result = numpy_compat.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is NotImplemented:
            result = numpy_compat.maybe_dispatch_ufunc_to_spark_func(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        else:
            raise NotImplementedError(
                f'Koalas objects currently do not support {ufunc}.'
            )

    @property
    def func_3yo3g8jn(self) -> Dtype:
        """Return the dtype object of the underlying data.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3])
        >>> s.dtype
        dtype('int64')

        >>> s = ks.Series(list('abc'))
        >>> s.dtype
        dtype('O')

        >>> s = ks.Series(pd.date_range('20130101', periods=3))
        >>> s.dtype
        dtype('<M8[ns]')

        >>> s.rename("a").to_frame().set_index("a").index.dtype
        dtype('<M8[ns]')
        """
        return self._internal.data_dtypes[0]

    @property
    def func_45s06kt1(self) -> bool:
        """
        Returns true if the current object is empty. Otherwise, returns false.

        >>> ks.range(10).id.empty
        False

        >>> ks.range(0).id.empty
        True

        >>> ks.DataFrame({}, index=list('abc')).index.empty
        False
        """
        return self._internal.resolved_copy.spark_frame.rdd.isEmpty()

    @property
    def func_6zq36i5i(self) -> bool:
        """
        Return True if it has any missing values. Otherwise, it returns False.

        >>> ks.DataFrame({}, index=list('abc')).index.hasnans
        False

        >>> ks.Series(['a', None]).hasnans
        True

        >>> ks.Series([1.0, 2.0, np.nan]).hasnans
        True

        >>> ks.Series([1, 2, 3]).hasnans
        False

        >>> (ks.Series([1.0, 2.0, np.nan]) + 1).hasnans
        True

        >>> ks.Series([1, 2, 3]).rename("a").to_frame().set_index("a").index.hasnans
        False
        """
        sdf = self._internal.spark_frame
        scol = self.spark.column
        if isinstance(self.spark.data_type, (DoubleType, FloatType)):
            return sdf.select(F.max(scol.isNull() | F.isnan(scol))).collect()[0][0]
        else:
            return sdf.select(F.max(scol.isNull())).collect()[0][0]

    @property
    def func_cndni9z8(self) -> bool:
        """
        Return boolean if values in the object are monotonically increasing.

        .. note:: the current implementation of is_monotonic requires to shuffle
            and aggregate multiple times to check the order locally and globally,
            which is potentially expensive. In case of multi-index, all data are
            transferred to single node which can easily cause out-of-memory error currently.

        .. note:: Disable the Spark config `spark.sql.optimizer.nestedSchemaPruning.enabled`
            for multi-index if you're using Koalas < 1.7.0 with PySpark 3.1.1.

        Returns
        -------
        is_monotonic : bool

        Examples
        --------
        >>> ser = ks.Series(['1/1/2018', '3/1/2018', '4/1/2018'])
        >>> ser.is_monotonic
        True

        >>> df = ks.DataFrame({'dates': [None, '1/1/2018', '2/1/2018', '3/1/2018']})
        >>> df.dates.is_monotonic
        False

        >>> df.index.is_monotonic
        True

        >>> ser = ks.Series([1])
        >>> ser.is_monotonic
        True

        >>> ser = ks.Series([])
        >>> ser.is_monotonic
        True

        >>> ser.rename("a").to_frame().set_index("a").index.is_monotonic
        True

        >>> ser = ks.Series([5, 4, 3, 2, 1], index=[1, 2, 3, 4, 5])
        >>> ser.is_monotonic
        False

        >>> ser.index.is_monotonic
        True

        Support for MultiIndex

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('x', 'a'), ('x', 'b'), ('y', 'c'), ('y', 'd'), ('z', 'e')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('z', 'e')],
                   )
        >>> midx.is_monotonic
        True

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('z', 'a'), ('z', 'b'), ('y', 'c'), ('y', 'd'), ('x', 'e')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('z', 'a'),
                    ('z', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('x', 'e')],
                   )
        >>> midx.is_monotonic
        False
        """
        return self._is_monotonic('increasing')
    is_monotonic_increasing = is_monotonic

    @property
    def func_0vh80e77(self) -> bool:
        """
        Return boolean if values in the object are monotonically decreasing.

        .. note:: the current implementation of is_monotonic_decreasing requires to shuffle
            and aggregate multiple times to check the order locally and globally,
            which is potentially expensive. In case of multi-index, all data are transferred
            to single node which can easily cause out-of-memory error currently.

        .. note:: Disable the Spark config `spark.sql.optimizer.nestedSchemaPruning.enabled`
            for multi-index if you're using Koalas < 1.7.0 with PySpark 3.1.1.

        Returns
        -------
        is_monotonic : bool

        Examples
        --------
        >>> ser = ks.Series(['4/1/2018', '3/1/2018', '1/1/2018'])
        >>> ser.is_monotonic_decreasing
        True

        >>> df = ks.DataFrame({'dates': [None, '3/1/2018', '2/1/2018', '1/1/2018']})
        >>> df.dates.is_monotonic_decreasing
        False

        >>> df.index.is_monotonic_decreasing
        False

        >>> ser = ks.Series([1])
        >>> ser.is_monotonic_decreasing
        True

        >>> ser = ks.Series([])
        >>> ser.is_monotonic_decreasing
        True

        >>> ser.rename("a").to_frame().set_index("a").index.is_monotonic_decreasing
        True

        >>> ser = ks.Series([5, 4, 3, 2, 1], index=[1, 2, 3, 4, 5])
        >>> ser.is_monotonic_decreasing
        True

        >>> ser.index.is_monotonic_decreasing
        False

        Support for MultiIndex

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('x', 'a'), ('x', 'b'), ('y', 'c'), ('y', 'd'), ('z', 'e')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('z', 'e')],
                   )
        >>> midx.is_monotonic_decreasing
        False

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('z', 'e'), ('z', 'd'), ('y', 'c'), ('y', 'b'), ('x', 'a')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('z', 'a'),
                    ('z', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('x', 'e')],
                   )
        >>> midx.is_monotonic_decreasing
        True
        """
        return self._is_monotonic('decreasing')

    def func_oqmp8xb1(self, order: str) -> Column:
        window = Window.partitionBy(F.col('__partition_id')).orderBy(
            NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)
        if order == 'increasing':
            return (F.col('__origin') >= F.lag(F.col('__origin'), 1).over(window)) & F.col('__origin').isNotNull()
        else:
            return (F.col('__origin') <= F.lag(F.col('__origin'), 1).over(window)) & F.col('__origin').isNotNull()

    def func_hiv89kqv(self, order: str) -> bool:
        assert order in ('increasing', 'decreasing')
        sdf = self._internal.spark_frame
        sdf = sdf.select(
            F.spark_partition_id().alias('__partition_id'),
            F.col(NATURAL_ORDER_COLUMN_NAME),
            self.spark.column.alias('__origin')
        ).select(
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
        sdf = sdf.select(
            comparison_col.alias('__comparison_between_partitions'),
            F.col('__comparison_within_partition')
        )
        ret = sdf.select(
            F.min(F.coalesce(F.col('__comparison_between_partitions'), F.lit(True))) &
            F.min(F.coalesce(F.col('__comparison_within_partition'), F.lit(True)))
        ).collect()[0][0]
        if ret is None:
            return True
        else:
            return ret

    @property
    def func_i845f3yz(self) -> int:
        """
        Return an int representing the number of array dimensions.

        Return 1 for Series / Index / MultiIndex.

        Examples
        --------

        For Series

        >>> s = ks.Series([None, 1, 2, 3, 4], index=[4, 5, 2, 1, 8])
        >>> s.ndim
        1

        For Index

        >>> s.index.ndim
        1

        For MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [1, 1, 1, 1, 1, 2, 1, 2, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        >>> s.index.ndim
        1
        """
        return 1

    def func_i5n73z9q(self, dtype: Union[str, np.dtype, type]) -> "IndexOpsMixin":
        """
        Cast a Koalas object to a specified dtype ``dtype``.

        Parameters
        ----------
        dtype : data type
            Use a numpy.dtype or Python type to cast entire pandas object to
            the same type.

        Returns
        -------
        casted : same type as caller

        See Also
        --------
        to_datetime : Convert argument to datetime.

        Examples
        --------
        >>> ser = ks.Series([1, 2], dtype='int32')
        >>> ser
        0    1
        1    2
        dtype: int32

        >>> ser.astype('int64')
        0    1
        1    2
        dtype: int64

        >>> ser.rename("a").to_frame().set_index("a").index.astype('int64')
        Int64Index([1, 2], dtype='int64', name='a')
        """
        dtype, spark_type = koalas_dtype(dtype)
        if not spark_type:
            raise ValueError(f'Type {dtype} not understood')
        if isinstance(self.dtype, CategoricalDtype):
            if isinstance(dtype, CategoricalDtype) and dtype.categories is None:
                return cast(Union[ks.Index, ks.Series], self).copy()
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

    def func_t6zr9gz2(self, values: Union[set, List[Any], np.ndarray, Any]) -> "IndexOpsMixin":
        """
        Check whether `values` are contained in Series or Index.

        Return a boolean Series or Index showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test.

        Returns
        -------
        isin : Series (bool dtype) or Index (bool dtype)

        Examples
        --------
        >>> s = ks.Series(['lama', 'cow', 'lama', 'beetle', 'lama',
        ...                'hippo'], name='animal')
        >>> s.isin(['cow', 'lama'])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('lama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['lama'])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        >>> s.rename("a").to_frame().set_index("a").index.isin(['lama'])
        Index([True, False, True, False, True, False], dtype='object', name='a')
        """
        if not is_list_like(values):
            raise TypeError(
                f'only list-like objects are allowed to be passed to isin(), you passed a [{type(values).__name__}]'
            )
        values_list: List[Any] = values.tolist() if isinstance(values, np.ndarray) else list(values)
        return self._with_new_scol(self.spark.column.isin(values_list))

    def func_06h48wad(self) -> "IndexOpsMixin":
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or numpy.NaN, gets mapped to True values.
        Everything else gets mapped to False values. Characters such as empty strings '' or
        numpy.inf are not considered NA values
        (unless you set pandas.options.mode.use_inf_as_na = True).

        Returns
        -------
        Series or Index : Mask of bool values for each element in Series
            that indicates whether an element is not an NA value.

        Examples
        --------
        >>> ser = ks.Series([5, 6, np.NaN])
        >>> ser.isna()  # doctest: +NORMALIZE_WHITESPACE
        0    False
        1    False
        2     True
        dtype: bool

        >>> ser.rename("a").to_frame().set_index("a").index.isna()
        Index([False, False, True], dtype='object', name='a')
        """
        from databricks.koalas.indexes import MultiIndex
        if isinstance(self, MultiIndex):
            raise NotImplementedError('isna is not defined for MultiIndex')
        if isinstance(self.spark.data_type, (FloatType, DoubleType)):
            return self._with_new_scol(self.spark.column.isNull() | F.isnan(self.spark.column))
        else:
            return self._with_new_scol(self.spark.column.isNull())
    isna = isnull

    def func_fh6mui06(self) -> "IndexOpsMixin":
        """
        Detect existing (non-missing) values.
        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True.
        Characters such as empty strings '' or numpy.inf are not considered NA values
        (unless you set pandas.options.mode.use_inf_as_na = True).
        NA values, such as None or numpy.NaN, get mapped to False values.

        Returns
        -------
        Series or Index : Mask of bool values for each element in Series
            that indicates whether an element is not an NA value.

        Examples
        --------
        Show which entries in a Series are not NA.

        >>> ser = ks.Series([5, 6, np.NaN])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool

        >>> ser.rename("a").to_frame().set_index("a").index.notna()
        Index([True, True, False], dtype='object', name='a')
        """
        from databricks.koalas.indexes import MultiIndex
        if isinstance(self, MultiIndex):
            raise NotImplementedError('notna is not defined for MultiIndex')
        return (~self.isnull()).rename(self.name)
    notna = notnull

    def all(self, axis: Union[int, str] = 0) -> bool:
        """
        Return whether all elements are True.

        Returns True unless there at least one element within a series that is
        False or equivalent (e.g. zero or empty)

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Examples
        --------
        >>> ks.Series([True, True]).all()
        True

        >>> ks.Series([True, False]).all()
        False

        >>> ks.Series([0, 1]).all()
        False

        >>> ks.Series([1, 2, 3]).all()
        True

        >>> ks.Series([True, True, None]).all()
        True

        >>> ks.Series([True, False, None]).all()
        False

        >>> ks.Series([]).all()
        True

        >>> ks.Series([np.nan]).all()
        True

        >>> df = ks.Series([True, False, None]).rename("a").to_frame()
        >>> df.set_index("a").index.all()
        False
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

        Returns False unless there at least one element within a series that is
        True or equivalent (e.g. non-zero or non-empty).

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Examples
        --------
        >>> ks.Series([False, False]).any()
        False

        >>> ks.Series([True, False]).any()
        True

        >>> ks.Series([0, 0]).any()
        False

        >>> ks.Series([0, 1, 2]).any()
        True

        >>> ks.Series([False, False, None]).any()
        False

        >>> ks.Series([True, False, None]).any()
        True

        >>> ks.Series([]).any()
        False

        >>> ks.Series([np.nan]).any()
        False

        >>> df = ks.Series([True, False, None]).rename("a").to_frame()
        >>> df.set_index("a").index.any()
        True
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

    def func_thv6kzqh(self, periods: int = 1, fill_value: Optional[Any] = None) -> "IndexOpsMixin":
        """
        Shift Series/Index by desired number of periods.

        .. note:: the current implementation of shift uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        periods : int
            Number of periods to shift. Can be positive or negative.
        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            The default depends on the dtype of self. For numeric data, np.nan is used.

        Returns
        -------
        Copy of input Series/Index, shifted.

        Examples
        --------
        >>> df = ks.DataFrame({'Col1': [10, 20, 15, 30, 45],
        ...                    'Col2': [13, 23, 18, 33, 48],
        ...                    'Col3': [17, 27, 22, 37, 52]},
        ...                   columns=['Col1', 'Col2', 'Col3'])

        >>> df.Col1.shift(periods=3)
        0     NaN
        1     NaN
        2     NaN
        3    10.0
        4    20.0
        Name: Col1, dtype: float64

        >>> df.Col2.shift(periods=3, fill_value=0)
        0     0
        1     0
        2     0
        3    13
        4    23
        Name: Col2, dtype: int64

        >>> df.index.shift(periods=3, fill_value=0)
        Int64Index([0, 0, 0, 0, 1], dtype='int64')
        """
        return self._shift(periods, fill_value).spark.analyzed

    def func_hqeww4su(
        self,
        periods: int,
        fill_value: Any,
        *,
        part_cols: Tuple[Any, ...] = ()
    ) -> "IndexOpsMixin":
        if not isinstance(periods, int):
            raise ValueError(f'periods should be an int; however, got [{type(periods).__name__}]')
        col = self.spark.column
        window = Window.partitionBy(*part_cols).orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        lag_col = F.lag(col, periods).over(window)
        col = F.when(lag_col.isNull() | F.isnan(lag_col), fill_value).otherwise(lag_col)
        return self._with_new_scol(col.alias(self._internal.data_spark_column_names[0]))

    def func_onc402ao(
        self,
        dropna: bool = True,
        approx: bool = False,
        rsd: float = 0.05,
        bins: Optional[Any] = None,
        sort: bool = True,
        normalize: bool = False,
        ascending: bool = False
    ) -> "Series":
        """
        Return a Series containing counts of unique values.
        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : boolean, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : boolean, default True
            Sort by values.
        ascending : boolean, default False
            Sort in ascending order.
        bins : Not Yet Supported
        dropna : boolean, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.

        Examples
        --------
        For Series

        >>> df = ks.DataFrame({'x':[0, 0, 1, 1, 1, np.nan]})
        >>> df.x.value_counts()  # doctest: +NORMALIZE_WHITESPACE
        1.0    3
        0.0    2
        Name: x, dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> df.x.value_counts(normalize=True)  # doctest: +NORMALIZE_WHITESPACE
        1.0    0.6
        0.0    0.4
        Name: x, dtype: float64

        **dropna**
        With `dropna` set to `False` we can also see NaN index values.

        >>> df.x.value_counts(dropna=False)  # doctest: +NORMALIZE_WHITESPACE
        1.0    3
        0.0    2
        NaN    1
        Name: x, dtype: int64

        For Index

        >>> idx = ks.Index([3, 1, 2, 3, 4, np.nan])
        >>> idx
        Float64Index([3.0, 1.0, 2.0, 3.0, 4.0, nan], dtype='float64')

        >>> idx.value_counts().sort_index()
        1.0    1
        2.0    1
        3.0    2
        4.0    1
        dtype: int64

        **sort**

        With `sort` set to `False`, the result wouldn't be sorted by number of count.

        >>> idx.value_counts(sort=True).sort_index()
        1.0    1
        2.0    1
        3.0    2
        4.0    1
        dtype: int64

        **normalize**

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> idx.value_counts(normalize=True).sort_index()
        1.0    0.2
        2.0    0.2
        3.0    0.4
        4.0    0.2
        dtype: float64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> idx.value_counts(dropna=False).sort_index()  # doctest: +SKIP
        1.0    1
        2.0    1
        3.0    2
        4.0    1
        NaN    1
        dtype: int64

        For MultiIndex.

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [1, 1, 1, 1, 1, 2, 1, 2, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        >>> s.index  # doctest: +SKIP
        MultiIndex([('lama', 'weight'),
                    ('lama', 'weight'),
                    ('lama', 'weight'),
                    ('cow', 'weight'),
                    ('cow', 'weight'),
                    ('cow', 'length'),
                    ('falcon', 'weight'),
                    ('falcon', 'length'),
                    ('falcon', 'length')],
                   )

        >>> s.index.value_counts().sort_index()
        (cow, length)       1
        (cow, weight)       2
        (falcon, length)    2
        (falcon, weight)    1
        (lama, weight)      3
        dtype: int64

        >>> s.index.value_counts(normalize=True).sort_index()
        (cow, length)       0.111111
        (cow, weight)       0.222222
        (falcon, length)    0.222222
        (falcon, weight)    0.111111
        (lama, weight)      0.333333
        dtype: float64

        If Index has name, keep the name up.

        >>> idx = ks.Index([0, 0, 0, 1, 1, 2, 3], name='koalas')
        >>> idx.value_counts().sort_index()
        0    3
        1    2
        2    1
        3    1
        Name: koalas, dtype: int64
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
            sum_count = sdf_dropna.count()
            sdf = sdf.withColumn('count', F.col('count') / F.lit(sum_count))
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, index_name)],
            column_labels=self._internal.column_labels,
            data_spark_columns=[scol_for(sdf, 'count')],
            column_label_names=self._internal.column_label_names
        )
        return first_series(DataFrame(internal))

    def func_gguz4uen(self, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> int:
        """
        Return number of unique elements in the object.
        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don’t include NaN in the count.
        approx: bool, default False
            If False, will use the exact algorithm and return the exact number of unique.
            If True, it uses the HyperLogLog approximate algorithm, which is significantly faster
            for large amount of data.
            Note: This parameter is specific to Koalas and is not found in pandas.
        rsd: float, default 0.05
            Maximum estimation error allowed in the HyperLogLog algorithm.
            Note: Just like ``approx`` this parameter is specific to Koalas.

        Returns
        -------
        int

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> ks.Series([1, 2, 3, np.nan]).nunique()
        3

        >>> ks.Series([1, 2, 3, np.nan]).nunique(dropna=False)
        4

        On big data, we recommend using the approximate algorithm to speed up this function.
        The result will be very close to the exact unique count.

        >>> ks.Series([1, 2, 3, np.nan]).nunique(approx=True)
        3

        >>> idx = ks.Index([1, 1, 2, None])
        >>> idx
        Float64Index([1.0, 1.0, 2.0, nan], dtype='float64')

        >>> idx.nunique()
        2

        >>> idx.nunique(dropna=False)
        3
        """
        res = self._internal.spark_frame.select([self._nunique(dropna, approx, rsd)])
        return res.collect()[0][0]

    def func_shm6w0mi(self, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> Column:
        colname = self._internal.data_spark_column_names[0]
        count_fn: Callable[[Column], Column]
        if approx:
            count_fn = partial(F.approx_count_distinct, rsd=rsd)
        else:
            count_fn = F.countDistinct
        if dropna:
            return count_fn(self.spark.column).alias(colname)
        else:
            return (count_fn(self.spark.column) + F.when(F.count(F.when(self.spark.column.isNull(), 1).otherwise(None)) >= 1, 1).otherwise(0)).alias(colname)

    def func_7d6s4zrf(self, indices: List[int]) -> Union["Series", "Index"]:
        """
        Return the elements in the given *positional* indices along an axis.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.

        Returns
        -------
        taken : same type as caller
            An array-like containing the elements taken from the object.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by labels.
        DataFrame.iloc : Select a subset of a DataFrame by positions.
        numpy.take : Take elements from an array along an axis.

        Examples
        --------

        Series

        >>> kser = ks.Series([100, 200, 300, 400, 500])
        >>> kser
        0    100
        1    200
        2    300
        3    400
        4    500
        dtype: int64

        >>> kser.take([0, 2, 4]).sort_index()
        0    100
        2    300
        4    500
        dtype: int64

        Index

        >>> kidx = ks.Index([100, 200, 300, 400, 500])
        >>> kidx
        Int64Index([100, 200, 300, 400, 500], dtype='int64')

        >>> kidx.take([0, 2, 4]).sort_values()
        Int64Index([100, 300, 500], dtype='int64')

        MultiIndex

        >>> kmidx = ks.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("x", "c")])
        >>> kmidx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('x', 'c')],
                   )

        >>> kmidx.take([0, 2])  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'c')],
                   )
        """
        if not is_list_like(indices) or isinstance(indices, (dict, set)):
            raise ValueError('`indices` must be a list-like except dict or set')
        if isinstance(self, ks.Series):
            return cast(ks.Series, self.iloc[indices])
        else:
            return self._kdf.iloc[indices].index

    def func_shfxfpkf(
        self,
        sort: bool = True,
        na_sentinel: Optional[int] = -1
    ) -> Tuple["Series", pd.Index]:
        """
        Encode the object as an enumerated type or categorical variable.

        This method is useful for obtaining a numeric representation of an
        array when all that matters is identifying distinct values.

        Parameters
        ----------
        sort : bool, default True
        na_sentinel : int or None, default -1
            Value to mark "not found". If None, will not drop the NaN
            from the uniques of the values.

        Returns
        -------
        codes : Series or Index
            A Series or Index that's an indexer into `uniques`.
            ``uniques.take(codes)`` will have the same values as `values`.
        uniques : pd.Index
            The unique valid values.

            .. note ::

               Even if there's a missing value in `values`, `uniques` will
               *not* contain an entry for it.

        Examples
        --------
        >>> kser = ks.Series(['b', None, 'a', 'c', 'b'])
        >>> codes, uniques = kser.factorize()
        >>> codes
        0    1
        1   -1
        2    0
        3    2
        4    1
        dtype: int32
        >>> uniques
        Index(['a', 'b', 'c'], dtype='object')

        >>> codes, uniques = kser.factorize(na_sentinel=None)
        >>> codes
        0    1
        1    3
        2    0
        3    2
        4    1
        dtype: int32
        >>> uniques
        Index(['a', 'b', 'c', None], dtype='object')

        >>> codes, uniques = kser.factorize(na_sentinel=-2)
        >>> codes
        0    1
        1   -2
        2    0
        3    2
        4    1
        dtype: int32
        >>> uniques
        Index(['a', 'b', 'c'], dtype='object')

        For Index:

        >>> kidx = ks.Index(['b', None, 'a', 'c', 'b'])
        >>> codes, uniques = kidx.factorize()
        >>> codes
        Int64Index([1, -1, 0, 2, 1], dtype='int64')
        >>> uniques
        Index(['a', 'b', 'c'], dtype='object')
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
            return codes, uniques.astype(self.dtype)
        uniq_sdf = self._internal.spark_frame.select(self.spark.column).distinct()
        max_compute_count = get_option('compute.max_rows')
        if max_compute_count is not None:
            uniq_pdf = uniq_sdf.limit(max_compute_count + 1).toPandas()
            if len(uniq_pdf) > max_compute_count:
                raise ValueError(
                    f"Current Series has more then {max_compute_count} unique values. Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option' to more than {max_compute_count} rows. Note that, before changing the 'compute.max_rows', this operation is considerably expensive."
                )
        else:
            uniq_pdf = uniq_sdf.toPandas()
        uniq_series = first_series(uniq_pdf).drop_duplicates()
        uniques_list = uniq_series.tolist()
        uniques_list = sorted(uniques_list, key=lambda x: (pd.isna(x), x))
        unique_to_code: dict[Any, int] = {}
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
        codes: "IndexOpsMixin" = self._with_new_scol(new_scol.alias(self._internal.data_spark_column_names[0]))
        if na_sentinel is not None:
            uniques_list = [x for x in uniques_list if not pd.isna(x)]
        uniques = pd.Index(uniques_list)
        return codes, uniques
