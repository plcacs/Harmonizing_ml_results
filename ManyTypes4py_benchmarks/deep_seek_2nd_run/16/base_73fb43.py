"""
Base and utility classes for Koalas objects.
"""
from abc import ABCMeta, abstractmethod
import datetime
from functools import wraps, partial
from itertools import chain
from typing import Any, Callable, Optional, Tuple, Union, cast, TYPE_CHECKING, List, Dict, Set, Sequence, TypeVar
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

T = TypeVar('T')

def should_alignment_for_column_op(self: 'IndexOpsMixin', other: Any) -> bool:
    from databricks.koalas.series import Series
    if isinstance(self, Series) and isinstance(other, Series):
        return not same_anchor(self, other)
    else:
        return self._internal.spark_frame is not other._internal.spark_frame

def align_diff_index_ops(func: Callable, this_index_ops: 'IndexOpsMixin', *args: Any) -> Union['Index', 'Series']:
    """
    Align the `IndexOpsMixin` objects and apply the function.
    """
    from databricks.koalas.indexes import Index
    from databricks.koalas.series import Series, first_series
    cols = [arg for arg in args if isinstance(arg, IndexOpsMixin)]
    if isinstance(this_index_ops, Series) and all((isinstance(col, Series) for col in cols)):
        combined = combine_frames(this_index_ops.to_frame(), *[cast(Series, col).rename(i) for i, col in enumerate(cols)], how='full')
        return column_op(func)(combined['this']._kser_for(combined['this']._internal.column_labels[0]), *[combined['that']._kser_for(label) for label in combined['that']._internal.column_labels]).rename(this_index_ops.name)
    else:
        self_len = len(this_index_ops)
        if any((len(col) != self_len for col in args if isinstance(col, IndexOpsMixin))):
            raise ValueError('operands could not be broadcast together with shapes')
        with option_context('compute.default_index_type', 'distributed-sequence'):
            if isinstance(this_index_ops, Index) and all((isinstance(col, Index) for col in cols)):
                return Index(column_op(func)(this_index_ops.to_series().reset_index(drop=True), *[arg.to_series().reset_index(drop=True) if isinstance(arg, Index) else arg for arg in args]).sort_index(), name=this_index_ops.name)
            elif isinstance(this_index_ops, Series):
                this = this_index_ops.reset_index()
                that = [cast(Series, col.to_series() if isinstance(col, Index) else col).rename(i).reset_index(drop=True) for i, col in enumerate(cols)]
                combined = combine_frames(this, *that, how='full').sort_index()
                combined = combined.set_index(combined._internal.column_labels[:this_index_ops._internal.index_level])
                combined.index.names = this_index_ops._internal.index_names
                return column_op(func)(first_series(combined['this']), *[combined['that']._kser_for(label) for label in combined['that']._internal.column_labels]).rename(this_index_ops.name)
            else:
                this = cast(Index, this_index_ops).to_frame().reset_index(drop=True)
                that_series = next((col for col in cols if isinstance(col, Series)))
                that_frame = that_series._kdf[[cast(Series, col.to_series() if isinstance(col, Index) else col).rename(i) for i, col in enumerate(cols)]]
                combined = combine_frames(this, that_frame.reset_index()).sort_index()
                self_index = combined['this'].set_index(combined['this']._internal.column_labels).index
                other = combined['that'].set_index(combined['that']._internal.column_labels[:that_series._internal.index_level])
                other.index.names = that_series._internal.index_names
                return column_op(func)(self_index, *[other._kser_for(label) for label, col in zip(other._internal.column_labels, cols)]).rename(that_series.name)

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
    A decorator that wraps APIs taking/returning Spark Column.
    """
    @wraps(f)
    def wrapper(self: 'IndexOpsMixin', *args: Any) -> 'IndexOpsMixin':
        from databricks.koalas.series import Series
        cols = [arg for arg in args if isinstance(arg, IndexOpsMixin)]
        if all((not should_alignment_for_column_op(self, col) for col in cols)):
            args = [arg.spark.column if isinstance(arg, IndexOpsMixin) else arg for arg in args]
            scol = f(self.spark.column, *args)
            spark_type = self._internal.spark_frame.select(scol).schema[0].dataType
            use_extension_dtypes = any((isinstance(col.dtype, extension_dtypes) for col in [self] + cols))
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
    def wrapper(self: 'IndexOpsMixin', *args: Any) -> 'IndexOpsMixin':
        new_args = []
        for arg in args:
            if isinstance(self.spark.data_type, LongType) and isinstance(arg, np.timedelta64):
                new_args.append(float(arg / np.timedelta64(1, 's')))
            else:
                new_args.append(arg)
        return column_op(f)(self, *new_args)
    return wrapper

class IndexOpsMixin(object, metaclass=ABCMeta):
    """common ops mixin to support a unified interface / docs for Series / Index"""

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @property
    @abstractmethod
    def _kdf(self) -> DataFrame:
        pass

    @abstractmethod
    def _with_new_scol(self, scol: Column, *, dtype: Optional[Dtype] = None) -> 'IndexOpsMixin':
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
        warnings.warn('Series.spark_column is deprecated as of Series.spark.column. Please use the API instead.', FutureWarning)
        return self.spark.column
    spark_column.__doc__ = SparkIndexOpsMethods.column.__doc__
    __neg__ = column_op(Column.__neg__)

    def __add__(self, other: Any) -> 'IndexOpsMixin':
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

    def __sub__(self, other: Any) -> 'IndexOpsMixin':
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('substraction can not be applied to string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            msg = "Note that there is a behavior difference of timestamp subtraction. The timestamp subtraction returns an integer in seconds, whereas pandas returns 'timedelta64[ns]'."
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

    def __mul__(self, other: Any) -> 'IndexOpsMixin':
        if isinstance(other, str):
            raise TypeError('multiplication can not be applied to a string literal.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('multiplication can not be applied to date times.')
        if isinstance(self.spark.data_type, IntegralType) and isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType):
            return column_op(SF.repeat)(other, self)
        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, IntegralType) or isinstance(other, int):
                return column_op(SF.repeat)(self, other)
            else:
                raise TypeError('a string series can only be multiplied to an int series or literal')
        return column_op(Column.__mul__)(self, other)

    def __truediv__(self, other: Any) -> 'IndexOpsMixin':
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('division can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def truediv(left: Column, right: Any) -> Column:
            return F.when(F.lit(right != 0) | F.lit(right).isNull(), left.__div__(right)).otherwise(F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left).otherwise(F.lit(np.inf).__div__(left)))
        return numpy_column_op(truediv)(self, other)

    def __mod__(self, other: Any) -> 'IndexOpsMixin':
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('modulo can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('modulo can not be applied to date times.')

        def mod(left: Column, right: Any) -> Column:
            return (left % right + right) % right
        return column_op(mod)(self, other)

    def __radd__(self, other: Any) -> 'IndexOpsMixin':
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

    def __rsub__(self, other: Any) -> 'IndexOpsMixin':
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError('substraction can not be applied to string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            msg = "Note that there is a behavior difference of timestamp subtraction. The timestamp subtraction returns an integer in seconds, whereas pandas returns 'timedelta64[ns]'."
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

    def __rmul__(self, other: Any) -> 'IndexOpsMixin':
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

    def __rtruediv__(self, other: Any) -> 'IndexOpsMixin':
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError('division can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def rtruediv(left: Any, right: Column) -> Column:
            return F.when(left == 0, F.lit(np.inf).__div__(right)).otherwise(F.lit(right).__truediv__(left))
        return numpy_column_op(rtruediv)(self, other)

    def __floordiv__(self, other: Any) -> 'IndexOpsMixin':
        if isinstance(self.spark.data_type, StringType) or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType)) or isinstance(other, str):
            raise TypeError('division can not be applied on string series or literals.')
        if isinstance(self.spark.data_type, TimestampType):
            raise TypeError('division can not be applied to date times.')

        def floordiv(left: Column, right: Any) -> Column:
            return F.when(F.lit