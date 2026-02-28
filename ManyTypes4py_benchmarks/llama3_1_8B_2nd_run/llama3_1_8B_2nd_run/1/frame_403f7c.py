from typing import Any, Optional, List, Tuple, Union, Generic, TypeVar, Iterable, Iterator, Dict, Callable, cast, TYPE_CHECKING
import re
import warnings
import inspect
import json
import types
from functools import partial, reduce
import sys
from itertools import zip_longest
from typing import Any, Optional, List, Tuple, Union, Generic, TypeVar, Iterable, Iterator, Dict, Callable, cast, TYPE_CHECKING
import datetime
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.api.types import is_list_like, is_dict_like, is_scalar
from pandas.api.extensions import ExtensionDtype
from pandas.tseries.frequencies import DateOffset, to_offset
if TYPE_CHECKING:
    from pandas.io.formats.style import Styler
if LooseVersion(pd.__version__) >= LooseVersion('0.24'):
    from pandas.core.dtypes.common import infer_dtype_from_object
else:
    from pandas.core.dtypes.common import _get_dtype_from_object as infer_dtype_from_object
from pandas.core.accessor import CachedAccessor
from pandas.core.dtypes.inference import is_sequence
import pyspark
from pyspark import StorageLevel
from pyspark import sql as spark
from pyspark.sql import Column, DataFrame as SparkDataFrame, functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType, StringType, StructType, StructField, ArrayType
from pyspark.sql.window import Window
from databricks import koalas as ks
from databricks.koalas.accessors import KoalasFrameMethods
from databricks.koalas.config import option_context, get_option
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkFrameMethods, CachedSparkFrameMethods
from databricks.koalas.utils import align_diff_frames, column_labels_level, combine_frames, default_session, is_name_like_tuple, is_name_like_value, is_testing, name_like_string, same_anchor, scol_for, validate_arguments_and_invoke_function, validate_axis, validate_bool_kwarg, validate_how, verify_temp_column_name
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.generic import Frame
from databricks.koalas.internal import InternalFrame, HIDDEN_COLUMNS, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT, SPARK_DEFAULT_INDEX_NAME, SPARK_DEFAULT_SERIES_NAME
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.typedef import as_spark_type, infer_return_type, spark_type_to_pandas_dtype, DataFrameType, SeriesType, Scalar, ScalarType
from databricks.koalas.plot import KoalasPlotAccessor
if TYPE_CHECKING:
    from databricks.koalas.indexes import Index
    from databricks.koalas.series import Series
T = TypeVar('T')

class DataFrame(Generic[T]):
    """
    Koalas DataFrame that corresponds to pandas DataFrame logically. This holds Spark DataFrame
    internally.

    :ivar _internal: an internal immutable Frame to manage metadata.
    :type _internal: InternalFrame

    Parameters
    ----------
    data : numpy ndarray (structured or homogeneous), dict, pandas DataFrame, Spark DataFrame or Koalas Series
        Dict can contain Series, arrays, constants, or list-like objects
        If data is a dict, argument order is maintained for Python 3.6
        and later.
        Note that if `data` is a pandas DataFrame, a Spark DataFrame, and a Koalas Series,
        other arguments should not be used.
    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided
    columns : Index or array-like
        Column labels to use for resulting frame. Will default to
        RangeIndex (0, 1, 2, ..., n) if no column labels are provided
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer
    copy : boolean, default False
        Copy data from inputs. Only affects DataFrame / 2d ndarray input

    Examples
    --------
    Constructing DataFrame from a dictionary.

    >>> d = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = ks.DataFrame(data=d, columns=['col1', 'col2'])
    >>> df
       col1  col2
    0     1     3
    1     2     4

    Constructing DataFrame from pandas DataFrame

    >>> df = ks.DataFrame(pd.DataFrame(data=d, columns=['col1', 'col2']))
    >>> df
       col1  col2
    0     1     3
    1     2     4

    Notice that the inferred dtype is int64.

    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object

    To enforce a single dtype:

    >>> df = ks.DataFrame(data=d, dtype=np.int8)
    >>> df.dtypes
    col1    int8
    col2    int8
    dtype: object

    Constructing DataFrame from numpy ndarray:

    >>> df2 = ks.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),
    ...                    columns=['a', 'b', 'c', 'd', 'e'])
    >>> df2  # doctest: +SKIP
       a  b  c  d  e
    0  3  1  4  9  8
    1  4  8  4  8  4
    2  7  6  5  6  7
    3  8  7  9  1  0
    4  2  5  4  3  9
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        if isinstance(data, InternalFrame):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            internal = data
        elif isinstance(data, spark.DataFrame):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            internal = InternalFrame(spark_frame=data, index_spark_columns=None)
        elif isinstance(data, ks.Series):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            data = data.to_frame()
            internal = data._internal
        else:
            if isinstance(data, pd.DataFrame):
                assert index is None
                assert columns is None
                assert dtype is None
                assert not copy
                pdf = data
            else:
                pdf = pd.DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
            internal = InternalFrame.from_pandas(pdf)
        object.__setattr__(self, '_internal_frame', internal)

    @property
    def _ksers(self):
        """ Return a dict of column label -> Series which anchors `self`. """
        from databricks.koalas.series import Series
        if not hasattr(self, '_kseries'):
            object.__setattr__(self, '_kseries', {label: Series(data=self, index=label) for label in self._internal.column_labels})
        else:
            kseries = self._kseries
            assert len(self._internal.column_labels) == len(kseries), (len(self._internal.column_labels), len(kseries))
            if any((self is not kser._kdf for kser in kseries.values())):
                self._kseries = {label: kser._kdf._kseries[label] if self is kser._kdf else Series(data=self, index=label) for label in self._internal.column_labels}
        return self._kseries

    @property
    def _internal(self):
        return self._internal_frame

    def _update_internal_frame(self, internal, requires_same_anchor=True):
        """
        Update InternalFrame with the given one.

        If the column_label is changed or the new InternalFrame is not the same `anchor`,
        disconnect the link to the Series and create a new one.

        If `requires_same_anchor` is `False`, checking whether or not the same anchor is ignored
        and force to update the InternalFrame, e.g., replacing the internal with the resolved_copy,
        updating the underlying Spark DataFrame which need to combine a different Spark DataFrame.

        :param internal: the new InternalFrame
        :param requires_same_anchor: whether checking the same anchor
        """
        from databricks.koalas.series import Series
        if hasattr(self, '_kseries'):
            kseries = {}
            for old_label, new_label in zip_longest(self._internal.column_labels, internal.column_labels):
                if old_label is not None:
                    kser = self._ksers[old_label]
                    renamed = old_label != new_label
                    not_same_anchor = requires_same_anchor and (not same_anchor(internal, kser))
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
        self._internal_frame = internal
        if hasattr(self, '_repr_pandas_cache'):
            del self._repr_pandas_cache

    @property
    def ndim(self):
        """
        Return an int representing the dimensionality of the DataFrame.

        return 2 for DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', None],
        ...                   columns=['max_speed', 'shield'])
        >>> df
               max_speed  shield
        cobra          1       2
        viper          4       5
        NaN            7       8
        >>> df.ndim
        2
        """
        return 2

    @property
    def axes(self):
        """
        Return a list representing the axes of the DataFrame.

        It has the row axis labels and column axis labels as the only members.
        They are returned in that order.

        Examples
        --------

        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.axes
        [Int64Index([0, 1], dtype='int64'), Index(['col1', 'col2'], dtype='object')]
        """
        return [self.index, self.columns]

    def _reduce_for_stat_function(self, sfun, name, axis=None, numeric_only=True, **kwargs):
        """
        Applies sfun to each column and returns a pd.Series where the number of rows equal the
        number of columns.

        Parameters
        ----------
        sfun : either an 1-arg function that takes a Column and returns a Column, or
            a 2-arg function that takes a Column and its DataType and returns a Column.
            axis: used only for sanity check because series only support index axis.
        name : original pandas API name.
        axis : axis to apply. 0 or 1, or 'index' or 'columns.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility. Only 'DataFrame.count' uses this parameter
            currently.
        """
        from inspect import signature
        from databricks.koalas.series import Series, first_series
        axis = validate_axis(axis)
        if axis == 0:
            min_count = kwargs.get('min_count', 0)
            exprs = [F.lit(None).cast(StringType()).alias(SPARK_DEFAULT_INDEX_NAME)]
            new_column_labels = []
            num_args = len(signature(sfun).parameters)
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
                        scol = F.when(Frame._count_expr(spark_column, spark_type) >= min_count, scol)
                    exprs.append(scol.alias(name_like_string(label)))
                    new_column_labels.append(label)
            if len(exprs) == 1:
                return Series([])
            sdf = self._internal.spark_frame.select(*exprs)
            with ks.option_context('compute.max_rows', 1):
                internal = InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, SPARK_DEFAULT_INDEX_NAME)], column_labels=new_column_labels, column_label_names=self._internal.column_label_names)
                return first_series(DataFrame(internal).transpose())
        else:
            limit = get_option('compute.shortcut_limit')
            pdf = self.head(limit + 1)._to_internal_pandas()
            pser = getattr(pdf, name)(axis=axis, numeric_only=numeric_only, **kwargs)
            if len(pdf) <= limit:
                return Series(pser)

            @pandas_udf(returnType=as_spark_type(pser.dtype.type))
            def calculate_columns_axis(*cols):
                return getattr(pd.concat(cols, axis=1), name)(axis=axis, numeric_only=numeric_only, **kwargs)
            column_name = verify_temp_column_name(self._internal.spark_frame.select(self._internal.index_spark_columns + [calculate_columns_axis(*self._internal.data_spark_columns).alias(column_name)]), '__calculate_columns_axis__')
            sdf = self._internal.spark_frame.select(self._internal.index_spark_columns + [calculate_columns_axis(*self._internal.data_spark_columns).alias(column_name)])
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=[scol_for(sdf, col) for col in self._internal.index_spark_column_names], index_names=self._internal.index_names, index_dtypes=self._internal.index_dtypes)
            return first_series(DataFrame(internal)).rename(pser.name)

    def _kser_for(self, label):
        """
        Create Series with a proper column label.

        The given label must be verified to exist in `InternalFrame.column_labels`.

        For example, in some method, self is like:

        >>> self = ks.range(3)

        `self._kser_for(label)` can be used with `InternalFrame.column_labels`:

        >>> self._kser_for(self._internal.column_labels[0])
        0    0
        1    1
        2    2
        Name: id, dtype: int64

        `self._kser_for(label)` must not be used directly with user inputs.
        In that case, `self[label]` should be used instead, which checks the label exists or not:

        >>> self['id']
        0    0
        1    1
        2    2
        Name: id, dtype: int64
        """
        return self._ksers[label]

    def _apply_series_op(self, op, should_resolve=False):
        applied = []
        for label in self._internal.column_labels:
            applied.append(op(self._kser_for(label)))
        internal = self._internal.with_new_columns(applied)
        if should_resolve:
            internal = internal.resolved_copy
        return DataFrame(internal)

    def _map_series_op(self, op, other):
        from databricks.koalas.base import IndexOpsMixin
        if not isinstance(other, DataFrame) and (isinstance(other, IndexOpsMixin) or is_sequence(other)):
            raise ValueError('%s with a sequence is currently not supported; however, got %s.' % (op, type(other).__name__))
        if isinstance(other, DataFrame):
            if self._internal.column_labels_level != other._internal.column_labels_level:
                raise ValueError('cannot join with no overlapping index names')
            if not same_anchor(self, other):

                def apply_op(kdf, this_column_labels, that_column_labels):
                    for this_label, that_label in zip(this_column_labels, that_column_labels):
                        yield (getattr(kdf._kser_for(this_label), op)(kdf._kser_for(that_label)).rename(this_label), this_label)
                return align_diff_frames(apply_op, self, other, fillna=True, how='full')
            else:
                applied = []
                column_labels = []
                for label in self._internal.column_labels:
                    if label in other._internal.column_labels:
                        applied.append(getattr(self._kser_for(label), op)(self._kser_for(label)))
                    else:
                        applied.append(F.lit(None).cast(self._internal.spark_type_for(label)).alias(name_like_string(label)))
                    column_labels.append(label)
                for label in other._internal.column_labels:
                    if label not in column_labels:
                        applied.append(F.lit(None).cast(other._internal.spark_type_for(label)).alias(name_like_string(label)))
                        column_labels.append(label)
                internal = self._internal.with_new_columns(applied, column_labels=column_labels)
                return DataFrame(internal)
        else:
            return self._apply_series_op(lambda kser: getattr(kser, op)(other))

    def __add__(self, other):
        return self._map_series_op('add', other)

    def __radd__(self, other):
        return self._map_series_op('radd', other)

    def __div__(self, other):
        return self._map_series_op('div', other)

    def __rdiv__(self, other):
        return self._map_series_op('rdiv', other)

    def __truediv__(self, other):
        return self._map_series_op('truediv', other)

    def __rtruediv__(self, other):
        return self._map_series_op('rtruediv', other)

    def __mul__(self, other):
        return self._map_series_op('mul', other)

    def __rmul__(self, other):
        return self._map_series_op('rmul', other)

    def __sub__(self, other):
        return self._map_series_op('sub', other)

    def __rsub__(self, other):
        return self._map_series_op('rsub', other)

    def __pow__(self, other):
        return self._map_series_op('pow', other)

    def __rpow__(self, other):
        return self._map_series_op('rpow', other)

    def __mod__(self, other):
        return self._map_series_op('mod', other)

    def __rmod__(self, other):
        return self._map_series_op('rmod', other)

    def __floordiv__(self, other):
        return self._map_series_op('floordiv', other)

    def __rfloordiv__(self, other):
        return self._map_series_op('rfloordiv', other)

    def __abs__(self):
        return self._apply_series_op(lambda kser: abs(kser))

    def __neg__(self):
        return self._apply_series_op(lambda kser: -kser)

    def add(self, other):
        return self + other
    plot = CachedAccessor('plot', KoalasPlotAccessor)
    hist = CachedAccessor('hist', KoalasPlotAccessor)
    kde = CachedAccessor('kde', KoalasPlotAccessor)
    add.__doc__ = _flex_doc_FRAME.format(desc='Addition', op_name='+', equiv='dataframe + other', reverse='radd')

    def radd(self, other):
        return other + self
    radd.__doc__ = _flex_doc_FRAME.format(desc='Addition', op_name='+', equiv='other + dataframe', reverse='add')

    def div(self, other):
        return self / other
    div.__doc__ = _flex_doc_FRAME.format(desc='Floating division', op_name='/', equiv='dataframe / other', reverse='rdiv')
    divide = div

    def rdiv(self, other):
        return other / self
    rdiv.__doc__ = _flex_doc_FRAME.format(desc='Floating division', op_name='/', equiv='other / dataframe', reverse='div')

    def truediv(self, other):
        return self / other
    truediv.__doc__ = _flex_doc_FRAME.format(desc='Floating division', op_name='/', equiv='dataframe / other', reverse='rtruediv')

    def rtruediv(self, other):
        return other / self
    rtruediv.__doc__ = _flex_doc_FRAME.format(desc='Floating division', op_name='/', equiv='other / dataframe', reverse='truediv')

    def mul(self, other):
        return self * other
    mul.__doc__ = _flex_doc_FRAME.format(desc='Multiplication', op_name='*', equiv='dataframe * other', reverse='rmul')
    multiply = mul

    def rmul(self, other):
        return other * self
    rmul.__doc__ = _flex_doc_FRAME.format(desc='Multiplication', op_name='*', equiv='other * dataframe', reverse='mul')

    def sub(self, other):
        return self - other
    sub.__doc__ = _flex_doc_FRAME.format(desc='Subtraction', op_name='-', equiv='dataframe - other', reverse='rsub')
    subtract = sub

    def rsub(self, other):
        return other - self
    rsub.__doc__ = _flex_doc_FRAME.format(desc='Subtraction', op_name='-', equiv='other - dataframe', reverse='sub')

    def mod(self, other):
        return self % other
    mod.__doc__ = _flex_doc_FRAME.format(desc='Modulo', op_name='%', equiv='dataframe % other', reverse='rmod')

    def rmod(self, other):
        return other % self
    rmod.__doc__ = _flex_doc_FRAME.format(desc='Modulo', op_name='%', equiv='other % dataframe', reverse='mod')

    def pow(self, other):
        return self ** other
    pow.__doc__ = _flex_doc_FRAME.format(desc='Exponential power of series', op_name='**', equiv='dataframe ** other', reverse='rpow')

    def rpow(self, other):
        return other ** self
    rpow.__doc__ = _flex_doc_FRAME.format(desc='Exponential power', op_name='**', equiv='other ** dataframe', reverse='pow')

    def floordiv(self, other):
        return self // other
    floordiv.__doc__ = _flex_doc_FRAME.format(desc='Integer division', op_name='//', equiv='dataframe // other', reverse='rfloordiv')

    def rfloordiv(self, other):
        return other // self
    rfloordiv.__doc__ = _flex_doc_FRAME.format(desc='Integer division', op_name='//', equiv='other // dataframe', reverse='floordiv')

    def __eq__(self, other):
        return self._map_series_op('eq', other)

    def __ne__(self, other):
        return self._map_series_op('ne', other)

    def __lt__(self, other):
        return self._map_series_op('lt', other)

    def __le__(self, other):
        return self._map_series_op('le', other)

    def __ge__(self, other):
        return self._map_series_op('ge', other)

    def __gt__(self, other):
        return self._map_series_op('gt', other)

    def eq(self, other):
        """
        Compare if the current value is equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.eq(1)
               a      b
        a   True  False
        b  False  False
        c  False  False
        d  False  False
        """
        return self == other
    equals = eq

    def gt(self, other):
        """
        Compare if the current value is greater than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.gt(2)
               a      b
        a  False  False
        b  False  False
        c   True  False
        d   True  False
        """
        return self > other

    def ge(self, other):
        """
        Compare if the current value is greater than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.ge(1)
              a      b
        a   True  False
        b  False  False
        c  False  False
        d  False  False
        """
        return self >= other

    def lt(self, other):
        """
        Compare if the current value is less than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.lt(1)
               a      b
        a  False  False
        b  False  False
        c  False  False
        d  False  False
        """
        return self < other

    def le(self, other):
        """
        Compare if the current value is less than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.le(2)
               a      b
        a   True  False
        b  False  False
        c   True  False
        d  False  False
        """
        return self <= other

    def ne(self, other):
        """
        Compare if the current value is not equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.ne(1)
               a      b
        a  False  False
        b   True  False
        c   True  False
        d   True  False
        """
        return self != other

    def applymap(self, func):
        """
        Apply a function to a Dataframe elementwise.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> np.int32:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.

        Returns
        -------
        DataFrame
            Transformed DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2.12], [3.356, 4.567]])
        >>> df
               0      1
        0  1.000  2.120
        1  3.356  4.567

        >>> def str_len(x) -> int:
        ...     return len(str(x))
        >>> df.applymap(str_len)
           0  1
        0  3  4
        1  5  5

        >>> df.applymap(lambda x: x ** 2)
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489
        """
        return self._apply_series_op(lambda kser: kser.apply(func))

    def aggregate(self, func):
        """Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : dict or a list
             a dict mapping from column name (string) to
             aggregate functions (list of strings).
             If a list is given, the aggregation is performed against
             all columns.

        Returns
        -------
        DataFrame

        Notes
        -----
        `agg` is an alias for `aggregate`. Use the alias.

        See Also
        --------
        DataFrame.apply : Invoke function on DataFrame.
        DataFrame.transform : Only perform transforming type operations.
        DataFrame.groupby : Perform operations over groups.
        Series.aggregate : The equivalent function for Series.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2, 3],
        ...                    [4, 5, 6],
        ...                    [7, 8, 9],
        ...                    [np.nan, np.nan, np.nan]],
        ...                   columns=['A', 'B', 'C'])

        >>> df
             A    B    C
        0  1.0  2.0  3.0
        1  4.0  5.0  6.0
        2  7.0  8.0  9.0
        3  NaN  NaN  NaN

        Aggregate these functions over the rows.

        >>> df.agg(['sum', 'min'])
                A     B     C
        sum  21.0  15.0  18.0

        Different aggregations per column.

        >>> df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})
                A    B
        sum   21.0  5.0
        min   1.0  2.0
        """
        from databricks.koalas.groupby import GroupBy
        if isinstance(func, list):
            if all((isinstance(f, str) for f in func)):
                func = dict([(column, func) for column in self.columns])
            else:
                raise ValueError('If the given function is a list, it should only contains function names as strings.')
        if not isinstance(func, dict) or not all((is_name_like_value(key) and isinstance(value, str) or (isinstance(value, list) and all((is_name_like_value(v) for v in value)))) for key, value in func.items())):
            raise ValueError('aggfs must be a dict mapping from column name to aggregate functions (string or list of strings).')
        with option_context('compute.default_index_type', 'distributed'):
            kdf = DataFrame(GroupBy._spark_groupby(self, func))
            if LooseVersion(pyspark.__version__) >= LooseVersion('2.4'):
                return kdf.stack().droplevel(0)[list(func.keys())]
            else:
                pdf = kdf._to_internal_pandas().stack()
                pdf.index = pdf.index.droplevel()
                return ks.from_pandas(pdf[list(func.keys())])

    def corr(self, method='pearson'):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        y : DataFrame

        See Also
        --------
        Series.corr

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr('pearson')
                  dogs      cats
        dogs  1.000000 -0.851064
        cats -0.851064  1.000000

        >>> df.corr('spearman')
                  dogs      cats
        dogs  1.000000 -0.948683
        cats -0.948683  1.000000

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * the `method` argument only accepts 'pearson', 'spearman'
        * the data should not contain NaNs. Koalas will return an error.
        * Koalas doesn't support the following argument(s).

            * `min_periods` argument is not supported
        """
        return ks.from_pandas(corr(self, method))

    def iteritems(self):
        """
        Iterator over (column name, Series) pairs.

        Iterates over the DataFrame columns, returning a tuple with
        the column name and the content as a Series.

        Returns
        -------
        label : object
            The column names for the DataFrame being iterated over.
        content : Series
            The column entries belonging to each label, as a Series.

        Examples
        --------
        >>> df = ks.DataFrame({'species': ['bear', 'bear', 'marsupial'],
        ...                    'population': [1864, 22000, 80000]},
        ...                   index=['panda', 'polar', 'koala'],
        ...                   columns=['species', 'population'])
        >>> df
                 species  population
        panda       bear        1864
        polar       bear       22000
        koala  marsupial       80000

        >>> for label, content in df.iteritems():
        ...    print('label:', label)
        ...    print('content:', content.to_string())
        ...
        label: species
        content: panda         bear
        polar       bear
        koala  marsupial
        label: population
        content: panda     1864
        polar    22000
        koala   80000
        """
        return ((label if len(label) > 1 else label[0], self._kser_for(label)) for label in self._internal.column_labels)

    def iterrows(self):
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : pandas.Series
            The data of the row as a Series.

        it : generator
            A generator that iterates over the rows of the frame.

        Notes
        -----

        1. Because ``iterrows`` returns a Series for each row,
           it does **not** preserve dtypes across the rows (dtypes are
           preserved across columns for DataFrames). For example,

           >>> df = ks.DataFrame([[1, 1.5]], columns=['int', 'float'])
           >>> row = next(df.iterrows())[1]
           >>> row
           int      1.0
           float    1.5
           Name: 0, dtype: float64
           >>> print(row['int'].dtype)
           float64
           >>> print(df['int'].dtype)
           int64

           To preserve dtypes while iterating over the rows, it is better
           to use :meth:`itertuples` which returns namedtuples of the values
           and which is generally faster than ``iterrows``.

        2. You should **never modify** something you are iterating over.
           This is not guaranteed to work in all cases. Depending on the
           data types, the iterator returns a copy and not a view, and writing
           to it will have no effect.
        """
        columns = self.columns
        internal_index_columns = self._internal.index_spark_column_names
        internal_data_columns = self._internal.data_spark_column_names

        def extract_kv_from_spark_row(row):
            k = row[internal_index_columns[0]] if len(internal_index_columns) == 1 else tuple((row[c] for c in internal_index_columns))
            v = [row[c] for c in internal_data_columns]
            return (k, v)
        for k, v in map(extract_kv_from_spark_row, self._internal.resolved_copy.spark_frame.toLocalIterator()):
            s = pd.Series(v, index=columns, name=k)
            yield (k, s)

    def itertuples(self, index=True, name='Koalas'):
        """
        Iterate over DataFrame rows as namedtuples.

        Parameters
        ----------
        index : bool, default True
            If True, return the index as the first element of the tuple.
        name : str or None, default "Koalas"
            The name of the returned namedtuples or None to return regular
            tuples.

        Returns
        -------
        iterator
            An object to iterate over namedtuples for each row in the
            DataFrame with the first field possibly being the index and
            following fields being the column values.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series)
            pairs.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        The column names will be renamed to positional names if they are
        invalid Python identifiers, repeated, or start with an underscore.
        On python versions < 3.7 regular tuples are returned for DataFrames
        with a large number of columns (>254).

        Examples
        --------
        >>> df = ks.DataFrame({'num_legs': [4, 2], 'num_wings': [2, 2]},
        ...                   index=['dog', 'hawk'])
        >>> df
              num_legs  num_wings
        dog          4          2
        hawk         2          2

        Viewing the first row

        >>> df.itertuples()
        Koalas(index='dog', num_legs=4, num_wings=2)

        Viewing the first row with index

        >>> df.itertuples(index=True)
        Koalas(index='dog', num_legs=4, num_wings=2)

        Viewing the first row with index and name

        >>> df.itertuples(index=True, name='Animal')
        Animal(index='dog', num_legs=4, num_wings=2)

        Viewing the first row with index and name

        >>> df.itertuples(index=True, name='Animal')
        Animal(index='dog', num_legs=4, num_wings=2)

        Viewing the first row without index

        >>> df.itertuples(index=False)
        Koalas(num_legs=4, num_wings=2)

        Viewing the first row without index and name

        >>> df.itertuples(index=False, name='Animal')
        Animal(num_legs=4, num_wings=2)
        """
        fields = list(self.columns)
        internal_index_columns = self._internal.index_spark_column_names
        internal_data_columns = self._internal.data_spark_column_names

        def extract_kv_from_spark_row(row):
            k = row[internal_index_columns[0]] if len(internal_index_columns) == 1 else tuple((row[c] for c in internal_index_columns))
            v = [row[c] for c in internal_data_columns]
            return (k, v)
        can_return_named_tuples = sys.version_info >= (3, 7) or len(self.columns) + index < 255
        if name is not None and can_return_named_tuples:
            itertuple = namedtuple(name, fields, rename=True)
            for k, v in map(extract_kv_from_spark_row, self._internal.resolved_copy.spark_frame.toLocalIterator()):
                yield itertuple._make(([k] if index else []) + list(v))
        else:
            for k, v in map(extract_kv_from_spark_row, self._internal.resolved_copy.spark_frame.toLocalIterator()):
                yield tuple(([k] if index else []) + list(v))

    def items(self):
        """This is an alias of ``iteritems``."""
        return self.iteritems()

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        """
        Copy object to the system clipboard.

        Write a text representation of object to the system clipboard.
        This can be pasted into Excel, for example.

        .. note:: This method should only be used if the resulting DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        excel : bool, default True
            - True, use the provided separator, writing in a csv format for
              allowing easy pasting into excel.
            - False, write a string representation of the object to the
              clipboard.

        sep : str, default ``'\\t'``
            Field delimiter.
        **kwargs
            These parameters will be passed to DataFrame.to_csv.

        Notes
        -----
        Requirements for your platform.

          - Linux : `xclip`, or `xsel` (with `gtk` or `PyQt4` modules)
          - Windows : none
          - OS X : none

        See Also
        --------
        read_clipboard : Read text from clipboard.

        Examples
        --------
        Copy the contents of a DataFrame to the clipboard.

        >>> df = ks.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])  # doctest: +SKIP
        >>> df.to_clipboard(sep=',')  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # ,A,B,C
        ... # 0,1,2,3
        ... # 1,4,5,6

        We can omit the index by passing the keyword `index` and setting
        it to false.

        >>> df.to_clipboard(sep=',', index=False)  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # A,B,C
        ... # 1,2,3
        ... # 4,5,6
        """
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(kdf._to_internal_pandas(), self.to_clipboard, pd.DataFrame.to_clipboard, args)

    def to_html(self, buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', bold_rows=True, classes=None, escape=True, notebook=False, border=None, table_id=None, render_links=False):
        """
        Render a DataFrame as an HTML table.

        .. note:: This method should only be used if the resulting pandas object is expected
            to be small, as all the data is loaded into the driver's memory. If the input
            is large, set max_rows parameter.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        columns : sequence, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : int, optional
            The minimum width of each column.
        header : bool, optional
            Write out the column names. If a list of strings is given, it
            is assumed to be aliases for the column names
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of NAN to use.
        formatters : list of functions or dict of {str: function}, optional
            Formatter functions to apply to columns’ elements by position or name. The result of each function must be a unicode string. List must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns’ elements if they are
            floats. The result of each function must be a unicode string.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
        bold_rows : bool, default True
            Make the row labels bold in the output.
        classes : str or list or tuple, default None
            CSS class(es) to apply to the resulting html table.
        escape : bool, default True
            Convert the characters <, >, and & to HTML-safe sequences.
        notebook : {True, False}, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            `<table>` tag. Default ``pd.options.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links (only works with pandas 0.24+).

        Returns
        -------
        str (or unicode, depending on data and options)
            String representation of the dataframe.

        See Also
        --------
        to_string : Convert DataFrame to a string.
        """
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self
        return validate_arguments_and_invoke_function(kdf._to_internal_pandas(), self.to_html, pd.DataFrame.to_html, args)

    def to_string(self, buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None):
        """
        Render a DataFrame to a console-friendly tabular output.

        .. note:: This method should only be used if the resulting pandas object is expected
            to be small, as all the data is loaded into the driver's memory. If the input
            is large, set max_rows parameter.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        columns : sequence, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : int, optional
            The minimum width of each column.
        header : bool, optional
            Write out the column names. If a list of strings is given, it
            is assumed to be aliases for the column names
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of NAN to use.
        formatters : list of functions or dict of {str: function}, optional
            Formatter functions to apply to columns’ elements by position or
            name. The result of each function must be a unicode string. List
            must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns’ elements if they are
            floats. The result of each function must be a unicode string.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
        line_width : int, optional
            Width to wrap a line in characters.

        Returns
        -------
        str (or unicode, depending on data and options)
            String representation of the dataframe.

        See Also
        --------
        to_html : Convert DataFrame to HTML.

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, columns=['col1', 'col2'])
        >>> print(df.to_string())
           col1  col2
        0     1    4
        1     2    5
        2     3    6

        >>> print(df.to_string(max_rows=2))
           col1  col2
        0     1    4
        1     2    5
        """
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self
        return validate_arguments_and_invoke_function(kdf._to_internal_pandas(), self.to_string, pd.DataFrame.to_string, args)

    def to_dict(self, orient='dict', into=dict):
        """
        Convert the DataFrame to a dictionary.

        The type of the key-value pairs can be customized with the parameters
        (see below).

        .. note:: This method should only be used if the resulting pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'records', 'index'}
            Determines the type of the values of the dictionary.

            - 'dict' (default) : dict like {column -> {index -> value}}
            - 'list' : dict like {column -> [values]}
            - 'series' : dict like {column -> Series(values)}
            - 'split' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            - 'records' : list like
              [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}

            Abbreviations are allowed. `s` indicates `series` and `sp`
            indicates `split`.

        into : class, default dict
            The collections.abc.Mapping subclass used for all Mappings
            in the return value.  Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        dict, list or collections.abc.Mapping
            Return a collections.abc.Mapping object representing the DataFrame.
            The resulting transformation depends on the `orient` parameter.

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2],
        ...                    'col2': [0.5, 0.75]},
        ...                   index=['row1', 'row2'],
        ...                   columns=['col1', 'col2'])
        >>> df
              col1  col2
        row1     1  0.50
        row2     2  0.75

        >>> df_dict = df.to_dict()
        >>> sorted([(key, sorted(values.items())) for key, values in df_dict.items()])
        [('col1', [('row1', 1), ('row2', 2)]), ('col2', [('row1', 0.5), ('row2', 0.75)])]

        You can specify the return orientation.

        >>> df_dict = df.to_dict('series')
        >>> sorted(df_dict.items())
        [('col1', row1    1
        row2    2
        Name: col1, dtype: int64), ('col2', row1    0.5
        row2    0.75
        Name: col2, dtype: float64)]

        >>> df_dict = df.to_dict('split')
        >>> sorted(df_dict.items())  # doctest: +ELLIPSIS
        [('columns', ['col1', 'col2']), ('data', [[1..., 0.75]]), ('index', ['row1', 'row2'])]

        >>> df_dict = df.to_dict('records')
        >>> [sorted(values.items()) for values in df_dict]  # doctest: +ELLIPSIS
        [[('col1', 1), ('col2', 0.5)], [('col1', 2), ('col2', 0.75)]]

        >>> df_dict = df.to_dict('index')
        >>> sorted([(key, sorted(values.items())) for key, values in df_dict.items()])
        [('row1', [('col1', 1), ('col2', 0.5)]), ('row2', [('col1', 2), ('col2', 0.75)])]

        You can also specify the mapping type.

        >>> from collections import OrderedDict, defaultdict
        >>> df.to_dict(into=OrderedDict)
        OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])), ('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])

        If you want a `defaultdict`, you need to initialize it:

        >>> dd = defaultdict(list)
        >>> df.to_dict('records', into=dd)  # doctest: +ELLIPSIS
        [defaultdict(<class 'list'>, {'col1': [1, 2], 'col2': [0.5, 0.75]}), defaultdict(<class 'list'>, {'col1': [1, 2], 'col2': [0.5, 0.75]})]
        """
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(kdf._to_internal_pandas(), self.to_dict, pd.DataFrame.to_dict, args)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, bold_rows=True, column_format=None, longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None):
        """
        Render an object to a LaTeX tabular environment table.

        Render an object to a tabular environment table. You can splice this into a LaTeX
        document. Requires usepackage{booktabs}.

        .. note:: This method should only be used if the resulting pandas object is expected
            to be small, as all the data is loaded into the driver's memory. If the input
            is large, consider alternative formats.

        Parameters
        ----------
        buf : file descriptor or None
            Buffer to write to. If None, the output is returned as a string.
        columns : list of label, optional
            The subset of columns to write. Writes all columns by default.
        col_space : int, optional
            The minimum width of each column.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given, it is assumed to be aliases
            for the column names.
        index : bool, default True
            Write row names (index).
        na_rep : str, default ‘NaN’
            Missing data representation.
        formatters : list of functions or dict of {str: function}, optional
            Formatter functions to apply to columns’ elements by position or name. The result of
            each function must be a unicode string. List must be of length equal to the number of
            columns.
        float_format : str, optional
            Format string for floating point numbers.
        sparsify : bool, default True
            Set to False for a DataFrame with a hierarchical index to print every multiindex key at
            each row. By default, the value will be read from the pandas config module.
        index_names : bool, default True
            Prints the names of the indexes. By default, the value will be read from the pandas
            config module.
        bold_rows : bool, default True
            Make the row labels bold in the output. By default, the value will be read from the
            pandas config module.
        column_format : str, optional
            The columns format as specified in LaTeX table format e.g. ‘rcl’ for 3 columns. By
            default, ‘l’ will be used for all columns except columns of numbers, which default to
            ‘r’.
        longtable : bool, optional
            By default, the value will be read from the pandas config module. Use a longtable
            environment instead of tabular. Requires adding a usepackage{longtable} to your LaTeX
            preamble.
        escape : bool, optional
            By default, the value will be read from the pandas config module. When set to False
            prevents from escaping LaTeX special characters in column names.
        encoding : str, optional
            A string representing the encoding to use in the output file, defaults to ‘ascii’ on
            Python 2 and ‘utf-8’ on Python 3.
        decimal : str, default ‘.’
            Character recognized as decimal separator, e.g. ‘,’ in Europe.
        multicolumn : bool, default True
            Use multicolumn to enhance MultiIndex columns. The default will be read from the config
            module.
        multicolumn_format : str, default ‘l’
            The alignment for multicolumns, similar to column_format The default will be read from
            the config module.
        multirow : bool, default False
            Use multirow to enhance MultiIndex rows. Requires adding a usepackage{multirow} to your
            LaTeX preamble. Will print centered labels (instead of top-aligned) across the contained
            rows, separating groups via clines. The default will be read from the pandas config
            module.

        Returns
        -------
        str or None
            If buf is None, returns the resulting LateX format as a string. Otherwise returns None.

        See Also
        --------
        to_string : Render a DataFrame to a console-friendly tabular output.
        to_html : Render a DataFrame as an HTML table.

        Examples
        --------
        >>> df = ks.DataFrame({'name': ['Raphael', 'Donatello'],
        ...                    'mask': ['red', 'purple'],
        ...                    'weapon': ['sai', 'bo staff']},
        ...                   columns=['name', 'mask', 'weapon'])
        >>> df
                 name     mask     weapon
        0    Raphael      red         sai
        1  Donatello   purple    bo staff

        Create a new LaTeX table, using the ‘rcl’ format for the columns.

        >>> df.to_latex(index=False, column_format='rcl')
        <BLANKLINE>
        \\begin{tabular}{rcl}
        \\toprule
        name &     mask &     weapon \\\\
        \\midrule
        Raphael &      red &         sai \\\\
        Donatello &   purple &    bo staff \\\\
        \\bottomrule
        \\end{tabular}
        <BLANKLINE>
        """
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(kdf._to_internal_pandas(), self.to_latex, pd.DataFrame.to_latex, args)

    def transpose(self):
        """
        Transpose index and columns.

        Reflect the DataFrame over its main diagonal by writing rows as columns
        and vice-versa. The property :attr:`.T` is an accessor to the method
        :meth:`transpose`.

        .. note:: the current implementation of transpose uses Spark’s Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

                >>> from databricks.koalas.config import option_context
                >>> with option_context(
                ...     'compute.max_rows', 1000):  # doctest: +NORMALIZE_WHITESPACE
                ...     ks.DataFrame({'a': range(1001)}).transpose()
                Traceback (most recent call last):
                  ...
                ValueError: Current DataFrame has more then the given limit 1000 rows.
                Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option'
                to retrieve to retrieve more than 1000 rows. Note that, before changing the
                'compute.max_rows', this operation is considerably expensive.

        Returns
        -------
        DataFrame

        Notes
        -----
        Transposing a DataFrame with mixed dtypes will result in a homogeneous
        DataFrame with the coerced dtype. For instance, if int and float have
        to be placed in same column, it becomes float. If type coercion is not
        possible, it fails.

        Also, note that the values in index should be unique because they become
        unique column names.

        In addition, if Spark 2.3 is used, the types should always be exactly same.

        Examples
        --------
        **Square DataFrame with homogeneous dtype**

        >>> d1 = {'col1': [1, 2], 'col2': [3, 4]}
        >>> df1 = ks.DataFrame(data=d1, columns=['col1', 'col2'])
        >>> df1
           col1  col2
        0     1     3
        1     2     4

        When the dtype is homogeneous in the original DataFrame, we get a
        transposed DataFrame with the same dtype:

        >>> df1.dtypes
        col1    int64
        col2    int64
        dtype: object

        **Non-square DataFrame with mixed dtypes**

        >>> d2 = {'score': [9.5, 8],
        ...       'kids': [0, 0],
        ...       'age': [12, 22]}
        >>> df2 = ks.DataFrame(data=d2, columns=['score', 'kids', 'age'])
        >>> df2
           score  kids  age
        0    9.5     0   12
        1    8.0     0   22

        When the DataFrame has mixed dtypes, we get a transposed DataFrame with
        the coerced dtype:

        >>> df2.dtypes
        score    float64
        kids       int64
        age        int64
        dtype: object
        """
        max_compute_count = get_option('compute.max_rows')
        if max_compute_count is not None:
            pdf = self.head(max_compute_count + 1)._to_internal_pandas()
            if len(pdf) > max_compute_count:
                raise ValueError("Current DataFrame has more then the given limit {} rows. Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option' to retrieve to retrieve more than {} rows. Note that, before changing the 'compute.max_rows', this operation is considerably expensive.".format(max_compute_count, max_compute_count))
            return DataFrame(pdf.transpose())
        pairs = F.explode(F.array(*[F.struct([F.lit(col).alias(SPARK_INDEX_NAME_FORMAT(i)) for i, col in enumerate(label)] + [self._internal.spark_column_for(label).alias('value')]) for label in self._internal.column_labels]))
        exploded_df = self._internal.resolved_copy.spark_frame.withColumn('pairs', pairs).select([F.to_json(F.struct(F.array([scol_for(spark_df, SPARK_INDEX_NAME_FORMAT(i)) for i in range(self._internal.column_labels_level)))).alias(SPARK_DEFAULT_INDEX_NAME)] + [F.col('pairs.*')])
        internal = InternalFrame(spark_frame=exploded_df.groupBy(F.array([scol_for(exploded_df, SPARK_INDEX_NAME_FORMAT(i)) for i in range(self._internal.column_labels_level)]).alias(SPARK_DEFAULT_INDEX_NAME)).pivot('index'))
        transposed_df = internal.spark_frame.agg(F.first(F.col('value')))
        new_data_columns = list(filter(lambda x: x not in [SPARK_DEFAULT_INDEX_NAME], transposed_df.columns))
        column_labels = [None if len(label) == 1 and label[0] is None else label for label in (tuple(json.loads(col)['a']) for col in new_data_columns)]
        internal = InternalFrame(spark_frame=transposed_df, index_spark_columns=[scol_for(transposed_df, col) for col in internal.index_spark_columns], index_names=self._internal.index_names, index_dtypes=self._internal.index_dtypes, column_labels=column_labels, data_spark_columns=[scol_for(transposed_df, col) for col in new_data_columns], column_label_names=self._internal.index_names)
        return DataFrame(internal)

    def apply_batch(self, func, args=(), **kwargs):
        warnings.warn('DataFrame.apply_batch is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.', FutureWarning)
        return self.koalas.apply_batch(func, args=args, **kwargs)
    apply_batch.__doc__ = KoalasFrameMethods.apply_batch.__doc__

    def map_in_pandas(self, func):
        warnings.warn('DataFrame.map_in_pandas is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.', FutureWarning)
        return self.koalas.apply_batch(func)
    map_in_pandas.__doc__ = KoalasFrameMethods.apply_batch.__doc__

    def apply(self, func, axis=0, args=(), **kwargs):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is
        either the DataFrame's index (``axis=0``) or the DataFrame's columns
        (``axis=1``).

        See also `Transform and apply a function
        <https://koalas.readthedocs.io/en/latest/user_guide/transform_apply.html>`_.

        .. note:: when `axis` is 0 or 'index', the `func` is unable to access
            to the whole input series. Koalas internally splits the input series into multiple
            batches and calls `func` with each batch multiple times. Therefore, operations
            such as global aggregations are impossible. See the example below.

            >>> # This case does not return the length of whole series but of the batch internally
            ... # used.
            ... # See https://github.com/python/typing/issues/193
            ... def length(s) -> int:
            ...     return len(s)
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.apply(length, axis=0)  # doctest: +SKIP
            0     83
            1     83
            2     83
            ...
            10    83
            11    83
            dtype: int32

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type as `Series` or scalar value in ``func``,
            for instance, as below:

            >>> def square(s) -> ks.Series[np.int32]:
            ...     return s ** 2

            Koalas uses return type hint and does not try to infer the type.

        .. note:: the `func` should specify a scalar or a series as its type hints when `axis` is 0 or 'index';
            however, the return type is specified as `Series` when `axis` is 1.

            >>> def plus_one(x) -> ks.DataFrame[float, float]:
            ...     return x + 1

            >>> def plus_one(x) -> ks.DataFrame['a': float, 'b': float]:
            ...     return x + 1

        Parameters
        ----------
        func : function
            Function to use for transforming the data. It must work when pandas Series
            is passed.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Can only be set to 0 at the moment.
        args : tuple
            Positional arguments to pass to `func` in addition to the
            array/series.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        Series or DataFrame
            Result of applying ``func`` along the given axis of the
            DataFrame.

        See Also
        --------
        DataFrame.applymap : For elementwise operations.
        DataFrame.aggregate : Only perform aggregating type operations.
        DataFrame.transform : Only perform transforming type operations.
        Series.apply : The equivalent function for Series.

        Examples
        --------
        >>> df = ks.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        >>> df
           A  B
        0  4  9
        1  4  9
        2  4  9

        Using a numpy universal function (in this case the same as
        ``np.sqrt(df)``):

        >>> def sqrt(x) -> ks.Series[np.int32]:
        ...     return np.sqrt(x)
        ...
        >>> df.apply(sqrt, axis=0)
             A    B
        0  2.0  3.0
        1  2.0  3.0
        2  2.0  3.0

        You can omit the type hint and let Koalas infer its type.

        >>> df.apply(np.sqrt, axis=0)
             A    B
        0  2.0  3.0
        1  2.0  3.0
        2  2.0  3.0

        When `axis` is 1 or 'columns', it applies the function for each row.

        >>> def summation(x) -> np.int64:
        ...     return np.sum(x)
        ...
        >>> df.apply(summation, axis=1)
        0    13
        1    13
        2    13
        dtype: int64

        Likewise, you can omit the type hint and let Koalas infer its type.

        >>> df.apply(np.sum, axis=1)
        0    13
        1    13
        2    13
        dtype: int64

        >>> df.apply(max, axis=1)
        0    9
        1    9
        2    9
        dtype: int64

        Returning a list-like will result in a Series

        >>> df.apply(lambda x: [1, 2], axis=1)
        0    [1, 2]
        1    [1, 2]
        2    [1, 2]
        dtype: object

        In order to specify the types when `axis` is '1', it should use DataFrame[...]
        annotation. In this case, the column names are automatically generated.

        >>> def identify(x) -> ks.DataFrame['A': np.int64, 'B': np.int64]:
        ...     return x
        ...
        >>> df.apply(identify, axis=1)
           A  B
        0  4  9
        1  4  9
        2  4  9

        You can also specify extra arguments.

        >>> def plus_two(a, b, c) -> ks.DataFrame[np.int64, np.int64]:
        ...     return a + b + c
        ...
        >>> df.apply(plus_two, axis=1, args=(1,), c=3)
           A  B
        0  8  13
        1  8  13
        2  8  13
        """
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.series import Series
        if not isinstance(func, types.FunctionType):
            assert callable(func), 'the first argument should be a callable function.'
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = validate_axis(axis)
        should_return_series = False
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get('return', None)
        should_infer_schema = return_sig is None
        if should_infer_schema:
            limit = get_option('compute.shortcut_limit')
            pdf = self.head(limit + 1)._to_internal_pandas()
            if len(pdf) > limit:
                return Series(pdf.apply(func, axis=axis, args=args, **kwargs))
            kdf = DataFrame(pdf)
            if axis == 0:
                applied = []
                for input_label, output_label in zip(self._internal.column_labels, kdf._internal.column_labels):
                    kser = self._kser_for(input_label)
                    dtype = kdf._internal.dtype_for(output_label)
                    return_schema = force_decimal_precision_scale(as_nullable_spark_type(kdf._internal.to_internal_spark_frame.schema))
                    applied.append(kser.koalas._transform_batch(func=lambda c: func(c, *args, **kwargs), return_type=SeriesType(dtype, return_schema)))
                internal = self._internal.with_new_columns(applied, data_dtypes=kdf._internal.data_dtypes)
                return DataFrame(internal)
            else:
                return self._apply_series_op(lambda kser: kser.koalas.transform_batch(func, *args, **kwargs))
        else:
            return_sig = infer_return_type(func)
            require_index_axis = isinstance(return_sig, SeriesType)
            require_column_axis = isinstance(return_sig, DataFrameType)
            if require_index_axis:
                if axis != 0:
                    raise TypeError("The given function should specify a scalar or a series as its type hints when axis is 0 or 'index'; however, the return type was %s" % return_sig)
                return_schema = cast(SeriesType, return_sig).spark_type
                fields_types = zip(self._internal.column_labels, [return_schema] * len(self._internal.column_labels))
                return_schema = StructType([StructField(c, t) for c, t in fields_types])
                data_dtypes = [cast(SeriesType, return_sig).dtype] * len(self._internal.column_labels)
            elif require_column_axis:
                if axis != 1:
                    raise TypeError("The given function should specify a scalar or a frame as its type hints when axis is 1 or 'column'; however, the return type was %s" % return_sig)
                return_schema = cast(DataFrameType, return_sig).spark_type
                data_dtypes = cast(DataFrameType, return_sig).dtypes
            else:
                should_return_series = True
                return_schema = cast(ScalarType, return_sig).spark_type
                return_schema = StructType([StructField(SPARK_DEFAULT_SERIES_NAME, return_schema)])
                data_dtypes = [cast(ScalarType, return_sig).dtype]
                column_labels = [None]
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(self, apply_func, return_schema, retain_index=True)
                sdf = self._internal.to_internal_spark_frame.mapInPandas(lambda iterator: map(output_func, iterator), schema=return_schema)
            else:
                sdf = GroupBy._spark_group_map_apply(self, apply_func, (F.spark_partition_id(),), return_schema, retain_index=True)
            internal = self._internal.with_new_sdf(sdf)
        result = DataFrame(internal)
        if should_return_series:
            return first_series(result)
        else:
            return result

    def transform(self, func, axis=0, *args, **kwargs):
        """
        Call ``func`` on self producing a Series with transformed values
        and that has the same length as its input.

        See also `Transform and apply a function
        <https://koalas.readthedocs.io/en/latest/user_guide/transform_apply.html>`_.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> ks.Series[np.int32]:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

        .. note:: the series within ``func`` is actually multiple pandas series as the
            segments of the whole Koalas series; therefore, the length of each series
            is not guaranteed. As an example, an aggregation against each series
            does work as a global aggregation but an aggregation of each segment. See
            below:

            >>> def func(x) -> ks.Series[np.int32]:
            ...     return x + sum(x)

        Parameters
        ----------
        func : function
            Function to use for transforming the data. It must work when pandas Series
            is passed.
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        DataFrame
            A DataFrame that must have the same length as self.

        Raises
        ------
        Exception : If the returned DataFrame has a different length than self.

        See Also
        --------
        DataFrame.apply : Invoke function on DataFrame.
        DataFrame.aggregate : Only perform aggregating type operations.
        DataFrame.apply : Invoke function on DataFrame.
        Series.transform : The equivalent function for Series.

        Examples
        --------
        >>> df = ks.DataFrame({'A': range(3), 'B': range(1, 4)}, columns=['A', 'B'])
        >>> df
           A  B
        0  0  1
        1  1  2
        2  2  3

        >>> def square(x) -> ks.Series[np.int32]:
        ...     return x ** 2
        >>> df.transform(square)
           A  B
        0  0  1
        1  1  4
        2  4  9

        You can omit the type hint and let Koalas infer its type.

        >>> df.transform(lambda x: x ** 2)
           A  B
        0  0  1
        1  1  4
        2  4  9

        For multi-index columns:

        >>> df.columns = [('X', 'A'), ('X', 'B'), ('Y', 'C')]
        >>> df.transform(square)  # doctest: +NORMALIZE_WHITESPACE
           X
           A  B
        0  0  1
        1  1  4
        2  4  9

        >>> (df * -1).transform(abs)  # doctest: +NORMALIZE_WHITESPACE
           X
           A  B
        0  0  1
        1  1  2
        2  2  3

        You can also specify extra arguments.

        >>> def calculation(x) -> ks.Series[int]:
        ...     return x ** 2 + 1
        >>> df.transform(calculation, x=10, y=20)  # doctest: +NORMALIZE_WHITESPACE
           X
           A  B
        0  1  1
        1  1  21
        2  5  41
        """
        if not isinstance(func, types.FunctionType):
            assert callable(func), 'the first argument should be a callable function.'
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get('return', None)
        should_infer_schema = return_sig is None
        if should_infer_schema:
            limit = get_option('compute.shortcut_limit')
            pdf = self.head(limit + 1)._to_internal_pandas()
            if len(pdf) > limit:
                return Series(pdf.transform(func, axis=axis, *args, **kwargs))
            kdf = DataFrame(pdf)
            if axis == 0:
                applied = []
                for input_label, output_label in zip(self._internal.column_labels, kdf._internal.column_labels):
                    kser = self._kser_for(input_label)
                    dtype = kdf._internal.dtype_for(output_label)
                    return_schema = force_decimal_precision_scale(as_nullable_spark_type(kdf._internal.to_internal_spark_frame.schema))
                    applied.append(kser.koalas._transform_batch(func=lambda c: func(c, *args, **kwargs), return_type=SeriesType(dtype, return_schema)))
                internal = self._internal.with_new_columns(applied, data_dtypes=kdf._internal.data_dtypes)
                return DataFrame(internal)
            else:
                return self._apply_series_op(lambda kser: kser.koalas.transform_batch(func, *args, **kwargs))
        else:
            return_sig = infer_return_type(func)
            require_index_axis = isinstance(return_sig, SeriesType)
            require_column_axis = isinstance(return_sig, DataFrameType)
            if require_index_axis:
                if axis != 0:
                    raise TypeError("The given function should specify a scalar or a series as its type hints when axis is 0 or 'index'; however, the return type was %s" % return_sig)
                return_schema = cast(SeriesType, return_sig).spark_type
                fields_types = zip(self._internal.column_labels, [return_schema] * len(self._internal.column_labels))
                return_schema = StructType([StructField(c, t) for c, t in fields_types])
                data_dtypes = [cast(SeriesType, return_sig).dtype] * len(self._internal.column_labels)
            elif require_column_axis:
                if axis != 1:
                    raise TypeError("The given function should specify a scalar or a frame as its type hints when axis is 1 or 'column'; however, the return type was %s" % return_sig)
                return_schema = cast(DataFrameType, return_sig).spark_type
                data_dtypes = cast(DataFrameType, return_sig).dtypes
            else:
                should_return_series = True
                return_schema = cast(ScalarType, return_sig).spark_type
                return_schema = StructType([StructField(SPARK_DEFAULT_SERIES_NAME, return_schema)])
                data_dtypes = [cast(ScalarType, return_sig).dtype]
                column_labels = [None]
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(self, apply_func, return_schema, retain_index=True)
                sdf = self._internal.to_internal_spark_frame.mapInPandas(lambda iterator: map(output_func, iterator), schema=return_schema)
            else:
                sdf = GroupBy._spark_group_map_apply(self, apply_func, (F.spark_partition_id(),), return_schema, retain_index=True)
            internal = self._internal.with_new_sdf(sdf)
        result = DataFrame(internal)
        if should_return_series:
            return first_series(result)
        else:
            return result

    def pop(self, item):
        """
        Return item and drop from frame. Raise KeyError if not found.

        Parameters
        ----------
        item : str
            Label of column to be popped.

        Returns
        -------
        Series

        Examples
        --------
        >>> df = ks.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey','mammal', np.nan)],
        ...                   columns=('name', 'class', 'max_speed'))

        >>> df
                 name   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        >>> df.pop('class')
        0      bird
        1      bird
        2    mammal
        3    mammal
        Name: class, dtype: object

        >>> df
                 name  max_speed
        0  falcon      389.0
        1  parrot       24.0
        2    lion       80.5
        3  monkey        NaN

        Also support for MultiIndex

        >>> df = ks.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey','mammal', np.nan)],
        ...                   columns=('name', 'class', 'max_speed'))
        >>> columns = [('a', 'name'), ('a', 'class'), ('b', 'max_speed')]
        >>> df.columns = pd.MultiIndex.from_tuples(columns)
        >>> df
                a                 b
           name   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        >>> df.pop('a')
             name   class
        0  falcon    bird
        1  parrot    bird
        2    lion  mammal
        3  monkey  mammal

        >>> df
                b
           max_speed
        0      389.0
        1       24.0
        2       80.5
        3        NaN
        """
        result = self[item]
        self._update_internal_frame(self.drop(item)._internal)
        return result

    def xs(self, key, axis=0, level=None):
        """
        Return cross-section from the DataFrame.

        This method takes a `key` argument to select data at a particular
        level of a MultiIndex.

        Parameters
        ----------
        key : label or tuple of label
            Label contained in the index, or partially in a MultiIndex.
        axis : 0 or 'index', default 0
            Axis to retrieve cross-section on.
            currently only support 0 or 'index'
        level : object, defaults to first n levels (n=1 or len(key))
            In case of a key partially contained in a MultiIndex, indicate
            which levels are used. Levels can be referred by label or position.

        Returns
        -------
        DataFrame or Series
            Cross-section from the original DataFrame
            corresponding to the selected index levels.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        DataFrame.loc : Access a group of rows and columns
            by label(s) or a boolean array.
        DataFrame.iloc : Purely integer-location based indexing
            for selection by position.

        Examples
        --------
        >>> df = ks.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
        ...                   index=['falcon', 'dog'],
        ...                   columns=['num_legs', 'num_wings'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        Get values at specified index

        >>> df.xs('falcon')
                num_legs  num_wings
        falcon         2          2

        Get values at specified index

        >>> df.xs('dog')
                num_legs  num_wings
        dog            4          0

        Get values at specified index

        >>> df.xs(('falcon', 'dog'))
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        Get values at specified index

        >>> df.xs(('falcon', 'dog'), level=0)
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        Get values at specified index

        >>> df.xs(('falcon', 'dog'), level=1)
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        Get values at specified index

        >>> df.xs(('falcon', 'dog'), level=0, axis=1)
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        Get values at specified index

        >>> df.xs(('falcon', 'dog'), level=1, axis=1)
                num_legs  num_wings
        falcon         2          2
        dog            4          0
        """
        from databricks.koalas.series import Series
        if not is_name_like_value(key):
            raise ValueError('key should be a scalar value or tuple that contains scalar values')
        if level is not None and is_name_like_tuple(key):
            raise KeyError('Level {} not found'.format(key))
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if not is_name_like_tuple(key):
            key = (key,)
        if len(key) > self._internal.index_level:
            raise KeyError('Key length ({}) exceeds index depth ({})'.format(len(key), self._internal.index_level))
        if level is None:
            level = 0
        rows = [self._internal.index_spark_columns[l] == index for lvl, index in enumerate(key, level)]
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
            index_spark_columns = internal.index_spark_columns[:level] + internal.index_spark_columns[level + len(key):]
            index_names = internal.index_names[:level] + internal.index_names[level + len(key):]
            index_dtypes = internal.index_dtypes[:level] + internal.index_dtypes[level + len(key):]
            internal = internal.copy(index_spark_columns=index_spark_columns, index_names=index_names, index_dtypes=index_dtypes).resolved_copy
            return DataFrame(internal)

    def between_time(self, start_time, end_time, include_start=True, axis=0):
        """
        Select values between particular times of the day (e.g., 9:00-9:30 AM).

        By setting ``start_time`` to be later than ``end_time``,
        you can get the times that are *not* between the two times.

        Parameters
        ----------
        start_time : datetime.time or str
            Initial time as a time filter limit.
        end_time : datetime.time or str
            End time as a time filter limit.
        include_start : bool, default True
            Whether the start time needs to be included in the result.
        axis : {0 or 'index'}, default 0
            Axis to retrieve cross-section on.

        Returns
        -------
        DataFrame
            Data from the original object filtered to the specified dates range.

        Raises
        ------
        TypeError
            If the index is not a :class:`DatetimeIndex`

        See Also
        --------
        at_time : Select values at a particular time of the day.
        first : Select initial periods of time series data based on a date offset.
        DatetimeIndex.indexer_between_time : Get just the index locations for
            values between particular times of the day.

        Examples
        --------
        >>> index = pd.date_range('2018-04-09', periods=4, freq='1D20min')
        >>> kdf = ks.DataFrame({'A': [1, 2, 3, 4]}, index=index)
        >>> kdf
                             A
        2018-04-09 00:00:00  1
        2018-04-11 00:20:00  2
        2018-04-13 00:40:00  3
        2018-04-15 01:00:00  4

        Get the rows for the last 3 days:

        >>> kdf.between_time('0:15', '0:45')
                             A
        2018-04-11 00:20:00  2
        2018-04-13 00:40:00  3

        Note how shuffling of the objects does not change the result.

        >>> kser2 = kser.reindex([1, 0, 2, 3])
        >>> kdf.between_time('0:45', '0:15')
                             A
        2018-04-09 00:00:00  1
        2018-04-11 00:20:00  2
        2018-04-13 00:40:00  3
        """
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError("'between_time' only supports a DatetimeIndex")
        kdf = self.copy()
        kdf.index.name = verify_temp_column_name(kdf, '__index_name__')
        return_types = [kdf.index.dtype] + list(kdf.dtypes)

        def pandas_between_time(pdf):
            return pdf.between_time(start_time, end_time, include_start, axis).reset_index()
        with option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_between_time)
        return DataFrame(self._internal.copy(spark_frame=kdf._internal.spark_frame, index_spark_columns=kdf._internal.data_spark_columns[:1], data_spark_columns=kdf._internal.data_spark_columns[1:]))

    def at_time(self, time, asof=False, axis=0):
        """
        Select values at particular time of day (e.g., 9:30 AM).

        When having a DataFrame with dates as index, this function can
        select the rows at a particular time of day.

        Parameters
        ----------
        time : datetime.time or str
        axis : {0 or ‘index’}, default 0

        Returns
        -------
        DataFrame

        Raises
        ------
        TypeError
            If the index is not a :class:`DatetimeIndex`

        Examples
        --------

        >>> index = pd.date_range('2018-04-09', periods=4, freq='12H')
        >>> kdf = ks.DataFrame({'A': [1, 2, 3, 4]}, index=index)
        >>> kdf
                             A
        2018-04-09 00:00:00  1
        2018-04-09 12:00:00  2
        2018-04-10 00:00:00  3
        2018-04-10 12:00:00  4

        Get the rows at 12:00:

        >>> kdf.at_time('12:00')
                             A
        2018-04-09 12:00:00  2
        2018-04-10 12:00:00  4

        Note how shuffling of the objects does not change the result.

        >>> kser2 = kser.reindex([1, 0, 2, 3])
        >>> kdf.at_time('12:00')
                             A
        2018-04-09 12:00:00  2
        2018-04-10 12:00:00  4
        """
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError("'at_time' only supports a DatetimeIndex")
        kdf = self.copy()
        kdf.index.name = verify_temp_column_name(kdf, '__index_name__')
        if LooseVersion(pd.__version__) < LooseVersion('0.24'):

            def pandas_at_time(pdf):
                return pdf.at_time(time, asof, axis).reset_index()
        else:

            def pandas_at_time(pdf):
                return pdf.at_time(time, asof, axis).reset_index()
        with option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_at_time)
        return DataFrame(self._internal.copy(spark_frame=kdf._internal.spark_frame, index_spark_columns=kdf._internal.data_spark_columns[:1], data_spark_columns=kdf._internal.data_spark_columns[1:]))

    def where(self, cond, other=np.nan):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : boolean DataFrame
            Where cond is True, keep the original value. Where False,
            replace with corresponding value from other.
        other : scalar, DataFrame
            Entries where cond is False are replaced with corresponding value from other.

        Returns
        -------
        DataFrame

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> df1 = ks.DataFrame({'A': [0, 1, 2, 3, 4], 'B':[100, 200, 300, 400, 500]})
        >>> df2 = ks.DataFrame({'A': [0, -1, -2, -3, -4], 'B':[-100, -200, -300, -400, -500]})
        >>> df1
           A    B
        0  0  100
        1  1  200
        2  2  300
        3  3  400
        4  4  500
        >>> df2
           A    B
        0  0 -100
        1 -1 -200
        2 -2 -300
        3 -3 -400
        4 -4 -500

        Replace all NaN elements with 0s.

        >>> df1.where(df1 > 0).sort_index()
             A      B
        0  0.0  100.0
        1  1.0  200.0
        2  2.0  300.0
        3  3.0  400.0
        4  4.0  500.0

        We can also propagate non-null values forward or backward.

        >>> df1.where(df1 > 1, 10).sort_index()
             A      B
        0  10.0  100.0
        1  10.0  200.0
        2  2.0  300.0
        3  3.0  400.0
        4  4.0  500.0

        Replace all NaN elements in column 'A', 'B', 'C', and 'D', with 0, 1,
        2, and 3 respectively.

        >>> values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        >>> df1.where(df1 > 1, df2).sort_index()
           A   B   C   D
        0  0  100  2  3
        1  1  200  2  3
        2  2  300  2  3
        3  3  400  2  3
        4  4  500  2  3

        When the column name of cond is different from self, it treats all values are False

        >>> cond = ks.DataFrame({'C': [0, -1, -2, -3, -4], 'D':[4, 3, 2, 1, 0]}) % 3 == 0
        >>> cond
               C      D
        0   True  False
        1  False   True
        2  False  False
        3   True  False
        4  False   True

        >>> df1.where(cond).sort_index()
             A      B      C      D
        0  NaN  NaN  NaN  NaN
        1  NaN  NaN  NaN  NaN
        2  NaN  NaN  NaN  NaN
        3  NaN  NaN  NaN  NaN
        4  NaN  NaN  NaN  NaN

        When the type of cond is Series, it just check boolean regardless of column name

        >>> cond = ks.Series([1, 2]) > 1
        >>> cond
        0    False
        1     True
        dtype: bool

        >>> df1.where(cond).sort_index()
             A      B      C      D
        0  NaN  NaN  NaN  NaN
        1  1.0  200.0  NaN  NaN
        2  NaN  NaN  NaN  NaN
        3  NaN  NaN  NaN  NaN
        4  NaN  NaN  NaN  NaN

        >>> reset_option("compute.ops_on_diff_frames")
        """
        from databricks.koalas.series import Series
        tmp_cond_col_name = '__tmp_cond_col_{}__'.format
        tmp_other_col_name = '__tmp_other_col_{}__'.format
        kdf = self.copy()
        tmp_cond_col_names = [tmp_cond_col_name(name_like_string(label)) for label in self._internal.column_labels]
        if isinstance(cond, DataFrame):
            cond = cond[[(cond._internal.spark_column_for(label) if label in cond._internal.column_labels else F.lit(False)).alias(name) for label, name in zip(self._internal.column_labels, tmp_cond_col_names)]]
            kdf[tmp_cond_col_names] = cond
        elif isinstance(cond, Series):
            cond = cond.to_frame()
            cond = cond[[cond._internal.data_spark_columns[0].alias(name) for name in tmp_cond_col_names]]
            kdf[tmp_cond_col_names] = cond
        else:
            raise ValueError('type of cond must be a DataFrame or Series')
        tmp_other_col_names = [tmp_other_col_name(name_like_string(label)) for label in self._internal.column_labels]
        if isinstance(other, DataFrame):
            other = other[[(other._internal.spark_column_for(label) if label in other._internal.column_labels else F.lit(np.nan)).alias(name) for label, name in zip(self._internal.column_labels, tmp_other_col_names)]]
            kdf[tmp_other_col_names] = other
        elif isinstance(other, Series):
            other = other.to_frame()
            other = other[[other._internal.data_spark_columns[0].alias(name) for name in tmp_other_col_names]]
            kdf[tmp_other_col_names] = other
        else:
            for label in self._internal.column_labels:
                kdf[tmp_other_col_name(name_like_string(label))] = other
        data_spark_columns = []
        for label in self._internal.column_labels:
            data_spark_columns.append(F.when(kdf[tmp_cond_col_name(name_like_string(label))].spark.column, kdf._internal.spark_column_for(label)).otherwise(kdf[tmp_other_col_name(name_like_string(label))].spark.column).alias(kdf._internal.spark_column_name_for(label)))
        return DataFrame(kdf._internal.with_new_columns(data_spark_columns, column_labels=self._internal.column_labels))

    def mask(self, cond, other=np.nan):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : boolean DataFrame
            Where cond is True, replace with corresponding value from other.
            Where False, keep the original value.
        other : scalar, DataFrame
            Entries where cond is True are replaced with corresponding value from other.

        Returns
        -------
        DataFrame

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> df1 = ks.DataFrame({'A': [0, 1, 2, 3, 4], 'B':[100, 200, 300, 400, 500]})
        >>> df2 = ks.DataFrame({'A': [0, -1, -2, -3, -4], 'B':[-100, -200, -300, -400, -500]})
        >>> df1
           A    B
        0  0  100
        1  1  200
        2  2  300
        3  3  400
        4  4  500
        >>> df2
           A    B
        0  0 -100
        1 -1 -200
        2 -2 -300
        3 -3 -400
        4 -4 -500

        Replace all NaN elements with 0s.

        >>> df1.mask(df1 > 0).sort_index()
             A      B
        0  0.0  100.0
        1  1.0  200.0
        2  2.0  300.0
        3  3.0  400.0
        4  4.0  500.0

        We can also propagate non-null values forward or backward.

        >>> df1.mask(df1 > 1, 10).sort_index()
             A      B
        0  10.0  100.0
        1  10.0  200.0
        2  2.0  300.0
        3  3.0  400.0
        4  4.0  500.0

        Replace all NaN elements in column 'A', 'B', 'C', and 'D', with 0, 1,
        2, and 3 respectively.

        >>> values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        >>> df1.mask(df1 > 1, df2).sort_index()
           A   B   C   D
        0  0  100  2  3
        1  1  200  2  3
        2  2  300  2  3
        3  3  400  2  3
        4  4  500  2  3

        >>> reset_option("compute.ops_on_diff_frames")
        """
        from databricks.koalas.series import Series
        if not isinstance(cond, (DataFrame, Series)):
            raise ValueError('type of cond must be a DataFrame or Series')
        tmp_cond_col_name = '__tmp_cond_col_{}__'.format
        tmp_other_col_name = '__tmp_other_col_{}__'.format
        kdf = self.copy()
        tmp_cond_col_names = [tmp_cond_col_name(name_like_string(label)) for label in self._internal.column_labels]
        if isinstance(cond, DataFrame):
            cond = cond[[(cond._internal.spark_column_for(label) if label in cond._internal.column_labels else F.lit(False)).alias(name) for label, name in zip(self._internal.column_labels, tmp_cond_col_names)]]
            kdf[tmp_cond_col_names] = cond
        elif isinstance(cond, Series):
            cond = cond.to_frame()
            cond = cond[[cond._internal.data_spark_columns[0].alias(name) for name in tmp_cond_col_names]]
            kdf[tmp_cond_col_names] = cond
        else:
            raise ValueError('type of cond must be a DataFrame or Series')
        tmp_other_col_names = [tmp_other_col_name(name_like_string(label)) for label in self._internal.column_labels]
        if isinstance(other, DataFrame):
            other = other[[(other._internal.spark_column_for(label) if label in other._internal.column_labels else F.lit(np.nan)).alias(name) for label, name in zip(self._internal.column_labels, tmp_other_col_names)]]
            kdf[tmp_other_col_names] = other
        elif isinstance(other, Series):
            other = other.to_frame()
            other = other[[other._internal.data_spark_columns[0].alias(name) for name in tmp_other_col_names]]
            kdf[tmp_other_col_names] = other
        else:
            for label in self._internal.column_labels:
                kdf[tmp_other_col_name(name_like_string(label))] = other
        data_spark_columns = []
        for label in self._internal.column_labels:
            data_spark_columns.append(F.when(kdf[tmp_cond_col_name(name_like_string(label))].spark.column, kdf._internal.spark_column_for(label)).otherwise(kdf[tmp_other_col_name(name_like_string(label))].spark.column).alias(kdf._internal.spark_column_name_for(label)))
        return DataFrame(kdf._internal.with_new_columns(data_spark_columns, column_labels=self._internal.column_labels))

    @property
    def index(self):
        """The index (row labels) Column of the DataFrame.

        Currently not supported when the DataFrame has no index.

        See Also
        --------
        Index
        """
        from databricks.koalas.indexes.base import Index
        return Index._new_instance(self)

    @property
    def empty(self):
        """
        Returns true if the current DataFrame is empty. Otherwise, returns false.

        Examples
        --------
        >>> ks.range(10).empty
        False

        >>> ks.range(0).empty
        True

        >>> ks.DataFrame({}, index=list('abc')).empty
        True
        """
        return len(self._internal.column_labels) == 0 or self._internal.resolved_copy.spark_frame.rdd.isEmpty()

    @property
    def style(self):
        """
        Property returning a Styler object containing methods for
        building a styled HTML representation for the DataFrame.

        .. note:: currently it collects top 1000 rows and return its
            pandas `pandas.io.formats.style.Styler` instance.

        Examples
        --------
        >>> ks.range(1001).style  # doctest: +ELLIPSIS
        <pandas.io.formats.style.Styler object at ...>
        """
        max_results = get_option('compute.max_rows')
        pdf = self.head(max_results + 1)._to_internal_pandas()
        if len(pdf) > max_results:
            warnings.warn("'style' property will only use top %s rows." % max_results, UserWarning)
        return pdf.head(max_results).style

    def set_index(self, keys, drop=True, append=False, inplace=False):
        """Set the DataFrame index (row labels) using one or more existing columns.

        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index can replace the
        existing index or expand on it.

        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single array of
            the same length as the calling DataFrame, or a list containing an
            arbitrary combination of column keys and arrays. Here, "array"
            encompasses :class:`Series`, :class:`Index` and ``np.ndarray``.
        drop : bool, default True
            Delete columns to be used as the new index.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            If True, do operation inplace and return None.

        Returns
        -------
        DataFrame
            Changed row labels.

        See Also
        --------
        DataFrame.reset_index : Opposite of set_index.

        Examples
        --------
        >>> df = ks.DataFrame({'month': [1, 4, 7, 10],
        ...                    'year': [2012, 2014, 2013, 2014],
        ...                    'sale': [55, 40, 84, 31]},
        ...                   columns=['month', 'year', 'sale'])
        >>> df
           month  year  sale
        0      1  2012    55
        1      4  2014    40
        2      7  2013    84
        3     10  2014    31

        Set the index to become the 'month' column:

        >>> df.set_index('month')  # doctest: +NORMALIZE_WHITESPACE
               year  sale
        month
        1      2012    55
        4      2014    40
        7      2013    84
        10     2014    31

        Create a MultiIndex using columns 'year' and 'month':

        >>> df.set_index(['year', 'month'])  # doctest: +NORMALIZE_WHITESPACE
                    sale
        year  month
        2012  1      55
        2014  4      40
        2013  7      84
        2014  10     31
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if is_name_like_tuple(keys):
            keys = [keys]
        elif is_name_like_value(keys):
            keys = [(keys,)]
        else:
            keys = [key if is_name_like_tuple(key) else (key,) for key in keys]
        columns = set(self._internal.column_labels)
        for key in keys:
            if key not in columns:
                raise KeyError(name_like_string(key))
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
        internal = self._internal.copy(index_spark_columns=index_spark_columns, index_names=index_names, index_dtypes=index_dtypes, column_labels=column_labels, data_spark_columns=[self._internal.spark_column_for(label) for label in column_labels], data_dtypes=[self._internal.dtype_for(label) for label in column_labels])
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        """Reset the index, or a level of it.

        For DataFrame with multi-level index, return new DataFrame with labeling information in
        the columns under the index names, defaulting to 'level_0', 'level_1', etc. if any are None.
        For a standard index, the index name will be used (if set), otherwise a default 'index' or
        'level_0' (if 'index' is already taken) will be used.

        Parameters
        ----------
        level : int, str, tuple, or list, default None
            Only remove the given levels from the index. Removes all levels by default.
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).
        col_level : int or str, default 0
            If the columns have multiple levels, determines which level the
            labels are inserted into. By default it is inserted into the first
            level.
        col_fill : object, default ''
            If the columns have multiple levels, determines how the other levels
            are named. If None then the index name is repeated.

        Returns
        -------
        DataFrame
            DataFrame with the new index.

        See Also
        --------
        DataFrame.set_index : Opposite of reset_index.

        Examples
        --------
        >>> df = ks.DataFrame([('bird', 389.0),
        ...                    ('bird', 24.0),
        ...                    ('mammal', 80.5),
        ...                    ('mammal', np.nan)],
        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
        ...                   columns=('class', 'max_speed'))
        >>> df
                 class  max_speed
        falcon    bird      389.0
        parrot    bird       24.0
        lion    mammal       80.5
        monkey  mammal        NaN

        When we reset the index, the old index is added as a column. Unlike pandas, Koalas
        does not automatically add a sequential index. The following 0, 1, 2, 3 are only
        there when we display the DataFrame.

        >>> df.reset_index()
            index   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        We can use the `drop` parameter to avoid the old index being added as
        a column:

        >>> df.reset_index(drop=True)
            class  max_speed
        0    bird      389.0
        1    bird       24.0
        2  mammal       80.5
        3  mammal        NaN

        You can also use `reset_index` with `MultiIndex`.

        >>> index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
        ...                                    ('bird', 'parrot'),
        ...                                    ('mammal', 'lion'),
        ...                                    ('mammal', 'monkey')],
        ...                                   names=['class', 'name'])
        >>> columns = pd.MultiIndex.from_tuples([('speed', 'max'),
        ...                                      ('species', 'type')])
        >>> df = ks.DataFrame([(389.0, 'fly'),
        ...                    ( 24.0, 'fly'),
        ...                    ( 80.5, 'run'),
        ...                    (np.nan, 'jump')],
        ...                   index=index,
        ...                   columns=columns)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                       speed species
                         max     type
        (bird, falcon)  389.0     fly
        (bird, parrot)   24.0     fly
        (mammal, lion)   80.5     run
        (mammal, monkey)  NaN     jump

        If the index has multiple levels, we can reset a subset of them:

        >>> df.reset_index(level='class')  # doctest: +NORMALIZE_WHITESPACE
                           speed species
        (bird, falcon)  389.0     fly
        (bird, parrot)   24.0     fly
        (mammal, lion)   80.5     run
        (mammal, monkey)  NaN     jump

        If we are not dropping the index, by default, it is placed in the top
        level. We can place it in another level:

        >>> df.reset_index(level='class', col_level=1)  # doctest: +NORMALIZE_WHITESPACE
                       speed species
        (bird, falcon)  389.0     fly
        (bird, parrot)   24.0     fly
        (mammal, lion)   80.5     run
        (mammal, monkey)  NaN     jump

        When the index is inserted under another level, we can specify under
        which one with the parameter `col_fill`:

        >>> df.reset_index(level='class', col_level=1,
        ...                col_fill='species')  # doctest: +NORMALIZE_WHITESPACE
                       species  speed species
        (bird, falcon)  fly     389.0     fly
        (bird, parrot)  fly     24.0     fly
        (mammal, lion)  run     80.5     run
        (mammal, monkey) jump     0.0     jump

        If we specify a nonexistent level for `col_fill`, it is created:

        >>> df.reset_index(level='class', col_level=1,
        ...                col_fill='genus')  # doctest: +NORMALIZE_WHITESPACE
                       genus  speed species
        (bird, falcon)  bird   389.0     fly
        (bird, parrot)  bird   24.0     fly
        (mammal, lion)  mammal  80.5     run
        (mammal, monkey) mammal  NaN     jump
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        multi_index = self._internal.index_level > 1

        def rename(index):
            if multi_index:
                return ('level_{}'.format(index),)
            elif ('index',) not in self._internal.column_labels:
                return ('index',)
            else:
                return ('level_{}'.format(index),)
        if level is None:
            new_column_labels = [name if name is None or len(name) > 1 else name[0] for name in self._internal.column_label_names]
            if self._internal.column_labels_level > 1:
                column_labels = [tuple(list(label) + [''] * (self._internal.column_labels_level - len(label))) for label in self._internal.column_labels]
                column_label_names = [None] * self._internal.column_labels_level + self._internal.column_label_names
                internal = self._internal.with_new_columns([self._kser_for(label) for label in column_labels], column_label_names=column_label_names)
                return DataFrame(internal)
            else:
                column_labels = self._internal.column_labels
                internal = self._internal.copy(column_labels=column_labels)
                return DataFrame(internal)
        else:
            if is_list_like(level):
                level = list(level)
            if isinstance(level, int) or is_name_like_tuple(level):
                level = [level]
            elif is_name_like_value(level):
                level = [(level,)]
            else:
                level = [lvl if isinstance(lvl, int) or is_name_like_tuple(lvl) else (lvl,) for lvl in level]
            if any((isinstance(l, int) for l in level)) and any((l < 0 for l in level)):
                raise IndexError('Too many levels: Index has only {} levels, not {}'.format(self._internal.index_level, max(level) + 1))
            if any((isinstance(l, int) for l in level)) and any((l >= self._internal.index_level for l in level)):
                raise IndexError('Too many levels: Index has only {} levels, not {}'.format(self._internal.index_level, max(level) + 1))
            if any((isinstance(l, str) for l in level)) and any((l not in self._internal.index_names for l in level)):
                raise KeyError('Level {} not found'.format(level))
            if self._internal.index_level == 1 and any((l > 0 for l in level)):
                raise ValueError('Cannot remove {} levels from an index with {} levels: at least one level must be left.'.format(len(level), self._internal.index_level))
            index_map = list(zip(self._internal.index_spark_columns, self._internal.index_names, self._internal.index_dtypes))
            index_map = [item for i, item in enumerate(index_map) if i not in level]
            index_spark_columns, index_names, index_dtypes = zip(*index_map)
            if drop:
                column_labels = [label for label in self._internal.column_labels if label not in [label for label in self._internal.column_labels if label[0] in level]]
            else:
                column_labels = self._internal.column_labels
            if self._internal.column_labels_level > 1:
                column_label_names = self._internal.column_label_names
                if col_level >= self._internal.column_labels_level:
                    raise IndexError('Too many levels: Index has only {} levels, not {}'.format(self._internal.column_labels_level, col_level + 1))
                if any((col_level + len(label) > self._internal.column_labels_level for label in column_labels)):
                    raise ValueError('Item must have length equal to number of levels.')
                new_column_labels = [tuple([col_fill] * col_level + list(label) + [col_fill] * (self._internal.column_labels_level - (len(label) + col_level))) for label in column_labels]
                internal = self._internal.with_new_columns([self._kser_for(label) for label in new_column_labels], column_labels=new_column_labels, data_dtypes=self._internal.data_dtypes)
                return DataFrame(internal)
            else:
                internal = self._internal.copy(index_spark_columns=index_spark_columns, index_names=index_names, index_dtypes=index_dtypes, column_labels=column_labels, data_spark_columns=[self._internal.spark_column_for(label) for label in column_labels], data_dtypes=self._internal.data_dtypes)
                return DataFrame(internal)

    def isnull(self):
        """
        Detects missing values for items in the current Dataframe.

        Return a boolean same-sized Dataframe indicating if the values are NA.
        NA values, such as None or numpy.NaN, gets mapped to True values.
        Everything else gets mapped to False values.

        See Also
        --------
        DataFrame.notnull

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)])
        >>> df.isnull()
               0      1
        0  False  False
        1  False   True
        2  False   True
        3  False  False

        >>> df = ks.DataFrame([[None, 'bee', None], ['dog', None, 'fly']])
        >>> df.isnull()
               0      1      2
        0   True  False   True
        1  False   True  False
        """
        return self._apply_series_op(lambda kser: kser.isnull())

    def notnull(self):
        """
        Detects non-missing values for items in the current Dataframe.

        This function takes a dataframe and indicates whether it's
        values are valid (not missing, which is ``NaN`` in numeric
        datatypes, ``None`` or ``NaN`` in objects and ``NaT`` in datetimelike).

        See Also
        --------
        DataFrame.isnull

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)])
        >>> df.notnull()
               0      1
        0  True   True
        1  True  False
        2  True  False
        3  True   True

        >>> df = ks.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
        >>> df.notnull()
               0      1      2
        0  True   True  True
        1  True  False  True
        """
        return self._apply_series_op(lambda kser: kser.notnull())
    notna = notnull

    def insert(self, loc, column, value, allow_duplicates=False):
        """
        Insert column into DataFrame at specified location.

        Raises a ValueError if `column` is already contained in the DataFrame,
        unless `allow_duplicates` is set to True.

        Parameters
        ----------
        loc : int
            Insertion index. Must verify 0 <= loc <= len(columns).
        column : str, number, or hashable object
            Label of the inserted column.
        value : int, Series, or array-like
        allow_duplicates : bool, optional

        Examples
        --------
        >>> kdf = ks.DataFrame([1, 2, 3])
        >>> kdf.sort_index()
           0
        0  1
        1  2
        2  3
        >>> kdf.insert(0, 'x', 4)
        >>> kdf.sort_index()
           x  0
        0  4  1
        1  4  2
        2  4  3

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)

        >>> kdf.insert(1, 'y', [5, 6, 7])
        >>> kdf.sort_index()
           x  y  0
        0  4  5  1
        1  4  6  2
        2  4  7  3

        >>> kdf.insert(2, 'z', ks.Series([8, 9, 10]))
        >>> kdf.sort_index()
           x  y   z  0
        0  4  5   8  1
        1  4  6   9  2
        2  4  7  10  3

        >>> reset_option("compute.ops_on_diff_frames")
        """
        if not isinstance(loc, int):
            raise TypeError('loc must be int')
        assert 0 <= loc <= len(self.columns)
        assert allow_duplicates is False
        if not is_name_like_value(column):
            raise ValueError('"column" should be a scalar value or tuple that contains scalar values')
        if is_name_like_tuple(column):
            if len(column) != len(self.columns.levels):
                raise ValueError('"column" must have length equal to number of column levels.')
        if column in self.columns:
            raise ValueError('cannot insert %s, already exists' % column)
        kdf = self.copy()
        kdf[column] = value
        columns = kdf.columns[:-1].insert(loc, kdf.columns[-1])
        kdf = kdf[columns]
        self._update_internal_frame(kdf._internal)

    def shift(self, periods=1, fill_value=None):
        """
        Shift DataFrame by desired number of periods.

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
        DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({'Col1': [10, 20, 30, 40, 45],
        ...                    'Col2': [13, 23, 33, 43, 53],
        ...                    'Col3': [17, 27, 37, 47, 57]},
        ...                   columns=['Col1', 'Col2', 'Col3'])

        >>> df.shift(periods=3)
           Col1  Col2  Col3
        0  NaN   NaN   NaN
        1  NaN   NaN   NaN
        2  NaN   NaN   NaN
        3  10.0  13.0  17.0
        4  20.0  23.0  27.0

        >>> df.shift(periods=3, fill_value=0)
           Col1  Col2  Col3
        0  0.0  0.0  0.0
        1  0.0  0.0  0.0
        2  0.0  0.0  0.0
        3  10.0  13.0  17.0
        4  20.0  23.0  27.0

        """
        return self._apply_series_op(lambda kser: kser._shift(periods, fill_value), should_resolve=True)

    def diff(self, periods=1, axis=0):
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another element in the
        DataFrame (default is the element in the same column of the previous row).

        .. note:: the current implementation of diff uses Spark’s Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.

        Returns
        -------
        diffed : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                    'b': [1, 1, 2, 3, 5, 8],
        ...                    'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        >>> df
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36

        >>> df.diff()
             a    b     c
        0  NaN  NaN   NaN
        1  1.0  0.0   3.0
        2  1.0  1.0   5.0
        3  1.0  1.0   7.0
        4  1.0  2.0   9.0
        5  1.0  3.0  11.0

        Difference with previous column

        >>> df.diff(periods=3)
             a    b     c
        0  NaN  NaN   NaN
        1  NaN  NaN   NaN
        2  NaN  NaN   NaN
        3  3.0  2.0  15.0
        4  3.0  4.0  21.0
        5  3.0  6.0  27.0

        Difference with following row

        >>> df.diff(periods=-1)
             a    b     c
        0 -1.0  0.0  -3.0
        1 -1.0 -1.0  -5.0
        2 -1.0 -1.0  -7.0
        3 -1.0 -2.0  -9.0
        4 -1.0 -3.0 -11.0
        5  NaN  NaN   NaN
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        return self._apply_series_op(lambda kser: kser._diff(periods), should_resolve=True)

    def nunique(self, axis=0, dropna=True, approx=False, rsd=0.05):
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.
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
        The number of unique values per column as a Koalas Series.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2], 'B': [np.nan, 3]})
        >>> df.nunique()
        A    2
        B    1
        dtype: int64

        >>> df.nunique(dropna=False)
        A    2
        B    2
        dtype: int64

        On big data, we recommend using the approximate algorithm to speed up this function.
        The result will be very close to the exact unique count.

        >>> df.nunique(approx=True)
        A    2
        B    1
        dtype: int64
        """
        from databricks.koalas.series import first_series
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        sdf = self._internal.resolved_copy.spark_frame
        column = verify_temp_column_name(sdf, '__duplicated__')
        if approx:
            if rsd <= 0.0 or rsd > 1.0:
                raise ValueError('rsd must be in the interval [0.0, 1.0]')
            if rsd == 0.0:
                return sdf.select([F.lit(None).cast(StringType()).alias(SPARK_DEFAULT_INDEX_NAME)] + [self._internal.spark_column_for(label).nunique(dropna) for label in self._internal.column_labels])
            if rsd == 1.0:
                return sdf.select([F.lit(None).cast(StringType()).alias(SPARK_DEFAULT_INDEX_NAME)] + [F.count(F.col(label)).alias(name_like_string(label)) for label in self._internal.column_labels])
            return sdf.select([F.lit(None).cast(StringType()).alias(SPARK_DEFAULT_INDEX_NAME)] + [self._internal.spark_column_for(label).nunique_approx(rsd=rsd) for label in self._internal.column_labels])
        else:
            return sdf.select([F.lit(None).cast(StringType()).alias(SPARK_DEFAULT_INDEX_NAME)] + [self._internal.spark_column_for(label).nunique(dropna) for label in self._internal.column_labels])

    def round(self, decimals=0):
        """
        Round a DataFrame to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. If an int is
            given, round each column to the same number of places.
            Otherwise dict and Series round to variable numbers of places.
            Column names should be in the keys if `decimals` is a
            dict-like, or in the index if `decimals` is a Series. Any
            columns not included in `decimals` will be left as is. Elements
            of `decimals` which are not columns of the input will be ignored.

            .. note:: If `decimals` is a Series, it is expected to be small,
                as all the data is loaded into the driver's memory.

        Returns
        -------
        DataFrame

        See Also
        --------
        Series.round

        Examples
        --------
        >>> df = ks.DataFrame({'A':[0.028208, 0.038683, 0.877076],
        ...                    'B':[0.992815, 0.645646, 0.149370],
        ...                    'C':[0.173891, 0.577595, 0.491027]},
        ...                   columns=['A', 'B', 'C'],
        ...                   index=['first', 'second', 'third'])
        >>> df
                       A         B         C
        first   0.028208  0.992815  0.173891
        second  0.038683  0.645646  0.577595
        third   0.877076  0.149370  0.491027

        >>> df.round(2)
                   A         B         C
        first   0.03    0.99    0.17
        second  0.04    0.65    0.58
        third   0.88    0.15    0.49

        >>> df.round({'A': 1, 'C': 2})
                  A         B         C
        first   0.0  0.992815  0.17
        second  0.0  0.645646  0.58
        third   0.9  0.149370  0.49

        >>> decimals = ks.Series([1, 0, 2], index=['A', 'B', 'C'])
        >>> df.round(decimals)
                  A         B         C
        first   0.0  0.992815  0.17
        second  0.0  0.645646  0.58
        third   0.9  0.149370  0.49
        """
        if isinstance(decimals, ks.Series):
            decimals = {k if isinstance(k, tuple) else (k,): v for k, v in decimals._to_internal_pandas().items()}
        elif isinstance(decimals, dict):
            decimals = {k if isinstance(k, tuple) else (k,): v for k, v in decimals.items()}
        elif isinstance(decimals, int):
            decimals = {k: decimals for k in self._internal.column_labels}
        else:
            raise ValueError('decimals must be an integer, a dict-like or a Series')

        def op(kser):
            label = kser._column_label
            if label in decimals:
                return F.round(kser.spark.column, decimals[label]).alias(kser._internal.data_spark_column_names[0])
            else:
                return kser
        return self._apply_series_op(op)

    def _mark_duplicates(self, subset=None, keep='first'):
        if subset is None:
            subset = self._internal.column_labels
        else:
            if is_name_like_tuple(subset):
                subset = [subset]
            elif is_name_like_value(subset):
                subset = [(subset,)]
            else:
                subset = [sub if is_name_like_tuple(sub) else (sub,) for sub in subset]
            diff = set(subset).difference(set(self._internal.column_labels))
            if len(diff) > 0:
                raise KeyError(', '.join([name_like_string(d) for d in diff]))
        group_cols = [self._internal.spark_column_name_for(label) for label in subset]
        sdf = self._internal.resolved_copy.spark_frame
        column = verify_temp_column_name(sdf, '__duplicated__')
        if keep == 'first' or keep == 'last':
            if keep == 'first':
                ord_func = spark.functions.asc
            else:
                ord_func = spark.functions.desc
            window = Window.partitionBy(group_cols).orderBy(ord_func(NATURAL_ORDER_COLUMN_NAME)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            sdf = sdf.withColumn(column, F.row_number().over(window) > 1)
        elif not keep:
            window = Window.partitionBy(group_cols).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            sdf = sdf.withColumn(column, F.count('*').over(window) > 1)
        else:
            raise ValueError("'keep' only supports 'first', 'last' and False")
        return (sdf, column)

    def duplicated(self, subset=None, keep='first'):
        """
        Return boolean Series denoting duplicate rows, optionally only considering certain columns.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns
        keep : {'first', 'last', False}, default 'first'
           - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
           - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
           - False : Mark all duplicates as ``True``.

        Returns
        -------
        duplicated : Series

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 1, 3], 'b': [1, 3, 3, 4]}, columns=['a', 'b'])
        >>> df
           a  b
        0  1  1
        1  1  3
        2  1  3
        3  3  4

        Drop the rows where at least one element is missing.

        >>> df.duplicated().sort_index()
        0    False
        1     True
        2     True
        3    False
        dtype: bool

        Mark duplicates as ``True`` except for the last occurrence.

        >>> df.duplicated(keep='last').sort_index()
        0     True
        1     True
        2    False
        3    False
        dtype: bool

        Mark all duplicates as ``True``.

        >>> df.duplicated(keep=False).sort_index()
        0     True
        1     True
        2     True
        3    False
        dtype: bool
        """
        from databricks.koalas.series import Series
        sdf, column = self._mark_duplicates(subset, keep)
        sdf = sdf.select(self._internal.index_spark_columns + [scol_for(sdf, column).alias(SPARK_DEFAULT_SERIES_NAME)])
        return Series(sdf.select(self._internal.index_spark_columns + [scol_for(sdf, column).alias(SPARK_DEFAULT_SERIES_NAME)])[SPARK_DEFAULT_SERIES_NAME])

    def dot(self, other):
        """
        Compute the matrix multiplication between the DataFrame and other.

        This method computes the matrix product between the DataFrame and the
        values of an other Series

        It can also be called using ``self @ other`` in Python >= 3.5.

        .. note:: This method is based on an expensive operation due to the nature
            of big data. Internally it needs to generate each row for each value, and
            then group twice - it is a huge operation. To prevent misusage, this method
            has the 'compute.max_rows' default limit of input length, and raises a ValueError.

                >>> from databricks.koalas.config import option_context
                >>> with option_context(
                ...     'compute.max_rows', 1000, "compute.ops_on_diff_frames", True
                ... ):  # doctest: +NORMALIZE_WHITESPACE
                ...     kdf = ks.DataFrame({'a': range(1001)})
                ...     kser = ks.Series([2], index=['a'])
                ...     kdf.dot(kser)
                Traceback (most recent call last):
                  ...
                ValueError: Current DataFrame has more then the given limit 1000 rows.
                Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option'
                to retrieve to retrieve more than 1000 rows. Note that, before changing the
                'compute.max_rows', this operation is considerably expensive.

        Parameters
        ----------
        other : Series
            The other object to compute the matrix product with.

        Returns
        -------
        Series
            Return the matrix product between self and other as a Series.

        See Also
        --------
        Series.dot: Similar method for Series.

        Notes
        -----
        The dimensions of DataFrame and other must be compatible in order to
        compute the matrix multiplication. In addition, the column names of
        DataFrame and the index of other must contain the same values, as they
        will be aligned prior to the multiplication.

        The dot method for Series computes the inner product, instead of the
        matrix product here.

        Examples
        --------
        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> kdf = ks.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        >>> kser = ks.Series([1, 1, 2, 1])
        >>> kdf.dot(kser)
        0    4
        1    5
        dtype: int64

        Note how shuffling of the objects does not change the result.

        >>> kser2 = kser.reindex([1, 0, 2, 3])
        >>> kdf.dot(kser2)
        0    4
        1    5
        dtype: int64
        >>> reset_option("compute.ops_on_diff_frames")
        """
        if not isinstance(other, ks.Series):
            raise TypeError('Unsupported type {}'.format(type(other).__name__))
        else:
            return cast(ks.Series, other.dot(self.transpose())).rename(None)

    def __matmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.dot(other)

    def to_koalas(self, index_col=None):
        """
        Converts the existing DataFrame into a Koalas DataFrame.

        This method is monkey-patched into Spark's DataFrame and can be used
        to convert a Spark DataFrame into a Koalas DataFrame. If running on
        an existing Koalas DataFrame, the method returns itself.

        If a Koalas DataFrame is converted to a Spark DataFrame and then back
        to Koalas, it will lose the index information and the original index
        will be turned into a normal column.

        Parameters
        ----------
        index_col: str or list of str, optional, default: None
            Index column of table in Spark.

        See Also
        --------
        DataFrame.to_spark

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, columns=['col1', 'col2'])
        >>> df
           col1  col2
        0     1     3
        1     2     4

        >>> spark_df = df.to_spark()
        >>> spark_df
        DataFrame[col1: bigint, col2: bigint]

        >>> kdf = spark_df.to_koalas()
        >>> kdf
           col1  col2
        0     1     3
        1     2     4

        We can specify the index columns.

        >>> kdf = spark_df.to_koalas(index_col='col1')
        >>> kdf  # doctest: +NORMALIZE_WHITESPACE
              col2
        col1
        1        3
        2        4

        Calling to_koalas on a Koalas DataFrame simply returns itself.

        >>> df.to_koalas()
           col1  col2
        0     1     3
        1     2     4
        """
        if isinstance(self, DataFrame):
            return self
        else:
            assert isinstance(self, spark.DataFrame), type(self)
            from databricks.koalas.namespace import _get_index_map
            index_spark_columns, index_names = _get_index_map(self, index_col)
            internal = InternalFrame(spark_frame=self, index_spark_columns=index_spark_columns, index_names=index_names)
            return DataFrame(internal)

    def cache(self):
        warnings.warn('DataFrame.cache is deprecated as of DataFrame.spark.cache. Please use the API instead.', FutureWarning)
        return self.spark.cache()
    cache.__doc__ = SparkFrameMethods.cache.__doc__

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        warnings.warn('DataFrame.persist is deprecated as of DataFrame.spark.persist. Please use the API instead.', FutureWarning)
        return self.spark.persist(storage_level)
    persist.__doc__ = SparkFrameMethods.persist.__doc__

    def hint(self, name, *parameters):
        warnings.warn('DataFrame.hint is deprecated as of DataFrame.spark.hint. Please use the API instead.', FutureWarning)
        return self.spark.hint(name, *parameters)
    hint.__doc__ = SparkFrameMethods.hint.__doc__

    def to_table(self, name, format=None, mode='overwrite', partition_cols=None, index_col=None, **options):
        return self.spark.to_table(name, format, mode, partition_cols, index_col, **options)
    to_table.__doc__ = SparkFrameMethods.to_table.__doc__

    def to_delta(self, path, mode='overwrite', partition_cols=None, index_col=None, **options):
        """
        Write the DataFrame out as a Delta Lake table.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when the destination
            exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options : dict
            All other options passed directly into Delta Lake.

        See Also
        --------
        read_delta
        DataFrame.to_parquet
        DataFrame.to_table
        DataFrame.to_spark_io

        Examples
        --------

        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df
                         date country  code
        0 2012-01-31 12:00:00      KR     1
        1 2012-02-29 12:00:00      US     2
        2 2012-03-31 12:00:00      JP     3

        Create a new Delta Lake table, partitioned by one column:

        >>> df.to_delta('%s/to_delta/foo' % path, partition_cols='date')

        Partitioned by two columns:

        >>> df.to_delta('%s/to_delta/bar' % path, partition_cols=['date', 'country'])

        Overwrite an existing table's partitions, using the 'replaceWhere' capability in Delta:

        >>> df.to_delta('%s/to_delta/bar' % path,
        ...             mode='overwrite', replaceWhere='date >= "2012-01-01"')
        """
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='delta', partition_cols=partition_cols, index_col=index_col, **options)

    def to_parquet(self, path, mode='overwrite', partition_cols=None, compression=None, index_col=None, **options):
        """
        Write the DataFrame out as a Parquet file or directory.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when the destination
            exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        compression : str {'none', 'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4', 'zstd'}
            Compression codec to use when saving to file. If None is set, it uses the
            value specified in `spark.sql.parquet.compression.codec`.
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options : dict
            All other options passed directly into Spark's data source.

        See Also
        --------
        read_parquet
        DataFrame.to_delta
        DataFrame.to_table
        DataFrame.to_spark_io

        Examples
        --------

        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df
                         date country  code
        0 2012-01-31 12:00:00      KR     1
        1 2012-02-29 12:00:00      US     2
        2 2012-03-31 12:00:00      JP     3

        >>> df.to_parquet('%s/to_parquet/foo.parquet' % path, partition_cols='date')

        >>> df.to_parquet(
        ...     '%s/to_parquet/foo.parquet' % path,
        ...     mode = 'overwrite',
        ...     partition_cols=['date', 'country'])
        """
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='parquet', partition_cols=partition_cols, index_col=index_col, **options)

    def to_orc(self, path, mode='overwrite', partition_cols=None, index_col=None, **options):
        """
        Write the DataFrame out as a ORC file or directory.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when the destination
            exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options : dict
            All other options passed directly into Spark's data source.

        See Also
        --------
        read_orc
        DataFrame.to_delta
        DataFrame.to_parquet
        DataFrame.to_table
        DataFrame.to_spark_io

        Examples
        --------

        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df
                         date country  code
        0 2012-01-31 12:00:00      KR     1
        1 2012-02-29 12:00:00      US     2
        2 2012-03-31 12:00:00      JP     3

        >>> df.to_orc('%s/to_orc/foo.orc' % path, partition_cols='date')

        >>> df.to_orc(
        ...     '%s/to_orc/foo.orc' % path,
        ...     mode = 'overwrite',
        ...     partition_cols=['date', 'country'])
        """
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='orc', partition_cols=partition_cols, index_col=index_col, **options)

    def to_spark_io(self, path=None, format=None, mode='overwrite', partition_cols=None, index_col=None, **options):
        return self.spark.to_spark_io(path, format, mode, partition_cols, index_col, **options)
    to_spark_io.__doc__ = SparkFrameMethods.to_spark_io.__doc__

    def to_spark(self, index_col=None):
        return self.spark.frame(index_col)
    to_spark.__doc__ = SparkFrameMethods.__doc__

    def to_pandas(self):
        """
        Return a pandas DataFrame.

        .. note:: This method should only be used if the resulting pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.to_pandas()
           dogs  cats
        0   0.2   0.3
        1   0.0   0.6
        2   0.6   0.0
        3   0.2   0.1
        """
        return self._internal.to_pandas_frame.copy()

    def toPandas(self):
        warnings.warn('DataFrame.toPandas is deprecated as of DataFrame.to_pandas. Please use the API instead.', FutureWarning)
        return self.to_pandas()
    toPandas.__doc__ = to_pandas.__doc__

    def assign(self, **kwargs):
        """
        Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : dict of {str: callable, Series or Index}
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though Koalas doesn't check it).
            If the values are not callable, (e.g. a Series or a literal),
            they are simply assigned.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to
            all the existing columns.

        Examples
        --------
        >>> df = ks.DataFrame({'temp_c': [17.0, 25.0]},
        ...                   index=['Portland', 'Berkeley'])
        >>> df
                  temp_c
        Portland    17.0
        Berkeley    25.0

        Where the value is a callable, evaluated on `df`:

        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        Alternatively, the same behavior can be achieved by directly
        referencing an existing Series or sequence and you can also
        create multiple columns within the same assign.

        >>> df.assign(temp_f=df['temp_c'] * 9 / 5 + 32,
        ...          temp_k=df['temp_c'] + 273.15,
        ...          temp_idx=df.index)
                         temp_c  temp_f  temp_k  temp_idx
        Portland    17.0    62.6  290.15  Portland
        Berkeley    25.0    77.0  298.15  Berkeley
        """
        return self._assign(kwargs)

    def _assign(self, kwargs):
        assert isinstance(kwargs, dict)
        from databricks.koalas.indexes import MultiIndex
        from databricks.koalas.series import IndexOpsMixin
        for k, v in kwargs.items():
            is_invalid_assignee = not (isinstance(v, (IndexOpsMixin, spark.Column)) or callable(v) or is_scalar(v)) or isinstance(v, MultiIndex)
            if is_invalid_assignee:
                raise TypeError("Column assignment doesn't support type {}".format(type(v).__name__))
            if callable(v):
                kwargs[k] = v(self)
        pairs = {k if is_name_like_tuple(k) else (k,): (v.spark.column, v.dtype) if isinstance(v, IndexOpsMixin) and (not isinstance(v, MultiIndex)) else (v, None) for k, v in kwargs.items()}
        scols = []
        data_dtypes = []
        for label in self