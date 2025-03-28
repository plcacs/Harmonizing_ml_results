"""
A wrapper class for Spark Column to behave similar to pandas Series.
"""
import datetime
import re
import inspect
import sys
import warnings
from collections.abc import Mapping
from distutils.version import LooseVersion
from functools import partial, wraps, reduce
from typing import Any, Callable, Dict, Generator, Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, cast
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
from pyspark.sql.types import BooleanType, DoubleType, FloatType, IntegerType, LongType, NumericType, StructType, IntegralType, ArrayType
from pyspark.sql.window import Window
from databricks import koalas as ks
from databricks.koalas.accessors import KoalasSeriesMethods
from databricks.koalas.categorical import CategoricalAccessor
from databricks.koalas.config import get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.exceptions import SparkPandasIndexingError
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import Frame
from databricks.koalas.internal import InternalFrame, DEFAULT_SERIES_NAME, NATURAL_ORDER_COLUMN_NAME, SPARK_DEFAULT_INDEX_NAME, SPARK_DEFAULT_SERIES_NAME
from databricks.koalas.missing.series import MissingPandasLikeSeries
from databricks.koalas.plot import KoalasPlotAccessor
from databricks.koalas.ml import corr
from databricks.koalas.utils import combine_frames, is_name_like_tuple, is_name_like_value, name_like_string, same_anchor, scol_for, sql_conf, validate_arguments_and_invoke_function, validate_axis, validate_bool_kwarg, verify_temp_column_name, SPARK_CONF_ARROW_ENABLED
from databricks.koalas.datetimes import DatetimeMethods
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkSeriesMethods
from databricks.koalas.strings import StringMethods
from databricks.koalas.typedef import infer_return_type, spark_type_to_pandas_dtype, ScalarType, Scalar, SeriesType
REPR_PATTERN = re.compile('Length: (?P<length>[0-9]+)')
_flex_doc_SERIES = """
Return {desc} of series and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``

Parameters
----------
other : Series or scalar value

Returns
-------
Series
    The result of the operation.

See Also
--------
Series.{reverse}

{series_examples}
"""
_add_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.add(df.b)
a    4.0
b    NaN
c    6.0
d    NaN
dtype: float64

>>> df.a.radd(df.b)
a    4.0
b    NaN
c    6.0
d    NaN
dtype: float64
"""
_sub_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.subtract(df.b)
a    0.0
b    NaN
c    2.0
d    NaN
dtype: float64

>>> df.a.rsub(df.b)
a    0.0
b    NaN
c   -2.0
d    NaN
dtype: float64
"""
_mul_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.multiply(df.b)
a    4.0
b    NaN
c    8.0
d    NaN
dtype: float64

>>> df.a.rmul(df.b)
a    4.0
b    NaN
c    8.0
d    NaN
dtype: float64
"""
_div_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.divide(df.b)
a    1.0
b    NaN
c    2.0
d    NaN
dtype: float64

>>> df.a.rdiv(df.b)
a    1.0
b    NaN
c    0.5
d    NaN
dtype: float64
"""
_pow_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.pow(df.b)
a     4.0
b     NaN
c    16.0
d     NaN
dtype: float64

>>> df.a.rpow(df.b)
a     4.0
b     NaN
c    16.0
d     NaN
dtype: float64
"""
_mod_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.mod(df.b)
a    0.0
b    NaN
c    0.0
d    NaN
dtype: float64

>>> df.a.rmod(df.b)
a    0.0
b    NaN
c    2.0
d    NaN
dtype: float64
"""
_floordiv_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.floordiv(df.b)
a    1.0
b    NaN
c    2.0
d    NaN
dtype: float64

>>> df.a.rfloordiv(df.b)
a    1.0
b    NaN
c    0.0
d    NaN
dtype: float64
"""
T = TypeVar('T')
str_type = str


def _create_type_for_series_type(param):
    from databricks.koalas.typedef import NameTypeHolder
    if isinstance(param, ExtensionDtype):
        new_class = type('NameType', (NameTypeHolder,), {})
        new_class.tpe = param
    else:
        new_class = param.type if isinstance(param, np.dtype) else param
    return SeriesType[new_class]


if (3, 5) <= sys.version_info < (3, 7):
    from typing import GenericMeta
    old_getitem = GenericMeta.__getitem__

    def new_getitem(self, params):
        if hasattr(self, 'is_series'):
            return old_getitem(self, _create_type_for_series_type(params))
        else:
            return old_getitem(self, params)
    GenericMeta.__getitem__ = new_getitem


class Series(Frame, IndexOpsMixin, Generic[T]):
    """
    Koalas Series that corresponds to pandas Series logically. This holds Spark Column
    internally.

    :ivar _internal: an internal immutable Frame to manage metadata.
    :type _internal: InternalFrame
    :ivar _kdf: Parent's Koalas DataFrame
    :type _kdf: ks.DataFrame

    Parameters
    ----------
    data : array-like, dict, or scalar value, pandas Series
        Contains data stored in Series
        If data is a dict, argument order is maintained for Python 3.6
        and later.
        Note that if `data` is a pandas Series, other arguments should not be used.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If both a dict and index
        sequence are used, the index will override the keys found in the
        dict.
    dtype : numpy.dtype or None
        If None, dtype will be inferred
    copy : boolean, default False
        Copy input data
    """

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=
        False, fastpath=False):
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
                s = pd.Series(data=data, index=index, dtype=dtype, name=
                    name, copy=copy, fastpath=fastpath)
            internal = InternalFrame.from_pandas(pd.DataFrame(s))
            if s.name is None:
                internal = internal.copy(column_labels=[None])
            anchor = DataFrame(internal)
            self._anchor = anchor
            self._col_label = anchor._internal.column_labels[0]
            object.__setattr__(anchor, '_kseries', {self._column_label: self})

    @property
    def _kdf(self):
        return self._anchor

    @property
    def _internal(self):
        return self._kdf._internal.select_column(self._column_label)

    @property
    def _column_label(self):
        return self._col_label

    def _update_anchor(self, kdf):
        assert kdf._internal.column_labels == [self._column_label], (kdf.
            _internal.column_labels, [self._column_label])
        self._anchor = kdf
        object.__setattr__(kdf, '_kseries', {self._column_label: self})

    def _with_new_scol(self, scol, *, dtype: Optional[np.dtype]=None):
        """
        Copy Koalas Series with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Series
        """
        internal = self._internal.copy(data_spark_columns=[scol.alias(
            name_like_string(self._column_label))], data_dtypes=[dtype])
        return first_series(DataFrame(internal))
    spark: 'SparkSeriesMethods' = CachedAccessor('spark', SparkSeriesMethods)

    @property
    def dtypes(self):
        """Return the dtype object of the underlying data.

        >>> s = ks.Series(list('abc'))
        >>> s.dtype == s.dtypes
        True
        """
        return self.dtype

    @property
    def axes(self):
        """
        Return a list of the row axis labels.

        Examples
        --------

        >>> kser = ks.Series([1, 2, 3])
        >>> kser.axes
        [Int64Index([0, 1, 2], dtype='int64')]
        """
        return [self.index]

    @property
    def spark_type(self):
        warnings.warn(
            'Series.spark_type is deprecated as of Series.spark.data_type. Please use the API instead.'
            , FutureWarning)
        return self.spark.data_type
    spark_type.__doc__ = SparkSeriesMethods.data_type.__doc__

    def add(self, other):
        return self + other
    add.__doc__ = _flex_doc_SERIES.format(desc='Addition', op_name='+',
        equiv='series + other', reverse='radd', series_examples=
        _add_example_SERIES)

    def radd(self, other):
        return other + self
    radd.__doc__ = _flex_doc_SERIES.format(desc='Reverse Addition', op_name
        ='+', equiv='other + series', reverse='add', series_examples=
        _add_example_SERIES)

    def div(self, other):
        return self / other
    div.__doc__ = _flex_doc_SERIES.format(desc='Floating division', op_name
        ='/', equiv='series / other', reverse='rdiv', series_examples=
        _div_example_SERIES)

    def divide(self, other):
        return self.div(other)
    divide.__doc__ = Series.div.__doc__

    def rdiv(self, other):
        return other / self
    rdiv.__doc__ = _flex_doc_SERIES.format(desc='Reverse Floating division',
        op_name='/', equiv='other / series', reverse='div', series_examples
        =_div_example_SERIES)

    def truediv(self, other):
        return self / other
    truediv.__doc__ = _flex_doc_SERIES.format(desc='Floating division',
        op_name='/', equiv='series / other', reverse='rtruediv',
        series_examples=_div_example_SERIES)

    def rtruediv(self, other):
        return other / self
    rtruediv.__doc__ = _flex_doc_SERIES.format(desc=
        'Reverse Floating division', op_name='/', equiv='other / series',
        reverse='truediv', series_examples=_div_example_SERIES)

    def mul(self, other):
        return self * other
    mul.__doc__ = _flex_doc_SERIES.format(desc='Multiplication', op_name=
        '*', equiv='series * other', reverse='rmul', series_examples=
        _mul_example_SERIES)

    def multiply(self, other):
        return self.mul(other)
    multiply.__doc__ = Series.mul.__doc__

    def rmul(self, other):
        return other * self
    rmul.__doc__ = _flex_doc_SERIES.format(desc='Reverse Multiplication',
        op_name='*', equiv='other * series', reverse='mul', series_examples
        =_mul_example_SERIES)

    def sub(self, other):
        return self - other
    sub.__doc__ = _flex_doc_SERIES.format(desc='Subtraction', op_name='-',
        equiv='series - other', reverse='rsub', series_examples=
        _sub_example_SERIES)

    def subtract(self, other):
        return self.sub(other)
    subtract.__doc__ = Series.sub.__doc__

    def rsub(self, other):
        return other - self
    rsub.__doc__ = _flex_doc_SERIES.format(desc='Reverse Subtraction',
        op_name='-', equiv='other - series', reverse='sub', series_examples
        =_sub_example_SERIES)

    def mod(self, other):
        return self % other
    mod.__doc__ = _flex_doc_SERIES.format(desc='Modulo', op_name='%', equiv
        ='series % other', reverse='rmod', series_examples=_mod_example_SERIES)

    def rmod(self, other):
        return other % self
    rmod.__doc__ = _flex_doc_SERIES.format(desc='Reverse Modulo', op_name=
        '%', equiv='other % series', reverse='mod', series_examples=
        _mod_example_SERIES)

    def pow(self, other):
        return self ** other
    pow.__doc__ = _flex_doc_SERIES.format(desc=
        'Exponential power of series', op_name='**', equiv=
        'series ** other', reverse='rpow', series_examples=_pow_example_SERIES)

    def rpow(self, other):
        return other ** self
    rpow.__doc__ = _flex_doc_SERIES.format(desc='Reverse Exponential power',
        op_name='**', equiv='other ** series', reverse='pow',
        series_examples=_pow_example_SERIES)

    def floordiv(self, other):
        return self // other
    floordiv.__doc__ = _flex_doc_SERIES.format(desc='Integer division',
        op_name='//', equiv='series // other', reverse='rfloordiv',
        series_examples=_floordiv_example_SERIES)

    def rfloordiv(self, other):
        return other // self
    rfloordiv.__doc__ = _flex_doc_SERIES.format(desc=
        'Reverse Integer division', op_name='//', equiv='other // series',
        reverse='floordiv', series_examples=_floordiv_example_SERIES)
    koalas: 'KoalasSeriesMethods' = CachedAccessor('koalas',
        KoalasSeriesMethods)

    def eq(self, other):
        """
        Compare if the current value is equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a == 1
        a     True
        b    False
        c    False
        d    False
        Name: a, dtype: bool

        >>> df.b.eq(1)
        a     True
        b    False
        c     True
        d    False
        Name: b, dtype: bool
        """
        return self == other
    equals: Callable[[Any], 'Series'] = eq

    def gt(self, other):
        """
        Compare if the current value is greater than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a > 1
        a    False
        b     True
        c     True
        d     True
        Name: a, dtype: bool

        >>> df.b.gt(1)
        a    False
        b    False
        c    False
        d    False
        Name: b, dtype: bool
        """
        return self > other

    def ge(self, other):
        """
        Compare if the current value is greater than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a >= 2
        a    False
        b     True
        c     True
        d     True
        Name: a, dtype: bool

        >>> df.b.ge(2)
        a    False
        b    False
        c    False
        d    False
        Name: b, dtype: bool
        """
        return self >= other

    def lt(self, other):
        """
        Compare if the current value is less than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a < 1
        a    False
        b    False
        c    False
        d    False
        Name: a, dtype: bool

        >>> df.b.lt(2)
        a     True
        b    False
        c     True
        d    False
        Name: b, dtype: bool
        """
        return self < other

    def le(self, other):
        """
        Compare if the current value is less than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a <= 2
        a     True
        b     True
        c    False
        d    False
        Name: a, dtype: bool

        >>> df.b.le(2)
        a     True
        b    False
        c     True
        d    False
        Name: b, dtype: bool
        """
        return self <= other

    def ne(self, other):
        """
        Compare if the current value is not equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a != 1
        a    False
        b     True
        c     True
        d     True
        Name: a, dtype: bool

        >>> df.b.ne(1)
        a    False
        b     True
        c    False
        d     True
        Name: b, dtype: bool
        """
        return self != other

    def divmod(self, other):
        """
        Return Integer division and modulo of series and other, element-wise
        (binary operator `divmod`).

        Parameters
        ----------
        other : Series or scalar value

        Returns
        -------
        2-Tuple of Series
            The result of the operation.

        See Also
        --------
        Series.rdivmod
        """
        return self.floordiv(other), self.mod(other)

    def rdivmod(self, other):
        """
        Return Integer division and modulo of series and other, element-wise
        (binary operator `rdivmod`).

        Parameters
        ----------
        other : Series or scalar value

        Returns
        -------
        2-Tuple of Series
            The result of the operation.

        See Also
        --------
        Series.divmod
        """
        return self.rfloordiv(other), self.rmod(other)

    def between(self, left, right, inclusive=True):
        """
        Return boolean Series equivalent to left <= series <= right.
        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : scalar or list-like
            Left boundary.
        right : scalar or list-like
            Right boundary.
        inclusive : bool, default True
            Include boundaries.

        Returns
        -------
        Series
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> s = ks.Series([2, 0, 4, 8, np.nan])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        With `inclusive` set to ``False`` boundary values are excluded:

        >>> s.between(1, 4, inclusive=False)
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = ks.Series(['Alice', 'Bob', 'Carol', 'Eve'])
        >>> s.between('Anna', 'Daniel')
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        if inclusive:
            lmask = self >= left
            rmask = self <= right
        else:
            lmask = self > left
            rmask = self < right
        return lmask & rmask

    def map(self, arg):
        """
        Map values of Series according to input correspondence.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict``.

        .. note:: make sure the size of the dictionary is not huge because it could
            downgrade the performance or throw OutOfMemoryError due to a huge
            expression within Spark. Consider the input as a functions as an
            alternative instead in this case.

        Parameters
        ----------
        arg : function or dict
            Mapping correspondence.

        Returns
        -------
        Series
            Same index as caller.

        See Also
        --------
        Series.apply : For applying more complex functions on a Series.
        DataFrame.applymap : Apply a function elementwise on a whole DataFrame.

        Notes
        -----
        When ``arg`` is a dictionary, values in Series that are not in the
        dictionary (as keys) are converted to ``None``. However, if the
        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
        provides a method for default values), then this default is used
        rather than ``None``.

        Examples
        --------
        >>> s = ks.Series(['cat', 'dog', None, 'rabbit'])
        >>> s
        0       cat
        1       dog
        2      None
        3    rabbit
        dtype: object

        ``map`` accepts a ``dict``. Values that are not found
        in the ``dict`` are converted to ``None``, unless the dict has a default
        value (e.g. ``defaultdict``):

        >>> s.map({'cat': 'kitten', 'dog': 'puppy'})
        0    kitten
        1     puppy
        2      None
        3      None
        dtype: object

        It also accepts a function:

        >>> def format(x) -> str:
        ...     return 'I am a {}'.format(x)

        >>> s.map(format)
        0       I am a cat
        1       I am a dog
        2      I am a None
        3    I am a rabbit
        dtype: object
        """
        if isinstance(arg, dict):
            is_start = True
            current: Column = F.when(F.lit(False), F.lit(None).cast(self.
                spark.data_type))
            for to_replace, value in arg.items():
                if is_start:
                    current = F.when(self.spark.column == F.lit(to_replace),
                        value)
                    is_start = False
                else:
                    current = current.when(self.spark.column == F.lit(
                        to_replace), value)
            if hasattr(arg, '__missing__'):
                tmp_val = arg[np._NoValue]
                del arg[np._NoValue]
                current = current.otherwise(F.lit(tmp_val))
            else:
                current = current.otherwise(F.lit(None).cast(self.spark.
                    data_type))
            return self._with_new_scol(current)
        else:
            return self.apply(arg)

    def alias(self, name):
        """An alias for :meth:`Series.rename`."""
        warnings.warn(
            'Series.alias is deprecated as of Series.rename. Please use the API instead.'
            , FutureWarning)
        return self.rename(name)

    @property
    def shape(self):
        """Return a tuple of the shape of the underlying data."""
        return len(self),

    @property
    def name(self):
        """Return name of the Series."""
        name = self._column_label
        if name is not None and len(name) == 1:
            return name[0]
        else:
            return name

    @name.setter
    def name(self, name):
        self.rename(name, inplace=True)

    def rename(self, index=None, inplace=False):
        """
        Alter Series name.

        Parameters
        ----------
        index : scalar
            Scalar will alter the ``Series.name`` attribute.
        inplace : bool, default False
            Whether to return a new Series. If True then value of copy is
            ignored.

        Returns
        -------
        Series
            Series with name altered.

        Examples
        --------

        >>> s = ks.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64

        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        """
        if index is None:
            pass
        elif not is_hashable(index):
            raise TypeError('Series.name must be a hashable type')
        elif not isinstance(index, tuple):
            index = index,
        scol: Column = self.spark.column.alias(name_like_string(index))
        internal = self._internal.copy(column_labels=[index],
            data_spark_columns=[scol], column_label_names=None)
        kdf: DataFrame = DataFrame(internal)
        if kwargs.get('inplace', False):
            self._col_label = index
            self._update_anchor(kdf)
            return self
        else:
            return first_series(kdf)

    def rename_axis(self, mapper=None, index=None, inplace=False):
        """
        Set the name of the axis for the index or columns.

        Parameters
        ----------
        mapper, index :  scalar, list-like, dict-like or function, optional
            A scalar, list-like, dict-like or functions transformations to
            apply to the index values.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Series.

        Returns
        -------
        Series, or None if `inplace` is True.

        See Also
        --------
        Series.rename : Alter Series index labels or name.
        DataFrame.rename : Alter DataFrame index labels or name.
        Index.rename : Set new names on index.

        Examples
        --------
        >>> s = ks.Series(["dog", "cat", "monkey"], name="animal")
        >>> s  # doctest: +NORMALIZE_WHITESPACE
        0       dog
        1       cat
        2    monkey
        Name: animal, dtype: object
        >>> s.rename_axis("index").sort_index()  # doctest: +NORMALIZE_WHITESPACE
        index
        0       dog
        1       cat
        2    monkey
        Name: animal, dtype: object

        **MultiIndex**

        >>> index = pd.MultiIndex.from_product([['mammal'],
        ...                                        ['dog', 'cat', 'monkey']],
        ...                                       names=['type', 'name'])
        >>> s = ks.Series([4, 4, 2], index=index, name='num_legs')
        >>> s  # doctest: +NORMALIZE_WHITESPACE
        type    name
        mammal  dog       4
                cat       4
                monkey    2
        Name: num_legs, dtype: int64
        >>> s.rename_axis(index={'type': 'class'}).sort_index()  # doctest: +NORMALIZE_WHITESPACE
        class   name
        mammal  cat       4
                dog       4
                monkey    2
        Name: num_legs, dtype: int64
        >>> s.rename_axis(index=str.upper).sort_index()  # doctest: +NORMALIZE_WHITESPACE
        TYPE    NAME
        mammal  cat       4
                dog       4
                monkey    2
        Name: num_legs, dtype: int64
        """
        kdf = self.to_frame().rename_axis(mapper=mapper, index=index,
            inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    @property
    def index(self):
        """The index (axis labels) Column of the Series.

        See Also
        --------
        Index
        """
        return self._kdf.index

    @property
    def is_unique(self):
        """
        Return boolean if values in the object are unique

        Returns
        -------
        is_unique : boolean

        >>> ks.Series([1, 2, 3]).is_unique
        True
        >>> ks.Series([1, 2, 2]).is_unique
        False
        >>> ks.Series([1, 2, 3, None]).is_unique
        True
        """
        scol: Column = self.spark.column
        return self._internal.spark_frame.select((F.count(scol) == F.
            countDistinct(scol)) & (F.count(F.when(scol.isNull(), 1).
            otherwise(None)) <= 1)).collect()[0][0]

    def reset_index(self, level=None, drop=False, name=None, inplace=False):
        """
        Generate a new DataFrame or Series with the index reset.

        This is useful when the index needs to be treated as a column,
        or when the index is meaningless and needs to be reset
        to the default before another operation.

        Parameters
        ----------
        level : int, str, tuple, or list, default optional
            For a Series with a MultiIndex, only remove the specified levels from the index.
            Removes all levels by default.
        drop : bool, default False
            Just reset the index, without inserting it as a column in the new DataFrame.
        name : object, optional
            The name to use for the column containing the original Series values.
            Uses self.name by default. This argument is ignored when drop is True.
        inplace : bool, default False
            Modify the Series in place (do not create a new object).

        Returns
        -------
        Series or DataFrame
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.
            In either case, if ``inplace=True``, no value is returned.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4], index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))

        Generate a DataFrame with default index.

        >>> s.reset_index()
          idx  0
        0   a  1
        1   b  2
        2   c  3
        3   d  4

        To specify the name of the new column use `name`.

        >>> s.reset_index(name='values')
          idx  values
        0   a       1
        1   b       2
        2   c       3
        3   d       4

        To generate a new Series with the default set `drop` to True.

        >>> s.reset_index(drop=True)
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        To update the Series in place, without generating a new one
        set `inplace` to True. Note that it also requires ``drop=True``.

        >>> s.reset_index(inplace=True, drop=True)
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if inplace and not drop:
            raise TypeError(
                'Cannot reset_index inplace on a Series to create a DataFrame')
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

    def to_frame(self, name=None):
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, default None
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        Examples
        --------
        >>> s = ks.Series(["a", "b", "c"])
        >>> s.to_frame()
           0
        0  a
        1  b
        2  c

        >>> s = ks.Series(["a", "b", "c"], name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        """
        if name is not None:
            renamed = self.rename(name)
        elif self._column_label is None:
            renamed = self.rename(DEFAULT_SERIES_NAME)
        else:
            renamed = self
        return DataFrame(renamed._internal)
    to_dataframe = to_frame

    def to_string(self, buf=None, na_rep='NaN', float_format=None, header=
        True, index=True, length=False, dtype=False, name=False, max_rows=None
        ):
        """
        Render a string representation of the Series.

        .. note:: This method should only be used if the resulting pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Parameters
        ----------
        buf : StringIO-like, optional
            buffer to write to
        na_rep : string, optional
            string representation of NAN to use, default 'NaN'
        float_format : one-parameter function, optional
            formatter function to apply to columns' elements if they are floats
            default None
        header : boolean, default True
            Add the Series header (index name)
        index : bool, optional
            Add index (row) labels, default True
        length : boolean, default False
            Add the Series length
        dtype : boolean, default False
            Add the Series dtype
        name : boolean, default False
            Add the Series name if not None
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.

        Returns
        -------
        formatted : string (if not buffer passed)

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)], columns=['dogs', 'cats'])
        >>> print(df['dogs'].to_string())
        0    0.2
        1    0.0
        2    0.6
        3    0.2

        >>> print(df['dogs'].to_string(max_rows=2))
        0    0.2
        1    0.0
        """
        args = locals()
        if max_rows is not None:
            kser = self.head(max_rows)
        else:
            kser = self
        return validate_arguments_and_invoke_function(kser.
            _to_internal_pandas(), self.to_string, pd.Series.to_string, args)

    def to_clipboard(self, excel=True, sep=None, **kwargs: Any):
        args = locals()
        kser = self
        return validate_arguments_and_invoke_function(kser.
            _to_internal_pandas(), self.to_clipboard, pd.Series.
            to_clipboard, args)
    to_clipboard.__doc__ = DataFrame.to_clipboard.__doc__

    def to_dict(self, into=dict):
        """
        Convert Series to {label -> value} dict or dict-like object.

        .. note:: This method should only be used if the resulting pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.Mapping subclass to use as the return
            object. Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        collections.abc.Mapping
            Key-value representation of Series.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s_dict = s.to_dict()
        >>> sorted(s_dict.items())
        [(0, 1), (1, 2), (2, 3), (3, 4)]

        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])

        >>> dd = defaultdict(list)
        >>> s.to_dict(dd)  # doctest: +ELLIPSIS
        defaultdict(<class 'list'>, {...})
        """
        args = locals()
        kser = self
        return validate_arguments_and_invoke_function(kser.
            _to_internal_pandas(), self.to_dict, pd.Series.to_dict, args)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True,
        index=True, na_rep='NaN', formatters=None, float_format=None,
        sparsify=None, index_names=True, bold_rows=False, column_format=
        None, longtable=False, escape=None, encoding=None, decimal='.',
        multicolumn=None, multicolumn_format=None, multirow=None):
        args = locals()
        kser = self
        return validate_arguments_and_invoke_function(kser.
            _to_internal_pandas(), self.to_latex, pd.Series.to_latex, args)
    to_latex.__doc__ = DataFrame.to_latex.__doc__

    def to_pandas(self):
        """
        Return a pandas Series.

        .. note:: This method should only be used if the resulting pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)], columns=['dogs', 'cats'])
        >>> df['dogs'].to_pandas()
        0    0.2
        1    0.0
        2    0.6
        3    0.2
        Name: dogs, dtype: float64
        """
        return self._to_internal_pandas().copy()

    def toPandas(self):
        warnings.warn(
            'Series.toPandas is deprecated as of Series.to_pandas. Please use the API instead.'
            , FutureWarning)
        return self.to_pandas()
    toPandas.__doc__ = to_pandas.__doc__

    def to_list(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        .. note:: This method should only be used if the resulting list is expected
            to be small, as all the data is loaded into the driver's memory.

        """
        return self._to_internal_pandas().tolist()
    tolist = to_list

    def drop_duplicates(self, keep='first', inplace=False):
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            Method to handle dropping duplicates:
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.

        Returns
        -------
        Series
            Series with duplicates dropped.

        Examples
        --------
        Generate a Series with duplicated entries.

        >>> s = ks.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'],
        ...               name='animal')
        >>> s.sort_index()
        0      lama
        1       cow
        2      lama
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        With the 'keep' parameter, the selection behaviour of duplicated values
        can be changed. The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> s.drop_duplicates().sort_index()
        0      lama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter 'keep' keeps the last occurrence for
        each set of duplicated entries.

        >>> s.drop_duplicates(keep='last').sort_index()
        1       cow
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        The value ``False`` for parameter 'keep' discards all sets of
        duplicated entries. Setting the value of 'inplace' to ``True`` performs
        the operation inplace and returns ``None``.

        >>> s.drop_duplicates(keep=False, inplace=True)
        >>> s.sort_index()
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf = self._kdf[[self.name]].drop_duplicates(keep=keep)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def reindex(self, index=None, fill_value=None):
        """
        Conform Series to new index with optional filling logic, placing
        NA/NaN in locations having no value in the previous index. A new object
        is produced.

        Parameters
        ----------
        index: array-like, optional
            New labels / index to conform to, should be specified using keywords.
            Preferably an Index object to avoid duplicating data
        fill_value : scalar, default np.NaN
            Value to use for missing values. Defaults to NaN, but can be any
            "compatible" value.

        Returns
        -------
        Series with changed index.

        See Also
        --------
        Series.reset_index : Remove row labels or move them to new columns.

        Examples
        --------

        Create a series with some fictional data.

        >>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
        >>> ser = ks.Series([200, 200, 404, 404, 301],
        ...                 index=index, name='http_status')
        >>> ser
        Firefox      200
        Chrome       200
        Safari       404
        IE10         404
        Konqueror    301
        Name: http_status, dtype: int64

        Create a new index and reindex the Series. By default
        values in the new index that do not have corresponding
        records in the Series are assigned ``NaN``.

        >>> new_index= ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
        ...             'Chrome']
        >>> ser.reindex(new_index).sort_index()
        Chrome           200.0
        Comodo Dragon      NaN
        IE10             404.0
        Iceweasel          NaN
        Safari           404.0
        Name: http_status, dtype: float64

        We can fill in the missing values by passing a value to
        the keyword ``fill_value``.

        >>> ser.reindex(new_index, fill_value=0).sort_index()
        Chrome           200
        Comodo Dragon      0
        IE10             404
        Iceweasel          0
        Safari           404
        Name: http_status, dtype: int64

        To further illustrate the filling functionality in
        ``reindex``, we will create a Series with a
        monotonically increasing index (for example, a sequence
        of dates).

        >>> date_index = pd.date_range('1/1/2010', periods=6, freq='D')
        >>> ser2 = ks.Series([100, 101, np.nan, 100, 89, 88],
        ...                  name='prices', index=date_index)
        >>> ser2.sort_index()
        2010-01-01    100.0
        2010-01-02    101.0
        2010-01-03      NaN
        2010-01-04    100.0
        2010-01-05     89.0
        2010-01-06     88.0
        Name: prices, dtype: float64

        Suppose we decide to expand the series to cover a wider
        date range.

        >>> date_index2 = pd.date_range('12/29/2009', periods=10, freq='D')
        >>> ser2.reindex(date_index2).sort_index()
        2009-12-29      NaN
        2009-12-30      NaN
        2009-12-31      NaN
        2010-01-01    100.0
        2010-01-02    101.0
        2010-01-03      NaN
        2010-01-04    100.0
        2010-01-05     89.0
        2010-01-06     88.0
        2010-01-07      NaN
        Name: prices, dtype: float64
        """
        return first_series(self.to_frame().reindex(index=index, fill_value
            =fill_value)).rename(self.name)

    def reindex_like(self, other):
        """
        Return a Series with matching indices as other object.

        Conform the object to the same index on all axes. Places NA/NaN in locations
        having no value in the previous index.

        Parameters
        ----------
        other : Series or DataFrame
            Its row and column indices are used to define the new indices
            of this object.

        Returns
        -------
        Series
            Series with changed indices on each axis.

        See Also
        --------
        DataFrame.set_index : Set row labels.
        DataFrame.reset_index : Remove row labels or move them to new columns.
        DataFrame.reindex : Change to new indices or expand indices.

        Notes
        -----
        Same as calling
        ``.reindex(index=other.index, ...)``.

        Examples
        --------

        >>> s1 = ks.Series([24.3, 31.0, 22.0, 35.0],
        ...                index=pd.date_range(start='2014-02-12',
        ...                                    end='2014-02-15', freq='D'),
        ...                name="temp_celsius")
        >>> s1
        2014-02-12    24.3
        2014-02-13    31.0
        2014-02-14    22.0
        2014-02-15    35.0
        Name: temp_celsius, dtype: float64

        >>> s2 = ks.Series(["low", "low", "medium"],
        ...                index=pd.DatetimeIndex(['2014-02-12', '2014-02-13',
        ...                                        '2014-02-15']),
        ...                name="winspeed")
        >>> s2
        2014-02-12       low
        2014-02-13       low
        2014-02-15    medium
        Name: winspeed, dtype: object

        >>> s2.reindex_like(s1).sort_index()
        2014-02-12       low
        2014-02-13       low
        2014-02-14      None
        2014-02-15    medium
        Name: winspeed, dtype: object
        """
        if isinstance(other, (Series, DataFrame)):
            return self.reindex(index=other.index)
        else:
            raise TypeError('other must be a Koalas Series or DataFrame')

    def fillna(self, value=None, method=None, axis=None, inplace=False,
        limit=None):
        """
        Fill NA/NaN values.

        .. note:: the current implementation of 'method' parameter in fillna uses Spark's Window
            without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        value : scalar, dict, Series
            Value to use to fill holes. alternately a dict/Series of values
            specifying which value to use for each column.
            DataFrame is not supported.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series pad / ffill: propagate last valid
            observation forward to next valid backfill / bfill:
            use NEXT valid observation to fill gap
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values to
            forward/backward fill. In other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None

        Returns
        -------
        Series
            Series with NA entries filled.

        Examples
        --------
        >>> s = ks.Series([np.nan, 2, 3, 4, np.nan, 6], name='x')
        >>> s
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        4    NaN
        5    6.0
        Name: x, dtype: float64

        Replace all NaN elements with 0s.

        >>> s.fillna(0)
        0    0.0
        1    2.0
        2    3.0
        3    4.0
        4    0.0
        5    6.0
        Name: x, dtype: float64

        We can also propagate non-null values forward or backward.

        >>> s.fillna(method='ffill')
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        4    4.0
        5    6.0
        Name: x, dtype: float64

        >>> s = ks.Series([np.nan, 'a', 'b', 'c', np.nan], name='x')
        >>> s.fillna(method='ffill')
        0    None
        1       a
        2       b
        3       c
        4       c
        Name: x, dtype: object
        """
        kser = self._fillna(value=value, method=method, axis=axis, limit=limit)
        if method is not None:
            kser = DataFrame(kser._kdf._internal.resolved_copy)._kser_for(self
                ._column_label)
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if inplace:
            self._kdf._update_internal_frame(kser._kdf._internal,
                requires_same_anchor=False)
            return None
        else:
            return kser._with_new_scol(kser.spark.column)

    def _fillna(self, value=None, method=None, axis=None, limit=None,
        part_cols=()):
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                "fillna currently only works for axis=0 or axis='index'")
        if value is None and method is None:
            raise ValueError(
                "Must specify a fillna 'value' or 'method' parameter.")
        if method is not None and method not in ['ffill', 'pad', 'backfill',
            'bfill']:
            raise ValueError("Expecting 'pad', 'ffill', 'backfill' or 'bfill'."
                )
        scol: Column = self.spark.column
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
                raise ValueError('limit parameter for value is not support now'
                    )
            scol = F.when(cond, value).otherwise(scol)
        else:
            if method in ['ffill', 'pad']:
                func: Callable[[Column], Column] = F.last
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
            window: Window = Window.partitionBy(*part_cols).orderBy(
                NATURAL_ORDER_COLUMN_NAME).rowsBetween(begin, end)
            scol = F.when(cond, func(scol, True).over(window)).otherwise(scol)
        return DataFrame(self._kdf._internal.with_new_spark_column(self.
            _column_label, scol.alias(name_like_string(self.name))))

    def dropna(self, axis=0, inplace=False, **kwargs: Any):
        """
        Return a new Series with missing values removed.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            There is only one axis to drop values from.
        inplace : bool, default False
            If True, do operation inplace and return None.
        **kwargs
            Not in use.

        Returns
        -------
        Series
            Series with NA entries dropped from it.

        Examples
        --------
        >>> ser = ks.Series([1., 2., np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        dtype: float64

        Keep the Series with valid entries in the same variable.

        >>> ser.dropna(inplace=True)
        >>> ser
        0    1.0
        1    2.0
        dtype: float64
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf = self._kdf[[self.name]].dropna(axis=axis, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def clip(self, lower=None, upper=None):
        """
        Trim values at input threshold(s).

        Assigns values outside boundary to boundary values.

        Parameters
        ----------
        lower : float or int, default None
            Minimum threshold value. All values below this threshold will be set to it.
        upper : float or int, default None
            Maximum threshold value. All values above this threshold will be set to it.

        Returns
        -------
        Series
            Series with the values outside the clip boundaries replaced

        Examples
        --------
        >>> ks.Series([0, 2, 4]).clip(1, 3)
        0    1
        1    2
        2    3
        dtype: int64

        Notes
        -----
        One difference between this implementation and pandas is that running
        `pd.Series(['a', 'b']).clip(0, 1)` will crash with "TypeError: '<=' not supported between
        instances of 'str' and 'int'" while `ks.Series(['a', 'b']).clip(0, 1)` will output the
        original Series, simply ignoring the incompatible types.
        """
        if is_list_like(lower) or is_list_like(upper):
            raise ValueError(
                "List-like value are not supported for 'lower' and 'upper' at the "
                 + 'moment')
        if lower is None and upper is None:
            return self
        if isinstance(self.spark.data_type, NumericType):
            scol: Column = self.spark.column
            if lower is not None:
                scol = F.when(scol < lower, lower).otherwise(scol)
            if upper is not None:
                scol = F.when(scol > upper, upper).otherwise(scol)
            return self._with_new_scol(scol, dtype=self.dtype)
        else:
            return self

    def corr(self, other, method='pearson'):
        """
        Compute correlation with `other` Series, excluding missing values.

        Parameters
        ----------
        other : Series
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        correlation : float

        Examples
        --------
        >>> df = ks.DataFrame({'s1': [.2, .0, .6, .2],
        ...                    's2': [.3, .6, .0, .1]})
        >>> s1 = df.s1
        >>> s2 = df.s2
        >>> s1.corr(s2, method='pearson')  # doctest: +ELLIPSIS
        -0.851064...

        >>> s1.corr(s2, method='spearman')  # doctest: +ELLIPSIS
        -0.948683...

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * the `method` argument only accepts 'pearson', 'spearman'
        * the data should not contain NaNs. Koalas will return an error.
        * Koalas doesn't support the following argument(s).

          * `min_periods` argument is not supported
        """
        columns = ['__corr_arg1__', '__corr_arg2__']
        kdf = self._kdf.assign(__corr_arg1__=self, __corr_arg2__=other)[columns
            ]
        c: pd.Series = corr(kdf, method=method)
        return c.loc[tuple(columns)]

    def nsmallest(self, n=5):
        """
        Return the smallest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.

        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.

        See Also
        --------
        Series.nlargest: Get the `n` largest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values().head(n)`` for small `n` relative to
        the size of the ``Series`` object.
        In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Examples
        --------
        >>> data = [1, 2, 3, 4, np.nan ,6, 7, 8]
        >>> s = ks.Series(data)
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        4    NaN
        5    6.0
        6    7.0
        7    8.0
        dtype: float64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nsmallest()
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        5    6.0
        dtype: float64

        >>> s.nsmallest(3)
        0    1.0
        1    2.0
        2    3.0
        dtype: float64
        """
        return self.sort_values(ascending=True).head(n)

    def nlargest(self, n=5):
        """
        Return the largest `n` elements.

        Parameters
        ----------
        n : int, default 5

        Returns
        -------
        Series
            The `n` largest values in the Series, sorted in decreasing order.

        See Also
        --------
        Series.nsmallest: Get the `n` smallest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values(ascending=False).head(n)`` for small `n`
        relative to the size of the ``Series`` object.

        In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Examples
        --------
        >>> data = [1, 2, 3, 4, np.nan ,6, 7, 8]
        >>> s = ks.Series(data)
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        4    NaN
        5    6.0
        6    7.0
        7    8.0
        dtype: float64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nlargest()
        7    8.0
        6    7.0
        5    6.0
        3    4.0
        2    3.0
        dtype: float64

        >>> s.nlargest(n=3)
        7    8.0
        6    7.0
        5    6.0
        dtype: float64


        """
        return self.sort_values(ascending=False).head(n)

    def append(self, to_append, ignore_index=False, verify_integrity=False):
        """
        Concatenate two or more Series.

        Parameters
        ----------
        to_append : Series or list/tuple of Series
        ignore_index : boolean, default False
            If True, do not use the index labels.
        verify_integrity : boolean, default False
            If True, raise Exception on creating index with duplicates

        Returns
        -------
        appended : Series

        Examples
        --------

        >>> s1 = ks.Series([1, 2, 3])
        >>> s2 = ks.Series([4, 5, 6])
        >>> s3 = ks.Series([4, 5, 6], index=[3,4,5])

        >>> s1.append(s2)
        0    1
        1    2
        2    3
        0    4
        1    5
        2    6
        dtype: int64

        >>> s1.append(s3)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        dtype: int64

        With ignore_index set to True:

        >>> s1.append(s2, ignore_index=True)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        dtype: int64
        """
        return first_series(self.to_frame().append(to_append.to_frame(),
            ignore_index, verify_integrity)).rename(self.name)

    def sample(self, n=None, frac=None, replace=False, random_state=None):
        return first_series(self.to_frame().sample(n=n, frac=frac, replace=
            replace, random_state=random_state)).rename(self.name)
    sample.__doc__ = DataFrame.sample.__doc__

    def hist(self, bins=10, **kwds: Any):
        return self.plot.hist(bins, **kwds)
    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def apply(self, func, args=(), **kwds: Any):
        """
        Invoke function on values of Series.

        Can be a Python function that only works on the Series.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> np.int32:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

        Parameters
        ----------
        func : function
            Python function to apply. Note that type hint for return type is required.
        args : tuple
            Positional arguments passed to func after the series value.
        **kwds
            Additional keyword arguments passed to func.

        Returns
        -------
        Series

        See Also
        --------
        Series.aggregate : Only perform aggregating type operations.
        Series.transform : Only perform transforming type operations.
        DataFrame.apply : The equivalent function for DataFrame.

        Examples
        --------

        Create a Series with typical summer temperatures for each city.

        >>> s = ks.Series(range(3))
        >>> s
        0    0
        1    1
        2    2
        dtype: int64

        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x) -> np.int64:
        ...     return x ** 2
        >>> s.apply(square)
        0    0
        1    1
        2    4
        dtype: int64

        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword

        >>> def subtract_custom_value(x: Any, custom_value: int) -> Any:
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        0   -5
        1   -4
        2   -3
        dtype: int64

        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``

        >>> def add_custom_values(x: Any, **kwargs: Any) -> Any:
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        0    55
        1    56
        2    57
        dtype: int64

        Use a function from the Numpy library

        >>> def numpy_log(col: Any) -> float:
        ...     return np.log(col)
        >>> s.apply(numpy_log)
        0    -inf
        1     0.0
        2    0.693147
        dtype: float64

        You can omit the type hint and let Koalas infer its type.

        >>> s.apply(np.log)
        0    -inf
        1     0.0
        2    0.693147
        dtype: float64

        """
        assert callable(func
            ), 'the first argument should be a callable function.'
        try:
            spec = inspect.getfullargspec(func)
            return_sig = spec.annotations.get('return', None)
            should_infer_schema = return_sig is None
        except TypeError:
            should_infer_schema = True
        apply_each: Callable[[Series], 'Series'] = wraps(func)(lambda s: s.
            apply(func, args=args, **kwds))
        if should_infer_schema:
            return self.koalas._transform_batch(apply_each, None)
        else:
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, ScalarType):
                raise ValueError(
                    'Expected the return type of this function to be of scalar type, but found type {}'
                    .format(sig_return))
            return_type: ScalarType = cast(ScalarType, sig_return)
            return self.koalas._transform_batch(apply_each, return_type)

    def aggregate(self, func):
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : str or a list of str
            function name(s) as string apply to series.

        Returns
        -------
        scalar, Series
            The return can be:
            - scalar : when Series.agg is called with single function
            - Series : when Series.agg is called with several functions

        Notes
        -----
        `agg` is an alias for `aggregate`. Use the alias.

        See Also
        --------
        Series.apply : Invoke function on Series.
        Series.transform : Only perform transforming type operations.
        Series.groupby : Perform operations over groups.
        DataFrame.aggregate : The equivalent function for DataFrame.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s.agg('min')
        1

        >>> s.agg(['min', 'max']).sort_index()
        max    4
        min    1
        dtype: int64
        """
        if isinstance(func, list):
            return first_series(self.to_frame().aggregate(func)).rename(self
                .name)
        elif isinstance(func, str):
            return getattr(self, func)()
        else:
            raise ValueError('func must be a string or list of strings')
    agg: Callable[[Union[str, List[str]]], Union[Scalar, 'Series']] = aggregate

    def transpose(self, *args: Any, **kwargs: Any):
        """
        Return the transpose, which is by definition self.

        Examples
        --------
        It returns the same object as the transpose of the given series object, which is by
        definition self.

        >>> s = ks.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64

        >>> s.transpose()
        0    1
        1    2
        2    3
        dtype: int64
        """
        return self.copy()
    T: 'Series[T]' = property(transpose)

    def transform(self, func, axis=0, *args: Any, **kwargs: Any):
        """
        Call ``func`` producing the same type as `self` with transformed values
        and that has the same axis length as input.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> np.int32:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

        Parameters
        ----------
        func : function or list
            A function or a list of functions to use for transforming the data.
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        An instance of the same type with `self` that must have the same length as input.

        See Also
        --------
        Series.aggregate : Only perform aggregating type operations.
        Series.apply : Invoke function on Series.
        DataFrame.transform : The equivalent function for DataFrame.

        Examples
        --------

        >>> s = ks.Series(range(3))
        >>> s
        0    0
        1    1
        2    2
        dtype: int64

        >>> def sqrt(x) -> float:
        ...     return np.sqrt(x)
        >>> s.transform(sqrt)
        0    0.000000
        1    1.000000
        2    1.414214
        dtype: float64

        Even though the resulting instance must have the same length as the
        input, it is possible to provide several input functions:

        >>> def exp(x) -> float:
        ...     return np.exp(x)
        >>> s.transform([sqrt, exp])
               sqrt       exp
        0  0.000000  1.000000
        1  1.000000  2.718282
        2  1.414214  7.389056

        You can omit the type hint and let Koalas infer its type.

        >>> s.transform([np.sqrt, np.exp])
               sqrt       exp
        0  0.000000  1.000000
        1  1.000000  2.718282
        2  1.414214  7.389056
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        if isinstance(func, list):
            applied: List['Series'] = []
            for f in func:
                applied.append(self.apply(f, args=args, **kwargs).rename(f.
                    __name__))
            internal = self._internal.with_new_columns(applied)
            return DataFrame(internal)
        else:
            return self.apply(func, args=args, **kwargs)

    def transform_batch(self, func, *args: Any, **kwargs: Any):
        warnings.warn(
            'Series.transform_batch is deprecated as of Series.koalas.transform_batch. Please use the API instead.'
            , FutureWarning)
        return self.koalas.transform_batch(func, *args, **kwargs)
    transform_batch.__doc__ = KoalasSeriesMethods.transform_batch.__doc__

    def round(self, decimals=0):
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int
            Number of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of
            positions to the left of the decimal point.

        Returns
        -------
        Series object

        See Also
        --------
        DataFrame.round

        Examples
        --------
        >>> df = ks.Series([0.028208, 0.038683, 0.877076], name='x')
        >>> df
        0    0.028208
        1    0.038683
        2    0.877076
        Name: x, dtype: float64

        >>> df.round(2)
        0    0.03
        1    0.04
        2    0.88
        Name: x, dtype: float64
        """
        if not isinstance(decimals, int):
            raise ValueError('decimals must be an integer')
        scol: Column = F.round(self.spark.column, decimals)
        return self._with_new_scol(scol)

    def quantile(self, q=0.5, accuracy=10000):
        """
        Return value at the given quantile.

        .. note:: Unlike pandas', the quantile in Koalas is an approximated quantile based upon
            approximate percentile computation because computing quantile across a large dataset
            is extremely expensive.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute.
        accuracy : int, optional
            Default accuracy of approximation. Larger value means better accuracy.
            The relative error can be deduced by 1.0 / accuracy.

        Returns
        -------
        float or Series
            If the current object is a Series and ``q`` is an array, a Series will be
            returned where the index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4, 5])
        >>> s.quantile(.5)
        3.0

        >>> (s + 1).quantile(.5)
        4.0

        >>> s.quantile([.25, .5, .75])
        0.25    2.0
        0.50    3.0
        0.75    4.0
        dtype: float64

        >>> (s + 1).quantile([.25, .5, .75])
        0.25    3.0
        0.50    4.0
        0.75    5.0
        dtype: float64
        """
        if isinstance(q, Iterable):
            return first_series(self.to_frame().quantile(q=q, axis=0,
                numeric_only=False, accuracy=accuracy)).rename(self.name)
        else:
            if not isinstance(accuracy, int):
                raise ValueError(
                    'accuracy must be an integer; however, got [%s]' % type
                    (accuracy).__name__)
            if not isinstance(q, float):
                raise ValueError(
                    'q must be a float or an array of floats; however, [%s] found.'
                     % type(q))
            if q < 0.0 or q > 1.0:
                raise ValueError(
                    'percentiles should all be in the interval [0, 1].')

            def quantile(spark_column, spark_type):
                if isinstance(spark_type, (BooleanType, NumericType)):
                    return SF.percentile_approx(spark_column.cast(
                        DoubleType()), q, accuracy)
                else:
                    raise TypeError('Could not convert {} ({}) to numeric'.
                        format(spark_type_to_pandas_dtype(spark_type),
                        spark_type.simpleString()))
            return self._reduce_for_stat_function(quantile, name='quantile')

    def rank(self, method='average', ascending=True):
        """
        Compute numerical data ranks (1 through n) along axis. Equal values are
        assigned a rank that is the average of the ranks of those values.

        .. note:: the current implementation of rank uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}
            * average: average rank of group
            * min: lowest rank in group
            * max: highest rank in group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups
        ascending : boolean, default True
            False for ranks by high (1) to low (N)

        Returns
        -------
        ranks : same type as caller

        Examples
        --------
        >>> s = ks.Series([1, 2, 2, 3], name='A')
        >>> s
        0    1
        1    2
        2    2
        3    3
        dtype: int64

        >>> s.rank()
        0    1.0
        1    2.5
        2    2.5
        3    4.0
        dtype: float64

        If method is set to 'min', it use lowest rank in group.

        >>> s.rank(method='min')
        0    1.0
        1    2.0
        2    2.0
        3    4.0
        dtype: float64

        If method is set to 'max', it use highest rank in group.

        >>> s.rank(method='max')
        0    1.0
        1    3.0
        2    3.0
        3    4.0
        dtype: float64

        If method is set to 'first', it is assigned rank in order without groups.

        >>> s.rank(method='first')
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        dtype: float64

        If method is set to 'dense', it leaves no gaps in group.

        >>> s.rank(method='dense')
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        dtype: float64
        """
        return self._rank(method, ascending).spark.analyzed

    def _rank(self, method='average', ascending=True, *, part_cols: Tuple[
        Any, ...]=()):
        if method not in ['average', 'min', 'max', 'first', 'dense']:
            msg = (
                "method must be one of 'average', 'min', 'max', 'first', 'dense'"
                )
            raise ValueError(msg)
        if self._internal.index_level > 1:
            raise ValueError('rank do not support index now')
        if ascending:
            asc_func = lambda scol: scol.asc()
        else:
            asc_func = lambda scol: scol.desc()
        if method == 'first':
            window = Window.orderBy(asc_func(self.spark.column), asc_func(F
                .col(NATURAL_ORDER_COLUMN_NAME))).partitionBy(*part_cols
                ).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            scol: Column = F.row_number().over(window)
        elif method == 'dense':
            window = Window.orderBy(asc_func(self.spark.column)).partitionBy(*
                part_cols).rowsBetween(Window.unboundedPreceding, Window.
                currentRow)
            scol = F.dense_rank().over(window)
        else:
            if method == 'average':
                stat_func: Callable[[Column], Column] = F.mean
            elif method == 'min':
                stat_func = F.min
            elif method == 'max':
                stat_func = F.max
            window1 = Window.orderBy(asc_func(self.spark.column)).partitionBy(*
                part_cols).rowsBetween(Window.unboundedPreceding, Window.
                currentRow)
            window2 = Window.partitionBy([self.spark.column] + list(part_cols)
                ).rowsBetween(Window.unboundedPreceding, Window.
                unboundedFollowing)
            scol = stat_func(F.row_number().over(window1)).over(window2)
        kser: 'Series[T]' = self._with_new_scol(scol).astype(np.float64)
        return kser

    def iteritems(self):
        """
        Lazily iterate over (index, value) tuples.

        This method returns an iterable tuple (index, value). This is
        convenient if you want to create a lazy iterator.

        .. note:: Unlike pandas', the iteritems in Koalas returns generator rather zip object

        Returns
        -------
        iterable
            Iterable of tuples containing the (index, value) pairs from a
            Series.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.

        Examples
        --------
        >>> s = ks.Series(['A', 'B', 'C'])
        >>> for index, value in s.items():
        ...     print("Index : {}, Value : {}".format(index, value))
        Index : 0, Value : A
        Index : 1, Value : B
        Index : 2, Value : C
        """
        internal_index_columns: List[str
            ] = self._internal.index_spark_column_names
        internal_data_column: str = self._internal.data_spark_column_names[0]

        def extract_kv_from_spark_row(row):
            k: Union[Any, Tuple[Any, ...]]
            v: Any
            if len(internal_index_columns) == 1:
                k = row[internal_index_columns[0]]
            else:
                k = tuple(row[c] for c in internal_index_columns)
            v = row[internal_data_column]
            return k, v
        return (extract_kv_from_spark_row(row) for row in self._internal.
            resolved_copy.spark_frame.toLocalIterator())

    def items(self):
        """This is an alias of ``iteritems``."""
        return self.iteritems()

    def droplevel(self, level):
        """
        Return Series with requested index level(s) removed.

        Parameters
        ----------
        level : int, str, or list-like
            If a string is given, must be the name of a level
            If list-like, elements must be names or positional indexes
            of levels.

        Returns
        -------
        Series
            Series with requested index level(s) removed.

        Examples
        --------
        >>> kser = ks.Series(
        ...     [1, 2, 3],
        ...     index=pd.MultiIndex.from_tuples(
        ...         [("x", "a"), ("x", "b"), ("y", "c")], names=["level_1", "level_2"]
        ...     ),
        ... )
        >>> kser
        level_1  level_2
        x        a          1
                b          2
        y        c          3
        dtype: int64

        Removing specific index level by level

        >>> kser.droplevel(0)
        level_2
        a    1
        b    2
        c    3
        dtype: int64

        Removing specific index level by name

        >>> kser.droplevel("level_2")
        level_1
        x    1
        x    2
        y    3
        dtype: int64
        """
        return first_series(self.to_frame().droplevel(level=level, axis=0)
            ).rename(self.name)

    def tail(self, n=5):
        """
        Return the last `n` rows.

        This function returns last `n` rows from the object based on
        position. It is useful for quickly verifying data, for example,
        after sorting or appending rows.

        For negative values of `n`, this function returns all rows except
        the first `n` rows, equivalent to ``df[n:]``.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        type of caller
            The last `n` rows of the caller object.

        See Also
        --------
        DataFrame.head : The first `n` rows of the caller object.

        Examples
        --------
        >>> kser = ks.Series([1, 2, 3, 4, 5])
        >>> kser
        0    1
        1    2
        2    3
        3    4
        4    5
        dtype: int64

        >>> kser.tail(3)  # doctest: +SKIP
        2    3
        3    4
        4    5
        dtype: int64
        """
        return first_series(self.to_frame().tail(n=n)).rename(self.name)

    def explode(self):
        """
        Transform each element of a list-like to a row.

        Returns
        -------
        Series
            Exploded lists to rows; index will be duplicated for these rows.

        See Also
        --------
        Series.str.split : Split string values on specified separator.
        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
            to produce DataFrame.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

        Examples
        --------
        >>> kser = ks.Series([[1, 2, 3], [], [3, 4]])
        >>> kser
        0    [1, 2, 3]
        1           []
        2       [3, 4]
        dtype: object

        >>> kser.explode()  # doctest: +SKIP
        0    1.0
        0    2.0
        0    3.0
        1    NaN
        2    3.0
        2    4.0
        dtype: float64
        """
        if not isinstance(self.spark.data_type, ArrayType):
            return self.copy()
        scol: Column = F.explode_outer(self.spark.column).alias(
            name_like_string(self._column_label))
        internal = self._internal.with_new_columns([scol], keep_order=False)
        return first_series(DataFrame(internal))

    def corr(self, other, method='pearson'):
        """
        Compute correlation with `other` Series, excluding missing values.

        Parameters
        ----------
        other : Series, DataFrame
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        correlation : float

        Examples
        --------
        >>> df = ks.DataFrame({'s1': [.2, .0, .6, .2],
        ...                    's2': [.3, .6, .0, .1]})
        >>> s1 = df.s1
        >>> s2 = df.s2
        >>> s1.corr(s2, method='pearson')  # doctest: +ELLIPSIS
        -0.851064...

        >>> s1.corr(s2, method='spearman')  # doctest: +ELLIPSIS
        -0.948683...

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * the `method` argument only accepts 'pearson', 'spearman'
        * the data should not contain NaNs. Koalas will return an error.
        * Koalas doesn't support the following argument(s).

          * `min_periods` argument is not supported
        """
        columns = ['__corr_arg1__', '__corr_arg2__']
        kdf = self._kdf.assign(__corr_arg1__=self, __corr_arg2__=other)[columns
            ]
        c: pd.Series = corr(kdf, method=method)
        return c.loc[tuple(columns)]

    def at_time(self, time, asof=False, axis=0):
        """
        Select values at particular time of day (e.g., 9:30AM).

        Parameters
        ----------
        time : datetime.time or str
        axis : {0 or 'index', 1 or 'columns'}, default 0

        Returns
        -------
        Series

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        between_time : Select values between particular times of the day.
        DatetimeIndex.indexer_at_time : Get just the index locations for
            values at particular time of the day.

        Examples
        --------
        >>> idx = pd.date_range('2018-04-09', periods=4, freq='12H')
        >>> kser = ks.Series([1, 2, 3, 4], index=idx)
        >>> kser
        2018-04-09 00:00:00    1
        2018-04-09 12:00:00    2
        2018-04-10 00:00:00    3
        2018-04-10 12:00:00    4
        dtype: int64

        >>> kser.at_time('12:00')
        2018-04-09 12:00:00    2
        2018-04-10 12:00:00    4
        dtype: int64
        """
        return first_series(self.to_frame().at_time(time, asof, axis)).rename(
            self.name)

    def _cum(self, func, skipna, part_cols=(), ascending=True):
        if ascending:
            window = Window.orderBy(F.asc(NATURAL_ORDER_COLUMN_NAME)
                ).partitionBy(*part_cols).rowsBetween(Window.
                unboundedPreceding, Window.currentRow)
        else:
            window = Window.orderBy(F.desc(NATURAL_ORDER_COLUMN_NAME)
                ).partitionBy(*part_cols).rowsBetween(Window.
                unboundedPreceding, Window.currentRow)
        if skipna:
            scol: Column = F.when(self.spark.column.isNull(), None).otherwise(
                func(self.spark.column).over(window))
        else:
            scol = F.when(F.max(self.spark.column.isNull()).over(window), None
                ).otherwise(func(self.spark.column).over(window))
        return self._with_new_scol(scol)

    def _cumsum(self, skipna, part_cols=()):
        kser: 'Series[T]' = self
        if isinstance(kser.spark.data_type, BooleanType):
            kser = kser.spark.transform(lambda scol: scol.cast(LongType()))
        elif not isinstance(kser.spark.data_type, NumericType):
            raise TypeError('Could not convert {} ({}) to numeric'.format(
                spark_type_to_pandas_dtype(kser.spark.data_type), kser.
                spark.data_type.simpleString()))
        return kser._cum(F.sum, skipna, part_cols)

    def _cumprod(self, skipna, part_cols=()):
        if isinstance(self.spark.data_type, BooleanType):
            scol: Column = self._cum(lambda scol: F.min(F.coalesce(scol, F.
                lit(True))), skipna, part_cols).spark.column.cast(LongType())
        elif isinstance(self.spark.data_type, NumericType):
            num_zeros: Column = self._cum(lambda scol: F.sum(F.when(scol ==
                0, 1).otherwise(0)), skipna, part_cols).spark.column
            num_negatives: Column = self._cum(lambda scol: F.sum(F.when(
                scol < 0, 1).otherwise(0)), skipna, part_cols).spark.column
            sign: Column = F.when(num_negatives % 2 == 0, 1).otherwise(-1)
            abs_prod: Column = F.exp(self._cum(lambda scol: F.sum(F.log(F.
                abs(scol))), skipna, part_cols).spark.column)
            scol: Column = F.when(num_zeros > 0, 0).otherwise(sign * abs_prod)
            if isinstance(self.spark.data_type, IntegralType):
                scol = F.round(scol).cast(LongType())
        else:
            raise TypeError('Could not convert {} ({}) to numeric'.format(
                spark_type_to_pandas_dtype(self.spark.data_type), self.
                spark.data_type.simpleString()))
        return self._with_new_scol(scol)
    dt: 'DatetimeMethods' = CachedAccessor('dt', DatetimeMethods)
    str: 'StringMethods' = CachedAccessor('str', StringMethods)
    cat: 'CategoricalAccessor' = CachedAccessor('cat', CategoricalAccessor)
    plot: 'KoalasPlotAccessor' = CachedAccessor('plot', KoalasPlotAccessor)

    def _apply_series_op(self, op, should_resolve=False):
        kser: 'Series[T]' = op(self)
        if should_resolve:
            internal: InternalFrame = kser._internal.resolved_copy
            return first_series(DataFrame(internal))
        else:
            return kser

    def _reduce_for_stat_function(self, sfun, name, axis=None, numeric_only
        =None, **kwargs: Any):
        """
        Applies sfun to the column and returns a scalar

        Parameters
        ----------
        sfun : the stats function to be used for aggregation
        name : original pandas API name.
        axis : used only for sanity check because series only support index axis.
        numeric_only : not used by this implementation, but passed down by stats functions
        """
        from inspect import signature
        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError('Series does not support columns axis.')
        num_args = len(signature(sfun).parameters)
        spark_column: Column = self.spark.column
        spark_type = self.spark.data_type
        if num_args == 1:
            scol: Column = sfun(spark_column)
        else:
            assert num_args == 2
            scol = sfun(spark_column, spark_type)
        min_count: int = kwargs.get('min_count', 0)
        if min_count > 0:
            scol = F.when(F.col('count') >= min_count, scol)
        result: Any = unpack_scalar(self._internal.spark_frame.select(scol))
        return result if result is not None else np.nan

    def empty(self):
        """
        Whether Series is entirely empty (no elements).

        >>> ks.Series([]).empty
        True
        >>> ks.Series([1, 2, 3]).empty
        False
        """
        return len(self) == 0

    def len(self):
        """
        Return the length of the Series.

        >>> s = ks.Series([1, 2, 3])
        >>> len(s)
        3
        """
        return len(self)

    def __len__(self):
        return self.len()

    def __repr__(self):
        max_display_count: Optional[int] = get_option('display.max_rows')
        if max_display_count is None:
            return self._to_internal_pandas().to_string(name=self.name,
                dtype=self.dtype)
        pser: pd.Series = self._kdf._get_or_create_repr_pandas_cache(
            max_display_count)[self.name]
        pser_length: int = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string: str = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit('\n', 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length: str = match.group('length')
                dtype_name: str = str(self.dtype.name)
                if self.name is None:
                    footer: str = (
                        '\ndtype: {dtype}\nShowing only the first {length}'
                        .format(length=length, dtype=pprint_thing(dtype_name)))
                else:
                    footer = (
                        '\nName: {name}, dtype: {dtype}\nShowing only the first {length}'
                        .format(length=length, name=self.name, dtype=
                        pprint_thing(dtype_name)))
                return rest + footer
        return pser.to_string(name=self.name, dtype=self.dtype)

    def __dir__(self):
        if not isinstance(self.spark.data_type, StructType):
            fields: List[str] = []
        else:
            fields = [f for f in self.spark.data_type.fieldNames() if ' '
                 not in f]
        return super().__dir__() + fields

    def __iter__(self):
        return MissingPandasLikeSeries.__iter__(self)
    if sys.version_info >= (3, 7):

        @classmethod
        def __class_getitem__(cls, params):
            return _create_type_for_series_type(params)
    elif (3, 5) <= sys.version_info < (3, 7):
        is_series = None


def unpack_scalar(sdf):
    """
    Takes a dataframe that is supposed to contain a single row with a single scalar value,
    and returns this value.
    """
    l = sdf.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row: pd.Series = l.iloc[0]
    l2: List[Any] = list(row)
    assert len(l2) == 1, (row, l2)
    return l2[0]


def first_series(df):
    """
    Takes a DataFrame and returns the first column of the DataFrame as a Series
    """
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    if isinstance(df, DataFrame):
        return df._kser_for(df._internal.column_labels[0])
    else:
        return df[df.columns[0]]
