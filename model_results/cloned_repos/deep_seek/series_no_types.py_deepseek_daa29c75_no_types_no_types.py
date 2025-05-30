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
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar, Union, cast
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

    def _with_new_scol(self, scol, *, dtype=None):
        """
        Copy Koalas Series with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Series
        """
        internal = self._internal.copy(data_spark_columns=[scol.alias(
            name_like_string(self._column_label))], data_dtypes=[dtype])
        return first_series(DataFrame(internal))
    spark = CachedAccessor('spark', SparkSeriesMethods)

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
    divide = div

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
    multiply = mul

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
    subtract = sub

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
    koalas = CachedAccessor('koalas', KoalasSeriesMethods)

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
    equals = eq

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
            current = F.when(F.lit(False), F.lit(None).cast(self.spark.
                data_type))
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

    def rename(self, index=None, **kwargs):
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
        scol = self.spark.column.alias(name_like_string(index))
        internal = self._internal.copy(column_labels=[index],
            data_spark_columns=[scol], column_label_names=None)
        kdf = DataFrame(internal)
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
        scol = self.spark.column
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
            kseries = self.head(max_rows)
        else:
            kseries = self
        return validate_arguments_and_invoke_function(kseries.
            _to_internal_pandas(), self.to_string, pd.Series.to_string, args)

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(kseries.
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
        kseries = self
        return validate_arguments_and_invoke_function(kseries.
            _to_internal_pandas(), self.to_dict, pd.Series.to_dict, args)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True,
        index=True, na_rep='NaN', formatters=None, float_format=None,
        sparsify=None, index_names=True, bold_rows=False, column_format=
        None, longtable=None, escape=None, encoding=None, decimal='.',
        multicolumn=None, multicolumn_format=None, multirow=None):
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(kseries.
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
        """Fill NA/NaN values.

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
                raise ValueError('limit parameter for value is not support now'
                    )
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
            window = Window.partitionBy(*part_cols).orderBy(
                NATURAL_ORDER_COLUMN_NAME).rowsBetween(begin, end)
            scol = F.when(cond, func(scol, True).over(window)).otherwise(scol)
        return DataFrame(self._kdf._internal.with_new_spark_column(self.
            _column_label, scol.alias(name_like_string(self.name))))._kser_for(
            self._column_label)

    def dropna(self, axis=0, inplace=False, **kwargs):
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
            scol = self.spark.column
            if lower is not None:
                scol = F.when(scol < lower, lower).otherwise(scol)
            if upper is not None:
                scol = F.when(scol > upper, upper).otherwise(scol)
            return self._with_new_scol(scol, dtype=self.dtype)
        else:
            return self

    def drop(self, labels=None, index=None, level=None):
        """
        Return Series with specified index labels removed.

        Remove elements of a Series based on specifying the index labels.
        When using a multi-index, labels on different levels can be removed by specifying the level.

        Parameters
        ----------
        labels : single label or list-like
            Index labels to drop.
        index : None
            Redundant for application on Series, but index can be used instead of labels.
        level : int or level name, optional
            For MultiIndex, level for which the labels will be removed.

        Returns
        -------
        Series
            Series with specified index labels removed.

        See Also
        --------
        Series.dropna

        Examples
        --------
        >>> s = ks.Series(data=np.arange(3), index=['A', 'B', 'C'])
        >>> s
        A    0
        B    1
        C    2
        dtype: int64

        Drop single label A

        >>> s.drop('A')
        B    1
        C    2
        dtype: int64

        Drop labels B and C

        >>> s.drop(labels=['B', 'C'])
        A    0
        dtype: int64

        With 'index' rather than 'labels' returns exactly same result.

        >>> s.drop(index='A')
        B    1
        C    2
        dtype: int64

        >>> s.drop(index=['B', 'C'])
        A    0
        dtype: int64

        Also support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        lama    speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.drop(labels='weight', level=1)
        lama    speed      45.0
                length      1.2
        cow     speed      30.0
                length      1.5
        falcon  speed     320.0
                length      0.3
        dtype: float64

        >>> s.drop(('lama', 'weight'))
        lama    speed      45.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.drop([('lama', 'speed'), ('falcon', 'weight')])
        lama    weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                length      0.3
        dtype: float64
        """
        return first_series(self._drop(labels=labels, index=index, level=level)
            )

    def _drop(self, labels=None, index=None, level=None):
        if labels is not None:
            if index is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'")
            return self._drop(index=labels, level=level)
        if index is not None:
            internal = self._internal
            if level is None:
                level = 0
            if level >= internal.index_level:
                raise ValueError(
                    "'level' should be less than the number of indexes")
            if is_name_like_tuple(index):
                index = [index]
            elif is_name_like_value(index):
                index = [(index,)]
            elif all(is_name_like_value(idxes, allow_tuple=False) for idxes in
                index):
                index = [(idex,) for idex in index]
            elif not all(is_name_like_tuple(idxes) for idxes in index):
                raise ValueError(
                    'If the given index is a list, it should only contains names as all tuples or all non tuples that contain index names'
                    )
            drop_index_scols = []
            for idxes in index:
                try:
                    index_scols = [(internal.index_spark_columns[lvl] ==
                        idx) for lvl, idx in enumerate(idxes, level)]
                except IndexError:
                    raise KeyError('Key length ({}) exceeds index depth ({})'
                        .format(internal.index_level, len(idxes)))
                drop_index_scols.append(reduce(lambda x, y: x & y, index_scols)
                    )
            cond = ~reduce(lambda x, y: x | y, drop_index_scols)
            return DataFrame(internal.with_filter(cond))
        else:
            raise ValueError(
                "Need to specify at least one of 'labels' or 'index'")

    def head(self, n=5):
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position.
        It is useful for quickly testing if your object has the right type of data in it.

        Parameters
        ----------
        n : Integer, default =  5

        Returns
        -------
        The first n rows of the caller object.

        Examples
        --------
        >>> df = ks.DataFrame({'animal':['alligator', 'bee', 'falcon', 'lion']})
        >>> df.animal.head(2)  # doctest: +NORMALIZE_WHITESPACE
        0     alligator
        1     bee
        Name: animal, dtype: object
        """
        return first_series(self.to_frame().head(n)).rename(self.name)

    def last(self, offset):
        """
        Select final periods of time series data based on a date offset.

        When having a Series with dates as index, this function can
        select the last few elements based on a date offset.

        Parameters
        ----------
        offset : str or DateOffset
            The offset length of the data that will be selected. For instance,
            '3D' will display all the rows having their index within the last 3 days.

        Returns
        -------
        Series
            A subset of the caller.

        Raises
        ------
        TypeError
            If the index is not a :class:`DatetimeIndex`

        Examples
        --------
        >>> index = pd.date_range('2018-04-09', periods=4, freq='2D')
        >>> kser = ks.Series([1, 2, 3, 4], index=index)
        >>> kser
        2018-04-09    1
        2018-04-11    2
        2018-04-13    3
        2018-04-15    4
        dtype: int64

        Get the rows for the last 3 days:

        >>> kser.last('3D')
        2018-04-13    3
        2018-04-15    4
        dtype: int64

        Notice the data for 3 last calendar days were returned, not the last
        3 observed days in the dataset, and therefore data for 2018-04-11 was
        not returned.
        """
        return first_series(self.to_frame().last(offset)).rename(self.name)

    def first(self, offset):
        """
        Select first periods of time series data based on a date offset.

        When having a Series with dates as index, this function can
        select the first few elements based on a date offset.

        Parameters
        ----------
        offset : str or DateOffset
            The offset length of the data that will be selected. For instance,
            '3D' will display all the rows having their index within the first 3 days.

        Returns
        -------
        Series
            A subset of the caller.

        Raises
        ------
        TypeError
            If the index is not a :class:`DatetimeIndex`

        Examples
        --------
        >>> index = pd.date_range('2018-04-09', periods=4, freq='2D')
        >>> kser = ks.Series([1, 2, 3, 4], index=index)
        >>> kser
        2018-04-09    1
        2018-04-11    2
        2018-04-13    3
        2018-04-15    4
        dtype: int64

        Get the rows for the first 3 days:

        >>> kser.first('3D')
        2018-04-09    1
        2018-04-11    2
        dtype: int64

        Notice the data for 3 first calendar days were returned, not the first
        3 observed days in the dataset, and therefore data for 2018-04-13 was
        not returned.
        """
        return first_series(self.to_frame().first(offset)).rename(self.name)

    def unique(self):
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        .. note:: This method returns newly created Series whereas pandas returns
                  the unique values as a NumPy array.

        Returns
        -------
        Returns the unique values as a Series.

        See Also
        --------
        Index.unique
        groupby.SeriesGroupBy.unique

        Examples
        --------
        >>> kser = ks.Series([2, 1, 3, 3], name='A')
        >>> kser.unique().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1
        ...  2
        ...  3
        Name: A, dtype: int64

        >>> ks.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()
        0   2016-01-01
        dtype: datetime64[ns]

        >>> kser.name = ('x', 'a')
        >>> kser.unique().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1
        ...  2
        ...  3
        Name: (x, a), dtype: int64
        """
        sdf = self._internal.spark_frame.select(self.spark.column).distinct()
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=None,
            column_labels=[self._column_label], data_spark_columns=[
            scol_for(sdf, self._internal.data_spark_column_names[0])],
            data_dtypes=[self.dtype], column_label_names=self._internal.
            column_label_names)
        return first_series(DataFrame(internal))

    def sort_values(self, ascending=True, inplace=False, na_position='last'):
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some criterion.

        Parameters
        ----------
        ascending : bool or list of bool, default True
             Sort ascending vs. descending. Specify list for multiple sort
             orders.  If this is a list of bools, must match the length of
             the by.
        inplace : bool, default False
             if True, perform operation in-place
        na_position : {'first', 'last'}, default 'last'
             `first` puts NaNs at the beginning, `last` puts NaNs at the end

        Returns
        -------
        sorted_obj : Series ordered by values.

        Examples
        --------
        >>> s = ks.Series([np.nan, 1, 3, 10, 5])
        >>> s
        0     NaN
        1     1.0
        2     3.0
        3    10.0
        4     5.0
        dtype: float64

        Sort values ascending order (default behaviour)

        >>> s.sort_values(ascending=True)
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        0     NaN
        dtype: float64

        Sort values descending order

        >>> s.sort_values(ascending=False)
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        dtype: float64

        Sort values inplace

        >>> s.sort_values(ascending=False, inplace=True)
        >>> s
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        dtype: float64

        Sort values putting NAs first

        >>> s.sort_values(na_position='first')
        0     NaN
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        dtype: float64

        Sort a series of strings

        >>> s = ks.Series(['z', 'b', 'd', 'a', 'c'])
        >>> s
        0    z
        1    b
        2    d
        3    a
        4    c
        dtype: object

        >>> s.sort_values()
        3    a
        1    b
        4    c
        2    d
        0    z
        dtype: object
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf = self._kdf[[self.name]]._sort(by=[self.spark.column],
            ascending=ascending, inplace=False, na_position=na_position)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def sort_index(self, axis=0, level=None, ascending=True, inplace=False,
        kind=None, na_position='last'):
        """
        Sort object by labels (along an axis)

        Parameters
        ----------
        axis : index, columns to direct sorting. Currently, only axis = 0 is supported.
        level : int or level name or list of ints or list of level names
            if not None, sort on values in specified index level(s)
        ascending : boolean, default True
            Sort ascending vs. descending
        inplace : bool, default False
            if True, perform operation in-place
        kind : str, default None
            Koalas does not allow specifying the sorting algorithm at the moment, default None
        na_position : {‘first’, ‘last’}, default ‘last’
            first puts NaNs at the beginning, last puts NaNs at the end. Not implemented for
            MultiIndex.

        Returns
        -------
        sorted_obj : Series

        Examples
        --------
        >>> df = ks.Series([2, 1, np.nan], index=['b', 'a', np.nan])

        >>> df.sort_index()
        a      1.0
        b      2.0
        NaN    NaN
        dtype: float64

        >>> df.sort_index(ascending=False)
        b      2.0
        a      1.0
        NaN    NaN
        dtype: float64

        >>> df.sort_index(na_position='first')
        NaN    NaN
        a      1.0
        b      2.0
        dtype: float64

        >>> df.sort_index(inplace=True)
        >>> df
        a      1.0
        b      2.0
        NaN    NaN
        dtype: float64

        >>> df = ks.Series(range(4), index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]], name='0')

        >>> df.sort_index()
        a  0    3
           1    2
        b  0    1
           1    0
        Name: 0, dtype: int64

        >>> df.sort_index(level=1)  # doctest: +SKIP
        a  0    3
        b  0    1
        a  1    2
        b  1    0
        Name: 0, dtype: int64

        >>> df.sort_index(level=[1, 0])
        a  0    3
        b  0    1
        a  1    2
        b  1    0
        Name: 0, dtype: int64
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf = self._kdf[[self.name]].sort_index(axis=axis, level=level,
            ascending=ascending, kind=kind, na_position=na_position)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def swaplevel(self, i=-2, j=-1, copy=True):
        """
        Swap levels i and j in a MultiIndex.
        Default is to swap the two innermost levels of the index.

        Parameters
        ----------
        i, j : int, str
            Level of the indices to be swapped. Can pass level name as string.
        copy : bool, default True
            Whether to copy underlying data. Must be True.

        Returns
        -------
        Series
            Series with levels swapped in MultiIndex.

        Examples
        --------
        >>> midx = pd.MultiIndex.from_arrays([['a', 'b'], [1, 2]], names = ['word', 'number'])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 1),
                    ('b', 2)],
                   names=['word', 'number'])
        >>> kser = ks.Series(['x', 'y'], index=midx)
        >>> kser
        word  number
        a     1         x
        b     2         y
        dtype: object
        >>> kser.swaplevel()
        number  word
        1       a       x
        2       b       y
        dtype: object
        >>> kser.swaplevel(0, 1)
        number  word
        1       a       x
        2       b       y
        dtype: object
        >>> kser.swaplevel('number', 'word')
        number  word
        1       a       x
        2       b       y
        dtype: object
        """
        assert copy is True
        return first_series(self.to_frame().swaplevel(i, j, axis=0)).rename(
            self.name)

    def swapaxes(self, i, j, copy=True):
        """
        Interchange axes and swap values axes appropriately.

        Parameters
        ----------
        i: {0 or 'index', 1 or 'columns'}. The axis to swap.
        j: {0 or 'index', 1 or 'columns'}. The axis to swap.
        copy : bool, default True.

        Returns
        -------
        Series

        Examples
        --------
        >>> kser = ks.Series([1, 2, 3], index=["x", "y", "z"])
        >>> kser
        x    1
        y    2
        z    3
        dtype: int64
        >>>
        >>> kser.swapaxes(0, 0)
        x    1
        y    2
        z    3
        dtype: int64
        """
        assert copy is True
        i = validate_axis(i)
        j = validate_axis(j)
        if not i == j == 0:
            raise ValueError('Axis must be 0 for Series')
        return self.copy()

    def add_prefix(self, prefix):
        """
        Prefix labels with string `prefix`.

        For Series, the row labels are prefixed.
        For DataFrame, the column labels are prefixed.

        Parameters
        ----------
        prefix : str
           The string to add before each label.

        Returns
        -------
        Series
           New Series with updated labels.

        See Also
        --------
        Series.add_suffix: Suffix column labels with string `suffix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_prefix('item_')
        item_0    1
        item_1    2
        item_2    3
        item_3    4
        dtype: int64
        """
        assert isinstance(prefix, str)
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame.select([F.concat(F.lit(prefix),
            index_spark_column).alias(index_spark_column_name) for 
            index_spark_column, index_spark_column_name in zip(internal.
            index_spark_columns, internal.index_spark_column_names)] +
            internal.data_spark_columns)
        return first_series(DataFrame(internal.with_new_sdf(sdf,
            index_dtypes=[None] * internal.index_level)))

    def add_suffix(self, suffix):
        """
        Suffix labels with string suffix.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        suffix : str
           The string to add after each label.

        Returns
        -------
        Series
           New Series with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.add_suffix('_item')
        0_item    1
        1_item    2
        2_item    3
        3_item    4
        dtype: int64
        """
        assert isinstance(suffix, str)
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame.select([F.concat(index_spark_column, F.
            lit(suffix)).alias(index_spark_column_name) for 
            index_spark_column, index_spark_column_name in zip(internal.
            index_spark_columns, internal.index_spark_column_names)] +
            internal.data_spark_columns)
        return first_series(DataFrame(internal.with_new_sdf(sdf,
            index_dtypes=[None] * internal.index_level)))

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
        kdf.columns = columns
        c = corr(kdf, method=method)
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

    def hist(self, bins=10, **kwds):
        return self.plot.hist(bins, **kwds)
    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def apply(self, func, args=(), **kwds):
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

        >>> s = ks.Series([20, 21, 12],
        ...               index=['London', 'New York', 'Helsinki'])
        >>> s
        London      20
        New York    21
        Helsinki    12
        dtype: int64


        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x) -> np.int64:
        ...     return x ** 2
        >>> s.apply(square)
        London      400
        New York    441
        Helsinki    144
        dtype: int64


        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword

        >>> def subtract_custom_value(x, custom_value) -> np.int64:
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        dtype: int64


        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``

        >>> def add_custom_values(x, **kwargs) -> np.int64:
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        dtype: int64


        Use a function from the Numpy library

        >>> def numpy_log(col) -> np.float64:
        ...     return np.log(col)
        >>> s.apply(numpy_log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        dtype: float64


        You can omit the type hint and let Koalas infer its type.

        >>> s.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
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
        apply_each = wraps(func)(lambda s: s.apply(func, args=args, **kwds))
        if should_infer_schema:
            return self.koalas._transform_batch(apply_each, None)
        else:
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, ScalarType):
                raise ValueError(
                    'Expected the return type of this function to be of scalar type, but found type {}'
                    .format(sig_return))
            return_type = cast(ScalarType, sig_return)
            return self.koalas._transform_batch(apply_each, return_type)

    def aggregate(self, func):
        """Aggregate using one or more operations over the specified axis.

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
        Series.apply : Invoke function on a Series.
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
    agg = aggregate

    def transpose(self, *args, **kwargs):
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
    T = property(transpose)

    def transform(self, func, axis=0, *args, **kwargs):
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
            applied = []
            for f in func:
                applied.append(self.apply(f, args=args, **kwargs).rename(f.
                    __name__))
            internal = self._internal.with_new_columns(applied)
            return DataFrame(internal)
        else:
            return self.apply(func, args=args, **kwargs)

    def transform_batch(self, func, *args, **kwargs):
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
        scol = F.round(self.spark.column, decimals)
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
        Name: A, dtype: int64

        >>> s.rank()
        0    1.0
        1    2.5
        2    2.5
        3    4.0
        Name: A, dtype: float64

        If method is set to 'min', it use lowest rank in group.

        >>> s.rank(method='min')
        0    1.0
        1    2.0
        2    2.0
        3    4.0
        Name: A, dtype: float64

        If method is set to 'max', it use highest rank in group.

        >>> s.rank(method='max')
        0    1.0
        1    3.0
        2    3.0
        3    4.0
        Name: A, dtype: float64

        If method is set to 'first', it is assigned rank in order without groups.

        >>> s.rank(method='first')
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        Name: A, dtype: float64

        If method is set to 'dense', it leaves no gaps in group.

        >>> s.rank(method='dense')
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        Name: A, dtype: float64
        """
        return self._rank(method, ascending).spark.analyzed

    def _rank(self, method='average', ascending=True, *, part_cols=()):
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
            scol = F.row_number().over(window)
        elif method == 'dense':
            window = Window.orderBy(asc_func(self.spark.column)).partitionBy(*
                part_cols).rowsBetween(Window.unboundedPreceding, Window.
                currentRow)
            scol = F.dense_rank().over(window)
        else:
            if method == 'average':
                stat_func = F.mean
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
        kser = self._with_new_scol(scol)
        return kser.astype(np.float64)

    def filter(self, items=None, like=None, regex=None, axis=None):
        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError('Series does not support columns axis.')
        return first_series(self.to_frame().filter(items=items, like=like,
            regex=regex, axis=axis)).rename(self.name)
    filter.__doc__ = DataFrame.filter.__doc__

    def describe(self, percentiles=None):
        return first_series(self.to_frame().describe(percentiles)).rename(self
            .name)
    describe.__doc__ = DataFrame.describe.__doc__

    def diff(self, periods=1):
        """
        First discrete difference of element.

        Calculates the difference of a Series element compared with another element in the
        DataFrame (default is the element in the same column of the previous row).

        .. note:: the current implementation of diff uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.

        Returns
        -------
        diffed : Series

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

        >>> df.b.diff()
        0    NaN
        1    0.0
        2    1.0
        3    1.0
        4    2.0
        5    3.0
        Name: b, dtype: float64

        Difference with previous value

        >>> df.c.diff(periods=3)
        0     NaN
        1     NaN
        2     NaN
        3    15.0
        4    21.0
        5    27.0
        Name: c, dtype: float64

        Difference with following value

        >>> df.c.diff(periods=-1)
        0    -3.0
        1    -5.0
        2    -7.0
        3    -9.0
        4   -11.0
        5     NaN
        Name: c, dtype: float64
        """
        return self._diff(periods).spark.analyzed

    def _diff(self, periods, *, part_cols=()):
        if not isinstance(periods, int):
            raise ValueError('periods should be an int; however, got [%s]' %
                type(periods).__name__)
        window = Window.partitionBy(*part_cols).orderBy(
            NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        scol = self.spark.column - F.lag(self.spark.column, periods).over(
            window)
        return self._with_new_scol(scol, dtype=self.dtype)

    def idxmax(self, skipna=True):
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.

        Returns
        -------
        Index
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Examples
        --------
        >>> s = ks.Series(data=[1, None, 4, 3, 5],
        ...               index=['A', 'B', 'C', 'D', 'E'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        E    5.0
        dtype: float64

        >>> s.idxmax()
        'E'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmax(skipna=False)
        nan

        In case of multi-index, you get a tuple:

        >>> index = pd.MultiIndex.from_arrays([
        ...     ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        >>> s = ks.Series(data=[1, None, 4, 5], index=index)
        >>> s
        first  second
        a      c         1.0
               d         NaN
        b      e         4.0
               f         5.0
        dtype: float64

        >>> s.idxmax()
        ('b', 'f')

        If multiple values equal the maximum, the first row label with that
        value is returned.

        >>> s = ks.Series([1, 100, 1, 100, 1, 100], index=[10, 3, 5, 2, 1, 8])
        >>> s
        10      1
        3     100
        5       1
        2     100
        1       1
        8     100
        dtype: int64

        >>> s.idxmax()
        3
        """
        sdf = self._internal.spark_frame
        scol = self.spark.column
        index_scols = self._internal.index_spark_columns
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_last()),
                NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_first()),
                NATURAL_ORDER_COLUMN_NAME)
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

    def idxmin(self, skipna=True):
        """
        Return the row label of the minimum value.

        If multiple values equal the minimum, the first row label with that
        value is returned.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.

        Returns
        -------
        Index
            Label of the minimum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        Series.idxmax : Return index *label* of the first occurrence
            of maximum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmin``. This method
        returns the label of the minimum, while ``ndarray.argmin`` returns
        the position. To get the position, use ``series.values.argmin()``.

        Examples
        --------
        >>> s = ks.Series(data=[1, None, 4, 0],
        ...               index=['A', 'B', 'C', 'D'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    0.0
        dtype: float64

        >>> s.idxmin()
        'D'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmin(skipna=False)
        nan

        In case of multi-index, you get a tuple:

        >>> index = pd.MultiIndex.from_arrays([
        ...     ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        >>> s = ks.Series(data=[1, None, 4, 0], index=index)
        >>> s
        first  second
        a      c         1.0
               d         NaN
        b      e         4.0
               f         0.0
        dtype: float64

        >>> s.idxmin()
        ('b', 'f')

        If multiple values equal the minimum, the first row label with that
        value is returned.

        >>> s = ks.Series([1, 100, 1, 100, 1, 100], index=[10, 3, 5, 2, 1, 8])
        >>> s
        10      1
        3     100
        5       1
        2     100
        1       1
        8     100
        dtype: int64

        >>> s.idxmin()
        10
        """
        sdf = self._internal.spark_frame
        scol = self.spark.column
        index_scols = self._internal.index_spark_columns
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_last()),
                NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_first()),
                NATURAL_ORDER_COLUMN_NAME)
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

    def pop(self, item):
        """
        Return item and drop from series.

        Parameters
        ----------
        item : str
            Label of index to be popped.

        Returns
        -------
        Value that is popped from series.

        Examples
        --------
        >>> s = ks.Series(data=np.arange(3), index=['A', 'B', 'C'])
        >>> s
        A    0
        B    1
        C    2
        dtype: int64

        >>> s.pop('A')
        0

        >>> s
        B    1
        C    2
        dtype: int64

        >>> s = ks.Series(data=np.arange(3), index=['A', 'A', 'C'])
        >>> s
        A    0
        A    1
        C    2
        dtype: int64

        >>> s.pop('A')
        A    0
        A    1
        dtype: int64

        >>> s
        C    2
        dtype: int64

        Also support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        lama    speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.pop('lama')
        speed      45.0
        weight    200.0
        length      1.2
        dtype: float64

        >>> s
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        Also support for MultiIndex with several indexs.

        >>> midx = pd.MultiIndex([['a', 'b', 'c'],
        ...                       ['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 0, 0, 0, 1, 1, 1],
        ...                       [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 0, 2]]
        ...  )
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...              index=midx)
        >>> s
        a  lama    speed      45.0
                   weight    200.0
                   length      1.2
           cow     speed      30.0
                   weight    250.0
                   length      1.5
        b  falcon  speed     320.0
                   speed       1.0
                   length      0.3
        dtype: float64

        >>> s.pop(('a', 'lama'))
        speed      45.0
        weight    200.0
        length      1.2
        dtype: float64

        >>> s
        a  cow     speed      30.0
                   weight    250.0
                   length      1.5
        b  falcon  speed     320.0
                   speed       1.0
                   length      0.3
        dtype: float64

        >>> s.pop(('b', 'falcon', 'speed'))
        (b, falcon, speed)    320.0
        (b, falcon, speed)      1.0
        dtype: float64
        """
        if not is_name_like_value(item):
            raise ValueError(
                "'key' should be string or tuple that contains strings")
        if not is_name_like_tuple(item):
            item = item,
        if self._internal.index_level < len(item):
            raise KeyError('Key length ({}) exceeds index depth ({})'.
                format(len(item), self._internal.index_level))
        internal = self._internal
        scols = internal.index_spark_columns[len(item):] + [self.spark.column]
        rows = [(internal.spark_columns[level] == index) for level, index in
            enumerate(item)]
        sdf = internal.spark_frame.filter(reduce(lambda x, y: x & y, rows)
            ).select(scols)
        kdf = self._drop(item)
        self._update_anchor(kdf)
        if self._internal.index_level == len(item):
            pdf = sdf.limit(2).toPandas()
            length = len(pdf)
            if length == 1:
                return pdf[internal.data_spark_column_names[0]].iloc[0]
            item_string = name_like_string(item)
            sdf = sdf.withColumn(SPARK_DEFAULT_INDEX_NAME, F.lit(str(
                item_string)))
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
                scol_for(sdf, SPARK_DEFAULT_INDEX_NAME)], column_labels=[
                self._column_label], data_dtypes=[self.dtype])
            return first_series(DataFrame(internal))
        else:
            internal = internal.copy(spark_frame=sdf, index_spark_columns=[
                scol_for(sdf, col) for col in internal.
                index_spark_column_names[len(item):]], index_dtypes=
                internal.index_dtypes[len(item):], index_names=self.
                _internal.index_names[len(item):], data_spark_columns=[
                scol_for(sdf, internal.data_spark_column_names[0])])
            return first_series(DataFrame(internal))

    def copy(self, deep=None):
        """
        Make a copy of this object's indices and data.

        Parameters
        ----------
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Returns
        -------
        copy : Series

        Examples
        --------
        >>> s = ks.Series([1, 2], index=["a", "b"])
        >>> s
        a    1
        b    2
        dtype: int64
        >>> s_copy = s.copy()
        >>> s_copy
        a    1
        b    2
        dtype: int64
        """
        return self._kdf.copy()._kser_for(self._column_label)

    def mode(self, dropna=True):
        """
        Return the mode(s) of the dataset.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series.

        Examples
        --------
        >>> s = ks.Series([0, 0, 1, 1, 1, np.nan, np.nan, np.nan])
        >>> s
        0    0.0
        1    0.0
        2    1.0
        3    1.0
        4    1.0
        5    NaN
        6    NaN
        7    NaN
        dtype: float64

        >>> s.mode()
        0    1.0
        dtype: float64

        If there are several same modes, all items are shown

        >>> s = ks.Series([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
        ...                np.nan, np.nan, np.nan])
        >>> s
        0     0.0
        1     0.0
        2     1.0
        3     1.0
        4     1.0
        5     2.0
        6     2.0
        7     2.0
        8     3.0
        9     3.0
        10    3.0
        11    NaN
        12    NaN
        13    NaN
        dtype: float64

        >>> s.mode().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1.0
        ...  2.0
        ...  3.0
        dtype: float64

        With 'dropna' set to 'False', we can also see NaN in the result

        >>> s.mode(False).sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1.0
        ...  2.0
        ...  3.0
        ...  NaN
        dtype: float64
        """
        ser_count = self.value_counts(dropna=dropna, sort=False)
        sdf_count = ser_count._internal.spark_frame
        most_value = ser_count.max()
        sdf_most_value = sdf_count.filter('count == {}'.format(most_value))
        sdf = sdf_most_value.select(F.col(SPARK_DEFAULT_INDEX_NAME).alias(
            SPARK_DEFAULT_SERIES_NAME))
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=None,
            column_labels=[None])
        return first_series(DataFrame(internal))

    def keys(self):
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.

        Examples
        --------
        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)

        >>> kser.keys()  # doctest: +SKIP
        MultiIndex([(  'lama',  'speed'),
                    (  'lama', 'weight'),
                    (  'lama', 'length'),
                    (   'cow',  'speed'),
                    (   'cow', 'weight'),
                    (   'cow', 'length'),
                    ('falcon',  'speed'),
                    ('falcon', 'weight'),
                    ('falcon', 'length')],
                   )
        """
        return self.index

    def replace(self, to_replace=None, value=None, regex=False):
        """
        Replace values given in to_replace with value.
        Values of the Series are replaced with other values dynamically.

        Parameters
        ----------
        to_replace : str, list, tuple, dict, Series, int, float, or None
            How to find the values that will be replaced.
            * numeric, str:

                - numeric: numeric values equal to to_replace will be replaced with value
                - str: string exactly matching to_replace will be replaced with value

            * list of str or numeric:

                - if to_replace and value are both lists or tuples, they must be the same length.
                - str and numeric rules apply as above.

            * dict:

                - Dicts can be used to specify different replacement values for different
                  existing values.
                  For example, {'a': 'b', 'y': 'z'} replaces the value ‘a’ with ‘b’ and ‘y’
                  with ‘z’. To use a dict in this way the value parameter should be None.
                - For a DataFrame a dict can specify that different values should be replaced
                  in different columns. For example, {'a': 1, 'b': 'z'} looks for the value 1
                  in column ‘a’ and the value ‘z’ in column ‘b’ and replaces these values with
                  whatever is specified in value.
                  The value parameter should not be None in this case.
                  You can treat this as a special case of passing two lists except that you are
                  specifying the column to search in.

            See the examples section for examples of each of these.

        value : scalar, dict, list, tuple, str default None
            Value to replace any values matching to_replace with.
            For a DataFrame a dict of values can be used to specify which value to use
            for each column (columns not in the dict will not be filled).
            Regular expressions, strings and lists or dicts of such objects are also allowed.

        Returns
        -------
        Series
            Object after replacement.

        Examples
        --------

        Scalar `to_replace` and `value`

        >>> s = ks.Series([0, 1, 2, 3, 4])
        >>> s
        0    0
        1    1
        2    2
        3    3
        4    4
        dtype: int64

        >>> s.replace(0, 5)
        0    5
        1    1
        2    2
        3    3
        4    4
        dtype: int64

        List-like `to_replace`

        >>> s.replace([0, 4], 5000)
        0    5000
        1       1
        2       2
        3       3
        4    5000
        dtype: int64

        >>> s.replace([1, 2, 3], [10, 20, 30])
        0     0
        1    10
        2    20
        3    30
        4     4
        dtype: int64

        Dict-like `to_replace`

        >>> s.replace({1: 1000, 2: 2000, 3: 3000, 4: 4000})
        0       0
        1    1000
        2    2000
        3    3000
        4    4000
        dtype: int64

        Also support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        lama    speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.replace(45, 450)
        lama    speed     450.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.replace([45, 30, 320], 500)
        lama    speed     500.0
                weight    200.0
                length      1.2
        cow     speed     500.0
                weight    250.0
                length      1.5
        falcon  speed     500.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.replace({45: 450, 30: 300})
        lama    speed     450.0
                weight    200.0
                length      1.2
        cow     speed     300.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64
        """
        if to_replace is None:
            return self.fillna(method='ffill')
        if not isinstance(to_replace, (str, list, tuple, dict, int, float)):
            raise ValueError(
                "'to_replace' should be one of str, list, tuple, dict, int, float"
                )
        if regex:
            raise NotImplementedError('replace currently not support for regex'
                )
        to_replace = list(to_replace) if isinstance(to_replace, tuple
            ) else to_replace
        value = list(value) if isinstance(value, tuple) else value
        if isinstance(to_replace, list) and isinstance(value, list):
            if not len(to_replace) == len(value):
                raise ValueError(
                    'Replacement lists must match in length. Expecting {} got {}'
                    .format(len(to_replace), len(value)))
            to_replace = {k: v for k, v in zip(to_replace, value)}
        if isinstance(to_replace, dict):
            is_start = True
            if len(to_replace) == 0:
                current = self.spark.column
            else:
                for to_replace_, value in to_replace.items():
                    cond = F.isnan(self.spark.column
                        ) | self.spark.column.isNull() if pd.isna(to_replace_
                        ) else self.spark.column == F.lit(to_replace_)
                    if is_start:
                        current = F.when(cond, value)
                        is_start = False
                    else:
                        current = current.when(cond, value)
                current = current.otherwise(self.spark.column)
        else:
            cond = self.spark.column.isin(to_replace)
            if np.array(pd.isna(to_replace)).any():
                cond = cond | F.isnan(self.spark.column
                    ) | self.spark.column.isNull()
            current = F.when(cond, value).otherwise(self.spark.column)
        return self._with_new_scol(current)

    def update(self, other):
        """
        Modify Series in place using non-NA values from passed Series. Aligns on index.

        Parameters
        ----------
        other : Series

        Examples
        --------
        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> s = ks.Series([1, 2, 3])
        >>> s.update(ks.Series([4, 5, 6]))
        >>> s.sort_index()
        0    4
        1    5
        2    6
        dtype: int64

        >>> s = ks.Series(['a', 'b', 'c'])
        >>> s.update(ks.Series(['d', 'e'], index=[0, 2]))
        >>> s.sort_index()
        0    d
        1    b
        2    e
        dtype: object

        >>> s = ks.Series([1, 2, 3])
        >>> s.update(ks.Series([4, 5, 6, 7, 8]))
        >>> s.sort_index()
        0    4
        1    5
        2    6
        dtype: int64

        >>> s = ks.Series([1, 2, 3], index=[10, 11, 12])
        >>> s
        10    1
        11    2
        12    3
        dtype: int64

        >>> s.update(ks.Series([4, 5, 6]))
        >>> s.sort_index()
        10    1
        11    2
        12    3
        dtype: int64

        >>> s.update(ks.Series([4, 5, 6], index=[11, 12, 13]))
        >>> s.sort_index()
        10    1
        11    4
        12    5
        dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> s = ks.Series([1, 2, 3])
        >>> s.update(ks.Series([4, np.nan, 6]))
        >>> s.sort_index()
        0    4.0
        1    2.0
        2    6.0
        dtype: float64

        >>> reset_option("compute.ops_on_diff_frames")
        """
        if not isinstance(other, Series):
            raise ValueError("'other' must be a Series")
        combined = combine_frames(self._kdf, other._kdf, how='leftouter')
        this_scol = combined['this']._internal.spark_column_for(self.
            _column_label)
        that_scol = combined['that']._internal.spark_column_for(other.
            _column_label)
        scol = F.when(that_scol.isNotNull(), that_scol).otherwise(this_scol
            ).alias(self._kdf._internal.spark_column_name_for(self.
            _column_label))
        internal = combined['this']._internal.with_new_spark_column(self.
            _column_label, scol)
        self._kdf._update_internal_frame(internal.resolved_copy,
            requires_same_anchor=False)

    def where(self, cond, other=np.nan):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : boolean Series
            Where cond is True, keep the original value. Where False,
            replace with corresponding value from other.
        other : scalar, Series
            Entries where cond is False are replaced with corresponding value from other.

        Returns
        -------
        Series

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> s1 = ks.Series([0, 1, 2, 3, 4])
        >>> s2 = ks.Series([100, 200, 300, 400, 500])
        >>> s1.where(s1 > 0).sort_index()
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        4    4.0
        dtype: float64

        >>> s1.where(s1 > 1, 10).sort_index()
        0    10
        1    10
        2     2
        3     3
        4     4
        dtype: int64

        >>> s1.where(s1 > 1, s1 + 100).sort_index()
        0    100
        1    101
        2      2
        3      3
        4      4
        dtype: int64

        >>> s1.where(s1 > 1, s2).sort_index()
        0    100
        1    200
        2      2
        3      3
        4      4
        dtype: int64

        >>> reset_option("compute.ops_on_diff_frames")
        """
        assert isinstance(cond, Series)
        should_try_ops_on_diff_frame = not same_anchor(cond, self
            ) or isinstance(other, Series) and not same_anchor(other, self)
        if should_try_ops_on_diff_frame:
            kdf = self.to_frame()
            tmp_cond_col = verify_temp_column_name(kdf, '__tmp_cond_col__')
            tmp_other_col = verify_temp_column_name(kdf, '__tmp_other_col__')
            kdf[tmp_cond_col] = cond
            kdf[tmp_other_col] = other
            condition = F.when(kdf[tmp_cond_col].spark.column, kdf.
                _kser_for(kdf._internal.column_labels[0]).spark.column
                ).otherwise(kdf[tmp_other_col].spark.column).alias(kdf.
                _internal.data_spark_column_names[0])
            internal = kdf._internal.with_new_columns([condition],
                column_labels=self._internal.column_labels)
            return first_series(DataFrame(internal))
        else:
            if isinstance(other, Series):
                other = other.spark.column
            condition = F.when(cond.spark.column, self.spark.column).otherwise(
                other).alias(self._internal.data_spark_column_names[0])
            return self._with_new_scol(condition)

    def mask(self, cond, other=np.nan):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : boolean Series
            Where cond is False, keep the original value. Where True,
            replace with corresponding value from other.
        other : scalar, Series
            Entries where cond is True are replaced with corresponding value from other.

        Returns
        -------
        Series

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> s1 = ks.Series([0, 1, 2, 3, 4])
        >>> s2 = ks.Series([100, 200, 300, 400, 500])
        >>> s1.mask(s1 > 0).sort_index()
        0    0.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        dtype: float64

        >>> s1.mask(s1 > 1, 10).sort_index()
        0     0
        1     1
        2    10
        3    10
        4    10
        dtype: int64

        >>> s1.mask(s1 > 1, s1 + 100).sort_index()
        0      0
        1      1
        2    102
        3    103
        4    104
        dtype: int64

        >>> s1.mask(s1 > 1, s2).sort_index()
        0      0
        1      1
        2    300
        3    400
        4    500
        dtype: int64

        >>> reset_option("compute.ops_on_diff_frames")
        """
        return self.where(~cond, other)

    def xs(self, key, level=None):
        """
        Return cross-section from the Series.

        This method takes a `key` argument to select data at a particular
        level of a MultiIndex.

        Parameters
        ----------
        key : label or tuple of label
            Label contained in the index, or partially in a MultiIndex.
        level : object, defaults to first n levels (n=1 or len(key))
            In case of a key partially contained in a MultiIndex, indicate
            which levels are used. Levels can be referred by label or position.

        Returns
        -------
        Series
            Cross-section from the original Series
            corresponding to the selected index levels.

        Examples
        --------
        >>> midx = pd.MultiIndex([['a', 'b', 'c'],
        ...                       ['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        a  lama    speed      45.0
                   weight    200.0
                   length      1.2
        b  cow     speed      30.0
                   weight    250.0
                   length      1.5
        c  falcon  speed     320.0
                   weight      1.0
                   length      0.3
        dtype: float64

        Get values at specified index

        >>> s.xs('a')
        lama  speed      45.0
              weight    200.0
              length      1.2
        dtype: float64

        Get values at several indexes

        >>> s.xs(('a', 'lama'))
        speed      45.0
        weight    200.0
        length      1.2
        dtype: float64

        Get values at specified index and level

        >>> s.xs('lama', level=1)
        a  speed      45.0
           weight    200.0
           length      1.2
        dtype: float64
        """
        if not isinstance(key, tuple):
            key = key,
        if level is None:
            level = 0
        internal = self._internal
        scols = internal.index_spark_columns[:level
            ] + internal.index_spark_columns[level + len(key):] + [self.
            spark.column]
        rows = [(internal.spark_columns[lvl] == index) for lvl, index in
            enumerate(key, level)]
        sdf = internal.spark_frame.filter(reduce(lambda x, y: x & y, rows)
            ).select(scols)
        if internal.index_level == len(key):
            pdf = sdf.limit(2).toPandas()
            length = len(pdf)
            if length == 1:
                return pdf[self._internal.data_spark_column_names[0]].iloc[0]
        index_spark_column_names = internal.index_spark_column_names[:level
            ] + internal.index_spark_column_names[level + len(key):]
        index_names = internal.index_names[:level] + internal.index_names[
            level + len(key):]
        index_dtypes = internal.index_dtypes[:level] + internal.index_dtypes[
            level + len(key):]
        internal = internal.copy(spark_frame=sdf, index_spark_columns=[
            scol_for(sdf, col) for col in index_spark_column_names],
            index_names=index_names, index_dtypes=index_dtypes,
            data_spark_columns=[scol_for(sdf, internal.
            data_spark_column_names[0])])
        return first_series(DataFrame(internal))

    def pct_change(self, periods=1):
        """
        Percentage change between the current and a prior element.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.

        Returns
        -------
        Series

        Examples
        --------

        >>> kser = ks.Series([90, 91, 85], index=[2, 4, 1])
        >>> kser
        2    90
        4    91
        1    85
        dtype: int64

        >>> kser.pct_change()
        2         NaN
        4    0.011111
        1   -0.065934
        dtype: float64

        >>> kser.sort_index().pct_change()
        1         NaN
        2    0.058824
        4    0.011111
        dtype: float64

        >>> kser.pct_change(periods=2)
        2         NaN
        4         NaN
        1   -0.055556
        dtype: float64
        """
        scol = self.spark.column
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-
            periods, -periods)
        prev_row = F.lag(scol, periods).over(window)
        return self._with_new_scol((scol - prev_row) / prev_row).spark.analyzed

    def combine_first(self, other):
        """
        Combine Series values, choosing the calling Series's values first.

        Parameters
        ----------
        other : Series
            The value(s) to be combined with the `Series`.

        Returns
        -------
        Series
            The result of combining the Series with the other object.

        See Also
        --------
        Series.combine : Perform elementwise operation on two Series
            using a given function.

        Notes
        -----
        Result index will be the union of the two indexes.

        Examples
        --------
        >>> s1 = ks.Series([1, np.nan])
        >>> s2 = ks.Series([3, 4])
        >>> with ks.option_context("compute.ops_on_diff_frames", True):
        ...     s1.combine_first(s2)
        0    1.0
        1    4.0
        dtype: float64
        """
        if not isinstance(other, ks.Series):
            raise ValueError(
                '`combine_first` only allows `Series` for parameter `other`')
        if same_anchor(self, other):
            this = self.spark.column
            that = other.spark.column
            combined = self._kdf
        else:
            combined = combine_frames(self._kdf, other._kdf)
            this = combined['this']._internal.spark_column_for(self.
                _column_label)
            that = combined['that']._internal.spark_column_for(other.
                _column_label)
        cond = F.when(this.isNull(), that).otherwise(this)
        if same_anchor(self, other):
            return self._with_new_scol(cond)
        index_scols = combined._internal.index_spark_columns
        sdf = combined._internal.spark_frame.select(*index_scols, cond.
            alias(self._internal.data_spark_column_names[0])).distinct()
        internal = self._internal.with_new_sdf(sdf, data_dtypes=[None])
        return first_series(DataFrame(internal))

    def dot(self, other):
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each columns of a DataFrame.

        It can also be called using `self @ other` in Python >= 3.5.

        .. note:: This API is slightly different from pandas when indexes from both Series
            are not aligned. To match with pandas', it requires to read the whole data for,
            for example, counting. pandas raises an exception; however, Koalas just proceeds
            and performs by ignoring mismatches with NaN permissively.

            >>> pdf1 = pd.Series([1, 2, 3], index=[0, 1, 2])
            >>> pdf2 = pd.Series([1, 2, 3], index=[0, 1, 3])
            >>> pdf1.dot(pdf2)  # doctest: +SKIP
            ...
            ValueError: matrices are not aligned

            >>> kdf1 = ks.Series([1, 2, 3], index=[0, 1, 2])
            >>> kdf2 = ks.Series([1, 2, 3], index=[0, 1, 3])
            >>> kdf1.dot(kdf2)  # doctest: +SKIP
            5

        Parameters
        ----------
        other : Series, DataFrame.
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each rows of
            other if other is a DataFrame.

        Notes
        -----
        The Series and other has to share the same index if other is a Series
        or a DataFrame.

        Examples
        --------
        >>> s = ks.Series([0, 1, 2, 3])

        >>> s.dot(s)
        14

        >>> s @ s
        14

        >>> kdf = ks.DataFrame({'x': [0, 1, 2, 3], 'y': [0, -1, -2, -3]})
        >>> kdf
           x  y
        0  0  0
        1  1 -1
        2  2 -2
        3  3 -3

        >>> with ks.option_context("compute.ops_on_diff_frames", True):
        ...     s.dot(kdf)
        ...
        x    14
        y   -14
        dtype: int64
        """
        if isinstance(other, DataFrame):
            if not same_anchor(self, other):
                if not self.index.sort_values().equals(other.index.
                    sort_values()):
                    raise ValueError('matrices are not aligned')
            other = other.copy()
            column_labels = other._internal.column_labels
            self_column_label = verify_temp_column_name(other,
                '__self_column__')
            other[self_column_label] = self
            self_kser = other._kser_for(self_column_label)
            product_ksers = [(other._kser_for(label) * self_kser) for label in
                column_labels]
            dot_product_kser = DataFrame(other._internal.with_new_columns(
                product_ksers, column_labels=column_labels)).sum()
            return cast(Series, dot_product_kser).rename(self.name)
        else:
            assert isinstance(other, Series)
            if not same_anchor(self, other):
                if len(self.index) != len(other.index):
                    raise ValueError('matrices are not aligned')
            return (self * other).sum()

    def __matmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.dot(other)

    def repeat(self, repeats):
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or Series
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.

        Returns
        -------
        Series
            Newly created Series with repeated elements.

        See Also
        --------
        Index.repeat : Equivalent function for Index.

        Examples
        --------
        >>> s = ks.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: object
        >>> s.repeat(2)
        0    a
        1    b
        2    c
        0    a
        1    b
        2    c
        dtype: object
        >>> ks.Series([1, 2, 3]).repeat(0)
        Series([], dtype: int64)
        """
        if not isinstance(repeats, (int, Series)):
            raise ValueError(
                '`repeats` argument must be integer or Series, but got {}'.
                format(type(repeats)))
        if isinstance(repeats, Series):
            if LooseVersion(pyspark.__version__) < LooseVersion('2.4'):
                raise ValueError(
                    '`repeats` argument must be integer with Spark<2.4, but got {}'
                    .format(type(repeats)))
            if not same_anchor(self, repeats):
                kdf = self.to_frame()
                temp_repeats = verify_temp_column_name(kdf, '__temp_repeats__')
                kdf[temp_repeats] = repeats
                return kdf._kser_for(kdf._internal.column_labels[0]).repeat(kdf
                    [temp_repeats]).rename(self.name)
            else:
                scol = F.explode(SF.array_repeat(self.spark.column, repeats
                    .astype('int32').spark.column)).alias(name_like_string(
                    self.name))
                sdf = self._internal.spark_frame.select(self._internal.
                    index_spark_columns + [scol])
                internal = self._internal.copy(spark_frame=sdf,
                    index_spark_columns=[scol_for(sdf, col) for col in self
                    ._internal.index_spark_column_names],
                    data_spark_columns=[scol_for(sdf, name_like_string(self
                    .name))])
                return first_series(DataFrame(internal))
        else:
            if repeats < 0:
                raise ValueError('negative dimensions are not allowed')
            kdf = self._kdf[[self.name]]
            if repeats == 0:
                return first_series(DataFrame(kdf._internal.with_filter(F.
                    lit(False))))
            else:
                return first_series(ks.concat([kdf] * repeats))

    def asof(self, where):
        """
        Return the last row(s) without any NaNs before `where`.

        The last row (for each element in `where`, if list) without any
        NaN is taken.

        If there is no good value, NaN is returned.

        .. note:: This API is dependent on :meth:`Index.is_monotonic_increasing`
            which can be expensive.

        Parameters
        ----------
        where : index or array-like of indices

        Returns
        -------
        scalar or Series

            The return can be:

            * scalar : when `self` is a Series and `where` is a scalar
            * Series: when `self` is a Series and `where` is an array-like

            Return scalar or Series

        Notes
        -----
        Indices are assumed to be sorted. Raises if this is not the case.

        Examples
        --------
        >>> s = ks.Series([1, 2, np.nan, 4], index=[10, 20, 30, 40])
        >>> s
        10    1.0
        20    2.0
        30    NaN
        40    4.0
        dtype: float64

        A scalar `where`.

        >>> s.asof(20)
        2.0

        For a sequence `where`, a Series is returned. The first value is
        NaN, because the first element of `where` is before the first
        index value.

        >>> s.asof([5, 20]).sort_index()
        5     NaN
        20    2.0
        dtype: float64

        Missing values are not considered. The following is ``2.0``, not
        NaN, even though NaN is at the index location for ``30``.

        >>> s.asof(30)
        2.0
        """
        should_return_series = True
        if isinstance(self.index, ks.MultiIndex):
            raise ValueError('asof is not supported for a MultiIndex')
        if isinstance(where, (ks.Index, ks.Series, DataFrame)):
            raise ValueError('where cannot be an Index, Series or a DataFrame')
        if not self.index.is_monotonic_increasing:
            raise ValueError('asof requires a sorted index')
        if not is_list_like(where):
            should_return_series = False
            where = [where]
        index_scol = self._internal.index_spark_columns[0]
        index_type = self._internal.spark_type_for(index_scol)
        cond = [F.max(F.when(index_scol <= F.lit(index).cast(index_type),
            self.spark.column)) for index in where]
        sdf = self._internal.spark_frame.select(cond)
        if not should_return_series:
            with sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
                result = sdf.limit(1).toPandas().iloc[0, 0]
            return result if result is not None else np.nan
        with ks.option_context('compute.default_index_type', 'distributed',
            'compute.max_rows', 1):
            kdf = ks.DataFrame(sdf)
            kdf.columns = pd.Index(where)
            return first_series(kdf.transpose()).rename(self.name)

    def mad(self):
        """
        Return the mean absolute deviation of values.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.mad()
        1.0
        """
        sdf = self._internal.spark_frame
        spark_column = self.spark.column
        avg = unpack_scalar(sdf.select(F.avg(spark_column)))
        mad = unpack_scalar(sdf.select(F.avg(F.abs(spark_column - avg))))
        return mad

    def unstack(self, level=-1):
        """
        Unstack, a.k.a. pivot, Series with MultiIndex to produce DataFrame.
        The level involved will automatically get sorted.

        Notes
        -----
        Unlike pandas, Koalas doesn't check whether an index is duplicated or not
        because the checking of duplicated index requires scanning whole data which
        can be quite expensive.

        Parameters
        ----------
        level : int, str, or list of these, default last level
            Level(s) to unstack, can pass level name.

        Returns
        -------
        DataFrame
            Unstacked Series.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4],
        ...               index=pd.MultiIndex.from_product([['one', 'two'],
        ...                                                 ['a', 'b']]))
        >>> s
        one  a    1
             b    2
        two  a    3
             b    4
        dtype: int64

        >>> s.unstack(level=-1).sort_index()
             a  b
        one  1  2
        two  3  4

        >>> s.unstack(level=0).sort_index()
           one  two
        a    1    3
        b    2    4
        """
        if not isinstance(self.index, ks.MultiIndex):
            raise ValueError('Series.unstack only support for a MultiIndex')
        index_nlevels = self.index.nlevels
        if level > 0 and level > index_nlevels - 1:
            raise IndexError(
                'Too many levels: Index has only {} levels, not {}'.format(
                index_nlevels, level + 1))
        elif level < 0 and level < -index_nlevels:
            raise IndexError(
                'Too many levels: Index has only {} levels, {} is not a valid level number'
                .format(index_nlevels, level))
        internal = self._internal.resolved_copy
        index_map = list(zip(internal.index_spark_column_names, internal.
            index_names))
        pivot_col, column_label_names = index_map.pop(level)
        index_scol_names, index_names = zip(*index_map)
        col = internal.data_spark_column_names[0]
        sdf = internal.spark_frame
        sdf = sdf.groupby(list(index_scol_names)).pivot(pivot_col).agg(F.
            first(scol_for(sdf, col)))
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
            scol_for(sdf, col) for col in index_scol_names], index_names=
            list(index_names), column_label_names=[column_label_names])
        return DataFrame(internal)

    def item(self):
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of Series.

        Raises
        ------
        ValueError
            If the data is not length-1.

        Examples
        --------
        >>> kser = ks.Series([10])
        >>> kser.item()
        10
        """
        return self.head(2)._to_internal_pandas().item()

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
        internal_index_columns = self._internal.index_spark_column_names
        internal_data_column = self._internal.data_spark_column_names[0]

        def extract_kv_from_spark_row(row):
            k = row[internal_index_columns[0]] if len(internal_index_columns
                ) == 1 else tuple(row[c] for c in internal_index_columns)
            v = row[internal_data_column]
            return k, v
        for k, v in map(extract_kv_from_spark_row, self._internal.
            resolved_copy.spark_frame.toLocalIterator()):
            yield k, v

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
        scol = F.explode_outer(self.spark.column).alias(name_like_string(
            self._column_label))
        internal = self._internal.with_new_columns([scol], keep_order=False)
        return first_series(DataFrame(internal))

    def argsort(self):
        """
        Return the integer indices that would sort the Series values.
        Unlike pandas, the index order is not preserved in the result.

        Returns
        -------
        Series
            Positions of values within the sort order with -1 indicating
            nan values.

        Examples
        --------
        >>> kser = ks.Series([3, 3, 4, 1, 6, 2, 3, 7, 8, 7, 10])
        >>> kser
        0      3
        1      3
        2      4
        3      1
        4      6
        5      2
        6      3
        7      7
        8      8
        9      7
        10    10
        dtype: int64

        >>> kser.argsort().sort_index()
        0      3
        1      5
        2      0
        3      1
        4      6
        5      2
        6      4
        7      7
        8      9
        9      8
        10    10
        dtype: int64
        """
        notnull = self.loc[self.notnull()]
        sdf_for_index = notnull._internal.spark_frame.select(notnull.
            _internal.index_spark_columns)
        tmp_join_key = verify_temp_column_name(sdf_for_index,
            '__tmp_join_key__')
        sdf_for_index = InternalFrame.attach_distributed_sequence_column(
            sdf_for_index, tmp_join_key)
        sdf_for_data = notnull._internal.spark_frame.select(notnull.spark.
            column.alias('values'), NATURAL_ORDER_COLUMN_NAME)
        sdf_for_data = InternalFrame.attach_distributed_sequence_column(
            sdf_for_data, SPARK_DEFAULT_SERIES_NAME)
        sdf_for_data = sdf_for_data.sort(scol_for(sdf_for_data, 'values'),
            NATURAL_ORDER_COLUMN_NAME).drop('values', NATURAL_ORDER_COLUMN_NAME
            )
        tmp_join_key = verify_temp_column_name(sdf_for_data, '__tmp_join_key__'
            )
        sdf_for_data = InternalFrame.attach_distributed_sequence_column(
            sdf_for_data, tmp_join_key)
        sdf = sdf_for_index.join(sdf_for_data, on=tmp_join_key).drop(
            tmp_join_key)
        internal = self._internal.with_new_sdf(spark_frame=sdf,
            data_columns=[SPARK_DEFAULT_SERIES_NAME], data_dtypes=[None])
        kser = first_series(DataFrame(internal))
        return cast(Series, ks.concat([kser, self.loc[self.isnull()].spark.
            transform(lambda _: F.lit(-1))]))

    def argmax(self):
        """
        Return int position of the largest value in the Series.

        If the maximum is achieved in multiple locations,
        the first row position is returned.

        Returns
        -------
        int
            Row position of the maximum value.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = ks.Series({'Corn Flakes': 100.0, 'Almond Delight': 110.0,
        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0})
        >>> s  # doctest: +SKIP
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmax()  # doctest: +SKIP
        2
        """
        sdf = self._internal.spark_frame.select(self.spark.column,
            NATURAL_ORDER_COLUMN_NAME)
        max_value = sdf.select(F.max(scol_for(sdf, self._internal.
            data_spark_column_names[0])), F.first(NATURAL_ORDER_COLUMN_NAME)
            ).head()
        if max_value[1] is None:
            raise ValueError('attempt to get argmax of an empty sequence')
        elif max_value[0] is None:
            return -1
        seq_col_name = verify_temp_column_name(sdf,
            '__distributed_sequence_column__')
        sdf = InternalFrame.attach_distributed_sequence_column(sdf.drop(
            NATURAL_ORDER_COLUMN_NAME), seq_col_name)
        return sdf.filter(scol_for(sdf, self._internal.
            data_spark_column_names[0]) == max_value[0]).head()[0]

    def argmin(self):
        """
        Return int position of the smallest value in the Series.

        If the minimum is achieved in multiple locations,
        the first row position is returned.

        Returns
        -------
        int
            Row position of the minimum value.

        Examples
        --------
        Consider dataset containing cereal calories

        >>> s = ks.Series({'Corn Flakes': 100.0, 'Almond Delight': 110.0,
        ...                'Cinnamon Toast Crunch': 120.0, 'Cocoa Puff': 110.0})
        >>> s  # doctest: +SKIP
        Corn Flakes              100.0
        Almond Delight           110.0
        Cinnamon Toast Crunch    120.0
        Cocoa Puff               110.0
        dtype: float64

        >>> s.argmin()  # doctest: +SKIP
        0
        """
        sdf = self._internal.spark_frame.select(self.spark.column,
            NATURAL_ORDER_COLUMN_NAME)
        min_value = sdf.select(F.min(scol_for(sdf, self._internal.
            data_spark_column_names[0])), F.first(NATURAL_ORDER_COLUMN_NAME)
            ).head()
        if min_value[1] is None:
            raise ValueError('attempt to get argmin of an empty sequence')
        elif min_value[0] is None:
            return -1
        seq_col_name = verify_temp_column_name(sdf,
            '__distributed_sequence_column__')
        sdf = InternalFrame.attach_distributed_sequence_column(sdf.drop(
            NATURAL_ORDER_COLUMN_NAME), seq_col_name)
        return sdf.filter(scol_for(sdf, self._internal.
            data_spark_column_names[0]) == min_value[0]).head()[0]

    def compare(self, other, keep_shape=False, keep_equal=False):
        """
        Compare to another Series and show the differences.

        Parameters
        ----------
        other : Series
            Object to compare with.
        keep_shape : bool, default False
            If true, all rows and columns are kept.
            Otherwise, only the ones with different values are kept.
        keep_equal : bool, default False
            If true, the result keeps values that are equal.
            Otherwise, equal values are shown as NaNs.

        Returns
        -------
        DataFrame

        Notes
        -----
        Matching NaNs will not appear as a difference.

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> s1 = ks.Series(["a", "b", "c", "d", "e"])
        >>> s2 = ks.Series(["a", "a", "c", "b", "e"])

        Align the differences on columns

        >>> s1.compare(s2).sort_index()
          self other
        1    b     a
        3    d     b

        Keep all original rows

        >>> s1.compare(s2, keep_shape=True).sort_index()
           self other
        0  None  None
        1     b     a
        2  None  None
        3     d     b
        4  None  None

        Keep all original rows and also all original values

        >>> s1.compare(s2, keep_shape=True, keep_equal=True).sort_index()
          self other
        0    a     a
        1    b     a
        2    c     c
        3    d     b
        4    e     e

        >>> reset_option("compute.ops_on_diff_frames")
        """
        if same_anchor(self, other):
            self_column_label = verify_temp_column_name(other.to_frame(),
                '__self_column__')
            other_column_label = verify_temp_column_name(self.to_frame(),
                '__other_column__')
            combined = DataFrame(self._internal.with_new_columns([self.
                rename(self_column_label), other.rename(other_column_label)]))
        else:
            if not self.index.equals(other.index):
                raise ValueError(
                    'Can only compare identically-labeled Series objects')
            combined = combine_frames(self.to_frame(), other.to_frame())
        this_column_label = 'self'
        that_column_label = 'other'
        if keep_equal and keep_shape:
            combined.columns = pd.Index([this_column_label, that_column_label])
            return combined
        this_data_scol = combined._internal.data_spark_columns[0]
        that_data_scol = combined._internal.data_spark_columns[1]
        index_scols = combined._internal.index_spark_columns
        sdf = combined._internal.spark_frame
        if keep_shape:
            this_scol = F.when(this_data_scol == that_data_scol, None
                ).otherwise(this_data_scol).alias(this_column_label)
            that_scol = F.when(this_data_scol == that_data_scol, None
                ).otherwise(that_data_scol).alias(that_column_label)
        else:
            sdf = sdf.filter(~this_data_scol.eqNullSafe(that_data_scol))
            this_scol = this_data_scol.alias(this_column_label)
            that_scol = that_data_scol.alias(that_column_label)
        sdf = sdf.select(index_scols + [this_scol, that_scol,
            NATURAL_ORDER_COLUMN_NAME])
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
            scol_for(sdf, col) for col in self._internal.
            index_spark_column_names], index_names=self._internal.
            index_names, index_dtypes=self._internal.index_dtypes,
            column_labels=[(this_column_label,), (that_column_label,)],
            data_spark_columns=[scol_for(sdf, this_column_label), scol_for(
            sdf, that_column_label)], column_label_names=[None])
        return DataFrame(internal)

    def align(self, other, join='outer', axis=None, copy=True):
        """
        Align two objects on their axes with the specified join method.

        Join method is specified for each axis Index.

        Parameters
        ----------
        other : DataFrame or Series
        join : {{'outer', 'inner', 'left', 'right'}}, default 'outer'
        axis : allowed axis of the other object, default None
            Align on index (0), columns (1), or both (None).
        copy : bool, default True
            Always returns new objects. If copy=False and no reindexing is
            required then original objects are returned.

        Returns
        -------
        (left, right) : (Series, type of other)
            Aligned objects.

        Examples
        --------
        >>> ks.set_option("compute.ops_on_diff_frames", True)
        >>> s1 = ks.Series([7, 8, 9], index=[10, 11, 12])
        >>> s2 = ks.Series(["g", "h", "i"], index=[10, 20, 30])

        >>> aligned_l, aligned_r = s1.align(s2)
        >>> aligned_l.sort_index()
        10    7.0
        11    8.0
        12    9.0
        20    NaN
        30    NaN
        dtype: float64
        >>> aligned_r.sort_index()
        10       g
        11    None
        12    None
        20       h
        30       i
        dtype: object

        Align with the join type "inner":

        >>> aligned_l, aligned_r = s1.align(s2, join="inner")
        >>> aligned_l.sort_index()
        10    7
        dtype: int64
        >>> aligned_r.sort_index()
        10    g
        dtype: object

        Align with a DataFrame:

        >>> df = ks.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}, index=[10, 20, 30])
        >>> aligned_l, aligned_r = s1.align(df)
        >>> aligned_l.sort_index()
        10    7.0
        11    8.0
        12    9.0
        20    NaN
        30    NaN
        dtype: float64
        >>> aligned_r.sort_index()
              a     b
        10  1.0     a
        11  NaN  None
        12  NaN  None
        20  2.0     b
        30  3.0     c

        >>> ks.reset_option("compute.ops_on_diff_frames")
        """
        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError('Series does not support columns axis.')
        self_df = self.to_frame()
        left, right = self_df.align(other, join=join, axis=axis, copy=False)
        if left is self_df:
            left_ser = self
        else:
            left_ser = first_series(left).rename(self.name)
        return (left_ser.copy(), right.copy()) if copy else (left_ser, right)

    def between_time(self, start_time, end_time, include_start=True,
        include_end=True, axis=0):
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
        include_end : bool, default True
            Whether the end time needs to be included in the result.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine range time on index or columns value.

        Returns
        -------
        Series
            Data from the original object filtered to the specified dates range.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        at_time : Select values at a particular time of the day.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_between_time : Get just the index locations for
            values between particular times of the day.

        Examples
        --------
        >>> idx = pd.date_range('2018-04-09', periods=4, freq='1D20min')
        >>> kser = ks.Series([1, 2, 3, 4], index=idx)
        >>> kser
        2018-04-09 00:00:00    1
        2018-04-10 00:20:00    2
        2018-04-11 00:40:00    3
        2018-04-12 01:00:00    4
        dtype: int64

        >>> kser.between_time('0:15', '0:45')
        2018-04-10 00:20:00    2
        2018-04-11 00:40:00    3
        dtype: int64
        """
        return first_series(self.to_frame().between_time(start_time,
            end_time, include_start, include_end, axis)).rename(self.name)

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
            scol = F.when(self.spark.column.isNull(), F.lit(None)).otherwise(
                func(self.spark.column).over(window))
        else:
            scol = F.when(F.max(self.spark.column.isNull()).over(window), F
                .lit(None)).otherwise(func(self.spark.column).over(window))
        return self._with_new_scol(scol)

    def _cumsum(self, skipna, part_cols=()):
        kser = self
        if isinstance(kser.spark.data_type, BooleanType):
            kser = kser.spark.transform(lambda scol: scol.cast(LongType()))
        elif not isinstance(kser.spark.data_type, NumericType):
            raise TypeError('Could not convert {} ({}) to numeric'.format(
                spark_type_to_pandas_dtype(kser.spark.data_type), kser.
                spark.data_type.simpleString()))
        return kser._cum(F.sum, skipna, part_cols)

    def _cumprod(self, skipna, part_cols=()):
        if isinstance(self.spark.data_type, BooleanType):
            scol = self._cum(lambda scol: F.min(F.coalesce(scol, F.lit(True
                ))), skipna, part_cols).spark.column.cast(LongType())
        elif isinstance(self.spark.data_type, NumericType):
            num_zeros = self._cum(lambda scol: F.sum(F.when(scol == 0, 1).
                otherwise(0)), skipna, part_cols).spark.column
            num_negatives = self._cum(lambda scol: F.sum(F.when(scol < 0, 1
                ).otherwise(0)), skipna, part_cols).spark.column
            sign = F.when(num_negatives % 2 == 0, 1).otherwise(-1)
            abs_prod = F.exp(self._cum(lambda scol: F.sum(F.log(F.abs(scol)
                )), skipna, part_cols).spark.column)
            scol = F.when(num_zeros > 0, 0).otherwise(sign * abs_prod)
            if isinstance(self.spark.data_type, IntegralType):
                scol = F.round(scol).cast(LongType())
        else:
            raise TypeError('Could not convert {} ({}) to numeric'.format(
                spark_type_to_pandas_dtype(self.spark.data_type), self.
                spark.data_type.simpleString()))
        return self._with_new_scol(scol)
    dt = CachedAccessor('dt', DatetimeMethods)
    str = CachedAccessor('str', StringMethods)
    cat = CachedAccessor('cat', CategoricalAccessor)
    plot = CachedAccessor('plot', KoalasPlotAccessor)

    def _apply_series_op(self, op, should_resolve=False):
        kser = op(self)
        if should_resolve:
            internal = kser._internal.resolved_copy
            return first_series(DataFrame(internal))
        else:
            return kser

    def _reduce_for_stat_function(self, sfun, name, axis=None, numeric_only
        =None, **kwargs):
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
        spark_column = self.spark.column
        spark_type = self.spark.data_type
        if num_args == 1:
            scol = sfun(spark_column)
        else:
            assert num_args == 2
            scol = sfun(spark_column, spark_type)
        min_count = kwargs.get('min_count', 0)
        if min_count > 0:
            scol = F.when(Frame._count_expr(spark_column, spark_type) >=
                min_count, scol)
        result = unpack_scalar(self._internal.spark_frame.select(scol))
        return result if result is not None else np.nan

    def __getitem__(self, key):
        try:
            if isinstance(key, slice) and any(type(n) == int for n in [key.
                start, key.stop]) or type(key) == int and not isinstance(self
                .index.spark.data_type, (IntegerType, LongType)):
                return self.iloc[key]
            return self.loc[key]
        except SparkPandasIndexingError:
            raise KeyError('Key length ({}) exceeds index depth ({})'.
                format(len(key), self._internal.index_level))

    def __getattr__(self, item):
        if item.startswith('__'):
            raise AttributeError(item)
        if hasattr(MissingPandasLikeSeries, item):
            property_or_func = getattr(MissingPandasLikeSeries, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError("'Series' object has no attribute '{}'".format
            (item))

    def _to_internal_pandas(self):
        """
        Return a pandas Series directly from _internal to avoid overhead of copy.

        This method is for internal use only.
        """
        return self._kdf._internal.to_pandas_frame[self.name]

    def __repr__(self):
        max_display_count = get_option('display.max_rows')
        if max_display_count is None:
            return self._to_internal_pandas().to_string(name=self.name,
                dtype=self.dtype)
        pser = self._kdf._get_or_create_repr_pandas_cache(max_display_count)[
            self.name]
        pser_length = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit('\n', 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group('length')
                dtype_name = str(self.dtype.name)
                if self.name is None:
                    footer = (
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
            fields = []
        else:
            fields = [f for f in self.spark.data_type.fieldNames() if ' '
                 not in f]
        return super().__dir__() + fields

    def __iter__(self):
        return MissingPandasLikeSeries.__iter__(self)
    if sys.version_info >= (3, 7):

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
    row = l.iloc[0]
    l2 = list(row)
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
