"""
A wrapper class for Spark Column to behave similar to pandas Series.
"""
import datetime
import re
import inspect
import sys
import warnings
from collections.abc import Mapping, Iterable
from distutils.version import LooseVersion
from functools import partial, wraps, reduce
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union, cast, Callable

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
from databricks import koalas as ks
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

REPR_PATTERN = re.compile("Length: (?P<length>[0-9]+)")
_flex_doc_SERIES = (
    "\nReturn {desc} of series and other, element-wise (binary operator `{op_name}`).\n\n"
    "Equivalent to ``{equiv}``\n\nParameters\n----------\nother : Series or scalar value\n\n"
    "Returns\n-------\nSeries\n    The result of the operation.\n\nSee Also\n--------\nSeries.{reverse}\n\n"
    "{series_examples}\n"
)
_add_example_SERIES = (
    "\nExamples\n--------\n>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],\n...                    "
    "'b': [2, np.nan, 2, np.nan]},\n...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])\n"
    ">>> df\n     a    b\na  2.0  2.0\nb  2.0  NaN\nc  4.0  2.0\nd  NaN  NaN\n\n>>> df.a.add(df.b)\na    4.0\nb    NaN\nc    6.0\nd    NaN\ndtype: float64\n\n"
    ">>> df.a.radd(df.b)\na    4.0\nb    NaN\nc    6.0\nd    NaN\ndtype: float64\n"
)
_sub_example_SERIES = (
    "\nExamples\n--------\n>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],\n...                    "
    "'b': [2, np.nan, 2, np.nan]},\n...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])\n"
    ">>> df\n     a    b\na  2.0  2.0\nb  2.0  NaN\nc  4.0  2.0\nd  NaN  NaN\n\n>>> df.a.subtract(df.b)\na    0.0\nb    NaN\nc    2.0\nd    NaN\ndtype: float64\n\n>>> df.a.rsub(df.b)\na    0.0\nb    NaN\nc   -2.0\nd    NaN\ndtype: float64\n"
)
_mul_example_SERIES = (
    "\nExamples\n--------\n>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],\n...                    "
    "'b': [2, np.nan, 2, np.nan]},\n...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])\n"
    ">>> df\n     a    b\na  2.0  2.0\nb  2.0  NaN\nc  4.0  2.0\nd  NaN  NaN\n\n>>> df.a.multiply(df.b)\na    4.0\nb    NaN\nc    8.0\nd    NaN\ndtype: float64\n\n>>> df.a.rmul(df.b)\na    4.0\nb    NaN\nc    8.0\nd    NaN\ndtype: float64\n"
)
_div_example_SERIES = (
    "\nExamples\n--------\n>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],\n...                    "
    "'b': [2, np.nan, 2, np.nan]},\n...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])\n"
    ">>> df\n     a    b\na  2.0  2.0\nb  2.0  NaN\nc  4.0  2.0\nd  NaN  NaN\n\n>>> df.a.divide(df.b)\na    1.0\nb    NaN\nc    2.0\nd    NaN\ndtype: float64\n\n>>> df.a.rdiv(df.b)\na    1.0\nb    NaN\nc    0.5\nd    NaN\ndtype: float64\n"
)
_pow_example_SERIES = (
    "\nExamples\n--------\n>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],\n...                    "
    "'b': [2, np.nan, 2, np.nan]},\n...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])\n"
    ">>> df\n     a    b\na  2.0  2.0\nb  2.0  NaN\nc  4.0  2.0\nd  NaN  NaN\n\n>>> df.a.pow(df.b)\na     4.0\nb     NaN\nc    16.0\nd     NaN\ndtype: float64\n\n>>> df.a.rpow(df.b)\na     4.0\nb     NaN\nc    16.0\nd     NaN\ndtype: float64\n"
)
_mod_example_SERIES = (
    "\nExamples\n--------\n>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],\n...                    "
    "'b': [2, np.nan, 2, np.nan]},\n...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])\n"
    ">>> df\n     a    b\na  2.0  2.0\nb  2.0  NaN\nc  4.0  2.0\nd  NaN  NaN\n\n>>> df.a.mod(df.b)\na    0.0\nb    NaN\nc    0.0\nd    NaN\ndtype: float64\n\n>>> df.a.rmod(df.b)\na    0.0\nb    NaN\nc    2.0\nd    NaN\ndtype: float64\n"
)
_floordiv_example_SERIES = (
    "\nExamples\n--------\n>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],\n...                    "
    "'b': [2, np.nan, 2, np.nan]},\n...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])\n"
    ">>> df\n     a    b\na  2.0  2.0\nb  2.0  NaN\nc  4.0  2.0\nd  NaN  NaN\n\n>>> df.a.floordiv(df.b)\na    1.0\nb    NaN\nc    2.0\nd    NaN\ndtype: float64\n\n>>> df.a.rfloordiv(df.b)\na    1.0\nb    NaN\nc    0.0\nd    NaN\ndtype: float64\n"
)

T = TypeVar("T")
str_type = str


def _create_type_for_series_type(param: Any) -> Any:
    from databricks.koalas.typedef import NameTypeHolder

    if isinstance(param, ExtensionDtype):
        new_class = type("NameType", (NameTypeHolder,), {})
        new_class.tpe = param
    else:
        new_class = param.type if isinstance(param, np.dtype) else param
    return SeriesType[new_class]


if (3, 5) <= sys.version_info < (3, 7):
    from typing import GenericMeta

    old_getitem = GenericMeta.__getitem__

    def new_getitem(self: GenericMeta, params: Any) -> Any:
        if hasattr(self, "is_series"):
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

    def __init__(
        self,
        data: Optional[Union[np.ndarray, pd.Series, dict, Any]] = None,
        index: Optional[Union[List[Any], pd.Index]] = None,
        dtype: Optional[np.dtype] = None,
        name: Optional[Any] = None,
        copy: bool = False,
        fastpath: bool = False,
    ) -> None:
        assert data is not None
        if isinstance(data, DataFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath
            self._anchor: DataFrame = data
            self._col_label: Tuple[Any, ...] = index
        else:
            if isinstance(data, pd.Series):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert not fastpath
                s: pd.Series = data
            else:
                s: pd.Series = pd.Series(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
                )
            internal: InternalFrame = InternalFrame.from_pandas(pd.DataFrame(s))
            if s.name is None:
                internal = internal.copy(column_labels=[None])
            anchor: DataFrame = DataFrame(internal)
            self._anchor = anchor
            self._col_label = anchor._internal.column_labels[0]
            object.__setattr__(anchor, "_kseries", {self._col_label: self})

    @property
    def _kdf(self) -> DataFrame:
        return self._anchor

    @property
    def _internal(self) -> InternalFrame:
        return self._kdf._internal.select_column(self._column_label)

    @property
    def _column_label(self) -> Tuple[Any, ...]:
        return self._col_label

    def _update_anchor(self, kdf: DataFrame) -> None:
        assert kdf._internal.column_labels == [self._column_label], (
            kdf._internal.column_labels,
            [self._column_label],
        )
        self._anchor = kdf
        object.__setattr__(kdf, "_kseries", {self._column_label: self})

    def _with_new_scol(self, scol: Column, *, dtype: Optional[np.dtype] = None) -> "Series":
        """
        Copy Koalas Series with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Series
        """
        internal = self._internal.copy(
            data_spark_columns=[scol.alias(name_like_string(self._column_label))],
            data_dtypes=[dtype],
        )
        return first_series(DataFrame(internal))

    spark: CachedAccessor = CachedAccessor("spark", SparkSeriesMethods)

    @property
    def dtypes(self) -> np.dtype:
        """Return the dtype object of the underlying data.

        >>> s = ks.Series(list('abc'))
        >>> s.dtype == s.dtypes
        True
        """
        return self.dtype

    @property
    def axes(self) -> List[pd.Index]:
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
    def spark_type(self) -> ExtensionDtype:
        warnings.warn(
            "Series.spark_type is deprecated as of Series.spark.data_type. Please use the API instead.",
            FutureWarning,
        )
        return self.spark.data_type

    spark_type.__doc__ = SparkSeriesMethods.data_type.__doc__

    def add(self, other: Union["Series", Any]) -> "Series":
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

    def div(self, other: Union["Series", Any]) -> "Series":
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

    def truediv(self, other: Union["Series", Any]) -> "Series":
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

    def mul(self, other: Union["Series", Any]) -> "Series":
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

    def sub(self, other: Union["Series", Any]) -> "Series":
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

    def mod(self, other: Union["Series", Any]) -> "Series":
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

    def pow(self, other: Union["Series", Any]) -> "Series":
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

    def floordiv(self, other: Union["Series", Any]) -> "Series":
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
    koalas: CachedAccessor = CachedAccessor("koalas", KoalasSeriesMethods)

    def eq(self, other: Any) -> "Series":
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

    def gt(self, other: Any) -> "Series":
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

    def ge(self, other: Any) -> "Series":
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

    def lt(self, other: Any) -> "Series":
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

    def le(self, other: Any) -> "Series":
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

    def ne(self, other: Any) -> "Series":
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

    def divmod(self, other: Union["Series", Any]) -> Tuple["Series", "Series"]:
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
        return (self.floordiv(other), self.mod(other))

    def rdivmod(self, other: Union["Series", Any]) -> Tuple["Series", "Series"]:
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
        return (self.rfloordiv(other), self.rmod(other))

    def between(
        self, left: Union[Any, Iterable[Any]], right: Union[Any, Iterable[Any]], inclusive: bool = True
    ) -> "Series":
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

    def map(self, arg: Union[Callable[[T], Any], Mapping[Any, Any]]) -> "Series":
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
            current = F.when(
                F.lit(False),
                F.lit(None).cast(self.spark.data_type),
            )
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
        """An alias for :meth:`Series.rename`."""
        warnings.warn(
            "Series.alias is deprecated as of Series.rename. Please use the API instead.",
            FutureWarning,
        )
        return self.rename(name)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return a tuple of the shape of the underlying data."""
        return (len(self),)

    @property
    def name(self) -> Optional[Any]:
        """Return name of the Series."""
        name = self._column_label
        if name is not None and len(name) == 1:
            return name[0]
        else:
            return name

    @name.setter
    def name(self, name: Any) -> None:
        self.rename(name, inplace=True)

    def rename(
        self,
        index: Optional[Any] = None,
        inplace: bool = False,
    ) -> Optional["Series"]:
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
            raise TypeError("Series.name must be a hashable type")
        elif not isinstance(index, tuple):
            index = (index,)
        scol = self.spark.column.alias(name_like_string(index))
        internal = self._internal.copy(
            column_labels=[index],
            data_spark_columns=[scol],
            column_label_names=None,
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
        mapper: Optional[Union[Callable[[Any], Any], Mapping[Any, Any]]] = None,
        index: Optional[
            Union[
                Callable[[Any], Any],
                Mapping[Any, Any],
                List[Any],
                Tuple[Any, ...],
            ]
        ] = None,
        inplace: bool = False,
    ) -> Optional["Series"]:
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
        kdf = self.to_frame().rename_axis(mapper=mapper, index=index, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    @property
    def index(self) -> Any:
        """The index (axis labels) Column of the Series.

        See Also
        --------
        Index
        """
        return self._kdf.index

    @property
    def is_unique(self) -> bool:
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
        return (
            self._internal.spark_frame.select(
                (F.count(scol) == F.countDistinct(scol))
                & (F.count(F.when(scol.isNull(), 1).otherwise(None)) <= 1)
            )
            .collect()[0][0]
        )

    def reset_index(
        self,
        level: Optional[Union[int, List[Union[int, str]], Tuple[Union[int, str], ...]]] = None,
        drop: bool = False,
        name: Optional[Any] = None,
        inplace: bool = False,
    ) -> Optional[Union["Series", DataFrame]]:
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace and (not drop):
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

    def to_frame(self, name: Optional[Any] = None) -> DataFrame:
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
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_string, pd.Series.to_string, args
        )

    def to_clipboard(
        self, excel: bool = True, sep: Optional[str] = None, **kwargs: Any
    ) -> None:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_clipboard, pd.Series.to_clipboard, args
        )

    to_clipboard.__doc__ = DataFrame.to_clipboard.__doc__

    def to_dict(self, into: type = dict) -> Mapping[Any, Any]:
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
        float_format: Optional[Callable[[float], str]] = None,
        sparsify: Optional[Any] = None,
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
    ) -> str:
        """
        Render a LaTeX representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            buffer to write to
        columns : 
            Not used
        col_space : 
            Not used
        header : boolean, default True
            Add the Series header (index name)
        index : bool, optional
            Add index (row) labels, default True
        na_rep : string, optional
            string representation of NAN to use, default 'NaN'
        formatters : 
            Not used
        float_format : one-parameter function, optional
            formatter function to apply to columns' elements if they are floats
            default None
        sparsify : 
            Not used
        index_names : bool, default True
            Whether to include the index names
        bold_rows : bool, default False
            Whether to bold the header row
        column_format : str, optional
            Column format string
        longtable : bool, optional
            Whether to use longtable environment
        escape : bool, optional
            Whether to escape LaTeX special characters
        encoding : str, optional
            Encoding to use for the output string
        decimal : str, default '.'
            Decimal separator for numbers
        multicolumn : bool, optional
            Whether to use multicolumn for column headers
        multicolumn_format : str, optional
            Format string for multicolumn
        multirow : bool, optional
            Whether to use multirow for row headers

        Returns
        -------
        formatted : string

        Examples
        --------

        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)], columns=['dogs', 'cats'])
        >>> print(df['dogs'].to_latex())
        0    0.2
        1    0.0
        2    0.6
        3    0.2
        dtype: float64
        """
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_latex, pd.Series.to_latex, args
        )

    to_latex.__doc__ = DataFrame.to_latex.__doc__

    def to_pandas(self) -> pd.Series:
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

    def toPandas(self) -> pd.Series:
        warnings.warn(
            "Series.toPandas is deprecated as of Series.to_pandas. Please use the API instead.",
            FutureWarning,
        )
        return self.to_pandas()

    toPandas.__doc__ = to_pandas.__doc__

    def to_list(self) -> List[Any]:
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

    def drop_duplicates(
        self, keep: Union[str, bool] = "first", inplace: bool = False
    ) -> Optional["Series"]:
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].drop_duplicates(keep=keep)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def reindex(
        self, index: Optional[List[Any]] = None, fill_value: Optional[Any] = None
    ) -> "Series":
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
        return first_series(self.to_frame().reindex(index=index, fill_value=fill_value)).rename(self.name)

    def reindex_like(self, other: Union["Series", DataFrame]) -> "Series":
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
        ...                index=pd.date_range(start='2014-02-12', end='2014-02-15', freq='D'),
        ...                name="temp_celsius")
        >>> s1
        2014-02-12    24.3
        2014-02-13    31.0
        2014-02-14    22.0
        2014-02-15    35.0
        Name: temp_celsius, dtype: float64

        >>> s2 = ks.Series(["low", "low", "medium"],
        ...                index=pd.DatetimeIndex(['2014-02-12', '2014-02-13', '2014-02-15']),
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
            raise TypeError("other must be a Koalas Series or DataFrame")

    def fillna(
        self,
        value: Optional[Union[Any, "Series"]] = None,
        method: Optional[str] = None,
        axis: Optional[int] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Optional["Series"]:
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

        >>> s = ks.Series([np.nan, 'a', 'b', 'c'], name='x')
        >>> s.fillna(method='ffill')
        0    None
        1       a
        2       b
        3       c
        Name: x, dtype: object
        """
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
        self,
        value: Optional[Any] = None,
        method: Optional[str] = None,
        axis: Optional[int] = None,
        limit: Optional[int] = None,
        part_cols: Tuple[Any, ...] = (),
    ) -> "DataFrame":
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError("fillna currently only works for axis=0 or axis='index'")
        if value is None and method is None:
            raise ValueError("Must specify a fillna 'value' or 'method' parameter.")
        if method is not None and method not in ["ffill", "pad", "backfill", "bfill"]:
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
                raise TypeError(f"Unsupported type {type(value).__name__}")
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
            window = Window.partitionBy(*part_cols).orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(begin, end)
            scol = F.when(cond, func(scol, True).over(window)).otherwise(scol)
        return DataFrame(
            self._kdf._internal.with_new_spark_column(
                self._column_label, scol.alias(name_like_string(self.name))
            )
        )

    def dropna(
        self,
        axis: int = 0,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["Series"]:
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].dropna(axis=axis, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def clip(self, lower: Optional[Any] = None, upper: Optional[Any] = None) -> "Series":
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
            raise ValueError("List-like value are not supported for 'lower' and 'upper' at the " + "moment")
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
        self,
        labels: Optional[Union[Any, List[Any], Tuple[Any, ...]]] = None,
        index: Optional[Union[Any, List[Any], Tuple[Any, ...]]] = None,
        level: Optional[Union[int, str]] = None,
    ) -> "Series":
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

        >>> midx = pd.MultiIndex.from_product([['lama', 'cow', 'falcon'],
        ...                                       ['speed', 'weight', 'length']],
        ...                                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
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
        return first_series(self._drop(labels=labels, index=index, level=level))

    def _drop(
        self,
        labels: Optional[Union[Any, List[Any], Tuple[Any, ...]]] = None,
        index: Optional[Union[Any, List[Any], Tuple[Any, ...]]] = None,
        level: Optional[Union[int, str]] = None,
    ) -> "Series":
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
            elif all((is_name_like_value(idxes, allow_tuple=False) for idxes in index)):
                index = [(idx,) for idx in index]
            elif not all((is_name_like_tuple(idxes) for idxes in index)):
                raise ValueError(
                    "If the given index is a list, it should only contains names as all tuples or all non tuples that contain index names"
                )
            drop_index_scols: List[Column] = []
            for idxes in index:
                try:
                    index_scols = [
                        internal.index_spark_columns[lvl] == idx
                        for lvl, idx in enumerate(idxes, level)
                    ]
                except IndexError:
                    raise KeyError(f"Key length ({len(idxes)}) exceeds index depth ({internal.index_level})")
                drop_index_scols.append(reduce(lambda x, y: x & y, index_scols))
            cond = ~reduce(lambda x, y: x | y, drop_index_scols)
            return DataFrame(internal.with_filter(cond))
        else:
            raise ValueError("Need to specify at least one of 'labels' or 'index'")

    def head(self, n: int = 5) -> "Series":
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
        0    alligator
        1          bee
        Name: animal, dtype: object
        """
        return first_series(self.to_frame().head(n)).rename(self.name)

    def last(self, offset: Union[str, DateOffset]) -> "Series":
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

    def first(self, offset: Union[str, DateOffset]) -> "Series":
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

    def unique(self) -> "Series":
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        .. note:: This method returns newly created Series whereas pandas returns
                  the unique values as a NumPy array.

        Returns
        -------
        Series
            Unique values as a Series.

        See Also
        --------
        Index.unique
        groupby.SeriesGroupBy.unique

        Examples
        --------
        >>> s = ks.Series([2, 1, 3, 3], name='A')
        >>> s.unique().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1
        ...  2
        ...  3
        Name: A, dtype: int64

        >>> ks.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()
        0   2016-01-01
        dtype: datetime64[ns]

        >>> s.name = ('x', 'a')
        >>> s.unique().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1
        ...  2
        ...  3
        Name: (x, a), dtype: int64
        """
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
        >>> s = ks.Series([0, 2, 4], name='A')
        >>> s
        0    0
        1    2
        2    4
        dtype: int64

        >>> s.sort_values(ascending=True)
        0    0
        1    2
        2    4
        dtype: int64

        >>> s.sort_values(ascending=False)
        2    4
        1    2
        0    0
        dtype: int64

        >>> s.sort_values(inplace=True)
        >>> s
        0    0
        1    2
        2    4
        dtype: int64

        >>> s = ks.Series([0, 2, 4, np.nan], name='A')
        >>> s.sort_values(na_position='first')
        3    NaN
        0    0.0
        1    2.0
        2    4.0
        dtype: float64

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
        level: Optional[Union[int, str, List[Union[int, str]], Tuple[Union[int, str], ...]]] = None,
        ascending: bool = True,
        inplace: bool = False,
        kind: Optional[str] = None,
        na_position: str = "last",
    ) -> Optional["Series"]:
        """
        Sort object by labels (along an axis)

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine range time on index or columns value.
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        kdf = self._kdf[[self.name]].sort_index(
            axis=axis,
            level=level,
            ascending=ascending,
            kind=kind,
            na_position=na_position,
        )
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def swaplevel(
        self, i: Union[int, str] = -2, j: Union[int, str] = -1, copy: bool = True
    ) -> "Series":
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
        >>> midx = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=['first', 'second'])
        >>> s = ks.Series([4, 200, 3, 1], index=midx)
        >>> s
        first  second
        a      c        4
               d      200
        b      e        3
               f        1
        dtype: int64

        >>> s.swaplevel()
        second  first
        c       a        4
        d       a      200
        e       b        3
        f       b        1
        dtype: int64

        >>> s.swaplevel(0, 1)
        second  first
        c       a        4
        d       a      200
        e       b        3
        f       b        1
        dtype: int64

        >>> s.swaplevel("first", "second")
        second  first
        c       a        4
        d       a      200
        e       b        3
        f       b        1
        dtype: int64
        """
        assert copy is True
        return first_series(self.to_frame().swaplevel(i, j, axis=0)).rename(self.name)

    def swapaxes(
        self, i: Union[int, str], j: Union[int, str], copy: bool = True
    ) -> "Series":
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
            raise ValueError("Axis must be 0 for Series")
        return self.copy()

    def add_prefix(self, prefix: str) -> "Series":
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
        sdf = internal.spark_frame.select(
            [
                F.concat(F.lit(prefix), index_spark_column).alias(index_spark_column_name)
                for index_spark_column, index_spark_column_name in zip(
                    internal.index_spark_columns, internal.index_spark_column_names
                )
            ]
            + internal.data_spark_columns
        )
        internal = internal.copy(
            spark_frame=sdf, index_dtypes=[None] * internal.index_level
        )
        return first_series(DataFrame(internal))

    def add_suffix(self, suffix: str) -> "Series":
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
        sdf = internal.spark_frame.select(
            [
                F.concat(index_spark_column, F.lit(suffix)).alias(index_spark_column_name)
                for index_spark_column, index_spark_column_name in zip(
                    internal.index_spark_columns, internal.index_spark_column_names
                )
            ]
            + internal.data_spark_columns
        )
        internal = internal.copy(
            spark_frame=sdf, index_dtypes=[None] * internal.index_level
        )
        return first_series(DataFrame(internal))

    def corr(
        self, other: "Series", method: str = "pearson"
    ) -> Optional[float]:
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
        columns = ["__corr_arg1__", "__corr_arg2__"]
        kdf = self._kdf.assign(__corr_arg1__=self, __corr_arg2__=other)[columns]
        c = corr(kdf, method=method)
        return c.loc[tuple(columns)]

    def nsmallest(self, n: int = 5) -> "Series":
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

    def nlargest(self, n: int = 5) -> "Series":
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

    def append(
        self,
        to_append: Union["Series", List["Series"], Tuple["Series", ...]],
        ignore_index: bool = False,
        verify_integrity: bool = False,
    ) -> "Series":
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
        return first_series(self.to_frame().append(to_append.to_frame(), ignore_index, verify_integrity)).rename(self.name)

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[Any] = None,
    ) -> "Series":
        return first_series(self.to_frame().sample(n=n, frac=frac, replace=replace, random_state=random_state)).rename(self.name)

    sample.__doc__ = DataFrame.sample.__doc__

    def hist(
        self,
        bins: int = 10,
        **kwds: Any,
    ) -> None:
        """
        Plot a histogram of the Series.

        Parameters
        ----------
        bins : int, default 10
            Number of histogram bins to use.

        **kwds
            Additional keyword arguments to pass to matplotlib's hist function.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s.hist()

        """
        return self.plot.hist(bins, **kwds)

    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def apply(
        self,
        func: Callable[[T], Any],
        args: Tuple[Any, ...] = (),
        **kwds: Any,
    ) -> "Series":
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

        >>> s = ks.Series([1, 2, 3], name='A')
        >>> s
        0    1
        1    2
        2    3
        dtype: int64


        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x) -> np.int64:
        ...     return x ** 2
        >>> s.apply(square)
        0    1
        1    4
        2    9
        dtype: int64


        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword

        >>> def subtract_custom_value(x, custom_value) -> np.int64:
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        0   -4
        1   -3
        2   -2
        dtype: int64


        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``

        >>> def add_custom_values(x, **kwargs) -> np.int64:
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        0    76
        1    77
        2    78
        dtype: int64


        Use a function from the Numpy library

        >>> def numpy_log(col) -> float:
        ...     return np.log(col)
        >>> s.apply(numpy_log)
        0    0.0
        1    0.693147
        2    1.098612
        dtype: float64


        You can omit the type hint and let Koalas infer its type.

        >>> s.apply(np.log)
        0    0.0
        1    0.693147
        2    1.098612
        dtype: float64

        """
        assert callable(func), "the first argument should be a callable function."
        try:
            spec = inspect.getfullargspec(func)
            return_sig = spec.annotations.get("return", None)
            should_infer_schema = return_sig is None
        except TypeError:
            should_infer_schema = True
        apply_each: Callable[[Any], Any] = wraps(func)(lambda s: s.apply(func, args=args, **kwds))
        if should_infer_schema:
            return self.koalas._transform_batch(apply_each, None)
        else:
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, ScalarType):
                raise ValueError(
                    f"Expected the return type of this function to be of scalar type, but found type {sig_return}"
                )
            return_type = cast(ScalarType, sig_return)
            return self.koalas._transform_batch(apply_each, return_type)

    def aggregate(self, func: Union[str, List[str]]) -> Union[Any, "Series"]:
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

        >>> s = ks.Series([1, 2, 3])
        >>> s.agg('min')
        1

        >>> s.agg(['min', 'max']).sort_index()
        max    3
        min    1
        dtype: int64
        """
        if isinstance(func, list):
            return first_series(self.to_frame().aggregate(func)).rename(self.name)
        elif isinstance(func, str):
            return getattr(self, func)()
        else:
            raise ValueError("func must be a string or list of strings")

    agg = aggregate

    def transpose(self) -> "Series":
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

    def transform(
        self,
        func: Union[Callable[[T], Any], List[Callable[[T], Any]]],
        axis: int = 0,
        *args: Any,
        **kwds: Any,
    ) -> Union["Series", DataFrame]:
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
        **kwds
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
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if isinstance(func, list):
            applied: List["Series"]
            applied = []
            for f in func:
                applied.append(self.apply(f, args=args, **kwds).rename(f.__name__))
            internal = self._internal.with_new_columns(applied)
            return DataFrame(internal)
        else:
            return self.apply(func, args=args, **kwds)

    def transform_batch(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> "Series":
        warnings.warn(
            "Series.transform_batch is deprecated as of Series.koalas.transform_batch. Please use the API instead.",
            FutureWarning,
        )
        return self.koalas.transform_batch(func, *args, **kwargs)

    transform_batch.__doc__ = KoalasSeriesMethods.transform_batch.__doc__

    def round(self, decimals: int = 0) -> "Series":
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
            raise ValueError("decimals must be an integer")
        scol = F.round(self.spark.column, decimals)
        return self._with_new_scol(scol)

    def quantile(
        self, q: Union[float, Iterable[float]] = 0.5, accuracy: int = 10000
    ) -> Union[float, "Series"]:
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
            return first_series(
                self.to_frame().quantile(q=q, axis=0, numeric_only=False, accuracy=accuracy)
            ).rename(self.name)
        else:
            if not isinstance(accuracy, int):
                raise ValueError(
                    "accuracy must be an integer; however, got [%s]" % type(accuracy).__name__
                )
            if not isinstance(q, float):
                raise ValueError("q must be a float or an array of floats; however, [%s] found." % type(q))
            if q < 0.0 or q > 1.0:
                raise ValueError("percentiles should all be in the interval [0, 1].")

            def quantile(
                spark_column: Column, spark_type: ExtensionDtype
            ) -> Union[Column, None]:
                if isinstance(spark_type, (BooleanType, NumericType)):
                    return SF.percentile_approx(spark_column.cast(DoubleType()), q, accuracy)
                else:
                    raise TypeError(
                        "Could not convert {} ({}) to numeric".format(
                            spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                        )
                    )

            return self._reduce_for_stat_function(quantile, name="quantile")

    def rank(self, method: str = "average", ascending: bool = True) -> "Column":
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
        ascending : bool, default True
            False for ranks by high (1) to low (N)

        Returns
        -------
        Column
            Sorted column.

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

    def _rank(
        self,
        method: str,
        ascending: bool,
        *,
        part_cols: Tuple[Any, ...] = (),
    ) -> "Series":
        if method not in ["average", "min", "max", "first", "dense"]:
            raise ValueError("method must be one of 'average', 'min', 'max', 'first', 'dense'")
        if self._internal.index_level > 1:
            raise ValueError("rank do not support index now")
        if ascending:
            asc_func = lambda scol: scol.asc()
        else:
            asc_func = lambda scol: scol.desc()
        if method == "first":
            window = Window.orderBy(
                asc_func(self.spark.column), asc_func(F.col(NATURAL_ORDER_COLUMN_NAME))
            ).partitionBy(*part_cols).rowsBetween(Window.unboundedPreceding, Window.currentRow)
            scol = F.row_number().over(window)
        elif method == "dense":
            window = Window.orderBy(asc_func(self.spark.column)).partitionBy(*part_cols).rowsBetween(
                Window.unboundedPreceding, Window.currentRow
            )
            scol = F.dense_rank().over(window)
        else:
            if method == "average":
                stat_func: Callable[[Column], Column] = F.mean
            elif method == "min":
                stat_func = F.min
            elif method == "max":
                stat_func = F.max
            window1 = Window.orderBy(asc_func(self.spark.column)).partitionBy(*part_cols).rowsBetween(
                Window.unboundedPreceding, Window.currentRow
            )
            window2 = Window.partitionBy([self.spark.column] + list(part_cols)).rowsBetween(
                Window.unboundedPreceding, Window.unboundedFollowing
            )
            scol = stat_func(F.row_number().over(window1)).over(window2)
        kser = self._with_new_scol(scol)
        return cast(Series, kser.astype(np.float64))

    def filter(
        self,
        items: Optional[Union[Any, List[Any], Tuple[Any, ...]]] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
        axis: Optional[int] = None,
    ) -> "Series":
        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError("Series does not support columns axis.")
        return first_series(self.to_frame().filter(items=items, like=like, regex=regex, axis=axis)).rename(self.name)

    filter.__doc__ = DataFrame.filter.__doc__

    def describe(
        self, percentiles: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> "Series":
        return first_series(self.to_frame().describe(percentiles)).rename(self.name)

    describe.__doc__ = DataFrame.describe.__doc__

    def diff(self, periods: int = 1) -> "Column":
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
        Column
            Diffed column.

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                    'b': [1, 1, 2, 3, 5, 8]},
        ...                   index=['a', 'b', 'c', 'd', 'e', 'f'])
        >>> df
           a  b
        a  1  1
        b  2  1
        c  3  2
        d  4  3
        e  5  5
        f  6  8

        >>> df.b.diff()
        a    NaN
        b    0.0
        c    1.0
        d    1.0
        e    2.0
        f    3.0
        Name: b, dtype: float64

        Difference with previous value

        >>> df.c.diff(periods=3)
        a     NaN
        b     NaN
        c     NaN
        d    15.0
        e    21.0
        f    27.0
        Name: c, dtype: float64

        Difference with following value

        >>> df.c.diff(periods=-1)
        a    -3.0
        b    -5.0
        c    -7.0
        d    -9.0
        e   -11.0
        f     NaN
        Name: c, dtype: float64
        """
        return self._diff(periods).spark.analyzed

    def _diff(
        self, periods: int, *, part_cols: Tuple[Any, ...] = ()
    ) -> "Series":
        if not isinstance(periods, int):
            raise ValueError("periods should be an int; however, got [%s]" % type(periods).__name__)
        window = (
            Window.orderBy(asc(NATURAL_ORDER_COLUMN_NAME))
            if periods > 0
            else Window.orderBy(desc(NATURAL_ORDER_COLUMN_NAME))
        ).partitionBy(*part_cols).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        return self._with_new_scol(F.lag(self.spark.column, periods).over(window))

    def idxmax(self, skipna: bool = True) -> Union[Any, Tuple[Any, ...], float]:
        """
        Return the row label of the maximum value.

        If the maximum is achieved in multiple locations,
        the first row label is returned.

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
        Consider dataset containing cereal calories

        >>> s = ks.Series(data=[1, None, 4, 3], index=['A', 'B', 'C', 'D'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        dtype: float64

        >>> s.idxmax()
        'C'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmax(skipna=False)
        nan

        In case of multi-index, you get a tuple:

        >>> index = pd.MultiIndex.from_arrays([
        ...     ['a', 'a', 'b', 'b'], ['lama', 'cow', 'falcon', 'speed']],
        ...     names=('first', 'second')
        ... )
        >>> s = ks.Series(data=[1, None, 4, 3], index=index)
        >>> s
        first  second
        a      lama      1.0
               cow      NaN
        b      falcon    4.0
               speed     3.0
        dtype: float64

        >>> s.idxmax()
        ('b', 'falcon')

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

    def idxmin(self, skipna: bool = True) -> Union[Any, Tuple[Any, ...], float]:
        """
        Return the row label of the minimum value.

        If the minimum is achieved in multiple locations,
        the first row label is returned.

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
        Consider dataset containing cereal calories

        >>> s = ks.Series(data=[1, None, 4, 0], index=['A', 'B', 'C', 'D'])
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
        ...     ['a', 'a', 'b', 'b'], ['lama', 'cow', 'falcon', 'speed']],
        ...     names=('first', 'second')
        ... )
        >>> s = ks.Series(data=[1, None, 4, 0], index=index)
        >>> s
        first  second
        a      lama      1.0
               cow      NaN
        b      falcon    4.0
               speed     0.0
        dtype: float64

        >>> s.idxmin()
        ('b', 'speed')

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

    def pop(self, item: Any) -> Union[Any, "Series"]:
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

        >>> s = ks.Series([1, 100, 1, 100, 1, 100], index=[10, 3, 5, 2, 1, 8])
        >>> s
        10      1
        3     100
        5       1
        2     100
        1       1
        8     100
        dtype: int64

        >>> s.pop(('b', 'c'))
        ValueError: Key length (2) exceeds index depth (1)

        """
        if not is_name_like_value(item):
            raise ValueError("'key' should be string or tuple that contains strings")
        if not is_name_like_tuple(item):
            item = (item,)
        if self._internal.index_level < len(item):
            raise KeyError(f"Key length ({len(item)}) exceeds index depth ({self._internal.index_level})")
        internal = self._internal
        scols: List[Column] = internal.index_spark_columns[len(item) :] + [self.spark.column]
        rows = [internal.spark_columns[lvl] == idx for lvl, idx in enumerate(item, len(item))]
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
                column_labels=[None],
                data_spark_columns=[scol_for(sdf, name_like_string(self.name))],
            )
            return first_series(DataFrame(internal))
        else:
            internal = internal.copy(
                spark_frame=sdf,
                index_spark_columns=[scol_for(sdf, col) for col in internal.index_spark_column_names[len(item) :]],
                index_dtypes=internal.index_dtypes[len(item) :],
                index_names=self._internal.index_names[len(item) :],
                data_spark_columns=[scol_for(sdf, internal.data_spark_column_names[0])],
            )
            return first_series(DataFrame(internal))

    def copy(self, deep: Optional[bool] = None) -> "Series":
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

    def mad(self) -> float:
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

    def unstack(self, level: int = -1) -> DataFrame:
        """
        Unstack, a.k.a. pivot, Series with MultiIndex to produce DataFrame.
        The level involved will automatically get sorted.

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
        ...               index=pd.MultiIndex.from_product([['one', 'two'], ['a', 'b']]))
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
        a    1     3
        b    2     4

        >>> s.unstack(level=[0, 1])
        """
        if not isinstance(self.index, ks.MultiIndex):
            raise ValueError("Series.unstack only support for a MultiIndex")
        index_nlevels = self.index.nlevels
        if level > 0 and level > index_nlevels - 1:
            raise IndexError(
                f"Too many levels: Index has only {index_nlevels} levels, not {level + 1}"
            )
        elif level < 0 and level < -index_nlevels:
            raise IndexError(
                f"Too many levels: Index has only {index_nlevels} levels, {level} is not a valid level number"
            )
        internal = self._internal.resolved_copy
        index_map: List[Tuple[str, Optional[str]]] = list(
            zip(internal.index_spark_column_names, internal.index_names)
        )
        pivot_col, column_label_names = index_map.pop(level)
        index_scol_names, index_names = zip(*index_map)
        col = internal.data_spark_column_names[0]
        sdf = internal.spark_frame
        sdf = sdf.groupby(list(index_scol_names)).pivot(pivot_col).agg(F.first(scol_for(sdf, col)))
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in index_scol_names],
            index_names=list(index_names),
            column_label_names=[None],
        )
        return DataFrame(internal)

    def corr(
        self, other: "Series", method: str = "pearson"
    ) -> Optional[float]:
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
        """
        return self.corr(other, method=method)

    def assign(self, **kwargs: Any) -> "DataFrame":
        """
        Assign new columns to a DataFrame, returning a new object (a copy)
        with the new columns added.

        The operation does not change any of the original DataFrame's data.

        Parameters
        ----------
        **kwargs : keyword arguments
            New columns to add to the object.

        Returns
        -------
        DataFrame
            Series rerouted to DataFrame.

        Examples
        --------

        >>> ser = ks.Series([1, 2, 3], index=['a', 'b', 'c'])

        >>> ser.assign(x=ser, y=ser+1)  # doctest: +NORMALIZE_WHITESPACE
           x  y
        a  1  2
        b  2  3
        c  3  4

        >>> ser.assign(**{"x": ser, "y": ser+1})
           x  y
        a  1  2
        b  2  3
        c  3  4
        """
        return self.to_frame().assign(**kwargs)

    def isin(self, values: Union[Any, Iterable[Any]]) -> "Series":
        """
        Return boolean Series showing whether each element is exactly contained in values.

        Returns
        -------
        Series
            Series of booleans.

        See Also
        --------
        Series.equals : Test if Series elements are equal to another Series.
        Series.filter : Subset a Series based on some criteria.
        Series.where : Replace values where condition is False.

        Examples
        --------
        >>> s = ks.Series(['peanut', 'butter', 'and', 'jelly', None])
        >>> s
        0    peanut
        1    butter
        2       and
        3     jelly
        4      None
        dtype: object

        >>> s.isin(['peanut', 'jelly'])
        0     True
        1    False
        2    False
        3     True
        4    False
        dtype: bool

        >>> s.isin(['peanut', 'oreo'])
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool
        """
        if isinstance(values, tuple):
            values = list(values)
        elif not isinstance(values, Iterable):
            values = [values]
        return self.spark.column.isin(values).cast(BooleanType()).alias(name_like_string(self.name))

    def clip_lower(self, threshold: Any) -> "Series":
        """
        Trim values below threshold.

        Assign all values below the threshold to the threshold value.

        Parameters
        ----------
        threshold : float or int
            The minimum threshold value. All values below this threshold will be set to it.

        Returns
        -------
        Series
            Series with the values below the threshold replaced

        Examples
        --------
        >>> ks.Series([0, 2, 4]).clip_lower(1)
        0    1
        1    2
        2    4
        dtype: int64
        """
        return self.clip(lower=threshold)

    def clip_upper(self, threshold: Any) -> "Series":
        """
        Trim values above threshold.

        Assign all values above the threshold to the threshold value.

        Parameters
        ----------
        threshold : float or int
            The maximum threshold value. All values above this threshold will be set to it.

        Returns
        -------
        Series
            Series with the values above the threshold replaced

        Examples
        --------
        >>> ks.Series([0, 2, 4]).clip_upper(3)
        0    0
        1    2
        2    3
        dtype: int64
        """
        return self.clip(upper=threshold)

    def cumsum(self, skipna: bool = True, **kwargs: Any) -> "Series":
        """
        Return cumulative sum over requested axis.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values.

        Returns
        -------
        Series of cumulative sums

        Examples
        --------
        >>> s = ks.Series([1, 0,  2, 3, np.nan, 4])
        >>> s
        0    1.0
        1    0.0
        2    2.0
        3    3.0
        4    NaN
        5    4.0
        dtype: float64

        >>> s.cumsum()
        0     1.0
        1     1.0
        2     3.0
        3     6.0
        4     6.0
        5    10.0
        dtype: float64
        """
        return self._cumsum(skipna).spark.analyzed

    def cumsum(
        self,
        skipna: bool = True,
        *,
        part_cols: Tuple[Any, ...] = (),
    ) -> "Series":
        return self._cumsum(skipna, part_cols=part_cols).spark.analyzed

    def cumsum(
        self,
        skipna: bool = True,
        *,
        part_cols: Tuple[Any, ...] = (),
    ) -> "Series":
        kser = self
        if isinstance(kser.spark.data_type, BooleanType):
            kser = kser.spark.transform(lambda scol: scol.cast(LongType()))
        elif not isinstance(kser.spark.data_type, NumericType):
            raise TypeError(
                f"Could not convert {spark_type_to_pandas_dtype(self.spark.data_type)} ({self.spark.data_type.simpleString()}) to numeric"
            )
        return kser._cum(F.sum, skipna, part_cols)

    def cumprod(self, skipna: bool = True, *, part_cols: Tuple[Any, ...] = ()) -> "Series":
        """
        Return cumulative product over requested axis.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values.

        Returns
        -------
        Series of cumulative product

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.cumprod()
        0     1
        1     2
        2     6
        3    24
        dtype: int64
        """
        if isinstance(self.spark.data_type, BooleanType):
            scol = self._cum(lambda scol: F.min(F.coalesce(scol, F.lit(True))), skipna=True, part_cols=part_cols).spark.column.cast(LongType())
        elif isinstance(self.spark.data_type, NumericType):
            num_zeros = self._cum(lambda scol: F.sum(F.when(scol == 0, 1).otherwise(0)), skipna=True, part_cols=part_cols).spark.column
            num_negatives = self._cum(lambda scol: F.sum(F.when(scol < 0, 1).otherwise(0)), skipna=True, part_cols=part_cols).spark.column
            sign = F.when(num_negatives % 2 == 0, 1).otherwise(-1)
            abs_prod = F.exp(self._cum(lambda scol: F.sum(F.log(F.abs(scol))), skipna=True, part_cols=part_cols).spark.column)
            scol = F.when(num_zeros > 0, 0).otherwise(sign * abs_prod)
            if isinstance(self.spark.data_type, IntegralType):
                scol = F.round(scol).cast(LongType())
        else:
            raise TypeError(
                f"Could not convert {spark_type_to_pandas_dtype(self.spark.data_type)} ({self.spark.data_type.simpleString()}) to numeric"
            )
        return self._with_new_scol(scol)

    dt: DatetimeMethods = CachedAccessor("dt", DatetimeMethods)
    str: StringMethods = CachedAccessor("str", StringMethods)
    cat: CategoricalAccessor = CachedAccessor("cat", CategoricalAccessor)
    plot: KoalasPlotAccessor = CachedAccessor("plot", KoalasPlotAccessor)

    def _apply_series_op(
        self, op: Callable[["Series"], "Series"], should_resolve: bool = False
    ) -> "Series":
        kser = op(self)
        if should_resolve:
            internal = kser._internal.resolved_copy
            return first_series(DataFrame(internal))
        else:
            return kser

    def _reduce_for_stat_function(
        self,
        sfun: Callable[[Column, ExtensionDtype], Union[Column, None]],
        name: str,
        axis: Optional[int] = None,
        numeric_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Any, float]:
        """
        Applies sfun to the column and returns a scalar

        Parameters
        ----------
        sfun : the stats function to be used for aggregation
        name : original pandas API name.
        axis : used only for sanity check because series only support index axis.
        numeric_only : not used by this implementation, but passed down by stats functions

        Returns
        -------
        float
            Scalar result of aggregation
        """
        from inspect import signature

        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError("Series does not support columns axis.")

        num_args = len(signature(sfun).parameters)
        spark_column = self.spark.column
        spark_type = self.spark.data_type
        if num_args == 1:
            scol = sfun(spark_column, spark_type)
        else:
            assert num_args == 2
            scol = sfun(spark_column, spark_type)
        min_count = kwargs.get("min_count", 0)
        if min_count > 0:
            scol = F.when(Frame._count_expr(spark_column, spark_type) >= min_count, scol)
        result = unpack_scalar(self._internal.spark_frame.select(scol))
        return result if result is not None else np.nan

    def __getitem__(self, key: Union[Any, slice]) -> Union[Any, "Series"]:
        try:
            if isinstance(key, slice) and any((isinstance(n, int) for n in [key.start, key.stop])) or (
                isinstance(key, int) and not isinstance(self.index.spark.data_type, (IntegerType, LongType))
            ):
                return self.iloc[key]
            return self.loc[key]
        except SparkPandasIndexingError:
            raise KeyError(
                f"Key length ({len(key)}) exceeds index depth ({self._internal.index_level})"
            )

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        if hasattr(MissingPandasLikeSeries, item):
            property_or_func: Any = getattr(MissingPandasLikeSeries, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError(f"'Series' object has no attribute '{item}'")

    def _to_internal_pandas(self) -> pd.Series:
        """
        Return a pandas Series directly from _internal to avoid overhead of copy.

        This method is for internal use only.
        """
        return self._kdf._internal.to_pandas_frame[self.name]

    def __repr__(self) -> str:
        max_display_count: Optional[int] = get_option("display.max_rows")
        if max_display_count is None:
            return self._to_internal_pandas().to_string(name=self.name, dtype=self.dtype)
        pser: pd.Series = self._kdf._get_or_create_repr_pandas_cache(max_display_count)[self.name]
        pser_length: int = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit("\n", 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group("length")
                dtype_name = str(self.dtype.name)
                if self.name is None:
                    footer = f"\ndtype: {pprint_thing(dtype_name)}\nShowing only the first {length}"
                else:
                    footer = f"\nName: {self.name}, dtype: {pprint_thing(dtype_name)}\nShowing only the first {length}"
                return rest + footer
        return pser.to_string(name=self.name, dtype=self.dtype)

    def __dir__(self) -> List[str]:
        if not isinstance(self.spark.data_type, StructType):
            fields: List[str] = []
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

    def unpack_scalar(sdf: pyspark.sql.DataFrame) -> Any:
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

    def first_series(df: Union[DataFrame, pd.DataFrame]) -> "Series":
        """
        Takes a DataFrame and returns the first column of the DataFrame as a Series
        """
        assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
        if isinstance(df, DataFrame):
            return df._kser_for(df._internal.column_labels[0])
        else:
            return df[df.columns[0]]
