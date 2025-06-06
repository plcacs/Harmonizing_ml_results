"""
A wrapper class for Spark DataFrame to behave similar to pandas DataFrame.
"""
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Mapping
from distutils.version import LooseVersion
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
from databricks.koalas.ml import corr
from databricks.koalas.typedef import as_spark_type, infer_return_type, spark_type_to_pandas_dtype, DataFrameType, SeriesType, Scalar, ScalarType
from databricks.koalas.plot import KoalasPlotAccessor
if TYPE_CHECKING:
    from databricks.koalas.indexes import Index
    from databricks.koalas.series import Series
REPR_PATTERN = re.compile(
    '\\n\\n\\[(?P<rows>[0-9]+) rows x (?P<columns>[0-9]+) columns\\]$')
REPR_HTML_PATTERN = re.compile(
    '\\n\\<p\\>(?P<rows>[0-9]+) rows × (?P<columns>[0-9]+) columns\\<\\/p\\>\\n\\<\\/div\\>$'
    )
_flex_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``. With reverse version, `{reverse}`.

Among flexible wrappers (`add`, `sub`, `mul`, `div`) to
arithmetic operators: `+`, `-`, `*`, `/`, `//`.

Parameters
----------
other : scalar
    Any single data

Returns
-------
DataFrame
    Result of the arithmetic operation.

Examples
--------
>>> df = ks.DataFrame({{'angles': [0, 3, 4],
...                    'degrees': [360, 180, 360]}},
...                   index=['circle', 'triangle', 'rectangle'],
...                   columns=['angles', 'degrees'])
>>> df
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Add a scalar with operator version which return the same
results. Also reverse version.

>>> df + 1
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

>>> df.add(1)
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

>>> df.add(df)
           angles  degrees
circle          0      720
triangle        6      360
rectangle       8      720

>>> df + df + df
           angles  degrees
circle          0     1080
triangle        9      540
rectangle      12     1080

>>> df.radd(1)
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

Divide and true divide by constant with reverse version.

>>> df / 10
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.div(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rdiv(10)
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

>>> df.truediv(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rtruediv(10)
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

Subtract by constant with reverse version.

>>> df - 1
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

>>> df.sub(1)
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

>>> df.rsub(1)
           angles  degrees
circle          1     -359
triangle       -2     -179
rectangle      -3     -359

Multiply by constant with reverse version.

>>> df * 1
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

>>> df.mul(1)
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

>>> df.rmul(1)
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Floor Divide by constant with reverse version.

>>> df // 10
           angles  degrees
circle        0.0     36.0
triangle      0.0     18.0
rectangle     0.0     36.0

>>> df.floordiv(10)
           angles  degrees
circle        0.0     36.0
triangle      0.0     18.0
rectangle     0.0     36.0

>>> df.rfloordiv(10)  # doctest: +SKIP
           angles  degrees
circle        inf      0.0
triangle      3.0      0.0
rectangle     2.0      0.0

Mod by constant with reverse version.

>>> df % 2
           angles  degrees
circle          0        0
triangle        1        0
rectangle       0        0

>>> df.mod(2)
           angles  degrees
circle          0        0
triangle        1        0
rectangle       0        0

>>> df.rmod(2)
           angles  degrees
circle        NaN        2
triangle      2.0        2
rectangle     2.0        2

Power by constant with reverse version.

>>> df ** 2
           angles   degrees
circle        0.0  129600.0
triangle      9.0   32400.0
rectangle    16.0  129600.0

>>> df.pow(2)
           angles   degrees
circle        0.0  129600.0
triangle      9.0   32400.0
rectangle    16.0  129600.0

>>> df.rpow(2)
           angles        degrees
circle        1.0  2.348543e+108
triangle      8.0   1.532496e+54
rectangle    16.0  2.348543e+108
"""
T = TypeVar('T')


def _create_tuple_for_frame_type(params):
    """
    This is a workaround to support variadic generic in DataFrame.

    See https://github.com/python/typing/issues/193
    we always wraps the given type hints by a tuple to mimic the variadic generic.
    """
    from databricks.koalas.typedef import NameTypeHolder
    if isinstance(params, zip):
        params = [slice(name, tpe) for name, tpe in params]
    if isinstance(params, slice):
        params = params,
    if hasattr(params, '__len__') and isinstance(params, Iterable) and all(
        isinstance(param, slice) for param in params):
        for param in params:
            if isinstance(param.start, str) and param.step is not None:
                raise TypeError(
                    "Type hints should be specified as DataFrame['name': type]; however, got %s"
                     % param)
        name_classes = []
        for param in params:
            new_class = type('NameType', (NameTypeHolder,), {})
            new_class.name = param.start
            new_class.tpe = param.stop.type if isinstance(param.stop, np.dtype
                ) else param.stop
            name_classes.append(new_class)
        return Tuple[tuple(name_classes)]
    if not isinstance(params, Iterable):
        params = [params]
    new_params = []
    for param in params:
        if isinstance(param, ExtensionDtype):
            new_class = type('NameType', (NameTypeHolder,), {})
            new_class.tpe = param
            new_params.append(new_class)
        else:
            new_params.append(param.type if isinstance(param, np.dtype) else
                param)
    return Tuple[tuple(new_params)]


if (3, 5) <= sys.version_info < (3, 7):
    from typing import GenericMeta
    old_getitem = GenericMeta.__getitem__

    def new_getitem(self, params):
        if hasattr(self, 'is_dataframe'):
            return old_getitem(self, _create_tuple_for_frame_type(params))
        else:
            return old_getitem(self, params)
    GenericMeta.__getitem__ = new_getitem


class DataFrame(Frame, Generic[T]):
    """
    Koalas DataFrame that corresponds to pandas DataFrame logically. This holds Spark DataFrame
    internally.

    :ivar _internal: an internal immutable Frame to manage metadata.
    :type _internal: InternalFrame

    Parameters
    ----------
    data : numpy ndarray (structured or homogeneous), dict, pandas DataFrame, Spark DataFrame         or Koalas Series
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

    def __init__(self, data=None, index=None, columns=None, dtype=None,
        copy=False):
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
            internal = InternalFrame(spark_frame=data, index_spark_columns=None
                )
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
                pdf = pd.DataFrame(data=data, index=index, columns=columns,
                    dtype=dtype, copy=copy)
            internal = InternalFrame.from_pandas(pdf)
        object.__setattr__(self, '_internal_frame', internal)

    @property
    def _ksers(self):
        """ Return a dict of column label -> Series which anchors `self`. """
        from databricks.koalas.series import Series
        if not hasattr(self, '_kseries'):
            object.__setattr__(self, '_kseries', {label: Series(data=self,
                index=label) for label in self._internal.column_labels})
        else:
            kseries = self._kseries
            assert len(self._internal.column_labels) == len(kseries), (len(
                self._internal.column_labels), len(kseries))
            if any(self is not kser._kdf for kser in kseries.values()):
                self._kseries = {label: (kseries[label] if self is kseries[
                    label]._kdf else Series(data=self, index=label)) for
                    label in self._internal.column_labels}
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
            for old_label, new_label in zip_longest(self._internal.
                column_labels, internal.column_labels):
                if old_label is not None:
                    kser = self._ksers[old_label]
                    renamed = old_label != new_label
                    not_same_anchor = requires_same_anchor and not same_anchor(
                        internal, kser)
                    if renamed or not_same_anchor:
                        kdf = DataFrame(self._internal.select_column(old_label)
                            )
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
        Return an int representing the number of array dimensions.

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

    def _reduce_for_stat_function(self, sfun, name, axis=None, numeric_only
        =True, **kwargs):
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
            exprs = [F.lit(None).cast(StringType()).alias(
                SPARK_DEFAULT_INDEX_NAME)]
            new_column_labels = []
            num_args = len(signature(sfun).parameters)
            for label in self._internal.column_labels:
                spark_column = self._internal.spark_column_for(label)
                spark_type = self._internal.spark_type_for(label)
                is_numeric_or_boolean = isinstance(spark_type, (NumericType,
                    BooleanType))
                keep_column = not numeric_only or is_numeric_or_boolean
                if keep_column:
                    if num_args == 1:
                        scol = sfun(spark_column)
                    else:
                        assert num_args == 2
                        scol = sfun(spark_column, spark_type)
                    if min_count > 0:
                        scol = F.when(Frame._count_expr(spark_column,
                            spark_type) >= min_count, scol)
                    exprs.append(scol.alias(name_like_string(label)))
                    new_column_labels.append(label)
            if len(exprs) == 1:
                return Series([])
            sdf = self._internal.spark_frame.select(*exprs)
            with ks.option_context('compute.max_rows', 1):
                internal = InternalFrame(spark_frame=sdf,
                    index_spark_columns=[scol_for(sdf,
                    SPARK_DEFAULT_INDEX_NAME)], column_labels=
                    new_column_labels, column_label_names=self._internal.
                    column_label_names)
                return first_series(DataFrame(internal).transpose())
        else:
            limit = get_option('compute.shortcut_limit')
            pdf = self.head(limit + 1)._to_internal_pandas()
            pser = getattr(pdf, name)(axis=axis, numeric_only=numeric_only,
                **kwargs)
            if len(pdf) <= limit:
                return Series(pser)

            @pandas_udf(returnType=as_spark_type(pser.dtype.type))
            def calculate_columns_axis(*cols):
                return getattr(pd.concat(cols, axis=1), name)(axis=axis,
                    numeric_only=numeric_only, **kwargs)
            column_name = verify_temp_column_name(self._internal.
                spark_frame.select(self._internal.index_spark_columns),
                '__calculate_columns_axis__')
            sdf = self._internal.spark_frame.select(self._internal.
                index_spark_columns + [calculate_columns_axis(*self.
                _internal.data_spark_columns).alias(column_name)])
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.
                index_spark_column_names], index_names=self._internal.
                index_names, index_dtypes=self._internal.index_dtypes)
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
        if not isinstance(other, DataFrame) and (isinstance(other,
            IndexOpsMixin) or is_sequence(other)):
            raise ValueError(
                '%s with a sequence is currently not supported; however, got %s.'
                 % (op, type(other).__name__))
        if isinstance(other, DataFrame):
            if (self._internal.column_labels_level != other._internal.
                column_labels_level):
                raise ValueError('cannot join with no overlapping index names')
            if not same_anchor(self, other):

                def apply_op(kdf, this_column_labels, that_column_labels):
                    for this_label, that_label in zip(this_column_labels,
                        that_column_labels):
                        yield getattr(kdf._kser_for(this_label), op)(kdf.
                            _kser_for(that_label)).rename(this_label
                            ), this_label
                return align_diff_frames(apply_op, self, other, fillna=True,
                    how='full')
            else:
                applied = []
                column_labels = []
                for label in self._internal.column_labels:
                    if label in other._internal.column_labels:
                        applied.append(getattr(self._kser_for(label), op)(
                            other._kser_for(label)))
                    else:
                        applied.append(F.lit(None).cast(self._internal.
                            spark_type_for(label)).alias(name_like_string(
                            label)))
                    column_labels.append(label)
                for label in other._internal.column_labels:
                    if label not in column_labels:
                        applied.append(F.lit(None).cast(other._internal.
                            spark_type_for(label)).alias(name_like_string(
                            label)))
                        column_labels.append(label)
                internal = self._internal.with_new_columns(applied,
                    column_labels=column_labels)
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
    spark = CachedAccessor('spark', SparkFrameMethods)
    koalas = CachedAccessor('koalas', KoalasFrameMethods)

    def hist(self, bins=10, **kwds):
        return self.plot.hist(bins, **kwds)
    hist.__doc__ = KoalasPlotAccessor.hist.__doc__

    def kde(self, bw_method=None, ind=None, **kwds):
        return self.plot.kde(bw_method, ind, **kwds)
    kde.__doc__ = KoalasPlotAccessor.kde.__doc__
    add.__doc__ = _flex_doc_FRAME.format(desc='Addition', op_name='+',
        equiv='dataframe + other', reverse='radd')

    def radd(self, other):
        return other + self
    radd.__doc__ = _flex_doc_FRAME.format(desc='Addition', op_name='+',
        equiv='other + dataframe', reverse='add')

    def div(self, other):
        return self / other
    div.__doc__ = _flex_doc_FRAME.format(desc='Floating division', op_name=
        '/', equiv='dataframe / other', reverse='rdiv')
    divide = div

    def rdiv(self, other):
        return other / self
    rdiv.__doc__ = _flex_doc_FRAME.format(desc='Floating division', op_name
        ='/', equiv='other / dataframe', reverse='div')

    def truediv(self, other):
        return self / other
    truediv.__doc__ = _flex_doc_FRAME.format(desc='Floating division',
        op_name='/', equiv='dataframe / other', reverse='rtruediv')

    def rtruediv(self, other):
        return other / self
    rtruediv.__doc__ = _flex_doc_FRAME.format(desc='Floating division',
        op_name='/', equiv='other / dataframe', reverse='truediv')

    def mul(self, other):
        return self * other
    mul.__doc__ = _flex_doc_FRAME.format(desc='Multiplication', op_name='*',
        equiv='dataframe * other', reverse='rmul')
    multiply = mul

    def rmul(self, other):
        return other * self
    rmul.__doc__ = _flex_doc_FRAME.format(desc='Multiplication', op_name=
        '*', equiv='other * dataframe', reverse='mul')

    def sub(self, other):
        return self - other
    sub.__doc__ = _flex_doc_FRAME.format(desc='Subtraction', op_name='-',
        equiv='dataframe - other', reverse='rsub')
    subtract = sub

    def rsub(self, other):
        return other - self
    rsub.__doc__ = _flex_doc_FRAME.format(desc='Subtraction', op_name='-',
        equiv='other - dataframe', reverse='sub')

    def mod(self, other):
        return self % other
    mod.__doc__ = _flex_doc_FRAME.format(desc='Modulo', op_name='%', equiv=
        'dataframe % other', reverse='rmod')

    def rmod(self, other):
        return other % self
    rmod.__doc__ = _flex_doc_FRAME.format(desc='Modulo', op_name='%', equiv
        ='other % dataframe', reverse='mod')

    def pow(self, other):
        return self ** other
    pow.__doc__ = _flex_doc_FRAME.format(desc='Exponential power of series',
        op_name='**', equiv='dataframe ** other', reverse='rpow')

    def rpow(self, other):
        return other ** self
    rpow.__doc__ = _flex_doc_FRAME.format(desc='Exponential power', op_name
        ='**', equiv='other ** dataframe', reverse='pow')

    def floordiv(self, other):
        return self // other
    floordiv.__doc__ = _flex_doc_FRAME.format(desc='Integer division',
        op_name='//', equiv='dataframe // other', reverse='rfloordiv')

    def rfloordiv(self, other):
        return other // self
    rfloordiv.__doc__ = _flex_doc_FRAME.format(desc='Integer division',
        op_name='//', equiv='other // dataframe', reverse='floordiv')

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
        a   True   True
        b  False  False
        c  False   True
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
        a  True   True
        b  True  False
        c  True   True
        d  True  False
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
        a   True   True
        b   True  False
        c  False   True
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
        b   True   True
        c   True  False
        d   True   True
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

        >>> def power(x) -> float:
        ...     return x ** 2
        >>> df.applymap(power)
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489

        You can omit the type hint and let Koalas infer its type.

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

        >>> df.agg(['sum', 'min'])[['A', 'B', 'C']].sort_index()
                A     B     C
        min   1.0   2.0   3.0
        sum  12.0  15.0  18.0

        Different aggregations per column.

        >>> df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})[['A', 'B']].sort_index()
                A    B
        max   NaN  8.0
        min   1.0  2.0
        sum  12.0  NaN

        For multi-index columns:

        >>> df.columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C")])
        >>> df.agg(['sum', 'min'])[[("X", "A"), ("X", "B"), ("Y", "C")]].sort_index()
                X           Y
                A     B     C
        min   1.0   2.0   3.0
        sum  12.0  15.0  18.0

        >>> aggregated = df.agg({("X", "A") : ['sum', 'min'], ("X", "B") : ['min', 'max']})
        >>> aggregated[[("X", "A"), ("X", "B")]].sort_index()  # doctest: +NORMALIZE_WHITESPACE
                X
                A    B
        max   NaN  8.0
        min   1.0  2.0
        sum  12.0  NaN
        """
        from databricks.koalas.groupby import GroupBy
        if isinstance(func, list):
            if all(isinstance(f, str) for f in func):
                func = dict([(column, func) for column in self.columns])
            else:
                raise ValueError(
                    'If the given function is a list, it should only contains function names as strings.'
                    )
        if not isinstance(func, dict) or not all(is_name_like_value(key) and
            (isinstance(value, str) or isinstance(value, list) and all(
            isinstance(v, str) for v in value)) for key, value in func.items()
            ):
            raise ValueError(
                'aggs must be a dict mapping from column name to aggregate functions (string or list of strings).'
                )
        with option_context('compute.default_index_type', 'distributed'):
            kdf = DataFrame(GroupBy._spark_groupby(self, func))
            if LooseVersion(pyspark.__version__) >= LooseVersion('2.4'):
                return kdf.stack().droplevel(0)[list(func.keys())]
            else:
                pdf = kdf._to_internal_pandas().stack()
                pdf.index = pdf.index.droplevel()
                return ks.from_pandas(pdf[list(func.keys())])
    agg = aggregate

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
        polar         bear
        koala    marsupial
        label: population
        content: panda     1864
        polar    22000
        koala    80000
        """
        return ((label if len(label) > 1 else label[0], self._kser_for(
            label)) for label in self._internal.column_labels)

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
            k = row[internal_index_columns[0]] if len(internal_index_columns
                ) == 1 else tuple(row[c] for c in internal_index_columns)
            v = [row[c] for c in internal_data_columns]
            return k, v
        for k, v in map(extract_kv_from_spark_row, self._internal.
            resolved_copy.spark_frame.toLocalIterator()):
            s = pd.Series(v, index=columns, name=k)
            yield k, s

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
        >>> df = ks.DataFrame({'num_legs': [4, 2], 'num_wings': [0, 2]},
        ...                   index=['dog', 'hawk'])
        >>> df
              num_legs  num_wings
        dog          4          0
        hawk         2          2

        >>> for row in df.itertuples():
        ...     print(row)
        ...
        Koalas(Index='dog', num_legs=4, num_wings=0)
        Koalas(Index='hawk', num_legs=2, num_wings=2)

        By setting the `index` parameter to False we can remove the index
        as the first element of the tuple:

        >>> for row in df.itertuples(index=False):
        ...     print(row)
        ...
        Koalas(num_legs=4, num_wings=0)
        Koalas(num_legs=2, num_wings=2)

        With the `name` parameter set we set a custom name for the yielded
        namedtuples:

        >>> for row in df.itertuples(name='Animal'):
        ...     print(row)
        ...
        Animal(Index='dog', num_legs=4, num_wings=0)
        Animal(Index='hawk', num_legs=2, num_wings=2)
        """
        fields = list(self.columns)
        if index:
            fields.insert(0, 'Index')
        index_spark_column_names = self._internal.index_spark_column_names
        data_spark_column_names = self._internal.data_spark_column_names

        def extract_kv_from_spark_row(row):
            k = row[index_spark_column_names[0]] if len(
                index_spark_column_names) == 1 else tuple(row[c] for c in
                index_spark_column_names)
            v = [row[c] for c in data_spark_column_names]
            return k, v
        can_return_named_tuples = sys.version_info >= (3, 7) or len(self.
            columns) + index < 255
        if name is not None and can_return_named_tuples:
            itertuple = namedtuple(name, fields, rename=True)
            for k, v in map(extract_kv_from_spark_row, self._internal.
                resolved_copy.spark_frame.toLocalIterator()):
                yield itertuple._make(([k] if index else []) + list(v))
        else:
            for k, v in map(extract_kv_from_spark_row, self._internal.
                resolved_copy.spark_frame.toLocalIterator()):
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

        This function also works for Series:

        >>> df = ks.Series([1, 2, 3, 4, 5, 6, 7], name='x')  # doctest: +SKIP
        >>> df.to_clipboard(sep=',')  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # 0, 1
        ... # 1, 2
        ... # 2, 3
        ... # 3, 4
        ... # 4, 5
        ... # 5, 6
        ... # 6, 7
        """
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(kdf.
            _to_internal_pandas(), self.to_clipboard, pd.DataFrame.
            to_clipboard, args)

    def to_html(self, buf=None, columns=None, col_space=None, header=True,
        index=True, na_rep='NaN', formatters=None, float_format=None,
        sparsify=None, index_names=True, justify=None, max_rows=None,
        max_cols=None, show_dimensions=False, decimal='.', bold_rows=True,
        classes=None, escape=True, notebook=False, border=None, table_id=
        None, render_links=False):
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
        formatters : list or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. The result of this function must be a unicode string.
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
        return validate_arguments_and_invoke_function(kdf.
            _to_internal_pandas(), self.to_html, pd.DataFrame.to_html, args)

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
        index=True, na_rep='NaN', formatters=None, float_format=None,
        sparsify=None, index_names=True, justify=None, max_rows=None,
        max_cols=None, show_dimensions=False, decimal='.', line_width=None):
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
        formatters : list or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. The result of this function must be a unicode string.
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
        0     1     4
        1     2     5
        2     3     6

        >>> print(df.to_string(max_rows=2))
           col1  col2
        0     1     4
        1     2     5
        """
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self
        return validate_arguments_and_invoke_function(kdf.
            _to_internal_pandas(), self.to_string, pd.DataFrame.to_string, args
            )

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
        Name: col1, dtype: int64), ('col2', row1    0.50
        row2    0.75
        Name: col2, dtype: float64)]

        >>> df_dict = df.to_dict('split')
        >>> sorted(df_dict.items())  # doctest: +ELLIPSIS
        [('columns', ['col1', 'col2']), ('data', [[1..., 0.75]]), ('index', ['row1', 'row2'])]

        >>> df_dict = df.to_dict('records')
        >>> [sorted(values.items()) for values in df_dict]  # doctest: +ELLIPSIS
        [[('col1', 1...), ('col2', 0.5)], [('col1', 2...), ('col2', 0.75)]]

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
        [defaultdict(<class 'list'>, {'col..., 'col...}), defaultdict(<class 'list'>, {'col..., 'col...})]
        """
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(kdf.
            _to_internal_pandas(), self.to_dict, pd.DataFrame.to_dict, args)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True,
        index=True, na_rep='NaN', formatters=None, float_format=None,
        sparsify=None, index_names=True, bold_rows=False, column_format=
        None, longtable=None, escape=None, encoding=None, decimal='.',
        multicolumn=None, multicolumn_format=None, multirow=None):
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
        sparsify : bool, optional
            Set to False for a DataFrame with a hierarchical index to print every multiindex key at
            each row. By default, the value will be read from the config module.
        index_names : bool, default True
            Prints the names of the indexes.
        bold_rows : bool, default False
            Make the row labels bold in the output.
        column_format : str, optional
            The columns format as specified in LaTeX table format e.g. ‘rcl’ for 3 columns. By
            default, ‘l’ will be used for all columns except columns of numbers, which default
            to ‘r’.
        longtable : bool, optional
            By default, the value will be read from the pandas config module. Use a longtable
            environment instead of tabular. Requires adding a usepackage{longtable} to your LaTeX
            preamble.
        escape : bool, optional
            By default, the value will be read from the pandas config module. When set to False
            prevents from escaping latex special characters in column names.
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
        DataFrame.to_string : Render a DataFrame to a console-friendly
            tabular output.
        DataFrame.to_html : Render a DataFrame as an HTML table.


        Examples
        --------
        >>> df = ks.DataFrame({'name': ['Raphael', 'Donatello'],
        ...                    'mask': ['red', 'purple'],
        ...                    'weapon': ['sai', 'bo staff']},
        ...                   columns=['name', 'mask', 'weapon'])
        >>> print(df.to_latex(index=False)) # doctest: +NORMALIZE_WHITESPACE
        \\begin{tabular}{lll}
        \\toprule
              name &    mask &    weapon \\\\
        \\midrule
           Raphael &     red &       sai \\\\
         Donatello &  purple &  bo staff \\\\
        \\bottomrule
        \\end{tabular}
        <BLANKLINE>
        """
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(kdf.
            _to_internal_pandas(), self.to_latex, pd.DataFrame.to_latex, args)

    def transpose(self):
        """
        Transpose index and columns.

        Reflect the DataFrame over its main diagonal by writing rows as columns
        and vice-versa. The property :attr:`.T` is an accessor to the method
        :meth:`transpose`.

        .. note:: This method is based on an expensive operation due to the nature
            of big data. Internally it needs to generate each row for each value, and
            then group twice - it is a huge operation. To prevent misusage, this method
            has the 'compute.max_rows' default limit of input length, and raises a ValueError.

                >>> from databricks.koalas.config import option_context
                >>> with option_context('compute.max_rows', 1000):  # doctest: +NORMALIZE_WHITESPACE
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
            The transposed DataFrame.

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

        >>> df1_transposed = df1.T.sort_index()  # doctest: +SKIP
        >>> df1_transposed  # doctest: +SKIP
              0  1
        col1  1  2
        col2  3  4

        When the dtype is homogeneous in the original DataFrame, we get a
        transposed DataFrame with the same dtype:

        >>> df1.dtypes
        col1    int64
        col2    int64
        dtype: object
        >>> df1_transposed.dtypes  # doctest: +SKIP
        0    int64
        1    int64
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

        >>> df2_transposed = df2.T.sort_index()  # doctest: +SKIP
        >>> df2_transposed  # doctest: +SKIP
                  0     1
        age    12.0  22.0
        kids    0.0   0.0
        score   9.5   8.0

        When the DataFrame has mixed dtypes, we get a transposed DataFrame with
        the coerced dtype:

        >>> df2.dtypes
        score    float64
        kids       int64
        age        int64
        dtype: object

        >>> df2_transposed.dtypes  # doctest: +SKIP
        0    float64
        1    float64
        dtype: object
        """
        max_compute_count = get_option('compute.max_rows')
        if max_compute_count is not None:
            pdf = self.head(max_compute_count + 1)._to_internal_pandas()
            if len(pdf) > max_compute_count:
                raise ValueError(
                    "Current DataFrame has more then the given limit {0} rows. Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option' to retrieve to retrieve more than {0} rows. Note that, before changing the 'compute.max_rows', this operation is considerably expensive."
                    .format(max_compute_count))
            return DataFrame(pdf.transpose())
        pairs = F.explode(F.array(*[F.struct([F.lit(col).alias(
            SPARK_INDEX_NAME_FORMAT(i)) for i, col in enumerate(label)] + [
            self._internal.spark_column_for(label).alias('value')]) for
            label in self._internal.column_labels]))
        exploded_df = self._internal.spark_frame.withColumn('pairs', pairs
            ).select([F.to_json(F.struct(F.array([scol for scol in self.
            _internal.index_spark_columns]).alias('a'))).alias('index'), F.
            col('pairs.*')])
        internal_index_columns = [SPARK_INDEX_NAME_FORMAT(i) for i in range
            (self._internal.column_labels_level)]
        pivoted_df = exploded_df.groupBy(internal_index_columns).pivot('index')
        transposed_df = pivoted_df.agg(F.first(F.col('value')))
        new_data_columns = list(filter(lambda x: x not in
            internal_index_columns, transposed_df.columns))
        column_labels = [(None if len(label) == 1 and label[0] is None else
            label) for label in (tuple(json.loads(col)['a']) for col in
            new_data_columns)]
        internal = InternalFrame(spark_frame=transposed_df,
            index_spark_columns=[scol_for(transposed_df, col) for col in
            internal_index_columns], index_names=self._internal.
            column_label_names, column_labels=column_labels,
            data_spark_columns=[scol_for(transposed_df, col) for col in
            new_data_columns], column_label_names=self._internal.index_names)
        return DataFrame(internal)
    T = property(transpose)

    def apply_batch(self, func, args=(), **kwds):
        warnings.warn(
            'DataFrame.apply_batch is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.'
            , FutureWarning)
        return self.koalas.apply_batch(func, args=args, **kwds)
    apply_batch.__doc__ = KoalasFrameMethods.apply_batch.__doc__

    def map_in_pandas(self, func):
        warnings.warn(
            'DataFrame.map_in_pandas is deprecated as of DataFrame.koalas.apply_batch. Please use the API instead.'
            , FutureWarning)
        return self.koalas.apply_batch(func)
    map_in_pandas.__doc__ = KoalasFrameMethods.apply_batch.__doc__

    def apply(self, func, axis=0, args=(), **kwds):
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

            To avoid this, specify the return type as `Series` or scalar value in ``func``,
            for instance, as below:

            >>> def square(s) -> ks.Series[np.int32]:
            ...     return s ** 2

            Koalas uses return type hint and does not try to infer the type.

            In case when axis is 1, it requires to specify `DataFrame` or scalar value
            with type hints as below:

            >>> def plus_one(x) -> ks.DataFrame[float, float]:
            ...     return x + 1

            If the return type is specified as `DataFrame`, the output column names become
            `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
            DataFrame in ``func``.

            To specify the column names, you can assign them in a pandas friendly style as below:

            >>> def plus_one(x) -> ks.DataFrame["a": float, "b": float]:
            ...     return x + 1

            >>> pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]})
            >>> def plus_one(x) -> ks.DataFrame[zip(pdf.dtypes, pdf.columns)]:
            ...     return x + 1

            However, this way switches the index type to default index type in the output
            because the type hint cannot express the index type at this moment. Use
            `reset_index()` to keep index as a workaround.

            When the given function has the return type annotated, the original index of the
            DataFrame will be lost and then a default index will be attached to the result.
            Please be careful about configuring the default index. See also `Default Index Type
            <https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type>`_.

        Parameters
        ----------
        func : function
            Function to apply to each column or row.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the function is applied:

            * 0 or 'index': apply function to each column.
            * 1 or 'columns': apply function to each row.
        args : tuple
            Positional arguments to pass to `func` in addition to the
            array/series.
        **kwds
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

        >>> def sqrt(x) -> ks.Series[float]:
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
           c0  c1
        0   8  13
        1   8  13
        2   8  13
        """
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.series import first_series
        if not isinstance(func, types.FunctionType):
            assert callable(func
                ), 'the first argument should be a callable function.'
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = validate_axis(axis)
        should_return_series = False
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get('return', None)
        should_infer_schema = return_sig is None
        should_use_map_in_pandas = LooseVersion(pyspark.__version__) >= '3.0'

        def apply_func(pdf):
            pdf_or_pser = pdf.apply(func, axis=axis, args=args, **kwds)
            if isinstance(pdf_or_pser, pd.Series):
                return pdf_or_pser.to_frame()
            else:
                return pdf_or_pser
        self_applied = DataFrame(self._internal.resolved_copy)
        column_labels = None
        if should_infer_schema:
            limit = get_option('compute.shortcut_limit')
            pdf = self_applied.head(limit + 1)._to_internal_pandas()
            applied = pdf.apply(func, axis=axis, args=args, **kwds)
            kser_or_kdf = ks.from_pandas(applied)
            if len(pdf) <= limit:
                return kser_or_kdf
            kdf = kser_or_kdf
            if isinstance(kser_or_kdf, ks.Series):
                should_return_series = True
                kdf = kser_or_kdf._kdf
            return_schema = force_decimal_precision_scale(
                as_nullable_spark_type(kdf._internal.
                to_internal_spark_frame.schema))
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(self_applied
                    , apply_func, return_schema, retain_index=True)
                sdf = (self_applied._internal.to_internal_spark_frame.
                    mapInPandas(lambda iterator: map(output_func, iterator),
                    schema=return_schema))
            else:
                sdf = GroupBy._spark_group_map_apply(self_applied,
                    apply_func, (F.spark_partition_id(),), return_schema,
                    retain_index=True)
            internal = kdf._internal.with_new_sdf(sdf)
        else:
            return_type = infer_return_type(func)
            require_index_axis = isinstance(return_type, SeriesType)
            require_column_axis = isinstance(return_type, DataFrameType)
            if require_index_axis:
                if axis != 0:
                    raise TypeError(
                        "The given function should specify a scalar or a series as its type hints when axis is 0 or 'index'; however, the return type was %s"
                         % return_sig)
                return_schema = cast(SeriesType, return_type).spark_type
                fields_types = zip(self_applied.columns, [return_schema] *
                    len(self_applied.columns))
                return_schema = StructType([StructField(c, t) for c, t in
                    fields_types])
                data_dtypes = [cast(SeriesType, return_type).dtype] * len(
                    self_applied.columns)
            elif require_column_axis:
                if axis != 1:
                    raise TypeError(
                        "The given function should specify a scalar or a frame as its type hints when axis is 1 or 'column'; however, the return type was %s"
                         % return_sig)
                return_schema = cast(DataFrameType, return_type).spark_type
                data_dtypes = cast(DataFrameType, return_type).dtypes
            else:
                should_return_series = True
                return_schema = cast(ScalarType, return_type).spark_type
                return_schema = StructType([StructField(
                    SPARK_DEFAULT_SERIES_NAME, return_schema)])
                data_dtypes = [cast(ScalarType, return_type).dtype]
                column_labels = [None]
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(self_applied
                    , apply_func, return_schema, retain_index=False)
                sdf = (self_applied._internal.to_internal_spark_frame.
                    mapInPandas(lambda iterator: map(output_func, iterator),
                    schema=return_schema))
            else:
                sdf = GroupBy._spark_group_map_apply(self_applied,
                    apply_func, (F.spark_partition_id(),), return_schema,
                    retain_index=False)
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=
                None, column_labels=column_labels, data_dtypes=data_dtypes)
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

        >>> df.columns = [('X', 'A'), ('X', 'B')]
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

        >>> def calculation(x, y, z) -> ks.Series[int]:
        ...     return x ** y + z
        >>> df.transform(calculation, y=10, z=20)  # doctest: +NORMALIZE_WHITESPACE
              X
              A      B
        0    20     21
        1    21   1044
        2  1044  59069
        """
        if not isinstance(func, types.FunctionType):
            assert callable(func
                ), 'the first argument should be a callable function.'
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get('return', None)
        should_infer_schema = return_sig is None
        if should_infer_schema:
            limit = get_option('compute.shortcut_limit')
            pdf = self.head(limit + 1)._to_internal_pandas()
            transformed = pdf.transform(func, axis, *args, **kwargs)
            kdf = DataFrame(transformed)
            if len(pdf) <= limit:
                return kdf
            applied = []
            for input_label, output_label in zip(self._internal.
                column_labels, kdf._internal.column_labels):
                kser = self._kser_for(input_label)
                dtype = kdf._internal.dtype_for(output_label)
                return_schema = force_decimal_precision_scale(
                    as_nullable_spark_type(kdf._internal.spark_type_for(
                    output_label)))
                applied.append(kser.koalas._transform_batch(func=lambda c:
                    func(c, *args, **kwargs), return_type=SeriesType(dtype,
                    return_schema)))
            internal = self._internal.with_new_columns(applied, data_dtypes
                =kdf._internal.data_dtypes)
            return DataFrame(internal)
        else:
            return self._apply_series_op(lambda kser: kser.koalas.
                transform_batch(func, *args, **kwargs))

    def transform_batch(self, func, *args, **kwargs):
        warnings.warn(
            'DataFrame.transform_batch is deprecated as of DataFrame.koalas.transform_batch. Please use the API instead.'
            , FutureWarning)
        return self.koalas.transform_batch(func, *args, **kwargs)
    transform_batch.__doc__ = KoalasFrameMethods.transform_batch.__doc__

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
             name   class max_speed
        0  falcon    bird     389.0
        1  parrot    bird      24.0
        2    lion  mammal      80.5
        3  monkey  mammal       NaN

        >>> df.pop('a')
             name   class
        0  falcon    bird
        1  parrot    bird
        2    lion  mammal
        3  monkey  mammal

        >>> df
                  b
          max_speed
        0     389.0
        1      24.0
        2      80.5
        3       NaN
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

        See Also
        --------
        DataFrame.loc : Access a group of rows and columns
            by label(s) or a boolean array.
        DataFrame.iloc : Purely integer-location based indexing
            for selection by position.

        Examples
        --------
        >>> d = {'num_legs': [4, 4, 2, 2],
        ...      'num_wings': [0, 0, 2, 2],
        ...      'class': ['mammal', 'mammal', 'mammal', 'bird'],
        ...      'animal': ['cat', 'dog', 'bat', 'penguin'],
        ...      'locomotion': ['walks', 'walks', 'flies', 'walks']}
        >>> df = ks.DataFrame(data=d)
        >>> df = df.set_index(['class', 'animal', 'locomotion'])
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                                   num_legs  num_wings
        class  animal  locomotion
        mammal cat     walks              4          0
               dog     walks              4          0
               bat     flies              2          2
        bird   penguin walks              2          2

        Get values at specified index

        >>> df.xs('mammal')  # doctest: +NORMALIZE_WHITESPACE
                           num_legs  num_wings
        animal locomotion
        cat    walks              4          0
        dog    walks              4          0
        bat    flies              2          2

        Get values at several indexes

        >>> df.xs(('mammal', 'dog'))  # doctest: +NORMALIZE_WHITESPACE
                    num_legs  num_wings
        locomotion
        walks              4          0

        >>> df.xs(('mammal', 'dog', 'walks'))  # doctest: +NORMALIZE_WHITESPACE
        num_legs     4
        num_wings    0
        Name: (mammal, dog, walks), dtype: int64

        Get values at specified index and level

        >>> df.xs('cat', level=1)  # doctest: +NORMALIZE_WHITESPACE
                           num_legs  num_wings
        class  locomotion
        mammal walks              4          0
        """
        from databricks.koalas.series import first_series
        if not is_name_like_value(key):
            raise ValueError(
                "'key' should be a scalar value or tuple that contains scalar values"
                )
        if level is not None and is_name_like_tuple(key):
            raise KeyError(key)
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        if not is_name_like_tuple(key):
            key = key,
        if len(key) > self._internal.index_level:
            raise KeyError('Key length ({}) exceeds index depth ({})'.
                format(len(key), self._internal.index_level))
        if level is None:
            level = 0
        rows = [(self._internal.index_spark_columns[lvl] == index) for lvl,
            index in enumerate(key, level)]
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
            index_spark_columns = internal.index_spark_columns[:level
                ] + internal.index_spark_columns[level + len(key):]
            index_names = internal.index_names[:level] + internal.index_names[
                level + len(key):]
            index_dtypes = internal.index_dtypes[:level
                ] + internal.index_dtypes[level + len(key):]
            internal = internal.copy(index_spark_columns=
                index_spark_columns, index_names=index_names, index_dtypes=
                index_dtypes).resolved_copy
            return DataFrame(internal)

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
        DataFrame
            Data from the original object filtered to the specified dates range.

        Raises
        ------
        TypeError
            If the index is not  a :class:`DatetimeIndex`

        See Also
        --------
        at_time : Select values at a particular time of the day.
        first : Select initial periods of time series based on a date offset.
        last : Select final periods of time series based on a date offset.
        DatetimeIndex.indexer_between_time : Get just the index locations for
            values between particular times of the day.

        Examples
        --------
        >>> idx = pd.date_range('2018-04-09', periods=4, freq='1D20min')
        >>> kdf = ks.DataFrame({'A': [1, 2, 3, 4]}, index=idx)
        >>> kdf
                             A
        2018-04-09 00:00:00  1
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3
        2018-04-12 01:00:00  4

        >>> kdf.between_time('0:15', '0:45')
                             A
        2018-04-10 00:20:00  2
        2018-04-11 00:40:00  3

        You get the times that are *not* between two times by setting
        ``start_time`` later than ``end_time``:

        >>> kdf.between_time('0:45', '0:15')
                             A
        2018-04-09 00:00:00  1
        2018-04-12 01:00:00  4
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'between_time currently only works for axis=0')
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError('Index must be DatetimeIndex')
        kdf = self.copy()
        kdf.index.name = verify_temp_column_name(kdf, '__index_name__')
        return_types = [kdf.index.dtype] + list(kdf.dtypes)

        def pandas_between_time(pdf):
            return pdf.between_time(start_time, end_time, include_start,
                include_end).reset_index()
        with option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_between_time)
        return DataFrame(self._internal.copy(spark_frame=kdf._internal.
            spark_frame, index_spark_columns=kdf._internal.
            data_spark_columns[:1], data_spark_columns=kdf._internal.
            data_spark_columns[1:]))

    def at_time(self, time, asof=False, axis=0):
        """
        Select values at particular time of day (e.g., 9:30AM).

        Parameters
        ----------
        time : datetime.time or str
        axis : {0 or 'index', 1 or 'columns'}, default 0

        Returns
        -------
        DataFrame

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
        >>> kdf = ks.DataFrame({'A': [1, 2, 3, 4]}, index=idx)
        >>> kdf
                             A
        2018-04-09 00:00:00  1
        2018-04-09 12:00:00  2
        2018-04-10 00:00:00  3
        2018-04-10 12:00:00  4

        >>> kdf.at_time('12:00')
                             A
        2018-04-09 12:00:00  2
        2018-04-10 12:00:00  4
        """
        if asof:
            raise NotImplementedError("'asof' argument is not supported")
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('at_time currently only works for axis=0'
                )
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError('Index must be DatetimeIndex')
        kdf = self.copy()
        kdf.index.name = verify_temp_column_name(kdf, '__index_name__')
        return_types = [kdf.index.dtype] + list(kdf.dtypes)
        if LooseVersion(pd.__version__) < LooseVersion('0.24'):

            def pandas_at_time(pdf):
                return pdf.at_time(time, asof).reset_index()
        else:

            def pandas_at_time(pdf):
                return pdf.at_time(time, asof, axis).reset_index()
        with option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_at_time)
        return DataFrame(self._internal.copy(spark_frame=kdf._internal.
            spark_frame, index_spark_columns=kdf._internal.
            data_spark_columns[:1], data_spark_columns=kdf._internal.
            data_spark_columns[1:]))

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

        >>> df1.where(df1 > 0).sort_index()
             A      B
        0  NaN  100.0
        1  1.0  200.0
        2  2.0  300.0
        3  3.0  400.0
        4  4.0  500.0

        >>> df1.where(df1 > 1, 10).sort_index()
            A    B
        0  10  100
        1  10  200
        2   2  300
        3   3  400
        4   4  500

        >>> df1.where(df1 > 1, df1 + 100).sort_index()
             A    B
        0  100  100
        1  101  200
        2    2  300
        3    3  400
        4    4  500

        >>> df1.where(df1 > 1, df2).sort_index()
           A    B
        0  0  100
        1 -1  200
        2  2  300
        3  3  400
        4  4  500

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
            A   B
        0 NaN NaN
        1 NaN NaN
        2 NaN NaN
        3 NaN NaN
        4 NaN NaN

        When the type of cond is Series, it just check boolean regardless of column name

        >>> cond = ks.Series([1, 2]) > 1
        >>> cond
        0    False
        1     True
        dtype: bool

        >>> df1.where(cond).sort_index()
             A      B
        0  NaN    NaN
        1  1.0  200.0
        2  NaN    NaN
        3  NaN    NaN
        4  NaN    NaN

        >>> reset_option("compute.ops_on_diff_frames")
        """
        from databricks.koalas.series import Series
        tmp_cond_col_name = '__tmp_cond_col_{}__'.format
        tmp_other_col_name = '__tmp_other_col_{}__'.format
        kdf = self.copy()
        tmp_cond_col_names = [tmp_cond_col_name(name_like_string(label)) for
            label in self._internal.column_labels]
        if isinstance(cond, DataFrame):
            cond = cond[[(cond._internal.spark_column_for(label) if label in
                cond._internal.column_labels else F.lit(False)).alias(name) for
                label, name in zip(self._internal.column_labels,
                tmp_cond_col_names)]]
            kdf[tmp_cond_col_names] = cond
        elif isinstance(cond, Series):
            cond = cond.to_frame()
            cond = cond[[cond._internal.data_spark_columns[0].alias(name) for
                name in tmp_cond_col_names]]
            kdf[tmp_cond_col_names] = cond
        else:
            raise ValueError('type of cond must be a DataFrame or Series')
        tmp_other_col_names = [tmp_other_col_name(name_like_string(label)) for
            label in self._internal.column_labels]
        if isinstance(other, DataFrame):
            other = other[[(other._internal.spark_column_for(label) if 
                label in other._internal.column_labels else F.lit(np.nan)).
                alias(name) for label, name in zip(self._internal.
                column_labels, tmp_other_col_names)]]
            kdf[tmp_other_col_names] = other
        elif isinstance(other, Series):
            other = other.to_frame()
            other = other[[other._internal.data_spark_columns[0].alias(name
                ) for name in tmp_other_col_names]]
            kdf[tmp_other_col_names] = other
        else:
            for label in self._internal.column_labels:
                kdf[tmp_other_col_name(name_like_string(label))] = other
        data_spark_columns = []
        for label in self._internal.column_labels:
            data_spark_columns.append(F.when(kdf[tmp_cond_col_name(
                name_like_string(label))].spark.column, kdf._internal.
                spark_column_for(label)).otherwise(kdf[tmp_other_col_name(
                name_like_string(label))].spark.column).alias(kdf._internal
                .spark_column_name_for(label)))
        return DataFrame(kdf._internal.with_new_columns(data_spark_columns,
            column_labels=self._internal.column_labels))

    def mask(self, cond, other=np.nan):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : boolean DataFrame
            Where cond is False, keep the original value. Where True,
            replace with corresponding value from other.
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

        >>> df1.mask(df1 > 0).sort_index()
             A   B
        0  0.0 NaN
        1  NaN NaN
        2  NaN NaN
        3  NaN NaN
        4  NaN NaN

        >>> df1.mask(df1 > 1, 10).sort_index()
            A   B
        0   0  10
        1   1  10
        2  10  10
        3  10  10
        4  10  10

        >>> df1.mask(df1 > 1, df1 + 100).sort_index()
             A    B
        0    0  200
        1    1  300
        2  102  400
        3  103  500
        4  104  600

        >>> df1.mask(df1 > 1, df2).sort_index()
           A    B
        0  0 -100
        1  1 -200
        2 -2 -300
        3 -3 -400
        4 -4 -500

        >>> reset_option("compute.ops_on_diff_frames")
        """
        from databricks.koalas.series import Series
        if not isinstance(cond, (DataFrame, Series)):
            raise ValueError('type of cond must be a DataFrame or Series')
        cond_inversed = cond._apply_series_op(lambda kser: ~kser)
        return self.where(cond_inversed, other)

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
        return len(self._internal.column_labels
            ) == 0 or self._internal.resolved_copy.spark_frame.rdd.isEmpty()

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
            warnings.warn("'style' property will only use top %s rows." %
                max_results, UserWarning)
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
            Modify the DataFrame in place (do not create a new object).

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
        2012  1     55
        2014  4     40
        2013  7     84
        2014  10    31
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if is_name_like_tuple(keys):
            keys = [keys]
        elif is_name_like_value(keys):
            keys = [(keys,)]
        else:
            keys = [(key if is_name_like_tuple(key) else (key,)) for key in
                keys]
        columns = set(self._internal.column_labels)
        for key in keys:
            if key not in columns:
                raise KeyError(name_like_string(key))
        if drop:
            column_labels = [label for label in self._internal.
                column_labels if label not in keys]
        else:
            column_labels = self._internal.column_labels
        if append:
            index_spark_columns = self._internal.index_spark_columns + [self
                ._internal.spark_column_for(label) for label in keys]
            index_names = self._internal.index_names + keys
            index_dtypes = self._internal.index_dtypes + [self._internal.
                dtype_for(label) for label in keys]
        else:
            index_spark_columns = [self._internal.spark_column_for(label) for
                label in keys]
            index_names = keys
            index_dtypes = [self._internal.dtype_for(label) for label in keys]
        internal = self._internal.copy(index_spark_columns=
            index_spark_columns, index_names=index_names, index_dtypes=
            index_dtypes, column_labels=column_labels, data_spark_columns=[
            self._internal.spark_column_for(label) for label in
            column_labels], data_dtypes=[self._internal.dtype_for(label) for
            label in column_labels])
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def reset_index(self, level=None, drop=False, inplace=False, col_level=
        0, col_fill=''):
        """Reset the index, or a level of it.

        For DataFrame with multi-level index, return new DataFrame with labeling information in
        the columns under the index names, defaulting to 'level_0', 'level_1', etc. if any are None.
        For a standard index, the index name will be used (if set), otherwise a default 'index' or
        'level_0' (if 'index' is already taken) will be used.

        Parameters
        ----------
        level : int, str, tuple, or list, default None
            Only remove the given levels from the index. Removes all levels by
            default.
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
            If the columns have multiple levels, determines how the other
            levels are named. If None then the index name is repeated.

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
                         max    type
        class  name
        bird   falcon  389.0     fly
               parrot   24.0     fly
        mammal lion     80.5     run
               monkey    NaN    jump

        If the index has multiple levels, we can reset a subset of them:

        >>> df.reset_index(level='class')  # doctest: +NORMALIZE_WHITESPACE
                 class  speed species
                          max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        If we are not dropping the index, by default, it is placed in the top
        level. We can place it in another level:

        >>> df.reset_index(level='class', col_level=1)  # doctest: +NORMALIZE_WHITESPACE
                        speed species
                 class    max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        When the index is inserted under another level, we can specify under
        which one with the parameter `col_fill`:

        >>> df.reset_index(level='class', col_level=1,
        ...                col_fill='species')  # doctest: +NORMALIZE_WHITESPACE
                      species  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump

        If we specify a nonexistent level for `col_fill`, it is created:

        >>> df.reset_index(level='class', col_level=1,
        ...                col_fill='genus')  # doctest: +NORMALIZE_WHITESPACE
                        genus  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        multi_index = self._internal.index_level > 1

        def rename(index):
            if multi_index:
                return 'level_{}'.format(index),
            elif ('index',) not in self._internal.column_labels:
                return 'index',
            else:
                return 'level_{}'.format(index),
        if level is None:
            new_column_labels = [(name if name is not None else rename(i)) for
                i, name in enumerate(self._internal.index_names)]
            new_data_spark_columns = [scol.alias(name_like_string(label)) for
                scol, label in zip(self._internal.index_spark_columns,
                new_column_labels)]
            new_data_dtypes = self._internal.index_dtypes
            index_spark_columns = []
            index_names = []
            index_dtypes = []
        else:
            if is_list_like(level):
                level = list(level)
            if isinstance(level, int) or is_name_like_tuple(level):
                level = [level]
            elif is_name_like_value(level):
                level = [(level,)]
            else:
                level = [(lvl if isinstance(lvl, int) or is_name_like_tuple
                    (lvl) else (lvl,)) for lvl in level]
            if all(isinstance(l, int) for l in level):
                for lev in level:
                    if lev >= self._internal.index_level:
                        raise IndexError(
                            'Too many levels: Index has only {} level, not {}'
                            .format(self._internal.index_level, lev + 1))
                idx = level
            elif all(is_name_like_tuple(lev) for lev in level):
                idx = []
                for l in level:
                    try:
                        i = self._internal.index_names.index(l)
                        idx.append(i)
                    except ValueError:
                        if multi_index:
                            raise KeyError('Level unknown not found')
                        else:
                            raise KeyError(
                                'Level unknown must be same as name ({})'.
                                format(name_like_string(self._internal.
                                index_names[0])))
            else:
                raise ValueError('Level should be all int or all string.')
            idx.sort()
            new_column_labels = []
            new_data_spark_columns = []
            new_data_dtypes = []
            index_spark_columns = self._internal.index_spark_columns.copy()
            index_names = self._internal.index_names.copy()
            index_dtypes = self._internal.index_dtypes.copy()
            for i in idx[::-1]:
                name = index_names.pop(i)
                new_column_labels.insert(0, name if name is not None else
                    rename(i))
                scol = index_spark_columns.pop(i)
                new_data_spark_columns.insert(0, scol.alias(
                    name_like_string(name)))
                new_data_dtypes.insert(0, index_dtypes.pop(i))
        if drop:
            new_data_spark_columns = []
            new_column_labels = []
            new_data_dtypes = []
        for label in new_column_labels:
            if label in self._internal.column_labels:
                raise ValueError('cannot insert {}, already exists'.format(
                    name_like_string(label)))
        if self._internal.column_labels_level > 1:
            column_depth = len(self._internal.column_labels[0])
            if col_level >= column_depth:
                raise IndexError(
                    'Too many levels: Index has only {} levels, not {}'.
                    format(column_depth, col_level + 1))
            if any(col_level + len(label) > column_depth for label in
                new_column_labels):
                raise ValueError(
                    'Item must have length equal to number of levels.')
            new_column_labels = [tuple([col_fill] * col_level + list(label) +
                [col_fill] * (column_depth - (len(label) + col_level))) for
                label in new_column_labels]
        internal = self._internal.copy(index_spark_columns=
            index_spark_columns, index_names=index_names, index_dtypes=
            index_dtypes, column_labels=new_column_labels + self._internal.
            column_labels, data_spark_columns=new_data_spark_columns + self
            ._internal.data_spark_columns, data_dtypes=new_data_dtypes +
            self._internal.data_dtypes)
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
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
        >>> df = ks.DataFrame([(.2, .3), (.0, None), (.6, None), (.2, .1)])
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
    isna = isnull

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
        >>> df = ks.DataFrame([(.2, .3), (.0, None), (.6, None), (.2, .1)])
        >>> df.notnull()
              0      1
        0  True   True
        1  True  False
        2  True  False
        3  True   True

        >>> df = ks.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
        >>> df.notnull()
              0      1     2
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
            raise ValueError(
                '"column" should be a scalar value or tuple that contains scalar values'
                )
        if is_name_like_tuple(column):
            if len(column) != len(self.columns.levels):
                raise ValueError(
                    '"column" must have length equal to number of column levels.'
                    )
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
        Copy of input DataFrame, shifted.

        Examples
        --------
        >>> df = ks.DataFrame({'Col1': [10, 20, 15, 30, 45],
        ...                    'Col2': [13, 23, 18, 33, 48],
        ...                    'Col3': [17, 27, 22, 37, 52]},
        ...                   columns=['Col1', 'Col2', 'Col3'])

        >>> df.shift(periods=3)
           Col1  Col2  Col3
        0   NaN   NaN   NaN
        1   NaN   NaN   NaN
        2   NaN   NaN   NaN
        3  10.0  13.0  17.0
        4  20.0  23.0  27.0

        >>> df.shift(periods=3, fill_value=0)
           Col1  Col2  Col3
        0     0     0     0
        1     0     0     0
        2     0     0     0
        3    10    13    17
        4    20    23    27

        """
        return self._apply_series_op(lambda kser: kser._shift(periods,
            fill_value), should_resolve=True)

    def diff(self, periods=1, axis=0):
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another element in the
        DataFrame (default is the element in the same column of the previous row).

        .. note:: the current implementation of diff uses Spark's Window without
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
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        return self._apply_series_op(lambda kser: kser._diff(periods),
            should_resolve=True)

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
        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [np.nan, 3, np.nan]})
        >>> df.nunique()
        A    3
        B    1
        dtype: int64

        >>> df.nunique(dropna=False)
        A    3
        B    2
        dtype: int64

        On big data, we recommend using the approximate algorithm to speed up this function.
        The result will be very close to the exact unique count.

        >>> df.nunique(approx=True)
        A    3
        B    1
        dtype: int64
        """
        from databricks.koalas.series import first_series
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        sdf = self._internal.spark_frame.select([F.lit(None).cast(
            StringType()).alias(SPARK_DEFAULT_INDEX_NAME)] + [self.
            _kser_for(label)._nunique(dropna, approx, rsd) for label in
            self._internal.column_labels])
        with ks.option_context('compute.max_rows', 1):
            internal = self._internal.copy(spark_frame=sdf,
                index_spark_columns=[scol_for(sdf, SPARK_DEFAULT_INDEX_NAME
                )], index_names=[None], index_dtypes=[None],
                data_spark_columns=[scol_for(sdf, col) for col in self.
                _internal.data_spark_column_names], data_dtypes=None)
            return first_series(DataFrame(internal).transpose())

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
            of `decimals` which are not columns of the input will be
            ignored.

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
        ...                    columns=['A', 'B', 'C'],
        ...                    index=['first', 'second', 'third'])
        >>> df
                       A         B         C
        first   0.028208  0.992815  0.173891
        second  0.038683  0.645646  0.577595
        third   0.877076  0.149370  0.491027

        >>> df.round(2)
                   A     B     C
        first   0.03  0.99  0.17
        second  0.04  0.65  0.58
        third   0.88  0.15  0.49

        >>> df.round({'A': 1, 'C': 2})
                  A         B     C
        first   0.0  0.992815  0.17
        second  0.0  0.645646  0.58
        third   0.9  0.149370  0.49

        >>> decimals = ks.Series([1, 0, 2], index=['A', 'B', 'C'])
        >>> df.round(decimals)
                  A    B     C
        first   0.0  1.0  0.17
        second  0.0  1.0  0.58
        third   0.9  0.0  0.49
        """
        if isinstance(decimals, ks.Series):
            decimals = {(k if isinstance(k, tuple) else (k,)): v for k, v in
                decimals._to_internal_pandas().items()}
        elif isinstance(decimals, dict):
            decimals = {(k if is_name_like_tuple(k) else (k,)): v for k, v in
                decimals.items()}
        elif isinstance(decimals, int):
            decimals = {k: decimals for k in self._internal.column_labels}
        else:
            raise ValueError(
                'decimals must be an integer, a dict-like or a Series')

        def op(kser):
            label = kser._column_label
            if label in decimals:
                return F.round(kser.spark.column, decimals[label]).alias(kser
                    ._internal.data_spark_column_names[0])
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
                subset = [(sub if is_name_like_tuple(sub) else (sub,)) for
                    sub in subset]
            diff = set(subset).difference(set(self._internal.column_labels))
            if len(diff) > 0:
                raise KeyError(', '.join([name_like_string(d) for d in diff]))
        group_cols = [self._internal.spark_column_name_for(label) for label in
            subset]
        sdf = self._internal.resolved_copy.spark_frame
        column = verify_temp_column_name(sdf, '__duplicated__')
        if keep == 'first' or keep == 'last':
            if keep == 'first':
                ord_func = spark.functions.asc
            else:
                ord_func = spark.functions.desc
            window = Window.partitionBy(group_cols).orderBy(ord_func(
                NATURAL_ORDER_COLUMN_NAME)).rowsBetween(Window.
                unboundedPreceding, Window.currentRow)
            sdf = sdf.withColumn(column, F.row_number().over(window) > 1)
        elif not keep:
            window = Window.partitionBy(group_cols).rowsBetween(Window.
                unboundedPreceding, Window.unboundedFollowing)
            sdf = sdf.withColumn(column, F.count('*').over(window) > 1)
        else:
            raise ValueError("'keep' only supports 'first', 'last' and False")
        return sdf, column

    def duplicated(self, subset=None, keep='first'):
        """
        Return boolean Series denoting duplicate rows, optionally only considering certain columns.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates,
            by default use all of the columns
        keep : {'first', 'last', False}, default 'first'
           - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
           - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
           - False : Mark all duplicates as ``True``.

        Returns
        -------
        duplicated : Series

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 1, 3], 'b': [1, 1, 1, 4], 'c': [1, 1, 1, 5]},
        ...                   columns = ['a', 'b', 'c'])
        >>> df
           a  b  c
        0  1  1  1
        1  1  1  1
        2  1  1  1
        3  3  4  5

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
        from databricks.koalas.series import first_series
        sdf, column = self._mark_duplicates(subset, keep)
        sdf = sdf.select(self._internal.index_spark_columns + [scol_for(sdf,
            column).alias(SPARK_DEFAULT_SERIES_NAME)])
        return first_series(DataFrame(InternalFrame(spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self.
            _internal.index_spark_column_names], index_names=self._internal
            .index_names, index_dtypes=self._internal.index_dtypes,
            column_labels=[None], data_spark_columns=[scol_for(sdf,
            SPARK_DEFAULT_SERIES_NAME)])))

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
        0   -4
        1    5
        dtype: int64

        Note how shuffling of the objects does not change the result.

        >>> kser2 = kser.reindex([1, 0, 2, 3])
        >>> kdf.dot(kser2)
        0   -4
        1    5
        dtype: int64
        >>> kdf @ kser2
        0   -4
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
            internal = InternalFrame(spark_frame=self, index_spark_columns=
                index_spark_columns, index_names=index_names)
            return DataFrame(internal)

    def cache(self):
        warnings.warn(
            'DataFrame.cache is deprecated as of DataFrame.spark.cache. Please use the API instead.'
            , FutureWarning)
        return self.spark.cache()
    cache.__doc__ = SparkFrameMethods.cache.__doc__

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        warnings.warn(
            'DataFrame.persist is deprecated as of DataFrame.spark.persist. Please use the API instead.'
            , FutureWarning)
        return self.spark.persist(storage_level)
    persist.__doc__ = SparkFrameMethods.persist.__doc__

    def hint(self, name, *parameters):
        warnings.warn(
            'DataFrame.hint is deprecated as of DataFrame.spark.hint. Please use the API instead.'
            , FutureWarning)
        return self.spark.hint(name, *parameters)
    hint.__doc__ = SparkFrameMethods.hint.__doc__

    def to_table(self, name, format=None, mode='overwrite', partition_cols=
        None, index_col=None, **options):
        return self.spark.to_table(name, format, mode, partition_cols,
            index_col, **options)
    to_table.__doc__ = SparkFrameMethods.to_table.__doc__

    def to_delta(self, path, mode='overwrite', partition_cols=None,
        index_col=None, **options):
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
        if 'options' in options and isinstance(options.get('options'), dict
            ) and len(options) == 1:
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='delta',
            partition_cols=partition_cols, index_col=index_col, **options)

    def to_parquet(self, path, mode='overwrite', partition_cols=None,
        compression=None, index_col=None, **options):
        """
        Write the DataFrame out as a Parquet file or directory.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'},
            default 'overwrite'. Specifies the behavior of the save operation when the
            destination exists already.

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
        if 'options' in options and isinstance(options.get('options'), dict
            ) and len(options) == 1:
            options = options.get('options')
        builder = self.to_spark(index_col=index_col).write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format('parquet').save(path)

    def to_orc(self, path, mode='overwrite', partition_cols=None, index_col
        =None, **options):
        """
        Write the DataFrame out as a ORC file or directory.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'},
            default 'overwrite'. Specifies the behavior of the save operation when the
            destination exists already.

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
        if 'options' in options and isinstance(options.get('options'), dict
            ) and len(options) == 1:
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='orc',
            partition_cols=partition_cols, index_col=index_col, **options)

    def to_spark_io(self, path=None, format=None, mode='overwrite',
        partition_cols=None, index_col=None, **options):
        return self.spark.to_spark_io(path, format, mode, partition_cols,
            index_col, **options)
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
        warnings.warn(
            'DataFrame.toPandas is deprecated as of DataFrame.to_pandas. Please use the API instead.'
            , FutureWarning)
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

        >>> assigned = df.assign(temp_f=df['temp_c'] * 9 / 5 + 32,
        ...                      temp_k=df['temp_c'] + 273.15,
        ...                      temp_idx=df.index)
        >>> assigned[['temp_c', 'temp_f', 'temp_k', 'temp_idx']]
                  temp_c  temp_f  temp_k  temp_idx
        Portland    17.0    62.6  290.15  Portland
        Berkeley    25.0    77.0  298.15  Berkeley

        Notes
        -----
        Assigning multiple columns within the same ``assign`` is possible
        but you cannot refer to newly created or modified columns. This
        feature is supported in pandas for Python 3.6 and later but not in
        Koalas. In Koalas, all items are computed first, and then assigned.
        """
        return self._assign(kwargs)

    def _assign(self, kwargs):
        assert isinstance(kwargs, dict)
        from databricks.koalas.indexes import MultiIndex
        from databricks.koalas.series import IndexOpsMixin
        for k, v in kwargs.items():
            is_invalid_assignee = not (isinstance(v, (IndexOpsMixin, spark.
                Column)) or callable(v) or is_scalar(v)) or isinstance(v,
                MultiIndex)
            if is_invalid_assignee:
                raise TypeError("Column assignment doesn't support type {0}"
                    .format(type(v).__name__))
            if callable(v):
                kwargs[k] = v(self)
        pairs = {(k if is_name_like_tuple(k) else (k,)): ((v.spark.column,
            v.dtype) if isinstance(v, IndexOpsMixin) and not isinstance(v,
            MultiIndex) else (v, None) if isinstance(v, spark.Column) else
            (F.lit(v), None)) for k, v in kwargs.items()}
        scols = []
        data_dtypes = []
        for label in self._internal.column_labels:
            for i in range(len(label)):
                if label[:len(label) - i] in pairs:
                    scol, dtype = pairs[label[:len(label) - i]]
                    scol = scol.alias(self._internal.spark_column_name_for(
                        label))
                    break
            else:
                scol = self._internal.spark_column_for(label)
                dtype = self._internal.dtype_for(label)
            scols.append(scol)
            data_dtypes.append(dtype)
        column_labels = self._internal.column_labels.copy()
        for label, (scol, dtype) in pairs.items():
            if label not in set(i[:len(label)] for i in self._internal.
                column_labels):
                scols.append(scol.alias(name_like_string(label)))
                column_labels.append(label)
                data_dtypes.append(dtype)
        level = self._internal.column_labels_level
        column_labels = [tuple(list(label) + [''] * (level - len(label))) for
            label in column_labels]
        internal = self._internal.with_new_columns(scols, column_labels=
            column_labels, data_dtypes=data_dtypes)
        return DataFrame(internal)

    @staticmethod
    def from_records(data, index=None, exclude=None, columns=None,
        coerce_float=False, nrows=None):
        """
        Convert structured or record ndarray to DataFrame.

        Parameters
        ----------
        data : ndarray (structured dtype), list of tuples, dict, or DataFrame
        index : string, list of fields, array-like
            Field of array to use as the index, alternately a specific set of input labels to use
        exclude : sequence, default None
            Columns or fields to exclude
        columns : sequence, default None
            Column names to use. If the passed data do not have names associated with them, this
            argument provides names for the columns. Otherwise this argument indicates the order of
            the columns in the result (any names not found in the data will become all-NA columns)
        coerce_float : boolean, default False
            Attempt to convert values of non-string, non-numeric objects (like decimal.Decimal) to
            floating point, useful for SQL result sets
        nrows : int, default None
            Number of rows to read if data is an iterator

        Returns
        -------
        df : DataFrame

        Examples
        --------
        Use dict as input

        >>> ks.DataFrame.from_records({'A': [1, 2, 3]})
           A
        0  1
        1  2
        2  3

        Use list of tuples as input

        >>> ks.DataFrame.from_records([(1, 2), (3, 4)])
           0  1
        0  1  2
        1  3  4

        Use NumPy array as input

        >>> ks.DataFrame.from_records(np.eye(3))
             0    1    2
        0  1.0  0.0  0.0
        1  0.0  1.0  0.0
        2  0.0  0.0  1.0
        """
        return DataFrame(pd.DataFrame.from_records(data, index, exclude,
            columns, coerce_float, nrows))

    def to_records(self, index=True, column_dtypes=None, index_dtypes=None):
        """
        Convert DataFrame to a NumPy record array.

        Index will be included as the first field of the record array if
        requested.

        .. note:: This method should only be used if the resulting NumPy ndarray is
            expected to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        index : bool, default True
            Include index in resulting record array, stored in 'index'
            field or using the index label, if set.
        column_dtypes : str, type, dict, default None
            If a string or type, the data type to store all columns. If
            a dictionary, a mapping of column names and indices (zero-indexed)
            to specific data types.
        index_dtypes : str, type, dict, default None
            If a string or type, the data type to store all index levels. If
            a dictionary, a mapping of index level names and indices
            (zero-indexed) to specific data types.
            This mapping is applied only if `index=True`.

        Returns
        -------
        numpy.recarray
            NumPy ndarray with the DataFrame labels as fields and each row
            of the DataFrame as entries.

        See Also
        --------
        DataFrame.from_records: Convert structured or record ndarray
            to DataFrame.
        numpy.recarray: An ndarray that allows field access using
            attributes, analogous to typed columns in a
            spreadsheet.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2], 'B': [0.5, 0.75]},
        ...                   index=['a', 'b'])
        >>> df
           A     B
        a  1  0.50
        b  2  0.75

        >>> df.to_records() # doctest: +SKIP
        rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
                  dtype=[('index', 'O'), ('A', '<i8'), ('B', '<f8')])

        The index can be excluded from the record array:

        >>> df.to_records(index=False) # doctest: +SKIP
        rec.array([(1, 0.5 ), (2, 0.75)],
                  dtype=[('A', '<i8'), ('B', '<f8')])

        Specification of dtype for columns is new in pandas 0.24.0.
        Data types can be specified for the columns:

        >>> df.to_records(column_dtypes={"A": "int32"}) # doctest: +SKIP
        rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
                  dtype=[('index', 'O'), ('A', '<i4'), ('B', '<f8')])

        Specification of dtype for index is new in pandas 0.24.0.
        Data types can also be specified for the index:

        >>> df.to_records(index_dtypes="<S2") # doctest: +SKIP
        rec.array([(b'a', 1, 0.5 ), (b'b', 2, 0.75)],
                  dtype=[('index', 'S2'), ('A', '<i8'), ('B', '<f8')])
        """
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(kdf.
            _to_internal_pandas(), self.to_records, pd.DataFrame.to_records,
            args)

    def copy(self, deep=None):
        """
        Make a copy of this object's indices and data.

        Parameters
        ----------
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Returns
        -------
        copy : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 'w': [7, 8]},
        ...                   columns=['x', 'y', 'z', 'w'])
        >>> df
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8
        >>> df_copy = df.copy()
        >>> df_copy
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8
        """
        return DataFrame(self._internal)

    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False
        ):
        """
        Remove missing values.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Determine if rows or columns which contain missing values are
            removed.

            * 0, or 'index' : Drop rows which contain missing values.
        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we have
            at least one NA or all NA.

            * 'any' : If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.

        thresh : int, optional
            Require that many non-NA values.
        subset : array-like, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.
        inplace : bool, default False
            If True, do operation inplace and return None.

        Returns
        -------
        DataFrame
            DataFrame with NA entries dropped from it.

        See Also
        --------
        DataFrame.drop : Drop specified labels from columns.
        DataFrame.isnull: Indicate missing values.
        DataFrame.notnull : Indicate existing (non-missing) values.

        Examples
        --------
        >>> df = ks.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
        ...                    "toy": [None, 'Batmobile', 'Bullwhip'],
        ...                    "born": [None, "1940-04-25", None]},
        ...                   columns=['name', 'toy', 'born'])
        >>> df
               name        toy        born
        0    Alfred       None        None
        1    Batman  Batmobile  1940-04-25
        2  Catwoman   Bullwhip        None

        Drop the rows where at least one element is missing.

        >>> df.dropna()
             name        toy        born
        1  Batman  Batmobile  1940-04-25

        Drop the columns where at least one element is missing.

        >>> df.dropna(axis='columns')
               name
        0    Alfred
        1    Batman
        2  Catwoman

        Drop the rows where all elements are missing.

        >>> df.dropna(how='all')
               name        toy        born
        0    Alfred       None        None
        1    Batman  Batmobile  1940-04-25
        2  Catwoman   Bullwhip        None

        Keep only the rows with at least 2 non-NA values.

        >>> df.dropna(thresh=2)
               name        toy        born
        1    Batman  Batmobile  1940-04-25
        2  Catwoman   Bullwhip        None

        Define in which columns to look for missing values.

        >>> df.dropna(subset=['name', 'born'])
             name        toy        born
        1  Batman  Batmobile  1940-04-25

        Keep the DataFrame with valid entries in the same variable.

        >>> df.dropna(inplace=True)
        >>> df
             name        toy        born
        1  Batman  Batmobile  1940-04-25
        """
        axis = validate_axis(axis)
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if thresh is None:
            if how is None:
                raise TypeError('must specify how or thresh')
            elif how not in ('any', 'all'):
                raise ValueError('invalid how option: {h}'.format(h=how))
        if subset is not None:
            if isinstance(subset, str):
                labels = [(subset,)]
            elif isinstance(subset, tuple):
                labels = [subset]
            else:
                labels = [(sub if isinstance(sub, tuple) else (sub,)) for
                    sub in subset]
        else:
            labels = None
        if axis == 0:
            if labels is not None:
                invalids = [label for label in labels if label not in self.
                    _internal.column_labels]
                if len(invalids) > 0:
                    raise KeyError(invalids)
            else:
                labels = self._internal.column_labels
            cnt = reduce(lambda x, y: x + y, [F.when(self._kser_for(label).
                notna().spark.column, 1).otherwise(0) for label in labels],
                F.lit(0))
            if thresh is not None:
                pred = cnt >= F.lit(int(thresh))
            elif how == 'any':
                pred = cnt == F.lit(len(labels))
            elif how == 'all':
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
                    raise ValueError(
                        'The length of each subset must be the same as the index size.'
                        )
                cond = reduce(lambda x, y: x | y, [reduce(lambda x, y: x &
                    y, [(scol == F.lit(l)) for l, scol in zip(lbl, internal
                    .index_spark_columns)]) for lbl in labels])
                internal = internal.with_filter(cond)
            null_counts = []
            for label in internal.column_labels:
                scol = internal.spark_column_for(label)
                if isinstance(internal.spark_type_for(label), (FloatType,
                    DoubleType)):
                    cond = scol.isNull() | F.isnan(scol)
                else:
                    cond = scol.isNull()
                null_counts.append(F.sum(F.when(~cond, 1).otherwise(0)).
                    alias(name_like_string(label)))
            counts = internal.spark_frame.select(null_counts + [F.count('*')]
                ).head()
            if thresh is not None:
                column_labels = [label for label, cnt in zip(internal.
                    column_labels, counts) if (cnt or 0) >= int(thresh)]
            elif how == 'any':
                column_labels = [label for label, cnt in zip(internal.
                    column_labels, counts) if (cnt or 0) == counts[-1]]
            elif how == 'all':
                column_labels = [label for label, cnt in zip(internal.
                    column_labels, counts) if (cnt or 0) > 0]
            kdf = self[column_labels]
            if inplace:
                self._update_internal_frame(kdf._internal)
                return None
            else:
                return kdf

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
        DataFrame
            DataFrame with NA entries filled.

        Examples
        --------
        >>> df = ks.DataFrame({
        ...     'A': [None, 3, None, None],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> df
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  NaN  NaN  NaN  5
        3  NaN  3.0  1.0  4

        Replace all NaN elements with 0s.

        >>> df.fillna(0)
             A    B    C  D
        0  0.0  2.0  0.0  0
        1  3.0  4.0  0.0  1
        2  0.0  0.0  0.0  5
        3  0.0  3.0  1.0  4

        We can also propagate non-null values forward or backward.

        >>> df.fillna(method='ffill')
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  3.0  4.0  NaN  5
        3  3.0  3.0  1.0  4

        Replace all NaN elements in column 'A', 'B', 'C', and 'D', with 0, 1,
        2, and 3 respectively.

        >>> values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        >>> df.fillna(value=values)
             A    B    C  D
        0  0.0  2.0  2.0  0
        1  3.0  4.0  2.0  1
        2  0.0  1.0  2.0  5
        3  0.0  3.0  1.0  4
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                "fillna currently only works for axis=0 or axis='index'")
        if value is not None:
            if not isinstance(value, (float, int, str, bool, dict, pd.Series)):
                raise TypeError('Unsupported type %s' % type(value).__name__)
            if limit is not None:
                raise ValueError('limit parameter for value is not support now'
                    )
            if isinstance(value, pd.Series):
                value = value.to_dict()
            if isinstance(value, dict):
                for v in value.values():
                    if not isinstance(v, (float, int, str, bool)):
                        raise TypeError('Unsupported type %s' % type(v).
                            __name__)
                value = {(k if is_name_like_tuple(k) else (k,)): v for k, v in
                    value.items()}

                def op(kser):
                    label = kser._column_label
                    for k, v in value.items():
                        if k == label[:len(k)]:
                            return kser._fillna(value=value[k], method=
                                method, axis=axis, limit=limit)
                    else:
                        return kser
            else:
                op = lambda kser: kser._fillna(value=value, method=method,
                    axis=axis, limit=limit)
        elif method is not None:
            op = lambda kser: kser._fillna(value=value, method=method, axis
                =axis, limit=limit)
        else:
            raise ValueError(
                "Must specify a fillna 'value' or 'method' parameter.")
        kdf = self._apply_series_op(op, should_resolve=method is not None)
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if inplace:
            self._update_internal_frame(kdf._internal, requires_same_anchor
                =False)
            return None
        else:
            return kdf

    def replace(self, to_replace=None, value=None, inplace=False, limit=
        None, regex=False, method='pad'):
        """
        Returns a new DataFrame replacing a value with another value.

        Parameters
        ----------
        to_replace : int, float, string, list, tuple or dict
            Value to be replaced.
        value : int, float, string, list or tuple
            Value to use to replace holes. The replacement value must be an int, float,
            or string.
            If value is a list or tuple, value should be of the same length with to_replace.
        inplace : boolean, default False
            Fill in place (do not create a new object)

        Returns
        -------
        DataFrame
            Object after replacement.

        Examples
        --------
        >>> df = ks.DataFrame({"name": ['Ironman', 'Captain America', 'Thor', 'Hulk'],
        ...                    "weapon": ['Mark-45', 'Shield', 'Mjolnir', 'Smash']},
        ...                   columns=['name', 'weapon'])
        >>> df
                      name   weapon
        0          Ironman  Mark-45
        1  Captain America   Shield
        2             Thor  Mjolnir
        3             Hulk    Smash

        Scalar `to_replace` and `value`

        >>> df.replace('Ironman', 'War-Machine')
                      name   weapon
        0      War-Machine  Mark-45
        1  Captain America   Shield
        2             Thor  Mjolnir
        3             Hulk    Smash

        List like `to_replace` and `value`

        >>> df.replace(['Ironman', 'Captain America'], ['Rescue', 'Hawkeye'], inplace=True)
        >>> df
              name   weapon
        0   Rescue  Mark-45
        1  Hawkeye   Shield
        2     Thor  Mjolnir
        3     Hulk    Smash

        Dicts can be used to specify different replacement values for different existing values
        To use a dict in this way the value parameter should be None

        >>> df.replace({'Mjolnir': 'Stormbuster'})
              name       weapon
        0   Rescue      Mark-45
        1  Hawkeye       Shield
        2     Thor  Stormbuster
        3     Hulk        Smash

        Dict can specify that different values should be replaced in different columns
        The value parameter should not be None in this case

        >>> df.replace({'weapon': 'Mjolnir'}, 'Stormbuster')
              name       weapon
        0   Rescue      Mark-45
        1  Hawkeye       Shield
        2     Thor  Stormbuster
        3     Hulk        Smash

        Nested dictionaries
        The value parameter should be None to use a nested dict in this way

        >>> df.replace({'weapon': {'Mjolnir': 'Stormbuster'}})
              name       weapon
        0   Rescue      Mark-45
        1  Hawkeye       Shield
        2     Thor  Stormbuster
        3     Hulk        Smash
        """
        if method != 'pad':
            raise NotImplementedError(
                "replace currently works only for method='pad")
        if limit is not None:
            raise NotImplementedError(
                'replace currently works only when limit=None')
        if regex is not False:
            raise NotImplementedError(
                "replace currently doesn't supports regex")
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if value is not None and not isinstance(value, (int, float, str,
            list, tuple, dict)):
            raise TypeError('Unsupported type {}'.format(type(value).__name__))
        if to_replace is not None and not isinstance(to_replace, (int,
            float, str, list, tuple, dict)):
            raise TypeError('Unsupported type {}'.format(type(to_replace).
                __name__))
        if isinstance(value, (list, tuple)) and isinstance(to_replace, (
            list, tuple)):
            if len(value) != len(to_replace):
                raise ValueError('Length of to_replace and value must be same')
        if isinstance(to_replace, dict) and (value is not None or all(
            isinstance(i, dict) for i in to_replace.values())):

            def op(kser):
                if kser.name in to_replace:
                    return kser.replace(to_replace=to_replace[kser.name],
                        value=value, regex=regex)
                else:
                    return kser
        else:
            op = lambda kser: kser.replace(to_replace=to_replace, value=
                value, regex=regex)
        kdf = self._apply_series_op(op)
        if inplace:
            self._update_internal_frame(kdf._internal)
            return None
        else:
            return kdf

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
        DataFrame
            DataFrame with the values outside the clip boundaries replaced.

        Examples
        --------
        >>> ks.DataFrame({'A': [0, 2, 4]}).clip(1, 3)
           A
        0  1
        1  2
        2  3

        Notes
        -----
        One difference between this implementation and pandas is that running
        pd.DataFrame({'A': ['a', 'b']}).clip(0, 1) will crash with "TypeError: '<=' not supported
        between instances of 'str' and 'int'" while ks.DataFrame({'A': ['a', 'b']}).clip(0, 1)
        will output the original DataFrame, simply ignoring the incompatible types.
        """
        if is_list_like(lower) or is_list_like(upper):
            raise ValueError(
                "List-like value are not supported for 'lower' and 'upper' at the "
                 + 'moment')
        if lower is None and upper is None:
            return self
        return self._apply_series_op(lambda kser: kser.clip(lower=lower,
            upper=upper))

    def head(self, n=5):
        """
        Return the first `n` rows.

        This function returns the first `n` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        obj_head : same type as caller
            The first `n` rows of the caller object.

        Examples
        --------
        >>> df = ks.DataFrame({'animal':['alligator', 'bee', 'falcon', 'lion',
        ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the first 5 lines

        >>> df.head()
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey

        Viewing the first `n` lines (three in this case)

        >>> df.head(3)
              animal
        0  alligator
        1        bee
        2     falcon
        """
        if n < 0:
            n = len(self) + n
        if n <= 0:
            return DataFrame(self._internal.with_filter(F.lit(False)))
        else:
            sdf = self._internal.resolved_copy.spark_frame
            if get_option('compute.ordered_head'):
                sdf = sdf.orderBy(NATURAL_ORDER_COLUMN_NAME)
            return DataFrame(self._internal.with_new_sdf(sdf.limit(n)))

    def last(self, offset):
        """
        Select final periods of time series data based on a date offset.

        When having a DataFrame with dates as index, this function can
        select the last few rows based on a date offset.

        Parameters
        ----------
        offset : str or DateOffset
            The offset length of the data that will be selected. For instance,
            '3D' will display all the rows having their index within the last 3 days.

        Returns
        -------
        DataFrame
            A subset of the caller.

        Raises
        ------
        TypeError
            If the index is not a :class:`DatetimeIndex`

        Examples
        --------

        >>> index = pd.date_range('2018-04-09', periods=4, freq='2D')
        >>> kdf = ks.DataFrame({'A': [1, 2, 3, 4]}, index=index)
        >>> kdf
                    A
        2018-04-09  1
        2018-04-11  2
        2018-04-13  3
        2018-04-15  4

        Get the rows for the last 3 days:

        >>> kdf.last('3D')
                    A
        2018-04-13  3
        2018-04-15  4

        Notice the data for 3 last calendar days were returned, not the last
        3 observed days in the dataset, and therefore data for 2018-04-11 was
        not returned.
        """
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError("'last' only supports a DatetimeIndex")
        offset = to_offset(offset)
        from_date = self.index.max() - offset
        return cast(DataFrame, self.loc[from_date:])

    def first(self, offset):
        """
        Select first periods of time series data based on a date offset.

        When having a DataFrame with dates as index, this function can
        select the first few rows based on a date offset.

        Parameters
        ----------
        offset : str or DateOffset
            The offset length of the data that will be selected. For instance,
            '3D' will display all the rows having their index within the first 3 days.

        Returns
        -------
        DataFrame
            A subset of the caller.

        Raises
        ------
        TypeError
            If the index is not a :class:`DatetimeIndex`

        Examples
        --------

        >>> index = pd.date_range('2018-04-09', periods=4, freq='2D')
        >>> kdf = ks.DataFrame({'A': [1, 2, 3, 4]}, index=index)
        >>> kdf
                    A
        2018-04-09  1
        2018-04-11  2
        2018-04-13  3
        2018-04-15  4

        Get the rows for the last 3 days:

        >>> kdf.first('3D')
                    A
        2018-04-09  1
        2018-04-11  2

        Notice the data for 3 first calendar days were returned, not the first
        3 observed days in the dataset, and therefore data for 2018-04-13 was
        not returned.
        """
        if not isinstance(self.index, ks.DatetimeIndex):
            raise TypeError("'first' only supports a DatetimeIndex")
        offset = to_offset(offset)
        to_date = self.index.min() + offset
        return cast(DataFrame, self.loc[:to_date])

    def pivot_table(self, values=None, index=None, columns=None, aggfunc=
        'mean', fill_value=None):
        """
        Create a spreadsheet-style pivot table as a DataFrame. The levels in
        the pivot table will be stored in MultiIndex objects (hierarchical
        indexes) on the index and columns of the result DataFrame.

        Parameters
        ----------
        values : column to aggregate.
            They should be either a list less than three or a string.
        index : column (string) or list of columns
            If an array is passed, it must be the same length as the data.
            The list should contain string.
        columns : column
            Columns used in the pivot operation. Only one column is supported and
            it should be a string.
        aggfunc : function (string), dict, default mean
            If dict is passed, the key is column to aggregate and value
            is function or list of functions.
        fill_value : scalar, default None
            Value to replace missing values with.

        Returns
        -------
        table : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
        ...                          "bar", "bar", "bar", "bar"],
        ...                    "B": ["one", "one", "one", "two", "two",
        ...                          "one", "one", "two", "two"],
        ...                    "C": ["small", "large", "large", "small",
        ...                          "small", "large", "small", "small",
        ...                          "large"],
        ...                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
        ...                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]},
        ...                   columns=['A', 'B', 'C', 'D', 'E'])
        >>> df
             A    B      C  D  E
        0  foo  one  small  1  2
        1  foo  one  large  2  4
        2  foo  one  large  2  5
        3  foo  two  small  3  5
        4  foo  two  small  3  6
        5  bar  one  large  4  6
        6  bar  one  small  5  8
        7  bar  two  small  6  9
        8  bar  two  large  7  9

        This first example aggregates values by taking the sum.

        >>> table = df.pivot_table(values='D', index=['A', 'B'],
        ...                        columns='C', aggfunc='sum')
        >>> table.sort_index()  # doctest: +NORMALIZE_WHITESPACE
        C        large  small
        A   B
        bar one    4.0      5
            two    7.0      6
        foo one    4.0      1
            two    NaN      6

        We can also fill missing values using the `fill_value` parameter.

        >>> table = df.pivot_table(values='D', index=['A', 'B'],
        ...                        columns='C', aggfunc='sum', fill_value=0)
        >>> table.sort_index()  # doctest: +NORMALIZE_WHITESPACE
        C        large  small
        A   B
        bar one      4      5
            two      7      6
        foo one      4      1
            two      0      6

        We can also calculate multiple types of aggregations for any given
        value column.

        >>> table = df.pivot_table(values=['D'], index =['C'],
        ...                        columns="A", aggfunc={'D': 'mean'})
        >>> table.sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 D
        A      bar       foo
        C
        large  5.5  2.000000
        small  5.5  2.333333

        The next example aggregates on multiple values.

        >>> table = df.pivot_table(index=['C'], columns="A", values=['D', 'E'],
        ...                         aggfunc={'D': 'mean', 'E': 'sum'})
        >>> table.sort_index() # doctest: +NORMALIZE_WHITESPACE
                 D             E
        A      bar       foo bar foo
        C
        large  5.5  2.000000  15   9
        small  5.5  2.333333  17  13
        """
        if not is_name_like_value(columns):
            raise ValueError('columns should be one column name.')
        if not is_name_like_value(values) and not (isinstance(values, list) and
            all(is_name_like_value(v) for v in values)):
            raise ValueError('values should be one column or list of columns.')
        if not isinstance(aggfunc, str) and (not isinstance(aggfunc, dict) or
            not all(is_name_like_value(key) and isinstance(value, str) for 
            key, value in aggfunc.items())):
            raise ValueError(
                'aggfunc must be a dict mapping from column name to aggregate functions (string).'
                )
        if isinstance(aggfunc, dict) and index is None:
            raise NotImplementedError(
                "pivot_table doesn't support aggfunc as dict and without index."
                )
        if isinstance(values, list) and index is None:
            raise NotImplementedError("values can't be a list without index.")
        if columns not in self.columns:
            raise ValueError('Wrong columns {}.'.format(name_like_string(
                columns)))
        if not is_name_like_tuple(columns):
            columns = columns,
        if isinstance(values, list):
            values = [(col if is_name_like_tuple(col) else (col,)) for col in
                values]
            if not all(isinstance(self._internal.spark_type_for(col),
                NumericType) for col in values):
                raise TypeError('values should be a numeric type.')
        else:
            values = values if is_name_like_tuple(values) else (values,)
            if not isinstance(self._internal.spark_type_for(values),
                NumericType):
                raise TypeError('values should be a numeric type.')
        if isinstance(aggfunc, str):
            if isinstance(values, list):
                agg_cols = [F.expr('{1}(`{0}`) as `{0}`'.format(self.
                    _internal.spark_column_name_for(value), aggfunc)) for
                    value in values]
            else:
                agg_cols = [F.expr('{1}(`{0}`) as `{0}`'.format(self.
                    _internal.spark_column_name_for(values), aggfunc))]
        elif isinstance(aggfunc, dict):
            aggfunc = {(key if is_name_like_tuple(key) else (key,)): value for
                key, value in aggfunc.items()}
            agg_cols = [F.expr('{1}(`{0}`) as `{0}`'.format(self._internal.
                spark_column_name_for(key), value)) for key, value in
                aggfunc.items()]
            agg_columns = [key for key, _ in aggfunc.items()]
            if set(agg_columns) != set(values):
                raise ValueError(
                    'Columns in aggfunc must be the same as values.')
        sdf = self._internal.resolved_copy.spark_frame
        if index is None:
            sdf = sdf.groupBy().pivot(pivot_col=self._internal.
                spark_column_name_for(columns)).agg(*agg_cols)
        elif isinstance(index, list):
            index = [(label if is_name_like_tuple(label) else (label,)) for
                label in index]
            sdf = sdf.groupBy([self._internal.spark_column_name_for(label) for
                label in index]).pivot(pivot_col=self._internal.
                spark_column_name_for(columns)).agg(*agg_cols)
        else:
            raise ValueError('index should be a None or a list of columns.')
        if fill_value is not None and isinstance(fill_value, (int, float)):
            sdf = sdf.fillna(fill_value)
        if index is not None:
            index_columns = [self._internal.spark_column_name_for(label) for
                label in index]
            index_dtypes = [self._internal.dtype_for(label) for label in index]
            if isinstance(values, list):
                data_columns = [column for column in sdf.columns if column
                     not in index_columns]
                if len(values) > 1:
                    data_columns.sort(key=lambda x: x.split('_', 1)[1])
                    sdf = sdf.select(index_columns + data_columns)
                    column_name_to_index = dict(zip(self._internal.
                        data_spark_column_names, self._internal.column_labels))
                    column_labels = [tuple(list(column_name_to_index[name.
                        split('_')[1]]) + [name.split('_')[0]]) for name in
                        data_columns]
                    column_label_names = [None] * column_labels_level(values
                        ) + [columns]
                    internal = InternalFrame(spark_frame=sdf,
                        index_spark_columns=[scol_for(sdf, col) for col in
                        index_columns], index_names=index, index_dtypes=
                        index_dtypes, column_labels=column_labels,
                        data_spark_columns=[scol_for(sdf, col) for col in
                        data_columns], column_label_names=column_label_names)
                    kdf = DataFrame(internal)
                else:
                    column_labels = [tuple(list(values[0]) + [column]) for
                        column in data_columns]
                    column_label_names = [None] * len(values[0]) + [columns]
                    internal = InternalFrame(spark_frame=sdf,
                        index_spark_columns=[scol_for(sdf, col) for col in
                        index_columns], index_names=index, index_dtypes=
                        index_dtypes, column_labels=column_labels,
                        data_spark_columns=[scol_for(sdf, col) for col in
                        data_columns], column_label_names=column_label_names)
                    kdf = DataFrame(internal)
            else:
                internal = InternalFrame(spark_frame=sdf,
                    index_spark_columns=[scol_for(sdf, col) for col in
                    index_columns], index_names=index, index_dtypes=
                    index_dtypes, column_label_names=[columns])
                kdf = DataFrame(internal)
        else:
            if isinstance(values, list):
                index_values = values[-1]
            else:
                index_values = values
            index_map = OrderedDict()
            for i, index_value in enumerate(index_values):
                colname = SPARK_INDEX_NAME_FORMAT(i)
                sdf = sdf.withColumn(colname, F.lit(index_value))
                index_map[colname] = None
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
                scol_for(sdf, col) for col in index_map.keys()],
                index_names=list(index_map.values()), column_label_names=[
                columns])
            kdf = DataFrame(internal)
        kdf_columns = kdf.columns
        if isinstance(kdf_columns, pd.MultiIndex):
            kdf.columns = kdf_columns.set_levels(kdf_columns.levels[-1].
                astype(spark_type_to_pandas_dtype(self._kser_for(columns).
                spark.data_type)), level=-1)
        else:
            kdf.columns = kdf_columns.astype(spark_type_to_pandas_dtype(
                self._kser_for(columns).spark.data_type))
        return kdf

    def pivot(self, index=None, columns=None, values=None):
        """
        Return reshaped DataFrame organized by given index / column values.

        Reshape data (produce a "pivot" table) based on column values. Uses
        unique values from specified `index` / `columns` to form axes of the
        resulting DataFrame. This function does not support data
        aggregation.

        Parameters
        ----------
        index : string, optional
            Column to use to make new frame's index. If None, uses
            existing index.
        columns : string
            Column to use to make new frame's columns.
        values : string, object or a list of the previous
            Column(s) to use for populating new frame's values.

        Returns
        -------
        DataFrame
            Returns reshaped DataFrame.

        See Also
        --------
        DataFrame.pivot_table : Generalization of pivot that can handle
            duplicate values for one index/column pair.

        Examples
        --------
        >>> df = ks.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
        ...                            'two'],
        ...                    'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
        ...                    'baz': [1, 2, 3, 4, 5, 6],
        ...                    'zoo': ['x', 'y', 'z', 'q', 'w', 't']},
        ...                   columns=['foo', 'bar', 'baz', 'zoo'])
        >>> df
           foo bar  baz zoo
        0  one   A    1   x
        1  one   B    2   y
        2  one   C    3   z
        3  two   A    4   q
        4  two   B    5   w
        5  two   C    6   t

        >>> df.pivot(index='foo', columns='bar', values='baz').sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
        bar  A  B  C
        foo
        one  1  2  3
        two  4  5  6

        >>> df.pivot(columns='bar', values='baz').sort_index()  # doctest: +NORMALIZE_WHITESPACE
        bar  A    B    C
        0  1.0  NaN  NaN
        1  NaN  2.0  NaN
        2  NaN  NaN  3.0
        3  4.0  NaN  NaN
        4  NaN  5.0  NaN
        5  NaN  NaN  6.0

        Notice that, unlike pandas raises an ValueError when duplicated values are found,
        Koalas' pivot still works with its first value it meets during operation because pivot
        is an expensive operation and it is preferred to permissively execute over failing fast
        when processing large data.

        >>> df = ks.DataFrame({"foo": ['one', 'one', 'two', 'two'],
        ...                    "bar": ['A', 'A', 'B', 'C'],
        ...                    "baz": [1, 2, 3, 4]}, columns=['foo', 'bar', 'baz'])
        >>> df
           foo bar  baz
        0  one   A    1
        1  one   A    2
        2  two   B    3
        3  two   C    4

        >>> df.pivot(index='foo', columns='bar', values='baz').sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
        bar    A    B    C
        foo
        one  1.0  NaN  NaN
        two  NaN  3.0  4.0

        It also support multi-index and multi-index column.
        >>> df.columns = pd.MultiIndex.from_tuples([('a', 'foo'), ('a', 'bar'), ('b', 'baz')])

        >>> df = df.set_index(('a', 'bar'), append=True)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                      a   b
                    foo baz
          (a, bar)
        0 A         one   1
        1 A         one   2
        2 B         two   3
        3 C         two   4

        >>> df.pivot(columns=('a', 'foo'), values=('b', 'baz')).sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
        ('a', 'foo')  one  two
          (a, bar)
        0 A           1.0  NaN
        1 A           2.0  NaN
        2 B           NaN  3.0
        3 C           NaN  4.0

        """
        if columns is None:
            raise ValueError('columns should be set.')
        if values is None:
            raise ValueError('values should be set.')
        should_use_existing_index = index is not None
        if should_use_existing_index:
            df = self
            index = [index]
        else:
            with option_context('compute.default_index_type', 'distributed'):
                df = self.reset_index()
            index = df._internal.column_labels[:self._internal.index_level]
        df = df.pivot_table(index=index, columns=columns, values=values,
            aggfunc='first')
        if should_use_existing_index:
            return df
        else:
            internal = df._internal.copy(index_names=self._internal.index_names
                )
            return DataFrame(internal)

    @property
    def columns(self):
        """The column labels of the DataFrame."""
        names = [(name if name is None or len(name) > 1 else name[0]) for
            name in self._internal.column_label_names]
        if self._internal.column_labels_level > 1:
            columns = pd.MultiIndex.from_tuples(self._internal.
                column_labels, names=names)
        else:
            columns = pd.Index([label[0] for label in self._internal.
                column_labels], name=names[0])
        return columns

    @columns.setter
    def columns(self, columns):
        if isinstance(columns, pd.MultiIndex):
            column_labels = columns.tolist()
        else:
            column_labels = [(col if is_name_like_tuple(col, allow_none=
                False) else (col,)) for col in columns]
        if len(self._internal.column_labels) != len(column_labels):
            raise ValueError(
                'Length mismatch: Expected axis has {} elements, new values have {} elements'
                .format(len(self._internal.column_labels), len(column_labels)))
        if isinstance(columns, pd.Index):
            column_label_names = [(name if is_name_like_tuple(name) else (
                name,)) for name in columns.names]
        else:
            column_label_names = None
        ksers = [self._kser_for(label).rename(name) for label, name in zip(
            self._internal.column_labels, column_labels)]
        self._update_internal_frame(self._internal.with_new_columns(ksers,
            column_label_names=column_label_names))

    @property
    def dtypes(self):
        """Return the dtypes in the DataFrame.

        This returns a Series with the data type of each column. The result's index is the original
        DataFrame's columns. Columns with mixed types are stored with the object dtype.

        Returns
        -------
        pd.Series
            The data type of each column.

        Examples
        --------
        >>> df = ks.DataFrame({'a': list('abc'),
        ...                    'b': list(range(1, 4)),
        ...                    'c': np.arange(3, 6).astype('i1'),
        ...                    'd': np.arange(4.0, 7.0, dtype='float64'),
        ...                    'e': [True, False, True],
        ...                    'f': pd.date_range('20130101', periods=3)},
        ...                   columns=['a', 'b', 'c', 'd', 'e', 'f'])
        >>> df.dtypes
        a            object
        b             int64
        c              int8
        d           float64
        e              bool
        f    datetime64[ns]
        dtype: object
        """
        return pd.Series([self._kser_for(label).dtype for label in self.
            _internal.column_labels], index=pd.Index([(label if len(label) >
            1 else label[0]) for label in self._internal.column_labels]))

    def spark_schema(self, index_col=None):
        warnings.warn(
            'DataFrame.spark_schema is deprecated as of DataFrame.spark.schema. Please use the API instead.'
            , FutureWarning)
        return self.spark.schema(index_col)
    spark_schema.__doc__ = SparkFrameMethods.schema.__doc__

    def print_schema(self, index_col=None):
        warnings.warn(
            'DataFrame.print_schema is deprecated as of DataFrame.spark.print_schema. Please use the API instead.'
            , FutureWarning)
        return self.spark.print_schema(index_col)
    print_schema.__doc__ = SparkFrameMethods.print_schema.__doc__

    def select_dtypes(self, include=None, exclude=None):
        """
        Return a subset of the DataFrame's columns based on the column dtypes.

        Parameters
        ----------
        include, exclude : scalar or list-like
            A selection of dtypes or strings to be included/excluded. At least
            one of these parameters must be supplied. It also takes Spark SQL
            DDL type strings, for instance, 'string' and 'date'.

        Returns
        -------
        DataFrame
            The subset of the frame including the dtypes in ``include`` and
            excluding the dtypes in ``exclude``.

        Raises
        ------
        ValueError
            * If both of ``include`` and ``exclude`` are empty

                >>> df = ks.DataFrame({'a': [1, 2] * 3,
                ...                    'b': [True, False] * 3,
                ...                    'c': [1.0, 2.0] * 3})
                >>> df.select_dtypes()
                Traceback (most recent call last):
                ...
                ValueError: at least one of include or exclude must be nonempty

            * If ``include`` and ``exclude`` have overlapping elements

                >>> df = ks.DataFrame({'a': [1, 2] * 3,
                ...                    'b': [True, False] * 3,
                ...                    'c': [1.0, 2.0] * 3})
                >>> df.select_dtypes(include='a', exclude='a')
                Traceback (most recent call last):
                ...
                ValueError: include and exclude overlap on {'a'}

        Notes
        -----
        * To select datetimes, use ``np.datetime64``, ``'datetime'`` or
          ``'datetime64'``

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2] * 3,
        ...                    'b': [True, False] * 3,
        ...                    'c': [1.0, 2.0] * 3,
        ...                    'd': ['a', 'b'] * 3}, columns=['a', 'b', 'c', 'd'])
        >>> df
           a      b    c  d
        0  1   True  1.0  a
        1  2  False  2.0  b
        2  1   True  1.0  a
        3  2  False  2.0  b
        4  1   True  1.0  a
        5  2  False  2.0  b

        >>> df.select_dtypes(include='bool')
               b
        0   True
        1  False
        2   True
        3  False
        4   True
        5  False

        >>> df.select_dtypes(include=['float64'], exclude=['int'])
             c
        0  1.0
        1  2.0
        2  1.0
        3  2.0
        4  1.0
        5  2.0

        >>> df.select_dtypes(exclude=['int'])
               b    c  d
        0   True  1.0  a
        1  False  2.0  b
        2   True  1.0  a
        3  False  2.0  b
        4   True  1.0  a
        5  False  2.0  b

        Spark SQL DDL type strings can be used as well.

        >>> df.select_dtypes(exclude=['string'])
           a      b    c
        0  1   True  1.0
        1  2  False  2.0
        2  1   True  1.0
        3  2  False  2.0
        4  1   True  1.0
        5  2  False  2.0
        """
        from pyspark.sql.types import _parse_datatype_string
        if not is_list_like(include):
            include = (include,) if include is not None else ()
        if not is_list_like(exclude):
            exclude = (exclude,) if exclude is not None else ()
        if not any((include, exclude)):
            raise ValueError(
                'at least one of include or exclude must be nonempty')
        if set(include).intersection(set(exclude)):
            raise ValueError('include and exclude overlap on {inc_ex}'.
                format(inc_ex=set(include).intersection(set(exclude))))
        include_spark_type = []
        for inc in include:
            try:
                include_spark_type.append(_parse_datatype_string(inc))
            except:
                pass
        exclude_spark_type = []
        for exc in exclude:
            try:
                exclude_spark_type.append(_parse_datatype_string(exc))
            except:
                pass
        include_numpy_type = []
        for inc in include:
            try:
                include_numpy_type.append(infer_dtype_from_object(inc))
            except:
                pass
        exclude_numpy_type = []
        for exc in exclude:
            try:
                exclude_numpy_type.append(infer_dtype_from_object(exc))
            except:
                pass
        column_labels = []
        for label in self._internal.column_labels:
            if len(include) > 0:
                should_include = infer_dtype_from_object(self._kser_for(
                    label).dtype.name
                    ) in include_numpy_type or self._internal.spark_type_for(
                    label) in include_spark_type
            else:
                should_include = not (infer_dtype_from_object(self.
                    _kser_for(label).dtype.name) in exclude_numpy_type or 
                    self._internal.spark_type_for(label) in exclude_spark_type)
            if should_include:
                column_labels.append(label)
        return DataFrame(self._internal.with_new_columns([self._kser_for(
            label) for label in column_labels]))

    def droplevel(self, level, axis=0):
        """
        Return DataFrame with requested index / column level(s) removed.

        Parameters
        ----------
        level: int, str, or list-like
            If a string is given, must be the name of a level If list-like, elements must
            be names or positional indexes of levels.

        axis: {0 or ‘index’, 1 or ‘columns’}, default 0

        Returns
        -------
        DataFrame with requested index / column level(s) removed.

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[3, 4], [7, 8], [11, 12]],
        ...     index=pd.MultiIndex.from_tuples([(1, 2), (5, 6), (9, 10)], names=["a", "b"]),
        ... )

        >>> df.columns = pd.MultiIndex.from_tuples([
        ...   ('c', 'e'), ('d', 'f')
        ... ], names=['level_1', 'level_2'])

        >>> df  # doctest: +NORMALIZE_WHITESPACE
        level_1   c   d
        level_2   e   f
        a b
        1 2      3   4
        5 6      7   8
        9 10    11  12

        >>> df.droplevel('a')  # doctest: +NORMALIZE_WHITESPACE
        level_1   c   d
        level_2   e   f
        b
        2        3   4
        6        7   8
        10      11  12

        >>> df.droplevel('level_2', axis=1)  # doctest: +NORMALIZE_WHITESPACE
        level_1   c   d
        a b
        1 2      3   4
        5 6      7   8
        9 10    11  12
        """
        axis = validate_axis(axis)
        if axis == 0:
            if not isinstance(level, (tuple, list)):
                level = [level]
            index_names = self.index.names
            nlevels = self._internal.index_level
            int_level = set()
            for n in level:
                if isinstance(n, int):
                    if n < 0:
                        n = n + nlevels
                        if n < 0:
                            raise IndexError(
                                'Too many levels: Index has only {} levels, {} is not a valid level number'
                                .format(nlevels, n - nlevels))
                    if n >= nlevels:
                        raise IndexError(
                            'Too many levels: Index has only {} levels, not {}'
                            .format(nlevels, n + 1))
                else:
                    if n not in index_names:
                        raise KeyError('Level {} not found'.format(n))
                    n = index_names.index(n)
                int_level.add(n)
            if len(level) >= nlevels:
                raise ValueError(
                    'Cannot remove {} levels from an index with {} levels: at least one level must be left.'
                    .format(len(level), nlevels))
            index_spark_columns, index_names, index_dtypes = zip(*[item for
                i, item in enumerate(zip(self._internal.index_spark_columns,
                self._internal.index_names, self._internal.index_dtypes)) if
                i not in int_level])
            internal = self._internal.copy(index_spark_columns=list(
                index_spark_columns), index_names=list(index_names),
                index_dtypes=list(index_dtypes))
            return DataFrame(internal)
        else:
            kdf = self.copy()
            kdf.columns = kdf.columns.droplevel(level)
            return kdf

    def drop(self, labels=None, axis=1, columns=None):
        """
        Drop specified labels from columns.

        Remove columns by specifying label names and axis=1 or columns.
        When specifying both labels and columns, only labels will be dropped.
        Removing rows is yet to be implemented.

        Parameters
        ----------
        labels : single label or list-like
            Column labels to drop.
        axis : {1 or 'columns'}, default 1
            .. dropna currently only works for axis=1 'columns'
               axis=0 is yet to be implemented.
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).

        Returns
        -------
        dropped : DataFrame

        See Also
        --------
        Series.dropna

        Examples
        --------
        >>> df = ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 'w': [7, 8]},
        ...                   columns=['x', 'y', 'z', 'w'])
        >>> df
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8

        >>> df.drop('x', axis=1)
           y  z  w
        0  3  5  7
        1  4  6  8

        >>> df.drop(['y', 'z'], axis=1)
           x  w
        0  1  7
        1  2  8

        >>> df.drop(columns=['y', 'z'])
           x  w
        0  1  7
        1  2  8

        Also support for MultiIndex

        >>> df = ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 'w': [7, 8]},
        ...                   columns=['x', 'y', 'z', 'w'])
        >>> columns = [('a', 'x'), ('a', 'y'), ('b', 'z'), ('b', 'w')]
        >>> df.columns = pd.MultiIndex.from_tuples(columns)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
           a     b
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8
        >>> df.drop('a')  # doctest: +NORMALIZE_WHITESPACE
           b
           z  w
        0  5  7
        1  6  8

        Notes
        -----
        Currently only axis = 1 is supported in this function,
        axis = 0 is yet to be implemented.
        """
        if labels is not None:
            axis = validate_axis(axis)
            if axis == 1:
                return self.drop(columns=labels)
            raise NotImplementedError('Drop currently only works for axis=1')
        elif columns is not None:
            if is_name_like_tuple(columns):
                columns = [columns]
            elif is_name_like_value(columns):
                columns = [(columns,)]
            else:
                columns = [(col if is_name_like_tuple(col) else (col,)) for
                    col in columns]
            drop_column_labels = set(label for label in self._internal.
                column_labels for col in columns if label[:len(col)] == col)
            if len(drop_column_labels) == 0:
                raise KeyError(columns)
            cols, labels = zip(*((column, label) for column, label in zip(
                self._internal.data_spark_column_names, self._internal.
                column_labels) if label not in drop_column_labels))
            internal = self._internal.with_new_columns([self._kser_for(
                label) for label in labels])
            return DataFrame(internal)
        else:
            raise ValueError(
                "Need to specify at least one of 'labels' or 'columns'")

    def _sort(self, by, ascending, inplace, na_position):
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        if len(ascending) != len(by):
            raise ValueError('Length of ascending ({}) != length of by ({})'
                .format(len(ascending), len(by)))
        if na_position not in ('first', 'last'):
            raise ValueError("invalid na_position: '{}'".format(na_position))
        mapper = {(True, 'first'): lambda x: Column(getattr(x._jc,
            'asc_nulls_first')()), (True, 'last'): lambda x: Column(getattr
            (x._jc, 'asc_nulls_last')()), (False, 'first'): lambda x:
            Column(getattr(x._jc, 'desc_nulls_first')()), (False, 'last'): 
            lambda x: Column(getattr(x._jc, 'desc_nulls_last')())}
        by = [mapper[asc, na_position](scol) for scol, asc in zip(by,
            ascending)]
        sdf = self._internal.resolved_copy.spark_frame.sort(*(by + [
            NATURAL_ORDER_COLUMN_NAME]))
        kdf = DataFrame(self._internal.with_new_sdf(sdf))
        if inplace:
            self._update_internal_frame(kdf._internal)
            return None
        else:
            return kdf

    def sort_values(self, by, ascending=True, inplace=False, na_position='last'
        ):
        """
        Sort by the values along either axis.

        Parameters
        ----------
        by : str or list of str
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
        sorted_obj : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({
        ...     'col1': ['A', 'B', None, 'D', 'C'],
        ...     'col2': [2, 9, 8, 7, 4],
        ...     'col3': [0, 9, 4, 2, 3],
        ...   },
        ...   columns=['col1', 'col2', 'col3'])
        >>> df
           col1  col2  col3
        0     A     2     0
        1     B     9     9
        2  None     8     4
        3     D     7     2
        4     C     4     3

        Sort by col1

        >>> df.sort_values(by=['col1'])
           col1  col2  col3
        0     A     2     0
        1     B     9     9
        4     C     4     3
        3     D     7     2
        2  None     8     4

        Sort Descending

        >>> df.sort_values(by='col1', ascending=False)
           col1  col2  col3
        3     D     7     2
        4     C     4     3
        1     B     9     9
        0     A     2     0
        2  None     8     4

        Sort by multiple columns

        >>> df = ks.DataFrame({
        ...     'col1': ['A', 'A', 'B', None, 'D', 'C'],
        ...     'col2': [2, 1, 9, 8, 7, 4],
        ...     'col3': [0, 1, 9, 4, 2, 3],
        ...   },
        ...   columns=['col1', 'col2', 'col3'])
        >>> df.sort_values(by=['col1', 'col2'])
           col1  col2  col3
        1     A     1     1
        0     A     2     0
        2     B     9     9
        5     C     4     3
        4     D     7     2
        3  None     8     4
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if is_name_like_value(by):
            by = [by]
        else:
            assert is_list_like(by), type(by)
        new_by = []
        for colname in by:
            ser = self[colname]
            if not isinstance(ser, ks.Series):
                raise ValueError(
                    'The column %s is not unique. For a multi-index, the label must be a tuple with elements corresponding to each level.'
                     % name_like_string(colname))
            new_by.append(ser.spark.column)
        return self._sort(by=new_by, ascending=ascending, inplace=inplace,
            na_position=na_position)

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
        sorted_obj : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({'A': [2, 1, np.nan]}, index=['b', 'a', np.nan])

        >>> df.sort_index()
               A
        a    1.0
        b    2.0
        NaN  NaN

        >>> df.sort_index(ascending=False)
               A
        b    2.0
        a    1.0
        NaN  NaN

        >>> df.sort_index(na_position='first')
               A
        NaN  NaN
        a    1.0
        b    2.0

        >>> df.sort_index(inplace=True)
        >>> df
               A
        a    1.0
        b    2.0
        NaN  NaN

        >>> df = ks.DataFrame({'A': range(4), 'B': range(4)[::-1]},
        ...                   index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]],
        ...                   columns=['A', 'B'])

        >>> df.sort_index()
             A  B
        a 0  3  0
          1  2  1
        b 0  1  2
          1  0  3

        >>> df.sort_index(level=1)  # doctest: +SKIP
             A  B
        a 0  3  0
        b 0  1  2
        a 1  2  1
        b 1  0  3

        >>> df.sort_index(level=[1, 0])
             A  B
        a 0  3  0
        b 0  1  2
        a 1  2  1
        b 1  0  3
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'No other axis than 0 are supported at the moment')
        if kind is not None:
            raise NotImplementedError(
                'Specifying the sorting algorithm is not supported at the moment.'
                )
        if level is None or is_list_like(level) and len(level) == 0:
            by = self._internal.index_spark_columns
        elif is_list_like(level):
            by = [self._internal.index_spark_columns[l] for l in level]
        else:
            by = [self._internal.index_spark_columns[level]]
        return self._sort(by=by, ascending=ascending, inplace=inplace,
            na_position=na_position)

    def swaplevel(self, i=-2, j=-1, axis=0):
        """
        Swap levels i and j in a MultiIndex on a particular axis.

        Parameters
        ----------
        i, j : int or str
            Levels of the indices to be swapped. Can pass level name as string.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to swap levels on. 0 or 'index' for row-wise, 1 or
            'columns' for column-wise.

        Returns
        -------
        DataFrame
            DataFrame with levels swapped in MultiIndex.

        Examples
        --------
        >>> midx = pd.MultiIndex.from_arrays(
        ...     [['red', 'blue'], [1, 2], ['s', 'm']], names = ['color', 'number', 'size'])
        >>> midx  # doctest: +SKIP
        MultiIndex([( 'red', 1, 's'),
                    ('blue', 2, 'm')],
                   names=['color', 'number', 'size'])

        Swap levels in a MultiIndex on index.

        >>> kdf = ks.DataFrame({'x': [5, 6], 'y':[5, 6]}, index=midx)
        >>> kdf  # doctest: +NORMALIZE_WHITESPACE
                           x  y
        color number size
        red   1      s     5  5
        blue  2      m     6  6

        >>> kdf.swaplevel()  # doctest: +NORMALIZE_WHITESPACE
                           x  y
        color size number
        red   s    1       5  5
        blue  m    2       6  6

        >>> kdf.swaplevel(0, 1)  # doctest: +NORMALIZE_WHITESPACE
                           x  y
        number color size
        1      red   s     5  5
        2      blue  m     6  6

        >>> kdf.swaplevel('number', 'size')  # doctest: +NORMALIZE_WHITESPACE
                           x  y
        color size number
        red   s    1       5  5
        blue  m    2       6  6

        Swap levels in a MultiIndex on columns.

        >>> kdf = ks.DataFrame({'x': [5, 6], 'y':[5, 6]})
        >>> kdf.columns = midx
        >>> kdf
        color  red blue
        number   1    2
        size     s    m
        0        5    5
        1        6    6

        >>> kdf.swaplevel(axis=1)
        color  red blue
        size     s    m
        number   1    2
        0        5    5
        1        6    6

        >>> kdf.swaplevel(axis=1)
        color  red blue
        size     s    m
        number   1    2
        0        5    5
        1        6    6

        >>> kdf.swaplevel(0, 1, axis=1)
        number   1    2
        color  red blue
        size     s    m
        0        5    5
        1        6    6

        >>> kdf.swaplevel('number', 'color', axis=1)
        number   1    2
        color  red blue
        size     s    m
        0        5    5
        1        6    6
        """
        axis = validate_axis(axis)
        if axis == 0:
            internal = self._swaplevel_index(i, j)
        else:
            assert axis == 1
            internal = self._swaplevel_columns(i, j)
        return DataFrame(internal)

    def swapaxes(self, i, j, copy=True):
        """
        Interchange axes and swap values axes appropriately.

        .. note:: This method is based on an expensive operation due to the nature
            of big data. Internally it needs to generate each row for each value, and
            then group twice - it is a huge operation. To prevent misusage, this method
            has the 'compute.max_rows' default limit of input length, and raises a ValueError.

                >>> from databricks.koalas.config import option_context
                >>> with option_context('compute.max_rows', 1000):  # doctest: +NORMALIZE_WHITESPACE
                ...     ks.DataFrame({'a': range(1001)}).swapaxes(i=0, j=1)
                Traceback (most recent call last):
                  ...
                ValueError: Current DataFrame has more then the given limit 1000 rows.
                Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option'
                to retrieve to retrieve more than 1000 rows. Note that, before changing the
                'compute.max_rows', this operation is considerably expensive.

        Parameters
        ----------
        i: {0 or 'index', 1 or 'columns'}. The axis to swap.
        j: {0 or 'index', 1 or 'columns'}. The axis to swap.
        copy : bool, default True.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> kdf = ks.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['x', 'y', 'z'], columns=['a', 'b', 'c']
        ... )
        >>> kdf
           a  b  c
        x  1  2  3
        y  4  5  6
        z  7  8  9
        >>> kdf.swapaxes(i=1, j=0)
           x  y  z
        a  1  4  7
        b  2  5  8
        c  3  6  9
        >>> kdf.swapaxes(i=1, j=1)
           a  b  c
        x  1  2  3
        y  4  5  6
        z  7  8  9
        """
        assert copy is True
        i = validate_axis(i)
        j = validate_axis(j)
        return self.copy() if i == j else self.transpose()

    def _swaplevel_columns(self, i, j):
        assert isinstance(self.columns, pd.MultiIndex)
        for index in (i, j):
            if not isinstance(index, int) and index not in self.columns.names:
                raise KeyError('Level %s not found' % index)
        i = i if isinstance(i, int) else self.columns.names.index(i)
        j = j if isinstance(j, int) else self.columns.names.index(j)
        for index in (i, j):
            if index >= len(self.columns) or index < -len(self.columns):
                raise IndexError(
                    'Too many levels: Columns have only %s levels, %s is not a valid level number'
                     % (self._internal.index_level, index))
        column_label_names = self._internal.column_label_names.copy()
        column_label_names[i], column_label_names[j] = column_label_names[j
            ], column_label_names[i]
        column_labels = self._internal._column_labels
        column_label_list = [list(label) for label in column_labels]
        for label_list in column_label_list:
            label_list[i], label_list[j] = label_list[j], label_list[i]
        column_labels = [tuple(x) for x in column_label_list]
        internal = self._internal.copy(column_label_names=list(
            column_label_names), column_labels=list(column_labels))
        return internal

    def _swaplevel_index(self, i, j):
        assert isinstance(self.index, ks.MultiIndex)
        for index in (i, j):
            if not isinstance(index, int) and index not in self.index.names:
                raise KeyError('Level %s not found' % index)
        i = i if isinstance(i, int) else self.index.names.index(i)
        j = j if isinstance(j, int) else self.index.names.index(j)
        for index in (i, j):
            if (index >= self._internal.index_level or index < -self.
                _internal.index_level):
                raise IndexError(
                    'Too many levels: Index has only %s levels, %s is not a valid level number'
                     % (self._internal.index_level, index))
        index_map = list(zip(self._internal.index_spark_columns, self.
            _internal.index_names, self._internal.index_dtypes))
        index_map[i], index_map[j] = index_map[j], index_map[i]
        index_spark_columns, index_names, index_dtypes = zip(*index_map)
        internal = self._internal.copy(index_spark_columns=list(
            index_spark_columns), index_names=list(index_names),
            index_dtypes=list(index_dtypes))
        return internal

    def nlargest(self, n, columns):
        """
        Return the first `n` rows ordered by `columns` in descending order.

        Return the first `n` rows with the largest values in `columns`, in
        descending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to
        ``df.sort_values(columns, ascending=False).head(n)``, but more
        performant in pandas.
        In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : label or list of labels
            Column label(s) to order by.

        Returns
        -------
        DataFrame
            The first `n` rows ordered by the given columns in descending
            order.

        See Also
        --------
        DataFrame.nsmallest : Return the first `n` rows ordered by `columns` in
            ascending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Notes
        -----

        This function cannot be used with all column types. For example, when
        specifying columns with `object` or `category` dtypes, ``TypeError`` is
        raised.

        Examples
        --------
        >>> df = ks.DataFrame({'X': [1, 2, 3, 5, 6, 7, np.nan],
        ...                    'Y': [6, 7, 8, 9, 10, 11, 12]})
        >>> df
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8
        3  5.0   9
        4  6.0  10
        5  7.0  11
        6  NaN  12

        In the following example, we will use ``nlargest`` to select the three
        rows having the largest values in column "population".

        >>> df.nlargest(n=3, columns='X')
             X   Y
        5  7.0  11
        4  6.0  10
        3  5.0   9

        >>> df.nlargest(n=3, columns=['Y', 'X'])
             X   Y
        6  NaN  12
        5  7.0  11
        4  6.0  10

        """
        return self.sort_values(by=columns, ascending=False).head(n=n)

    def nsmallest(self, n, columns):
        """
        Return the first `n` rows ordered by `columns` in ascending order.

        Return the first `n` rows with the smallest values in `columns`, in
        ascending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to ``df.sort_values(columns, ascending=True).head(n)``,
        but more performant. In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Parameters
        ----------
        n : int
            Number of items to retrieve.
        columns : list or str
            Column name or names to order by.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.nlargest : Return the first `n` rows ordered by `columns` in
            descending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Examples
        --------
        >>> df = ks.DataFrame({'X': [1, 2, 3, 5, 6, 7, np.nan],
        ...                    'Y': [6, 7, 8, 9, 10, 11, 12]})
        >>> df
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8
        3  5.0   9
        4  6.0  10
        5  7.0  11
        6  NaN  12

        In the following example, we will use ``nsmallest`` to select the
        three rows having the smallest values in column "a".

        >>> df.nsmallest(n=3, columns='X') # doctest: +NORMALIZE_WHITESPACE
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8

        To order by the largest values in column "a" and then "c", we can
        specify multiple columns like in the next example.

        >>> df.nsmallest(n=3, columns=['Y', 'X']) # doctest: +NORMALIZE_WHITESPACE
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8
        """
        return self.sort_values(by=columns, ascending=True).head(n=n)

    def isin(self, values):
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : iterable or dict
           The sequence of values to test. If values is a dict,
           the keys must be the column names, which must match.
           Series and DataFrame are not supported.

        Returns
        -------
        DataFrame
            DataFrame of booleans showing whether each element in the DataFrame
            is contained in values.

        Examples
        --------
        >>> df = ks.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
        ...                   index=['falcon', 'dog'],
        ...                   columns=['num_legs', 'num_wings'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        When ``values`` is a list check whether every value in the DataFrame
        is present in the list (which animals have 0 or 2 legs or wings)

        >>> df.isin([0, 2])
                num_legs  num_wings
        falcon      True       True
        dog        False       True

        When ``values`` is a dict, we can pass values to check for each
        column separately:

        >>> df.isin({'num_wings': [0, 3]})
                num_legs  num_wings
        falcon     False      False
        dog        False       True
        """
        if isinstance(values, (pd.DataFrame, pd.Series)):
            raise NotImplementedError('DataFrame and Series are not supported')
        if isinstance(values, dict) and not set(values.keys()).issubset(self
            .columns):
            raise AttributeError("'DataFrame' object has no attribute %s" %
                set(values.keys()).difference(self.columns))
        data_spark_columns = []
        if isinstance(values, dict):
            for i, col in enumerate(self.columns):
                if col in values:
                    item = values[col]
                    item = item.tolist() if isinstance(item, np.ndarray
                        ) else list(item)
                    data_spark_columns.append(self._internal.
                        spark_column_for(self._internal.column_labels[i]).
                        isin(item).alias(self._internal.
                        data_spark_column_names[i]))
                else:
                    data_spark_columns.append(F.lit(False).alias(self.
                        _internal.data_spark_column_names[i]))
        elif is_list_like(values):
            values = values.tolist() if isinstance(values, np.ndarray
                ) else list(values)
            data_spark_columns += [self._internal.spark_column_for(label).
                isin(values).alias(self._internal.spark_column_name_for(
                label)) for label in self._internal.column_labels]
        else:
            raise TypeError(
                'Values should be iterable, Series, DataFrame or dict.')
        return DataFrame(self._internal.with_new_columns(data_spark_columns))

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.shape
        (2, 2)

        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4],
        ...                    'col3': [5, 6]})
        >>> df.shape
        (2, 3)
        """
        return len(self), len(self.columns)

    def merge(self, right, how='inner', on=None, left_on=None, right_on=
        None, left_index=False, right_index=False, suffixes=('_x', '_y')):
        """
        Merge DataFrame objects with a database-style join.

        The index of the resulting DataFrame will be one of the following:
            - 0...n if no index is used for merging
            - Index of the left DataFrame if merged only on the index of the right DataFrame
            - Index of the right DataFrame if merged only on the index of the left DataFrame
            - All involved indices if merged using the indices of both DataFrames
                e.g. if `left` with indices (a, x) and `right` with indices (b, x), the result will
                be an index (x, a, b)

        Parameters
        ----------
        right: Object to merge with.
        how: Type of merge to be performed.
            {'left', 'right', 'outer', 'inner'}, default 'inner'

            left: use only keys from left frame, similar to a SQL left outer join; not preserve
                key order unlike pandas.
            right: use only keys from right frame, similar to a SQL right outer join; not preserve
                key order unlike pandas.
            outer: use union of keys from both frames, similar to a SQL full outer join; sort keys
                lexicographically.
            inner: use intersection of keys from both frames, similar to a SQL inner join;
                not preserve the order of the left keys unlike pandas.
        on: Column or index level names to join on. These must be found in both DataFrames. If on
            is None and not merging on indexes then this defaults to the intersection of the
            columns in both DataFrames.
        left_on: Column or index level names to join on in the left DataFrame. Can also
            be an array or list of arrays of the length of the left DataFrame.
            These arrays are treated as if they are columns.
        right_on: Column or index level names to join on in the right DataFrame. Can also
            be an array or list of arrays of the length of the right DataFrame.
            These arrays are treated as if they are columns.
        left_index: Use the index from the left DataFrame as the join key(s). If it is a
            MultiIndex, the number of keys in the other DataFrame (either the index or a number of
            columns) must match the number of levels.
        right_index: Use the index from the right DataFrame as the join key. Same caveats as
            left_index.
        suffixes: Suffix to apply to overlapping column names in the left and right side,
            respectively.

        Returns
        -------
        DataFrame
            A DataFrame of the two merged objects.

        See Also
        --------
        DataFrame.join : Join columns of another DataFrame.
        DataFrame.update : Modify in place using non-NA values from another DataFrame.
        DataFrame.hint : Specifies some hint on the current DataFrame.
        broadcast : Marks a DataFrame as small enough for use in broadcast joins.

        Examples
        --------
        >>> df1 = ks.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
        ...                     'value': [1, 2, 3, 5]},
        ...                    columns=['lkey', 'value'])
        >>> df2 = ks.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
        ...                     'value': [5, 6, 7, 8]},
        ...                    columns=['rkey', 'value'])
        >>> df1
          lkey  value
        0  foo      1
        1  bar      2
        2  baz      3
        3  foo      5
        >>> df2
          rkey  value
        0  foo      5
        1  bar      6
        2  baz      7
        3  foo      8

        Merge df1 and df2 on the lkey and rkey columns. The value columns have
        the default suffixes, _x and _y, appended.

        >>> merged = df1.merge(df2, left_on='lkey', right_on='rkey')
        >>> merged.sort_values(by=['lkey', 'value_x', 'rkey', 'value_y'])  # doctest: +ELLIPSIS
          lkey  value_x rkey  value_y
        ...bar        2  bar        6
        ...baz        3  baz        7
        ...foo        1  foo        5
        ...foo        1  foo        8
        ...foo        5  foo        5
        ...foo        5  foo        8

        >>> left_kdf = ks.DataFrame({'A': [1, 2]})
        >>> right_kdf = ks.DataFrame({'B': ['x', 'y']}, index=[1, 2])

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True).sort_index()
           A  B
        1  2  x

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='left').sort_index()
           A     B
        0  1  None
        1  2     x

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='right').sort_index()
             A  B
        1  2.0  x
        2  NaN  y

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='outer').sort_index()
             A     B
        0  1.0  None
        1  2.0     x
        2  NaN     y

        Notes
        -----
        As described in #263, joining string columns currently returns None for missing values
            instead of NaN.
        """

        def to_list(os):
            if os is None:
                return []
            elif is_name_like_tuple(os):
                return [os]
            elif is_name_like_value(os):
                return [(os,)]
            else:
                return [(o if is_name_like_tuple(o) else (o,)) for o in os]
        if isinstance(right, ks.Series):
            right = right.to_frame()
        if on:
            if left_on or right_on:
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" and "right_on", not a combination of both.'
                    )
            left_key_names = list(map(self._internal.spark_column_name_for,
                to_list(on)))
            right_key_names = list(map(right._internal.
                spark_column_name_for, to_list(on)))
        else:
            if left_index:
                left_key_names = self._internal.index_spark_column_names
            else:
                left_key_names = list(map(self._internal.
                    spark_column_name_for, to_list(left_on)))
            if right_index:
                right_key_names = right._internal.index_spark_column_names
            else:
                right_key_names = list(map(right._internal.
                    spark_column_name_for, to_list(right_on)))
            if left_key_names and not right_key_names:
                raise ValueError('Must pass right_on or right_index=True')
            if right_key_names and not left_key_names:
                raise ValueError('Must pass left_on or left_index=True')
            if not left_key_names and not right_key_names:
                common = list(self.columns.intersection(right.columns))
                if len(common) == 0:
                    raise ValueError(
                        'No common columns to perform merge on. Merge options: left_on=None, right_on=None, left_index=False, right_index=False'
                        )
                left_key_names = list(map(self._internal.
                    spark_column_name_for, to_list(common)))
                right_key_names = list(map(right._internal.
                    spark_column_name_for, to_list(common)))
            if len(left_key_names) != len(right_key_names):
                raise ValueError('len(left_keys) must equal len(right_keys)')
        right_prefix = '__right_'
        right_key_names = [(right_prefix + right_key_name) for
            right_key_name in right_key_names]
        how = validate_how(how)

        def resolve(internal, side):
            rename = lambda col: '__{}_{}'.format(side, col)
            internal = internal.resolved_copy
            sdf = internal.spark_frame
            sdf = sdf.select([scol_for(sdf, col).alias(rename(col)) for col in
                sdf.columns if col not in HIDDEN_COLUMNS] + list(
                HIDDEN_COLUMNS))
            return internal.copy(spark_frame=sdf, index_spark_columns=[
                scol_for(sdf, rename(col)) for col in internal.
                index_spark_column_names], data_spark_columns=[scol_for(sdf,
                rename(col)) for col in internal.data_spark_column_names])
        left_internal = self._internal.resolved_copy
        right_internal = resolve(right._internal, 'right')
        left_table = left_internal.spark_frame.alias('left_table')
        right_table = right_internal.spark_frame.alias('right_table')
        left_key_columns = [scol_for(left_table, label) for label in
            left_key_names]
        right_key_columns = [scol_for(right_table, label) for label in
            right_key_names]
        join_condition = reduce(lambda x, y: x & y, [(lkey == rkey) for 
            lkey, rkey in zip(left_key_columns, right_key_columns)])
        joined_table = left_table.join(right_table, join_condition, how=how)
        left_suffix = suffixes[0]
        right_suffix = suffixes[1]
        duplicate_columns = set(left_internal.column_labels) & set(
            right_internal.column_labels)
        exprs = []
        data_columns = []
        column_labels = []
        left_scol_for = lambda label: scol_for(left_table, left_internal.
            spark_column_name_for(label))
        right_scol_for = lambda label: scol_for(right_table, right_internal
            .spark_column_name_for(label))
        for label in left_internal.column_labels:
            col = left_internal.spark_column_name_for(label)
            scol = left_scol_for(label)
            if label in duplicate_columns:
                spark_column_name = left_internal.spark_column_name_for(label)
                if (spark_column_name in left_key_names and right_prefix +
                    spark_column_name in right_key_names):
                    right_scol = right_scol_for(label)
                    if how == 'right':
                        scol = right_scol.alias(col)
                    elif how == 'full':
                        scol = F.when(scol.isNotNull(), scol).otherwise(
                            right_scol).alias(col)
                    else:
                        pass
                else:
                    col = col + left_suffix
                    scol = scol.alias(col)
                    label = tuple([str(label[0]) + left_suffix] + list(
                        label[1:]))
            exprs.append(scol)
            data_columns.append(col)
            column_labels.append(label)
        for label in right_internal.column_labels:
            col = right_internal.spark_column_name_for(label)[len(
                right_prefix):]
            scol = right_scol_for(label).alias(col)
            if label in duplicate_columns:
                spark_column_name = left_internal.spark_column_name_for(label)
                if (spark_column_name in left_key_names and right_prefix +
                    spark_column_name in right_key_names):
                    continue
                else:
                    col = col + right_suffix
                    scol = scol.alias(col)
                    label = tuple([str(label[0]) + right_suffix] + list(
                        label[1:]))
            exprs.append(scol)
            data_columns.append(col)
            column_labels.append(label)
        left_index_scols = left_internal.index_spark_columns
        right_index_scols = right_internal.index_spark_columns
        if left_index:
            if right_index:
                if how in ('inner', 'left'):
                    exprs.extend(left_index_scols)
                    index_spark_column_names = (left_internal.
                        index_spark_column_names)
                    index_names = left_internal.index_names
                elif how == 'right':
                    exprs.extend(right_index_scols)
                    index_spark_column_names = (right_internal.
                        index_spark_column_names)
                    index_names = right_internal.index_names
                else:
                    index_spark_column_names = (left_internal.
                        index_spark_column_names)
                    index_names = left_internal.index_names
                    for col, left_scol, right_scol in zip(
                        index_spark_column_names, left_index_scols,
                        right_index_scols):
                        scol = F.when(left_scol.isNotNull(), left_scol
                            ).otherwise(right_scol)
                        exprs.append(scol.alias(col))
            else:
                exprs.extend(right_index_scols)
                index_spark_column_names = (right_internal.
                    index_spark_column_names)
                index_names = right_internal.index_names
        elif right_index:
            exprs.extend(left_index_scols)
            index_spark_column_names = left_internal.index_spark_column_names
            index_names = left_internal.index_names
        else:
            index_spark_column_names = []
            index_names = []
        selected_columns = joined_table.select(*exprs)
        internal = InternalFrame(spark_frame=selected_columns,
            index_spark_columns=[scol_for(selected_columns, col) for col in
            index_spark_column_names], index_names=index_names,
            column_labels=column_labels, data_spark_columns=[scol_for(
            selected_columns, col) for col in data_columns])
        return DataFrame(internal)

    def join(self, right, on=None, how='left', lsuffix='', rsuffix=''):
        """
        Join columns of another DataFrame.

        Join columns with `right` DataFrame either on index or on a key column. Efficiently join
        multiple DataFrame objects by index at once by passing a list.

        Parameters
        ----------
        right: DataFrame, Series
        on: str, list of str, or array-like, optional
            Column or index level name(s) in the caller to join on the index in `right`, otherwise
            joins index-on-index. If multiple values given, the `right` DataFrame must have a
            MultiIndex. Can pass an array as the join key if it is not already contained in the
            calling DataFrame. Like an Excel VLOOKUP operation.
        how: {'left', 'right', 'outer', 'inner'}, default 'left'
            How to handle the operation of the two objects.

            * left: use `left` frame’s index (or column if on is specified).
            * right: use `right`’s index.
            * outer: form union of `left` frame’s index (or column if on is specified) with
              right’s index, and sort it. lexicographically.
            * inner: form intersection of `left` frame’s index (or column if on is specified)
              with `right`’s index, preserving the order of the `left`’s one.
        lsuffix : str, default ''
            Suffix to use from left frame's overlapping columns.
        rsuffix : str, default ''
            Suffix to use from `right` frame's overlapping columns.

        Returns
        -------
        DataFrame
            A dataframe containing columns from both the `left` and `right`.

        See Also
        --------
        DataFrame.merge: For column(s)-on-columns(s) operations.
        DataFrame.update : Modify in place using non-NA values from another DataFrame.
        DataFrame.hint : Specifies some hint on the current DataFrame.
        broadcast : Marks a DataFrame as small enough for use in broadcast joins.

        Notes
        -----
        Parameters on, lsuffix, and rsuffix are not supported when passing a list of DataFrame
        objects.

        Examples
        --------
        >>> kdf1 = ks.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
        ...                      'A': ['A0', 'A1', 'A2', 'A3']},
        ...                     columns=['key', 'A'])
        >>> kdf2 = ks.DataFrame({'key': ['K0', 'K1', 'K2'],
        ...                      'B': ['B0', 'B1', 'B2']},
        ...                     columns=['key', 'B'])
        >>> kdf1
          key   A
        0  K0  A0
        1  K1  A1
        2  K2  A2
        3  K3  A3
        >>> kdf2
          key   B
        0  K0  B0
        1  K1  B1
        2  K2  B2

        Join DataFrames using their indexes.

        >>> join_kdf = kdf1.join(kdf2, lsuffix='_left', rsuffix='_right')
        >>> join_kdf.sort_values(by=join_kdf.columns)
          key_left   A key_right     B
        0       K0  A0        K0    B0
        1       K1  A1        K1    B1
        2       K2  A2        K2    B2
        3       K3  A3      None  None

        If we want to join using the key columns, we need to set key to be the index in both df and
        right. The joined DataFrame will have key as its index.

        >>> join_kdf = kdf1.set_index('key').join(kdf2.set_index('key'))
        >>> join_kdf.sort_values(by=join_kdf.columns) # doctest: +NORMALIZE_WHITESPACE
              A     B
        key
        K0   A0    B0
        K1   A1    B1
        K2   A2    B2
        K3   A3  None

        Another option to join using the key columns is to use the on parameter. DataFrame.join
        always uses right’s index but we can use any column in df. This method not preserve the
        original DataFrame’s index in the result unlike pandas.

        >>> join_kdf = kdf1.join(kdf2.set_index('key'), on='key')
        >>> join_kdf.index
        Int64Index([0, 1, 2, 3], dtype='int64')
        """
        if isinstance(right, ks.Series):
            common = list(self.columns.intersection([right.name]))
        else:
            common = list(self.columns.intersection(right.columns))
        if len(common) > 0 and not lsuffix and not rsuffix:
            raise ValueError(
                'columns overlap but no suffix specified: {rename}'.format(
                rename=common))
        need_set_index = False
        if on:
            if not is_list_like(on):
                on = [on]
            if len(on) != right._internal.index_level:
                raise ValueError(
                    'len(left_on) must equal the number of levels in the index of "right"'
                    )
            need_set_index = len(set(on) & set(self.index.names)) == 0
        if need_set_index:
            self = self.set_index(on)
        join_kdf = self.merge(right, left_index=True, right_index=True, how
            =how, suffixes=(lsuffix, rsuffix))
        return join_kdf.reset_index() if need_set_index else join_kdf

    def append(self, other, ignore_index=False, verify_integrity=False,
        sort=False):
        """
        Append rows of other to the end of caller, returning a new object.

        Columns in other that are not in the caller are added as new columns.

        Parameters
        ----------
        other : DataFrame or Series/dict-like object, or list of these
            The data to append.

        ignore_index : boolean, default False
            If True, do not use the index labels.

        verify_integrity : boolean, default False
            If True, raise ValueError on creating index with duplicates.

        sort : boolean, default False
            Currently not supported.

        Returns
        -------
        appended : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2], [3, 4]], columns=list('AB'))

        >>> df.append(df)
           A  B
        0  1  2
        1  3  4
        0  1  2
        1  3  4

        >>> df.append(df, ignore_index=True)
           A  B
        0  1  2
        1  3  4
        2  1  2
        3  3  4
        """
        if isinstance(other, ks.Series):
            raise ValueError(
                'DataFrames.append() does not support appending Series to DataFrames'
                )
        if sort:
            raise NotImplementedError(
                "The 'sort' parameter is currently not supported")
        if not ignore_index:
            index_scols = self._internal.index_spark_columns
            if len(index_scols) != other._internal.index_level:
                raise ValueError(
                    'Both DataFrames have to have the same number of index levels'
                    )
            if verify_integrity and len(index_scols) > 0:
                if self._internal.spark_frame.select(index_scols).intersect(
                    other._internal.spark_frame.select(other._internal.
                    index_spark_columns)).count() > 0:
                    raise ValueError('Indices have overlapping values')
        from databricks.koalas.namespace import concat
        return cast(DataFrame, concat([self, other], ignore_index=ignore_index)
            )

    def update(self, other, join='left', overwrite=True):
        """
        Modify in place using non-NA values from another DataFrame.
        Aligns on indices. There is no return value.

        Parameters
        ----------
        other : DataFrame, or Series
        join : 'left', default 'left'
            Only left join is implemented, keeping the index and columns of the original object.
        overwrite : bool, default True
            How to handle non-NA values for overlapping keys:

            * True: overwrite original DataFrame's values with values from `other`.
            * False: only update values that are NA in the original DataFrame.

        Returns
        -------
        None : method directly changes calling object

        See Also
        --------
        DataFrame.merge : For column(s)-on-columns(s) operations.
        DataFrame.join : Join columns of another DataFrame.
        DataFrame.hint : Specifies some hint on the current DataFrame.
        broadcast : Marks a DataFrame as small enough for use in broadcast joins.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [400, 500, 600]}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': [4, 5, 6], 'C': [7, 8, 9]}, columns=['B', 'C'])
        >>> df.update(new_df)
        >>> df.sort_index()
           A  B
        0  1  4
        1  2  5
        2  3  6

        The DataFrame's length does not increase as a result of the update,
        only values at matching index/column labels are updated.

        >>> df = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': ['d', 'e', 'f', 'g', 'h', 'i']}, columns=['B'])
        >>> df.update(new_df)
        >>> df.sort_index()
           A  B
        0  a  d
        1  b  e
        2  c  f

        For Series, it's name attribute must be set.

        >>> df = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']}, columns=['A', 'B'])
        >>> new_column = ks.Series(['d', 'e'], name='B', index=[0, 2])
        >>> df.update(new_column)
        >>> df.sort_index()
           A  B
        0  a  d
        1  b  y
        2  c  e

        If `other` contains None the corresponding values are not updated in the original dataframe.

        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [400, 500, 600]}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': [4, None, 6]}, columns=['B'])
        >>> df.update(new_df)
        >>> df.sort_index()
           A      B
        0  1    4.0
        1  2  500.0
        2  3    6.0
        """
        if join != 'left':
            raise NotImplementedError('Only left join is supported')
        if isinstance(other, ks.Series):
            other = other.to_frame()
        update_columns = list(set(self._internal.column_labels).
            intersection(set(other._internal.column_labels)))
        update_sdf = self.join(other[update_columns], rsuffix='_new'
            )._internal.resolved_copy.spark_frame
        data_dtypes = self._internal.data_dtypes.copy()
        for column_labels in update_columns:
            column_name = self._internal.spark_column_name_for(column_labels)
            old_col = scol_for(update_sdf, column_name)
            new_col = scol_for(update_sdf, other._internal.
                spark_column_name_for(column_labels) + '_new')
            if overwrite:
                update_sdf = update_sdf.withColumn(column_name, F.when(
                    new_col.isNull(), old_col).otherwise(new_col))
            else:
                update_sdf = update_sdf.withColumn(column_name, F.when(
                    old_col.isNull(), new_col).otherwise(old_col))
            data_dtypes[self._internal.column_labels.index(column_labels)
                ] = None
        sdf = update_sdf.select([scol_for(update_sdf, col) for col in self.
            _internal.spark_column_names] + list(HIDDEN_COLUMNS))
        internal = self._internal.with_new_sdf(sdf, data_dtypes=data_dtypes)
        self._update_internal_frame(internal, requires_same_anchor=False)

    def sample(self, n=None, frac=None, replace=False, random_state=None):
        """
        Return a random sample of items from an axis of object.

        Please call this function using named argument by specifying the ``frac`` argument.

        You can use `random_state` for reproducibility. However, note that different from pandas,
        specifying a seed in Koalas/Spark does not guarantee the sampled rows will be fixed. The
        result set depends on not only the seed, but also how the data is distributed across
        machines and to some extent network randomness when shuffle operations are involved. Even
        in the simplest case, the result set will depend on the system's CPU core count.

        Parameters
        ----------
        n : int, optional
            Number of items to return. This is currently NOT supported. Use frac instead.
        frac : float, optional
            Fraction of axis items to return.
        replace : bool, default False
            Sample with or without replacement.
        random_state : int, optional
            Seed for the random number generator (if int).

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing the sampled items.

        Examples
        --------
        >>> df = ks.DataFrame({'num_legs': [2, 4, 8, 0],
        ...                    'num_wings': [2, 0, 0, 0],
        ...                    'num_specimen_seen': [10, 2, 1, 8]},
        ...                   index=['falcon', 'dog', 'spider', 'fish'],
        ...                   columns=['num_legs', 'num_wings', 'num_specimen_seen'])
        >>> df  # doctest: +SKIP
                num_legs  num_wings  num_specimen_seen
        falcon         2          2                 10
        dog            4          0                  2
        spider         8          0                  1
        fish           0          0                  8

        A random 25% sample of the ``DataFrame``.
        Note that we use `random_state` to ensure the reproducibility of
        the examples.

        >>> df.sample(frac=0.25, random_state=1)  # doctest: +SKIP
                num_legs  num_wings  num_specimen_seen
        falcon         2          2                 10
        fish           0          0                  8

        Extract 25% random elements from the ``Series`` ``df['num_legs']``, with replacement,
        so the same items could appear more than once.

        >>> df['num_legs'].sample(frac=0.4, replace=True, random_state=1)  # doctest: +SKIP
        falcon    2
        spider    8
        spider    8
        Name: num_legs, dtype: int64

        Specifying the exact number of items to return is not supported at the moment.

        >>> df.sample(n=5)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        NotImplementedError: Function sample currently does not support specifying ...
        """
        if n is not None:
            raise NotImplementedError(
                'Function sample currently does not support specifying exact number of items to return. Use frac instead.'
                )
        if frac is None:
            raise ValueError('frac must be specified.')
        sdf = self._internal.resolved_copy.spark_frame.sample(withReplacement
            =replace, fraction=frac, seed=random_state)
        return DataFrame(self._internal.with_new_sdf(sdf))

    def astype(self, dtype):
        """
        Cast a Koalas object to a specified dtype ``dtype``.

        Parameters
        ----------
        dtype : data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast entire Koalas object to
            the same type. Alternatively, use {col: dtype, ...}, where col is a
            column label and dtype is a numpy.dtype or Python type to cast one
            or more of the DataFrame's columns to column-specific types.

        Returns
        -------
        casted : same type as caller

        See Also
        --------
        to_datetime : Convert argument to datetime.

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]}, dtype='int64')
        >>> df
           a  b
        0  1  1
        1  2  2
        2  3  3

        Convert to float type:

        >>> df.astype('float')
             a    b
        0  1.0  1.0
        1  2.0  2.0
        2  3.0  3.0

        Convert to int64 type back:

        >>> df.astype('int64')
           a  b
        0  1  1
        1  2  2
        2  3  3

        Convert column a to float type:

        >>> df.astype({'a': float})
             a  b
        0  1.0  1
        1  2.0  2
        2  3.0  3

        """
        applied = []
        if is_dict_like(dtype):
            for col_name in dtype.keys():
                if col_name not in self.columns:
                    raise KeyError(
                        'Only a column name can be used for the key in a dtype mappings argument.'
                        )
            for col_name, col in self.items():
                if col_name in dtype:
                    applied.append(col.astype(dtype=dtype[col_name]))
                else:
                    applied.append(col)
        else:
            for col_name, col in self.items():
                applied.append(col.astype(dtype=dtype))
        return DataFrame(self._internal.with_new_columns(applied))

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
        DataFrame
           New DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        Series.add_suffix: Suffix row labels with string `suffix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]}, columns=['A', 'B'])
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_prefix('col_')
           col_A  col_B
        0      1      3
        1      2      4
        2      3      5
        3      4      6
        """
        assert isinstance(prefix, str)
        return self._apply_series_op(lambda kser: kser.rename(tuple([(
            prefix + i) for i in kser._column_label])))

    def add_suffix(self, suffix):
        """
        Suffix labels with string `suffix`.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        suffix : str
           The string to add before each label.

        Returns
        -------
        DataFrame
           New DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        Series.add_suffix: Suffix row labels with string `suffix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]}, columns=['A', 'B'])
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_suffix('_col')
           A_col  B_col
        0      1      3
        1      2      4
        2      3      5
        3      4      6
        """
        assert isinstance(suffix, str)
        return self._apply_series_op(lambda kser: kser.rename(tuple([(i +
            suffix) for i in kser._column_label])))

    def describe(self, percentiles=None):
        """
        Generate descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding
        ``NaN`` values.

        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types. The output
        will vary depending on what is provided. Refer to the notes
        below for more detail.

        Parameters
        ----------
        percentiles : list of ``float`` in range [0.0, 1.0], default [0.25, 0.5, 0.75]
            A list of percentiles to be computed.

        Returns
        -------
        DataFrame
            Summary statistics of the Dataframe provided.

        See Also
        --------
        DataFrame.count: Count number of non-NA/null observations.
        DataFrame.max: Maximum of the values in the object.
        DataFrame.min: Minimum of the values in the object.
        DataFrame.mean: Mean of the values.
        DataFrame.std: Standard deviation of the observations.

        Notes
        -----
        For numeric data, the result's index will include ``count``,
        ``mean``, ``std``, ``min``, ``25%``, ``50%``, ``75%``, ``max``.

        Currently only numeric data is supported.

        Examples
        --------
        Describing a numeric ``Series``.

        >>> s = ks.Series([1, 2, 3])
        >>> s.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.0
        50%      2.0
        75%      3.0
        max      3.0
        dtype: float64

        Describing a ``DataFrame``. Only numeric fields are returned.

        >>> df = ks.DataFrame({'numeric1': [1, 2, 3],
        ...                    'numeric2': [4.0, 5.0, 6.0],
        ...                    'object': ['a', 'b', 'c']
        ...                   },
        ...                   columns=['numeric1', 'numeric2', 'object'])
        >>> df.describe()
               numeric1  numeric2
        count       3.0       3.0
        mean        2.0       5.0
        std         1.0       1.0
        min         1.0       4.0
        25%         1.0       4.0
        50%         2.0       5.0
        75%         3.0       6.0
        max         3.0       6.0

        For multi-index columns:

        >>> df.columns = [('num', 'a'), ('num', 'b'), ('obj', 'c')]
        >>> df.describe()  # doctest: +NORMALIZE_WHITESPACE
               num
                 a    b
        count  3.0  3.0
        mean   2.0  5.0
        std    1.0  1.0
        min    1.0  4.0
        25%    1.0  4.0
        50%    2.0  5.0
        75%    3.0  6.0
        max    3.0  6.0

        >>> df[('num', 'b')].describe()
        count    3.0
        mean     5.0
        std      1.0
        min      4.0
        25%      4.0
        50%      5.0
        75%      6.0
        max      6.0
        Name: (num, b), dtype: float64

        Describing a ``DataFrame`` and selecting custom percentiles.

        >>> df = ks.DataFrame({'numeric1': [1, 2, 3],
        ...                    'numeric2': [4.0, 5.0, 6.0]
        ...                   },
        ...                   columns=['numeric1', 'numeric2'])
        >>> df.describe(percentiles = [0.85, 0.15])
               numeric1  numeric2
        count       3.0       3.0
        mean        2.0       5.0
        std         1.0       1.0
        min         1.0       4.0
        15%         1.0       4.0
        50%         2.0       5.0
        85%         3.0       6.0
        max         3.0       6.0

        Describing a column from a ``DataFrame`` by accessing it as
        an attribute.

        >>> df.numeric1.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.0
        50%      2.0
        75%      3.0
        max      3.0
        Name: numeric1, dtype: float64

        Describing a column from a ``DataFrame`` by accessing it as
        an attribute and selecting custom percentiles.

        >>> df.numeric1.describe(percentiles = [0.85, 0.15])
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        15%      1.0
        50%      2.0
        85%      3.0
        max      3.0
        Name: numeric1, dtype: float64
        """
        exprs = []
        column_labels = []
        for label in self._internal.column_labels:
            scol = self._internal.spark_column_for(label)
            spark_type = self._internal.spark_type_for(label)
            if isinstance(spark_type, DoubleType) or isinstance(spark_type,
                FloatType):
                exprs.append(F.nanvl(scol, F.lit(None)).alias(self.
                    _internal.spark_column_name_for(label)))
                column_labels.append(label)
            elif isinstance(spark_type, NumericType):
                exprs.append(scol)
                column_labels.append(label)
        if len(exprs) == 0:
            raise ValueError('Cannot describe a DataFrame without columns')
        if percentiles is not None:
            if any(p < 0.0 or p > 1.0 for p in percentiles):
                raise ValueError(
                    'Percentiles should all be in the interval [0, 1]')
            percentiles = percentiles + [0.5
                ] if 0.5 not in percentiles else percentiles
        else:
            percentiles = [0.25, 0.5, 0.75]
        formatted_perc = ['{:.0%}'.format(p) for p in sorted(percentiles)]
        stats = ['count', 'mean', 'stddev', 'min', *formatted_perc, 'max']
        sdf = self._internal.spark_frame.select(*exprs).summary(stats)
        sdf = sdf.replace('stddev', 'std', subset='summary')
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
            scol_for(sdf, 'summary')], column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, self._internal.
            spark_column_name_for(label)) for label in column_labels])
        return DataFrame(internal).astype('float64')

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        """
        Return DataFrame with duplicate rows removed, optionally only
        considering certain columns.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to keep.
            - ``first`` : Drop duplicates except for the first occurrence.
            - ``last`` : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.
        inplace : boolean, default False
            Whether to drop duplicates in place or to return a copy.

        Returns
        -------
        DataFrame
            DataFrame with duplicates removed or None if ``inplace=True``.

        >>> df = ks.DataFrame(
        ...     {'a': [1, 2, 2, 2, 3], 'b': ['a', 'a', 'a', 'c', 'd']}, columns = ['a', 'b'])
        >>> df
           a  b
        0  1  a
        1  2  a
        2  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates().sort_index()
           a  b
        0  1  a
        1  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates('a').sort_index()
           a  b
        0  1  a
        1  2  a
        4  3  d

        >>> df.drop_duplicates(['a', 'b']).sort_index()
           a  b
        0  1  a
        1  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates(keep='last').sort_index()
           a  b
        0  1  a
        2  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates(keep=False).sort_index()
           a  b
        0  1  a
        3  2  c
        4  3  d
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        sdf, column = self._mark_duplicates(subset, keep)
        sdf = sdf.where(~scol_for(sdf, column)).drop(column)
        internal = self._internal.with_new_sdf(sdf)
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def reindex(self, labels=None, index=None, columns=None, axis=None,
        copy=True, fill_value=None):
        """
        Conform DataFrame to new index with optional filling logic, placing
        NA/NaN in locations having no value in the previous index. A new object
        is produced unless the new index is equivalent to the current one and
        ``copy=False``.

        Parameters
        ----------
        labels: array-like, optional
            New labels / index to conform the axis specified by ‘axis’ to.
        index, columns: array-like, optional
            New labels / index to conform to, should be specified using keywords.
            Preferably an Index object to avoid duplicating data
        axis: int or str, optional
            Axis to target. Can be either the axis name (‘index’, ‘columns’) or
            number (0, 1).
        copy : bool, default True
            Return a new object, even if the passed indexes are the same.
        fill_value : scalar, default np.NaN
            Value to use for missing values. Defaults to NaN, but can be any
            "compatible" value.

        Returns
        -------
        DataFrame with changed index.

        See Also
        --------
        DataFrame.set_index : Set row labels.
        DataFrame.reset_index : Remove row labels or move them to new columns.

        Examples
        --------

        ``DataFrame.reindex`` supports two calling conventions

        * ``(index=index_labels, columns=column_labels, ...)``
        * ``(labels, axis={'index', 'columns'}, ...)``

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Create a dataframe with some fictional data.

        >>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
        >>> df = ks.DataFrame({
        ...      'http_status': [200, 200, 404, 404, 301],
        ...      'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
        ...       index=index,
        ...       columns=['http_status', 'response_time'])
        >>> df
                   http_status  response_time
        Firefox            200           0.04
        Chrome             200           0.02
        Safari             404           0.07
        IE10               404           0.08
        Konqueror          301           1.00

        Create a new index and reindex the dataframe. By default
        values in the new index that do not have corresponding
        records in the dataframe are assigned ``NaN``.

        >>> new_index= ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
        ...             'Chrome']
        >>> df.reindex(new_index).sort_index()
                       http_status  response_time
        Chrome               200.0           0.02
        Comodo Dragon          NaN            NaN
        IE10                 404.0           0.08
        Iceweasel              NaN            NaN
        Safari               404.0           0.07

        We can fill in the missing values by passing a value to
        the keyword ``fill_value``.

        >>> df.reindex(new_index, fill_value=0, copy=False).sort_index()
                       http_status  response_time
        Chrome                 200           0.02
        Comodo Dragon            0           0.00
        IE10                   404           0.08
        Iceweasel                0           0.00
        Safari                 404           0.07

        We can also reindex the columns.

        >>> df.reindex(columns=['http_status', 'user_agent']).sort_index()
                   http_status  user_agent
        Chrome             200         NaN
        Firefox            200         NaN
        IE10               404         NaN
        Konqueror          301         NaN
        Safari             404         NaN

        Or we can use "axis-style" keyword arguments

        >>> df.reindex(['http_status', 'user_agent'], axis="columns").sort_index()
                   http_status  user_agent
        Chrome             200         NaN
        Firefox            200         NaN
        IE10               404         NaN
        Konqueror          301         NaN
        Safari             404         NaN

        To further illustrate the filling functionality in
        ``reindex``, we will create a dataframe with a
        monotonically increasing index (for example, a sequence
        of dates).

        >>> date_index = pd.date_range('1/1/2010', periods=6, freq='D')
        >>> df2 = ks.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
        ...                    index=date_index)
        >>> df2.sort_index()
                    prices
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0

        Suppose we decide to expand the dataframe to cover a wider
        date range.

        >>> date_index2 = pd.date_range('12/29/2009', periods=10, freq='D')
        >>> df2.reindex(date_index2).sort_index()
                    prices
        2009-12-29     NaN
        2009-12-30     NaN
        2009-12-31     NaN
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN
        """
        if axis is not None and (index is not None or columns is not None):
            raise TypeError(
                "Cannot specify both 'axis' and any of 'index' or 'columns'.")
        if labels is not None:
            axis = validate_axis(axis)
            if axis == 0:
                index = labels
            elif axis == 1:
                columns = labels
            else:
                raise ValueError('No axis named %s for object type %s.' % (
                    axis, type(axis).__name__))
        if index is not None and not is_list_like(index):
            raise TypeError(
                'Index must be called with a collection of some kind, %s was passed'
                 % type(index))
        if columns is not None and not is_list_like(columns):
            raise TypeError(
                'Columns must be called with a collection of some kind, %s was passed'
                 % type(columns))
        df = self
        if index is not None:
            df = df._reindex_index(index, fill_value)
        if columns is not None:
            df = df._reindex_columns(columns, fill_value)
        if copy and df is self:
            return df.copy()
        else:
            return df

    def _reindex_index(self, index, fill_value):
        nlevels = self._internal.index_level
        assert nlevels <= 1 or isinstance(index, ks.MultiIndex
            ) and nlevels == index.nlevels, 'MultiIndex DataFrame can only be reindexed with a similar Koalas MultiIndex.'
        index_columns = self._internal.index_spark_column_names
        frame = self._internal.resolved_copy.spark_frame.drop(
            NATURAL_ORDER_COLUMN_NAME)
        if isinstance(index, ks.Index):
            if nlevels != index.nlevels:
                return DataFrame(index._internal.with_new_columns([])).reindex(
                    columns=self.columns, fill_value=fill_value)
            index_names = index._internal.index_names
            scols = index._internal.index_spark_columns
            labels = index._internal.spark_frame.select([scol.alias(
                index_column) for scol, index_column in zip(scols,
                index_columns)])
        else:
            kser = ks.Series(list(index))
            labels = kser._internal.spark_frame.select(kser.spark.column.
                alias(index_columns[0]))
            index_names = self._internal.index_names
        if fill_value is not None:
            frame_index_columns = [verify_temp_column_name(frame,
                '__frame_index_column_{}__'.format(i)) for i in range(nlevels)]
            index_scols = [scol_for(frame, index_col).alias(frame_index_col
                ) for index_col, frame_index_col in zip(index_columns,
                frame_index_columns)]
            scols = self._internal.resolved_copy.data_spark_columns
            frame = frame.select(index_scols + scols)
            temp_fill_value = verify_temp_column_name(frame, '__fill_value__')
            labels = labels.withColumn(temp_fill_value, F.lit(fill_value))
            frame_index_scols = [scol_for(frame, col) for col in
                frame_index_columns]
            labels_index_scols = [scol_for(labels, col) for col in
                index_columns]
            joined_df = frame.join(labels, on=[(fcol == lcol) for fcol,
                lcol in zip(frame_index_scols, labels_index_scols)], how=
                'right')
            joined_df = joined_df.select(*labels_index_scols, *[F.when(
                reduce(lambda c1, c2: c1 & c2, [(fcol.isNull() & lcol.
                isNotNull()) for fcol, lcol in zip(frame_index_scols,
                labels_index_scols)]), scol_for(joined_df, temp_fill_value)
                ).otherwise(scol_for(joined_df, col)).alias(col) for col in
                self._internal.data_spark_column_names])
        else:
            joined_df = frame.join(labels, on=index_columns, how='right')
        sdf = joined_df.drop(NATURAL_ORDER_COLUMN_NAME)
        internal = self._internal.copy(spark_frame=sdf, index_spark_columns
            =[scol_for(sdf, col) for col in self._internal.
            index_spark_column_names], index_names=index_names,
            index_dtypes=None, data_spark_columns=[scol_for(sdf, col) for
            col in self._internal.data_spark_column_names])
        return DataFrame(internal)

    def _reindex_columns(self, columns, fill_value):
        level = self._internal.column_labels_level
        if level > 1:
            label_columns = list(columns)
            for col in label_columns:
                if not isinstance(col, tuple):
                    raise TypeError('Expected tuple, got {}'.format(type(
                        col).__name__))
        else:
            label_columns = [(col,) for col in columns]
        for col in label_columns:
            if len(col) != level:
                raise ValueError("shape (1,{}) doesn't match the shape (1,{})"
                    .format(len(col), level))
        fill_value = np.nan if fill_value is None else fill_value
        scols_or_ksers, labels = [], []
        for label in label_columns:
            if label in self._internal.column_labels:
                scols_or_ksers.append(self._kser_for(label))
            else:
                scols_or_ksers.append(F.lit(fill_value).alias(
                    name_like_string(label)))
            labels.append(label)
        if isinstance(columns, pd.Index):
            column_label_names = [(name if is_name_like_tuple(name) else (
                name,)) for name in columns.names]
            internal = self._internal.with_new_columns(scols_or_ksers,
                column_labels=labels, column_label_names=column_label_names)
        else:
            internal = self._internal.with_new_columns(scols_or_ksers,
                column_labels=labels)
        return DataFrame(internal)

    def reindex_like(self, other, copy=True):
        """
        Return a DataFrame with matching indices as other object.

        Conform the object to the same index on all axes. Places NA/NaN in locations
        having no value in the previous index. A new object is produced unless the
        new index is equivalent to the current one and copy=False.

        Parameters
        ----------
        other : DataFrame
            Its row and column indices are used to define the new indices
            of this object.
        copy : bool, default True
            Return a new object, even if the passed indexes are the same.

        Returns
        -------
        DataFrame
            DataFrame with changed indices on each axis.

        See Also
        --------
        DataFrame.set_index : Set row labels.
        DataFrame.reset_index : Remove row labels or move them to new columns.
        DataFrame.reindex : Change to new indices or expand indices.

        Notes
        -----
        Same as calling
        ``.reindex(index=other.index, columns=other.columns,...)``.

        Examples
        --------

        >>> df1 = ks.DataFrame([[24.3, 75.7, 'high'],
        ...                     [31, 87.8, 'high'],
        ...                     [22, 71.6, 'medium'],
        ...                     [35, 95, 'medium']],
        ...                    columns=['temp_celsius', 'temp_fahrenheit',
        ...                             'windspeed'],
        ...                    index=pd.date_range(start='2014-02-12',
        ...                                        end='2014-02-15', freq='D'))
        >>> df1
                    temp_celsius  temp_fahrenheit windspeed
        2014-02-12          24.3             75.7      high
        2014-02-13          31.0             87.8      high
        2014-02-14          22.0             71.6    medium
        2014-02-15          35.0             95.0    medium

        >>> df2 = ks.DataFrame([[28, 'low'],
        ...                     [30, 'low'],
        ...                     [35.1, 'medium']],
        ...                    columns=['temp_celsius', 'windspeed'],
        ...                    index=pd.DatetimeIndex(['2014-02-12', '2014-02-13',
        ...                                            '2014-02-15']))
        >>> df2
                    temp_celsius windspeed
        2014-02-12          28.0       low
        2014-02-13          30.0       low
        2014-02-15          35.1    medium

        >>> df2.reindex_like(df1).sort_index() # doctest: +NORMALIZE_WHITESPACE
                    temp_celsius  temp_fahrenheit windspeed
        2014-02-12          28.0              NaN       low
        2014-02-13          30.0              NaN       low
        2014-02-14           NaN              NaN       None
        2014-02-15          35.1              NaN    medium
        """
        if isinstance(other, DataFrame):
            return self.reindex(index=other.index, columns=other.columns,
                copy=copy)
        else:
            raise TypeError('other must be a Koalas DataFrame')

    def melt(self, id_vars=None, value_vars=None, var_name=None, value_name
        ='value'):
        """
        Unpivot a DataFrame from wide format to long format, optionally
        leaving identifier variables set.

        This function is useful to massage a DataFrame into a format where one
        or more columns are identifier variables (`id_vars`), while all other
        columns, considered measured variables (`value_vars`), are "unpivoted" to
        the row axis, leaving just two non-identifier columns, 'variable' and
        'value'.

        Parameters
        ----------
        frame : DataFrame
        id_vars : tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
        value_vars : tuple, list, or ndarray, optional
            Column(s) to unpivot. If not specified, uses all columns that
            are not set as `id_vars`.
        var_name : scalar, default 'variable'
            Name to use for the 'variable' column. If None it uses `frame.columns.name` or
            ‘variable’.
        value_name : scalar, default 'value'
            Name to use for the 'value' column.

        Returns
        -------
        DataFrame
            Unpivoted DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
        ...                    'B': {0: 1, 1: 3, 2: 5},
        ...                    'C': {0: 2, 1: 4, 2: 6}},
        ...                   columns=['A', 'B', 'C'])
        >>> df
           A  B  C
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> ks.melt(df)
          variable value
        0        A     a
        1        B     1
        2        C     2
        3        A     b
        4        B     3
        5        C     4
        6        A     c
        7        B     5
        8        C     6

        >>> df.melt(id_vars='A')
           A variable  value
        0  a        B      1
        1  a        C      2
        2  b        B      3
        3  b        C      4
        4  c        B      5
        5  c        C      6

        >>> df.melt(value_vars='A')
          variable value
        0        A     a
        1        A     b
        2        A     c

        >>> ks.melt(df, id_vars=['A', 'B'])
           A  B variable  value
        0  a  1        C      2
        1  b  3        C      4
        2  c  5        C      6

        >>> df.melt(id_vars=['A'], value_vars=['C'])
           A variable  value
        0  a        C      2
        1  b        C      4
        2  c        C      6

        The names of 'variable' and 'value' columns can be customized:

        >>> ks.melt(df, id_vars=['A'], value_vars=['B'],
        ...         var_name='myVarname', value_name='myValname')
           A myVarname  myValname
        0  a         B          1
        1  b         B          3
        2  c         B          5
        """
        column_labels = self._internal.column_labels
        if id_vars is None:
            id_vars = []
        else:
            if isinstance(id_vars, tuple):
                if self._internal.column_labels_level == 1:
                    id_vars = [(idv if is_name_like_tuple(idv) else (idv,)) for
                        idv in id_vars]
                else:
                    raise ValueError(
                        'id_vars must be a list of tuples when columns are a MultiIndex'
                        )
            elif is_name_like_value(id_vars):
                id_vars = [(id_vars,)]
            else:
                id_vars = [(idv if is_name_like_tuple(idv) else (idv,)) for
                    idv in id_vars]
            non_existence_col = [idv for idv in id_vars if idv not in
                column_labels]
            if len(non_existence_col) != 0:
                raveled_column_labels = np.ravel(column_labels)
                missing = [nec for nec in np.ravel(non_existence_col) if 
                    nec not in raveled_column_labels]
                if len(missing) != 0:
                    raise KeyError(
                        "The following 'id_vars' are not present in the DataFrame: {}"
                        .format(missing))
                else:
                    raise KeyError('None of {} are in the {}'.format(
                        non_existence_col, column_labels))
        if value_vars is None:
            value_vars = []
        else:
            if isinstance(value_vars, tuple):
                if self._internal.column_labels_level == 1:
                    value_vars = [(valv if is_name_like_tuple(valv) else (
                        valv,)) for valv in value_vars]
                else:
                    raise ValueError(
                        'value_vars must be a list of tuples when columns are a MultiIndex'
                        )
            elif is_name_like_value(value_vars):
                value_vars = [(value_vars,)]
            else:
                value_vars = [(valv if is_name_like_tuple(valv) else (valv,
                    )) for valv in value_vars]
            non_existence_col = [valv for valv in value_vars if valv not in
                column_labels]
            if len(non_existence_col) != 0:
                raveled_column_labels = np.ravel(column_labels)
                missing = [nec for nec in np.ravel(non_existence_col) if 
                    nec not in raveled_column_labels]
                if len(missing) != 0:
                    raise KeyError(
                        "The following 'value_vars' are not present in the DataFrame: {}"
                        .format(missing))
                else:
                    raise KeyError('None of {} are in the {}'.format(
                        non_existence_col, column_labels))
        if len(value_vars) == 0:
            value_vars = column_labels
        column_labels = [label for label in column_labels if label not in
            id_vars]
        sdf = self._internal.spark_frame
        if var_name is None:
            if (self._internal.column_labels_level == 1 and self._internal.
                column_label_names[0] is None):
                var_name = ['variable']
            else:
                var_name = [(name_like_string(name) if name is not None else
                    'variable_{}'.format(i)) for i, name in enumerate(self.
                    _internal.column_label_names)]
        elif isinstance(var_name, str):
            var_name = [var_name]
        pairs = F.explode(F.array(*[F.struct(*([F.lit(c).alias(name) for c,
            name in zip(label, var_name)] + [self._internal.
            spark_column_for(label).alias(value_name)])) for label in
            column_labels if label in value_vars]))
        columns = [self._internal.spark_column_for(label).alias(
            name_like_string(label)) for label in id_vars] + [F.col(
            'pairs.`%s`' % name) for name in var_name] + [F.col(
            'pairs.`%s`' % value_name)]
        exploded_df = sdf.withColumn('pairs', pairs).select(columns)
        return DataFrame(InternalFrame(spark_frame=exploded_df,
            index_spark_columns=None, column_labels=[(label if len(label) ==
            1 else (name_like_string(label),)) for label in id_vars] + [(
            name,) for name in var_name] + [(value_name,)]))

    def stack(self):
        """
        Stack the prescribed level(s) from columns to index.

        Return a reshaped DataFrame or Series having a multi-level
        index with one or more new inner-most levels compared to the current
        DataFrame. The new inner-most levels are created by pivoting the
        columns of the current dataframe:

          - if the columns have a single level, the output is a Series;
          - if the columns have multiple levels, the new index
            level(s) is (are) taken from the prescribed level(s) and
            the output is a DataFrame.

        The new index levels are sorted.

        Returns
        -------
        DataFrame or Series
            Stacked dataframe or series.

        See Also
        --------
        DataFrame.unstack : Unstack prescribed level(s) from index axis
            onto column axis.
        DataFrame.pivot : Reshape dataframe from long format to wide
            format.
        DataFrame.pivot_table : Create a spreadsheet-style pivot table
            as a DataFrame.

        Notes
        -----
        The function is named by analogy with a collection of books
        being reorganized from being side by side on a horizontal
        position (the columns of the dataframe) to being stacked
        vertically on top of each other (in the index of the
        dataframe).

        Examples
        --------
        **Single level columns**

        >>> df_single_level_cols = ks.DataFrame([[0, 1], [2, 3]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=['weight', 'height'])

        Stacking a dataframe with a single level column axis returns a Series:

        >>> df_single_level_cols
             weight  height
        cat       0       1
        dog       2       3
        >>> df_single_level_cols.stack().sort_index()
        cat  height    1
             weight    0
        dog  height    3
             weight    2
        dtype: int64

        **Multi level columns: simple case**

        >>> multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('weight', 'pounds')])
        >>> df_multi_level_cols1 = ks.DataFrame([[1, 2], [2, 4]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol1)

        Stacking a dataframe with a multi-level column axis:

        >>> df_multi_level_cols1  # doctest: +NORMALIZE_WHITESPACE
            weight
                kg pounds
        cat      1      2
        dog      2      4
        >>> df_multi_level_cols1.stack().sort_index()
                    weight
        cat kg           1
            pounds       2
        dog kg           2
            pounds       4

        **Missing values**

        >>> multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('height', 'm')])
        >>> df_multi_level_cols2 = ks.DataFrame([[1.0, 2.0], [3.0, 4.0]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol2)

        It is common to have missing values when stacking a dataframe
        with multi-level columns, as the stacked dataframe typically
        has more values than the original dataframe. Missing values
        are filled with NaNs:

        >>> df_multi_level_cols2
            weight height
                kg      m
        cat    1.0    2.0
        dog    3.0    4.0
        >>> df_multi_level_cols2.stack().sort_index()  # doctest: +SKIP
                height  weight
        cat kg     NaN     1.0
            m      2.0     NaN
        dog kg     NaN     3.0
            m      4.0     NaN
        """
        from databricks.koalas.series import first_series
        if len(self._internal.column_labels) == 0:
            return DataFrame(self._internal.copy(column_label_names=self.
                _internal.column_label_names[:-1]).with_filter(F.lit(False)))
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
        column_labels = OrderedDict(sorted(column_labels.items(), key=lambda
            x: x[0]))
        index_name = self._internal.column_label_names[-1]
        column_label_names = self._internal.column_label_names[:-1]
        if len(column_label_names) == 0:
            column_label_names = [None]
        index_column = SPARK_INDEX_NAME_FORMAT(self._internal.index_level)
        data_columns = [name_like_string(label) for label in column_labels]
        structs = [F.struct([F.lit(value).alias(index_column)] + [(
            column_labels[label][value] if value in column_labels[label] else
            F.lit(None)).alias(name) for label, name in zip(column_labels,
            data_columns)]).alias(value) for value in index_values]
        pairs = F.explode(F.array(structs))
        sdf = self._internal.spark_frame.withColumn('pairs', pairs)
        sdf = sdf.select(self._internal.index_spark_columns + [sdf['pairs']
            [index_column].alias(index_column)] + [sdf['pairs'][name].alias
            (name) for name in data_columns])
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
            scol_for(sdf, col) for col in self._internal.
            index_spark_column_names + [index_column]], index_names=self.
            _internal.index_names + [index_name], index_dtypes=self.
            _internal.index_dtypes + [None], column_labels=list(
            column_labels), data_spark_columns=[scol_for(sdf, col) for col in
            data_columns], column_label_names=column_label_names)
        kdf = DataFrame(internal)
        if should_returns_series:
            return first_series(kdf)
        else:
            return kdf

    def unstack(self):
        """
        Pivot the (necessarily hierarchical) index labels.

        Returns a DataFrame having a new level of column labels whose inner-most level
        consists of the pivoted index labels.

        If the index is not a MultiIndex, the output will be a Series.

        .. note:: If the index is a MultiIndex, the output DataFrame could be very wide, and
            it could cause a serious performance degradation since Spark partitions it row based.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        DataFrame.pivot : Pivot a table based on column values.
        DataFrame.stack : Pivot a level of the column labels (inverse operation from unstack).

        Examples
        --------
        >>> df = ks.DataFrame({"A": {"0": "a", "1": "b", "2": "c"},
        ...                    "B": {"0": "1", "1": "3", "2": "5"},
        ...                    "C": {"0": "2", "1": "4", "2": "6"}},
        ...                   columns=["A", "B", "C"])
        >>> df
           A  B  C
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> df.unstack().sort_index()
        A  0    a
           1    b
           2    c
        B  0    1
           1    3
           2    5
        C  0    2
           1    4
           2    6
        dtype: object

        >>> df.columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C')])
        >>> df.unstack().sort_index()
        X  A  0    a
              1    b
              2    c
           B  0    1
              1    3
              2    5
        Y  C  0    2
              1    4
              2    6
        dtype: object

        For MultiIndex case:

        >>> df = ks.DataFrame({"A": ["a", "b", "c"],
        ...                    "B": [1, 3, 5],
        ...                    "C": [2, 4, 6]},
        ...                   columns=["A", "B", "C"])
        >>> df = df.set_index('A', append=True)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
             B  C
          A
        0 a  1  2
        1 b  3  4
        2 c  5  6
        >>> df.unstack().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B              C
        A    a    b    c    a    b    c
        0  1.0  NaN  NaN  2.0  NaN  NaN
        1  NaN  3.0  NaN  NaN  4.0  NaN
        2  NaN  NaN  5.0  NaN  NaN  6.0
        """
        from databricks.koalas.series import first_series
        if self._internal.index_level > 1:
            with option_context('compute.default_index_type', 'distributed'):
                df = self.reset_index()
            index = df._internal.column_labels[:self._internal.index_level - 1]
            columns = df.columns[self._internal.index_level - 1]
            df = df.pivot_table(index=index, columns=columns, values=self.
                _internal.column_labels, aggfunc='first')
            internal = df._internal.copy(index_names=self._internal.
                index_names[:-1], index_dtypes=self._internal.index_dtypes[
                :-1], column_label_names=df._internal.column_label_names[:-
                1] + [None if self._internal.index_names[-1] is None else
                df._internal.column_label_names[-1]])
            return DataFrame(internal)
        column_labels = self._internal.column_labels
        ser_name = SPARK_DEFAULT_SERIES_NAME
        sdf = self._internal.spark_frame
        new_index_columns = [SPARK_INDEX_NAME_FORMAT(i) for i in range(self
            ._internal.column_labels_level)]
        new_index_map = list(zip(new_index_columns, self._internal.
            column_label_names))
        pairs = F.explode(F.array(*[F.struct(*([F.lit(c).alias(name) for c,
            name in zip(idx, new_index_columns)] + [self._internal.
            spark_column_for(idx).alias(ser_name)])) for idx in column_labels])
            )
        columns = [F.col('pairs.%s' % name) for name in new_index_columns[:
            self._internal.column_labels_level]] + [F.col('pairs.%s' %
            ser_name)]
        new_index_len = len(new_index_columns)
        existing_index_columns = []
        for i, index_name in enumerate(self._internal.index_names):
            new_index_map.append((SPARK_INDEX_NAME_FORMAT(i + new_index_len
                ), index_name))
            existing_index_columns.append(self._internal.
                index_spark_columns[i].alias(SPARK_INDEX_NAME_FORMAT(i +
                new_index_len)))
        exploded_df = sdf.withColumn('pairs', pairs).select(
            existing_index_columns + columns)
        index_spark_column_names, index_names = zip(*new_index_map)
        return first_series(DataFrame(InternalFrame(exploded_df,
            index_spark_columns=[scol_for(exploded_df, col) for col in
            index_spark_column_names], index_names=list(index_names),
            column_labels=[None])))

    def all(self, axis=0):
        """
        Return whether all elements are True.

        Returns True unless there is at least one element within a series that is
        False or equivalent (e.g. zero or empty)

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Returns
        -------
        Series

        Examples
        --------
        Create a dataframe from a dictionary.

        >>> df = ks.DataFrame({
        ...    'col1': [True, True, True],
        ...    'col2': [True, False, False],
        ...    'col3': [0, 0, 0],
        ...    'col4': [1, 2, 3],
        ...    'col5': [True, True, None],
        ...    'col6': [True, False, None]},
        ...    columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])

        Default behaviour checks if column-wise values all return a boolean.

        >>> df.all()
        col1     True
        col2    False
        col3    False
        col4     True
        col5     True
        col6    False
        dtype: bool
        """
        from databricks.koalas.series import first_series
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        applied = []
        column_labels = self._internal.column_labels
        for label in column_labels:
            scol = self._internal.spark_column_for(label)
            all_col = F.min(F.coalesce(scol.cast('boolean'), F.lit(True)))
            applied.append(F.when(all_col.isNull(), True).otherwise(all_col))
        value_column = 'value'
        cols = []
        for label, applied_col in zip(column_labels, applied):
            cols.append(F.struct([F.lit(col).alias(SPARK_INDEX_NAME_FORMAT(
                i)) for i, col in enumerate(label)] + [applied_col.alias(
                value_column)]))
        sdf = self._internal.spark_frame.select(F.array(*cols).alias('arrays')
            ).select(F.explode(F.col('arrays')))
        sdf = sdf.selectExpr('col.*')
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
            scol_for(sdf, SPARK_INDEX_NAME_FORMAT(i)) for i in range(self.
            _internal.column_labels_level)], index_names=self._internal.
            column_label_names, column_labels=[None], data_spark_columns=[
            scol_for(sdf, value_column)])
        return first_series(DataFrame(internal))

    def any(self, axis=0):
        """
        Return whether any element is True.

        Returns False unless there is at least one element within a series that is
        True or equivalent (e.g. non-zero or non-empty).

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Returns
        -------
        Series

        Examples
        --------
        Create a dataframe from a dictionary.

        >>> df = ks.DataFrame({
        ...    'col1': [False, False, False],
        ...    'col2': [True, False, False],
        ...    'col3': [0, 0, 1],
        ...    'col4': [0, 1, 2],
        ...    'col5': [False, False, None],
        ...    'col6': [True, False, None]},
        ...    columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])

        Default behaviour checks if column-wise values all return a boolean.

        >>> df.any()
        col1    False
        col2     True
        col3     True
        col4     True
        col5    False
        col6     True
        dtype: bool
        """
        from databricks.koalas.series import first_series
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        applied = []
        column_labels = self._internal.column_labels
        for label in column_labels:
            scol = self._internal.spark_column_for(label)
            all_col = F.max(F.coalesce(scol.cast('boolean'), F.lit(False)))
            applied.append(F.when(all_col.isNull(), False).otherwise(all_col))
        value_column = 'value'
        cols = []
        for label, applied_col in zip(column_labels, applied):
            cols.append(F.struct([F.lit(col).alias(SPARK_INDEX_NAME_FORMAT(
                i)) for i, col in enumerate(label)] + [applied_col.alias(
                value_column)]))
        sdf = self._internal.spark_frame.select(F.array(*cols).alias('arrays')
            ).select(F.explode(F.col('arrays')))
        sdf = sdf.selectExpr('col.*')
        internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
            scol_for(sdf, SPARK_INDEX_NAME_FORMAT(i)) for i in range(self.
            _internal.column_labels_level)], index_names=self._internal.
            column_label_names, column_labels=[None], data_spark_columns=[
            scol_for(sdf, value_column)])
        return first_series(DataFrame(internal))

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
        >>> df = ks.DataFrame({'A': [1, 2, 2, 3], 'B': [4, 3, 2, 1]}, columns= ['A', 'B'])
        >>> df
           A  B
        0  1  4
        1  2  3
        2  2  2
        3  3  1

        >>> df.rank().sort_index()
             A    B
        0  1.0  4.0
        1  2.5  3.0
        2  2.5  2.0
        3  4.0  1.0

        If method is set to 'min', it use lowest rank in group.

        >>> df.rank(method='min').sort_index()
             A    B
        0  1.0  4.0
        1  2.0  3.0
        2  2.0  2.0
        3  4.0  1.0

        If method is set to 'max', it use highest rank in group.

        >>> df.rank(method='max').sort_index()
             A    B
        0  1.0  4.0
        1  3.0  3.0
        2  3.0  2.0
        3  4.0  1.0

        If method is set to 'dense', it leaves no gaps in group.

        >>> df.rank(method='dense').sort_index()
             A    B
        0  1.0  4.0
        1  2.0  3.0
        2  2.0  2.0
        3  3.0  1.0
        """
        return self._apply_series_op(lambda kser: kser._rank(method=method,
            ascending=ascending), should_resolve=True)

    def filter(self, items=None, like=None, regex=None, axis=None):
        """
        Subset rows or columns of dataframe according to labels in
        the specified index.

        Note that this routine does not filter a dataframe on its
        contents. The filter is applied to the labels of the index.

        Parameters
        ----------
        items : list-like
            Keep labels from axis which are in items.
        like : string
            Keep labels from axis for which "like in label == True".
        regex : string (regular expression)
            Keep labels from axis for which re.search(regex, label) == True.
        axis : int or string axis name
            The axis to filter on.  By default this is the info axis,
            'index' for Series, 'columns' for DataFrame.

        Returns
        -------
        same type as input object

        See Also
        --------
        DataFrame.loc

        Notes
        -----
        The ``items``, ``like``, and ``regex`` parameters are
        enforced to be mutually exclusive.

        ``axis`` defaults to the info axis that is used when indexing
        with ``[]``.

        Examples
        --------
        >>> df = ks.DataFrame(np.array(([1, 2, 3], [4, 5, 6])),
        ...                   index=['mouse', 'rabbit'],
        ...                   columns=['one', 'two', 'three'])

        >>> # select columns by name
        >>> df.filter(items=['one', 'three'])
                one  three
        mouse     1      3
        rabbit    4      6

        >>> # select columns by regular expression
        >>> df.filter(regex='e$', axis=1)
                one  three
        mouse     1      3
        rabbit    4      6

        >>> # select rows containing 'bbi'
        >>> df.filter(like='bbi', axis=0)
                one  two  three
        rabbit    4    5      6

        For a Series,

        >>> # select rows by name
        >>> df.one.filter(items=['rabbit'])
        rabbit    4
        Name: one, dtype: int64

        >>> # select rows by regular expression
        >>> df.one.filter(regex='e$')
        mouse    1
        Name: one, dtype: int64

        >>> # select rows containing 'bbi'
        >>> df.one.filter(like='bbi')
        rabbit    4
        Name: one, dtype: int64
        """
        if sum(x is not None for x in (items, like, regex)) > 1:
            raise TypeError(
                'Keyword arguments `items`, `like`, or `regex` are mutually exclusive'
                )
        axis = validate_axis(axis, none_axis=1)
        index_scols = self._internal.index_spark_columns
        if items is not None:
            if is_list_like(items):
                items = list(items)
            else:
                raise ValueError('items should be a list-like object.')
            if axis == 0:
                if len(index_scols) == 1:
                    col = None
                    for item in items:
                        if col is None:
                            col = index_scols[0] == F.lit(item)
                        else:
                            col = col | (index_scols[0] == F.lit(item))
                elif len(index_scols) > 1:
                    col = None
                    for item in items:
                        if not isinstance(item, tuple):
                            raise TypeError('Unsupported type {}'.format(
                                type(item).__name__))
                        if not item:
                            raise ValueError('The item should not be empty.')
                        midx_col = None
                        for i, element in enumerate(item):
                            if midx_col is None:
                                midx_col = index_scols[i] == F.lit(element)
                            else:
                                midx_col = midx_col & (index_scols[i] == F.
                                    lit(element))
                        if col is None:
                            col = midx_col
                        else:
                            col = col | midx_col
                else:
                    raise ValueError('Single or multi index must be specified.'
                        )
                return DataFrame(self._internal.with_filter(col))
            else:
                return self[items]
        elif like is not None:
            if axis == 0:
                col = None
                for index_scol in index_scols:
                    if col is None:
                        col = index_scol.contains(like)
                    else:
                        col = col | index_scol.contains(like)
                return DataFrame(self._internal.with_filter(col))
            else:
                column_labels = self._internal.column_labels
                output_labels = [label for label in column_labels if any(
                    like in i for i in label)]
                return self[output_labels]
        elif regex is not None:
            if axis == 0:
                col = None
                for index_scol in index_scols:
                    if col is None:
                        col = index_scol.rlike(regex)
                    else:
                        col = col | index_scol.rlike(regex)
                return DataFrame(self._internal.with_filter(col))
            else:
                column_labels = self._internal.column_labels
                matcher = re.compile(regex)
                output_labels = [label for label in column_labels if any(
                    matcher.search(i) is not None for i in label)]
                return self[output_labels]
        else:
            raise TypeError('Must pass either `items`, `like`, or `regex`')

    def rename(self, mapper=None, index=None, columns=None, axis='index',
        inplace=False, level=None, errors='ignore'):
        """
        Alter axes labels.
        Function / dict values must be unique (1-to-1). Labels not contained in a dict / Series
        will be left as-is. Extra labels listed don’t throw an error.

        Parameters
        ----------
        mapper : dict-like or function
            Dict-like or functions transformations to apply to that axis’ values.
            Use either `mapper` and `axis` to specify the axis to target with `mapper`, or `index`
            and `columns`.
        index : dict-like or function
            Alternative to specifying axis ("mapper, axis=0" is equivalent to "index=mapper").
        columns : dict-like or function
            Alternative to specifying axis ("mapper, axis=1" is equivalent to "columns=mapper").
        axis : int or str, default 'index'
            Axis to target with mapper. Can be either the axis name ('index', 'columns') or
            number (0, 1).
        inplace : bool, default False
            Whether to return a new DataFrame.
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified level.
        errors : {'ignore', 'raise}, default 'ignore'
            If 'raise', raise a `KeyError` when a dict-like `mapper`, `index`, or `columns`
            contains labels that are not present in the Index being transformed. If 'ignore',
            existing keys will be renamed and extra keys will be ignored.

        Returns
        -------
        DataFrame with the renamed axis labels.

        Raises
        ------
        `KeyError`
            If any of the labels is not found in the selected axis and "errors='raise'".

        Examples
        --------
        >>> kdf1 = ks.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> kdf1.rename(columns={"A": "a", "B": "c"})  # doctest: +NORMALIZE_WHITESPACE
           a  c
        0  1  4
        1  2  5
        2  3  6

        >>> kdf1.rename(index={1: 10, 2: 20})  # doctest: +NORMALIZE_WHITESPACE
            A  B
        0   1  4
        10  2  5
        20  3  6

        >>> def str_lower(s) -> str:
        ...     return str.lower(s)
        >>> kdf1.rename(str_lower, axis='columns')  # doctest: +NORMALIZE_WHITESPACE
           a  b
        0  1  4
        1  2  5
        2  3  6

        >>> def mul10(x) -> int:
        ...     return x * 10
        >>> kdf1.rename(mul10, axis='index')  # doctest: +NORMALIZE_WHITESPACE
            A  B
        0   1  4
        10  2  5
        20  3  6

        >>> idx = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Y', 'D')])
        >>> kdf2 = ks.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=idx)
        >>> kdf2.rename(columns=str_lower, level=0)  # doctest: +NORMALIZE_WHITESPACE
           x     y
           A  B  C  D
        0  1  2  3  4
        1  5  6  7  8

        >>> kdf3 = ks.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=idx, columns=list('ab'))
        >>> kdf3.rename(index=str_lower)  # doctest: +NORMALIZE_WHITESPACE
             a  b
        x a  1  2
          b  3  4
        y c  5  6
          d  7  8
        """

        def gen_mapper_fn(mapper):
            if isinstance(mapper, dict):
                if len(mapper) == 0:
                    if errors == 'raise':
                        raise KeyError(
                            'Index include label which is not in the `mapper`.'
                            )
                    else:
                        return DataFrame(self._internal)
                type_set = set(map(lambda x: type(x), mapper.values()))
                if len(type_set) > 1:
                    raise ValueError(
                        'Mapper dict should have the same value type.')
                spark_return_type = as_spark_type(list(type_set)[0])

                def mapper_fn(x):
                    if x in mapper:
                        return mapper[x]
                    else:
                        if errors == 'raise':
                            raise KeyError(
                                'Index include value which is not in the `mapper`'
                                )
                        return x
            elif callable(mapper):
                spark_return_type = cast(ScalarType, infer_return_type(mapper)
                    ).spark_type

                def mapper_fn(x):
                    return mapper(x)
            else:
                raise ValueError(
                    '`mapper` or `index` or `columns` should be either dict-like or function type.'
                    )
            return mapper_fn, spark_return_type
        index_mapper_fn = None
        index_mapper_ret_stype = None
        columns_mapper_fn = None
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if mapper:
            axis = validate_axis(axis)
            if axis == 0:
                index_mapper_fn, index_mapper_ret_stype = gen_mapper_fn(mapper)
            elif axis == 1:
                columns_mapper_fn, columns_mapper_ret_stype = gen_mapper_fn(
                    mapper)
            else:
                raise ValueError(
                    'argument axis should be either the axis name (‘index’, ‘columns’) or number (0, 1)'
                    )
        else:
            if index:
                index_mapper_fn, index_mapper_ret_stype = gen_mapper_fn(index)
            if columns:
                columns_mapper_fn, _ = gen_mapper_fn(columns)
            if not index and not columns:
                raise ValueError(
                    'Either `index` or `columns` should be provided.')
        kdf = self.copy()
        if index_mapper_fn:
            index_columns = kdf._internal.index_spark_column_names
            num_indices = len(index_columns)
            if level:
                if level < 0 or level >= num_indices:
                    raise ValueError(
                        'level should be an integer between [0, num_indices)')

            def gen_new_index_column(level):
                index_col_name = index_columns[level]
                index_mapper_udf = pandas_udf(lambda s: s.map(
                    index_mapper_fn), returnType=index_mapper_ret_stype)
                return index_mapper_udf(scol_for(kdf._internal.spark_frame,
                    index_col_name))
            sdf = kdf._internal.resolved_copy.spark_frame
            index_dtypes = self._internal.index_dtypes.copy()
            if level is None:
                for i in range(num_indices):
                    sdf = sdf.withColumn(index_columns[i],
                        gen_new_index_column(i))
                    index_dtypes[i] = None
            else:
                sdf = sdf.withColumn(index_columns[level],
                    gen_new_index_column(level))
                index_dtypes[level] = None
            kdf = DataFrame(kdf._internal.with_new_sdf(sdf, index_dtypes=
                index_dtypes))
        if columns_mapper_fn:
            if level:
                if level < 0 or level >= kdf._internal.column_labels_level:
                    raise ValueError(
                        'level should be an integer between [0, column_labels_level)'
                        )

            def gen_new_column_labels_entry(column_labels_entry):
                if isinstance(column_labels_entry, tuple):
                    if level is None:
                        return tuple(map(columns_mapper_fn,
                            column_labels_entry))
                    else:
                        entry_list = list(column_labels_entry)
                        entry_list[level] = columns_mapper_fn(entry_list[level]
                            )
                        return tuple(entry_list)
                else:
                    return columns_mapper_fn(column_labels_entry)
            new_column_labels = list(map(gen_new_column_labels_entry, kdf.
                _internal.column_labels))
            new_data_scols = [kdf._kser_for(old_label).rename(new_label) for
                old_label, new_label in zip(kdf._internal.column_labels,
                new_column_labels)]
            kdf = DataFrame(kdf._internal.with_new_columns(new_data_scols))
        if inplace:
            self._update_internal_frame(kdf._internal)
            return None
        else:
            return kdf

    def rename_axis(self, mapper=None, index=None, columns=None, axis=0,
        inplace=False):
        """
        Set the name of the axis for the index or columns.

        Parameters
        ----------
        mapper : scalar, list-like, optional
            A scalar, list-like, dict-like or functions transformations to
            apply to the axis name attribute.
        index, columns : scalar, list-like, dict-like or function, optional
            A scalar, list-like, dict-like or functions transformations to
            apply to that axis' values.

            Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index``
            and/or ``columns``.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to rename.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new DataFrame.

        Returns
        -------
        DataFrame, or None if `inplace` is True.

        See Also
        --------
        Series.rename : Alter Series index labels or name.
        DataFrame.rename : Alter DataFrame index labels or name.
        Index.rename : Set new names on index.

        Notes
        -----
        ``DataFrame.rename_axis`` supports two calling conventions

        * ``(index=index_mapper, columns=columns_mapper, ...)``
        * ``(mapper, axis={'index', 'columns'}, ...)``

        The first calling convention will only modify the names of
        the index and/or the names of the Index object that is the columns.

        The second calling convention will modify the names of the
        corresponding index specified by axis.

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Examples
        --------
        >>> df = ks.DataFrame({"num_legs": [4, 4, 2],
        ...                    "num_arms": [0, 0, 2]},
        ...                   index=["dog", "cat", "monkey"],
        ...                   columns=["num_legs", "num_arms"])
        >>> df
                num_legs  num_arms
        dog            4         0
        cat            4         0
        monkey         2         2

        >>> df = df.rename_axis("animal").sort_index()
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                num_legs  num_arms
        animal
        cat            4         0
        dog            4         0
        monkey         2         2

        >>> df = df.rename_axis("limbs", axis="columns").sort_index()
        >>> df # doctest: +NORMALIZE_WHITESPACE
        limbs   num_legs  num_arms
        animal
        cat            4         0
        dog            4         0
        monkey         2         2

        **MultiIndex**

        >>> index = pd.MultiIndex.from_product([['mammal'],
        ...                                     ['dog', 'cat', 'monkey']],
        ...                                    names=['type', 'name'])
        >>> df = ks.DataFrame({"num_legs": [4, 4, 2],
        ...                    "num_arms": [0, 0, 2]},
        ...                   index=index,
        ...                   columns=["num_legs", "num_arms"])
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                       num_legs  num_arms
        type   name
        mammal dog            4         0
               cat            4         0
               monkey         2         2

        >>> df.rename_axis(index={'type': 'class'}).sort_index()  # doctest: +NORMALIZE_WHITESPACE
                       num_legs  num_arms
        class  name
        mammal cat            4         0
               dog            4         0
               monkey         2         2

        >>> df.rename_axis(index=str.upper).sort_index()  # doctest: +NORMALIZE_WHITESPACE
                       num_legs  num_arms
        TYPE   NAME
        mammal cat            4         0
               dog            4         0
               monkey         2         2
        """

        def gen_names(v, curnames):
            if is_scalar(v):
                newnames = [v]
            elif is_list_like(v) and not is_dict_like(v):
                newnames = list(v)
            elif is_dict_like(v):
                newnames = [(v[name] if name in v else name) for name in
                    curnames]
            elif callable(v):
                newnames = [v(name) for name in curnames]
            else:
                raise ValueError(
                    '`mapper` or `index` or `columns` should be either dict-like or function type.'
                    )
            if len(newnames) != len(curnames):
                raise ValueError('Length of new names must be {}, got {}'.
                    format(len(curnames), len(newnames)))
            return [(name if is_name_like_tuple(name) else (name,)) for
                name in newnames]
        if mapper is not None and (index is not None or columns is not None):
            raise TypeError(
                "Cannot specify both 'mapper' and any of 'index' or 'columns'."
                )
        if mapper is not None:
            axis = validate_axis(axis)
            if axis == 0:
                index = mapper
            elif axis == 1:
                columns = mapper
        column_label_names = gen_names(columns, self.columns.names
            ) if columns is not None else self._internal.column_label_names
        index_names = gen_names(index, self.index.names
            ) if index is not None else self._internal.index_names
        internal = self._internal.copy(index_names=index_names,
            column_label_names=column_label_names)
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def keys(self):
        """
        Return alias for columns.

        Returns
        -------
        Index
            Columns of the DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', 'sidewinder'],
        ...                   columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8

        >>> df.keys()
        Index(['max_speed', 'shield'], dtype='object')
        """
        return self.columns

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
        DataFrame

        Examples
        --------
        Percentage change in French franc, Deutsche Mark, and Italian lira
        from 1980-01-01 to 1980-03-01.

        >>> df = ks.DataFrame({
        ...     'FR': [4.0405, 4.0963, 4.3149],
        ...     'GR': [1.7246, 1.7482, 1.8519],
        ...     'IT': [804.74, 810.01, 860.13]},
        ...     index=['1980-01-01', '1980-02-01', '1980-03-01'])
        >>> df
                        FR      GR      IT
        1980-01-01  4.0405  1.7246  804.74
        1980-02-01  4.0963  1.7482  810.01
        1980-03-01  4.3149  1.8519  860.13

        >>> df.pct_change()
                          FR        GR        IT
        1980-01-01       NaN       NaN       NaN
        1980-02-01  0.013810  0.013684  0.006549
        1980-03-01  0.053365  0.059318  0.061876

        You can set periods to shift for forming percent change

        >>> df.pct_change(2)
                          FR        GR       IT
        1980-01-01       NaN       NaN      NaN
        1980-02-01       NaN       NaN      NaN
        1980-03-01  0.067912  0.073814  0.06883
        """
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-
            periods, -periods)

        def op(kser):
            prev_row = F.lag(kser.spark.column, periods).over(window)
            return ((kser.spark.column - prev_row) / prev_row).alias(kser.
                _internal.data_spark_column_names[0])
        return self._apply_series_op(op, should_resolve=True)

    def idxmax(self, axis=0):
        """
        Return index of first occurrence of maximum over requested axis.
        NA/null values are excluded.

        .. note:: This API collect all rows with maximum value using `to_pandas()`
            because we suppose the number of rows with max values are usually small in general.

        Parameters
        ----------
        axis : 0 or 'index'
            Can only be set to 0 at the moment.

        Returns
        -------
        Series

        See Also
        --------
        Series.idxmax

        Examples
        --------
        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf
           a    b    c
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmax()
        a    2
        b    0
        c    2
        dtype: int64

        For Multi-column Index

        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
           a    b    c
           x    y    z
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmax()
        a  x    2
        b  y    0
        c  z    2
        dtype: int64
        """
        max_cols = map(lambda scol: F.max(scol), self._internal.
            data_spark_columns)
        sdf_max = self._internal.spark_frame.select(*max_cols).head()
        conds = (scol == max_val for scol, max_val in zip(self._internal.
            data_spark_columns, sdf_max))
        cond = reduce(lambda x, y: x | y, conds)
        kdf = DataFrame(self._internal.with_filter(cond))
        return cast(ks.Series, ks.from_pandas(kdf._to_internal_pandas().
            idxmax()))

    def idxmin(self, axis=0):
        """
        Return index of first occurrence of minimum over requested axis.
        NA/null values are excluded.

        .. note:: This API collect all rows with minimum value using `to_pandas()`
            because we suppose the number of rows with min values are usually small in general.

        Parameters
        ----------
        axis : 0 or 'index'
            Can only be set to 0 at the moment.

        Returns
        -------
        Series

        See Also
        --------
        Series.idxmin

        Examples
        --------
        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf
           a    b    c
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmin()
        a    0
        b    3
        c    1
        dtype: int64

        For Multi-column Index

        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
           a    b    c
           x    y    z
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmin()
        a  x    0
        b  y    3
        c  z    1
        dtype: int64
        """
        min_cols = map(lambda scol: F.min(scol), self._internal.
            data_spark_columns)
        sdf_min = self._internal.spark_frame.select(*min_cols).head()
        conds = (scol == min_val for scol, min_val in zip(self._internal.
            data_spark_columns, sdf_min))
        cond = reduce(lambda x, y: x | y, conds)
        kdf = DataFrame(self._internal.with_filter(cond))
        return cast(ks.Series, ks.from_pandas(kdf._to_internal_pandas().
            idxmin()))

    def info(self, verbose=None, buf=None, max_cols=None, null_counts=None):
        """
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and column dtypes, non-null values and memory usage.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        max_cols : int, optional
            When to switch from the verbose to the truncated output. If the
            DataFrame has more than `max_cols` columns, the truncated output
            is used.
        null_counts : bool, optional
            Whether to show the non-null counts.

        Returns
        -------
        None
            This method prints a summary of a DataFrame and returns None.

        See Also
        --------
        DataFrame.describe: Generate descriptive statistics of DataFrame
            columns.

        Examples
        --------
        >>> int_values = [1, 2, 3, 4, 5]
        >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> df = ks.DataFrame(
        ...     {"int_col": int_values, "text_col": text_values, "float_col": float_values},
        ...     columns=['int_col', 'text_col', 'float_col'])
        >>> df
           int_col text_col  float_col
        0        1    alpha       0.00
        1        2     beta       0.25
        2        3    gamma       0.50
        3        4    delta       0.75
        4        5  epsilon       1.00

        Prints information of all columns:

        >>> df.info(verbose=True)  # doctest: +SKIP
        <class 'databricks.koalas.frame.DataFrame'>
        Index: 5 entries, 0 to 4
        Data columns (total 3 columns):
         #   Column     Non-Null Count  Dtype
        ---  ------     --------------  -----
         0   int_col    5 non-null      int64
         1   text_col   5 non-null      object
         2   float_col  5 non-null      float64
        dtypes: float64(1), int64(1), object(1)

        Prints a summary of columns count and its dtypes but not per column
        information:

        >>> df.info(verbose=False)  # doctest: +SKIP
        <class 'databricks.koalas.frame.DataFrame'>
        Index: 5 entries, 0 to 4
        Columns: 3 entries, int_col to float_col
        dtypes: float64(1), int64(1), object(1)

        Pipe output of DataFrame.info to buffer instead of sys.stdout, get
        buffer content and writes to a text file:

        >>> import io
        >>> buffer = io.StringIO()
        >>> df.info(buf=buffer)
        >>> s = buffer.getvalue()
        >>> with open('%s/info.txt' % path, "w",
        ...           encoding="utf-8") as f:
        ...     _ = f.write(s)
        >>> with open('%s/info.txt' % path) as f:
        ...     f.readlines()  # doctest: +SKIP
        ["<class 'databricks.koalas.frame.DataFrame'>\\n",
        'Index: 5 entries, 0 to 4\\n',
        'Data columns (total 3 columns):\\n',
        ' #   Column     Non-Null Count  Dtype  \\n',
        '---  ------     --------------  -----  \\n',
        ' 0   int_col    5 non-null      int64  \\n',
        ' 1   text_col   5 non-null      object \\n',
        ' 2   float_col  5 non-null      float64\\n',
        'dtypes: float64(1), int64(1), object(1)']
        """
        with pd.option_context('display.max_info_columns', sys.maxsize,
            'display.max_info_rows', sys.maxsize):
            try:
                object.__setattr__(self, '_data', self)
                count_func = self.count
                self.count = lambda : count_func().to_pandas()
                return pd.DataFrame.info(self, verbose=verbose, buf=buf,
                    max_cols=max_cols, memory_usage=False, null_counts=
                    null_counts)
            finally:
                del self._data
                self.count = count_func

    def quantile(self, q=0.5, axis=0, numeric_only=True, accuracy=10000):
        """
        Return value at the given quantile.

        .. note:: Unlike pandas', the quantile in Koalas is an approximated quantile based upon
            approximate percentile computation because computing quantile across a large dataset
            is extremely expensive.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute.
        axis : int or str, default 0 or 'index'
            Can only be set to 0 at the moment.
        numeric_only : bool, default True
            If False, the quantile of datetime and timedelta data will be computed as well.
            Can only be set to True at the moment.
        accuracy : int, optional
            Default accuracy of approximation. Larger value means better accuracy.
            The relative error can be deduced by 1.0 / accuracy.

        Returns
        -------
        Series or DataFrame
            If q is an array, a DataFrame will be returned where the
            index is q, the columns are the columns of self, and the values are the quantiles.
            If q is a float, a Series will be returned where the
            index is the columns of self and the values are the quantiles.

        Examples
        --------
        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 0]})
        >>> kdf
           a  b
        0  1  6
        1  2  7
        2  3  8
        3  4  9
        4  5  0

        >>> kdf.quantile(.5)
        a    3.0
        b    7.0
        Name: 0.5, dtype: float64

        >>> kdf.quantile([.25, .5, .75])
                a    b
        0.25  2.0  6.0
        0.50  3.0  7.0
        0.75  4.0  8.0
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError(
                'axis should be either 0 or "index" currently.')
        if not isinstance(accuracy, int):
            raise ValueError(
                'accuracy must be an integer; however, got [%s]' % type(
                accuracy).__name__)
        if isinstance(q, Iterable):
            q = list(q)
        for v in (q if isinstance(q, list) else [q]):
            if not isinstance(v, float):
                raise ValueError(
                    'q must be a float or an array of floats; however, [%s] found.'
                     % type(v))
            if v < 0.0 or v > 1.0:
                raise ValueError(
                    'percentiles should all be in the interval [0, 1].')

        def quantile(spark_column, spark_type):
            if isinstance(spark_type, (BooleanType, NumericType)):
                return SF.percentile_approx(spark_column.cast(DoubleType()),
                    q, accuracy)
            else:
                raise TypeError('Could not convert {} ({}) to numeric'.
                    format(spark_type_to_pandas_dtype(spark_type),
                    spark_type.simpleString()))
        if isinstance(q, list):
            percentile_cols = []
            percentile_col_names = []
            column_labels = []
            for label, column in zip(self._internal.column_labels, self.
                _internal.data_spark_column_names):
                spark_type = self._internal.spark_type_for(label)
                is_numeric_or_boolean = isinstance(spark_type, (NumericType,
                    BooleanType))
                keep_column = not numeric_only or is_numeric_or_boolean
                if keep_column:
                    percentile_col = quantile(self._internal.
                        spark_column_for(label), spark_type)
                    percentile_cols.append(percentile_col.alias(column))
                    percentile_col_names.append(column)
                    column_labels.append(label)
            if len(percentile_cols) == 0:
                return DataFrame(index=q)
            sdf = self._internal.spark_frame.select(percentile_cols)
            cols_dict = OrderedDict()
            for column in percentile_col_names:
                cols_dict[column] = list()
                for i in range(len(q)):
                    cols_dict[column].append(scol_for(sdf, column).getItem(
                        i).alias(column))
            internal_index_column = SPARK_DEFAULT_INDEX_NAME
            cols = []
            for i, col in enumerate(zip(*cols_dict.values())):
                cols.append(F.struct(F.lit(q[i]).alias(
                    internal_index_column), *col))
            sdf = sdf.select(F.array(*cols).alias('arrays'))
            sdf = sdf.select(F.explode(F.col('arrays'))).selectExpr('col.*')
            internal = InternalFrame(spark_frame=sdf, index_spark_columns=[
                scol_for(sdf, internal_index_column)], column_labels=
                column_labels, data_spark_columns=[scol_for(sdf, col) for
                col in percentile_col_names])
            return DataFrame(internal)
        else:
            return self._reduce_for_stat_function(quantile, name='quantile',
                numeric_only=numeric_only).rename(q)

    def query(self, expr, inplace=False):
        """
        Query the columns of a DataFrame with a boolean expression.

        .. note:: Internal columns that starting with a '__' prefix are able to access, however,
            they are not supposed to be accessed.

        .. note:: This API delegates to Spark SQL so the syntax follows Spark SQL. Therefore, the
            pandas specific syntax such as `@` is not supported. If you want the pandas syntax,
            you can work around with :meth:`DataFrame.koalas.apply_batch`, but you should
            be aware that `query_func` will be executed at different nodes in a distributed manner.
            So, for example, to use `@` syntax, make sure the variable is serialized by, for
            example, putting it within the closure as below.

            >>> df = ks.DataFrame({'A': range(2000), 'B': range(2000)})
            >>> def query_func(pdf):
            ...     num = 1995
            ...     return pdf.query('A > @num')
            >>> df.koalas.apply_batch(query_func)
                     A     B
            1996  1996  1996
            1997  1997  1997
            1998  1998  1998
            1999  1999  1999

        Parameters
        ----------
        expr : str
            The query string to evaluate.

            You can refer to column names that contain spaces by surrounding
            them in backticks.

            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.

        inplace : bool
            Whether the query should modify the data in place or return
            a modified copy.

        Returns
        -------
        DataFrame
            DataFrame resulting from the provided query expression.

        Examples
        --------
        >>> df = ks.DataFrame({'A': range(1, 6),
        ...                    'B': range(10, 0, -2),
        ...                    'C C': range(10, 5, -1)})
        >>> df
           A   B  C C
        0  1  10   10
        1  2   8    9
        2  3   6    8
        3  4   4    7
        4  5   2    6

        >>> df.query('A > B')
           A  B  C C
        4  5  2    6

        The previous expression is equivalent to

        >>> df[df.A > df.B]
           A  B  C C
        4  5  2    6

        For columns with spaces in their name, you can use backtick quoting.

        >>> df.query('B == `C C`')
           A   B  C C
        0  1  10   10

        The previous expression is equivalent to

        >>> df[df.B == df['C C']]
           A   B  C C
        0  1  10   10
        """
        if isinstance(self.columns, pd.MultiIndex):
            raise ValueError("Doesn't support for MultiIndex columns")
        if not isinstance(expr, str):
            raise ValueError('expr must be a string to be evaluated, {} given'
                .format(type(expr).__name__))
        inplace = validate_bool_kwarg(inplace, 'inplace')
        data_columns = [label[0] for label in self._internal.column_labels]
        sdf = self._internal.spark_frame.select(self._internal.
            index_spark_columns + [scol.alias(col) for scol, col in zip(
            self._internal.data_spark_columns, data_columns)]).filter(expr)
        internal = self._internal.with_new_sdf(sdf, data_columns=data_columns)
        if inplace:
            self._update_internal_frame(internal)
            return None
        else:
            return DataFrame(internal)

    def explain(self, extended=None, mode=None):
        warnings.warn(
            'DataFrame.explain is deprecated as of DataFrame.spark.explain. Please use the API instead.'
            , FutureWarning)
        return self.spark.explain(extended, mode)
    explain.__doc__ = SparkFrameMethods.explain.__doc__

    def take(self, indices, axis=0, **kwargs):
        """
        Return the elements in the given *positional* indices along an axis.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            The axis on which to select elements. ``0`` means that we are
            selecting rows, ``1`` means that we are selecting columns.
        **kwargs
            For compatibility with :meth:`numpy.take`. Has no effect on the
            output.

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
        >>> df = ks.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey', 'mammal', np.nan)],
        ...                   columns=['name', 'class', 'max_speed'],
        ...                   index=[0, 2, 3, 1])
        >>> df
             name   class  max_speed
        0  falcon    bird      389.0
        2  parrot    bird       24.0
        3    lion  mammal       80.5
        1  monkey  mammal        NaN

        Take elements at positions 0 and 3 along the axis 0 (default).

        Note how the actual indices selected (0 and 1) do not correspond to
        our selected indices 0 and 3. That's because we are selecting the 0th
        and 3rd rows, not rows whose indices equal 0 and 3.

        >>> df.take([0, 3]).sort_index()
             name   class  max_speed
        0  falcon    bird      389.0
        1  monkey  mammal        NaN

        Take elements at indices 1 and 2 along the axis 1 (column selection).

        >>> df.take([1, 2], axis=1)
            class  max_speed
        0    bird      389.0
        2    bird       24.0
        3  mammal       80.5
        1  mammal        NaN

        We may take elements using negative integers for positive indices,
        starting from the end of the object, just like with Python lists.

        >>> df.take([-1, -2]).sort_index()
             name   class  max_speed
        1  monkey  mammal        NaN
        3    lion  mammal       80.5
        """
        axis = validate_axis(axis)
        if not is_list_like(indices) or isinstance(indices, (dict, set)):
            raise ValueError('`indices` must be a list-like except dict or set'
                )
        if axis == 0:
            return cast(DataFrame, self.iloc[indices, :])
        else:
            return cast(DataFrame, self.iloc[:, indices])

    def eval(self, expr, inplace=False):
        """
        Evaluate a string describing operations on DataFrame columns.

        Operates on columns only, not specific rows or elements. This allows
        `eval` to run arbitrary code, which can make you vulnerable to code
        injection if you pass user input to this function.

        Parameters
        ----------
        expr : str
            The expression string to evaluate.
        inplace : bool, default False
            If the expression contains an assignment, whether to perform the
            operation inplace and mutate the existing DataFrame. Otherwise,
            a new DataFrame is returned.

        Returns
        -------
        The result of the evaluation.

        See Also
        --------
        DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
        DataFrame.assign : Can evaluate an expression or function to create new
            values for a column.
        eval : Evaluate a Python expression as a string using various
            backends.

        Examples
        --------
        >>> df = ks.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2)})
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2
        >>> df.eval('A + B')
        0    11
        1    10
        2     9
        3     8
        4     7
        dtype: int64

        Assignment is allowed though by default the original DataFrame is not
        modified.

        >>> df.eval('C = A + B')
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2

        Use ``inplace=True`` to modify the original DataFrame.

        >>> df.eval('C = A + B', inplace=True)
        >>> df
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7
        """
        from databricks.koalas.series import first_series
        if isinstance(self.columns, pd.MultiIndex):
            raise ValueError('`eval` is not supported for multi-index columns')
        inplace = validate_bool_kwarg(inplace, 'inplace')
        should_return_series = False
        series_name = None
        should_return_scalar = False

        def eval_func(pdf):
            nonlocal should_return_series
            nonlocal series_name
            nonlocal should_return_scalar
            result_inner = pdf.eval(expr, inplace=inplace)
            if inplace:
                result_inner = pdf
            if isinstance(result_inner, pd.Series):
                should_return_series = True
                series_name = result_inner.name
                result_inner = result_inner.to_frame()
            elif is_scalar(result_inner):
                should_return_scalar = True
                result_inner = pd.Series(result_inner).to_frame()
            return result_inner
        result = self.koalas.apply_batch(eval_func)
        if inplace:
            self._update_internal_frame(result._internal,
                requires_same_anchor=False)
            return None
        elif should_return_series:
            return first_series(result).rename(series_name)
        elif should_return_scalar:
            return first_series(result)[0]
        else:
            return result

    def explode(self, column):
        """
        Transform each element of a list-like to a row, replicating index values.

        Parameters
        ----------
        column : str or tuple
            Column to explode.

        Returns
        -------
        DataFrame
            Exploded lists to rows of the subset columns;
            index will be duplicated for these rows.

        See Also
        --------
        DataFrame.unstack : Pivot a level of the (necessarily hierarchical)
            index labels.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [[1, 2, 3], [], [3, 4]], 'B': 1})
        >>> df
                   A  B
        0  [1, 2, 3]  1
        1         []  1
        2     [3, 4]  1

        >>> df.explode('A')
             A  B
        0  1.0  1
        0  2.0  1
        0  3.0  1
        1  NaN  1
        2  3.0  1
        2  4.0  1
        """
        from databricks.koalas.series import Series
        if not is_name_like_value(column):
            raise ValueError('column must be a scalar')
        kdf = DataFrame(self._internal.resolved_copy)
        kser = kdf[column]
        if not isinstance(kser, Series):
            raise ValueError(
                'The column %s is not unique. For a multi-index, the label must be a tuple with elements corresponding to each level.'
                 % name_like_string(column))
        if not isinstance(kser.spark.data_type, ArrayType):
            return self.copy()
        sdf = kdf._internal.spark_frame.withColumn(kser._internal.
            data_spark_column_names[0], F.explode_outer(kser.spark.column))
        data_dtypes = kdf._internal.data_dtypes.copy()
        data_dtypes[kdf._internal.column_labels.index(kser._column_label)
            ] = None
        internal = kdf._internal.with_new_sdf(sdf, data_dtypes=data_dtypes)
        return DataFrame(internal)

    def mad(self, axis=0):
        """
        Return the mean absolute deviation of values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        >>> df.mad()
        a    0.666667
        b    0.066667
        dtype: float64

        >>> df.mad(axis=1)
        0    0.45
        1    0.90
        2    1.35
        3     NaN
        dtype: float64
        """
        from databricks.koalas.series import first_series
        axis = validate_axis(axis)
        if axis == 0:

            def get_spark_column(kdf, label):
                scol = kdf._internal.spark_column_for(label)
                col_type = kdf._internal.spark_type_for(label)
                if isinstance(col_type, BooleanType):
                    scol = scol.cast('integer')
                return scol
            new_column_labels = []
            for label in self._internal.column_labels:
                dtype = self._kser_for(label).spark.data_type
                if isinstance(dtype, (NumericType, BooleanType)):
                    new_column_labels.append(label)
            new_columns = [F.avg(get_spark_column(self, label)).alias(
                name_like_string(label)) for label in new_column_labels]
            mean_data = self._internal.spark_frame.select(new_columns).first()
            new_columns = [F.avg(F.abs(get_spark_column(self, label) -
                mean_data[name_like_string(label)])).alias(name_like_string
                (label)) for label in new_column_labels]
            sdf = self._internal.spark_frame.select([F.lit(None).cast(
                StringType()).alias(SPARK_DEFAULT_INDEX_NAME)] + new_columns)
            with ks.option_context('compute.max_rows', 1):
                internal = InternalFrame(spark_frame=sdf,
                    index_spark_columns=[scol_for(sdf,
                    SPARK_DEFAULT_INDEX_NAME)], column_labels=
                    new_column_labels, column_label_names=self._internal.
                    column_label_names)
                return first_series(DataFrame(internal).transpose())
        else:

            @pandas_udf(returnType=DoubleType())
            def calculate_columns_axis(*cols):
                return pd.concat(cols, axis=1).mad(axis=1)
            internal = self._internal.copy(column_labels=[None],
                data_spark_columns=[calculate_columns_axis(*self._internal.
                data_spark_columns).alias(SPARK_DEFAULT_SERIES_NAME)],
                data_dtypes=[None], column_label_names=None)
            return first_series(DataFrame(internal))

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
        >>> df = ks.DataFrame({'animal': ['alligator', 'bee', 'falcon', 'lion',
        ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the last 5 lines

        >>> df.tail()  # doctest: +SKIP
           animal
        4  monkey
        5  parrot
        6   shark
        7   whale
        8   zebra

        Viewing the last `n` lines (three in this case)

        >>> df.tail(3)  # doctest: +SKIP
          animal
        6  shark
        7  whale
        8  zebra

        For negative values of `n`

        >>> df.tail(-3)  # doctest: +SKIP
           animal
        3    lion
        4  monkey
        5  parrot
        6   shark
        7   whale
        8   zebra
        """
        if LooseVersion(pyspark.__version__) < LooseVersion('3.0'):
            raise RuntimeError('tail can be used in PySpark >= 3.0')
        if not isinstance(n, int):
            raise TypeError("bad operand type for unary -: '{}'".format(
                type(n).__name__))
        if n < 0:
            n = len(self) + n
        if n <= 0:
            return ks.DataFrame(self._internal.with_filter(F.lit(False)))
        sdf = self._internal.resolved_copy.spark_frame
        rows = sdf.tail(n)
        new_sdf = default_session().createDataFrame(rows, sdf.schema)
        return DataFrame(self._internal.with_new_sdf(new_sdf))

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
        (left, right) : (DataFrame, type of other)
            Aligned objects.

        Examples
        --------
        >>> ks.set_option("compute.ops_on_diff_frames", True)
        >>> df1 = ks.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}, index=[10, 20, 30])
        >>> df2 = ks.DataFrame({"a": [4, 5, 6], "c": ["d", "e", "f"]}, index=[10, 11, 12])

        Align both axis:

        >>> aligned_l, aligned_r = df1.align(df2)
        >>> aligned_l.sort_index()
              a     b   c
        10  1.0     a NaN
        11  NaN  None NaN
        12  NaN  None NaN
        20  2.0     b NaN
        30  3.0     c NaN
        >>> aligned_r.sort_index()
              a   b     c
        10  4.0 NaN     d
        11  5.0 NaN     e
        12  6.0 NaN     f
        20  NaN NaN  None
        30  NaN NaN  None

        Align only axis=0 (index):

        >>> aligned_l, aligned_r = df1.align(df2, axis=0)
        >>> aligned_l.sort_index()
              a     b
        10  1.0     a
        11  NaN  None
        12  NaN  None
        20  2.0     b
        30  3.0     c
        >>> aligned_r.sort_index()
              a     c
        10  4.0     d
        11  5.0     e
        12  6.0     f
        20  NaN  None
        30  NaN  None

        Align only axis=1 (column):

        >>> aligned_l, aligned_r = df1.align(df2, axis=1)
        >>> aligned_l.sort_index()
            a  b   c
        10  1  a NaN
        20  2  b NaN
        30  3  c NaN
        >>> aligned_r.sort_index()
            a   b  c
        10  4 NaN  d
        11  5 NaN  e
        12  6 NaN  f

        Align with the join type "inner":

        >>> aligned_l, aligned_r = df1.align(df2, join="inner")
        >>> aligned_l.sort_index()
            a
        10  1
        >>> aligned_r.sort_index()
            a
        10  4

        Align with a Series:

        >>> s = ks.Series([7, 8, 9], index=[10, 11, 12])
        >>> aligned_l, aligned_r = df1.align(s, axis=0)
        >>> aligned_l.sort_index()
              a     b
        10  1.0     a
        11  NaN  None
        12  NaN  None
        20  2.0     b
        30  3.0     c
        >>> aligned_r.sort_index()
        10    7.0
        11    8.0
        12    9.0
        20    NaN
        30    NaN
        dtype: float64

        >>> ks.reset_option("compute.ops_on_diff_frames")
        """
        from databricks.koalas.series import Series, first_series
        if not isinstance(other, (DataFrame, Series)):
            raise TypeError('unsupported type: {}'.format(type(other).__name__)
                )
        how = validate_how(join)
        axis = validate_axis(axis, None)
        right_is_series = isinstance(other, Series)
        if right_is_series:
            if axis is None:
                raise ValueError('Must specify axis=0 or 1')
            elif axis != 0:
                raise NotImplementedError(
                    'align currently only works for axis=0 when right is Series'
                    )
        left = self
        right = other
        if (axis is None or axis == 0) and not same_anchor(left, right):
            combined = combine_frames(left, right, how=how)
            left = combined['this']
            right = combined['that']
            if right_is_series:
                right = first_series(right).rename(other.name)
        if (axis is None or axis == 1
            ) and left._internal.column_labels != right._internal.column_labels:
            if (left._internal.column_labels_level != right._internal.
                column_labels_level):
                raise ValueError('cannot join with no overlapping index names')
            left = left.copy()
            right = right.copy()
            if how == 'full':
                column_labels = sorted(list(set(left._internal.
                    column_labels) | set(right._internal.column_labels)))
            elif how == 'inner':
                column_labels = sorted(list(set(left._internal.
                    column_labels) & set(right._internal.column_labels)))
            elif how == 'left':
                column_labels = left._internal.column_labels
            else:
                column_labels = right._internal.column_labels
            for label in column_labels:
                if label not in left._internal.column_labels:
                    left[label] = F.lit(None).cast(DoubleType())
            left = left[column_labels]
            for label in column_labels:
                if label not in right._internal.column_labels:
                    right[label] = F.lit(None).cast(DoubleType())
            right = right[column_labels]
        return (left.copy(), right.copy()) if copy else (left, right)

    @staticmethod
    def from_dict(data, orient='columns', dtype=None, columns=None):
        """
        Construct DataFrame from dict of array-like or dicts.

        Creates DataFrame object from dictionary by columns or by index
        allowing dtype specification.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        orient : {'columns', 'index'}, default 'columns'
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass 'columns'
            (default). Otherwise if the keys should be rows, pass 'index'.
        dtype : dtype, default None
            Data type to force, otherwise infer.
        columns : list, default None
            Column labels to use when ``orient='index'``. Raises a ValueError
            if used with ``orient='columns'``.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_records : DataFrame from structured ndarray, sequence
            of tuples or dicts, or DataFrame.
        DataFrame : DataFrame object creation using constructor.

        Examples
        --------
        By default the keys of the dict become the DataFrame columns:

        >>> data = {'col_1': [3, 2, 1, 0], 'col_2': [10, 20, 30, 40]}
        >>> ks.DataFrame.from_dict(data)
           col_1  col_2
        0      3     10
        1      2     20
        2      1     30
        3      0     40

        Specify ``orient='index'`` to create the DataFrame using dictionary
        keys as rows:

        >>> data = {'row_1': [3, 2, 1, 0], 'row_2': [10, 20, 30, 40]}
        >>> ks.DataFrame.from_dict(data, orient='index').sort_index()
                0   1   2   3
        row_1   3   2   1   0
        row_2  10  20  30  40

        When using the 'index' orientation, the column names can be
        specified manually:

        >>> ks.DataFrame.from_dict(data, orient='index',
        ...                        columns=['A', 'B', 'C', 'D']).sort_index()
                A   B   C   D
        row_1   3   2   1   0
        row_2  10  20  30  40
        """
        return DataFrame(pd.DataFrame.from_dict(data, orient=orient, dtype=
            dtype, columns=columns))

    def _to_internal_pandas(self):
        """
        Return a pandas DataFrame directly from _internal to avoid overhead of copy.

        This method is for internal use only.
        """
        return self._internal.to_pandas_frame

    def _get_or_create_repr_pandas_cache(self, n):
        if not hasattr(self, '_repr_pandas_cache'
            ) or n not in self._repr_pandas_cache:
            object.__setattr__(self, '_repr_pandas_cache', {n: self.head(n +
                1)._to_internal_pandas()})
        return self._repr_pandas_cache[n]

    def __repr__(self):
        max_display_count = get_option('display.max_rows')
        if max_display_count is None:
            return self._to_internal_pandas().to_string()
        pdf = self._get_or_create_repr_pandas_cache(max_display_count)
        pdf_length = len(pdf)
        pdf = pdf.iloc[:max_display_count]
        if pdf_length > max_display_count:
            repr_string = pdf.to_string(show_dimensions=True)
            match = REPR_PATTERN.search(repr_string)
            if match is not None:
                nrows = match.group('rows')
                ncols = match.group('columns')
                footer = (
                    '\n\n[Showing only the first {nrows} rows x {ncols} columns]'
                    .format(nrows=nrows, ncols=ncols))
                return REPR_PATTERN.sub(footer, repr_string)
        return pdf.to_string()

    def _repr_html_(self):
        max_display_count = get_option('display.max_rows')
        bold_rows = not LooseVersion('0.25.1') == LooseVersion(pd.__version__)
        if max_display_count is None:
            return self._to_internal_pandas().to_html(notebook=True,
                bold_rows=bold_rows)
        pdf = self._get_or_create_repr_pandas_cache(max_display_count)
        pdf_length = len(pdf)
        pdf = pdf.iloc[:max_display_count]
        if pdf_length > max_display_count:
            repr_html = pdf.to_html(show_dimensions=True, notebook=True,
                bold_rows=bold_rows)
            match = REPR_HTML_PATTERN.search(repr_html)
            if match is not None:
                nrows = match.group('rows')
                ncols = match.group('columns')
                by = chr(215)
                footer = (
                    '\n<p>Showing only the first {rows} rows {by} {cols} columns</p>\n</div>'
                    .format(rows=nrows, by=by, cols=ncols))
                return REPR_HTML_PATTERN.sub(footer, repr_html)
        return pdf.to_html(notebook=True, bold_rows=bold_rows)

    def __getitem__(self, key):
        from databricks.koalas.series import Series
        if key is None:
            raise KeyError('none key')
        elif isinstance(key, Series):
            return self.loc[key.astype(bool)]
        elif isinstance(key, slice):
            if any(type(n) == int or None for n in [key.start, key.stop]):
                return self.iloc[key]
            return self.loc[key]
        elif is_name_like_value(key):
            return self.loc[:, key]
        elif is_list_like(key):
            return self.loc[:, list(key)]
        raise NotImplementedError(key)

    def __setitem__(self, key, value):
        from databricks.koalas.series import Series
        if isinstance(value, (DataFrame, Series)) and not same_anchor(value,
            self):
            level = self._internal.column_labels_level
            key = DataFrame._index_normalized_label(level, key)
            value = DataFrame._index_normalized_frame(level, value)

            def assign_columns(kdf, this_column_labels, that_column_labels):
                assert len(key) == len(that_column_labels)
                for k, this_label, that_label in zip_longest(key,
                    this_column_labels, that_column_labels):
                    yield kdf._kser_for(that_label), tuple(['that', *k])
                    if this_label is not None and this_label[1:] != k:
                        yield kdf._kser_for(this_label), this_label
            kdf = align_diff_frames(assign_columns, self, value, fillna=
                False, how='left')
        elif isinstance(value, list):
            if len(self) != len(value):
                raise ValueError(
                    'Length of values does not match length of index')
            with option_context('compute.default_index_type',
                'distributed-sequence', 'compute.ops_on_diff_frames', True):
                kdf = self.reset_index()
                kdf[key] = ks.DataFrame(value)
                kdf = kdf.set_index(kdf.columns[:self._internal.index_level])
                kdf.index.names = self.index.names
        elif isinstance(key, list):
            assert isinstance(value, DataFrame)
            field_names = value.columns
            kdf = self._assign({k: value[c] for k, c in zip(key, field_names)})
        else:
            kdf = self._assign({key: value})
        self._update_internal_frame(kdf._internal)

    @staticmethod
    def _index_normalized_label(level, labels):
        """
        Returns a label that is normalized against the current column index level.
        For example, the key "abc" can be ("abc", "", "") if the current Frame has
        a multi-index for its column
        """
        if is_name_like_tuple(labels):
            labels = [labels]
        elif is_name_like_value(labels):
            labels = [(labels,)]
        else:
            labels = [(k if is_name_like_tuple(k) else (k,)) for k in labels]
        if any(len(label) > level for label in labels):
            raise KeyError('Key length ({}) exceeds index depth ({})'.
                format(max(len(label) for label in labels), level))
        return [tuple(list(label) + [''] * (level - len(label))) for label in
            labels]

    @staticmethod
    def _index_normalized_frame(level, kser_or_kdf):
        """
        Returns a frame that is normalized against the current column index level.
        For example, the name in `pd.Series([...], name="abc")` can be can be
        ("abc", "", "") if the current DataFrame has a multi-index for its column
        """
        from databricks.koalas.series import Series
        if isinstance(kser_or_kdf, Series):
            kdf = kser_or_kdf.to_frame()
        else:
            assert isinstance(kser_or_kdf, DataFrame), type(kser_or_kdf)
            kdf = kser_or_kdf.copy()
        kdf.columns = pd.MultiIndex.from_tuples([tuple([name_like_string(
            label)] + [''] * (level - 1)) for label in kdf._internal.
            column_labels])
        return kdf

    def __getattr__(self, key):
        if key.startswith('__'):
            raise AttributeError(key)
        if hasattr(_MissingPandasLikeDataFrame, key):
            property_or_func = getattr(_MissingPandasLikeDataFrame, key)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        try:
            return self.loc[:, key]
        except KeyError:
            raise AttributeError("'%s' object has no attribute '%s'" % (
                self.__class__.__name__, key))

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)
            return object.__setattr__(self, key, value)
        except AttributeError:
            pass
        if (key,) in self._internal.column_labels:
            self[key] = value
        else:
            msg = (
                "Koalas doesn't allow columns to be created via a new attribute name"
                )
            if is_testing():
                raise AssertionError(msg)
            else:
                warnings.warn(msg, UserWarning)

    def __len__(self):
        return self._internal.resolved_copy.spark_frame.count()

    def __dir__(self):
        fields = [f for f in self._internal.resolved_copy.spark_frame.
            schema.fieldNames() if ' ' not in f]
        return super().__dir__() + fields

    def __iter__(self):
        return iter(self.columns)

    def __array_ufunc__(self, ufunc, method, *inputs: Any, **kwargs: Any):
        if all(isinstance(inp, DataFrame) for inp in inputs) and any(not
            same_anchor(inp, inputs[0]) for inp in inputs):
            assert len(inputs) == 2
            this = inputs[0]
            that = inputs[1]
            if (this._internal.column_labels_level != that._internal.
                column_labels_level):
                raise ValueError('cannot join with no overlapping index names')

            def apply_op(kdf, this_column_labels, that_column_labels):
                for this_label, that_label in zip(this_column_labels,
                    that_column_labels):
                    yield ufunc(kdf._kser_for(this_label), kdf._kser_for(
                        that_label), **kwargs).rename(this_label), this_label
            return align_diff_frames(apply_op, this, that, fillna=True, how
                ='full')
        else:
            applied = []
            this = inputs[0]
            assert all(inp is this for inp in inputs if isinstance(inp,
                DataFrame))
            for label in this._internal.column_labels:
                arguments = []
                for inp in inputs:
                    arguments.append(inp[label] if isinstance(inp,
                        DataFrame) else inp)
                applied.append(ufunc(*arguments, **kwargs).rename(label))
            internal = this._internal.with_new_columns(applied)
            return DataFrame(internal)
    if sys.version_info >= (3, 7):

        def __class_getitem__(cls, params):
            return _create_tuple_for_frame_type(params)
    elif (3, 5) <= sys.version_info < (3, 7):
        is_dataframe = None


def _reduce_spark_multi(sdf, aggs):
    """
    Performs a reduction on a spark DataFrame, the functions being known sql aggregate functions.
    """
    assert isinstance(sdf, spark.DataFrame)
    sdf0 = sdf.agg(*aggs)
    l = sdf0.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row = l.iloc[0]
    l2 = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2


class CachedDataFrame(DataFrame):
    """
    Cached Koalas DataFrame, which corresponds to pandas DataFrame logically, but internally
    it caches the corresponding Spark DataFrame.
    """

    def __init__(self, internal, storage_level=None):
        if storage_level is None:
            object.__setattr__(self, '_cached', internal.spark_frame.cache())
        elif isinstance(storage_level, StorageLevel):
            object.__setattr__(self, '_cached', internal.spark_frame.
                persist(storage_level))
        else:
            raise TypeError(
                'Only a valid pyspark.StorageLevel type is acceptable for the `storage_level`'
                )
        super().__init__(internal)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.spark.unpersist()
    spark = CachedAccessor('spark', CachedSparkFrameMethods)

    @property
    def storage_level(self):
        warnings.warn(
            'DataFrame.storage_level is deprecated as of DataFrame.spark.storage_level. Please use the API instead.'
            , FutureWarning)
        return self.spark.storage_level
    storage_level.__doc__ = CachedSparkFrameMethods.storage_level.__doc__

    def unpersist(self):
        warnings.warn(
            'DataFrame.unpersist is deprecated as of DataFrame.spark.unpersist. Please use the API instead.'
            , FutureWarning)
        return self.spark.unpersist()
    unpersist.__doc__ = CachedSparkFrameMethods.unpersist.__doc__
