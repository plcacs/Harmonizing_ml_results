from __future__ import annotations
import operator
from operator import le, lt
import textwrap
from typing import TYPE_CHECKING, Literal, Union, overload, Any, Optional, Sequence, Tuple, List, cast, Dict, Set, TypeVar, Generic
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import VALID_CLOSED, Interval, IntervalMixin, intervals_to_interval_bounds
from pandas._libs.missing import NA
from pandas._typing import ArrayLike, AxisInt, Dtype, IntervalClosedType, NpDtype, PositionalIndexer, ScalarIndexer, Self, SequenceIndexer, SortKind, TimeArrayLike, npt
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import LossySetitemError, maybe_upcast_numeric_to_64bit
from pandas.core.dtypes.common import is_float_dtype, is_integer_dtype, is_list_like, is_object_dtype, is_scalar, is_string_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, IntervalDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCDatetimeIndex, ABCIntervalIndex, ABCPeriodIndex
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna, notna
from pandas.core.algorithms import isin, take, unique, value_counts_internal as value_counts
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray, _extension_array_shared_docs
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import array as pd_array, ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import invalid_comparison, unpack_zerodim_and_defer
if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence as ABCSequence
    from pandas import Index, Series
    from numpy.typing import NDArray
    from pyarrow import Array as ArrowArray

IntervalSide = Union[TimeArrayLike, np.ndarray]
IntervalOrNA = Union[Interval, float]
_interval_shared_docs: Dict[str, str] = {}
_shared_docs_kwargs: Dict[str, str] = {'klass': 'IntervalArray', 'qualname': 'arrays.IntervalArray', 'name': ''}
_interval_shared_docs['class'] = "\n%(summary)s\n\nParameters\n----------\ndata : array-like (1-dimensional)\n    Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing\n    Interval objects from which to build the %(klass)s.\nclosed : {'left', 'right', 'both', 'neither'}, default 'right'\n    Whether the intervals are closed on the left-side, right-side, both or\n    neither.\ndtype : dtype or None, default None\n    If None, dtype will be inferred.\ncopy : bool, default False\n    Copy the input data.\n%(name)sverify_integrity : bool, default True\n    Verify that the %(klass)s is valid.\n\nAttributes\n----------\nleft\nright\nclosed\nmid\nlength\nis_empty\nis_non_overlapping_monotonic\n%(extra_attributes)s\nMethods\n-------\nfrom_arrays\nfrom_tuples\nfrom_breaks\ncontains\noverlaps\nset_closed\nto_tuples\n%(extra_methods)s\nSee Also\n--------\nIndex : The base pandas Index type.\nInterval : A bounded slice-like interval; the elements of an %(klass)s.\ninterval_range : Function to create a fixed frequency IntervalIndex.\ncut : Bin values into discrete Intervals.\nqcut : Bin values into equal-sized Intervals based on rank or sample quantiles.\n\nNotes\n-----\nSee the `user guide\n<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__\nfor more.\n\n%(examples)s"

@Appender(_interval_shared_docs['class'] % {'klass': 'IntervalArray', 'summary': 'Pandas array for interval data that are closed on the same side.', 'name': '', 'extra_attributes': '', 'extra_methods': '', 'examples': textwrap.dedent('    Examples\n    --------\n    A new ``IntervalArray`` can be constructed directly from an array-like of\n    ``Interval`` objects:\n\n    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])\n    <IntervalArray>\n    [(0, 1], (1, 5]]\n    Length: 2, dtype: interval[int64, right]\n\n    It may also be constructed using one of the constructor\n    methods: :meth:`IntervalArray.from_arrays`,\n    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.\n    ')})
class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na: bool = True
    _na_value: Any = _fill_value = np.nan

    @property
    def ndim(self) -> int:
        return 1

    def __new__(cls, data: Any, closed: Optional[IntervalClosedType] = None, dtype: Optional[Dtype] = None, copy: bool = False, verify_integrity: bool = True) -> Self:
        data = extract_array(data, extract_numpy=True)
        if isinstance(data, cls):
            left = data._left
            right = data._right
            closed = closed or data.closed
            dtype = IntervalDtype(left.dtype, closed=closed)
        else:
            if is_scalar(data):
                msg = f'{cls.__name__}(...) must be called with a collection of some kind, {data} was passed'
                raise TypeError(msg)
            data = _maybe_convert_platform_interval(data)
            left, right, infer_closed = intervals_to_interval_bounds(data, validate_closed=closed is None)
            if left.dtype == object:
                left = lib.maybe_convert_objects(left)
                right = lib.maybe_convert_objects(right)
            closed = closed or infer_closed
            left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed, copy=copy, dtype=dtype)
        if verify_integrity:
            cls._validate(left, right, dtype=dtype)
        return cls._simple_new(left, right, dtype=dtype)

    @classmethod
    def _simple_new(cls, left: Any, right: Any, dtype: IntervalDtype) -> Self:
        result = IntervalMixin.__new__(cls)
        result._left = left
        result._right = right
        result._dtype = dtype
        return result

    @classmethod
    def _ensure_simple_new_inputs(cls, left: Any, right: Any, closed: Optional[IntervalClosedType] = None, copy: bool = False, dtype: Optional[Dtype] = None) -> Tuple[Any, Any, IntervalDtype]:
        """Ensure correctness of input parameters for cls._simple_new."""
        from pandas.core.indexes.base import ensure_index
        left = ensure_index(left, copy=copy)
        left = maybe_upcast_numeric_to_64bit(left)
        right = ensure_index(right, copy=copy)
        right = maybe_upcast_numeric_to_64bit(right)
        if closed is None and isinstance(dtype, IntervalDtype):
            closed = dtype.closed
        closed = closed or 'right'
        if dtype is not None:
            dtype = pandas_dtype(dtype)
            if isinstance(dtype, IntervalDtype):
                if dtype.subtype is not None:
                    left = left.astype(dtype.subtype)
                    right = right.astype(dtype.subtype)
            else:
                msg = f'dtype must be an IntervalDtype, got {dtype}'
                raise TypeError(msg)
            if dtype.closed is None:
                dtype = IntervalDtype(dtype.subtype, closed)
            elif closed != dtype.closed:
                raise ValueError('closed keyword does not match dtype.closed')
        if is_float_dtype(left.dtype) and is_integer_dtype(right.dtype):
            right = right.astype(left.dtype)
        elif is_float_dtype(right.dtype) and is_integer_dtype(left.dtype):
            left = left.astype(right.dtype)
        if type(left) != type(right):
            msg = f'must not have differing left [{type(left).__name__}] and right [{type(right).__name__}] types'
            raise ValueError(msg)
        if isinstance(left.dtype, CategoricalDtype) or is_string_dtype(left.dtype):
            msg = 'category, object, and string subtypes are not supported for IntervalArray'
            raise TypeError(msg)
        if isinstance(left, ABCPeriodIndex):
            msg = 'Period dtypes are not supported, use a PeriodIndex instead'
            raise ValueError(msg)
        if isinstance(left, ABCDatetimeIndex) and str(left.tz) != str(right.tz):
            msg = f"left and right must have the same time zone, got '{left.tz}' and '{right.tz}'"
            raise ValueError(msg)
        elif needs_i8_conversion(left.dtype) and left.unit != right.unit:
            left_arr, right_arr = left._data._ensure_matching_resos(right._data)
            left = ensure_index(left_arr)
            right = ensure_index(right_arr)
        left = ensure_wrapped_if_datetimelike(left)
        left = extract_array(left, extract_numpy=True)
        right = ensure_wrapped_if_datetimelike(right)
        right = extract_array(right, extract_numpy=True)
        if isinstance(left, ArrowExtensionArray) or isinstance(right, ArrowExtensionArray):
            pass
        else:
            lbase = getattr(left, '_ndarray', left)
            lbase = getattr(lbase, '_data', lbase).base
            rbase = getattr(right, '_ndarray', right)
            rbase = getattr(rbase, '_data', rbase).base
            if lbase is not None and lbase is rbase:
                right = right.copy()
        dtype = IntervalDtype(left.dtype, closed=closed)
        return (left, right, dtype)

    @classmethod
    def _from_sequence(cls, scalars: Sequence[Any], *, dtype: Optional[Dtype] = None, copy: bool = False) -> Self:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: Any, original: Any) -> Self:
        return cls._from_sequence(values, dtype=original.dtype)
    _interval_shared_docs['from_breaks'] = textwrap.dedent("\n        Construct an %(klass)s from an array of splits.\n\n        Parameters\n        ----------\n        breaks : array-like (1-dimensional)\n            Left and right bounds for each interval.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.        %(name)s\n        copy : bool, default False\n            Copy the data.\n        dtype : dtype or None, default None\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_arrays : Construct from a left and right array.\n        %(klass)s.from_tuples : Construct from a sequence of tuples.\n\n        %(examples)s        ")

    @classmethod
    @Appender(_interval_shared_docs['from_breaks'] % {'klass': 'IntervalArray', 'name': '', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, dtype: interval[int64, right]\n        ')})
    def from_breaks(cls, breaks: Any, closed: IntervalClosedType = 'right', copy: bool = False, dtype: Optional[Dtype] = None) -> Self:
        breaks = _maybe_convert_platform_interval(breaks)
        return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)
    _interval_shared_docs['from_arrays'] = textwrap.dedent("\n        Construct from two arrays defining the left and right bounds.\n\n        Parameters\n        ----------\n        left : array-like (1-dimensional)\n            Left bounds for each interval.\n        right : array-like (1-dimensional)\n            Right bounds for each interval.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.        %(name)s\n        copy : bool, default False\n            Copy the data.\n        dtype : dtype, optional\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        Raises\n        ------\n        ValueError\n            When a value is missing in only one of `left` or `right`.\n            When a value in `left` is greater than the corresponding value\n            in `right`.\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_breaks : Construct an %(klass)s from an array of\n            splits.\n        %(klass)s.from_tuples : Construct an %(klass)s from an\n            array-like of tuples.\n\n        Notes\n        -----\n        Each element of `left` must be less than or equal to the `right`\n        element at the same position. If an element is missing, it must be\n        missing in both `left` and `right`. A TypeError is raised when\n        using an unsupported type for `left` or `right`. At the moment,\n        'category', 'object', and 'string' subtypes are not supported.\n\n        %(examples)s        ")

    @classmethod
    @Appender(_interval_shared_docs['from_arrays'] % {'klass': 'IntervalArray', 'name': '', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, dtype: interval[int64, right]\n        ')})
    def from_arrays(cls, left: Any, right: Any, closed: IntervalClosedType = 'right', copy: bool = False, dtype: Optional[Dtype] = None) -> Self:
        left = _maybe_convert_platform_interval(left)
        right = _maybe_convert_platform_interval(right)
        left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed, copy=copy, dtype=dtype)
        cls._validate(left, right, dtype=dtype)
        return cls._simple_new(left, right, dtype=dtype)
    _interval_shared_docs['from_tuples'] = textwrap.dedent("\n        Construct an %(klass)s from an array-like of tuples.\n\n        Parameters\n        ----------\n        data : array-like (1-dimensional)\n            Array of tuples.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.        %(name)s\n        copy : bool, default False\n            By-default copy the data, this is compat only and ignored.\n        dtype : dtype or None, default None\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_arrays : Construct an %(klass)s from a left and\n                                    right array.\n        %(klass)s.from_breaks : Construct an %(klass)s from an array of\n                                    splits.\n\n        %(examples)s        ")

    @classmethod
    @Appender(_interval_shared_docs['from_tuples'] % {'klass': 'IntervalArray', 'name': '', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])\n        <IntervalArray>\n        [(0, 1], (1, 2]]\n        Length: 2, dtype: interval[int64, right]\n        ')})
    def from_tuples(cls, data: Sequence[Tuple[Any, Any]], closed: IntervalClosedType = 'right', copy: bool = False, dtype: Optional[Dtype] = None) -> Self:
        if len(data):
            left, right = ([], [])
        else:
            left = right = data
        for d in data:
            if not isinstance(d, tuple) and isna(d):
                lhs = rhs = np.nan
            else:
                name = cls.__name__
                try:
                    lhs, rhs = d
                except ValueError as err:
                    msg = f'{name}.from_tuples requires tuples of length 2, got {d}'
                    raise ValueError(msg) from err
                except TypeError as err:
                    msg = f'{name}.from_tuples received an invalid item, {d}'
                    raise TypeError(msg) from