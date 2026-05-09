from __future__ import annotations
import operator
from operator import le, lt
import textwrap
from typing import TYPE_CHECKING, Literal, Union, overload
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
    from collections.abc import Callable, Iterator, Sequence
    from pandas import Index, Series

IntervalSide = Union[TimeArrayLike, np.ndarray]
IntervalOrNA = Union[Interval, float]
_interval_shared_docs = {}
_shared_docs_kwargs = {'klass': 'IntervalArray', 'qualname': 'arrays.IntervalArray', 'name': ''}
_interval_shared_docs['class'] = "\n%(summary)s\n\nParameters\n----------\ndata : array-like (1-dimensional)\n    Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing\n    Interval objects from which to build the %(klass)s.\nclosed : {'left', 'right', 'both', 'neither'}, default 'right'\n    Whether the intervals are closed on the left-side, right-side, both or\n    neither.\ndtype : dtype or None, default None\n    If None, dtype will be inferred.\ncopy : bool, default False\n    Copy the input data.\n%(name)sverify_integrity : bool, default True\n    Verify that the %(klass)s is valid.\n\nAttributes\n----------\nleft\nright\nclosed\nmid\nlength\nis_empty\nis_non_overlapping_monotonic\n%(extra_attributes)s\nMethods\n-------\nfrom_arrays\nfrom_tuples\nfrom_breaks\ncontains\noverlaps\nset_closed\nto_tuples\n%(extra_methods)s\nSee Also\n--------\nIndex : The base pandas Index type.\nInterval : A bounded slice-like interval; the elements of an %(klass)s.\ninterval_range : Function to create a fixed frequency IntervalIndex.\ncut : Bin values into discrete Intervals.\nqcut : Bin values into equal-sized Intervals based on rank or sample quantiles.\n\nNotes\n-----\nSee the `user guide\n<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__\nfor more.\n\n%(examples)s"

@Appender(_interval_shared_docs['class'] % {'klass': 'IntervalArray', 'summary': 'Pandas array for interval data that are closed on the same side.', 'name': '', 'extra_attributes': '', 'extra_methods': '', 'examples': textwrap.dedent('    Examples\n    --------\n    A new ``IntervalArray`` can be constructed directly from an array-like of\n    ``Interval`` objects:\n\n    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])\n    <IntervalArray>\n    [(0, 1], (1, 5]]\n    Length: 2, dtype: interval[int64, right]\n\n    It may also be constructed using one of the constructor\n    methods: :meth:`IntervalArray.from_arrays`,\n    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.\n    ')})
class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na: bool = True
    _na_value: IntervalOrNA = np.nan
    _fill_value: IntervalOrNA = np.nan

    @property
    def ndim(self) -> int:
        return 1

    def __new__(cls, data: ArrayLike, closed: IntervalClosedType = None, dtype: Dtype = None, copy: bool = False, verify_integrity: bool = True) -> IntervalArray:
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
    def _simple_new(cls, left: np.ndarray, right: np.ndarray, dtype: Dtype) -> IntervalArray:
        result = IntervalMixin.__new__(cls)
        result._left = left
        result._right = right
        result._dtype = dtype
        return result

    @classmethod
    def _ensure_simple_new_inputs(cls, left: np.ndarray, right: np.ndarray, closed: IntervalClosedType = None, copy: bool = False, dtype: Dtype = None) -> tuple[np.ndarray, np.ndarray, Dtype]:
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
    def _from_sequence(cls, scalars: ArrayLike, *, dtype: Dtype = None, copy: bool = False) -> IntervalArray:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: ArrayLike, original: IntervalArray) -> IntervalArray:
        return cls._from_sequence(values, dtype=original.dtype)

    _interval_shared_docs['from_breaks'] = textwrap.dedent("\n        Construct an %(klass)s from an array of splits.\n\n        Parameters\n        ----------\n        breaks : array-like (1-dimensional)\n            Left and right bounds for each interval.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.        %(name)s\n        copy : bool, default False\n            Copy the data.\n        dtype : dtype or None, default None\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_arrays : Construct from a left and right array.\n        %(klass)s.from_tuples : Construct from a sequence of tuples.\n\n        %(examples)s        ")

    @classmethod
    @Appender(_interval_shared_docs['from_breaks'] % {'klass': 'IntervalArray', 'name': '', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, dtype: interval[int64, right]\n        ')})
    def from_breaks(cls, breaks: ArrayLike, closed: IntervalClosedType = 'right', copy: bool = False, dtype: Dtype = None) -> IntervalArray:
        breaks = _maybe_convert_platform_interval(breaks)
        return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)

    _interval_shared_docs['from_arrays'] = textwrap.dedent("\n        Construct from two arrays defining the left and right bounds.\n\n        Parameters\n        ----------\n        left : array-like (1-dimensional)\n            Left bounds for each interval.\n        right : array-like (1-dimensional)\n            Right bounds for each interval.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.        %(name)s\n        copy : bool, default False\n            Copy the data.\n        dtype : dtype, optional\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        Raises\n        ------\n        ValueError\n            When a value is missing in only one of `left` or `right`.\n            When a value in `left` is greater than the corresponding value\n            in `right`.\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_breaks : Construct an %(klass)s from an array of\n            splits.\n        %(klass)s.from_tuples : Construct an %(klass)s from an\n            array-like of tuples.\n\n        Notes\n        -----\n        Each element of `left` must be less than or equal to the `right`\n        element at the same position. If an element is missing, it must be\n        missing in both `left` and `right`. A TypeError is raised when\n        using an unsupported type for `left` or `right`. At the moment,\n        'category', 'object', and 'string' subtypes are not supported.\n\n        %(examples)s        ")

    @classmethod
    @Appender(_interval_shared_docs['from_arrays'] % {'klass': 'IntervalArray', 'name': '', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])\n        <IntervalArray>\n        [(0, 1], (1, 2], (2, 3]]\n        Length: 3, dtype: interval[int64, right]\n        ')})
    def from_arrays(cls, left: ArrayLike, right: ArrayLike, closed: IntervalClosedType = 'right', copy: bool = False, dtype: Dtype = None) -> IntervalArray:
        left = _maybe_convert_platform_interval(left)
        right = _maybe_convert_platform_interval(right)
        left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed, copy=copy, dtype=dtype)
        cls._validate(left, right, dtype=dtype)
        return cls._simple_new(left, right, dtype=dtype)

    _interval_shared_docs['from_tuples'] = textwrap.dedent("\n        Construct an %(klass)s from an array-like of tuples.\n\n        Parameters\n        ----------\n        data : array-like (1-dimensional)\n            Array of tuples.\n        closed : {'left', 'right', 'both', 'neither'}, default 'right'\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.        %(name)s\n        copy : bool, default False\n            By-default copy the data, this is compat only and ignored.\n        dtype : dtype or None, default None\n            If None, dtype will be inferred.\n\n        Returns\n        -------\n        %(klass)s\n\n        See Also\n        --------\n        interval_range : Function to create a fixed frequency IntervalIndex.\n        %(klass)s.from_arrays : Construct an %(klass)s from a left and\n                                    right array.\n        %(klass)s.from_breaks : Construct an %(klass)s from an array of\n                                    splits.\n\n        %(examples)s        ")

    @classmethod
    @Appender(_interval_shared_docs['from_tuples'] % {'klass': 'IntervalArray', 'name': '', 'examples': textwrap.dedent('        Examples\n        --------\n        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])\n        <IntervalArray>\n        [(0, 1], (1, 2]]\n        Length: 2, dtype: interval[int64, right]\n        ')})
    def from_tuples(cls, data: ArrayLike, closed: IntervalClosedType = 'right', copy: bool = False, dtype: Dtype = None) -> IntervalArray:
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
                    raise TypeError(msg) from err
            left.append(lhs)
            right.append(rhs)
        return cls.from_arrays(left, right, closed, copy=False, dtype=dtype)

    @classmethod
    def _validate(cls, left: np.ndarray, right: np.ndarray, dtype: Dtype) -> None:
        """
        Verify that the IntervalArray is valid.

        Checks that

        * dtype is correct
        * left and right match lengths
        * left and right have the same missing values
        * left is always below right
        """
        if not isinstance(dtype, IntervalDtype):
            msg = f'invalid dtype: {dtype}'
            raise ValueError(msg)
        if len(left) != len(right):
            msg = 'left and right must have the same length'
            raise ValueError(msg)
        left_mask = notna(left)
        right_mask = notna(right)
        if not (left_mask == right_mask).all():
            msg = 'missing values must be missing in the same location both left and right sides'
            raise ValueError(msg)
        if not (left[left_mask] <= right[left_mask]).all():
            msg = 'left side of interval must be <= right side'
            raise ValueError(msg)

    def _shallow_copy(self, left: np.ndarray, right: np.ndarray) -> IntervalArray:
        """
        Return a new IntervalArray with the replacement attributes

        Parameters
        ----------
        left : Index
            Values to be used for the left-side of the intervals.
        right : Index
            Values to be used for the right-side of the intervals.
        """
        dtype = IntervalDtype(left.dtype, closed=self.closed)
        left, right, dtype = self._ensure_simple_new_inputs(left, right, dtype=dtype)
        return self._simple_new(left, right, dtype=dtype)

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self.left.nbytes + self.right.nbytes

    @property
    def size(self) -> int:
        return self.left.size

    def __iter__(self) -> Iterator[Interval]:
        return iter(np.asarray(self))

    def __len__(self) -> int:
        return len(self._left)

    @overload
    def __getitem__(self, key: PositionalIndexer) -> IntervalArray:
        ...

    @overload
    def __getitem__(self, key: ScalarIndexer) -> IntervalArray:
        ...

    def __getitem__(self, key: Union[PositionalIndexer, ScalarIndexer]) -> IntervalArray:
        key = check_array_indexer(self, key)
        left = self._left[key]
        right = self._right[key]
        if not isinstance(left, (np.ndarray, ExtensionArray)):
            if is_scalar(left) and isna(left):
                return self._fill_value
            return Interval(left, right, self.closed)
        if np.ndim(left) > 1:
            raise ValueError('multi-dimensional indexing not allowed')
        return self._simple_new(left, right, dtype=self.dtype)

    def __setitem__(self, key: Union[PositionalIndexer, ScalarIndexer], value: IntervalOrNA) -> None:
        value_left, value_right = self._validate_setitem_value(value)
        key = check_array_indexer(self, key)
        self._left[key] = value_left
        self._right[key] = value_right

    def _cmp_method(self, other: IntervalOrNA, op: operator.Operator) -> np.ndarray:
        if is_list_like(other):
            if len(self) != len(other):
                raise ValueError('Lengths must match to compare')
            other = pd_array(other)
        elif not isinstance(other, Interval):
            if other is NA:
                from pandas.core.arrays import BooleanArray
                arr = np.empty(self.shape, dtype=bool)
                mask = np.ones(self.shape, dtype=bool)
                return BooleanArray(arr, mask)
            return invalid_comparison(self, other, op)
        if isinstance(other, Interval):
            other_dtype = pandas_dtype('interval')
        elif not isinstance(other.dtype, CategoricalDtype):
            other_dtype = other.dtype
        else:
            other_dtype = other.categories.dtype
            if isinstance(other_dtype, IntervalDtype):
                if self.closed != other.categories.closed:
                    return invalid_comparison(self, other, op)
                other = other.categories._values.take(other.codes, allow_fill=True, fill_value=other.categories._na_value)
        if isinstance(other_dtype, IntervalDtype):
            if self.closed != other.closed:
                return invalid_comparison(self, other, op)
            elif not isinstance(other, Interval):
                other = type(self)(other)
            if op is operator.eq:
                return (self._left == other.left) & (self._right == other.right)
            elif op is operator.ne:
                return (self._left != other.left) | (self._right != other.right)
            elif op is operator.gt:
                return (self._left > other.left) | (self._left == other.left) & (self._right > other.right)
            elif op is operator.ge:
                return (self == other) | (self > other)
            elif op is operator.lt:
                return (self._left < other.left) | (self._left == other.left) & (self._right < other.right)
            else:
                return (self == other) | (self < other)
        if not is_object_dtype(other_dtype):
            return invalid_comparison(self, other, op)
        result = np.zeros(len(self), dtype=bool)
        for i, obj in enumerate(other):
            try:
                result[i] = op(self[i], obj)
            except TypeError:
                if obj is NA:
                    result = result.astype(object)
                    result[i] = NA
                else:
                    raise
        return result

    @unpack_zerodim_and_defer('__eq__')
    def __eq__(self, other: IntervalOrNA) -> np.ndarray:
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer('__ne__')
    def __ne__(self, other: IntervalOrNA) -> np.ndarray:
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer('__gt__')
    def __gt__(self, other: IntervalOrNA) -> np.ndarray:
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer('__ge__')
    def __ge__(self, other: IntervalOrNA) -> np.ndarray:
        return self._cmp_method(other, operator.ge)

    @unpack_zerodim_and_defer('__lt__')
    def __lt__(self, other: IntervalOrNA) -> np.ndarray:
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer('__le__')
    def __le__(self, other: IntervalOrNA) -> np.ndarray:
        return self._cmp_method(other, operator.le)

    def argsort(self, *, ascending: bool = True, kind: str = 'quicksort', na_position: str = 'last', **kwargs: Any) -> np.ndarray:
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)
        if ascending and kind == 'quicksort' and (na_position == 'last'):
            return np.lexsort((self.right, self.left))
        return super().argsort(ascending=ascending, kind=kind, na_position=na_position, **kwargs)

    def min(self, *, axis: AxisInt = None, skipna: bool = True) -> IntervalOrNA:
        nv.validate_minmax_axis(axis, self.ndim)
        if not len(self):
            return self._na_value
        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self
        indexer = obj.argsort()[0]
        return obj[indexer]

    def max(self, *, axis: AxisInt = None, skipna: bool = True) -> IntervalOrNA:
        nv.validate_minmax_axis(axis, self.ndim)
        if not len(self):
            return self._na_value
        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self
        indexer = obj.argsort()[-1]
        return obj[indexer]

    def fillna(self, value: IntervalOrNA, limit: int = None, copy: bool = True) -> IntervalArray:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, a Series or dict can be used to fill in different
            values for each index. The value should not be a list. The
            value(s) passed should be either Interval objects or NA/NaN.
        limit : int, default None
            (Not implemented yet for IntervalArray)
            The maximum number of entries where NA values will be filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        filled : IntervalArray with NA/NaN filled
        """
        if copy is False:
            raise NotImplementedError
        if limit is not None:
            raise ValueError('limit must be None')
        value_left, value_right = self._validate_scalar(value)
        left = self.left.fillna(value=value_left)
        right = self.right.fillna(value=value_right)
        return self._shallow_copy(left, right)

    def astype(self, dtype: Dtype, copy: bool = True) -> ExtensionArray:
        """
        Cast to an ExtensionArray or NumPy array with dtype 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ExtensionArray or ndarray
            ExtensionArray or NumPy ndarray with 'dtype' for its dtype.
        """
        from pandas import Index
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if isinstance(dtype, IntervalDtype):
            if dtype == self.dtype:
                return self.copy() if copy else self
            if is_float_dtype(self.dtype.subtype) and needs_i8_conversion(dtype.subtype):
                msg = f'Cannot convert {self.dtype} to {dtype}; subtypes are incompatible'
                raise TypeError(msg)
            try:
                new_left = Index(self._left, copy=False).astype(dtype.subtype)
                new_right = Index(self._right, copy=False).astype(dtype.subtype)
            except IntCastingNaNError:
                raise
            except (TypeError, ValueError) as err:
                msg = f'Cannot convert {self.dtype} to {dtype}; subtypes are incompatible'
                raise TypeError(msg) from err
            return self._shallow_copy(new_left, new_right)
        else:
            try:
                return super().astype(dtype, copy=copy)
            except (TypeError, ValueError) as err:
                msg = f'Cannot cast {type(self).__name__} to dtype {dtype}'
                raise TypeError(msg) from err

    def equals(self, other: IntervalArray) -> bool:
        if type(self) != type(other):
            return False
        return bool(self.closed == other.closed and self.left.equals(other.left) and self.right.equals(other.right))

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[IntervalArray]) -> IntervalArray:
        """
        Concatenate multiple IntervalArray

        Parameters
        ----------
        to_concat : sequence of IntervalArray

        Returns
        -------
        IntervalArray
        """
        closed_set = {interval.closed for interval in to_concat}
        if len(closed_set) != 1:
            raise ValueError('Intervals must all be closed on the same side.')
        closed = closed_set.pop()
        left = np.concatenate([interval.left for interval in to_concat])
        right = np.concatenate([interval.right for interval in to_concat])
        left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed)
        return cls._simple_new(left, right, dtype=dtype)

    def copy(self) -> IntervalArray:
        """
        Return a copy of the array.

        Returns
        -------
        IntervalArray
        """
        left = self._left.copy()
        right = self._right.copy()
        dtype = self.dtype
        return self._simple_new(left, right, dtype=dtype)

    def isna(self) -> np.ndarray:
        return isna(self._left)

    def shift(self, periods: int = 1, fill_value: IntervalOrNA = None) -> IntervalArray:
        if not len(self) or periods == 0:
            return self.copy()
        self._validate_scalar(fill_value)
        empty_len = min(abs(periods), len(self))
        if isna(fill_value):
            from pandas import Index
            fill_value = Index(self._left, copy=False)._na_value
            empty = IntervalArray.from_breaks([fill_value] * (empty_len + 1), closed=self.closed)
        else:
            empty = self._from_sequence([fill_value] * empty_len, dtype=self.dtype)
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods):]
            b = empty
        return self._concat_same_type([a, b])

    def take(self, indices: Sequence[int], *, allow_fill: bool = False, fill_value: IntervalOrNA = None, axis: AxisInt = None, **kwargs: Any) -> IntervalArray:
        """
        Take elements from the IntervalArray.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.

        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : Interval or NA, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        axis : any, default None
            Present for compat with IntervalIndex; does nothing.

        Returns
        -------
        IntervalArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        """
        nv.validate_take((), kwargs)
        fill_left = fill_right = fill_value
        if allow_fill:
            fill_left, fill_right = self._validate_scalar(fill_value)
        left_take = take(self._left, indices, allow_fill=allow_fill, fill_value=fill_left)
        right_take = take(self._right, indices, allow_fill=allow_fill, fill_value=fill_right)
        return self._shallow_copy(left_take, right_take)

    def _validate_listlike(self, value: IntervalOrNA) -> tuple[np.ndarray, np.ndarray]:
        try:
            array = IntervalArray(value)
            self._check_closed_matches(array, name='value')
            value_left, value_right = (array.left, array.right)
        except TypeError as err:
            msg = f"'value' should be an interval type, got {type(value)} instead."
            raise TypeError(msg) from err
        try:
            self.left._validate_fill_value(value_left)
        except (LossySetitemError, TypeError) as err:
            msg = f"'value' should be a compatible interval type, got {type(value)} instead."
            raise TypeError(msg) from err
        return (value_left, value_right)

    def _validate_scalar(self, value: IntervalOrNA) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(value, Interval):
            self._check_closed_matches(value, name='value')
            left, right = (value.left, value.right)
        elif is_valid_na_for_dtype(value, self.left.dtype):
            left = right = self.left._na_value
        else:
            raise TypeError('can only insert Interval objects and NA into an IntervalArray')
        return (left, right)

    def _validate_setitem_value(self, value: IntervalOrNA) -> tuple[np.ndarray, np.ndarray]:
        if is_valid_na_for_dtype(value, self.left.dtype):
            value = self.left._na_value
            if is_integer_dtype(self.dtype.subtype):
                raise TypeError('Cannot set float NaN to integer-backed IntervalArray')
            value_left, value_right = (value, value)
        elif isinstance(value, Interval):
            self._check_closed_matches(value, name='value')
            value_left, value_right = (value.left, value.right)
            self.left._validate_fill_value(value_left)
            self.left._validate_fill_value(value_right)
        else:
            return self._validate_listlike(value)
        return (value_left, value_right)

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of each interval.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        result = value_counts(np.asarray(self), dropna=dropna)
        result.index = result.index.astype(self.dtype)
        return result

    def _formatter(self, boxed: bool = False) -> Callable[[Interval], str]:
        return str

    @property
    def left(self) -> Index:
        """
        Return the left endpoints of each Interval in the IntervalArray as an Index.

        This property provides access to the left endpoints of the intervals
        contained within the IntervalArray. This can be useful for analyses where
        the starting point of each interval is of interest, such as in histogram
        creation, data aggregation, or any scenario requiring the identification
        of the beginning of defined ranges. This property returns a ``pandas.Index``
        object containing the midpoint for each interval.

        See Also
        --------
        arrays.IntervalArray.right : Return the right endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.mid : Return the midpoint of each Interval in the
            IntervalArray as an Index.
        arrays.IntervalArray.contains : Check elementwise if the Intervals contain
            the value.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.left
        Index([0, 2], dtype='int64')
        """
        from pandas import Index
        return Index(self._left, copy=False)

    @property
    def right(self) -> Index:
        """
        Return the right endpoints of each Interval in the IntervalArray as an Index.

        This property extracts the right endpoints from each interval contained within
        the IntervalArray. This can be helpful in use cases where you need to work
        with or compare only the upper bounds of intervals, such as when performing
        range-based filtering, determining interval overlaps, or visualizing the end
        boundaries of data segments.

        See Also
        --------
        arrays.IntervalArray.left : Return the left endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.mid : Return the midpoint of each Interval in the
            IntervalArray as an Index.
        arrays.IntervalArray.contains : Check elementwise if the Intervals contain
            the value.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.right
        Index([1, 5], dtype='int64')
        """
        from pandas import Index
        return Index(self._right, copy=False)

    @property
    def length(self) -> Index:
        """
        Return an Index with entries denoting the length of each Interval.

        The length of an interval is calculated as the difference between
        its `right` and `left` bounds. This property is particularly useful
        when working with intervals where the size of the interval is an important
        attribute, such as in time-series analysis or spatial data analysis.

        See Also
        --------
        arrays.IntervalArray.left : Return the left endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.right : Return the right endpoints of each Interval in
            the IntervalArray as an Index.
        arrays.IntervalArray.mid : Return the midpoint of each Interval in the
            IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.length
        Index([1, 4], dtype='int64')
        """
        return self.right - self.left

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.

        The midpoint of an interval is calculated as the average of its
        ``left`` and ``right`` bounds. This property returns a ``pandas.Index`` object
        containing the midpoint for each interval.

        See Also
        --------
        Interval.left : Return left bound for the interval.
        Interval.right : Return right bound for the interval.
        Interval.length : Return the length of each interval.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.mid
        Index([0.5, 3.0], dtype='float64')
        """
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            return self.left + 0.5 * self.length

    _interval_shared_docs['overlaps'] = textwrap.dedent("\n        Check elementwise if an Interval overlaps the values in the %(klass)s.\n\n        Two intervals overlap if they share a common point, including closed\n        endpoints. Intervals that only have an open endpoint in common do not\n        overlap.\n\n        Parameters\n        ----------\n        other : %(klass)s\n            Interval to check against for an overlap.\n\n        Returns\n        -------\n        ndarray\n            Boolean array positionally indicating where an overlap occurs.\n\n        See Also\n        --------\n        Interval.overlaps : Check whether two Interval objects overlap.\n\n        Examples\n        --------\n        %(examples)s\n        >>> intervals.overlaps(pd.Interval(0.5, 1.5))\n        array([ True,  True, False])\n    ")

    def overlaps(self, other: Interval) -> np.ndarray:
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the IntervalArray.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array
            A boolean mask whether the value is contained in the Intervals.

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        IntervalArray.overlaps : Check if an Interval overlaps the values in the
            IntervalArray.

        Examples
        --------
        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]

        >>> intervals.contains(0.5)
        array([ True, False, False])
        """
        if isinstance(other, (IntervalArray, ABCIntervalIndex)):
            raise NotImplementedError
        if not isinstance(other, Interval):
            msg = f'`other` must be Interval-like, got {type(other).__name__}'
            raise TypeError(msg)
        op1 = le if self.closed_left and other.closed_right else lt
        op2 = le if other.closed_left and self.closed_right else lt
        return op1(self.left, other.right) & op2(other.left, self.right)

    @property
    def closed(self) -> str:
        """
        String describing the inclusive side the intervals.

        Either ``left``, ``right``, ``both`` or ``neither``.

        See Also
        --------
        IntervalArray.closed : Returns inclusive side of the IntervalArray.
        Interval.closed : Returns inclusive side of the Interval.
        IntervalIndex.closed : Returns inclusive side of the IntervalIndex.

        Examples
        --------

        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.closed
        'right'

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.closed
        'right'
        """
        return self.dtype.closed

    _interval_shared_docs['set_closed'] = textwrap.dedent("\n        Return an identical %(klass)s closed on the specified side.\n\n        Parameters\n        ----------\n        closed : {'left', 'right', 'both', 'neither'}\n            Whether the intervals are closed on the left-side, right-side, both\n            or neither.\n\n        Returns\n        -------\n        %(klass)s\n\n        %(examples)s        ")

    def set_closed(self, closed: IntervalClosedType) -> IntervalArray:
        """
        Return an identical IntervalArray closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.

        Returns
        -------
        IntervalArray
            A new IntervalArray with the specified side closures.

        See Also
        --------
        IntervalArray.closed : Returns inclusive side of the Interval.
        arrays.IntervalArray.closed : Returns inclusive side of the IntervalArray.

        Examples
        --------
        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))
        >>> index
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        >>> index.set_closed("both")
        <IntervalArray>
        [[0, 1], [1, 2], [2, 3]]
        Length: 3, dtype: interval[int64, both]
        """
        if closed not in VALID_CLOSED:
            msg = f"invalid option for 'closed': {closed}"
            raise ValueError(msg)
        left, right = (self._left, self._right)
        dtype = IntervalDtype(left.dtype, closed=closed)
        return self._simple_new(left, right, dtype=dtype)

    _interval_shared_docs['is_non_overlapping_monotonic'] = "\n        Return a boolean whether the %(klass)s is non-overlapping and monotonic.\n\n        Non-overlapping means (no Intervals share points), and monotonic means\n        either monotonic increasing or monotonic decreasing.\n\n        Examples\n        --------\n        For arrays:\n\n        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])\n        >>> interv_arr\n        <IntervalArray>\n        [(0, 1], (1, 5]]\n        Length: 2, dtype: interval[int64, right]\n        >>> interv_arr.is_non_overlapping_monotonic\n        True\n\n        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1),\n        ...                                       pd.Interval(-1, 0.1)])\n        >>> interv_arr\n        <IntervalArray>\n        [(0.0, 1.0], (-1.0, 0.1]]\n        Length: 2, dtype: interval[float64, right]\n        >>> interv_arr.is_non_overlapping_monotonic\n        False\n\n        For Interval Index:\n\n        >>> interv_idx = pd.interval_range(start=0, end=2)\n        >>> interv_idx\n        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')\n        >>> interv_idx.is_non_overlapping_monotonic\n        True\n\n        >>> interv_idx = pd.interval_range(start=0, end=2, closed='both')\n        >>> interv_idx\n        IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')\n        >>> interv_idx.is_non_overlapping_monotonic\n        False\n        ")

    @property
    def is_non_overlapping_monotonic(self) -> bool:
        """
        Return a boolean whether the IntervalArray/IntervalIndex        is non-overlapping and monotonic.

        Non-overlapping means (no Intervals share points), and monotonic means
        either monotonic increasing or monotonic decreasing.

        See Also
        --------
        overlaps : Check if two IntervalIndex objects overlap.

        Examples
        --------
        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        True

        >>> interv_arr = pd.arrays.IntervalArray(
        ...     [pd.Interval(0, 1), pd.Interval(-1, 0.1)]
        ... )
        >>> interv_arr
        <IntervalArray>
        [(0.0, 1.0], (-1.0, 0.1]]
        Length: 2, dtype: interval[float64, right]
        >>> interv_arr.is_non_overlapping_monotonic
        False

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.is_non_overlapping_monotonic
        True

        >>> interv_idx = pd.interval_range(start=0, end=2, closed="both")
        >>> interv_idx
        IntervalIndex([[0, 1], [1, 2]], dtype='interval[int64, both]')
        >>> interv_idx.is_non_overlapping_monotonic
        False
        """
        if self.closed == 'both':
            return bool((self._right[:-1] < self._left[1:]).all() or (self._left[:-1] > self._right[1:]).all())
        return bool((self._right[:-1] <= self._left[1:]).all() or (self._left[:-1] >= self._right[1:]).all())

    def __array__(self, dtype: Dtype = None, copy: bool = False) -> np.ndarray:
        """
        Return the IntervalArray's data as a numpy array of Interval
        objects (with dtype='object')
        """
        if copy is False:
            raise ValueError('Unable to avoid copy while creating an array as requested.')
        left = self._left
        right = self._right
        mask = self.isna()
        closed = self.closed
        result = np.empty(len(left), dtype=object)
        for i, left_value in enumerate(left):
            if mask[i]:
                result[i] = np.nan
            else:
                result[i] = Interval(left_value, right[i], closed)
        return result

    def __arrow_array__(self, type: pyarrow.DataType = None) -> pyarrow.ExtensionArray:
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow
        from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
        try:
            subtype = pyarrow.from_numpy_dtype(self.dtype.subtype)
        except TypeError as err:
            raise TypeError(f"Conversion to arrow with subtype '{self.dtype.subtype}' is not supported") from err
        interval_type = ArrowIntervalType(subtype, self.closed)
        storage_array = pyarrow.StructArray.from_arrays([pyarrow.array(self._left, type=subtype, from_pandas=True), pyarrow.array(self._right, type=subtype, from_pandas=True)], names=['left', 'right'])
        mask = self.isna()
        if mask.any():
            null_bitmap = pyarrow.array(~mask).buffers()[1]
            storage_array = pyarrow.StructArray.from_buffers(storage_array.type, len(storage_array), [null_bitmap], children=[storage_array.field(0), storage_array.field(1)])
        if type is not None:
            if type.equals(interval_type.storage_type):
                return storage_array
            elif isinstance(type, ArrowIntervalType):
                if not type.equals(interval_type):
                    raise TypeError(f"Not supported to convert IntervalArray to type with different 'subtype' ({self.dtype.subtype} vs {type.subtype}) and 'closed' ({self.closed} vs {type.closed}) attributes")
            else:
                raise TypeError(f"Not supported to convert IntervalArray to '{type}' type")
        return pyarrow.ExtensionArray.from_storage(interval_type, storage_array)

    _interval_shared_docs['to_tuples'] = textwrap.dedent('\n        Return an %(return_type)s of tuples of the form (left, right).\n\n        Parameters\n        ----------\n        na_tuple : bool, default True\n            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,\n            just return ``NA`` as ``nan``.\n\n        Returns\n        -------\n        tuples: %(return_type)s\n        %(examples)s        ')

    def to_tuples(self, na_tuple: bool = True) -> np.ndarray:
        """
        Return an ndarray (if self is IntervalArray) or Index         (if self is IntervalIndex) of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.

        Returns
        -------
        ndarray or Index
            An ndarray of tuples representing the intervals
                if `self` is an IntervalArray.
            An Index of tuples representing the intervals
                if `self` is an IntervalIndex.

        See Also
        --------
        IntervalArray.to_list : Convert IntervalArray to a list of tuples.
        IntervalArray.to_numpy : Convert IntervalArray to a numpy array.
        IntervalArray.unique : Find unique intervals in an IntervalArray.

        Examples
        --------
        For :class:`pandas.IntervalArray`:

        >>> idx = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        >>> idx
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        >>> idx.to_tuples()
        array([(0, 1), (1, 2)], dtype=object)

        For :class:`pandas.IntervalIndex`:

        >>> idx = pd.interval_range(start=0, end=2)
        >>> idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> idx.to_tuples()
        Index([(0, 1), (1, 2)], dtype='object')
        """
        tuples = com.asarray_tuplesafe(zip(self._left, self._right))
        if not na_tuple:
            tuples = np.where(~self.isna(), tuples, np.nan)
        return tuples

    def _putmask(self, mask: np.ndarray, value: IntervalOrNA) -> None:
        value_left, value_right = self._validate_setitem_value(value)
        if isinstance(self._left, np.ndarray):
            np.putmask(self._left, mask, value_left)
            assert isinstance(self._right, np.ndarray)
            np.putmask(self._right, mask, value_right)
        else:
            self._left._putmask(mask, value_left)
            assert not isinstance(self._right, np.ndarray)
            self._right._putmask(mask, value_right)

    def insert(self, loc: int, item: IntervalOrNA) -> IntervalArray:
        """
        Return a new IntervalArray inserting new item at location. Follows
        Python numpy.insert semantics for negative values.  Only Interval
        objects and NA can be inserted into an IntervalIndex

        Parameters
        ----------
        loc : int
        item : Interval

        Returns
        -------
        IntervalArray
        """
        left_insert, right_insert = self._validate_scalar(item)
        new_left = self.left.insert(loc, left_insert)
        new_right = self.right.insert(loc, right_insert)
        return self._shallow_copy(new_left, new_right)

    def delete(self, loc: int) -> IntervalArray:
        if isinstance(self._left, np.ndarray):
            new_left = np.delete(self._left, loc)
            assert isinstance(self._right, np.ndarray)
            new_right = np.delete(self._right, loc)
        else:
            new_left = self._left.delete(loc)
            assert not isinstance(self._right, np.ndarray)
            new_right = self._right.delete(loc)
        return self._shallow_copy(left=new_left, right=new_right)

    @Appender(_extension_array_shared_docs['repeat'] % _shared_docs_kwargs)
    def repeat(self, repeats: int, axis: AxisInt = None) -> IntervalArray:
        nv.validate_repeat((), {'axis': axis})
        left_repeat = self.left.repeat(repeats)
        right_repeat = self.right.repeat(repeats)
        return self._shallow_copy(left=left_repeat, right=right_repeat)

    _interval_shared_docs['contains'] = textwrap.dedent('\n        Check elementwise if the Intervals contain the value.\n\n        Return a boolean mask whether the value is contained in the Intervals\n        of the %(klass)s.\n\n        Parameters\n        ----------\n        other : scalar\n            The value to check whether it is contained in the Intervals.\n\n        Returns\n        -------\n        boolean array\n\n        See Also\n        --------\n        Interval.contains : Check whether Interval object contains value.\n        %(klass)s.overlaps : Check if an Interval overlaps the values in the\n            %(klass)s.\n\n        Examples\n        --------\n        %(examples)s\n        >>> intervals.contains(0.5)\n        array([ True, False, False])\n    ')

    def contains(self, other: IntervalOrNA) -> np.ndarray:
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the IntervalArray.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array
            A boolean mask whether the value is contained in the Intervals.

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        IntervalArray.overlaps : Check if an Interval overlaps the values in the
            IntervalArray.

        Examples
        --------
        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]

        >>> intervals.contains(0.5)
        array([ True, False, False])
        """
        if isinstance(other, (IntervalArray, ABCIntervalIndex)):
            raise NotImplementedError
        if not isinstance(other, Interval):
            msg = f'`other` must be Interval-like, got {type(other).__name__}'
            raise TypeError(msg)
        op1 = le if self.closed_left and other.closed_right else lt
        op2 = le if other.closed_left and self.closed_right else lt
        return op1(self.left, other.right) & op2(other.left, self.right)

    def isin(self, values: ArrayLike) -> np.ndarray:
        if isinstance(values, IntervalArray):
            if self.closed != values.closed:
                return np.zeros(self.shape, dtype=bool)
            if self.dtype == values.dtype:
                left = self._combined.view('complex128')
                right = values._combined.view('complex128')
                return np.isin(left, right).ravel()
            elif needs_i8_conversion(self.left.dtype) ^ needs_i8_conversion(values.left.dtype):
                return np.zeros(self.shape, dtype=bool)
        return isin(self.astype(object), values.astype(object))

    @property
    def _combined(self) -> np.ndarray:
        left = self.left._values.reshape(-1, 1)
        right = self.right._values.reshape(-1, 1)
        if needs_i8_conversion(left.dtype):
            comb = left._concat_same_type([left, right], axis=1)
        else:
            comb = np.concatenate([left, right], axis=1)
        return comb

    def _from_combined(self, combined: np.ndarray) -> IntervalArray:
        """
        Create a new IntervalArray with our dtype from a 1D complex128 ndarray.
        """
        nc = combined.view('i8').reshape(-1, 2)
        dtype = self._left.dtype
        if needs_i8_conversion(dtype):
            assert isinstance(self._left, (DatetimeArray, TimedeltaArray))
            new_left = type(self._left)._from_sequence(nc[:, 0], dtype=dtype)
            assert isinstance(self._right, (DatetimeArray, TimedeltaArray))
            new_right = type(self._right)._from_sequence(nc[:, 1], dtype=dtype)
        else:
            assert isinstance(dtype, np.dtype)
            new_left = nc[:, 0].view(dtype)
            new_right = nc[:, 1].view(dtype)
        return self._shallow_copy(left=new_left, right=new_right)

    def unique(self) -> IntervalArray:
        nc = unique(self._combined.view('complex128')[:, 0])
        nc = nc[:, None]
        return self._from_combined(nc)
