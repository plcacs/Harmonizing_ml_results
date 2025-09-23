from __future__ import annotations
from operator import le, lt
import textwrap
from typing import TYPE_CHECKING, Any, Literal, Optional, Union
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import Interval, IntervalMixin, IntervalTree
from pandas._libs.tslibs import BaseOffset, Period, Timedelta, Timestamp, to_offset
from pandas.errors import InvalidIndexError
from pandas.util._decorators import Appender, cache_readonly, set_module
from pandas.util._exceptions import rewrite_exception
from pandas.core.dtypes.cast import find_common_type, infer_dtype_from_scalar, maybe_box_datetimelike, maybe_downcast_numeric, maybe_upcast_numeric_to_64bit
from pandas.core.dtypes.common import ensure_platform_int, is_float_dtype, is_integer, is_integer_dtype, is_list_like, is_number, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, IntervalDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import IntervalArray, _interval_shared_docs
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs, ensure_index, maybe_extract_name
from pandas.core.indexes.datetimes import DatetimeIndex, date_range
from pandas.core.indexes.extension import ExtensionIndex, inherit_names
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex, timedelta_range
if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import Dtype, DtypeObj, IntervalClosedType, Self, npt

_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({
    'klass': 'IntervalIndex',
    'qualname': 'IntervalIndex',
    'target_klass': 'IntervalIndex or list of Intervals',
    'name': textwrap.dedent('         name : object, optional\n              Name to be stored in the index.\n         ')
})


def _get_next_label(label: Any) -> Any:
    dtype = getattr(label, 'dtype', type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = 'datetime64[ns]'
    dtype = pandas_dtype(dtype)
    if lib.is_np_dtype(dtype, 'mM') or isinstance(dtype, DatetimeTZDtype):
        return label + np.timedelta64(1, 'ns')
    elif is_integer_dtype(dtype):
        return label + 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, np.inf)
    else:
        raise TypeError(f'cannot determine next label for type {type(label)!r}')


def _get_prev_label(label: Any) -> Any:
    dtype = getattr(label, 'dtype', type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = 'datetime64[ns]'
    dtype = pandas_dtype(dtype)
    if lib.is_np_dtype(dtype, 'mM') or isinstance(dtype, DatetimeTZDtype):
        return label - np.timedelta64(1, 'ns')
    elif is_integer_dtype(dtype):
        return label - 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, -np.inf)
    else:
        raise TypeError(f'cannot determine next label for type {type(label)!r}')


def _new_IntervalIndex(cls: type[IntervalIndex], d: dict[str, Any]) -> IntervalIndex:
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """
    return cls.from_arrays(**d)


@Appender(_interval_shared_docs['class'] % {'klass': 'IntervalIndex', 'summary': 'Immutable index of intervals that are closed on the same side.', 'name': _index_doc_kwargs['name'], 'extra_attributes': 'is_overlapping\nvalues\n', 'extra_methods': '', 'examples': textwrap.dedent("    Examples\n    --------\n    A new ``IntervalIndex`` is typically constructed using\n    :func:`interval_range`:\n\n    >>> pd.interval_range(start=0, end=5)\n    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],\n                  dtype='interval[int64, right]')\n\n    It may also be constructed using one of the constructor\n    methods: :meth:`IntervalIndex.from_arrays`,\n    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.\n\n    See further examples in the doc strings of ``interval_range`` and the\n    mentioned constructor methods.\n    ")})
@inherit_names(['set_closed', 'to_tuples'], IntervalArray, wrap=True)
@inherit_names(['__array__', 'overlaps', 'contains', 'closed_left', 'closed_right', 'open_left', 'open_right', 'is_empty'], IntervalArray)
@inherit_names(['is_non_overlapping_monotonic', 'closed'], IntervalArray, cache=True)
@set_module('pandas')
class IntervalIndex(ExtensionIndex):
    _typ: str = 'intervalindex'
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool
    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings: bool = False
    _data_cls = IntervalArray

    def __new__(
        cls: type[IntervalIndex],
        data: Any,
        closed: Optional[IntervalClosedType] = None,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
        name: Any = None,
        verify_integrity: bool = True
    ) -> IntervalIndex:
        name = maybe_extract_name(name, data, cls)
        with rewrite_exception('IntervalArray', cls.__name__):
            array = IntervalArray(data, closed=closed, copy=copy, dtype=dtype, verify_integrity=verify_integrity)
        return cls._simple_new(array, name)

    @classmethod
    @Appender(_interval_shared_docs['from_breaks'] % {'klass': 'IntervalIndex', 'name': textwrap.dedent('\n             name : str, optional\n                  Name of the resulting IntervalIndex.'), 'examples': textwrap.dedent("        Examples\n        --------\n        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])\n        IntervalIndex([(0, 1], (1, 2], (2, 3]],\n                      dtype='interval[int64, right]')\n        ")})
    def from_breaks(
        cls: type[IntervalIndex],
        breaks: Any,
        closed: IntervalClosedType = 'right',
        name: Optional[Any] = None,
        copy: bool = False,
        dtype: Optional[Dtype] = None
    ) -> IntervalIndex:
        with rewrite_exception('IntervalArray', cls.__name__):
            array = IntervalArray.from_breaks(breaks, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(_interval_shared_docs['from_arrays'] % {'klass': 'IntervalIndex', 'name': textwrap.dedent('\n             name : str, optional\n                  Name of the resulting IntervalIndex.'), 'examples': textwrap.dedent("        Examples\n        --------\n        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])\n        IntervalIndex([(0, 1], (1, 2], (2, 3]],\n                      dtype='interval[int64, right]')\n        ")})
    def from_arrays(
        cls: type[IntervalIndex],
        left: Any,
        right: Any,
        closed: IntervalClosedType = 'right',
        name: Optional[Any] = None,
        copy: bool = False,
        dtype: Optional[Dtype] = None
    ) -> IntervalIndex:
        with rewrite_exception('IntervalArray', cls.__name__):
            array = IntervalArray.from_arrays(left, right, closed, copy=copy, dtype=dtype)
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(_interval_shared_docs['from_tuples'] % {'klass': 'IntervalIndex', 'name': textwrap.dedent('\n             name : str, optional\n                  Name of the resulting IntervalIndex.'), 'examples': textwrap.dedent("        Examples\n        --------\n        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])\n        IntervalIndex([(0, 1], (1, 2]],\n                       dtype='interval[int64, right]')\n        ")})
    def from_tuples(
        cls: type[IntervalIndex],
        data: Any,
        closed: IntervalClosedType = 'right',
        name: Optional[Union[str, Hashable]] = None,
        copy: bool = False,
        dtype: Optional[Dtype] = None
    ) -> IntervalIndex:
        with rewrite_exception('IntervalArray', cls.__name__):
            arr = IntervalArray.from_tuples(data, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(arr, name=name)

    @cache_readonly
    def _engine(self) -> IntervalTree:
        left = self._maybe_convert_i8(self.left)
        left = maybe_upcast_numeric_to_64bit(left)
        right = self._maybe_convert_i8(self.right)
        right = maybe_upcast_numeric_to_64bit(right)
        return IntervalTree(left, right, closed=self.closed)

    def __contains__(self, key: Any) -> bool:
        """
        return a boolean if this key is IN the index
        We *only* accept an Interval

        Parameters
        ----------
        key : Interval

        Returns
        -------
        bool
        """
        hash(key)
        if not isinstance(key, Interval):
            if is_valid_na_for_dtype(key, self.dtype):
                return self.hasnans
            return False
        try:
            self.get_loc(key)
            return True
        except KeyError:
            return False

    def _getitem_slice(self, slobj: slice) -> IntervalIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._data[slobj]
        return type(self)._simple_new(res, name=self._name)

    @cache_readonly
    def _multiindex(self) -> MultiIndex:
        return MultiIndex.from_arrays([self.left, self.right], names=['left', 'right'])

    def __reduce__(self) -> tuple[Any, tuple[Any, dict[str, Any]], Any]:
        d = {'left': self.left, 'right': self.right, 'closed': self.closed, 'name': self.name}
        return (_new_IntervalIndex, (type(self), d), None)

    @property
    def inferred_type(self) -> str:
        """Return a string of the type inferred from the values"""
        return 'interval'

    @Appender(Index.memory_usage.__doc__)
    def memory_usage(self, deep: bool = False) -> int:
        return self.left.memory_usage(deep=deep) + self.right.memory_usage(deep=deep)

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if the IntervalIndex is monotonic decreasing (only equal or
        decreasing values), else False
        """
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def is_unique(self) -> bool:
        """
        Return True if the IntervalIndex contains unique elements, else False.
        """
        left = self.left
        right = self.right
        if self.isna().sum() > 1:
            return False
        if left.is_unique or right.is_unique:
            return True
        seen_pairs: set[tuple[Any, Any]] = set()
        check_idx = np.where(left.duplicated(keep=False))[0]
        for idx in check_idx:
            pair = (left[idx], right[idx])
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)
        return True

    @property
    def is_overlapping(self) -> bool:
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        See the complete docstring in the source.
        """
        return self._engine.is_overlapping

    def _needs_i8_conversion(self, key: Any) -> bool:
        """
        Check if a given key needs i8 conversion.
        """
        key_dtype = getattr(key, 'dtype', None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            return self._needs_i8_conversion(key.left)
        i8_types = (Timestamp, Timedelta, DatetimeIndex, TimedeltaIndex)
        return isinstance(key, i8_types)

    def _maybe_convert_i8(self, key: Any) -> Any:
        """
        Maybe convert a given key to its equivalent i8 value(s).
        """
        if is_list_like(key):
            key = ensure_index(key)
            key = maybe_upcast_numeric_to_64bit(key)
        if not self._needs_i8_conversion(key):
            return key
        scalar = is_scalar(key)
        key_dtype = getattr(key, 'dtype', None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            left = self._maybe_convert_i8(key.left)
            right = self._maybe_convert_i8(key.right)
            constructor = Interval if scalar else IntervalIndex.from_arrays
            return constructor(left, right, closed=self.closed)
        if scalar:
            (key_dtype, key_i8) = infer_dtype_from_scalar(key)
            if isinstance(key, Period):
                key_i8 = key.ordinal
            elif isinstance(key_i8, Timestamp):
                key_i8 = key_i8._value
            elif isinstance(key_i8, (np.datetime64, np.timedelta64)):
                key_i8 = key_i8.view('i8')
        else:
            (key_dtype, key_i8) = (key.dtype, Index(key.asi8))
            if key.hasnans:
                key_i8 = key_i8.where(~key._isnan)
        subtype = self.dtype.subtype
        if subtype != key_dtype:
            raise ValueError(f'Cannot index an IntervalIndex of subtype {subtype} with values of dtype {key_dtype}')
        return key_i8

    def _searchsorted_monotonic(self, label: Any, side: Literal['left', 'right'] = 'left') -> int:
        if not self.is_non_overlapping_monotonic:
            raise KeyError('can only get slices from an IntervalIndex if bounds are non-overlapping and all monotonic increasing or decreasing')
        if isinstance(label, (IntervalMixin, IntervalIndex)):
            raise NotImplementedError('Interval objects are not currently supported')
        if side == 'left' and self.left.is_monotonic_increasing or (side == 'right' and (not self.left.is_monotonic_increasing)):
            sub_idx = self.right
            if self.open_right:
                label = _get_next_label(label)
        else:
            sub_idx = self.left
            if self.open_left:
                label = _get_prev_label(label)
        return sub_idx._searchsorted_monotonic(label, side)

    def get_loc(self, key: Any) -> Union[int, slice, np.ndarray]:
        """
        Get integer location, slice or boolean mask for requested label.
        """
        self._check_indexing_error(key)
        if isinstance(key, Interval):
            if self.closed != key.closed:
                raise KeyError(key)
            mask = (self.left == key.left) & (self.right == key.right)
        elif is_valid_na_for_dtype(key, self.dtype):
            mask = self.isna()
        else:
            op_left = le if self.closed_left else lt
            op_right = le if self.closed_right else lt
            try:
                mask = op_left(self.left, key) & op_right(key, self.right)
            except TypeError as err:
                raise KeyError(key) from err
        matches = mask.sum()
        if matches == 0:
            raise KeyError(key)
        if matches == 1:
            return mask.argmax()
        res = lib.maybe_booleans_to_slice(mask.view('u1'))
        if isinstance(res, slice) and res.stop is None:
            res = slice(res.start, len(self), res.step)
        return res

    def _get_indexer(
        self,
        target: Index,
        method: Optional[str] = None,
        limit: Optional[int] = None,
        tolerance: Any = None
    ) -> npt.NDArray[np.intp]:
        if isinstance(target, IntervalIndex):
            indexer = self._get_indexer_unique_sides(target)
        elif not (is_object_dtype(target.dtype) or is_string_dtype(target.dtype)):
            target = self._maybe_convert_i8(target)
            indexer = self._engine.get_indexer(target.values)
        else:
            return self._get_indexer_pointwise(target)[0]
        return ensure_platform_int(indexer)

    @Appender(_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs)
    def get_indexer_non_unique(self, target: Index) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        target = ensure_index(target)
        if not self._should_compare(target) and (not self._should_partial_index(target)):
            return self._get_indexer_non_comparable(target, None, unique=False)
        elif isinstance(target, IntervalIndex):
            if self.left.is_unique and self.right.is_unique:
                indexer = self._get_indexer_unique_sides(target)
                missing = (indexer == -1).nonzero()[0]
            else:
                return self._get_indexer_pointwise(target)
        elif is_object_dtype(target.dtype) or not self._should_partial_index(target):
            return self._get_indexer_pointwise(target)
        else:
            target = self._maybe_convert_i8(target)
            (indexer, missing) = self._engine.get_indexer_non_unique(target.values)
        return (ensure_platform_int(indexer), ensure_platform_int(missing))

    def _get_indexer_unique_sides(self, target: IntervalIndex) -> npt.NDArray[np.intp]:
        """
        _get_indexer specialized to the case where both of our sides are unique.
        """
        left_indexer = self.left.get_indexer(target.left)
        right_indexer = self.right.get_indexer(target.right)
        indexer = np.where(left_indexer == right_indexer, left_indexer, -1)
        return indexer

    def _get_indexer_pointwise(self, target: Index) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
        indexer_list: list[np.ndarray] = []
        missing_list: list[int] = []
        for i, key in enumerate(target):
            try:
                locs = self.get_loc(key)
                if isinstance(locs, slice):
                    locs = np.arange(locs.start, locs.stop, locs.step, dtype='intp')
                elif lib.is_integer(locs):
                    locs = np.array(locs, ndmin=1)
                else:
                    locs = np.where(locs)[0]
            except (KeyError, InvalidIndexError):
                missing_list.append(i)
                locs = np.array([-1])
            indexer_list.append(locs)
        indexer = np.concatenate(indexer_list)
        return (ensure_platform_int(indexer), ensure_platform_int(np.array(missing_list, dtype='intp')))

    @cache_readonly
    def _index_as_unique(self) -> bool:
        return not self.is_overlapping and self._engine._na_count < 2

    _requires_unique_msg: str = 'cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique'

    def _convert_slice_indexer(self, key: slice, kind: Literal['loc', 'getitem']) -> Any:
        if not (key.step is None or key.step == 1):
            msg = 'label-based slicing with step!=1 is not supported for IntervalIndex'
            if kind == 'loc':
                raise ValueError(msg)
            if kind == 'getitem':
                if not is_valid_positional_slice(key):
                    raise ValueError(msg)
        return super()._convert_slice_indexer(key, kind)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        return self.dtype.subtype.kind in 'mM'

    def _maybe_cast_slice_bound(self, label: Any, side: str) -> Any:
        return getattr(self, side)._maybe_cast_slice_bound(label, side)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        if not isinstance(dtype, IntervalDtype):
            return False
        common_subtype = find_common_type([self.dtype, dtype])
        return not is_object_dtype(common_subtype)

    @cache_readonly
    def left(self) -> Index:
        """
        Return left bounds of the intervals in the IntervalIndex.
        """
        return Index(self._data.left, copy=False)

    @cache_readonly
    def right(self) -> Index:
        """
        Return right bounds of the intervals in the IntervalIndex.
        """
        return Index(self._data.right, copy=False)

    @cache_readonly
    def mid(self) -> Index:
        """
        Return the midpoint of each interval in the IntervalIndex as an Index.
        """
        return Index(self._data.mid, copy=False)

    @property
    def length(self) -> Index:
        """
        Calculate the length of each interval in the IntervalIndex.
        """
        return Index(self._data.length, copy=False)

    def _intersection(self, other: IntervalIndex, sort: bool = False) -> IntervalIndex:
        """
        intersection specialized to the case with matching dtypes.
        """
        if self.left.is_unique and self.right.is_unique:
            taken = self._intersection_unique(other)
        elif other.left.is_unique and other.right.is_unique and (self.isna().sum() <= 1):
            taken = other._intersection_unique(self)
        else:
            taken = self._intersection_non_unique(other)
        if sort:
            taken = taken.sort_values()
        return taken

    def _intersection_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does not have any common endpoint,
        no matter left or right.
        """
        lindexer = self.left.get_indexer(other.left)
        rindexer = self.right.get_indexer(other.right)
        match = (lindexer == rindexer) & (lindexer != -1)
        indexer = lindexer.take(match.nonzero()[0])
        indexer = unique(indexer)
        return self.take(indexer)

    def _intersection_non_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does have some common endpoints.
        """
        mask = np.zeros(len(self), dtype=bool)
        if self.hasnans and other.hasnans:
            first_nan_loc = np.arange(len(self))[self.isna()][0]
            mask[first_nan_loc] = True
        other_tups = set(zip(other.left, other.right))
        for i, tup in enumerate(zip(self.left, self.right)):
            if tup in other_tups:
                mask[i] = True
        return self[mask]

    def _get_engine_target(self) -> np.ndarray:
        raise NotImplementedError('IntervalIndex does not use libjoin fastpaths or pass values to IndexEngine objects')

    def _from_join_target(self, result: Any) -> Any:
        raise NotImplementedError('IntervalIndex does not use libjoin fastpaths')


def _is_valid_endpoint(endpoint: Any) -> bool:
    """
    Helper for interval_range to check if start/end are valid types.
    """
    return any([is_number(endpoint), isinstance(endpoint, Timestamp), isinstance(endpoint, Timedelta), endpoint is None])


def _is_type_compatible(a: Any, b: Any) -> bool:
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
    is_ts_compat = lambda x: isinstance(x, (Timestamp, BaseOffset))
    is_td_compat = lambda x: isinstance(x, (Timedelta, BaseOffset))
    return (is_number(a) and is_number(b)) or (is_ts_compat(a) and is_ts_compat(b)) or (is_td_compat(a) and is_td_compat(b)) or com.any_none(a, b)


def interval_range(
    start: Any = None,
    end: Any = None,
    periods: Optional[int] = None,
    freq: Any = None,
    name: Any = None,
    closed: IntervalClosedType = 'right'
) -> IntervalIndex:
    """
    Return a fixed frequency IntervalIndex.
    """
    start = maybe_box_datetimelike(start)
    end = maybe_box_datetimelike(end)
    endpoint = start if start is not None else end
    if freq is None and com.any_none(periods, start, end):
        freq = 1 if is_number(endpoint) else 'D'
    if com.count_not_none(start, end, periods, freq) != 3:
        raise ValueError('Of the four parameters: start, end, periods, and freq, exactly three must be specified')
    if not _is_valid_endpoint(start):
        raise ValueError(f'start must be numeric or datetime-like, got {start}')
    if not _is_valid_endpoint(end):
        raise ValueError(f'end must be numeric or datetime-like, got {end}')
    periods = validate_periods(periods)
    if freq is not None and (not is_number(freq)):
        try:
            freq = to_offset(freq)
        except ValueError as err:
            raise ValueError(f'freq must be numeric or convertible to DateOffset, got {freq}') from err
    if not all([_is_type_compatible(start, end), _is_type_compatible(start, freq), _is_type_compatible(end, freq)]):
        raise TypeError('start, end, freq need to be type compatible')
    if periods is not None:
        periods += 1
    breaks: Union[np.ndarray, TimedeltaIndex, DatetimeIndex]
    if is_number(endpoint):
        dtype: np.dtype = np.dtype('int64')
        if com.all_not_none(start, end, freq):
            if isinstance(start, (float, np.float16)) or isinstance(end, (float, np.float16)) or isinstance(freq, (float, np.float16)):
                dtype = np.dtype('float64')
            elif isinstance(start, (np.integer, np.floating)) and isinstance(end, (np.integer, np.floating)) and (start.dtype == end.dtype):
                dtype = start.dtype
            breaks = np.arange(start, end + freq * 0.1, freq)
            breaks = maybe_downcast_numeric(breaks, dtype)
        else:
            if periods is None:
                periods = int((end - start) // freq) + 1
            elif start is None:
                start = end - (periods - 1) * freq
            elif end is None:
                end = start + (periods - 1) * freq
            breaks = np.linspace(start, end, periods)
        if all((is_integer(x) for x in com.not_none(start, end, freq))):
            breaks = maybe_downcast_numeric(breaks, dtype)
    elif isinstance(endpoint, Timestamp):
        breaks = date_range(start=start, end=end, periods=periods, freq=freq)
    else:
        breaks = timedelta_range(start=start, end=end, periods=periods, freq=freq)
    return IntervalIndex.from_breaks(breaks, name=name, closed=closed, dtype=IntervalDtype(subtype=breaks.dtype, closed=closed))