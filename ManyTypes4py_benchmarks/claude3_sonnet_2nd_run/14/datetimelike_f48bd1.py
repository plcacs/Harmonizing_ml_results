"""
Base and utility classes for tseries type pandas objects.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast, final, Optional, Union, Literal, TypeVar, overload
import numpy as np
from pandas._libs import NaT, Timedelta, lib
from pandas._libs.tslibs import BaseOffset, Resolution, Tick, parsing, to_offset
from pandas.compat.numpy import function as nv
from pandas.errors import InvalidIndexError, NullFrequencyError
from pandas.util._decorators import Appender, cache_readonly, doc
from pandas.core.dtypes.common import is_integer, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype, PeriodDtype
from pandas.core.arrays import DatetimeArray, ExtensionArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta
if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pandas._typing import Axis, JoinHow, Self, npt
    from pandas import CategoricalIndex
_index_doc_kwargs = dict(ibase._index_doc_kwargs)

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """
    _can_hold_strings: bool = False

    @doc(DatetimeLikeArrayMixin.mean)
    def mean(self, *, skipna: bool = True, axis: int = 0) -> Timedelta:
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> Optional[BaseOffset]:
        """
        Return the frequency object if it is set, otherwise None.

        To learn more about the frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        See Also
        --------
        DatetimeIndex.freq : Return the frequency object if it is set, otherwise None.
        PeriodIndex.freq : Return the frequency object if it is set, otherwise None.

        Examples
        --------
        >>> datetimeindex = pd.date_range(
        ...     "2022-02-22 02:22:22", periods=10, tz="America/Chicago", freq="h"
        ... )
        >>> datetimeindex
        DatetimeIndex(['2022-02-22 02:22:22-06:00', '2022-02-22 03:22:22-06:00',
                       '2022-02-22 04:22:22-06:00', '2022-02-22 05:22:22-06:00',
                       '2022-02-22 06:22:22-06:00', '2022-02-22 07:22:22-06:00',
                       '2022-02-22 08:22:22-06:00', '2022-02-22 09:22:22-06:00',
                       '2022-02-22 10:22:22-06:00', '2022-02-22 11:22:22-06:00'],
                      dtype='datetime64[ns, America/Chicago]', freq='h')
        >>> datetimeindex.freq
        <Hour>
        """
        return self._data.freq

    @freq.setter
    def freq(self, value: Optional[BaseOffset]) -> None:
        self._data.freq = value

    @property
    def asi8(self) -> np.ndarray:
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> Optional[str]:
        from pandas import PeriodIndex
        if self._data.freqstr is not None and isinstance(self._data, (PeriodArray, PeriodIndex)):
            freq = PeriodDtype(self._data.freq)._freqstr
            return freq
        else:
            return self._data.freqstr

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Resolution:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> Resolution:
        return self._data.resolution

    @cache_readonly
    def hasnans(self) -> bool:
        return self._data._hasna

    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in 'iufc':
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                other = cast('CategoricalIndex', other)
                should_try = other.categories.inferred_type in inferable
            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    return False
        if self.dtype != other.dtype:
            return False
        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            return False
        return True

    def _convert_tolerance(self, tolerance: Any, target: np.ndarray) -> np.ndarray:
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tolerance, target)
    _default_na_rep: str = 'NaT'

    def _format_with_header(self, *, header: list[str], na_rep: str, date_format: Optional[str] = None) -> list[str]:
        return header + list(self._get_values_for_csv(na_rep=na_rep, date_format=date_format))

    @property
    def _formatter_func(self) -> Any:
        return self._data._formatter()

    def _format_attrs(self) -> list[tuple[str, str]]:
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            if attrib == 'freq':
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)
                attrs.append(('freq', freq))
        return attrs

    @Appender(Index._summary.__doc__)
    def _summary(self, name: Optional[str] = None) -> str:
        result = super()._summary(name=name)
        if self.freq:
            result += f'\nFreq: {self.freqstr}'
        return result

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        return reso > self._resolution_obj

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime) -> tuple[Any, Any]:
        raise NotImplementedError

    def _parse_with_reso(self, label: str) -> tuple[datetime, Resolution]:
        try:
            if self.freq is None or hasattr(self.freq, 'rule_code'):
                freq = self.freq
        except NotImplementedError:
            freq = getattr(self, 'freqstr', getattr(self, 'inferred_freq', None))
        if freq is not None and (not isinstance(freq, str)):
            freqstr = freq.rule_code
        else:
            freqstr = freq
        if isinstance(label, np.str_):
            label = str(label)
        parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso = Resolution.from_attrname(reso_str)
        return (parsed, reso)

    def _get_string_slice(self, key: str) -> Union[slice, np.ndarray]:
        parsed, reso = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def _partial_date_slice(self, reso: Resolution, parsed: datetime) -> Union[slice, np.ndarray]:
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
        if not self._can_partial_date_slice(reso):
            raise ValueError
        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox
        if self.is_monotonic_increasing:
            if len(self) and (t1 < self[0] and t2 < self[0] or (t1 > self[-1] and t2 > self[-1])):
                raise KeyError
            left = vals.searchsorted(unbox(t1), side='left')
            right = vals.searchsorted(unbox(t2), side='right')
            return slice(left, right)
        else:
            lhs_mask = vals >= unbox(t1)
            rhs_mask = vals <= unbox(t2)
            return (lhs_mask & rhs_mask).nonzero()[0]

    def _maybe_cast_slice_bound(self, label: Any, side: Literal['left', 'right']) -> Any:
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
        if isinstance(label, str):
            try:
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                self._raise_invalid_indexer('slice', label, err)
            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == 'left' else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer('slice', label)
        return label

    def shift(self, periods: int = 1, freq: Optional[Union[BaseOffset, str]] = None) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        raise NotImplementedError

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr: Any) -> Index:
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                res = com.asarray_tuplesafe(keyarr)
            else:
                res = keyarr
        return Index(res, dtype=res.dtype)

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """
    _comparables: list[str] = ['name', 'freq']
    _attributes: list[str] = ['name', 'freq']
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def unit(self) -> str:
        return self._data.unit

    def as_unit(self, unit: str) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        This method is for converting the dtype of a ``DatetimeIndex`` or
        ``TimedeltaIndex`` to a new dtype with the given unit
        resolution/precision.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}

        Returns
        -------
        same type as self
            Converted to the specified unit.

        See Also
        --------
        Timestamp.as_unit : Convert to the given unit.
        Timedelta.as_unit : Convert to the given unit.
        DatetimeIndex.as_unit : Convert to the given unit.
        TimedeltaIndex.as_unit : Convert to the given unit.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(["2020-01-02 01:02:03.004005006"])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit("s")
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(["1 day 3 min 2 us 42 ns"])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit("s")
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
        arr = self._data.as_unit(unit)
        return type(self)._simple_new(arr, name=self.name)

    def _with_freq(self, freq: Optional[BaseOffset]) -> Self:
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def values(self) -> np.ndarray:
        data = self._data._ndarray
        data = data.view()
        data.flags.writeable = False
        return data

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq: Optional[Union[BaseOffset, str]] = None) -> Self:
        if freq is not None and freq != self.freq:
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = periods * freq
            return self + offset
        if periods == 0 or len(self) == 0:
            return self.copy()
        if self.freq is None:
            raise NullFrequencyError('Cannot shift with no freq')
        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq
        result = self._data._generate_range(start=start, end=end, periods=None, freq=self.freq, unit=self.unit)
        return type(self)._simple_new(result, name=self.name)

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.inferred_freq)
    def inferred_freq(self) -> Optional[str]:
        return self._data.inferred_freq

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        freq = cast(Tick, self.freq)
        tick = Timedelta(freq).as_unit(self.unit)._value
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def _can_range_setop(self, other: DatetimeTimedeltaMixin) -> bool:
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def _wrap_range_setop(self, other: DatetimeTimedeltaMixin, res_i8: Union[RangeIndex, np.ndarray]) -> Self:
        new_freq = None
        if not len(res_i8):
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            new_freq = to_offset(Timedelta(res_i8.step, unit=self.unit).as_unit(self.unit))
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        result = type(self._data)._simple_new(res_values, dtype=self.dtype, freq=new_freq)
        return cast('Self', self._wrap_setop_result(other, result))

    def _range_intersect(self, other: DatetimeTimedeltaMixin, sort: bool) -> Self:
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _range_union(self, other: DatetimeTimedeltaMixin, sort: bool) -> Self:
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.union(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _intersection(self, other: Index, sort: bool = False) -> Self:
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
        other = cast('DatetimeTimedeltaMixin', other)
        if self._can_range_setop(other):
            return self._range_intersect(other, sort=sort)
        if not self._can_fast_intersect(other):
            result = Index._intersection(self, other, sort=sort)
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq('infer')
        else:
            return self._fast_intersect(other, sort)

    def _fast_intersect(self, other: DatetimeTimedeltaMixin, sort: bool) -> Self:
        if self[0] <= other[0]:
            left, right = (self, other)
        else:
            left, right = (other, self)
        end = min(left[-1], right[-1])
        start = right[0]
        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]
        return result

    def _can_fast_intersect(self, other: DatetimeTimedeltaMixin) -> bool:
        if self.freq is None:
            return False
        elif other.freq != self.freq:
            return False
        elif not self.is_monotonic_increasing:
            return False
        return self.freq.n == 1

    def _can_fast_union(self, other: DatetimeTimedeltaMixin) -> bool:
        freq = self.freq
        if freq is None or freq != other.freq:
            return False
        if not self.is_monotonic_increasing:
            return False
        if len(self) == 0 or len(other) == 0:
            return True
        if self[0] <= other[0]:
            left, right = (self, other)
        else:
            left, right = (other, self)
        right_start = right[0]
        left_end = left[-1]
        return right_start == left_end + freq or right_start in left

    def _fast_union(self, other: DatetimeTimedeltaMixin, sort: Optional[bool] = None) -> Self:
        if self[0] <= other[0]:
            left, right = (self, other)
        elif sort is False:
            left, right = (self, other)
            left_start = left[0]
            loc = right.searchsorted(left_start, side='left')
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = type(self)._simple_new(dates, name=self.name)
            return result
        else:
            left, right = (other, self)
        left_end = left[-1]
        right_end = right[-1]
        if left_end < right_end:
            loc = right.searchsorted(left_end, side='right')
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            assert isinstance(dates, type(self._data))
            assert dates._freq == self.freq
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def _union(self, other: Index, sort: Optional[bool]) -> Self:
        assert isinstance(other, type(self))
        assert self.dtype == other.dtype
        if self._can_range_setop(other):
            return self._range_union(other, sort=sort)
        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            return result
        else:
            return super()._union(other, sort)._with_freq('infer')

    def _get_join_freq(self, other: DatetimeTimedeltaMixin) -> Optional[BaseOffset]:
        """
        Get the freq to attach to the result of a join operation.
        """
        freq = None
        if self._can_fast_union(other):
            freq = self.freq
        return freq

    def _wrap_join_result(self, joined: Index, other: Index, lidx: np.ndarray, ridx: np.ndarray, how: JoinHow) -> tuple[Self, np.ndarray, np.ndarray]:
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        join_index, lidx, ridx = super()._wrap_join_result(joined, other, lidx, ridx, how)
        join_index._data._freq = self._get_join_freq(cast(DatetimeTimedeltaMixin, other))
        return (join_index, lidx, ridx)

    def _get_engine_target(self) -> np.ndarray:
        return self._data._ndarray.view('i8')

    def _from_join_target(self, result: np.ndarray) -> ExtensionArray:
        result = result.view(self._data._ndarray.dtype)
        return self._data._from_backing_data(result)

    def _get_delete_freq(self, loc: Union[int, Sequence[int], slice]) -> Optional[BaseOffset]:
        """
        Find the `freq` for self.delete(loc).
        """
        freq = None
        if self.freq is not None:
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    loc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq

    def _get_insert_freq(self, loc: int, item: Any) -> Optional[BaseOffset]:
        """
        Find the `freq` for self.insert(loc, item).
        """
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)
        freq = None
        if self.freq is not None:
            if self.size:
                if item is NaT:
                    pass
                elif loc in (0, -len(self)) and item + self.freq == self[0]:
                    freq = self.freq
                elif loc == len(self) and item - self.freq == self[-1]:
                    freq = self.freq
            elif isinstance(self.freq, Tick):
                freq = self.freq
            elif self.freq.is_on_offset(item):
                freq = self.freq
        return freq

    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self, loc: Union[int, Sequence[int]]) -> Self:
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: int, item: Any) -> Index:
        result = super().insert(loc, item)
        if isinstance(result, type(self)):
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def take(self, indices: Sequence[int], axis: int = 0, allow_fill: bool = True, 
             fill_value: Optional[Any] = None, **kwargs: Any) -> Self:
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)
        result = NDArrayBackedExtensionIndex.take(self, indices, axis, allow_fill, fill_value, **kwargs)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
