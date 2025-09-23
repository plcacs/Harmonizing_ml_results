"""
Base and utility classes for tseries type pandas objects.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast, final, overload
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
_index_doc_kwargs: dict[str, Any] = dict(ibase._index_doc_kwargs)

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """
    _can_hold_strings: bool = False
    _data: DatetimeArray | TimedeltaArray | PeriodArray

    @doc(DatetimeLikeArrayMixin.mean)
    def mean(self, *, skipna: bool = True, axis: int | None = 0) -> Any:
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> BaseOffset | None:
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
    def freq(self, value: BaseOffset | None) -> None:
        self._data.freq = value

    @property
    def asi8(self) -> npt.NDArray[np.int64] | None:
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> str | None:
        from pandas import PeriodIndex
        if self._data.freqstr is not None and isinstance(self._data, (PeriodArray, PeriodIndex)):
            freq: str | None = PeriodDtype(self._data.freq)._freqstr
            return freq
        else:
            return self._data.freqstr

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Resolution:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> str:
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
            should_try: bool = False
            inferable: list[str] = self._data._infer_matches
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
    def __contains__(self, key: object) -> bool:
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

    def _format_with_header(self, *, header: list[str], na_rep: str, date_format: str | None = None) -> list[str]:
        return header + list(self._get_values_for_csv(na_rep=na_rep, date_format=date_format))

    @property
    def _formatter_func(self):
        return self._data._formatter()

    def _format_attrs(self) -> list[tuple[str, str | None]]:
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs: list[tuple[str, str | None]] = super()._format_attrs()
        for attrib in self._attributes:
            if attrib == 'freq':
                freq: str | None = self.freqstr
                if freq is not None:
                    freq = repr(freq)
                attrs.append(('freq', freq))
        return attrs

    @Appender(Index._summary.__doc__)
    def _summary(self, name: str | None = None) -> str:
        result: str = super()._summary(name=name)
        if self.freq:
            result += f'\nFreq: {self.freqstr}'
        return result

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        return reso > self._resolution_obj

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime) -> tuple[datetime, datetime]:
        raise NotImplementedError

    def _parse_with_reso(self, label: str) -> tuple[datetime, Resolution]:
        try:
            if self.freq is None or hasattr(self.freq, 'rule_code'):
                freq: BaseOffset | None = self.freq
        except NotImplementedError:
            freq = getattr(self, 'freqstr', getattr(self, 'inferred_freq', None))
        freqstr: str | None
        if freq is not None and (not isinstance(freq, str)):
            freqstr = freq.rule_code
        else:
            freqstr = freq
        if isinstance(label, np.str_):
            label = str(label)
        (parsed, reso_str) = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso = Resolution.from_attrname(reso_str)
        return (parsed, reso)

    def _get_string_slice(self, key: str) -> slice | npt.NDArray[np.intp]:
        (parsed, reso) = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def _partial_date_slice(self, reso: Resolution, parsed: datetime) -> slice | npt.NDArray[np.intp]:
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
        (t1, t2) = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox
        if self.is_monotonic_increasing:
            if len(self) and (t1 < self[0] and t2 < self[0] or (t1 > self[-1] and t2 > self[-1])):
                raise KeyError
            left: int = vals.searchsorted(unbox(t1), side='left')
            right: int = vals.searchsorted(unbox(t2), side='right')
            return slice(left, right)
        else:
            lhs_mask: npt.NDArray[np.bool_] = vals >= unbox(t1)
            rhs_mask: npt.NDArray[np.bool_] = vals <= unbox(t2)
            return (lhs_mask & rhs_mask).nonzero()[0]

    def _maybe_cast_slice_bound(self, label: Any, side: str) -> Any:
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
                (parsed, reso) = self._parse_with_reso(label)
            except ValueError as err:
                self._raise_invalid_indexer('slice', label, err)
            (lower, upper) = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == 'left' else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer('slice', label)
        return label

    def shift(self, periods: int = 1, freq: Any = None) -> Self:
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
    _data: DatetimeArray | TimedeltaArray
    _comparables: list[str] = ['name', 'freq']
    _attributes: list[str] = ['name', 'freq']
    _is_monotonic_increasing: bool = Index.is_monotonic_increasing
    _is_monotonic_decreasing: bool = Index.is_monotonic_decreasing
    _is_unique: bool = Index.is_unique

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

    def _with_freq(self, freq: BaseOffset | None) -> Self:
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def values(self) -> np.ndarray:
        data = self._data._ndarray
        data = data.view()
        data.flags.writeable = False
        return data

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq: Any = None) -> Self:
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
    def inferred_freq(self) -> str | None:
        return self._data.inferred_freq

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        freq = cast(Tick, self.freq)
        tick = Timedelta(freq).as_unit(self.unit)._value
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def _can_range_setop(self, other: Self) -> bool:
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def _wrap_range_setop(self, other: Self, res_i8: RangeIndex | Index) -> Self:
        new_freq = None
        if not len(res_i8):
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            new_freq = to_offset(Timedelta(res_i8.step, unit=self.unit).as_unit(self.unit))
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        result = type(self._data)._simple_new(res_values, dtype=self.dtype, freq=new_freq)
        return cast(Self, self._wrap_setop_result(other, result))

    def _range_intersect(self, other: Self, sort: bool) -> Self:
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _range_union(self, other: Self, sort: bool) -> Self:
        left = self._as_range_index
        right = other._as_range