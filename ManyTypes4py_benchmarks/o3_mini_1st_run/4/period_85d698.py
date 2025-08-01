from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, TYPE_CHECKING, overload, Sequence
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset, NaT, Period, Resolution, Tick
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.period import PeriodArray, period_array, raise_on_incompatible, validate_dtype_freq
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex, Index
from pandas.core.indexes.extension import inherit_names

if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import Dtype, DtypeObj, Self, npt

_index_doc_kwargs: Dict[str, Any] = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'target_klass': 'PeriodIndex or list of Periods'})
_shared_doc_kwargs: Dict[str, Any] = {'klass': 'PeriodArray'}


def _new_PeriodIndex(cls: type[PeriodIndex], **d: Any) -> PeriodIndex:
    values = d.pop('data')
    if values.dtype == 'int64':
        freq = d.pop('freq', None)
        dtype = PeriodDtype(freq)
        values = PeriodArray(values, dtype=dtype)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)


@inherit_names(['strftime', 'start_time', 'end_time'] + PeriodArray._field_ops, PeriodArray, wrap=True)
@inherit_names(['is_leap_year'], PeriodArray)
@set_module('pandas')
class PeriodIndex(DatetimeIndexOpsMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    dtype : str or PeriodDtype, default None
        A dtype from which to extract a freq.
    copy : bool
        Make a copy of input ndarray.
    name : str, default None
        Name of the resulting PeriodIndex.

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp
    from_fields
    from_ordinals

    Raises
    ------
    ValueError
        Passing the parameter data as a list without specifying either freq or
        dtype will raise a ValueError: "freq not specified and cannot be inferred"

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    Examples
    --------
    >>> idx = pd.PeriodIndex(data=["2000Q1", "2002Q3"], freq="Q")
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """
    _typ: str = 'periodindex'
    _data_cls = PeriodArray
    _supports_partial_string_indexing: bool = True

    @property
    def _engine_type(self) -> Any:
        return libindex.PeriodEngine

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        return self.dtype._resolution_obj

    @doc(PeriodArray.asfreq, other='arrays.PeriodArray', other_name='PeriodArray', **_shared_doc_kwargs)
    def asfreq(self, freq: Optional[Any] = None, how: str = 'E') -> PeriodIndex:
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq: Optional[Any] = None, how: str = 'start') -> DatetimeIndex:
        arr = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)

    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Index:
        return Index(self._data.hour, name=self.name)

    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Index:
        return Index(self._data.minute, name=self.name)

    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Index:
        return Index(self._data.second, name=self.name)

    def __new__(
        cls: type[PeriodIndex],
        data: Optional[Any] = None,
        freq: Optional[Any] = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Optional[str] = None
    ) -> PeriodIndex:
        refs: Optional[Any] = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references
        name = maybe_extract_name(name, data, cls)
        freq = validate_dtype_freq(dtype, freq)
        if freq and isinstance(data, cls) and (data.freq != freq):
            data = data.asfreq(freq)
        data = period_array(data=data, freq=freq)
        if copy:
            data = data.copy()
        return cls._simple_new(data, name=name, refs=refs)

    @classmethod
    def from_fields(
        cls: type[PeriodIndex],
        *,
        year: Optional[Any] = None,
        quarter: Optional[Any] = None,
        month: Optional[Any] = None,
        day: Optional[Any] = None,
        hour: Optional[Any] = None,
        minute: Optional[Any] = None,
        second: Optional[Any] = None,
        freq: Optional[Any] = None
    ) -> PeriodIndex:
        """
        Construct a PeriodIndex from fields (year, month, day, etc.).

        Parameters
        ----------
        year : int, array, or Series, default None
            Year for the PeriodIndex.
        quarter : int, array, or Series, default None
            Quarter for the PeriodIndex.
        month : int, array, or Series, default None
            Month for the PeriodIndex.
        day : int, array, or Series, default None
            Day for the PeriodIndex.
        hour : int, array, or Series, default None
            Hour for the PeriodIndex.
        minute : int, array, or Series, default None
            Minute for the PeriodIndex.
        second : int, array, or Series, default None
            Second for the PeriodIndex.
        freq : str or period object, optional
            One of pandas period strings or corresponding objects.

        Returns
        -------
        PeriodIndex
        """
        fields: Dict[str, Any] = {
            'year': year,
            'quarter': quarter,
            'month': month,
            'day': day,
            'hour': hour,
            'minute': minute,
            'second': second
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        arr = PeriodArray._from_fields(fields=fields, freq=freq)
        return cls._simple_new(arr)

    @classmethod
    def from_ordinals(
        cls: type[PeriodIndex],
        ordinals: Union[Sequence[int], np.ndarray],
        *,
        freq: Any,
        name: Optional[str] = None
    ) -> PeriodIndex:
        """
        Construct a PeriodIndex from ordinals.

        Parameters
        ----------
        ordinals : array-like of int
            The period offsets from the proleptic Gregorian epoch.
        freq : str or period object
            One of pandas period strings or corresponding objects.
        name : str, default None
            Name of the resulting PeriodIndex.

        Returns
        -------
        PeriodIndex
        """
        ordinals_arr: np.ndarray = np.asarray(ordinals, dtype=np.int64)
        dtype = PeriodDtype(freq)
        data = PeriodArray._simple_new(ordinals_arr, dtype=dtype)
        return cls._simple_new(data, name=name)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other: Any) -> Union[int, np.ndarray]:
        """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]
        """
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, Tick):
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if other.base == self.freq.base:
                return other.n
            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            assert isinstance(other, int)
            return other
        raise raise_on_incompatible(self, None)

    def _is_comparable_dtype(self, dtype: Any) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return self.dtype == dtype

    def asof_locs(self, where: Any, mask: np.ndarray) -> Any:
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq)
        elif not isinstance(where, PeriodIndex):
            raise TypeError('asof_locs `where` must be DatetimeIndex or PeriodIndex')
        return super().asof_locs(where, mask)

    @property
    def is_full(self) -> bool:
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic_increasing:
            raise ValueError('Index is not monotonic')
        values = self.asi8
        return bool((values[1:] - values[:-1] < 2).all())

    @property
    def inferred_type(self) -> str:
        return 'period'

    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any:
        tolerance = super()._convert_tolerance(tolerance, target)
        if self.dtype == target.dtype:
            tolerance = self._maybe_convert_timedelta(tolerance)
        return tolerance

    def get_loc(self, key: Any) -> Union[int, np.ndarray]:
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : Period, NaT, str, or datetime
            String or datetime key must be parsable as Period.

        Returns
        -------
        loc : int or ndarray[int64]
        """
        orig_key = key
        self._check_indexing_error(key)
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT
        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except ValueError as err:
                raise KeyError(f"Cannot interpret '{key}' as period") from err
            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err
            if reso == self._resolution_obj:
                key = self._cast_partial_indexing_scalar(parsed)
            else:
                raise KeyError(key)
        elif isinstance(key, Period):
            self._disallow_mismatched_indexing(key)
        elif isinstance(key, datetime):
            key = self._cast_partial_indexing_scalar(key)
        else:
            raise KeyError(key)
        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    def _disallow_mismatched_indexing(self, key: Any) -> None:
        if key._dtype != self.dtype:
            raise KeyError(key)

    def _cast_partial_indexing_scalar(self, label: Any) -> Period:
        try:
            period = Period(label, freq=self.freq)
        except ValueError as err:
            raise KeyError(label) from err
        return period

    @doc(DatetimeIndexOpsMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label: Any, side: str) -> Any:
        if isinstance(label, datetime):
            label = self._cast_partial_indexing_scalar(label)
        return super()._maybe_cast_slice_bound(label, side)

    def _parsed_string_to_bounds(self, reso: Any, parsed: Any) -> tuple[Any, Any]:
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        iv: Period = Period(parsed, freq=freq)
        return (iv.asfreq(self.freq, how='start'), iv.asfreq(self.freq, how='end'))

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq: Optional[Any] = None) -> PeriodIndex:
        if freq is not None:
            raise TypeError(f'`freq` argument is not supported for {type(self).__name__}.shift')
        return self + periods


def period_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: Optional[Any] = None,
    name: Optional[str] = None
) -> PeriodIndex:
    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str, datetime, date, pandas.Timestamp, or period-like, default None
        Left bound for generating periods.
    end : str, datetime, date, pandas.Timestamp, or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    See Also
    --------
    date_range : Returns a fixed frequency DatetimeIndex.
    Period : Represents a period of time.
    PeriodIndex : Immutable ndarray holding ordinal values indicating regular periods
        in time.
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    if freq is None and (not isinstance(start, Period) and (not isinstance(end, Period))):
        freq = 'D'
    data, freq = PeriodArray._generate_range(start, end, periods, freq)
    dtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name)