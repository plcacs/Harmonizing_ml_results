from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Optional, Union, Tuple, Type
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

if False:
    from collections.abc import Hashable
    from pandas._typing import Dtype, DtypeObj, Self, npt
else:
    Hashable = object
    Dtype = Any
    DtypeObj = Any
    Self = Any
    npt = np

_index_doc_kwargs: dict[str, Any] = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'target_klass': 'PeriodIndex or list of Periods'})
_shared_doc_kwargs: dict[str, Any] = {'klass': 'PeriodArray'}


def _new_PeriodIndex(cls: Type[PeriodIndex], **d: Any) -> PeriodIndex:
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
    """
    _typ: str = 'periodindex'
    _data: PeriodArray
    freq: BaseOffset
    dtype: PeriodDtype
    _data_cls = PeriodArray
    _supports_partial_string_indexing: bool = True

    @property
    def _engine_type(self) -> Any:
        return libindex.PeriodEngine

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        return self.dtype._resolution_obj

    @doc(PeriodArray.asfreq, other='arrays.PeriodArray', other_name='PeriodArray', **_shared_doc_kwargs)
    def asfreq(self, freq: Optional[Union[str, BaseOffset]] = None, how: str = 'E') -> PeriodIndex:
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq: Optional[Union[str, BaseOffset]] = None, how: str = 'start') -> DatetimeIndex:
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

    def __new__(cls: Type[Self],
                data: Any = None,
                freq: Optional[Union[str, BaseOffset]] = None,
                dtype: Optional[Dtype] = None,
                copy: bool = False,
                name: Optional[Hashable] = None) -> Self:
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
    def from_fields(cls: Type[Self],
                    *,
                    year: Optional[Any] = None,
                    quarter: Optional[Any] = None,
                    month: Optional[Any] = None,
                    day: Optional[Any] = None,
                    hour: Optional[Any] = None,
                    minute: Optional[Any] = None,
                    second: Optional[Any] = None,
                    freq: Optional[Union[str, BaseOffset]] = None) -> Self:
        fields: dict[str, Any] = {'year': year, 'quarter': quarter, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second}
        fields = {key: value for (key, value) in fields.items() if value is not None}
        arr = PeriodArray._from_fields(fields=fields, freq=freq)
        return cls._simple_new(arr)

    @classmethod
    def from_ordinals(cls: Type[Self],
                      ordinals: Any,
                      *,
                      freq: Union[str, BaseOffset],
                      name: Optional[Hashable] = None) -> Self:
        ordinals = np.asarray(ordinals, dtype=np.int64)
        dtype = PeriodDtype(freq)
        data = PeriodArray._simple_new(ordinals, dtype=dtype)
        return cls._simple_new(data, name=name)

    @property
    def values(self) -> npt.NDArray[np.object_]:
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other: Any) -> Union[int, npt.NDArray[np.int64]]:
        """
        Convert timedelta-like input to an integer multiple of self.freq
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

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return self.dtype == dtype

    def asof_locs(self, where: Index, mask: npt.NDArray[np.bool_]) -> np.ndarray:
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
        """
        orig_key = key
        self._check_indexing_error(key)
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT
        elif isinstance(key, str):
            try:
                (parsed, reso) = self._parse_with_reso(key)
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

    def _disallow_mismatched_indexing(self, key: Period) -> None:
        if key._dtype != self.dtype:
            raise KeyError(key)

    def _cast_partial_indexing_scalar(self, label: datetime) -> Period:
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

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime) -> Tuple[Period, Period]:
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        iv = Period(parsed, freq=freq)
        return (iv.asfreq(self.freq, how='start'), iv.asfreq(self.freq, how='end'))

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq: Optional[Any] = None) -> Self:
        if freq is not None:
            raise TypeError(f'`freq` argument is not supported for {type(self).__name__}.shift')
        return self + periods


def period_range(start: Optional[Union[str, datetime, Period, Any]] = None,
                 end: Optional[Union[str, datetime, Period, Any]] = None,
                 periods: Optional[int] = None,
                 freq: Optional[Union[str, BaseOffset]] = None,
                 name: Optional[Any] = None) -> PeriodIndex:
    """
    Return a fixed frequency PeriodIndex.
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError('Of the three parameters: start, end, and periods, exactly two must be specified')
    if freq is None and (not isinstance(start, Period) and (not isinstance(end, Period))):
        freq = 'D'
    (data, freq) = PeriodArray._generate_range(start, end, periods, freq)
    dtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name)