from __future__ import annotations
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Union, Dict, List, Tuple, cast
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
_shared_doc_kwargs: Dict[str, str] = {'klass': 'PeriodArray'}


def func_8x3ogkx4(cls: Any, **d: Any) -> Any:
    values: Any = d.pop('data')
    if values.dtype == 'int64':
        freq: Optional[Any] = d.pop('freq', None)
        dtype: PeriodDtype = PeriodDtype(freq)
        values = PeriodArray(values, dtype=dtype)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)


@inherit_names(['strftime', 'start_time', 'end_time'] + PeriodArray.
    _field_ops, PeriodArray, wrap=True)
@inherit_names(['is_leap_year'], PeriodArray)
@set_module('pandas')
class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: str = 'periodindex'
    _data_cls: type = PeriodArray
    _supports_partial_string_indexing: bool = True

    @property
    def func_a02eylc5(self) -> Any:
        return libindex.PeriodEngine

    @cache_readonly
    def func_rp8iljxj(self) -> Resolution:
        return self.dtype._resolution_obj

    @doc(PeriodArray.asfreq, other='arrays.PeriodArray', other_name=
        'PeriodArray', **_shared_doc_kwargs)
    def func_506j5ls5(self, freq: Optional[Any] = None, how: str = 'E') -> Self:
        arr: PeriodArray = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def func_u1mz7rty(self, freq: Optional[Any] = None, how: str = 'start') -> DatetimeIndex:
        arr: np.ndarray = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)

    @property
    @doc(PeriodArray.hour.fget)
    def func_jl6n8usr(self) -> Index:
        return Index(self._data.hour, name=self.name)

    @property
    @doc(PeriodArray.minute.fget)
    def func_oyegjm5t(self) -> Index:
        return Index(self._data.minute, name=self.name)

    @property
    @doc(PeriodArray.second.fget)
    def func_wkqdwkbr(self) -> Index:
        return Index(self._data.second, name=self.name)

    def __new__(cls, data: Optional[Any] = None, freq: Optional[Any] = None, dtype: Optional[Dtype] = None, copy: bool = False, name: Optional[Union[str, Hashable]] = None) -> Self:
        refs: Optional[Any] = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references
        name = maybe_extract_name(name, data, cls)
        freq = validate_dtype_freq(dtype, freq)
        if freq and isinstance(data, cls) and data.freq != freq:
            data = data.asfreq(freq)
        data = period_array(data=data, freq=freq)
        if copy:
            data = data.copy()
        return cls._simple_new(data, name=name, refs=refs)

    @classmethod
    def func_4qg0xxig(cls, *, year: Optional[Union[int, np.ndarray, ABCSeries]] = None, quarter: Optional[Union[int, np.ndarray, ABCSeries]] = None, month: Optional[Union[int, np.ndarray, ABCSeries]] = None, day: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        hour: Optional[Union[int, np.ndarray, ABCSeries]] = None, minute: Optional[Union[int, np.ndarray, ABCSeries]] = None, second: Optional[Union[int, np.ndarray, ABCSeries]] = None, freq: Optional[Any] = None) -> Self:
        fields: Dict[str, Optional[Union[int, np.ndarray, ABCSeries]]] = {'year': year, 'quarter': quarter, 'month': month, 'day':
            day, 'hour': hour, 'minute': minute, 'second': second}
        fields = {key: value for key, value in fields.items() if value is not
            None}
        arr: PeriodArray = PeriodArray._from_fields(fields=fields, freq=freq)
        return cls._simple_new(arr)

    @classmethod
    def func_xy3pt5rx(cls, ordinals: Union[List[int], np.ndarray], *, freq: Any, name: Optional[Union[str, Hashable]] = None) -> Self:
        ordinals = np.asarray(ordinals, dtype=np.int64)
        dtype: PeriodDtype = PeriodDtype(freq)
        data: PeriodArray = PeriodArray._simple_new(ordinals, dtype=dtype)
        return cls._simple_new(data, name=name)

    @property
    def func_wydzpyq9(self) -> np.ndarray:
        return np.asarray(self, dtype=object)

    def func_ttvmwnlu(self, other: Any) -> Union[int, np.ndarray]:
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, Tick):
                delta: Union[int, np.ndarray] = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if other.base == self.freq.base:
                return other.n
            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            assert isinstance(other, int)
            return other
        raise raise_on_incompatible(self, None)

    def func_d27ge8qb(self, dtype: DtypeObj) -> bool:
        return self.dtype == dtype

    def func_48tpj0if(self, where: Any, mask: np.ndarray) -> np.ndarray:
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq)
        elif not isinstance(where, PeriodIndex):
            raise TypeError(
                'asof_locs `where` must be DatetimeIndex or PeriodIndex')
        return super().asof_locs(where, mask)

    @property
    def func_z6es85sn(self) -> bool:
        if len(self) == 0:
            return True
        if not self.is_monotonic_increasing:
            raise ValueError('Index is not monotonic')
        values: np.ndarray = self.asi8
        return bool((values[1:] - values[:-1] < 2).all())

    @property
    def func_6j88ei23(self) -> str:
        return 'period'

    def func_whide0vw(self, tolerance: Any, target: Any) -> Any:
        tolerance = super()._convert_tolerance(tolerance, target)
        if self.dtype == target.dtype:
            tolerance = self._maybe_convert_timedelta(tolerance)
        return tolerance

    def func_ybclxbyt(self, key: Any) -> Union[int, np.ndarray]:
        orig_key: Any = key
        self._check_indexing_error(key)
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT
        elif isinstance(key, str):
            try:
                parsed: Any
                reso: Any
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

    def func_x3s6b7p3(self, key: Any) -> None:
        if key._dtype != self.dtype:
            raise KeyError(key)

    def func_a7a7toeh(self, label: Any) -> Period:
        try:
            period: Period = Period(label, freq=self.freq)
        except ValueError as err:
            raise KeyError(label) from err
        return period

    @doc(DatetimeIndexOpsMixin._maybe_cast_slice_bound)
    def func_wz2cztgj(self, label: Any, side: str) -> Any:
        if isinstance(label, datetime):
            label = self._cast_partial_indexing_scalar(label)
        return super()._maybe_cast_slice_bound(label, side)

    def func_1i5a94zb(self, reso: Any, parsed: Any) -> Tuple[Period, Period]:
        freq: str = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        iv: Period = Period(parsed, freq=freq)
        return iv.asfreq(self.freq, how='start'), iv.asfreq(self.freq, how=
            'end')

    @doc(DatetimeIndexOpsMixin.shift)
    def func_6p1eiqgb(self, periods: int = 1, freq: Optional[Any] = None) -> Self:
        if freq is not None:
            raise TypeError(
                f'`freq` argument is not supported for {type(self).__name__}.shift'
                )
        return self + periods


def func_fu8sjihl(start: Optional[Any] = None, end: Optional[Any] = None, periods: Optional[int] = None, freq: Optional[Any] = None, name: Optional[Union[str, Hashable]] = None) -> PeriodIndex:
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            'Of the three parameters: start, end, and periods, exactly two must be specified'
            )
    if freq is None and (not isinstance(start, Period) and not isinstance(
        end, Period)):
        freq = 'D'
    data: np.ndarray
    data, freq = PeriodArray._generate_range(start, end, periods, freq)
    dtype: PeriodDtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name)
