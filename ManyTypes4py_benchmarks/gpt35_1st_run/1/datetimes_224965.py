from __future__ import annotations
import datetime as dt
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import NaT, Period, Timestamp, index as libindex, lib
from pandas._libs.tslibs import Resolution, Tick, Timedelta, periods_per_day, timezones, to_offset
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.datetimes import DatetimeArray, tz_to_dtype
import pandas.core.common as com
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time
if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import Dtype, DtypeObj, Frequency, IntervalClosedType, Self, TimeAmbiguous, TimeNonexistent, npt
    from pandas.core.api import DataFrame, PeriodIndex
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR

def _new_DatetimeIndex(cls, d: dict) -> DatetimeIndex:
    ...

@inherit_names(DatetimeArray._field_ops + [method for method in DatetimeArray._datetimelike_methods if method not in ('tz_localize', 'tz_convert', 'strftime')], DatetimeArray, wrap=True)
@inherit_names(['is_normalized'], DatetimeArray, cache=True)
@inherit_names(['tz', 'tzinfo', 'dtype', 'to_pydatetime', 'date', 'time', 'timetz', 'std'] + DatetimeArray._bool_ops, DatetimeArray)
@set_module('pandas')
class DatetimeIndex(DatetimeTimedeltaMixin):
    ...

    @property
    def _engine_type(self) -> libindex.DatetimeEngine:
        ...

    @doc(DatetimeArray.strftime)
    def strftime(self, date_format: str) -> Index:
        ...

    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz) -> DatetimeIndex:
        ...

    @doc(DatetimeArray.tz_localize)
    def tz_localize(self, tz, ambiguous='raise', nonexistent='raise') -> DatetimeIndex:
        ...

    @doc(DatetimeArray.to_period)
    def to_period(self, freq=None) -> PeriodIndex:
        ...

    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self) -> Index:
        ...

    @doc(DatetimeArray.isocalendar)
    def isocalendar(self) -> DataFrame:
        ...

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        ...

    def __new__(cls, data=None, freq=lib.no_default, tz=lib.no_default, ambiguous='raise', dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None) -> DatetimeIndex:
        ...

    @cache_readonly
    def _is_dates_only(self) -> bool:
        ...

    def __reduce__(self) -> tuple:
        ...

    def _is_comparable_dtype(self, dtype) -> bool:
        ...

    @cache_readonly
    def _formatter_func(self):
        ...

    def _can_range_setop(self, other) -> bool:
        ...

    def _get_time_micros(self) -> np.ndarray:
        ...

    def snap(self, freq='S') -> DatetimeIndex:
        ...

    def _parsed_string_to_bounds(self, reso, parsed) -> tuple:
        ...

    def _parse_with_reso(self, label) -> tuple:
        ...

    def _disallow_mismatched_indexing(self, key) -> None:
        ...

    def get_loc(self, key) -> int:
        ...

    @doc(DatetimeTimedeltaMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side) -> Timestamp:
        ...

    def slice_indexer(self, start=None, end=None, step=None) -> slice:
        ...

    @property
    def inferred_type(self) -> str:
        ...

    def indexer_at_time(self, time, asof=False) -> np.ndarray:
        ...

    def indexer_between_time(self, start_time, end_time, include_start=True, include_end=True) -> np.ndarray:
        ...

@set_module('pandas')
def date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, inclusive='both', *, unit=None, **kwargs) -> DatetimeIndex:
    ...

@set_module('pandas')
def bdate_range(start=None, end=None, periods=None, freq='B', tz=None, normalize=True, name=None, weekmask=None, holidays=None, inclusive='both', **kwargs) -> DatetimeIndex:
    ...
