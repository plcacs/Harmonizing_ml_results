```python
from __future__ import annotations
from typing import Any, overload
from datetime import datetime, timedelta
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset, NaTType, Period, Resolution, Tick
from pandas.core.arrays.period import PeriodArray
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.base import Index

def _new_PeriodIndex(cls: Any, **d: Any) -> Any: ...

class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: str = ...
    _data_cls: type[PeriodArray] = ...
    _supports_partial_string_indexing: bool = ...
    
    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]: ...
    
    @property
    def _resolution_obj(self) -> Resolution: ...
    
    def asfreq(self, freq: Any = None, how: str = 'E') -> PeriodIndex: ...
    
    def to_timestamp(self, freq: Any = None, how: str = 'start') -> DatetimeIndex: ...
    
    @property
    def hour(self) -> Index: ...
    
    @property
    def minute(self) -> Index: ...
    
    @property
    def second(self) -> Index: ...
    
    def __new__(
        cls,
        data: Any = None,
        freq: Any = None,
        dtype: Any = None,
        copy: bool = False,
        name: Any = None
    ) -> PeriodIndex: ...
    
    @classmethod
    def from_fields(
        cls,
        *,
        year: Any = None,
        quarter: Any = None,
        month: Any = None,
        day: Any = None,
        hour: Any = None,
        minute: Any = None,
        second: Any = None,
        freq: Any = None
    ) -> PeriodIndex: ...
    
    @classmethod
    def from_ordinals(
        cls,
        ordinals: Any,
        *,
        freq: Any,
        name: Any = None
    ) -> PeriodIndex: ...
    
    @property
    def values(self) -> np.ndarray: ...
    
    def _maybe_convert_timedelta(self, other: Any) -> Any: ...
    
    def _is_comparable_dtype(self, dtype: Any) -> bool: ...
    
    def asof_locs(self, where: Any, mask: np.ndarray) -> Any: ...
    
    @property
    def is_full(self) -> bool: ...
    
    @property
    def inferred_type(self) -> str: ...
    
    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any: ...
    
    def get_loc(self, key: Any) -> Any: ...
    
    def _disallow_mismatched_indexing(self, key: Period) -> None: ...
    
    def _cast_partial_indexing_scalar(self, label: Any) -> Period: ...
    
    def _maybe_cast_slice_bound(self, label: Any, side: str) -> Any: ...
    
    def _parsed_string_to_bounds(self, reso: Any, parsed: Any) -> tuple[Period, Period]: ...
    
    def shift(self, periods: int = 1, freq: Any = None) -> PeriodIndex: ...
    
    # Inherited properties from PeriodArray via inherit_names decorator
    @property
    def strftime(self) -> Any: ...
    
    @property
    def start_time(self) -> Any: ...
    
    @property
    def end_time(self) -> Any: ...
    
    @property
    def is_leap_year(self) -> Any: ...
    
    # Field operations from PeriodArray
    @property
    def day(self) -> Any: ...
    
    @property
    def dayofweek(self) -> Any: ...
    
    @property
    def day_of_week(self) -> Any: ...
    
    @property
    def dayofyear(self) -> Any: ...
    
    @property
    def day_of_year(self) -> Any: ...
    
    @property
    def days_in_month(self) -> Any: ...
    
    @property
    def daysinmonth(self) -> Any: ...
    
    @property
    def freq(self) -> Any: ...
    
    @property
    def freqstr(self) -> Any: ...
    
    @property
    def month(self) -> Any: ...
    
    @property
    def quarter(self) -> Any: ...
    
    @property
    def qyear(self) -> Any: ...
    
    @property
    def week(self) -> Any: ...
    
    @property
    def weekday(self) -> Any: ...
    
    @property
    def weekofyear(self) -> Any: ...
    
    @property
    def year(self) -> Any: ...

def period_range(
    start: Any = None,
    end: Any = None,
    periods: Any = None,
    freq: Any = None,
    name: Any = None
) -> PeriodIndex: ...
```