```python
from __future__ import annotations

from collections.abc import Hashable
from datetime import datetime, timedelta
from typing import Any, ClassVar, overload

import numpy as np
from pandas._libs.tslibs import BaseOffset, NaT, Period, Resolution, Tick
from pandas._typing import Dtype, DtypeObj, Self, npt
from pandas.core.arrays.period import PeriodArray
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex

_index_doc_kwargs: dict[str, Any]
_shared_doc_kwargs: dict[str, str]

def _new_PeriodIndex(cls: type[PeriodIndex], **d: Any) -> PeriodIndex: ...

class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: ClassVar[str]
    _data_cls: ClassVar[type[PeriodArray]]
    _supports_partial_string_indexing: ClassVar[bool]

    @property
    def _engine_type(self) -> type[Any]: ...
    @property
    def _resolution_obj(self) -> Resolution: ...
    def asfreq(self, freq: str | BaseOffset | None = None, how: str = "E") -> PeriodIndex: ...
    def to_timestamp(self, freq: str | BaseOffset | None = None, how: str = "start") -> DatetimeIndex: ...
    @property
    def hour(self) -> Index: ...
    @property
    def minute(self) -> Index: ...
    @property
    def second(self) -> Index: ...
    def __new__(
        cls,
        data: Any = None,
        freq: str | BaseOffset | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self: ...
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
        freq: str | BaseOffset | None = None,
    ) -> PeriodIndex: ...
    @classmethod
    def from_ordinals(
        cls,
        ordinals: npt.ArrayLike,
        *,
        freq: str | BaseOffset,
        name: Hashable | None = None,
    ) -> PeriodIndex: ...
    @property
    def values(self) -> np.ndarray[Any, np.dtype[np.object_]]: ...
    def _maybe_convert_timedelta(
        self,
        other: timedelta | np.timedelta64 | Tick | np.ndarray | BaseOffset | int,
    ) -> int | np.ndarray[Any, np.dtype[np.int64]]: ...
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool: ...
    def asof_locs(self, where: DatetimeIndex | PeriodIndex, mask: np.ndarray[Any, np.dtype[np.bool_]]) -> Any: ...
    @property
    def is_full(self) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any: ...
    def get_loc(self, key: Period | Any | str | datetime) -> int | np.ndarray[Any, np.dtype[np.intp]]: ...
    def _disallow_mismatched_indexing(self, key: Period) -> None: ...
    def _cast_partial_indexing_scalar(self, label: Any) -> Period: ...
    def _maybe_cast_slice_bound(self, label: datetime | Any, side: str) -> Any: ...
    def _parsed_string_to_bounds(self, reso: Resolution, parsed: Any) -> tuple[Period, Period]: ...
    def shift(self, periods: int = 1, freq: Any = None) -> PeriodIndex: ...

def period_range(
    start: str | datetime | Any | None = None,
    end: str | datetime | Any | None = None,
    periods: int | None = None,
    freq: str | BaseOffset | None = None,
    name: Hashable | None = None,
) -> PeriodIndex: ...
```