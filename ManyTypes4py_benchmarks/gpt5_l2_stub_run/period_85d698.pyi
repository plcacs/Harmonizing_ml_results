from __future__ import annotations

from datetime import datetime, timedelta
from collections.abc import Hashable, Sequence
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset, Period, Resolution, Tick
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.arrays.period import PeriodArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex, Index


def _new_PeriodIndex(cls: type[PeriodIndex], **d: object) -> PeriodIndex: ...


class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: str
    _data_cls: type[PeriodArray]
    _supports_partial_string_indexing: bool

    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]: ...

    @property
    def _resolution_obj(self) -> Resolution: ...

    def asfreq(self, freq: str | BaseOffset | Tick | None = ..., how: str = ...) -> PeriodIndex: ...
    def to_timestamp(self, freq: str | BaseOffset | Tick | None = ..., how: str = ...) -> DatetimeIndex: ...

    @property
    def hour(self) -> Index: ...
    @property
    def minute(self) -> Index: ...
    @property
    def second(self) -> Index: ...

    def __new__(
        cls,
        data: object | None = ...,
        freq: str | BaseOffset | Tick | None = ...,
        dtype: str | PeriodDtype | None = ...,
        copy: bool = ...,
        name: Hashable | None = ...,
    ) -> PeriodIndex: ...

    @classmethod
    def from_fields(
        cls,
        *,
        year: object | None = ...,
        quarter: object | None = ...,
        month: object | None = ...,
        day: object | None = ...,
        hour: object | None = ...,
        minute: object | None = ...,
        second: object | None = ...,
        freq: str | BaseOffset | Tick | None = ...,
    ) -> PeriodIndex: ...

    @classmethod
    def from_ordinals(
        cls,
        ordinals: Sequence[int] | np.ndarray,
        *,
        freq: str | BaseOffset | Tick,
        name: Hashable | None = ...,
    ) -> PeriodIndex: ...

    @property
    def values(self) -> np.ndarray: ...

    def _maybe_convert_timedelta(
        self, other: timedelta | np.timedelta64 | Tick | BaseOffset | int | np.ndarray
    ) -> int | np.ndarray: ...
    def _is_comparable_dtype(self, dtype: object) -> bool: ...
    def asof_locs(self, where: DatetimeIndex | PeriodIndex, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def is_full(self) -> bool: ...
    @property
    def inferred_type(self) -> str: ...

    def _convert_tolerance(self, tolerance: object, target: PeriodIndex) -> object: ...
    def get_loc(self, key: Period | str | datetime | object) -> int | np.ndarray: ...
    def _disallow_mismatched_indexing(self, key: Period) -> None: ...
    def _cast_partial_indexing_scalar(self, label: object) -> Period: ...
    def _maybe_cast_slice_bound(self, label: object, side: str) -> object: ...
    def _parsed_string_to_bounds(self, reso: Resolution, parsed: object) -> tuple[Period, Period]: ...
    def shift(self, periods: int = ..., freq: None = ...) -> PeriodIndex: ...


def period_range(
    start: object | None = ...,
    end: object | None = ...,
    periods: int | None = ...,
    freq: str | BaseOffset | Tick | None = ...,
    name: Hashable | None = ...,
) -> PeriodIndex: ...