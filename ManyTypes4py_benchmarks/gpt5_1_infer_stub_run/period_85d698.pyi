from __future__ import annotations

from collections.abc import Hashable, Iterable
from datetime import datetime, timedelta
import numpy as np
from numpy.typing import NDArray
from pandas._libs.tslibs import BaseOffset, Period, Resolution
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex, Index
from typing import Any, Self


class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: str
    _data_cls: Any
    _supports_partial_string_indexing: bool

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
        data: object | None = None,
        freq: str | Period | BaseOffset | None = None,
        dtype: str | PeriodDtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self: ...
    @classmethod
    def from_fields(
        cls,
        *,
        year: object | None = None,
        quarter: object | None = None,
        month: object | None = None,
        day: object | None = None,
        hour: object | None = None,
        minute: object | None = None,
        second: object | None = None,
        freq: str | Period | BaseOffset | None = None,
    ) -> PeriodIndex: ...
    @classmethod
    def from_ordinals(
        cls,
        ordinals: Iterable[int] | NDArray[np.int64],
        *,
        freq: str | Period | BaseOffset,
        name: Hashable | None = None,
    ) -> PeriodIndex: ...
    @property
    def values(self) -> NDArray[object]: ...
    def _maybe_convert_timedelta(
        self,
        other: timedelta | np.timedelta64 | BaseOffset | int | NDArray[object],
    ) -> int | NDArray[np.int64]: ...
    def _is_comparable_dtype(self, dtype: object) -> bool: ...
    def asof_locs(
        self, where: DatetimeIndex | PeriodIndex, mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.intp], NDArray[np.bool_]]: ...
    @property
    def is_full(self) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def _convert_tolerance(self, tolerance: object, target: PeriodIndex) -> object: ...
    def get_loc(self, key: object) -> int | NDArray[np.int64]: ...
    def _disallow_mismatched_indexing(self, key: Period) -> None: ...
    def _cast_partial_indexing_scalar(self, label: object) -> Period: ...
    def _maybe_cast_slice_bound(self, label: object, side: str) -> object: ...
    def _parsed_string_to_bounds(self, reso: Resolution, parsed: object) -> tuple[Period, Period]: ...
    def shift(self, periods: int = 1, freq: object | None = None) -> PeriodIndex: ...


def period_range(
    start: object | None = None,
    end: object | None = None,
    periods: int | None = None,
    freq: str | BaseOffset | None = None,
    name: Hashable | None = None,
) -> PeriodIndex: ...