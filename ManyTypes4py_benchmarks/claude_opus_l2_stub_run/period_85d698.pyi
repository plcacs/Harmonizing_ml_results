from __future__ import annotations

from collections.abc import Hashable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset, NaT, Period, Resolution, Tick
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR

from pandas.core.dtypes.dtypes import PeriodDtype

from pandas.core.arrays.period import PeriodArray
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex

if TYPE_CHECKING:
    from pandas._typing import Dtype, DtypeObj, Self

_index_doc_kwargs: dict[str, str]
_shared_doc_kwargs: dict[str, str]

def _new_PeriodIndex(cls: type, **d: object) -> PeriodIndex: ...

class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: str
    _data_cls: type[PeriodArray]
    _supports_partial_string_indexing: bool

    # Inherited from PeriodArray via inherit_names (wrap=True)
    def strftime(self, date_format: str) -> Index: ...
    @property
    def start_time(self) -> DatetimeIndex: ...
    @property
    def end_time(self) -> DatetimeIndex: ...
    @property
    def day_of_week(self) -> Index: ...
    @property
    def day(self) -> Index: ...
    @property
    def dayofweek(self) -> Index: ...
    @property
    def dayofyear(self) -> Index: ...
    @property
    def day_of_year(self) -> Index: ...
    @property
    def days_in_month(self) -> Index: ...
    @property
    def daysinmonth(self) -> Index: ...
    @property
    def month(self) -> Index: ...
    @property
    def quarter(self) -> Index: ...
    @property
    def qyear(self) -> Index: ...
    @property
    def week(self) -> Index: ...
    @property
    def weekday(self) -> Index: ...
    @property
    def weekofyear(self) -> Index: ...
    @property
    def year(self) -> Index: ...

    # Inherited from PeriodArray via inherit_names (no wrap)
    @property
    def is_leap_year(self) -> npt.NDArray[np.bool_]: ...

    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]: ...
    @property
    def _resolution_obj(self) -> Resolution: ...

    def asfreq(self, freq: str | BaseOffset | None = ..., how: str = ...) -> PeriodIndex: ...
    def to_timestamp(self, freq: str | BaseOffset | None = ..., how: str = ...) -> DatetimeIndex: ...

    @property
    def hour(self) -> Index: ...
    @property
    def minute(self) -> Index: ...
    @property
    def second(self) -> Index: ...

    def __new__(
        cls,
        data: object = ...,
        freq: str | BaseOffset | None = ...,
        dtype: Dtype | None = ...,
        copy: bool = ...,
        name: Hashable | None = ...,
    ) -> PeriodIndex: ...

    @classmethod
    def from_fields(
        cls,
        *,
        year: int | npt.NDArray[np.int64] | None = ...,
        quarter: int | npt.NDArray[np.int64] | None = ...,
        month: int | npt.NDArray[np.int64] | None = ...,
        day: int | npt.NDArray[np.int64] | None = ...,
        hour: int | npt.NDArray[np.int64] | None = ...,
        minute: int | npt.NDArray[np.int64] | None = ...,
        second: int | npt.NDArray[np.int64] | None = ...,
        freq: str | BaseOffset | None = ...,
    ) -> PeriodIndex: ...

    @classmethod
    def from_ordinals(
        cls,
        ordinals: npt.ArrayLike,
        *,
        freq: str | BaseOffset,
        name: Hashable | None = ...,
    ) -> PeriodIndex: ...

    @property
    def values(self) -> np.ndarray: ...
    def _maybe_convert_timedelta(self, other: timedelta | np.timedelta64 | BaseOffset | Tick | int | np.ndarray) -> int | npt.NDArray[np.int64]: ...
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool: ...
    def asof_locs(self, where: DatetimeIndex | PeriodIndex, mask: np.ndarray) -> npt.NDArray[np.intp]: ...

    @property
    def is_full(self) -> bool: ...
    @property
    def inferred_type(self) -> str: ...

    def _convert_tolerance(self, tolerance: object, target: Index) -> object: ...
    def get_loc(self, key: Period | str | datetime) -> int | slice | np.ndarray: ...
    def _disallow_mismatched_indexing(self, key: Period) -> None: ...
    def _cast_partial_indexing_scalar(self, label: object) -> Period: ...
    def _maybe_cast_slice_bound(self, label: object, side: str) -> object: ...
    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime) -> tuple[Period, Period]: ...
    def shift(self, periods: int = ..., freq: object = ...) -> PeriodIndex: ...

def period_range(
    start: str | datetime | Period | None = ...,
    end: str | datetime | Period | None = ...,
    periods: int | None = ...,
    freq: str | BaseOffset | None = ...,
    name: Hashable | None = ...,
) -> PeriodIndex: ...