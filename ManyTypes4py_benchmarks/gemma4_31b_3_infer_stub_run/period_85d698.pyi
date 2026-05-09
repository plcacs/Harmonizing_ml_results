from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Optional, Union, overload, Type, TypeVar
import numpy as np
from pandas._libs.tslibs import BaseOffset, NaT, Period, Resolution, Tick
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.arrays.period import PeriodArray
from pandas.core.indexes.datetimes import DatetimeIndex, Index

T = TypeVar("T", bound="PeriodIndex")

def _new_PeriodIndex(cls: Type[T], **d: Any) -> T: ...

class PeriodIndex(Index):
    _typ: str
    _data_cls: Type[PeriodArray]
    _supports_partial_string_indexing: bool

    def __new__(
        cls: Type[T],
        data: Optional[Union[np.ndarray, PeriodArray, Index, Any]] = None,
        freq: Optional[Union[str, Period]] = None,
        dtype: Optional[Union[str, PeriodDtype]] = None,
        copy: bool = False,
        name: Optional[str] = None,
    ) -> T: ...

    @property
    def _engine_type(self) -> Any: ...

    @property
    def _resolution_obj(self) -> Resolution: ...

    def asfreq(self, freq: Optional[Union[str, Period]] = None, how: str = 'E') -> T: ...

    def to_timestamp(self, freq: Optional[Union[str, Period]] = None, how: str = 'start') -> DatetimeIndex: ...

    @property
    def hour(self) -> Index: ...

    @property
    def minute(self) -> Index: ...

    @property
    def second(self) -> Index: ...

    @classmethod
    def from_fields(
        cls: Type[T],
        *,
        year: Optional[Union[int, np.ndarray, Any]] = None,
        quarter: Optional[Union[int, np.ndarray, Any]] = None,
        month: Optional[Union[int, np.ndarray, Any]] = None,
        day: Optional[Union[int, np.ndarray, Any]] = None,
        hour: Optional[Union[int, np.ndarray, Any]] = None,
        minute: Optional[Union[int, np.ndarray, Any]] = None,
        second: Optional[Union[int, np.ndarray, Any]] = None,
        freq: Optional[Union[str, Period]] = None,
    ) -> T: ...

    @classmethod
    def from_ordinals(cls: Type[T], ordinals: Union[list[int], np.ndarray], *, freq: Union[str, Period], name: Optional[str] = None) -> T: ...

    @property
    def values(self) -> np.ndarray[Any]: ...

    def _maybe_convert_timedelta(self, other: Union[timedelta, np.timedelta64, Tick, np.ndarray, BaseOffset, int]) -> Union[int, np.ndarray[Any]]: ...

    def _is_comparable_dtype(self, dtype: Any) -> bool: ...

    def asof_locs(self, where: Union[DatetimeIndex, PeriodIndex], mask: np.ndarray[bool]) -> np.ndarray[Any]: ...

    @property
    def is_full(self) -> bool: ...

    @property
    def inferred_type(self) -> str: ...

    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any: ...

    def get_loc(self, key: Union[Period, NaT, str, datetime]) -> Union[int, np.ndarray[Any]]: ...

    def _disallow_mismatched_indexing(self, key: Period) -> None: ...

    def _cast_partial_indexing_scalar(self, label: Any) -> Period: ...

    def _maybe_cast_slice_bound(self, label: Any, side: str) -> Any: ...

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: Any) -> tuple[Period, Period]: ...

    def shift(self, periods: int = 1, freq: Optional[Any] = None) -> T: ...

def period_range(
    start: Optional[Union[str, datetime, Any]],
    end: Optional[Union[str, datetime, Any]],
    periods: Optional[int] = None,
    freq: Optional[Union[str, BaseOffset]] = None,
    name: Optional[str] = None,
) -> PeriodIndex: ...