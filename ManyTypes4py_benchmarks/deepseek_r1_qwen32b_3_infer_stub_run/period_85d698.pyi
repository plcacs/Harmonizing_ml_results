from __future__ import annotations
from datetime import datetime, timedelta
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Union,
    overload,
)
import numpy as np
from pandas._libs.tslibs import BaseOffset, NaT, Period, Resolution
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.arrays.period import PeriodArray
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex

class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: str
    _data_cls: type[PeriodArray]
    _supports_partial_string_indexing: bool

    def __new__(
        cls,
        data: Optional[Union[Index, ABCSeries, PeriodArray]] = None,
        freq: Optional[Union[str, Period]] = None,
        dtype: Optional[Union[str, PeriodDtype]] = None,
        copy: bool = False,
        name: Optional[str] = None,
    ) -> Self:
        ...

    @classmethod
    def from_fields(
        cls,
        *,
        year: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        quarter: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        month: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        day: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        hour: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        minute: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        second: Optional[Union[int, np.ndarray, ABCSeries]] = None,
        freq: Optional[Union[str, Period]] = None,
    ) -> PeriodIndex:
        ...

    @classmethod
    def from_ordinals(
        cls,
        ordinals: Union[np.ndarray, List[int]],
        *,
        freq: Union[str, Period],
        name: Optional[str] = None,
    ) -> PeriodIndex:
        ...

    @property
    def values(self) -> np.ndarray:
        ...

    @property
    def inferred_type(self) -> str:
        ...

    @property
    def is_full(self) -> bool:
        ...

    def get_loc(self, key: Any) -> Union[int, np.ndarray]:
        ...

    def asof_locs(
        self,
        where: Union[DatetimeIndex, PeriodIndex],
        mask: np.ndarray,
    ) -> np.ndarray:
        ...

    def shift(self, periods: int = 1, freq: Optional[Any] = None) -> PeriodIndex:
        ...

def period_range(
    start: Optional[Union[str, datetime, Period]] = None,
    end: Optional[Union[str, datetime, Period]] = None,
    periods: Optional[int] = None,
    freq: Optional[Union[str, BaseOffset]] = None,
    name: Optional[str] = None,
) -> PeriodIndex:
    ...