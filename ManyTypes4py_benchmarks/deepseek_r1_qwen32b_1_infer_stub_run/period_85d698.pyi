from __future__ import annotations
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import numpy as np
from numpy import ndarray
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    Period,
    Resolution,
    Tick,
)
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays import PeriodArray
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex

if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import (
        Dtype,
        DtypeObj,
        npt,
    )

Self = TypeVar("Self", bound="PeriodIndex")

class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: ClassVar[str] = "periodindex"
    _data_cls: ClassVar[Type[PeriodArray]] = PeriodArray
    _supports_partial_string_indexing: ClassVar[bool] = True

    def __new__(
        cls,
        data: Optional[Any] = None,
        freq: Optional[Any] = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Optional[str] = None,
    ) -> PeriodIndex:
        ...

    @classmethod
    def from_fields(
        cls,
        *,
        year: Optional[Union[int, ndarray, ABCSeries]] = None,
        quarter: Optional[Union[int, ndarray, ABCSeries]] = None,
        month: Optional[Union[int, ndarray, ABCSeries]] = None,
        day: Optional[Union[int, ndarray, ABCSeries]] = None,
        hour: Optional[Union[int, ndarray, ABCSeries]] = None,
        minute: Optional[Union[int, ndarray, ABCSeries]] = None,
        second: Optional[Union[int, ndarray, ABCSeries]] = None,
        freq: Optional[Any] = None,
    ) -> PeriodIndex:
        ...

    @classmethod
    def from_ordinals(
        cls,
        ordinals: Union[ndarray, Iterable[int]],
        *,
        freq: Any,
        name: Optional[str] = None,
    ) -> PeriodIndex:
        ...

    @property
    def values(self) -> ndarray:
        ...

    @property
    def hour(self) -> Index:
        ...

    @property
    def minute(self) -> Index:
        ...

    @property
    def second(self) -> Index:
        ...

    def get_loc(self, key: Union[Period, str, datetime, NaT]) -> Union[int, ndarray]:
        ...

    def shift(self, periods: int = 1, freq: Optional[Any] = None) -> PeriodIndex:
        ...

    def asfreq(self, freq: Optional[Any] = None, how: Literal["E", "S"] = "E") -> PeriodIndex:
        ...

    def to_timestamp(self, freq: Optional[Any] = None, how: Literal["start", "end"] = "start") -> DatetimeIndex:
        ...

    def _maybe_convert_timedelta(self, other: Union[timedelta, np.timedelta64, Tick, BaseOffset, int, ndarray]) -> Union[int, ndarray]:
        ...

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        ...

    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any:
        ...

    @property
    def inferred_type(self) -> str:
        ...

    @property
    def is_full(self) -> bool:
        ...

    def __add__(self, other: Union[int, PeriodIndex, ndarray]) -> PeriodIndex:
        ...

    def __sub__(self, other: Union[int, PeriodIndex, ndarray]) -> PeriodIndex:
        ...

def period_range(
    start: Optional[Union[str, datetime, Period]] = None,
    end: Optional[Union[str, datetime, Period]] = None,
    periods: Optional[int] = None,
    freq: Optional[Any] = None,
    name: Optional[str] = None,
) -> PeriodIndex:
    ...