from __future__ import annotations
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
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
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.period import PeriodArray
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.extension import inherit_names
from pandas._typing import (
    ArrayLike,
    Dtype,
    DtypeObj,
    Self,
    npt,
)

if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import (
        ArrayLike,
        Dtype,
        DtypeObj,
        Self,
        npt,
    )

class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: str
    _data_cls: type[PeriodArray]
    _supports_partial_string_indexing: bool

    def __new__(
        cls,
        data: Union[ArrayLike, PeriodArray, Index, ABCSeries] = None,
        freq: Union[str, BaseOffset] = None,
        dtype: Union[str, PeriodDtype] = None,
        copy: bool = False,
        name: Optional[str] = None,
    ) -> Self:
        ...

    @classmethod
    def from_fields(
        cls,
        *,
        year: Union[int, np.ndarray, ABCSeries] = None,
        quarter: Union[int, np.ndarray, ABCSeries] = None,
        month: Union[int, np.ndarray, ABCSeries] = None,
        day: Union[int, np.ndarray, ABCSeries] = None,
        hour: Union[int, np.ndarray, ABCSeries] = None,
        minute: Union[int, np.ndarray, ABCSeries] = None,
        second: Union[int, np.ndarray, ABCSeries] = None,
        freq: Union[str, BaseOffset] = None,
    ) -> Self:
        ...

    @classmethod
    def from_ordinals(
        cls,
        ordinals: npt.NDArray[np.int64],
        *,
        freq: Union[str, BaseOffset],
        name: Optional[str] = None,
    ) -> Self:
        ...

    @property
    def values(self) -> npt.NDArray[object]:
        ...

    def asfreq(self, freq: Optional[str] = None, how: str = 'E') -> Self:
        ...

    def to_timestamp(self, freq: Optional[str] = None, how: str = 'start') -> DatetimeIndex:
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

    def get_loc(self, key: Union[Period, NaT, str, datetime]) -> Union[int, npt.NDArray[np.int64]]:
        ...

    def shift(self, periods: int = 1) -> Self:
        ...

def period_range(
    start: Optional[Union[str, datetime, Period]] = None,
    end: Optional[Union[str, datetime, Period]] = None,
    periods: Optional[int] = None,
    freq: Union[str, BaseOffset] = 'D',
    name: Optional[str] = None,
) -> PeriodIndex:
    ...