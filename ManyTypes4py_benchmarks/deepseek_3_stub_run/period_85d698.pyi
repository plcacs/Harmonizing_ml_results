from __future__ import annotations
from typing import (
    Any,
    ClassVar,
    Literal,
    overload,
    TYPE_CHECKING,
    Union,
    Optional,
    cast,
)
from datetime import datetime, timedelta
import numpy as np
from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset, NaT, Period, Resolution, Tick
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.period import PeriodArray, period_array, raise_on_incompatible, validate_dtype_freq
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex, Index
from pandas.core.indexes.extension import inherit_names

if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
        ArrayLike,
        ScalarLike,
        Frequency,
        NaTType,
    )
    from pandas import Series

_index_doc_kwargs: dict[str, Any] = ...
_shared_doc_kwargs: dict[str, str] = ...

def _new_PeriodIndex(cls: type[PeriodIndex], **d: Any) -> PeriodIndex: ...

@inherit_names(['strftime', 'start_time', 'end_time'] + PeriodArray._field_ops, PeriodArray, wrap=True)
@inherit_names(['is_leap_year'], PeriodArray)
@set_module('pandas')
class PeriodIndex(DatetimeIndexOpsMixin):
    _typ: ClassVar[str] = ...
    _data_cls: ClassVar[type[PeriodArray]] = ...
    _supports_partial_string_indexing: ClassVar[bool] = ...
    
    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]: ...
    
    @cache_readonly
    def _resolution_obj(self) -> Resolution: ...
    
    @doc(PeriodArray.asfreq, other='arrays.PeriodArray', other_name='PeriodArray', **_shared_doc_kwargs)
    def asfreq(self, freq: Frequency | None = None, how: str = 'E') -> Self: ...
    
    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq: Frequency | None = None, how: str = 'start') -> DatetimeIndex: ...
    
    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Index: ...
    
    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Index: ...
    
    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Index: ...
    
    def __new__(
        cls,
        data: ArrayLike | None = None,
        freq: Frequency | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None
    ) -> Self: ...
    
    @classmethod
    def from_fields(
        cls,
        *,
        year: int | ArrayLike | Series | None = None,
        quarter: int | ArrayLike | Series | None = None,
        month: int | ArrayLike | Series | None = None,
        day: int | ArrayLike | Series | None = None,
        hour: int | ArrayLike | Series | None = None,
        minute: int | ArrayLike | Series | None = None,
        second: int | ArrayLike | Series | None = None,
        freq: Frequency | None = None
    ) -> Self: ...
    
    @classmethod
    def from_ordinals(
        cls,
        ordinals: ArrayLike,
        *,
        freq: Frequency,
        name: Hashable | None = None
    ) -> Self: ...
    
    @property
    def values(self) -> npt.NDArray[np.object_]: ...
    
    def _maybe_convert_timedelta(
        self,
        other: timedelta | np.timedelta64 | Tick | BaseOffset | int | np.ndarray
    ) -> int | npt.NDArray[np.int64]: ...
    
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool: ...
    
    def asof_locs(
        self,
        where: DatetimeIndex | PeriodIndex,
        mask: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.intp]: ...
    
    @property
    def is_full(self) -> bool: ...
    
    @property
    def inferred_type(self) -> Literal['period']: ...
    
    def _convert_tolerance(
        self,
        tolerance: Any,
        target: Index
    ) -> Any: ...
    
    def get_loc(
        self,
        key: Period | NaTType | str | datetime
    ) -> int | npt.NDArray[np.int64]: ...
    
    def _disallow_mismatched_indexing(self, key: Period) -> None: ...
    
    def _cast_partial_indexing_scalar(self, label: datetime | Period) -> Period: ...
    
    @doc(DatetimeIndexOpsMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(
        self,
        label: Period | datetime | Any,
        side: Literal['left', 'right']
    ) -> Period: ...
    
    def _parsed_string_to_bounds(
        self,
        reso: Resolution,
        parsed: datetime
    ) -> tuple[Period, Period]: ...
    
    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq: Any = None) -> Self: ...

def period_range(
    start: str | datetime | Period | None = None,
    end: str | datetime | Period | None = None,
    periods: int | None = None,
    freq: Frequency | None = None,
    name: Hashable | None = None
) -> PeriodIndex: ...