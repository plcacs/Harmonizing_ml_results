"""
Stub file for 'datetimelike_f48bd1' module
"""

from __future__ import annotations
from abc import ABC
from typing import (
    Any,
    Optional,
    Union,
    Tuple,
    List,
    Final,
    TypeVar,
    overload,
    Sequence,
)
import numpy as np
from pandas._libs import (
    NaT,
    Timedelta,
    BaseOffset,
    Resolution,
    Tick,
    to_offset,
)
from pandas._libs.tslibs import parsing
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
    PeriodArray,
    DatetimeLikeArrayMixin,
)
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas._typing import (
    Axis,
    JoinHow,
    Self,
    npt,
)

T = TypeVar("T")

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    _can_hold_strings: Final[bool] = False

    @property
    def mean(self) -> Any:
        ...

    @property
    def freq(self) -> Optional[BaseOffset]:
        ...

    @freq.setter
    def freq(self, value: Optional[BaseOffset]) -> None:
        ...

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        ...

    @property
    def freqstr(self) -> Optional[str]:
        ...

    @property
    def resolution(self) -> Resolution:
        ...

    @property
    def hasnans(self) -> bool:
        ...

    def equals(self, other: Index) -> bool:
        ...

    def __contains__(self, key: Any) -> bool:
        ...

    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any:
        ...

    def shift(self, periods: int = 1, freq: Optional[Union[BaseOffset, str]] = None) -> Any:
        ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    _comparables: Final[List[str]] = ['name', 'freq']
    _attributes: Final[List[str]] = ['name', 'freq']

    @property
    def unit(self) -> str:
        ...

    def as_unit(self, unit: str) -> Self:
        ...

    def shift(self, periods: int = 1, freq: Optional[Union[BaseOffset, str]] = None) -> Self:
        ...

    @property
    def inferred_freq(self) -> Optional[BaseOffset]:
        ...

    def _as_range_index(self) -> RangeIndex:
        ...

    def _can_range_setop(self, other: Any) -> bool:
        ...

    def _wrap_range_setop(self, other: Any, res_i8: Any) -> Self:
        ...

    def _range_intersect(self, other: Any, sort: bool) -> Self:
        ...

    def _range_union(self, other: Any, sort: bool) -> Self:
        ...

    def _intersection(self, other: Any, sort: bool = False) -> Self:
        ...

    def _fast_intersect(self, other: Any, sort: bool) -> Self:
        ...

    def _can_fast_intersect(self, other: Any) -> bool:
        ...

    def _can_fast_union(self, other: Any) -> bool:
        ...

    def _fast_union(self, other: Any, sort: Optional[bool]) -> Self:
        ...

    def _union(self, other: Any, sort: bool) -> Self:
        ...

    def _get_join_freq(self, other: Any) -> Optional[BaseOffset]:
        ...

    def _wrap_join_result(
        self, joined: Any, other: Any, lidx: Any, ridx: Any, how: JoinHow
    ) -> Tuple[Index, Any, Any]:
        ...

    def _get_engine_target(self) -> npt.NDArray[np.int64]:
        ...

    def _from_join_target(self, result: Any) -> Self:
        ...

    def _get_delete_freq(self, loc: Any) -> Optional[BaseOffset]:
        ...

    def _get_insert_freq(self, loc: Any, item: Any) -> Optional[BaseOffset]:
        ...

    def delete(self, loc: Any) -> Self:
        ...

    def insert(self, loc: Any, item: Any) -> Self:
        ...

    def take(
        self,
        indices: Any,
        axis: int = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any
    ) -> Self:
        ...