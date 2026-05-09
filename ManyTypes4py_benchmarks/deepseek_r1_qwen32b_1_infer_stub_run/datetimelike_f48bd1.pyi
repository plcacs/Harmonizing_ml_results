"""
Stub file for 'datetimelike_f48bd1' module.
"""

from __future__ import annotations
from abc import ABC
from typing import (
    Any,
    Optional,
    Union,
    Sequence,
    List,
    Tuple,
    Final,
    Literal,
    Self,
    overload,
)
import numpy as np
from pandas._libs.tslibs import BaseOffset, Resolution, Tick
from pandas._libs.tslibs import to_offset
from pandas.core.indexes.base import Index
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.arrays import DatetimeArray, TimedeltaArray
from pandas.core.dtypes.dtypes import DatetimeTZDtype

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    _can_hold_strings: Final[bool] = False

    @property
    def freq(self) -> Optional[BaseOffset]:
        ...

    @freq.setter
    def freq(self, value: Optional[BaseOffset]) -> None:
        ...

    @property
    def asi8(self) -> np.ndarray:
        ...

    @property
    def freqstr(self) -> Optional[str]:
        ...

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        ...

    @cache_readonly
    def resolution(self) -> str:
        ...

    @cache_readonly
    def hasnans(self) -> bool:
        ...

    def equals(self, other: Index) -> bool:
        ...

    def __contains__(self, key: Any) -> bool:
        ...

    def _convert_tolerance(self, tolerance: Union[str, Timedelta], target: Any) -> Any:
        ...

    def _format_with_header(
        self, *, header: str, na_rep: str, date_format: Optional[str]
    ) -> str:
        ...

    @property
    def _formatter_func(self) -> Any:
        ...

    def _format_attrs(self) -> List[Tuple[str, str]]:
        ...

    def _summary(self, name: Optional[str] = None) -> str:
        ...

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        ...

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime) -> Tuple[Any, Any]:
        ...

    def _parse_with_reso(self, label: Union[str, datetime]) -> Tuple[datetime, Resolution]:
        ...

    def _get_string_slice(self, key: Union[str, datetime]) -> Any:
        ...

    @final
    def _partial_date_slice(self, reso: Resolution, parsed: datetime) -> Union[slice, np.ndarray]:
        ...

    def _maybe_cast_slice_bound(self, label: Any, side: Literal['left', 'right']) -> Any:
        ...

    @abstractmethod
    def shift(self, periods: int = 1, freq: Optional[Union[BaseOffset, str]] = None) -> Self:
        ...

    def _maybe_cast_listlike_indexer(self, keyarr: Any) -> Index:
        ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    _comparables: Final[List[str]] = ['name', 'freq']
    _attributes: Final[List[str]] = ['name', 'freq']

    @property
    def unit(self) -> str:
        ...

    def as_unit(self, unit: Literal['s', 'ms', 'us', 'ns']) -> Self:
        ...

    def _with_freq(self, freq: Optional[BaseOffset]) -> Self:
        ...

    @property
    def values(self) -> np.ndarray:
        ...

    def shift(self, periods: int = 1, freq: Optional[Union[BaseOffset, str]] = None) -> Self:
        ...

    @cache_readonly
    def inferred_freq(self) -> Optional[str]:
        ...

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        ...

    def _can_range_setop(self, other: Any) -> bool:
        ...

    def _wrap_range_setop(self, other: Any, res_i8: Any) -> Any:
        ...

    def _range_intersect(self, other: Any, sort: bool) -> Any:
        ...

    def _range_union(self, other: Any, sort: bool) -> Any:
        ...

    def _intersection(self, other: Any, sort: bool = False) -> Any:
        ...

    def _fast_intersect(self, other: Any, sort: bool) -> Any:
        ...

    def _can_fast_intersect(self, other: Any) -> bool:
        ...

    def _can_fast_union(self, other: Any) -> bool:
        ...

    def _fast_union(self, other: Any, sort: Optional[bool]) -> Any:
        ...

    def _union(self, other: Any, sort: bool) -> Any:
        ...

    def _get_join_freq(self, other: Any) -> Optional[BaseOffset]:
        ...

    def _wrap_join_result(
        self, joined: Any, other: Any, lidx: Any, ridx: Any, how: JoinHow
    ) -> Tuple[Any, Any, Any]:
        ...

    def _get_engine_target(self) -> np.ndarray:
        ...

    def _from_join_target(self, result: np.ndarray) -> Any:
        ...

    def _get_delete_freq(self, loc: Any) -> Optional[BaseOffset]:
        ...

    def _get_insert_freq(self, loc: Any, item: Any) -> Optional[BaseOffset]:
        ...

    def delete(self, loc: Any) -> Self:
        ...

    def insert(self, loc: int, item: Any) -> Self:
        ...

    def take(
        self,
        indices: Union[List[int], np.ndarray],
        axis: int = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any
    ) -> Self:
        ...