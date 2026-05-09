from __future__ import annotations
import numpy as np
from datetime import timedelta
from pandas._libs.tslibs import BaseOffset
from pandas.core.indexes.datetimes import DatetimeIndex
from typing import (
    Any,
    Optional,
    Tuple,
    Union,
    Dict,
    List,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from numpy import ndarray

class BaseIndexer:
    def __init__(self, index_array: Optional[np.ndarray] = None, window_size: int = 0, **kwargs: Any) -> None: ...
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(self, index_array: Optional[np.ndarray] = None, window_size: int = 0, index: DatetimeIndex, offset: BaseOffset, **kwargs: Any) -> None: ...
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

class GroupbyIndexer(BaseIndexer):
    def __init__(self, index_array: Optional[np.ndarray] = None, window_size: Union[int, BaseIndexer] = 0, groupby_indices: Optional[Dict[Any, List[int]]] = None, window_indexer: type[BaseIndexer] = BaseIndexer, indexer_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None: ...
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray[np.int64], np.ndarray[np.int64]]: ...