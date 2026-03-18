```python
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas.core.indexes.datetimes import DatetimeIndex

get_window_bounds_doc: str = ...

class BaseIndexer:
    index_array: Optional[np.ndarray]
    window_size: int
    
    def __init__(
        self,
        index_array: Optional[np.ndarray] = None,
        window_size: int = 0,
        **kwargs: Any
    ) -> None: ...
    
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    index: DatetimeIndex
    offset: BaseOffset
    
    def __init__(
        self,
        index_array: Optional[np.ndarray] = None,
        window_size: int = 0,
        index: Optional[DatetimeIndex] = None,
        offset: Optional[BaseOffset] = None,
        **kwargs: Any
    ) -> None: ...
    
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class GroupbyIndexer(BaseIndexer):
    groupby_indices: Dict[Any, Any]
    window_indexer: Any
    indexer_kwargs: Dict[str, Any]
    
    def __init__(
        self,
        index_array: Optional[np.ndarray] = None,
        window_size: Union[int, Any] = 0,
        groupby_indices: Optional[Dict[Any, Any]] = None,
        window_indexer: Any = BaseIndexer,
        indexer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None: ...
    
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]: ...
```