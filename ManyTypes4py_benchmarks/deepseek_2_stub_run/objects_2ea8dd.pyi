```python
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Type, Union
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas.core.indexes.datetimes import DatetimeIndex

get_window_bounds_doc: str = ...

class BaseIndexer:
    index_array: Any
    window_size: int
    
    def __init__(
        self,
        index_array: Any = ...,
        window_size: int = ...,
        **kwargs: Any
    ) -> None: ...
    
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    index: DatetimeIndex
    offset: BaseOffset
    
    def __init__(
        self,
        index_array: Any = ...,
        window_size: int = ...,
        index: Optional[DatetimeIndex] = ...,
        offset: Optional[BaseOffset] = ...,
        **kwargs: Any
    ) -> None: ...
    
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class GroupbyIndexer(BaseIndexer):
    groupby_indices: Dict[Any, Any]
    window_indexer: Type[BaseIndexer]
    indexer_kwargs: Dict[str, Any]
    
    def __init__(
        self,
        index_array: Any = ...,
        window_size: Union[int, BaseIndexer] = ...,
        groupby_indices: Optional[Dict[Any, Any]] = ...,
        window_indexer: Type[BaseIndexer] = ...,
        indexer_kwargs: Optional[Dict[str, Any]] = ...,
        **kwargs: Any
    ) -> None: ...
    
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
        win_type: Optional[str] = ...
    ) -> Tuple[np.ndarray, np.ndarray]: ...
```