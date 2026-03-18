from typing import Any, Dict, Optional, Sequence, Tuple, Type
import numpy as np

get_window_bounds_doc: str = ...

class BaseIndexer:
    index_array: Any
    window_size: int

    def __init__(self, index_array: Any = ..., window_size: int = ..., **kwargs: Any) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    index: Any
    offset: Any

    def __init__(
        self,
        index_array: Any = ...,
        window_size: int = ...,
        index: Any = ...,
        offset: Any = ...,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class GroupbyIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: Any = ...,
        window_size: int = ...,
        groupby_indices: Optional[Dict[Any, Sequence[int]]] = ...,
        window_indexer: Type[BaseIndexer] = BaseIndexer,
        indexer_kwargs: Optional[Dict[str, Any]] = ...,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray, np.ndarray]: ...