from typing import Any, Optional, Tuple, Dict, Type

get_window_bounds_doc: str = ...

class BaseIndexer:
    index_array: Any
    window_size: int

    def __init__(self, index_array: Optional[Any] = ..., window_size: int = ..., **kwargs: Any) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    index: Any
    offset: Any

    def __init__(
        self,
        index_array: Optional[Any] = ...,
        window_size: int = ...,
        index: Optional[Any] = ...,
        offset: Optional[Any] = ...,
        **kwargs: Any
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...

class GroupbyIndexer(BaseIndexer):
    groupby_indices: Dict[Any, Any]
    window_indexer: Type[BaseIndexer]
    indexer_kwargs: Dict[Any, Any]

    def __init__(
        self,
        index_array: Optional[Any] = ...,
        window_size: int = ...,
        groupby_indices: Optional[Dict[Any, Any]] = ...,
        window_indexer: Type[BaseIndexer] = ...,
        indexer_kwargs: Optional[Dict[Any, Any]] = ...,
        **kwargs: Any
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[Any, Any]: ...