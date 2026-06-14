from typing import Any, Mapping, Optional, Sequence, Tuple, Type
import numpy as np
from numpy.typing import NDArray
from pandas._libs.tslibs import BaseOffset
from pandas.core.indexes.datetimes import DatetimeIndex

get_window_bounds_doc: str = ...

class BaseIndexer:
    def __init__(self, index_array: Optional[np.ndarray] = ..., window_size: int = ..., **kwargs: Any) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: Optional[np.ndarray] = ...,
        window_size: int = ...,
        index: Optional[DatetimeIndex] = ...,
        offset: Optional[BaseOffset] = ...,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

class GroupbyIndexer(BaseIndexer):
    def __init__(
        self,
        index_array: Optional[np.ndarray] = ...,
        window_size: int = ...,
        groupby_indices: Optional[Mapping[object, Sequence[int]]] = ...,
        window_indexer: type[BaseIndexer] = ...,
        indexer_kwargs: Optional[dict[str, Any]] = ...,
        **kwargs: Any,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...