from __future__ import annotations
from datetime import timedelta
from typing import Any, Optional, Tuple, Union, Type, Dict, List
import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas.core.indexes.datetimes import DatetimeIndex

get_window_bounds_doc: str = ...

class BaseIndexer:
    index_array: Optional[np.ndarray]
    window_size: int

    def __init__(self, index_array: Optional[np.ndarray] = ..., window_size: int = 0, **kwargs: Any) -> None: ...

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    index: DatetimeIndex
    offset: BaseOffset

    def __init__(
        self,
        index_array: Optional[np.ndarray] = ...,
        window_size: int = 0,
        index: Optional[DatetimeIndex] = ...,
        offset: Optional[BaseOffset] = ...,
        **kwargs: Any,
    ) -> None: ...

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...

class GroupbyIndexer(BaseIndexer):
    groupby_indices: Dict[Any, List[int]]
    window_indexer: Type[BaseIndexer]
    indexer_kwargs: Dict[str, Any]

    def __init__(
        self,
        index_array: Optional[np.ndarray] = ...,
        window_size: Union[int, BaseIndexer] = 0,
        groupby_indices: Optional[Dict[Any, List[int]]] = ...,
        window_indexer: Type[BaseIndexer] = BaseIndexer,
        indexer_kwargs: Optional[Dict[str, Any]] = ...,
        **kwargs: Any,
    ) -> None: ...

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = ...,
        center: Optional[bool] = ...,
        closed: Optional[str] = ...,
        step: Optional[int] = ...,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any]]: ...