from __future__ import annotations

import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas.core.indexes.datetimes import DatetimeIndex

get_window_bounds_doc: str = ...

class BaseIndexer:
    index_array: np.ndarray | None
    window_size: int

    def __init__(
        self,
        index_array: np.ndarray | None = ...,
        window_size: int = ...,
        **kwargs: object,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class FixedWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class VariableWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class VariableOffsetWindowIndexer(BaseIndexer):
    index: DatetimeIndex
    offset: BaseOffset

    def __init__(
        self,
        index_array: np.ndarray | None = ...,
        window_size: int = ...,
        index: DatetimeIndex | None = ...,
        offset: BaseOffset | None = ...,
        **kwargs: object,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class ExpandingIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class FixedForwardWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class GroupbyIndexer(BaseIndexer):
    groupby_indices: dict[object, list[int]]
    window_indexer: type[BaseIndexer]
    indexer_kwargs: dict[str, object]

    def __init__(
        self,
        index_array: np.ndarray | None = ...,
        window_size: int = ...,
        groupby_indices: dict[object, list[int]] | None = ...,
        window_indexer: type[BaseIndexer] = ...,
        indexer_kwargs: dict[str, object] | None = ...,
        **kwargs: object,
    ) -> None: ...
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...

class ExponentialMovingWindowIndexer(BaseIndexer):
    def get_window_bounds(
        self,
        num_values: int = ...,
        min_periods: int | None = ...,
        center: bool | None = ...,
        closed: str | None = ...,
        step: int | None = ...,
    ) -> tuple[np.ndarray, np.ndarray]: ...