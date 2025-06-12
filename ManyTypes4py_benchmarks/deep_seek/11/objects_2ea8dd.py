"""Indexer objects for computing start/end window bounds for rolling operations"""
from __future__ import annotations
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import ensure_platform_int
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.tseries.offsets import Nano

get_window_bounds_doc = '\nComputes the bounds of a window.\n\nParameters\n----------\nnum_values : int, default 0\n    number of values that will be aggregated over\nwindow_size : int, default 0\n    the number of rows in a window\nmin_periods : int, default None\n    min_periods passed from the top level rolling API\ncenter : bool, default None\n    center passed from the top level rolling API\nclosed : str, default None\n    closed passed from the top level rolling API\nstep : int, default None\n    step passed from the top level rolling API\n    .. versionadded:: 1.5\nwin_type : str, default None\n    win_type passed from the top level rolling API\n\nReturns\n-------\nA tuple of ndarray[int64]s, indicating the boundaries of each\nwindow\n'

class BaseIndexer:
    """
    Base class for window bounds calculations.
    """

    def __init__(
        self,
        index_array: Optional[np.ndarray] = None,
        window_size: int = 0,
        **kwargs: Any
    ) -> None:
        self.index_array = index_array
        self.window_size = window_size
        for key, value in kwargs.items():
            setattr(self, key, value)

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class FixedWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of fixed length."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if center or self.window_size == 0:
            offset = (self.window_size - 1) // 2
        else:
            offset = 0
        end = np.arange(1 + offset, num_values + 1 + offset, step, dtype='int64')
        start = end - self.window_size
        if closed in ['left', 'both']:
            start -= 1
        if closed in ['left', 'neither']:
            end -= 1
        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)
        return (start, end)

class VariableWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of variable length, namely for time series."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return calculate_variable_window_bounds(num_values, self.window_size, min_periods, center, closed, self.index_array)

class VariableOffsetWindowIndexer(BaseIndexer):
    """
    Calculate window boundaries based on a non-fixed offset such as a BusinessDay.
    """

    def __init__(
        self,
        index_array: Optional[np.ndarray] = None,
        window_size: int = 0,
        index: Optional[DatetimeIndex] = None,
        offset: Optional[BaseOffset] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(index_array, window_size, **kwargs)
        if not isinstance(index, DatetimeIndex):
            raise ValueError('index must be a DatetimeIndex.')
        self.index = index
        if not isinstance(offset, BaseOffset):
            raise ValueError('offset must be a DateOffset-like object.')
        self.offset = offset

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if step is not None:
            raise NotImplementedError('step not implemented for variable offset window')
        if num_values <= 0:
            return (np.empty(0, dtype='int64'), np.empty(0, dtype='int64'))
        if closed is None:
            closed = 'right' if self.index is not None else 'both'
        right_closed = closed in ['right', 'both']
        left_closed = closed in ['left', 'both']
        if self.index[num_values - 1] < self.index[0]:
            index_growth_sign = -1
        else:
            index_growth_sign = 1
        offset_diff = index_growth_sign * self.offset
        start = np.empty(num_values, dtype='int64')
        start.fill(-1)
        end = np.empty(num_values, dtype='int64')
        end.fill(-1)
        start[0] = 0
        if right_closed:
            end[0] = 1
        else:
            end[0] = 0
        zero = timedelta(0)
        for i in range(1, num_values):
            end_bound = self.index[i]
            start_bound = end_bound - offset_diff
            if left_closed:
                start_bound -= Nano(1)
            start[i] = i
            for j in range(start[i - 1], i):
                start_diff = (self.index[j] - start_bound) * index_growth_sign
                if start_diff > zero:
                    start[i] = j
                    break
            end_diff = (self.index[end[i - 1]] - end_bound) * index_growth_sign
            if end_diff == zero and (not right_closed):
                end[i] = end[i - 1] + 1
            elif end_diff <= zero:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]
            if not right_closed:
                end[i] -= 1
        return (start, end)

class ExpandingIndexer(BaseIndexer):
    """Calculate expanding window bounds, mimicking df.expanding()"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (np.zeros(num_values, dtype=np.int64), np.arange(1, num_values + 1, dtype=np.int64))

class FixedForwardWindowIndexer(BaseIndexer):
    """
    Creates window boundaries for fixed-length windows that include the current row.
    """

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if center:
            raise ValueError("Forward-looking windows can't have center=True")
        if closed is not None:
            raise ValueError("Forward-looking windows don't support setting the closed argument")
        if step is None:
            step = 1
        start = np.arange(0, num_values, step, dtype='int64')
        end = start + self.window_size
        if self.window_size:
            end = np.clip(end, 0, num_values)
        return (start, end)

class GroupbyIndexer(BaseIndexer):
    """Calculate bounds to compute groupby rolling, mimicking df.groupby().rolling()"""

    def __init__(
        self,
        index_array: Optional[np.ndarray] = None,
        window_size: int = 0,
        groupby_indices: Optional[Dict[Any, np.ndarray]] = None,
        window_indexer: type = BaseIndexer,
        indexer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        self.groupby_indices = groupby_indices or {}
        self.window_indexer = window_indexer
        self.indexer_kwargs = indexer_kwargs.copy() if indexer_kwargs else {}
        super().__init__(index_array=index_array, window_size=self.indexer_kwargs.pop('window_size', window_size), **kwargs)

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        start_arrays = []
        end_arrays = []
        window_indices_start = 0
        for indices in self.groupby_indices.values():
            if self.index_array is not None:
                index_array = self.index_array.take(ensure_platform_int(indices))
            else:
                index_array = self.index_array
            indexer = self.window_indexer(index_array=index_array, window_size=self.window_size, **self.indexer_kwargs)
            start, end = indexer.get_window_bounds(len(indices), min_periods, center, closed, step)
            start = start.astype(np.int64)
            end = end.astype(np.int64)
            assert len(start) == len(end), 'these should be equal in length from get_window_bounds'
            window_indices = np.arange(window_indices_start, window_indices_start + len(indices))
            window_indices_start += len(indices)
            window_indices = np.append(window_indices, [window_indices[-1] + 1]).astype(np.int64, copy=False)
            start_arrays.append(window_indices.take(ensure_platform_int(start)))
            end_arrays.append(window_indices.take(ensure_platform_int(end)))
        if len(start_arrays) == 0:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        start = np.concatenate(start_arrays)
        end = np.concatenate(end_arrays)
        return (start, end)

class ExponentialMovingWindowIndexer(BaseIndexer):
    """Calculate ewm window bounds (the entire window)"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: Optional[int] = None,
        center: Optional[bool] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        win_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([0], dtype=np.int64), np.array([num_values], dtype=np.int64))
