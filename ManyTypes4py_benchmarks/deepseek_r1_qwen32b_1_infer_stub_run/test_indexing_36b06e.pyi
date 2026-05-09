import numpy as np
from datetime import datetime, timedelta
from pandas import Interval, IntervalIndex, Timestamp
from typing import Any, Callable, List, Optional, Tuple, Union

class TestGetItem:
    def test_getitem(self, closed: Optional[str]) -> None:
        ...

class TestWhere:
    def test_where(self, listlike_box: Optional[type]) -> None:
        ...

class TestTake:
    def test_take(self, closed: Optional[str]) -> None:
        ...

class TestGetLoc:
    def test_get_loc_interval(self, closed: str, side: str) -> None:
        ...

    def test_get_loc_scalar(self, closed: str, scalar: Union[int, float]) -> None:
        ...

    def test_get_loc_length_one_scalar(self, scalar: Union[int, float, datetime, Timestamp], closed: str) -> None:
        ...

    def test_get_loc_length_one_interval(self, left: Union[int, float, datetime, Timestamp], right: Union[int, float, datetime, Timestamp], closed: str, other_closed: str) -> None:
        ...

    def test_get_loc_datetimelike_nonoverlapping(self, breaks: Union[np.ndarray, List[datetime], List[timedelta]]) -> None:
        ...

    def test_get_loc_datetimelike_overlapping(self, arrays: Tuple[Union[np.ndarray, List[datetime], List[timedelta]], ...]) -> None:
        ...

    def test_get_loc_decreasing(self, values: Union[np.ndarray, List[datetime], List[timedelta]]) -> None:
        ...

    def test_get_loc_non_scalar_errors(self, key: Union[List[int], Tuple[int, int]]) -> None:
        ...

    def test_get_indexer_with_nans(self) -> None:
        ...

class TestGetIndexer:
    def test_get_indexer_with_interval(self, query: List[Interval], expected: List[int]) -> None:
        ...

    def test_get_indexer_with_int_and_float(self, query: List[Union[int, float]], expected: List[int]) -> None:
        ...

    def test_get_indexer_length_one(self, item: Union[List[int], np.ndarray], closed: str) -> None:
        ...

    def test_get_indexer_length_one_interval(self, size: int, closed: str) -> None:
        ...

    def test_get_indexer_categorical(self, target: Union[IntervalIndex, List[Union[int, float, str]]], ordered: bool) -> None:
        ...

    def test_get_indexer_categorical_with_nans(self) -> None:
        ...

    def test_get_indexer_datetime(self) -> None:
        ...

    def test_get_indexer_errors(self, tuples: List[Tuple[Union[int, float], Union[int, float]]], closed: str) -> None:
        ...

    def test_get_indexer_non_unique_with_int_and_float(self, query: List[Union[int, float]], expected: Tuple[List[int], List[int]]) -> None:
        ...

    def test_get_indexer_non_monotonic(self) -> None:
        ...

    def test_get_indexer_with_nans(self) -> None:
        ...

    def test_get_index_non_unique_non_monotonic(self) -> None:
        ...

    def test_get_indexer_multiindex_with_intervals(self) -> None:
        ...

    def test_get_indexer_interval_index(self, box: Callable) -> None:
        ...

    def test_get_indexer_read_only(self) -> None:
        ...

class TestSliceLocs:
    def test_slice_locs_with_interval(self) -> None:
        ...

    def test_slice_locs_with_ints_and_floats_succeeds(self) -> None:
        ...

    def test_slice_locs_with_ints_and_floats_errors(self, tuples: List[Tuple[Union[int, float], Union[int, float]]], query: Tuple[Union[int, float], Union[int, float]]) -> None:
        ...

class TestPutmask:
    def test_putmask_dt64(self, tz: Optional[str]) -> None:
        ...

    def test_putmask_td64(self) -> None:
        ...

class TestContains:
    def test_contains_dunder(self) -> None:
        ...