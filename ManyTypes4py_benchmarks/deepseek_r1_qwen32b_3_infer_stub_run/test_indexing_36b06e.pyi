import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
    NA,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Timedelta,
    Timestamp,
    array,
    date_range,
    interval_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm

class TestGetItem:
    def test_getitem(self, closed: str) -> None:
        ...
    
    def test_getitem_2d_deprecated(self) -> None:
        ...

class TestWhere:
    def test_where(self, listlike_box: Type[Index]) -> None:
        ...

class TestTake:
    def test_take(self, closed: str) -> None:
        ...

class TestGetLoc:
    @pytest.mark.parametrize('side', ['right', 'left', 'both', 'neither'])
    def test_get_loc_interval(self, closed: str, side: str) -> None:
        ...
    
    @pytest.mark.parametrize('scalar', [-1, 0, 0.5, 3, 4.5, 5, 6])
    def test_get_loc_scalar(self, closed: str, scalar: float) -> None:
        ...
    
    @pytest.mark.parametrize('scalar', [-1, 0, 0.5, 3, 4.5, 5, 6])
    def test_get_loc_length_one_scalar(self, scalar: float, closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('left, right', [(0, 5), (-1, 4), (-1, 6), (6, 7)])
    def test_get_loc_length_one_interval(
        self, left: float, right: float, closed: str, other_closed: str
    ) -> None:
        ...
    
    @pytest.mark.parametrize('breaks', [date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], ids=lambda x: str(x.dtype))
    def test_get_loc_datetimelike_nonoverlapping(self, breaks: Union[date_range, timedelta_range]) -> None:
        ...
    
    @pytest.mark.parametrize('arrays', [(date_range('20180101', periods=4), date_range('20180103', periods=4)), (date_range('20180101', periods=4, tz='US/Eastern'), date_range('20180103', periods=4, tz='US/Eastern')), (timedelta_range('0 days', periods=4), timedelta_range('2 days', periods=4))], ids=lambda x: str(x[0].dtype))
    def test_get_loc_datetimelike_overlapping(self, arrays: Tuple[Union[DatetimeIndex, TimedeltaIndex], ...]) -> None:
        ...
    
    @pytest.mark.parametrize('values', [date_range('2018-01-04', periods=4, freq='-1D'), date_range('2018-01-04', periods=4, freq='-1D', tz='US/Eastern'), timedelta_range('3 days', periods=4, freq='-1D'), np.arange(3.0, -1.0, -1.0), np.arange(3, -1, -1)], ids=lambda x: str(x.dtype))
    def test_get_loc_decreasing(self, values: Union[DatetimeIndex, TimedeltaIndex, np.ndarray]) -> None:
        ...
    
    @pytest.mark.parametrize('key', [[5], (2, 3)])
    def test_get_loc_non_scalar_errors(self, key: Union[List[int], Tuple[int, int]]) -> None:
        ...
    
    def test_get_indexer_with_nans(self) -> None:
        ...

class TestGetIndexer:
    @pytest.mark.parametrize('query, expected', [([Interval(2, 4, closed='right')], [1]), ([Interval(2, 4, closed='left')], [-1]), ...])
    def test_get_indexer_with_interval(self, query: List[Interval], expected: List[int]) -> None:
        ...
    
    @pytest.mark.parametrize('query, expected', [([-0.5], [-1]), ([0], [-1]), ...])
    def test_get_indexer_with_int_and_float(self, query: List[float], expected: List[int]) -> None:
        ...
    
    @pytest.mark.parametrize('item', [[3], np.arange(0.5, 5, 0.5)])
    def test_get_indexer_length_one(self, item: Union[List[Interval], np.ndarray], closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('size', [1, 5])
    def test_get_indexer_length_one_interval(self, size: int, closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('target', [IntervalIndex.from_tuples([(7, 8), (1, 2), (3, 4), (0, 1)]), ...])
    def test_get_indexer_categorical(self, target: Union[IntervalIndex, List[Interval], List[str], List[float]], ordered: bool) -> None:
        ...
    
    def test_get_indexer_categorical_with_nans(self) -> None:
        ...
    
    def test_get_indexer_datetime(self) -> None:
        ...
    
    @pytest.mark.parametrize('tuples, closed', [([(0, 2), (1, 3), (3, 4)], 'neither'), ...])
    def test_get_indexer_errors(self, tuples: List[Tuple[float, float]], closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('query, expected', [([-0.5], ([-1], [0])), ...])
    def test_get_indexer_non_unique_with_int_and_float(self, query: List[float], expected: Tuple[List[int], List[int]]) -> None:
        ...
    
    def test_get_indexer_non_monotonic(self) -> None:
        ...
    
    def test_get_indexer_with_nans(self) -> None:
        ...
    
    def test_get_index_non_unique_non_monotonic(self) -> None:
        ...
    
    def test_get_indexer_multiindex_with_intervals(self) -> None:
        ...
    
    @pytest.mark.parametrize('box', [IntervalIndex, array, list])
    def test_get_indexer_interval_index(self, box: Type[IntervalIndex]) -> None:
        ...
    
    def test_get_indexer_read_only(self) -> None:
        ...

class TestSliceLocs:
    def test_slice_locs_with_interval(self) -> None:
        ...
    
    def test_slice_locs_with_ints_and_floats_succeeds(self) -> None:
        ...
    
    @pytest.mark.parametrize('query', [[0, 1], [0, 2], [0, 3], [0, 4]])
    @pytest.mark.parametrize('tuples', [[(0, 2), (1, 3), (2, 4)], ...])
    def test_slice_locs_with_ints_and_floats_errors(self, tuples: List[Tuple[float, float]], query: List[int]) -> None:
        ...

class TestPutmask:
    @pytest.mark.parametrize('tz', ['US/Pacific', None])
    def test_putmask_dt64(self, tz: Optional[str]) -> None:
        ...
    
    def test_putmask_td64(self) -> None:
        ...

class TestContains:
    def test_contains_dunder(self) -> None:
        ...