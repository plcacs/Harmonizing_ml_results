from typing import List, Tuple, Union
import pandas as pd
from pandas import Index, Interval, IntervalIndex, Timedelta, Timestamp

class TestIntervalIndex:
    index: IntervalIndex = IntervalIndex.from_arrays([0, 1], [1, 2])

    def create_index(self, closed: str = 'right') -> IntervalIndex:
        return IntervalIndex.from_breaks(range(11), closed=closed)

    def create_index_with_nan(self, closed: str = 'right') -> IntervalIndex:
        mask = [True, False] + [True] * 8
        return IntervalIndex.from_arrays(np.where(mask, np.arange(10), np.nan), np.where(mask, np.arange(1, 11), np.nan), closed=closed)

    def test_properties(self, closed: str) -> None:
        ...

    def test_length(self, closed: str, breaks: List[Union[int, float, pd.Timestamp, pd.Timedelta]]) -> None:
        ...

    def test_with_nans(self, closed: str) -> None:
        ...

    def test_copy(self, closed: str) -> None:
        ...

    def test_ensure_copied_data(self, closed: str) -> None:
        ...

    def test_delete(self, closed: str) -> None:
        ...

    def test_insert(self, data: IntervalIndex) -> None:
        ...

    def test_is_unique_interval(self, closed: str) -> None:
        ...

    def test_monotonic(self, closed: str) -> None:
        ...

    def test_is_monotonic_with_nans(self) -> None:
        ...

    def test_maybe_convert_i8(self, breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex]) -> None:
        ...

    def test_maybe_convert_i8_nat(self, breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex]) -> None:
        ...

    def test_maybe_convert_i8_numeric(self, make_key: callable, any_real_numpy_dtype: np.dtype) -> None:
        ...

    def test_maybe_convert_i8_numeric_identical(self, make_key: callable, any_real_numpy_dtype: np.dtype) -> None:
        ...

    def test_maybe_convert_i8_errors(self, breaks1: Union[pd.DatetimeIndex, pd.TimedeltaIndex], breaks2: Union[pd.DatetimeIndex, pd.TimedeltaIndex], make_key: callable) -> None:
        ...

    def test_contains_method(self) -> None:
        ...

    def test_dropna(self, closed: str) -> None:
        ...

    def test_non_contiguous(self, closed: str) -> None:
        ...

    def test_isin(self, closed: str) -> None:
        ...

    def test_comparison(self) -> None:
        ...

    def test_missing_values(self, closed: str) -> None:
        ...

    def test_sort_values(self, closed: str) -> None:
        ...

    def test_datetime(self, tz: str) -> None:
        ...

    def test_append(self, closed: str) -> None:
        ...

    def test_is_non_overlapping_monotonic(self, closed: str) -> None:
        ...

    def test_is_overlapping(self, start: Union[pd.Timestamp, pd.Timedelta], shift: Union[pd.Timedelta, pd.Timestamp], na_value: Union[pd.Timestamp, pd.Timedelta], closed: str) -> None:
        ...

    def test_to_tuples(self, tuples: List[Tuple]) -> None:
        ...

    def test_to_tuples_na(self, tuples: List[Union[int, float, pd.Timestamp, pd.Timedelta]], na_tuple: bool) -> None:
        ...

    def test_nbytes(self) -> None:
        ...

    def test_set_closed(self, name: str, closed: str, other_closed: str) -> None:
        ...

    def test_set_closed_errors(self, bad_closed: Union[str, int]) -> None:
        ...

    def test_is_all_dates(self) -> None:
        ...

    def test_dir(self) -> None:
        ...

    def test_searchsorted_different_argument_classes(self, listlike_box: List) -> None:
        ...

    def test_searchsorted_invalid_argument(self, arg: List) -> None:
        ...
