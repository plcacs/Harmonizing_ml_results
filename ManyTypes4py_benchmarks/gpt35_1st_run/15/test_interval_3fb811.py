from typing import List, Tuple, Union
import pandas as pd
from pandas import Index, Interval, IntervalIndex, Timedelta, Timestamp, date_range, interval_range

class TestIntervalIndex:
    def create_index(self, closed: str = 'right') -> IntervalIndex:
    def create_index_with_nan(self, closed: str = 'right') -> IntervalIndex:
    def test_properties(self, closed: str) -> None:
    def test_length(self, closed: str, breaks: List[Union[List[int], List[float], pd.DatetimeIndex, pd.TimedeltaIndex]]) -> None:
    def test_with_nans(self, closed: str) -> None:
    def test_copy(self, closed: str) -> None:
    def test_ensure_copied_data(self, closed: str) -> None:
    def test_delete(self, closed: str) -> None:
    def test_insert(self, data: Union[pd.IntervalIndex, pd.Interval, pd.TimedeltaIndex]) -> None:
    def test_is_unique_interval(self, closed: str) -> None:
    def test_monotonic(self, closed: str) -> None:
    def test_is_monotonic_with_nans(self) -> None:
    def test_maybe_convert_i8(self, breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex]) -> None:
    def test_maybe_convert_i8_nat(self, breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex]) -> None:
    def test_maybe_convert_i8_numeric(self, make_key: callable, any_real_numpy_dtype: type) -> None:
    def test_maybe_convert_i8_numeric_identical(self, make_key: callable, any_real_numpy_dtype: type) -> None:
    def test_maybe_convert_i8_errors(self, breaks1: Union[pd.DatetimeIndex, pd.TimedeltaIndex], breaks2: Union[pd.DatetimeIndex, pd.TimedeltaIndex], make_key: callable) -> None:
    def test_contains_method(self) -> None:
    def test_dropna(self, closed: str) -> None:
    def test_non_contiguous(self, closed: str) -> None:
    def test_isin(self, closed: str) -> None:
    def test_comparison(self) -> None:
    def test_missing_values(self, closed: str) -> None:
    def test_sort_values(self, closed: str) -> None:
    def test_datetime(self, tz: str) -> None:
    def test_append(self, closed: str) -> None:
    def test_is_non_overlapping_monotonic(self, closed: str) -> None:
    def test_is_overlapping(self, start: Union[Timestamp, Timedelta], shift: Union[Timedelta, Timestamp], na_value: Union[pd.NA, Timestamp], closed: str) -> None:
    def test_to_tuples(self, tuples: List[Tuple]) -> None:
    def test_to_tuples_na(self, tuples: List[Tuple], na_tuple: bool) -> None:
    def test_nbytes(self) -> None:
    def test_set_closed(self, name: str, closed: str, other_closed: str) -> None:
    def test_set_closed_errors(self, bad_closed: Union[str, int]) -> None:
    def test_is_all_dates(self) -> None

def test_dir() -> None:
def test_searchsorted_different_argument_classes(listlike_box: callable) -> None:
def test_searchsorted_invalid_argument(arg: List[Union[int, str, Timestamp]]) -> None:
