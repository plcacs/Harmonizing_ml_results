from pandas import (
    Interval,
    IntervalIndex,
    Index,
    Timestamp,
    Timedelta,
    NaT,
    Series,
    DataFrame,
)
from typing import Any, List, Optional, Union, Dict, Tuple, Callable, Sequence

class TestIntervalIndex:
    index: IntervalIndex

    def test_properties(self, closed: str) -> None:
        ...

    def test_length(self, closed: str, breaks: Union[List[int], List[float], List[Timestamp], List[Timedelta]]) -> None:
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

    def test_maybe_convert_i8(self, breaks: Union[List[Timestamp], List[Timedelta]]) -> None:
        ...

    def test_maybe_convert_i8_nat(self, breaks: Union[List[Timestamp], List[Timedelta]]) -> None:
        ...

    def test_maybe_convert_i8_numeric(self, make_key: Callable, any_real_numpy_dtype: Any) -> None:
        ...

    def test_maybe_convert_i8_numeric_identical(self, make_key: Callable, any_real_numpy_dtype: Any) -> None:
        ...

    def test_maybe_convert_i8_errors(self, breaks1: Union[List[Timestamp], List[Timedelta]], breaks2: Union[List[Timestamp], List[Timedelta]], make_key: Callable) -> None:
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

    def test_datetime(self, tz: Optional[str]) -> None:
        ...

    def test_append(self, closed: str) -> None:
        ...

    def test_is_non_overlapping_monotonic(self, closed: str) -> None:
        ...

    def test_is_overlapping(self, start: Union[int, Timestamp, Timedelta], shift: Union[int, Timedelta], na_value: Union[float, NaT], closed: str) -> None:
        ...

    def test_to_tuples(self, tuples: List[Tuple[Union[int, Timestamp, Timedelta], Union[int, Timestamp, Timedelta]]]) -> None:
        ...

    def test_to_tuples_na(self, tuples: List[Union[Tuple[Union[int, Timestamp, Timedelta], Union[int, Timestamp, Timedelta]], float]], na_tuple: bool) -> None:
        ...

    def test_nbytes(self) -> None:
        ...

    def test_set_closed(self, name: Optional[str], closed: str, other_closed: str) -> None:
        ...

    def test_set_closed_errors(self, bad_closed: str) -> None:
        ...

    def test_is_all_dates(self) -> None:
        ...

def test_dir() -> None:
    ...

def test_searchsorted_different_argument_classes(listlike_box: Callable) -> None:
    ...

def test_searchsorted_invalid_argument(arg: List[Any]) -> None:
    ...