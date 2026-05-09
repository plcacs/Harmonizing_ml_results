from __future__ import annotations
from typing import (
    Any,
    AnyStr,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
import pytest
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    interval_range,
    isna,
    notna,
    timedelta_range,
)
from pandas._testing import tm
from pandas.core.common import com

class TestIntervalIndex:
    index: IntervalIndex
    def __init__(self) -> None:
        ...
    
    def create_index(self, closed: str = 'right') -> IntervalIndex:
        ...
    
    def create_index_with_nan(self, closed: str = 'right') -> IntervalIndex:
        ...
    
    def test_properties(self, closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('breaks', [Any, Any, Any, Any])
    def test_length(self, closed: str, breaks: Any) -> None:
        ...
    
    def test_with_nans(self, closed: str) -> None:
        ...
    
    def test_copy(self, closed: str) -> None:
        ...
    
    def test_ensure_copied_data(self, closed: str) -> None:
        ...
    
    def test_delete(self, closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('data', [Any, Any, Any, Any])
    def test_insert(self, data: Any) -> None:
        ...
    
    def test_is_unique_interval(self, closed: str) -> None:
        ...
    
    def test_monotonic(self, closed: str) -> None:
        ...
    
    def test_is_monotonic_with_nans(self) -> None:
        ...
    
    @pytest.mark.parametrize('breaks', [Any, Any, Any])
    def test_maybe_convert_i8(self, breaks: Any) -> None:
        ...
    
    @pytest.mark.parametrize('breaks', [Any, Any])
    def test_maybe_convert_i8_nat(self, breaks: Any) -> None:
        ...
    
    @pytest.mark.parametrize('make_key', [Any, Any])
    def test_maybe_convert_i8_numeric(self, make_key: Any, any_real_numpy_dtype: Any) -> None:
        ...
    
    @pytest.mark.parametrize('make_key', [Any, Any, Any, Any])
    def test_maybe_convert_i8_numeric_identical(self, make_key: Any, any_real_numpy_dtype: Any) -> None:
        ...
    
    @pytest.mark.parametrize('breaks1, breaks2', [Any, Any])
    @pytest.mark.parametrize('make_key', [Any, Any, Any, Any, Any])
    def test_maybe_convert_i8_errors(self, breaks1: Any, breaks2: Any, make_key: Any) -> None:
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
    
    @pytest.mark.parametrize('tz', [None, str])
    def test_datetime(self, tz: Optional[str]) -> None:
        ...
    
    def test_append(self, closed: str) -> None:
        ...
    
    def test_is_non_overlapping_monotonic(self, closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('start, shift, na_value', [(int, int, float), (Timestamp, Timedelta, Any), (Timedelta, Timedelta, Any)])
    def test_is_overlapping(self, start: Any, shift: Any, na_value: Any, closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('tuples', [List[Tuple[int, int]], List[Tuple[Timestamp, Timestamp]], List[Tuple[Timedelta, Timedelta]]])
    def test_to_tuples(self, tuples: List[Tuple[Any, Any]]) -> None:
        ...
    
    @pytest.mark.parametrize('tuples', [List[Tuple[int, int]], List[Tuple[Timestamp, Timestamp]], List[Tuple[Timedelta, Timedelta]]])
    @pytest.mark.parametrize('na_tuple', [bool])
    def test_to_tuples_na(self, tuples: List[Tuple[Any, Any]], na_tuple: bool) -> None:
        ...
    
    def test_nbytes(self) -> None:
        ...
    
    @pytest.mark.parametrize('name', [None, str])
    def test_set_closed(self, name: Optional[str], closed: str, other_closed: str) -> None:
        ...
    
    @pytest.mark.parametrize('bad_closed', [str, int, str, bool, bool])
    def test_set_closed_errors(self, bad_closed: Any) -> None:
        ...
    
    def test_is_all_dates(self) -> None:
        ...

def test_dir() -> None:
    ...

def test_searchsorted_different_argument_classes(listlike_box: Any) -> None:
    ...

@pytest.mark.parametrize('arg', [List[int], List[str], List[Timestamp]])
def test_searchsorted_invalid_argument(arg: Any) -> None:
    ...

class MarkDecorator:
    def parametrize(self, name: str, argvalues: Any) -> pytest.MarkDecorator:
        ...

@pytest.mark.parametrize = MarkDecorator().parametrize