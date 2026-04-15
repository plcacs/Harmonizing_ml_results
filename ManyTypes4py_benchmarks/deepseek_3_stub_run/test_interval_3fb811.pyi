from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Literal

import numpy as np
import pandas as pd
from pandas import (
    Index,
    Interval,
    IntervalIndex,
    Timedelta,
    Timestamp,
    date_range,
    interval_range,
    timedelta_range,
)
from pandas._testing import tm
import pytest

_T = TypeVar("_T")
_Closed = Literal["left", "right", "both", "neither"]

class TestIntervalIndex:
    index: IntervalIndex = ...
    
    def create_index(self, closed: _Closed = "right") -> IntervalIndex: ...
    
    def create_index_with_nan(self, closed: _Closed = "right") -> IntervalIndex: ...
    
    def test_properties(self, closed: _Closed) -> None: ...
    
    @pytest.mark.parametrize("breaks", ...)
    def test_length(
        self,
        closed: _Closed,
        breaks: Union[
            List[Union[int, float]],
            List[Union[float, int]],
            pd.DatetimeIndex,
            pd.TimedeltaIndex,
        ],
    ) -> None: ...
    
    def test_with_nans(self, closed: _Closed) -> None: ...
    
    def test_copy(self, closed: _Closed) -> None: ...
    
    def test_ensure_copied_data(self, closed: _Closed) -> None: ...
    
    def test_delete(self, closed: _Closed) -> None: ...
    
    @pytest.mark.parametrize("data", ...)
    def test_insert(
        self,
        data: Union[
            IntervalIndex,
            pd.core.indexes.interval.IntervalIndex,
        ],
    ) -> None: ...
    
    def test_is_unique_interval(self, closed: _Closed) -> None: ...
    
    def test_monotonic(self, closed: _Closed) -> None: ...
    
    def test_is_monotonic_with_nans(self) -> None: ...
    
    @pytest.mark.parametrize("breaks", ...)
    def test_maybe_convert_i8(
        self,
        breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex],
    ) -> None: ...
    
    @pytest.mark.parametrize("breaks", ...)
    def test_maybe_convert_i8_nat(
        self,
        breaks: Union[pd.DatetimeIndex, pd.TimedeltaIndex],
    ) -> None: ...
    
    @pytest.mark.parametrize("make_key", ...)
    def test_maybe_convert_i8_numeric(
        self,
        make_key: Callable[[np.ndarray], Union[np.ndarray, List[Any]]],
        any_real_numpy_dtype: Type[np.dtype],
    ) -> None: ...
    
    @pytest.mark.parametrize("make_key", ...)
    def test_maybe_convert_i8_numeric_identical(
        self,
        make_key: Callable[[np.ndarray], Union[IntervalIndex, Interval, Any]],
        any_real_numpy_dtype: Type[np.dtype],
    ) -> None: ...
    
    @pytest.mark.parametrize("breaks1, breaks2", ...)
    @pytest.mark.parametrize("make_key", ...)
    def test_maybe_convert_i8_errors(
        self,
        breaks1: Union[pd.DatetimeIndex, pd.TimedeltaIndex],
        breaks2: Union[pd.DatetimeIndex, pd.TimedeltaIndex],
        make_key: Callable[[Union[pd.DatetimeIndex, pd.TimedeltaIndex]], Any],
    ) -> None: ...
    
    def test_contains_method(self) -> None: ...
    
    def test_dropna(self, closed: _Closed) -> None: ...
    
    def test_non_contiguous(self, closed: _Closed) -> None: ...
    
    def test_isin(self, closed: _Closed) -> None: ...
    
    def test_comparison(self) -> None: ...
    
    def test_missing_values(self, closed: _Closed) -> None: ...
    
    def test_sort_values(self, closed: _Closed) -> None: ...
    
    @pytest.mark.parametrize("tz", ...)
    def test_datetime(self, tz: Optional[str]) -> None: ...
    
    def test_append(self, closed: _Closed) -> None: ...
    
    def test_is_non_overlapping_monotonic(self, closed: _Closed) -> None: ...
    
    @pytest.mark.parametrize("start, shift, na_value", ...)
    def test_is_overlapping(
        self,
        start: Union[int, Timestamp, Timedelta],
        shift: Union[int, Timedelta],
        na_value: Union[float, pd.NaTType],
        closed: _Closed,
    ) -> None: ...
    
    @pytest.mark.parametrize("tuples", ...)
    def test_to_tuples(
        self,
        tuples: Union[
            List[Tuple[int, int]],
            List[Tuple[Timestamp, Timestamp]],
            List[Tuple[Timedelta, Timedelta]],
        ],
    ) -> None: ...
    
    @pytest.mark.parametrize("tuples", ...)
    @pytest.mark.parametrize("na_tuple", ...)
    def test_to_tuples_na(
        self,
        tuples: Union[
            List[Union[Tuple[int, int], float]],
            List[Union[Tuple[Timestamp, Timestamp], float]],
            List[Union[Tuple[Timedelta, Timedelta], float]],
        ],
        na_tuple: bool,
    ) -> None: ...
    
    def test_nbytes(self) -> None: ...
    
    @pytest.mark.parametrize("name", ...)
    def test_set_closed(
        self,
        name: Optional[str],
        closed: _Closed,
        other_closed: _Closed,
    ) -> None: ...
    
    @pytest.mark.parametrize("bad_closed", ...)
    def test_set_closed_errors(self, bad_closed: Any) -> None: ...
    
    def test_is_all_dates(self) -> None: ...

def test_dir() -> None: ...

def test_searchsorted_different_argument_classes(
    listlike_box: Callable[[IntervalIndex], Any],
) -> None: ...

@pytest.mark.parametrize("arg", ...)
def test_searchsorted_invalid_argument(
    arg: Union[List[int], List[str], List[Timestamp]],
) -> None: ...