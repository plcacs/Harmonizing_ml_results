from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
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
import pandas._testing as tm
import pandas.core.common as com

class TestIntervalIndex:
    index: IntervalIndex = ...

    def create_index(self, closed: str) -> IntervalIndex:
        ...

    def create_index_with_nan(self, closed: str) -> IntervalIndex:
        ...

    def test_properties(self, closed: str) -> None:
        ...

    @pytest.mark.parametrize('breaks', [[1, 1, 2, 5, 15, 53, 217, 1014, 5335, 31240, 201608], [-np.inf, -100, -10, 0.5, 1, 1.5, 3.8, 101, 202, np.inf], date_range('2017-01-01', '2017-01-04'), pytest.param(date_range('2017-01-01', '2017-01-04', unit='s'), marks=pytest.mark.xfail(reason='mismatched result unit')), pd.to_timedelta(['1ns', '2ms', '3s', '4min', '5h', '6D'])])
    def test_length(self, closed: str, breaks: list) -> None:
        ...

    def test_with_nans(self, closed: str) -> None:
        ...

    def test_copy(self, closed: str) -> None:
        ...

    def test_ensure_copied_data(self, closed: str) -> None:
        ...

    def test_delete(self, closed: str) -> None:
        ...

    @pytest.mark.parametrize('data', [interval_range(0, periods=10, closed='neither'), interval_range(1.7, periods=8, freq=2.5, closed='both'), interval_range(Timestamp('20170101'), periods=12, closed='left'), interval_range(Timedelta('1 day'), periods=6, closed='right')])
    def test_insert(self, data: IntervalIndex) -> None:
        ...

    def test_is_unique_interval(self, closed: str) -> bool:
        ...

    def test_monotonic(self, closed: str) -> None:
        ...

    def test_is_monotonic_with_nans(self) -> None:
        ...

    @pytest.mark.parametrize('breaks', [date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], ids=lambda x: str(x.dtype))
    def test_maybe_convert_i8(self, breaks: pd.DatetimeIndex | pd.TimedeltaIndex) -> None:
        ...

    @pytest.mark.parametrize('breaks', [date_range('2018-01-01', periods=5), timedelta_range('0 days', periods=5)])
    def test_maybe_convert_i8_nat(self, breaks: pd.DatetimeIndex | pd.TimedeltaIndex) -> None:
        ...

    @pytest.mark.parametrize('make_key', [lambda breaks: breaks, list], ids=['lambda', 'list'])
    def test_maybe_convert_i8_numeric(self, make_key: callable, any_real_numpy_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('make_key', [IntervalIndex.from_breaks, lambda breaks: Interval(breaks[0], breaks[1]), lambda breaks: breaks[0]], ids=['IntervalIndex', 'Interval', 'scalar'])
    def test_maybe_convert_i8_numeric_identical(self, make_key: callable, any_real_numpy_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('breaks1, breaks2', permutations([date_range('20180101', periods=4), date_range('20180101', periods=4, tz='US/Eastern'), timedelta_range('0 days', periods=4)], 2), ids=lambda x: str(x.dtype))
    @pytest.mark.parametrize('make_key', [IntervalIndex.from_breaks, lambda breaks: Interval(breaks[0], breaks[1]), lambda breaks: breaks, lambda breaks: breaks[0], list], ids=['IntervalIndex', 'Interval', 'Index', 'scalar', 'list'])
    def test_maybe_convert_i8_errors(self, breaks1: pd.DatetimeIndex | pd.TimedeltaIndex, breaks2: pd.DatetimeIndex | pd.TimedeltaIndex, make_key: callable) -> None:
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

    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_datetime(self, tz: str | None) -> None:
        ...

    def test_append(self, closed: str) -> None:
        ...

    def test_is_non_overlapping_monotonic(self, closed: str) -> bool:
        ...

    @pytest.mark.parametrize('start, shift, na_value', [(0, 1, np.nan), (Timestamp('2018-01-01'), Timedelta('1 day'), pd.NaT), (Timedelta('0 days'), Timedelta('1 day'), pd.NaT)])
    def test_is_overlapping(self, start: int | Timestamp | Timedelta, shift: int | Timedelta, na_value: float | pd.NaT, closed: str) -> None:
        ...

    @pytest.mark.parametrize('tuples', [zip(range(10), range(1, 11)), zip(date_range('20170101', periods=10), date_range('20170101', periods=10)), zip(timedelta_range('0 days', periods=10), timedelta_range('1 day', periods=10))])
    def test_to_tuples(self, tuples: list[tuple]) -> None:
        ...

    @pytest.mark.parametrize('tuples', [list(zip(range(10), range(1, 11))) + [np.nan], list(zip(date_range('20170101', periods=10), date_range('20170101', periods=10))) + [np.nan], list(zip(timedelta_range('0 days', periods=10), timedelta_range('1 day', periods=10))) + [np.nan]])
    @pytest.mark.parametrize('na_tuple', [True, False])
    def test_to_tuples_na(self, tuples: list[tuple], na_tuple: bool) -> None:
        ...

    def test_nbytes(self) -> None:
        ...

    @pytest.mark.parametrize('name', [None, 'foo'])
    def test_set_closed(self, name: str | None, closed: str, other_closed: str) -> None:
        ...

    @pytest.mark.parametrize('bad_closed', ['foo', 10, 'LEFT', True, False])
    def test_set_closed_errors(self, bad_closed: str | int | bool) -> None:
        ...

    def test_is_all_dates(self) -> None:
        ...

def test_dir() -> None:
    ...

def test_searchsorted_different_argument_classes(listlike_box: callable) -> None:
    ...

@pytest.mark.parametrize('arg', [[1, 2], ['a', 'b'], [Timestamp('2020-01-01', tz='Europe/London')] * 2])
def test_searchsorted_invalid_argument(arg: list) -> None:
    ...