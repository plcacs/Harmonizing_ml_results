"""
Stub file for test_scalar_compat_34fd8d module
"""

from datetime import date, datetime, time
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import pytest
import numpy as np
from hypothesis import given
from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Timestamp,
    date_range,
    offsets,
)
from pandas._libs.tslibs import timezones
from pandas.core.arrays import DatetimeArray

class TestDatetimeIndexOps:
    def test_dti_no_millisecond_field(self) -> None:
        ...

    def test_dti_time(self) -> None:
        ...

    def test_dti_date(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', [None, 'datetime64[ns, CET]', 'datetime64[ns, EST]', 'datetime64[ns, UTC]'])
    def test_dti_date2(self, dtype: Optional[str]) -> None:
        ...

    @pytest.mark.parametrize('dtype', [None, 'datetime64[ns, CET]', 'datetime64[ns, EST]', 'datetime64[ns, UTC]'])
    def test_dti_time2(self, dtype: Optional[str]) -> None:
        ...

    def test_dti_timetz(self, tz_naive_fixture: Optional[str]) -> None:
        ...

    @pytest.mark.parametrize('field', ['dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'days_in_month', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end'])
    def test_dti_timestamp_fields(self, field: str) -> None:
        ...

    def test_dti_nanosecond(self) -> None:
        ...

    @pytest.mark.parametrize('prefix', ['', 'dateutil/'])
    def test_dti_hour_tzaware(self, prefix: str) -> None:
        ...

    @pytest.mark.parametrize('time_locale', [None] + tm.get_locales())
    def test_day_name_month_name(self, time_locale: Optional[str]) -> None:
        ...

    def test_dti_week(self) -> None:
        ...

    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_dti_fields(self, tz: Optional[str]) -> None:
        ...

    def test_dti_is_year_quarter_start(self) -> None:
        ...

    def test_dti_is_month_start(self) -> None:
        ...

    def test_dti_is_month_start_custom(self) -> None:
        ...

    @pytest.mark.parametrize('timestamp, freq, periods, expected_values', [('2017-12-01', 'MS', 3, np.array([False, True, False])), ('2017-12-01', 'QS', 3, np.array([True, False, False])), ('2017-12-01', 'YS', 3, np.array([True, True, True]))])
    def test_dti_dr_is_year_start(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        ...

    @pytest.mark.parametrize('timestamp, freq, periods, expected_values', [('2017-12-01', 'ME', 3, np.array([True, False, False])), ('2017-12-01', 'QE', 3, np.array([True, False, False])), ('2017-12-01', 'YE', 3, np.array([True, True, True]))])
    def test_dti_dr_is_year_end(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        ...

    @pytest.mark.parametrize('timestamp, freq, periods, expected_values', [('2017-12-01', 'MS', 3, np.array([False, True, False])), ('2017-12-01', 'QS', 3, np.array([True, True, True])), ('2017-12-01', 'YS', 3, np.array([True, True, True]))])
    def test_dti_dr_is_quarter_start(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        ...

    @pytest.mark.parametrize('timestamp, freq, periods, expected_values', [('2017-12-01', 'ME', 3, np.array([True, False, False])), ('2017-12-01', 'QE', 3, np.array([True, True, True])), ('2017-12-01', 'YE', 3, np.array([True, True, True]))])
    def test_dti_dr_is_quarter_end(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        ...

    @pytest.mark.parametrize('timestamp, freq, periods, expected_values', [('2017-12-01', 'MS', 3, np.array([True, True, True])), ('2017-12-01', 'QS', 3, np.array([True, True, True])), ('2017-12-01', 'YS', 3, np.array([True, True, True]))])
    def test_dti_dr_is_month_start(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        ...

    @pytest.mark.parametrize('timestamp, freq, periods, expected_values', [('2017-12-01', 'ME', 3, np.array([True, True, True])), ('2017-12-01', 'QE', 3, np.array([True, True, True])), ('2017-12-01', 'YE', 3, np.array([True, True, True]))])
    def test_dti_dr_is_month_end(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        ...

    def test_dti_is_year_quarter_start_doubledigit_freq(self) -> None:
        ...

    def test_dti_is_year_start_freq_custom_business_day_with_digit(self) -> None:
        ...

    @pytest.mark.parametrize('freq', ['3BMS', offsets.BusinessMonthBegin(3)])
    def test_dti_is_year_quarter_start_freq_business_month_begin(self, freq: Union[str, offsets.BusinessMonthBegin]) -> None:
        ...

@given(dt=st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)), n=st.integers(min_value=1, max_value=10), freq=st.sampled_from(['MS', 'QS', 'YS']))
@pytest.mark.slow
def test_against_scalar_parametric(freq: str, dt: datetime, n: int) -> None:
    ...