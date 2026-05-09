"""
Stub file for 'test_offsets_595cf3.py'
"""

from __future__ import annotations
from datetime import datetime
import numpy as np
from pandas._libs.tslibs import NaT, Timedelta, Timestamp
from pandas._libs.tslibs.offsets import (
    _get_offset,
    _offset_map,
    to_offset,
    liboffsets,
)
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import DataFrame, DatetimeIndex, Series
import pytest

_ARITHMETIC_DATE_OFFSET: list[str] = ...

@pytest.fixture
def month_classes(request) -> type[liboffsets.MonthOffset]:
    ...

@pytest.fixture
def offset_types(request) -> type:
    ...

@pytest.fixture
def dt() -> Timestamp:
    ...

@pytest.fixture
def expecteds() -> dict[str, Timestamp]:
    ...

class TestCommon:
    def test_immutable(self, offset_types: type) -> None:
        ...

    def test_return_type(self, offset_types: type) -> None:
        ...

    def test_offset_n(self, offset_types: type) -> None:
        ...

    def test_offset_timedelta64_arg(self, offset_types: type) -> None:
        ...

    def test_offset_mul_ndarray(self, offset_types: type) -> None:
        ...

    def test_offset_freqstr(self, offset_types: type) -> None:
        ...

    def _check_offsetfunc_works(
        self,
        offset: type,
        funcname: str,
        dt: datetime | np.datetime64,
        expected: Timestamp,
        normalize: bool = False,
    ) -> None:
        ...

    def test_apply(self, offset_types: type, expecteds: dict[str, Timestamp]) -> None:
        ...

    def test_rollforward(self, offset_types: type, expecteds: dict[str, Timestamp]) -> None:
        ...

    def test_rollback(self, offset_types: type) -> None:
        ...

    def test_is_on_offset(self, offset_types: type, expecteds: dict[str, Timestamp]) -> None:
        ...

    def test_add(self, offset_types: type, dt: datetime, expecteds: dict[str, Timestamp]) -> None:
        ...

    def test_add_empty_datetimeindex(
        self,
        performance_warning: pytest.Warning,
        offset_types: type,
        tz_naive_fixture: str | None,
    ) -> None:
        ...

    def test_pickle_roundtrip(self, offset_types: type) -> None:
        ...

    def test_offsets_hashable(self, offset_types: type) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:Non-vectorized DateOffset being applied to Series or DatetimeIndex')
    @pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
    def test_add_dt64_ndarray_non_nano(
        self,
        offset_types: type,
        unit: str,
    ) -> None:
        ...

class TestDateOffset:
    def setup_method(self) -> None:
        ...

    def test_repr(self) -> None:
        ...

    def test_mul(self) -> None:
        ...

    @pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
    def test_constructor(self, kwd: str, request: pytest.FixtureRequest) -> None:
        ...

    def test_default_constructor(self, dt: Timestamp) -> None:
        ...

    def test_copy(self) -> None:
        ...

    @pytest.mark.parametrize('arithmatic_offset_type, expected', zip(_ARITHMETIC_DATE_OFFSET, ['2009-01-02', '2008-02-02', '2008-01-09', '2008-01-03', '2008-01-02 01:00:00', '2008-01-02 00:01:00', '2008-01-02 00:00:01', '2008-01-02 00:00:00.001000000', '2008-01-02 00:00:00.000001000']))
    def test_add(self, arithmatic_offset_type: str, expected: str, dt: Timestamp) -> None:
        ...

    @pytest.mark.parametrize('arithmatic_offset_type, expected', zip(_ARITHMETIC_DATE_OFFSET, ['2007-01-02', '2007-12-02', '2007-12-26', '2008-01-01', '2008-01-01 23:00:00', '2008-01-01 23:59:00', '2008-01-01 23:59:59', '2008-01-01 23:59:59.999000000', '2008-01-01 23:59:59.999999000']))
    def test_sub(self, arithmatic_offset_type: str, expected: str, dt: Timestamp) -> None:
        ...

    @pytest.mark.parametrize('arithmatic_offset_type, n, expected', zip(_ARITHMETIC_DATE_OFFSET, range(1, 10), ['2009-01-02', '2008-03-02', '2008-01-23', '2008-01-06', '2008-01-02 05:00:00', '2008-01-02 00:06:00', '2008-01-02 00:00:07', '2008-01-02 00:00:00.008000000', '2008-01-02 00:00:00.000009000']))
    def test_mul_add(self, arithmatic_offset_type: str, n: int, expected: str, dt: Timestamp) -> None:
        ...

    @pytest.mark.parametrize('arithmatic_offset_type, n, expected', zip(_ARITHMETIC_DATE_OFFSET, range(1, 10), ['2007-01-02', '2007-11-02', '2007-12-12', '2007-12-29', '2008-01-01 19:00:00', '2008-01-01 23:54:00', '2008-01-01 23:59:53', '2008-01-01 23:59:59.992000000', '2008-01-01 23:59:59.999991000']))
    def test_mul_sub(self, arithmatic_offset_type: str, n: int, expected: str, dt: Timestamp) -> None:
        ...

    def test_leap_year(self) -> None:
        ...

    def test_eq(self) -> None:
        ...

    @pytest.mark.parametrize('offset_kwargs, expected_arg', [({'microseconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:00.001001'), ({'seconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:01.001'), ({'minutes': 1, 'milliseconds': 1}, '2022-01-01 00:01:00.001'), ({'hours': 1, 'milliseconds': 1}, '2022-01-01 01:00:00.001'), ({'days': 1, 'milliseconds': 1}, '2022-01-02 00:00:00.001'), ({'weeks': 1, 'milliseconds': 1}, '2022-01-08 00:00:00.001'), ({'months': 1, 'milliseconds': 1}, '2022-02-01 00:00:00.001'), ({'years': 1, 'milliseconds': 1}, '2023-01-01 00:00:00.001')])
    def test_milliseconds_combination(self, offset_kwargs: dict[str, int], expected_arg: str) -> None:
        ...

    def test_offset_invalid_arguments(self) -> None:
        ...

class TestOffsetNames:
    def test_get_offset_name(self) -> None:
        ...

def test_get_offset() -> None:
    ...

def test_get_offset_legacy() -> None:
    ...

class TestOffsetAliases:
    def setup_method(self) -> None:
        ...

    def test_alias_equality(self) -> None:
        ...

    def test_rule_code(self) -> None:
        ...

def test_freq_offsets() -> None:
    ...

class TestReprNames:
    def test_str_for_named_is_name(self) -> None:
        ...

def test_valid_default_arguments(offset_types: type) -> None:
    ...

@pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
def test_valid_month_attributes(kwd: str, month_classes: type) -> None:
    ...

def test_month_offset_name(month_classes: type) -> None:
    ...

@pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
def test_valid_relativedelta_kwargs(kwd: str, request: pytest.FixtureRequest) -> None:
    ...

@pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
def test_valid_tick_attributes(kwd: str, tick_classes: type) -> None:
    ...

def test_validate_n_error() -> None:
    ...

def test_require_integers(offset_types: type) -> None:
    ...

def test_tick_normalize_raises(tick_classes: type) -> None:
    ...

@pytest.mark.parametrize('offset_kwargs, expected_arg', [({'nanoseconds': 1}, '1970-01-01 00:00:00.000000001'), ({'nanoseconds': 5}, '1970-01-01 00:00:00.000000005'), ({'nanoseconds': -1}, '1969-12-31 23:59:59.999999999'), ({'microseconds': 1}, '1970-01-01 00:00:00.000001'), ({'microseconds': -1}, '1969-12-31 23:59:59.999999'), ({'seconds': 1}, '1970-01-01 00:00:01'), ({'seconds': -1}, '1969-12-31 23:59:59'), ({'minutes': 1}, '1970-01-01 00:01:00'), ({'minutes': -1}, '1969-12-31 23:59:00'), ({'hours': 1}, '1970-01-01 01:00:00'), ({'hours': -1}, '1969-12-31 23:00:00'), ({'days': 1}, '1970-01-02 00:00:00'), ({'days': -1}, '1969-12-31 00:00:00'), ({'weeks': 1}, '1970-01-08 00:00:00'), ({'weeks': -1}, '1969-12-25 00:00:00'), ({'months': 1}, '1970-02-01 00:00:00'), ({'months': -1}, '1969-12-01 00:00:00'), ({'years': 1}, '1971-01-01 00:00:00'), ({'years': -1}, '1969-01-01 00:00:00')])
def test_dateoffset_add_sub(offset_kwargs: dict[str, int], expected_arg: str) -> None:
    ...

def test_dateoffset_add_sub_timestamp_with_nano() -> None:
    ...

@pytest.mark.parametrize('attribute', ['hours', 'days', 'weeks', 'months', 'years'])
def test_dateoffset_immutable(attribute: str) -> None:
    ...

def test_dateoffset_misc() -> None:
    ...

@pytest.mark.parametrize('n', [-1, 1, 3])
def test_construct_int_arg_no_kwargs_assumed_days(n: int) -> None:
    ...

@pytest.mark.parametrize('offset, expected', [(DateOffset(minutes=7, nanoseconds=18), Timestamp('2022-01-01 00:07:00.000000018')), (DateOffset(nanoseconds=3), Timestamp('2022-01-01 00:00:00.000000003'))])
def test_dateoffset_add_sub_timestamp_series_with_nano(offset: DateOffset, expected: Timestamp) -> None:
    ...

@pytest.mark.parametrize('n_months, scaling_factor, start_timestamp, expected_timestamp', [(1, 2, '2020-01-30', '2020-03-30'), (2, 1, '2020-01-30', '2020-03-30'), (1, 0, '2020-01-30', '2020-01-30'), (2, 0, '2020-01-30', '2020-01-30'), (1, -1, '2020-01-30', '2019-12-30'), (2, -1, '2020-01-30', '2019-11-30')])
def test_offset_multiplication(n_months: int, scaling_factor: int, start_timestamp: str, expected_timestamp: str) -> None:
    ...

def test_dateoffset_operations_on_dataframes(performance_warning: pytest.Warning) -> None:
    ...

def test_is_yqm_start_end() -> None:
    ...

@pytest.mark.parametrize('left', [DateOffset(1), Nano(1)])
@pytest.mark.parametrize('right', [DateOffset(1), Nano(1)])
def test_multiply_dateoffset_typeerror(left: DateOffset | Nano, right: DateOffset | Nano) -> None:
    ...