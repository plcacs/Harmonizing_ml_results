"""
Stub file for test_offsets_595cf3.py
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Dict,
    Fixture,
    List,
    Optional,
    ParamSpec,
    Tuple,
    Type,
    Union,
    overload,
)
import numpy as np
import pytest
from pandas._libs.tslibs import (
    NaT,
    Timedelta,
    Timestamp,
    conversion,
    timezones,
)
from pandas._libs.tslibs.offsets import (
    BaseOffset,
    DateOffset,
    Easter,
    FY5253,
    FY5253Quarter,
    Week,
    WeekOfMonth,
    LastWeekOfMonth,
)
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
)
import pandas._testing as tm

P = ParamSpec("P")

@pytest.fixture
def month_classes(request) -> Type[BaseOffset]:
    ...

@pytest.fixture
def offset_types(request) -> Type[BaseOffset]:
    ...

@pytest.fixture
def dt() -> Timestamp:
    ...

@pytest.fixture
def expecteds() -> Dict[str, Timestamp]:
    ...

class TestCommon:
    def test_immutable(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def test_return_type(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def test_offset_n(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def test_offset_timedelta64_arg(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def test_offset_mul_ndarray(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def test_offset_freqstr(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def _check_offsetfunc_works(
        self,
        offset: Type[BaseOffset],
        funcname: str,
        dt: Union[datetime, np.datetime64],
        expected: Timestamp,
        normalize: bool = False,
    ) -> None:
        ...

    def test_apply(
        self,
        offset_types: Type[BaseOffset],
        expecteds: Dict[str, Timestamp],
    ) -> None:
        ...

    def test_rollforward(
        self,
        offset_types: Type[BaseOffset],
        expecteds: Dict[str, Timestamp],
    ) -> None:
        ...

    def test_rollback(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def test_is_on_offset(
        self,
        offset_types: Type[BaseOffset],
        expecteds: Dict[str, Timestamp],
    ) -> None:
        ...

    def test_add(
        self,
        offset_types: Type[BaseOffset],
        tz_naive_fixture: Optional[timezone],
        expecteds: Dict[str, Timestamp],
    ) -> None:
        ...

    def test_add_empty_datetimeindex(
        self,
        performance_warning: pytest.Warning,
        offset_types: Type[BaseOffset],
        tz_naive_fixture: Optional[timezone],
    ) -> None:
        ...

    def test_pickle_roundtrip(self, offset_types: Type[BaseOffset]) -> None:
        ...

    def test_pickle_dateoffset_odd_inputs(self) -> None:
        ...

    def test_offsets_hashable(self, offset_types: Type[BaseOffset]) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:Non-vectorized DateOffset being applied to Series or DatetimeIndex')
    @pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
    def test_add_dt64_ndarray_non_nano(
        self,
        offset_types: Type[BaseOffset],
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

    @pytest.mark.parametrize(
        'arithmatic_offset_type, expected',
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                '2009-01-02',
                '2008-02-02',
                '2008-01-09',
                '2008-01-03',
                '2008-01-02 01:00:00',
                '2008-01-02 00:01:00',
                '2008-01-02 00:00:01',
                '2008-01-02 00:00:00.001000000',
                '2008-01-02 00:00:00.000001000',
            ],
        ),
    )
    def test_add(
        self,
        arithmatic_offset_type: str,
        expected: str,
        dt: Timestamp,
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'arithmatic_offset_type, expected',
        zip(
            _ARITHMETIC_DATE_OFFSET,
            [
                '2007-01-02',
                '2007-12-02',
                '2007-12-26',
                '2008-01-01',
                '2008-01-01 23:00:00',
                '2008-01-01 23:59:00',
                '2008-01-01 23:59:59',
                '2008-01-01 23:59:59.999000000',
                '2008-01-01 23:59:59.999999000',
            ],
        ),
    )
    def test_sub(
        self,
        arithmatic_offset_type: str,
        expected: str,
        dt: Timestamp,
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'arithmatic_offset_type, n, expected',
        zip(
            _ARITHMETIC_DATE_OFFSET,
            range(1, 10),
            [
                '2009-01-02',
                '2008-03-02',
                '2008-01-23',
                '2008-01-06',
                '2008-01-02 05:00:00',
                '2008-01-02 00:06:00',
                '2008-01-02 00:00:07',
                '2008-01-02 00:00:00.008000000',
                '2008-01-02 00:00:00.000009000',
            ],
        ),
    )
    def test_mul_add(
        self,
        arithmatic_offset_type: str,
        n: int,
        expected: str,
        dt: Timestamp,
    ) -> None:
        ...

    @pytest.mark.parametrize(
        'arithmatic_offset_type, n, expected',
        zip(
            _ARITHMETIC_DATE_OFFSET,
            range(1, 10),
            [
                '2007-01-02',
                '2007-11-02',
                '2007-12-12',
                '2007-12-29',
                '2008-01-01 19:00:00',
                '2008-01-01 23:54:00',
                '2008-01-01 23:59:53',
                '2008-01-01 23:59:59.992000000',
                '2008-01-01 23:59:59.999991000',
            ],
        ),
    )
    def test_mul_sub(
        self,
        arithmatic_offset_type: str,
        n: int,
        expected: str,
        dt: Timestamp,
    ) -> None:
        ...

    def test_leap_year(self) -> None:
        ...

    def test_eq(self) -> None:
        ...

    @pytest.mark.parametrize(
        'offset_kwargs, expected_arg',
        [
            ({'microseconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:00.001001'),
            ({'seconds': 1, 'milliseconds': 1}, '2022-01-01 00:00:01.001'),
            ({'minutes': 1, 'milliseconds': 1}, '2022-01-01 00:01:00.001'),
            ({'hours': 1, 'milliseconds': 1}, '2022-01-01 01:00:00.001'),
            ({'days': 1, 'milliseconds': 1}, '2022-01-02 00:00:00.001'),
            ({'weeks': 1, 'milliseconds': 1}, '2022-01-08 00:00:00.001'),
            ({'months': 1, 'milliseconds': 1}, '2022-02-01 00:00:00.001'),
            ({'years': 1, 'milliseconds': 1}, '2023-01-01 00:00:00.001'),
        ],
    )
    def test_milliseconds_combination(
        self,
        offset_kwargs: Dict[str, int],
        expected_arg: str,
    ) -> None:
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

def test_valid_default_arguments(offset_types: Type[BaseOffset]) -> None:
    ...

@pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
def test_valid_month_attributes(kwd: str, month_classes: Type[BaseOffset]) -> None:
    ...

def test_month_offset_name(month_classes: Type[BaseOffset]) -> None:
    ...

@pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
def test_valid_relativedelta_kwargs(kwd: str, request: pytest.FixtureRequest) -> None:
    ...

@pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
def test_valid_tick_attributes(kwd: str, tick_classes: Type[BaseOffset]) -> None:
    ...

def test_validate_n_error() -> None:
    ...

def test_require_integers(offset_types: Type[BaseOffset]) -> None:
    ...

def test_tick_normalize_raises(tick_classes: Type[BaseOffset]) -> None:
    ...

@pytest.mark.parametrize(
    'offset_kwargs, expected_arg',
    [
        ({'nanoseconds': 1}, '1970-01-01 00:00:00.000000001'),
        ({'nanoseconds': 5}, '1970-01-01 00:00:00.000000005'),
        ({'nanoseconds': -1}, '1969-12-31 23:59:59.999999999'),
        ({'microseconds': 1}, '1970-01-01 00:00:00.000001'),
        ({'microseconds': -1}, '1969-12-31 23:59:59.999999'),
        ({'seconds': 1}, '1970-01-01 00:00:01'),
        ({'seconds': -1}, '1969-12-31 23:59:59'),
        ({'minutes': 1}, '1970-01-01 00:01:00'),
        ({'minutes': -1}, '1969-12-31 23:59:00'),
        ({'hours': 1}, '1970-01-01 01:00:00'),
        ({'hours': -1}, '1969-12-31 23:00:00'),
        ({'days': 1}, '1970-01-02 00:00:00'),
        ({'days': -1}, '1969-12-31 00:00:00'),
        ({'weeks': 1}, '1970-01-08 00:00:00'),
        ({'weeks': -1}, '1969-12-25 00:00:00'),
        ({'months': 1}, '1970-02-01 00:00:00'),
        ({'months': -1}, '1969-12-01 00:00:00'),
        ({'years': 1}, '1971-01-01 00:00:00'),
        ({'years': -1}, '1969-01-01 00:00:00'),
    ],
)
def test_dateoffset_add_sub(
    offset_kwargs: Dict[str, int],
    expected_arg: str,
) -> None:
    ...

def test_dateoffset_add_sub_timestamp_with_nano() -> None:
    ...

def test_dateoffset_immutable(attribute: str) -> None:
    ...

def test_dateoffset_misc() -> None:
    ...

@pytest.mark.parametrize('n', [-1, 1, 3])
def test_construct_int_arg_no_kwargs_assumed_days(n: int) -> None:
    ...

@pytest.mark.parametrize(
    'offset, expected',
    [
        (DateOffset(minutes=7, nanoseconds=18), Timestamp('2022-01-01 00:07:00.000000018')),
        (DateOffset(nanoseconds=3), Timestamp('2022-01-01 00:00:00.000000003')),
    ],
)
def test_dateoffset_add_sub_timestamp_series_with_nano(
    offset: DateOffset,
    expected: Timestamp,
) -> None:
    ...

def test_offset_operations_on_dataframes(performance_warning: pytest.Warning) -> None:
    ...

def test_is_yqm_start_end() -> None:
    ...

@pytest.mark.parametrize(
    'left',
    [DateOffset(1), Nano(1)],
)
@pytest.mark.parametrize(
    'right',
    [DateOffset(1), Nano(1)],
)
def test_multiply_dateoffset_typeerror(left: Union[DateOffset, Nano], right: Union[DateOffset, Nano]) -> None:
    ...