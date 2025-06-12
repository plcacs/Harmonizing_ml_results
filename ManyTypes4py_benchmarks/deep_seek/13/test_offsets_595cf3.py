"""
Tests of pandas.tseries.offsets
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union, cast
import numpy as np
import pytest
from pandas._libs.tslibs import NaT, Timedelta, Timestamp, conversion, timezones
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import _get_offset, _offset_map, to_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import DataFrame, DatetimeIndex, Series, date_range
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import FY5253, BDay, BMonthEnd, BusinessHour, CustomBusinessDay, CustomBusinessHour, CustomBusinessMonthBegin, CustomBusinessMonthEnd, DateOffset, Easter, FY5253Quarter, LastWeekOfMonth, MonthBegin, Nano, Tick, Week, WeekOfMonth

_ARITHMETIC_DATE_OFFSET: List[str] = ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds']

def _create_offset(klass: Any, value: int = 1, normalize: bool = False) -> Any:
    if klass is FY5253:
        klass = klass(n=value, startingMonth=1, weekday=1, variation='last', normalize=normalize)
    elif klass is FY5253Quarter:
        klass = klass(n=value, startingMonth=1, weekday=1, qtr_with_extra_week=1, variation='last', normalize=normalize)
    elif klass is LastWeekOfMonth:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is WeekOfMonth:
        klass = klass(n=value, week=1, weekday=5, normalize=normalize)
    elif klass is Week:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is DateOffset:
        klass = klass(days=value, normalize=normalize)
    else:
        klass = klass(value, normalize=normalize)
    return klass

@pytest.fixture(params=[getattr(offsets, o) for o in offsets.__all__ if issubclass(getattr(offsets, o), liboffsets.MonthOffset) and o != 'MonthOffset'])
def month_classes(request: pytest.FixtureRequest) -> Any:
    """
    Fixture for month based datetime offsets available for a time series.
    """
    return request.param

@pytest.fixture(params=[getattr(offsets, o) for o in offsets.__all__ if o not in ('Tick', 'BaseOffset')])
def offset_types(request: pytest.FixtureRequest) -> Any:
    """
    Fixture for all the datetime offsets available for a time series.
    """
    return request.param

@pytest.fixture
def dt() -> Timestamp:
    return Timestamp(datetime(2008, 1, 2))

@pytest.fixture
def expecteds() -> Dict[str, Timestamp]:
    return {'Day': Timestamp('2011-01-02 09:00:00'), 'DateOffset': Timestamp('2011-01-02 09:00:00'), 'BusinessDay': Timestamp('2011-01-03 09:00:00'), 'CustomBusinessDay': Timestamp('2011-01-03 09:00:00'), 'CustomBusinessMonthEnd': Timestamp('2011-01-31 09:00:00'), 'CustomBusinessMonthBegin': Timestamp('2011-01-03 09:00:00'), 'MonthBegin': Timestamp('2011-02-01 09:00:00'), 'BusinessMonthBegin': Timestamp('2011-01-03 09:00:00'), 'MonthEnd': Timestamp('2011-01-31 09:00:00'), 'SemiMonthEnd': Timestamp('2011-01-15 09:00:00'), 'SemiMonthBegin': Timestamp('2011-01-15 09:00:00'), 'BusinessMonthEnd': Timestamp('2011-01-31 09:00:00'), 'YearBegin': Timestamp('2012-01-01 09:00:00'), 'BYearBegin': Timestamp('2011-01-03 09:00:00'), 'YearEnd': Timestamp('2011-12-31 09:00:00'), 'BYearEnd': Timestamp('2011-12-30 09:00:00'), 'QuarterBegin': Timestamp('2011-03-01 09:00:00'), 'BQuarterBegin': Timestamp('2011-03-01 09:00:00'), 'QuarterEnd': Timestamp('2011-03-31 09:00:00'), 'BQuarterEnd': Timestamp('2011-03-31 09:00:00'), 'BusinessHour': Timestamp('2011-01-03 10:00:00'), 'CustomBusinessHour': Timestamp('2011-01-03 10:00:00'), 'WeekOfMonth': Timestamp('2011-01-08 09:00:00'), 'LastWeekOfMonth': Timestamp('2011-01-29 09:00:00'), 'FY5253Quarter': Timestamp('2011-01-25 09:00:00'), 'FY5253': Timestamp('2011-01-25 09:00:00'), 'Week': Timestamp('2011-01-08 09:00:00'), 'Easter': Timestamp('2011-04-24 09:00:00'), 'Hour': Timestamp('2011-01-01 10:00:00'), 'Minute': Timestamp('2011-01-01 09:01:00'), 'Second': Timestamp('2011-01-01 09:00:01'), 'Milli': Timestamp('2011-01-01 09:00:00.001000'), 'Micro': Timestamp('2011-01-01 09:00:00.000001'), 'Nano': Timestamp('2011-01-01T09:00:00.000000001')}

class TestCommon:

    def test_immutable(self, offset_types: Any) -> None:
        offset = _create_offset(offset_types)
        msg = 'objects is not writable|DateOffset objects are immutable'
        with pytest.raises(AttributeError, match=msg):
            offset.normalize = True
        with pytest.raises(AttributeError, match=msg):
            offset.n = 91

    def test_return_type(self, offset_types: Any) -> None:
        offset = _create_offset(offset_types)
        result = Timestamp('20080101') + offset
        assert isinstance(result, Timestamp)
        assert NaT + offset is NaT
        assert offset + NaT is NaT
        assert NaT - offset is NaT
        assert (-offset)._apply(NaT) is NaT

    def test_offset_n(self, offset_types: Any) -> None:
        offset = _create_offset(offset_types)
        assert offset.n == 1
        neg_offset = offset * -1
        assert neg_offset.n == -1
        mul_offset = offset * 3
        assert mul_offset.n == 3

    def test_offset_timedelta64_arg(self, offset_types: Any) -> None:
        off = _create_offset(offset_types)
        td64 = np.timedelta64(4567, 's')
        with pytest.raises(TypeError, match='argument must be an integer'):
            type(off)(n=td64, **off.kwds)

    def test_offset_mul_ndarray(self, offset_types: Any) -> None:
        off = _create_offset(offset_types)
        expected = np.array([[off, off * 2], [off * 3, off * 4]])
        result = np.array([[1, 2], [3, 4]]) * off
        tm.assert_numpy_array_equal(result, expected)
        result = off * np.array([[1, 2], [3, 4]])
        tm.assert_numpy_array_equal(result, expected)

    def test_offset_freqstr(self, offset_types: Any) -> None:
        offset = _create_offset(offset_types)
        freqstr = offset.freqstr
        if freqstr not in ('<Easter>', '<DateOffset: days=1>', 'LWOM-SAT'):
            code = _get_offset(freqstr)
            assert offset.rule_code == code

    def _check_offsetfunc_works(self, offset: Any, funcname: str, dt: Union[datetime, np.datetime64], expected: Timestamp, normalize: bool = False) -> None:
        if normalize and issubclass(offset, Tick):
            return
        offset_s = _create_offset(offset, normalize=normalize)
        func = getattr(offset_s, funcname)
        result = func(dt)
        assert isinstance(result, Timestamp)
        assert result == expected
        result = func(Timestamp(dt))
        assert isinstance(result, Timestamp)
        assert result == expected
        ts = Timestamp(dt) + Nano(5)
        with tm.assert_produces_warning(None):
            result = func(ts)
        assert isinstance(result, Timestamp)
        if normalize is False:
            assert result == expected + Nano(5)
        else:
            assert result == expected
        if isinstance(dt, np.datetime64):
            return
        for tz in [None, 'UTC', 'Asia/Tokyo', 'US/Eastern', 'dateutil/Asia/Tokyo', 'dateutil/US/Pacific']:
            expected_localize = expected.tz_localize(tz)
            tz_obj = timezones.maybe_get_tz(tz)
            dt_tz = conversion.localize_pydatetime(dt, tz_obj)
            result = func(dt_tz)
            assert isinstance(result, Timestamp)
            assert result == expected_localize
            result = func(Timestamp(dt, tz=tz))
            assert isinstance(result, Timestamp)
            assert result == expected_localize
            ts = Timestamp(dt, tz=tz) + Nano(5)
            with tm.assert_produces_warning(None):
                result = func(ts)
            assert isinstance(result, Timestamp)
            if normalize is False:
                assert result == expected_localize + Nano(5)
            else:
                assert result == expected_localize

    def test_apply(self, offset_types: Any, expecteds: Dict[str, Timestamp]) -> None:
        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64('2011-01-01 09:00')
        expected = expecteds[offset_types.__name__]
        expected_norm = Timestamp(expected.date())
        for dt in [sdt, ndt]:
            self._check_offsetfunc_works(offset_types, '_apply', dt, expected)
            self._check_offsetfunc_works(offset_types, '_apply', dt, expected_norm, normalize=True)

    def test_rollforward(self, offset_types: Any, expecteds: Dict[str, Timestamp]) -> None:
        expecteds = expecteds.copy()
        no_changes = ['Day', 'MonthBegin', 'SemiMonthBegin', 'YearBegin', 'Week', 'Hour', 'Minute', 'Second', 'Milli', 'Micro', 'Nano', 'DateOffset']
        for n in no_changes:
            expecteds[n] = Timestamp('2011/01/01 09:00')
        expecteds['BusinessHour'] = Timestamp('2011-01-03 09:00:00')
        expecteds['CustomBusinessHour'] = Timestamp('2011-01-03 09:00:00')
        norm_expected = expecteds.copy()
        for k in norm_expected:
            norm_expected[k] = Timestamp(norm_expected[k].date())
        normalized = {'Day': Timestamp('2011-01-02 00:00:00'), 'DateOffset': Timestamp('2011-01-02 00:00:00'), 'MonthBegin': Timestamp('2011-02-01 00:00:00'), 'SemiMonthBegin': Timestamp('2011-01-15 00:00:00'), 'YearBegin': Timestamp('2012-01-01 00:00:00'), 'Week': Timestamp('2011-01-08 00:00:00'), 'Hour': Timestamp('2011-01-01 00:00:00'), 'Minute': Timestamp('2011-01-01 00:00:00'), 'Second': Timestamp('2011-01-01 00:00:00'), 'Milli': Timestamp('2011-01-01 00:00:00'), 'Micro': Timestamp('2011-01-01 00:00:00')}
        norm_expected.update(normalized)
        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64('2011-01-01 09:00')
        for dt in [sdt, ndt]:
            expected = expecteds[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, 'rollforward', dt, expected)
            expected = norm_expected[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, 'rollforward', dt, expected, normalize=True)

    def test_rollback(self, offset_types: Any) -> None:
        expecteds = {'BusinessDay': Timestamp('2010-12-31 09:00:00'), 'CustomBusinessDay': Timestamp('2010-12-31 09:00:00'), 'CustomBusinessMonthEnd': Timestamp('2010-12-31 09:00:00'), 'CustomBusinessMonthBegin': Timestamp('2010-12-01 09:00:00'), 'BusinessMonthBegin': Timestamp('2010-12-01 09:00:00'), 'MonthEnd': Timestamp('2010-12-31 09:00:00'), 'SemiMonthEnd': Timestamp('2010-12-31 09:00:00'), 'BusinessMonthEnd': Timestamp('2010-12-31 09:00:00'), 'BYearBegin': Timestamp('2010-01-01 09:00:00'), 'YearEnd': Timestamp('2010-12-31 09:00:00'), 'BYearEnd': Timestamp('2010-12-31 09:00:00'), 'QuarterBegin': Timestamp('2010-12-01 09:00:00'), 'BQuarterBegin': Timestamp('2010-12-01 09:00:00'), 'QuarterEnd': Timestamp('2010-12-31 09:00:00'), 'BQuarterEnd': Timestamp('2010-12-31 09:00:00'), 'BusinessHour': Timestamp('2010-12-31 17:00:00'), 'CustomBusinessHour': Timestamp('2010-12-31 17:00:00'), 'WeekOfMonth': Timestamp('2010-12-11 09:00:00'), 'LastWeekOfMonth': Timestamp('2010-12-25 09:00:00'), 'FY5253Quarter': Timestamp('2010-10-26 09:00:00'), 'FY5253': Timestamp('2010-01-26 09:00:00'), 'Easter': Timestamp('2010-04-04 09:00:00')}
        for n in ['Day', 'MonthBegin', 'SemiMonthBegin', 'YearBegin', 'Week', 'Hour', 'Minute', 'Second', 'Milli', 'Micro', 'Nano', 'DateOffset']:
            expecteds[n] = Timestamp('2011/01/01 09:00')
        norm_expected = expecteds.copy()
        for k in norm_expected:
            norm_expected[k] = Timestamp(norm_expected[k].date())
        normalized = {'Day': Timestamp('2010-12-31 00:00:00'), 'DateOffset': Timestamp('2010-12-31 00:00:00'), 'MonthBegin': Timestamp('2010-12-01 00:00:00'), 'SemiMonthBegin': Timestamp('2010-12-15 00:00:00'), 'YearBegin': Timestamp('2010-01-01 00:00:00'), 'Week': Timestamp('2010-12-25 00:00:00'), 'Hour': Timestamp('2011-01-01 00:00:00'), 'Minute': Timestamp('2011-01-01 00:00:00'), 'Second': Timestamp('2011-01-01 00:00:00'), 'Milli': Timestamp('2011-01-01 00:00:00'), 'Micro': Timestamp('2011-01-01 00:00:00')}
        norm_expected.update(normalized)
        sdt = datetime(2011, 1, 1, 9, 0)
        ndt = np.datetime64('2011-01-01 09:00')
        for dt in [sdt, ndt]:
            expected = expecteds[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, 'rollback', dt, expected)
            expected = norm_expected[offset_types.__name__]
            self._check_offsetfunc_works(offset_types, 'rollback', dt, expected, normalize=True)

    def test_is_on_offset(self, offset_types: Any, expecteds: Dict[str, Timestamp]) -> None:
        dt = expecteds[offset_types.__name__]
        offset_s = _create_offset(offset_types)
        assert offset_s.is_on_offset(dt)
        if issubclass(offset_types, Tick):
            return
        offset_n = _create_offset(offset_types, normalize=True)
        assert not offset_n.is_on_offset(dt)
        if offset_types in (BusinessHour, CustomBusinessHour):
