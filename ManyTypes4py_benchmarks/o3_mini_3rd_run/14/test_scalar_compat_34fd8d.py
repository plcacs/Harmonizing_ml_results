#!/usr/bin/env python
"""
Tests for DatetimeIndex methods behaving like their Timestamp counterparts
"""
import calendar
import locale
import unicodedata
from datetime import date, datetime, time
from typing import Any, Optional

import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as st

from pandas._libs.tslibs import timezones
from pandas import DatetimeIndex, Index, NaT, Timestamp, date_range, offsets
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray


class TestDatetimeIndexOps:
    def test_dti_no_millisecond_field(self) -> None:
        msg: str = "type object 'DatetimeIndex' has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex.millisecond
        msg = "'DatetimeIndex' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex([]).millisecond

    def test_dti_time(self) -> None:
        rng = date_range('1/1/2000', freq='12min', periods=10)
        result = Index(rng).time
        expected = [t.time() for t in rng]
        assert (result == expected).all()

    def test_dti_date(self) -> None:
        rng = date_range('1/1/2000', freq='12h', periods=10)
        result = Index(rng).date
        expected = [t.date() for t in rng]
        assert (result == expected).all()

    @pytest.mark.parametrize(
        'dtype', [None, 'datetime64[ns, CET]', 'datetime64[ns, EST]', 'datetime64[ns, UTC]']
    )
    def test_dti_date2(self, dtype: Optional[str]) -> None:
        expected = np.array([date(2018, 6, 4), NaT])
        index = DatetimeIndex(['2018-06-04 10:00:00', NaT], dtype=dtype)
        result = index.date
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        'dtype', [None, 'datetime64[ns, CET]', 'datetime64[ns, EST]', 'datetime64[ns, UTC]']
    )
    def test_dti_time2(self, dtype: Optional[str]) -> None:
        expected = np.array([time(10, 20, 30), NaT])
        index = DatetimeIndex(['2018-06-04 10:20:30', NaT], dtype=dtype)
        result = index.time
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_timetz(self, tz_naive_fixture: str) -> None:
        tz = timezones.maybe_get_tz(tz_naive_fixture)
        expected = np.array([time(10, 20, 30, tzinfo=tz), NaT])
        index = DatetimeIndex(['2018-06-04 10:20:30', NaT], tz=tz)
        result = index.timetz
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        'field',
        ['dayofweek', 'day_of_week', 'dayofyear', 'day_of_year', 'quarter', 'days_in_month',
         'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']
    )
    def test_dti_timestamp_fields(self, field: str) -> None:
        idx = date_range('2020-01-01', periods=10)
        expected = getattr(idx, field)[-1]
        result = getattr(Timestamp(idx[-1]), field)
        assert result == expected

    def test_dti_nanosecond(self) -> None:
        dti = DatetimeIndex(np.arange(10))
        expected = Index(np.arange(10, dtype=np.int32))
        tm.assert_index_equal(dti.nanosecond, expected)

    @pytest.mark.parametrize('prefix', ['', 'dateutil/'])
    def test_dti_hour_tzaware(self, prefix: str) -> None:
        strdates = ['1/1/2012', '3/1/2012', '4/1/2012']
        rng = DatetimeIndex(strdates, tz=prefix + 'US/Eastern')
        assert (rng.hour == 0).all()
        dr = date_range('2011-10-02 00:00', freq='h', periods=10, tz=prefix + 'America/Atikokan')
        expected = Index(np.arange(10, dtype=np.int32))
        tm.assert_index_equal(dr.hour, expected)

    @pytest.mark.parametrize('time_locale', [None] + tm.get_locales())
    def test_day_name_month_name(self, time_locale: Optional[str]) -> None:
        if time_locale is None:
            expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            expected_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_days = list(calendar.day_name)
                expected_months = list(calendar.month_name[1:])
        dti = date_range(freq='D', start=datetime(1998, 1, 1), periods=365)
        english_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day, name, eng_name in zip(range(4, 11), expected_days, english_days):
            name_cap = name.capitalize()
            assert dti.day_name(locale=time_locale)[day] == name_cap
            assert dti.day_name(locale=None)[day] == eng_name
            ts = Timestamp(datetime(2016, 4, day))
            assert ts.day_name(locale=time_locale) == name_cap
        dti = dti.append(DatetimeIndex([NaT]))
        assert np.isnan(dti.day_name(locale=time_locale)[-1])
        ts = Timestamp(NaT)
        assert np.isnan(ts.day_name(locale=time_locale))
        dti = date_range(freq='ME', start='2012', end='2013')
        result = dti.month_name(locale=time_locale)
        expected = Index([month.capitalize() for month in expected_months])
        result = result.str.normalize('NFD')
        expected = expected.str.normalize('NFD')
        tm.assert_index_equal(result, expected)
        for item, exp_month in zip(dti, expected_months):
            result_name = item.month_name(locale=time_locale)
            expected_name = exp_month.capitalize()
            result_norm = unicodedata.normalize('NFD', result_name)
            expected_norm = unicodedata.normalize('NFD', expected_name)
            assert result_norm == expected_norm
        dti = dti.append(DatetimeIndex([NaT]))
        assert np.isnan(dti.month_name(locale=time_locale)[-1])

    def test_dti_week(self) -> None:
        dates = ['2013/12/29', '2013/12/30', '2013/12/31']
        dates = DatetimeIndex(dates, tz='Europe/Brussels')
        expected = [52, 1, 1]
        assert dates.isocalendar().week.tolist() == expected
        assert [d.weekofyear for d in dates] == expected

    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_dti_fields(self, tz: Optional[str]) -> None:
        dti = date_range(freq='D', start=datetime(1998, 1, 1), periods=365, tz=tz)
        assert dti.year[0] == 1998
        assert dti.month[0] == 1
        assert dti.day[0] == 1
        assert dti.hour[0] == 0
        assert dti.minute[0] == 0
        assert dti.second[0] == 0
        assert dti.microsecond[0] == 0
        assert dti.dayofweek[0] == 3
        assert dti.dayofyear[0] == 1
        assert dti.dayofyear[120] == 121
        assert dti.isocalendar().week.iloc[0] == 1
        assert dti.isocalendar().week.iloc[120] == 18
        assert dti.quarter[0] == 1
        assert dti.quarter[120] == 2
        assert dti.days_in_month[0] == 31
        assert dti.days_in_month[90] == 30
        assert dti.is_month_start[0]
        assert not dti.is_month_start[1]
        assert dti.is_month_start[31]
        assert dti.is_quarter_start[0]
        assert dti.is_quarter_start[90]
        assert dti.is_year_start[0]
        assert not dti.is_year_start[364]
        assert not dti.is_month_end[0]
        assert dti.is_month_end[30]
        assert not dti.is_month_end[31]
        assert dti.is_month_end[364]
        assert not dti.is_quarter_end[0]
        assert not dti.is_quarter_end[30]
        assert dti.is_quarter_end[89]
        assert dti.is_quarter_end[364]
        assert not dti.is_year_end[0]
        assert dti.is_year_end[364]
        assert len(dti.year) == 365
        assert len(dti.month) == 365
        assert len(dti.day) == 365
        assert len(dti.hour) == 365
        assert len(dti.minute) == 365
        assert len(dti.second) == 365
        assert len(dti.microsecond) == 365
        assert len(dti.dayofweek) == 365
        assert len(dti.dayofyear) == 365
        assert len(dti.isocalendar()) == 365
        assert len(dti.quarter) == 365
        assert len(dti.is_month_start) == 365
        assert len(dti.is_month_end) == 365
        assert len(dti.is_quarter_start) == 365
        assert len(dti.is_quarter_end) == 365
        assert len(dti.is_year_start) == 365
        assert len(dti.is_year_end) == 365
        dti.name = 'name'
        for accessor in DatetimeArray._field_ops:
            res = getattr(dti, accessor)
            assert len(res) == 365
            assert isinstance(res, Index)
            assert res.name == 'name'
        for accessor in DatetimeArray._bool_ops:
            res = getattr(dti, accessor)
            assert len(res) == 365
            assert isinstance(res, np.ndarray)
        res = dti[dti.is_quarter_start]
        exp = dti[[0, 90, 181, 273]]
        tm.assert_index_equal(res, exp)
        res = dti[dti.is_leap_year]
        exp = DatetimeIndex([], freq='D', tz=dti.tz, name='name').as_unit('ns')
        tm.assert_index_equal(res, exp)

    def test_dti_is_year_quarter_start(self) -> None:
        dti = date_range(freq='BQE-FEB', start=datetime(1998, 1, 1), periods=4)
        assert sum(dti.is_quarter_start) == 0
        assert sum(dti.is_quarter_end) == 4
        assert sum(dti.is_year_start) == 0
        assert sum(dti.is_year_end) == 1

    def test_dti_is_month_start(self) -> None:
        dti = DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'])
        assert dti.is_month_start[0] == 1

    def test_dti_is_month_start_custom(self) -> None:
        bday_egypt = offsets.CustomBusinessDay(weekmask='Sun Mon Tue Wed Thu')
        dti = date_range(datetime(2013, 4, 30), periods=5, freq=bday_egypt)
        msg: str = 'Custom business days is not supported by is_month_start'
        with pytest.raises(ValueError, match=msg):
            dti.is_month_start

    @pytest.mark.parametrize(
        'timestamp, freq, periods, expected_values',
        [
            ('2017-12-01', 'MS', 3, np.array([False, True, False])),
            ('2017-12-01', 'QS', 3, np.array([True, False, False])),
            ('2017-12-01', 'YS', 3, np.array([True, True, True])),
        ]
    )
    def test_dti_dr_is_year_start(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        result = date_range(timestamp, freq=freq, periods=periods).is_year_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        'timestamp, freq, periods, expected_values',
        [
            ('2017-12-01', 'ME', 3, np.array([True, False, False])),
            ('2017-12-01', 'QE', 3, np.array([True, False, False])),
            ('2017-12-01', 'YE', 3, np.array([True, True, True])),
        ]
    )
    def test_dti_dr_is_year_end(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        result = date_range(timestamp, freq=freq, periods=periods).is_year_end
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        'timestamp, freq, periods, expected_values',
        [
            ('2017-12-01', 'MS', 3, np.array([False, True, False])),
            ('2017-12-01', 'QS', 3, np.array([True, True, True])),
            ('2017-12-01', 'YS', 3, np.array([True, True, True])),
        ]
    )
    def test_dti_dr_is_quarter_start(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        result = date_range(timestamp, freq=freq, periods=periods).is_quarter_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        'timestamp, freq, periods, expected_values',
        [
            ('2017-12-01', 'ME', 3, np.array([True, False, False])),
            ('2017-12-01', 'QE', 3, np.array([True, True, True])),
            ('2017-12-01', 'YE', 3, np.array([True, True, True])),
        ]
    )
    def test_dti_dr_is_quarter_end(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        result = date_range(timestamp, freq=freq, periods=periods).is_quarter_end
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        'timestamp, freq, periods, expected_values',
        [
            ('2017-12-01', 'MS', 3, np.array([True, True, True])),
            ('2017-12-01', 'QS', 3, np.array([True, True, True])),
            ('2017-12-01', 'YS', 3, np.array([True, True, True])),
        ]
    )
    def test_dti_dr_is_month_start(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        result = date_range(timestamp, freq=freq, periods=periods).is_month_start
        tm.assert_numpy_array_equal(result, expected_values)

    @pytest.mark.parametrize(
        'timestamp, freq, periods, expected_values',
        [
            ('2017-12-01', 'ME', 3, np.array([True, True, True])),
            ('2017-12-01', 'QE', 3, np.array([True, True, True])),
            ('2017-12-01', 'YE', 3, np.array([True, True, True])),
        ]
    )
    def test_dti_dr_is_month_end(self, timestamp: str, freq: str, periods: int, expected_values: np.ndarray) -> None:
        result = date_range(timestamp, freq=freq, periods=periods).is_month_end
        tm.assert_numpy_array_equal(result, expected_values)

    def test_dti_is_year_quarter_start_doubledigit_freq(self) -> None:
        dr = date_range('2017-01-01', periods=2, freq='10YS')
        assert all(dr.is_year_start)
        dr = date_range('2017-01-01', periods=2, freq='10QS')
        assert all(dr.is_quarter_start)

    def test_dti_is_year_start_freq_custom_business_day_with_digit(self) -> None:
        dr = date_range('2020-01-01', periods=2, freq='2C')
        msg: str = 'Custom business days is not supported by is_year_start'
        with pytest.raises(ValueError, match=msg):
            dr.is_year_start

    @pytest.mark.parametrize('freq', ['3BMS', offsets.BusinessMonthBegin(3)])
    def test_dti_is_year_quarter_start_freq_business_month_begin(self, freq: Any) -> None:
        dr = date_range('2020-01-01', periods=5, freq=freq)
        result = [x.is_year_start for x in dr]
        assert result == [True, False, False, False, True]
        dr = date_range('2020-01-01', periods=4, freq=freq)
        result = [x.is_quarter_start for x in dr]
        assert all(dr.is_quarter_start)


@given(dt=st.datetimes(min_value=datetime(1960, 1, 1), max_value=datetime(1980, 1, 1)),
       n=st.integers(min_value=1, max_value=10),
       freq=st.sampled_from(['MS', 'QS', 'YS']))
@pytest.mark.slow
def test_against_scalar_parametric(freq: str, dt: datetime, n: int) -> None:
    freq = f'{n}{freq}'
    d = date_range(dt, periods=3, freq=freq)
    result = list(d.is_year_start)
    expected = [x.is_year_start for x in d]
    assert result == expected
