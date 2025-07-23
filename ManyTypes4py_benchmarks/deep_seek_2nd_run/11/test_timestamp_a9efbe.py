"""test the scalar Timestamp"""
import calendar
from datetime import datetime, timedelta, timezone
import locale
import time
import unicodedata
import zoneinfo
from dateutil.tz import tzlocal, tzutc
from hypothesis import given, strategies as st
import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz, get_timezone, maybe_get_tz, tz_compare
from pandas.compat import IS64
from pandas import NaT, Timedelta, Timestamp
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Any, Dict, List, Optional, Tuple, Union

class TestTimestampProperties:

    def test_properties_business(self) -> None:
        freq = to_offset('B')
        ts = Timestamp('2017-10-01')
        assert ts.dayofweek == 6
        assert ts.day_of_week == 6
        assert ts.is_month_start
        assert not freq.is_month_start(ts)
        assert freq.is_month_start(ts + Timedelta(days=1))
        assert not freq.is_quarter_start(ts)
        assert freq.is_quarter_start(ts + Timedelta(days=1))
        ts = Timestamp('2017-09-30')
        assert ts.dayofweek == 5
        assert ts.day_of_week == 5
        assert ts.is_month_end
        assert not freq.is_month_end(ts)
        assert freq.is_month_end(ts - Timedelta(days=1))
        assert ts.is_quarter_end
        assert not freq.is_quarter_end(ts)
        assert freq.is_quarter_end(ts - Timedelta(days=1))

    @pytest.mark.parametrize('attr, expected', [['year', 2014], ['month', 12], ['day', 31], ['hour', 23], ['minute', 59], ['second', 0], ['microsecond', 0], ['nanosecond', 0], ['dayofweek', 2], ['day_of_week', 2], ['quarter', 4], ['dayofyear', 365], ['day_of_year', 365], ['week', 1], ['daysinmonth', 31]])
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_fields(self, attr: str, expected: int, tz: Optional[str]) -> None:
        ts = Timestamp('2014-12-31 23:59:00', tz=tz)
        result = getattr(ts, attr)
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_millisecond_raises(self, tz: Optional[str]) -> None:
        ts = Timestamp('2014-12-31 23:59:00', tz=tz)
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            ts.millisecond

    @pytest.mark.parametrize('start', ['is_month_start', 'is_quarter_start', 'is_year_start'])
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_is_start(self, start: str, tz: Optional[str]) -> None:
        ts = Timestamp('2014-01-01 00:00:00', tz=tz)
        assert getattr(ts, start)

    @pytest.mark.parametrize('end', ['is_month_end', 'is_year_end', 'is_quarter_end'])
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_is_end(self, end: str, tz: Optional[str]) -> None:
        ts = Timestamp('2014-12-31 23:59:59', tz=tz)
        assert getattr(ts, end)

    @pytest.mark.parametrize('tz', [None, 'EST'])
    @pytest.mark.parametrize('time_locale', [None] + tm.get_locales())
    def test_names(self, tz: Optional[str], time_locale: Optional[str]) -> None:
        data = Timestamp('2017-08-28 23:00:00', tz=tz)
        if time_locale is None:
            expected_day = 'Monday'
            expected_month = 'August'
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_day = calendar.day_name[0].capitalize()
                expected_month = calendar.month_name[8].capitalize()
        result_day = data.day_name(time_locale)
        result_month = data.month_name(time_locale)
        expected_day = unicodedata.normalize('NFD', expected_day)
        expected_month = unicodedata.normalize('NFD', expected_month)
        result_day = unicodedata.normalize('NFD', result_day)
        result_month = unicodedata.normalize('NFD', result_month)
        assert result_day == expected_day
        assert result_month == expected_month
        nan_ts = Timestamp(NaT)
        assert np.isnan(nan_ts.day_name(time_locale))
        assert np.isnan(nan_ts.month_name(time_locale))

    def test_is_leap_year(self, tz_naive_fixture: Any) -> None:
        tz = tz_naive_fixture
        if not IS64 and tz == tzlocal():
            pytest.skip('tzlocal() on a 32 bit platform causes internal overflow errors')
        dt = Timestamp('2000-01-01 00:00:00', tz=tz)
        assert dt.is_leap_year
        assert isinstance(dt.is_leap_year, bool)
        dt = Timestamp('1999-01-01 00:00:00', tz=tz)
        assert not dt.is_leap_year
        dt = Timestamp('2004-01-01 00:00:00', tz=tz)
        assert dt.is_leap_year
        dt = Timestamp('2100-01-01 00:00:00', tz=tz)
        assert not dt.is_leap_year

    def test_woy_boundary(self) -> None:
        d = datetime(2013, 12, 31)
        result = Timestamp(d).week
        expected = 1
        assert result == expected
        d = datetime(2008, 12, 28)
        result = Timestamp(d).week
        expected = 52
        assert result == expected
        d = datetime(2009, 12, 31)
        result = Timestamp(d).week
        expected = 53
        assert result == expected
        d = datetime(2010, 1, 1)
        result = Timestamp(d).week
        expected = 53
        assert result == expected
        d = datetime(2010, 1, 3)
        result = Timestamp(d).week
        expected = 53
        assert result == expected
        result = np.array([Timestamp(datetime(*args)).week for args in [(2000, 1, 1), (2000, 1, 2), (2005, 1, 1), (2005, 1, 2)]])
        assert (result == [52, 52, 53, 53]).all()

    def test_resolution(self) -> None:
        dt = Timestamp('2100-01-01 00:00:00.000000000')
        assert dt.resolution == Timedelta(nanoseconds=1)
        assert Timestamp.resolution == Timedelta(nanoseconds=1)
        assert dt.as_unit('us').resolution == Timedelta(microseconds=1)
        assert dt.as_unit('ms').resolution == Timedelta(milliseconds=1)
        assert dt.as_unit('s').resolution == Timedelta(seconds=1)

    @pytest.mark.parametrize('date_string, expected', [('0000-2-29', 1), ('0000-3-1', 2), ('1582-10-14', 3), ('-0040-1-1', 4), ('2023-06-18', 6)])
    def test_dow_historic(self, date_string: str, expected: int) -> None:
        ts = Timestamp(date_string)
        dow = ts.weekday()
        assert dow == expected

    @given(ts=st.datetimes(), sign=st.sampled_from(['-', '']))
    def test_dow_parametric(self, ts: datetime, sign: str) -> None:
        ts = f'{sign}{str(ts.year).zfill(4)}-{str(ts.month).zfill(2)}-{str(ts.day).zfill(2)}'
        result = Timestamp(ts).weekday()
        expected = ((np.datetime64(ts) - np.datetime64('1970-01-01')).astype('int64') - 4) % 7
        assert result == expected

class TestTimestamp:

    @pytest.mark.parametrize('tz', [None, zoneinfo.ZoneInfo('US/Pacific')])
    def test_disallow_setting_tz(self, tz: Any) -> None:
        ts = Timestamp('2010')
        msg = 'Cannot directly set timezone'
        with pytest.raises(AttributeError, match=msg):
            ts.tz = tz

    def test_default_to_stdlib_utc(self) -> None:
        msg = 'Timestamp.utcnow is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert Timestamp.utcnow().tz is timezone.utc
        assert Timestamp.now('UTC').tz is timezone.utc
        assert Timestamp('2016-01-01', tz='UTC').tz is timezone.utc

    def test_tz(self) -> None:
        tstr = '2014-02-01 09:00'
        ts = Timestamp(tstr)
        local = ts.tz_localize('Asia/Tokyo')
        assert local.hour == 9
        assert local == Timestamp(tstr, tz='Asia/Tokyo')
        conv = local.tz_convert('US/Eastern')
        assert conv == Timestamp('2014-01-31 19:00', tz='US/Eastern')
        assert conv.hour == 19
        ts = Timestamp(tstr) + offsets.Nano(5)
        local = ts.tz_localize('Asia/Tokyo')
        assert local.hour == 9
        assert local.nanosecond == 5
        conv = local.tz_convert('US/Eastern')
        assert conv.nanosecond == 5
        assert conv.hour == 19

    def test_utc_z_designator(self) -> None:
        assert get_timezone(Timestamp('2014-11-02 01:00Z').tzinfo) is timezone.utc

    def test_asm8(self) -> None:
        ns = [Timestamp.min._value, Timestamp.max._value, 1000]
        for n in ns:
            assert Timestamp(n).asm8.view('i8') == np.datetime64(n, 'ns').view('i8') == n
        assert Timestamp('nat').asm8.view('i8') == np.datetime64('nat', 'ns').view('i8')

    def test_class_ops(self) -> None:
        def compare(x: Any, y: Any) -> None:
            assert int((Timestamp(x)._value - Timestamp(y)._value) / 1000000000.0) == 0
        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now('UTC'), datetime.now(timezone.utc))
        compare(Timestamp.now('UTC'), datetime.now(tzutc()))
        msg = 'Timestamp.utcnow is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            compare(Timestamp.utcnow(), datetime.now(timezone.utc))
        compare(Timestamp.today(), datetime.today())
        current_time = calendar.timegm(datetime.now().utctimetuple())
        msg = 'Timestamp.utcfromtimestamp is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ts_utc = Timestamp.utcfromtimestamp(current_time)
        assert ts_utc.timestamp() == current_time
        compare(Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time))
        compare(Timestamp.fromtimestamp(current_time, 'UTC'), datetime.fromtimestamp(current_time, timezone.utc))
        compare(Timestamp.fromtimestamp(current_time, tz='UTC'), datetime.fromtimestamp(current_time, timezone.utc))
        date_component = datetime.now(timezone.utc)
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(Timestamp.combine(date_component, time_component), datetime.combine(date_component, time_component))

    def test_basics_nanos(self) -> None:
        val = np.int64(946684800000000000).view('M8[ns]')
        stamp = Timestamp(val.view('i8') + 500)
        assert stamp.year == 2000
        assert stamp.month == 1
        assert stamp.microsecond == 0
        assert stamp.nanosecond == 500
        val = np.iinfo(np.int64).min + 80000000000000
        stamp = Timestamp(val)
        assert stamp.year == 1677
        assert stamp.month == 9
        assert stamp.day == 21
        assert stamp.microsecond == 145224
        assert stamp.nanosecond == 192

    def test_roundtrip(self) -> None:
        base = Timestamp('20140101 00:00:00').as_unit('ns')
        result = Timestamp(base._value + Timedelta('5ms')._value)
        assert result == Timestamp(f'{base}.005000')
        assert result.microsecond == 5000
        result = Timestamp(base._value + Timedelta('5us')._value)
        assert result == Timestamp(f'{base}.000005')
        assert result.microsecond == 5
        result = Timestamp(base._value + Timedelta('5ns')._value)
        assert result == Timestamp(f'{base}.000000005')
        assert result.nanosecond == 5
        assert result.microsecond == 0
        result = Timestamp(base._value + Timedelta('6ms 5us')._value)
        assert result == Timestamp(f'{base}.006005')
        assert result.microsecond == 5 + 6 * 1000
        result = Timestamp(base._value + Timedelta('200ms 5us')._value)
        assert result == Timestamp(f'{base}.200005')
        assert result.microsecond == 5 + 200 * 1000

    def test_hash_equivalent(self) -> None:
        d: Dict[datetime, int] = {datetime(2011, 1, 1): 5}
        stamp = Timestamp(datetime(2011, 1, 1))
        assert d[stamp] == 5

    @pytest.mark.parametrize('timezone, year, month, day, hour', [['America/Chicago', 2013, 11, 3, 1], ['America/Santiago', 2021, 4, 3, 23]])
    def test_hash_timestamp_with_fold(self, timezone: str, year: int, month: int, day: int, hour: int) -> None:
        test_timezone = gettz(timezone)
        transition_1 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
        transition_2 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
        assert hash(transition_1) == hash(transition_2)

class TestTimestampNsOperations:

    def test_nanosecond_string_parsing(self) -> None:
        ts = Timestamp('2013-05-01 07:15:45.123456789')
        expected_repr = '2013-05-01 07:15:45.123456789'
        expected_value = 1367392545123456789
        assert ts._value == expected_value
        assert expected_repr in repr(ts)
        ts = Timestamp('2013-05-01 07:15:45.123456789+09:00', tz='Asia/Tokyo')
        assert ts._value == expected_value - 9 * 3600 * 1000000000
        assert expected_repr in repr(ts)
        ts = Timestamp('2013-05-01 07:15:45.123456789', tz='UTC')
        assert ts._value == expected_value
        assert expected_repr in repr(ts)
        ts = Timestamp('2013-05-01 07:15:45.123456789', tz='US/Eastern')
        assert ts._value == expected_value + 4 * 3600 * 1000000000
        assert expected_repr in repr(ts)
        ts = Timestamp('20130501T071545.123456789')
        assert ts._value == expected_value
        assert expected_repr in repr(ts)

    def test_nanosecond_timestamp(self) -> None:
        expected = 1293840000000000005
        t = Timestamp('2011-01-01') + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
        assert t.nanosecond == 5
        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
       