#!/usr/bin/env python3
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
from typing import Any, Optional

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

    @pytest.mark.parametrize(
        'attr, expected',
        [
            ['year', 2014], 
            ['month', 12], 
            ['day', 31], 
            ['hour', 23], 
            ['minute', 59], 
            ['second', 0], 
            ['microsecond', 0], 
            ['nanosecond', 0], 
            ['dayofweek', 2], 
            ['day_of_week', 2], 
            ['quarter', 4], 
            ['dayofyear', 365], 
            ['day_of_year', 365], 
            ['week', 1], 
            ['daysinmonth', 31]
        ]
    )
    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_fields(self, attr: str, expected: int, tz: Optional[str]) -> None:
        ts = Timestamp('2014-12-31 23:59:00', tz=tz)
        result: int = getattr(ts, attr)
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize('tz', [None, 'US/Eastern'])
    def test_millisecond_raises(self, tz: Optional[str]) -> None:
        ts = Timestamp('2014-12-31 23:59:00', tz=tz)
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            _ = ts.millisecond

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
            expected_day: str = 'Monday'
            expected_month: str = 'August'
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_day = calendar.day_name[0].capitalize()
                expected_month = calendar.month_name[8].capitalize()
        result_day: str = data.day_name(time_locale)
        result_month: str = data.month_name(time_locale)
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
        result: int = Timestamp(d).week
        expected: int = 1
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

    @pytest.mark.parametrize(
        'date_string, expected',
        [
            ('0000-2-29', 1), 
            ('0000-3-1', 2), 
            ('1582-10-14', 3), 
            ('-0040-1-1', 4), 
            ('2023-06-18', 6)
        ]
    )
    def test_dow_historic(self, date_string: str, expected: int) -> None:
        ts = Timestamp(date_string)
        dow: int = ts.weekday()
        assert dow == expected

    @given(ts=st.datetimes(), sign=st.sampled_from(['-', '']))
    def test_dow_parametric(self, ts: datetime, sign: str) -> None:
        ts_str: str = f'{sign}{str(ts.year).zfill(4)}-{str(ts.month).zfill(2)}-{str(ts.day).zfill(2)}'
        result: int = Timestamp(ts_str).weekday()
        expected: int = ((np.datetime64(ts_str) - np.datetime64('1970-01-01')).astype('int64') - 4) % 7
        assert result == expected

class TestTimestamp:
    @pytest.mark.parametrize('tz', [None, zoneinfo.ZoneInfo('US/Pacific')])
    def test_disallow_setting_tz(self, tz: Optional[zoneinfo.ZoneInfo]) -> None:
        ts = Timestamp('2010')
        msg = 'Cannot directly set timezone'
        with pytest.raises(AttributeError, match=msg):
            ts.tz = tz  # type: ignore[assignment]

    def test_default_to_stdlib_utc(self) -> None:
        msg = 'Timestamp.utcnow is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert Timestamp.utcnow().tz is timezone.utc
        assert Timestamp.now('UTC').tz is timezone.utc
        assert Timestamp('2016-01-01', tz='UTC').tz is timezone.utc

    def test_tz(self) -> None:
        tstr: str = '2014-02-01 09:00'
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
        ns: list[int] = [Timestamp.min._value, Timestamp.max._value, 1000]
        for n in ns:
            assert Timestamp(n).asm8.view('i8') == np.datetime64(n, 'ns').view('i8') == n
        assert Timestamp('nat').asm8.view('i8') == np.datetime64('nat', 'ns').view('i8')

    def test_class_ops(self) -> None:

        def compare(x: Any, y: Any) -> None:
            assert int((Timestamp(x)._value - Timestamp(y)._value) / 1000000000) == 0

        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now('UTC'), datetime.now(timezone.utc))
        compare(Timestamp.now('UTC'), datetime.now(tzutc()))
        msg = 'Timestamp.utcnow is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            compare(Timestamp.utcnow(), datetime.now(timezone.utc))
        compare(Timestamp.today(), datetime.today())
        current_time: int = calendar.timegm(datetime.now().utctimetuple())
        msg = 'Timestamp.utcfromtimestamp is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ts_utc = Timestamp.utcfromtimestamp(current_time)
        assert ts_utc.timestamp() == current_time
        compare(Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time))
        compare(Timestamp.fromtimestamp(current_time, 'UTC'), datetime.fromtimestamp(current_time, timezone.utc))
        compare(Timestamp.fromtimestamp(current_time, tz='UTC'), datetime.fromtimestamp(current_time, timezone.utc))
        date_component: datetime = datetime.now(timezone.utc)
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(Timestamp.combine(date_component, time_component), datetime.combine(date_component, time_component))

    def test_basics_nanos(self) -> None:
        val: np.int64 = np.int64(946684800000000000).view('M8[ns]')
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
        d: dict[datetime, int] = {datetime(2011, 1, 1): 5}
        stamp = Timestamp(datetime(2011, 1, 1))
        assert d[stamp] == 5

    @pytest.mark.parametrize(
        'timezone, year, month, day, hour',
        [['America/Chicago', 2013, 11, 3, 1], ['America/Santiago', 2021, 4, 3, 23]]
    )
    def test_hash_timestamp_with_fold(self, timezone: str, year: int, month: int, day: int, hour: int) -> None:
        test_timezone = gettz(timezone)
        transition_1 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
        transition_2 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
        assert hash(transition_1) == hash(transition_2)

class TestTimestampNsOperations:
    def test_nanosecond_string_parsing(self) -> None:
        ts = Timestamp('2013-05-01 07:15:45.123456789')
        expected_repr = '2013-05-01 07:15:45.123456789'
        expected_value: int = 1367392545123456789
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
        expected: int = 1293840000000000005
        t = Timestamp('2011-01-01') + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
        assert t.nanosecond == 5
        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
        assert t.nanosecond == 5
        t = Timestamp('2011-01-01 00:00:00.000000005')
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000005')"
        assert t._value == expected
        assert t.nanosecond == 5
        expected = 1293840000000000010
        t = t + offsets.Nano(5)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t._value == expected
        assert t.nanosecond == 10
        t = Timestamp(t)
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t._value == expected
        assert t.nanosecond == 10
        t = Timestamp('2011-01-01 00:00:00.000000010')
        assert repr(t) == "Timestamp('2011-01-01 00:00:00.000000010')"
        assert t._value == expected
        assert t.nanosecond == 10

class TestTimestampConversion:
    def test_conversion(self) -> None:
        ts = Timestamp('2000-01-01').as_unit('ns')
        result: datetime = ts.to_pydatetime()
        expected: datetime = datetime(2000, 1, 1)
        assert result == expected
        assert type(result) == type(expected)
        result_np = ts.to_datetime64()
        expected_np = np.datetime64(ts._value, 'ns')
        assert result_np == expected_np
        assert type(result_np) == type(expected_np)
        assert result_np.dtype == expected_np.dtype

    def test_to_period_tz_warning(self) -> None:
        ts = Timestamp('2009-04-15 16:17:18', tz='US/Eastern')
        with tm.assert_produces_warning(UserWarning, match='drop timezone information'):
            ts.to_period('D')

    def test_to_numpy_alias(self) -> None:
        ts = Timestamp(datetime.now())
        assert ts.to_datetime64() == ts.to_numpy()
        msg = 'dtype and copy arguments are ignored'
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy('M8[s]')
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy(copy=True)

class TestNonNano:
    @pytest.fixture(params=['s', 'ms', 'us'])
    def reso(self, request: Any) -> str:
        return request.param

    @pytest.fixture
    def dt64(self, reso: str) -> np.datetime64:
        return np.datetime64('2016-01-01', reso)

    @pytest.fixture
    def ts(self, dt64: np.datetime64) -> Timestamp:
        return Timestamp._from_dt64(dt64)

    @pytest.fixture
    def ts_tz(self, ts: Timestamp, tz_aware_fixture: Any) -> Timestamp:
        tz = maybe_get_tz(tz_aware_fixture)
        return Timestamp._from_value_and_reso(ts._value, ts._creso, tz)

    def test_non_nano_construction(self, dt64: np.datetime64, ts: Timestamp, reso: str) -> None:
        assert ts._value == dt64.view('i8')
        if reso == 's':
            assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value
        elif reso == 'ms':
            assert ts._creso == NpyDatetimeUnit.NPY_FR_ms.value
        elif reso == 'us':
            assert ts._creso == NpyDatetimeUnit.NPY_FR_us.value

    def test_non_nano_fields(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        assert ts.year == alt.year
        assert ts.month == alt.month
        assert ts.day == alt.day
        assert ts.hour == ts.minute == ts.second == ts.microsecond == 0
        assert ts.nanosecond == 0
        assert ts.to_julian_date() == alt.to_julian_date()
        assert ts.weekday() == alt.weekday()
        assert ts.isoweekday() == alt.isoweekday()

    def test_start_end_fields(self, ts: Timestamp) -> None:
        assert ts.is_year_start
        assert ts.is_quarter_start
        assert ts.is_month_start
        assert not ts.is_year_end
        assert not ts.is_month_end
        assert not ts.is_month_end
        assert ts.is_year_start
        assert ts.is_quarter_start
        assert ts.is_month_start
        assert not ts.is_year_end
        assert not ts.is_month_end
        assert not ts.is_month_end

    def test_day_name(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        assert ts.day_name() == alt.day_name()

    def test_month_name(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        assert ts.month_name() == alt.month_name()

    def test_tz_convert(self, ts: Timestamp) -> None:
        ts = Timestamp._from_value_and_reso(ts._value, ts._creso, timezone.utc)
        tz = zoneinfo.ZoneInfo('US/Pacific')
        result = ts.tz_convert(tz)
        assert isinstance(result, Timestamp)
        assert result._creso == ts._creso
        assert tz_compare(result.tz, tz)

    def test_repr(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        assert str(ts) == str(alt)
        assert repr(ts) == repr(alt)

    def test_comparison(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        assert ts == dt64
        assert dt64 == ts
        assert ts == alt
        assert alt == ts
        assert not ts != dt64
        assert not dt64 != ts
        assert not ts != alt
        assert not alt != ts
        assert not ts < dt64
        assert not dt64 < ts
        assert not ts < alt
        assert not alt < ts
        assert not ts > dt64
        assert not dt64 > ts
        assert not ts > alt
        assert not alt > ts
        assert ts >= dt64
        assert dt64 >= ts
        assert ts >= alt
        assert alt >= ts
        assert ts <= dt64
        assert dt64 <= ts
        assert ts <= alt
        assert alt <= ts

    def test_cmp_cross_reso(self) -> None:
        dt64 = np.datetime64(9223372800, 's')
        ts = Timestamp._from_dt64(dt64)
        other = Timestamp(dt64 - 3600 * 24).as_unit('ns')
        assert other < ts
        assert other.asm8 > ts.asm8
        assert ts > other
        assert ts.asm8 < other.asm8
        assert not other == ts
        assert ts != other

    @pytest.mark.xfail(reason='Dispatches to np.datetime64 which is wrong')
    def test_cmp_cross_reso_reversed_dt64(self) -> None:
        dt64 = np.datetime64(106752, 'D')
        ts = Timestamp._from_dt64(dt64)
        other = Timestamp(dt64 - 1)
        assert other.asm8 < ts

    def test_pickle(self, ts: Timestamp, tz_aware_fixture: Any) -> None:
        tz = maybe_get_tz(tz_aware_fixture)
        ts = Timestamp._from_value_and_reso(ts._value, ts._creso, tz)
        rt = tm.round_trip_pickle(ts)
        assert rt._creso == ts._creso
        assert rt == ts

    def test_normalize(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        result = ts.normalize()
        assert result._creso == ts._creso
        assert result == alt.normalize()

    def test_asm8(self, dt64: np.datetime64, ts: Timestamp) -> None:
        rt = ts.asm8
        assert rt == dt64
        assert rt.dtype == dt64.dtype

    def test_to_numpy(self, dt64: np.datetime64, ts: Timestamp) -> None:
        res = ts.to_numpy()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_to_datetime64(self, dt64: np.datetime64, ts: Timestamp) -> None:
        res = ts.to_datetime64()
        assert res == dt64
        assert res.dtype == dt64.dtype

    def test_timestamp(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        assert ts.timestamp() == alt.timestamp()

    def test_to_period(self, dt64: np.datetime64, ts: Timestamp) -> None:
        alt = Timestamp(dt64)
        assert ts.to_period('D') == alt.to_period('D')

    @pytest.mark.parametrize('td', [timedelta(days=4), Timedelta(days=4), np.timedelta64(4, 'D')])
    def test_addsub_timedeltalike_non_nano(self, dt64: np.datetime64, ts: Timestamp, td: Any) -> None:
        exp_reso = max(ts._creso, Timedelta(td)._creso)
        result = ts - td
        expected = Timestamp(dt64) - td
        assert isinstance(result, Timestamp)
        assert result._creso == exp_reso
        assert result == expected
        result = ts + td
        expected = Timestamp(dt64) + td
        assert isinstance(result, Timestamp)
        assert result._creso == exp_reso
        assert result == expected
        result = td + ts
        expected = td + Timestamp(dt64)
        assert isinstance(result, Timestamp)
        assert result._creso == exp_reso
        assert result == expected

    def test_addsub_offset(self, ts_tz: Timestamp) -> None:
        off = offsets.YearEnd(1)
        result = ts_tz + off
        assert isinstance(result, Timestamp)
        assert result._creso == ts_tz._creso
        if ts_tz.month == 12 and ts_tz.day == 31:
            assert result.year == ts_tz.year + 1
        else:
            assert result.year == ts_tz.year
        assert result.day == 31
        assert result.month == 12
        assert tz_compare(result.tz, ts_tz.tz)
        result = ts_tz - off
        assert isinstance(result, Timestamp)
        assert result._creso == ts_tz._creso
        assert result.year == ts_tz.year - 1
        assert result.day == 31
        assert result.month == 12
        assert tz_compare(result.tz, ts_tz.tz)

    def test_sub_datetimelike_mismatched_reso(self, ts_tz: Timestamp) -> None:
        ts = ts_tz
        unit_map = {NpyDatetimeUnit.NPY_FR_us.value: 'ms', 
                    NpyDatetimeUnit.NPY_FR_ms.value: 's', 
                    NpyDatetimeUnit.NPY_FR_s.value: 'us'}
        unit: str = unit_map[ts._creso]
        other = ts.as_unit(unit)
        assert other._creso != ts._creso
        result = ts - other
        assert isinstance(result, Timedelta)
        assert result._value == 0
        assert result._creso == max(ts._creso, other._creso)
        result = other - ts
        assert isinstance(result, Timedelta)
        assert result._value == 0
        assert result._creso == max(ts._creso, other._creso)
        if ts._creso < other._creso:
            other2 = other + Timedelta._from_value_and_reso(1, other._creso)
            exp = ts.as_unit(other.unit) - other2
            res = ts - other2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
            res = other2 - ts
            assert res == -exp
            assert res._creso == max(ts._creso, other._creso)
        else:
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._creso)
            exp = ts2 - other.as_unit(ts2.unit)
            res = ts2 - other
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
            res = other - ts2
            assert res == -exp
            assert res._creso == max(ts._creso, other._creso)

    def test_sub_timedeltalike_mismatched_reso(self, ts_tz: Timestamp) -> None:
        ts = ts_tz
        unit_map = {NpyDatetimeUnit.NPY_FR_us.value: 'ms', 
                    NpyDatetimeUnit.NPY_FR_ms.value: 's', 
                    NpyDatetimeUnit.NPY_FR_s.value: 'us'}
        unit: str = unit_map[ts._creso]
        other = Timedelta(0).as_unit(unit)
        assert other._creso != ts._creso
        result = ts + other
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._creso == max(ts._creso, other._creso)
        result = other + ts
        assert isinstance(result, Timestamp)
        assert result == ts
        assert result._creso == max(ts._creso, other._creso)
        if ts._creso < other._creso:
            other2 = other + Timedelta._from_value_and_reso(1, other._creso)
            exp = ts.as_unit(other.unit) + other2
            res = ts + other2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
            res = other2 + ts
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
        else:
            ts2 = ts + Timedelta._from_value_and_reso(1, ts._creso)
            exp = ts2 + other.as_unit(ts2.unit)
            res = ts2 + other
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)
            res = other + ts2
            assert res == exp
            assert res._creso == max(ts._creso, other._creso)

    def test_addition_doesnt_downcast_reso(self) -> None:
        ts = Timestamp(year=2022, month=1, day=1, microsecond=999999).as_unit('us')
        td = Timedelta(microseconds=1).as_unit('us')
        res = ts + td
        assert res._creso == ts._creso

    def test_sub_timedelta64_mismatched_reso(self, ts_tz: Timestamp) -> None:
        ts = ts_tz
        res = ts + np.timedelta64(1, 'ns')
        exp = ts.as_unit('ns') + np.timedelta64(1, 'ns')
        assert exp == res
        assert exp._creso == NpyDatetimeUnit.NPY_FR_ns.value

    def test_min(self, ts: Timestamp) -> None:
        assert ts.min <= ts
        assert ts.min._creso == ts._creso
        assert ts.min._value == NaT._value + 1

    def test_max(self, ts: Timestamp) -> None:
        assert ts.max >= ts
        assert ts.max._creso == ts._creso
        assert ts.max._value == np.iinfo(np.int64).max

    def test_resolution(self, ts: Timestamp) -> None:
        expected = Timedelta._from_value_and_reso(1, ts._creso)
        result = ts.resolution
        assert result == expected
        assert result._creso == expected._creso

    def test_out_of_ns_bounds(self) -> None:
        result = Timestamp(-52700112000, unit='s')
        assert result == Timestamp('0300-01-01')
        assert result.to_numpy() == np.datetime64('0300-01-01T00:00:00', 's')

def test_timestamp_class_min_max_resolution() -> None:
    assert Timestamp.min == Timestamp(NaT._value + 1)
    assert Timestamp.min._creso == NpyDatetimeUnit.NPY_FR_ns.value
    assert Timestamp.max == Timestamp(np.iinfo(np.int64).max)
    assert Timestamp.max._creso == NpyDatetimeUnit.NPY_FR_ns.value
    assert Timestamp.resolution == Timedelta(1)
    assert Timestamp.resolution._creso == NpyDatetimeUnit.NPY_FR_ns.value

def test_delimited_date() -> None:
    with tm.assert_produces_warning(None):
        result = Timestamp('13-01-2000')
    expected = Timestamp(2000, 1, 13)
    assert result == expected

def test_utctimetuple() -> None:
    ts = Timestamp('2000-01-01', tz='UTC')
    result = ts.utctimetuple()
    expected = time.struct_time((2000, 1, 1, 0, 0, 0, 5, 1, 0))
    assert result == expected

def test_negative_dates() -> None:
    ts = Timestamp('-2000-01-01')
    msg = (" not yet supported on Timestamps which are outside the range of Python's standard library. "
           "For now, please call the components you need \\(such as `.year` and `.month`\\) and construct your string from there.$")
    func = '^strftime'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.strftime('%Y')
    msg = " not yet supported on Timestamps which are outside the range of Python's standard library. "
    func = '^date'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.date()
    func = '^isocalendar'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.isocalendar()
    func = '^timetuple'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.timetuple()
    func = '^toordinal'
    with pytest.raises(NotImplementedError, match=func + msg):
        ts.toordinal()