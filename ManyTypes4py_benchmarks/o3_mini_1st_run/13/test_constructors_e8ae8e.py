from __future__ import annotations
import calendar
from datetime import date, datetime, timedelta, timezone
import zoneinfo
import dateutil.tz
from dateutil.tz import gettz, tzoffset, tzutc
import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import NA, NaT, Period, Timedelta, Timestamp
import pandas._testing as tm
from typing import Any, Callable, Dict, Optional, Union

class TestTimestampConstructorUnitKeyword:
    @pytest.mark.parametrize('typ', [int, float])
    def test_constructor_int_float_with_YM_unit(self, typ: Callable[[int], Union[int, float]]) -> None:
        val: Union[int, float] = typ(150)
        ts: Timestamp = Timestamp(val, unit='Y')
        expected: Timestamp = Timestamp('2120-01-01')
        assert ts == expected
        ts = Timestamp(val, unit='M')
        expected = Timestamp('1982-07-01')
        assert ts == expected

    @pytest.mark.parametrize('typ', [int, float])
    def test_construct_from_int_float_with_unit_out_of_bound_raises(self, typ: Callable[[int], Union[int, float]]) -> None:
        val: Union[int, float] = typ(150000000000000)
        msg: str = f"cannot convert input {val} with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(val, unit='D')

    def test_constructor_float_not_round_with_YM_unit_raises(self) -> None:
        msg: str = 'Conversion of non-round float with unit=[MY] is ambiguous'
        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit='Y')
        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit='M')

    @pytest.mark.parametrize('value, check_kwargs', [
        [946688461000000000, {}],
        [946688461000000000 / 1000, {'unit': 'us'}],
        [946688461000000000 / 1000000, {'unit': 'ms'}],
        [946688461000000000 / 1000000000, {'unit': 's'}],
        [10957, {'unit': 'D', 'h': 0}],
        [(946688461000000000 + 500000) / 1000000000, {'unit': 's', 'us': 499, 'ns': 964}],
        [(946688461000000000 + 500000000) / 1000000000, {'unit': 's', 'us': 500000}],
        [(946688461000000000 + 500000) / 1000000, {'unit': 'ms', 'us': 500}],
        [(946688461000000000 + 500000) / 1000, {'unit': 'us', 'us': 500}],
        [(946688461000000000 + 500000000) / 1000000, {'unit': 'ms', 'us': 500000}],
        [946688461000000000 / 1000.0 + 5, {'unit': 'us', 'us': 5}],
        [946688461000000000 / 1000.0 + 5000, {'unit': 'us', 'us': 5000}],
        [946688461000000000 / 1000000.0 + 0.5, {'unit': 'ms', 'us': 500}],
        [946688461000000000 / 1000000.0 + 0.005, {'unit': 'ms', 'us': 5, 'ns': 5}],
        [946688461000000000 / 1000000000.0 + 0.5, {'unit': 's', 'us': 500000}],
        [10957 + 0.5, {'unit': 'D', 'h': 12}]
    ])
    def test_construct_with_unit(self, value: Union[int, float], check_kwargs: Dict[str, Any]) -> None:
        def check(value: Union[int, float], unit: Optional[str] = None, h: int = 1, s: int = 1, us: int = 0, ns: int = 0) -> None:
            stamp: Timestamp = Timestamp(value, unit=unit)
            assert stamp.year == 2000
            assert stamp.month == 1
            assert stamp.day == 1
            assert stamp.hour == h
            if unit != 'D':
                assert stamp.minute == 1
                assert stamp.second == s
                assert stamp.microsecond == us
            else:
                assert stamp.minute == 0
                assert stamp.second == 0
                assert stamp.microsecond == 0
            assert stamp.nanosecond == ns
        check(value, **check_kwargs)

class TestTimestampConstructorFoldKeyword:
    def test_timestamp_constructor_invalid_fold_raise(self) -> None:
        msg: str = 'Valid values for the fold argument are None, 0, or 1.'
        with pytest.raises(ValueError, match=msg):
            Timestamp(123, fold=2)

    def test_timestamp_constructor_pytz_fold_raise(self) -> None:
        pytz = pytest.importorskip('pytz')
        msg: str = 'pytz timezones do not support fold. Please use dateutil timezones.'
        tz = pytz.timezone('Europe/London')
        with pytest.raises(ValueError, match=msg):
            Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)

    @pytest.mark.parametrize('fold', [0, 1])
    @pytest.mark.parametrize('ts_input', [
        1572136200000000000,
        1.5721362e+18,
        np.datetime64(1572136200000000000, 'ns'),
        '2019-10-27 01:30:00+01:00',
        datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc)
    ])
    def test_timestamp_constructor_fold_conflict(self, ts_input: Union[int, float, np.datetime64, str, datetime], fold: int) -> None:
        msg: str = 'Cannot pass fold with possibly unambiguous input: int, float, numpy.datetime64, str, or timezone-aware datetime-like. Pass naive datetime-like or build Timestamp from components.'
        with pytest.raises(ValueError, match=msg):
            Timestamp(ts_input=ts_input, fold=fold)  # type: ignore[arg-type]

    @pytest.mark.parametrize('tz', ['dateutil/Europe/London', None])
    @pytest.mark.parametrize('fold', [0, 1])
    def test_timestamp_constructor_retain_fold(self, tz: Optional[Union[str, zoneinfo.ZoneInfo]], fold: int) -> None:
        ts: Timestamp = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
        result: int = ts.fold
        expected: int = fold
        assert result == expected

    @pytest.mark.parametrize('tz', ['dateutil/Europe/London', zoneinfo.ZoneInfo('Europe/London')])
    @pytest.mark.parametrize('ts_input,fold_out', [
        (1572136200000000000, 0),
        (1572139800000000000, 1),
        ('2019-10-27 01:30:00+01:00', 0),
        ('2019-10-27 01:30:00+00:00', 1),
        (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0),
        (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1)
    ])
    def test_timestamp_constructor_infer_fold_from_value(
        self,
        tz: Union[str, zoneinfo.ZoneInfo],
        ts_input: Union[int, float, np.datetime64, str, datetime],
        fold_out: int
    ) -> None:
        ts: Timestamp = Timestamp(ts_input, tz=tz)  # type: ignore[arg-type]
        result: int = ts.fold
        expected: int = fold_out
        assert result == expected

    @pytest.mark.parametrize('tz', ['dateutil/Europe/London'])
    @pytest.mark.parametrize('fold,value_out', [(0, 1572136200000000), (1, 1572139800000000)])
    def test_timestamp_constructor_adjust_value_for_fold(self, tz: str, fold: int, value_out: int) -> None:
        ts_input: datetime = datetime(2019, 10, 27, 1, 30)
        ts: Timestamp = Timestamp(ts_input, tz=tz, fold=fold)
        result: int = ts._value  # type: ignore[attr-defined]
        expected: int = value_out
        assert result == expected

class TestTimestampConstructorPositionalAndKeywordSupport:
    def test_constructor_positional(self) -> None:
        msg: str = "'NoneType' object cannot be interpreted as an integer"
        with pytest.raises(TypeError, match=msg):
            Timestamp(2000, 1)  # type: ignore[arg-type]
        msg = 'month must be in 1..12'
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 0, 1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 13, 1)  # type: ignore[arg-type]
        msg = 'day is out of range for month'
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 32)  # type: ignore[arg-type]
        assert repr(Timestamp(2015, 11, 12)) == repr(Timestamp('20151112'))
        assert repr(Timestamp(2015, 11, 12, 1, 2, 3, 999999)) == repr(Timestamp('2015-11-12 01:02:03.999999'))

    def test_constructor_keyword(self) -> None:
        msg: str = "function missing required argument 'day'|Required argument 'day'"
        with pytest.raises(TypeError, match=msg):
            Timestamp(year=2000, month=1)  # type: ignore[call-arg]
        msg = 'month must be in 1..12'
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=0, day=1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=13, day=1)
        msg = 'day is out of range for month'
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=32)
        assert repr(Timestamp(year=2015, month=11, day=12)) == repr(Timestamp('20151112'))
        assert repr(Timestamp(year=2015, month=11, day=12, hour=1, minute=2, second=3, microsecond=999999)) == repr(Timestamp('2015-11-12 01:02:03.999999'))

    @pytest.mark.parametrize('arg', ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'nanosecond'])
    def test_invalid_date_kwarg_with_string_input(self, arg: str) -> None:
        kwarg: Dict[str, int] = {arg: 1}
        msg: str = 'Cannot pass a date attribute keyword argument'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2010-10-10 12:59:59.999999999', **kwarg)

    @pytest.mark.parametrize('kwargs', [{}, {'year': 2020}, {'year': 2020, 'month': 1}])
    def test_constructor_missing_keyword(self, kwargs: Dict[str, Any]) -> None:
        msg1: str = "function missing required argument '(year|month|day)' \\(pos [123]\\)"
        msg2: str = "Required argument '(year|month|day)' \\(pos [123]\\) not found"
        msg: str = '|'.join([msg1, msg2])
        with pytest.raises(TypeError, match=msg):
            Timestamp(**kwargs)  # type: ignore[arg-type]

    def test_constructor_positional_with_tzinfo(self) -> None:
        ts: Timestamp = Timestamp(2020, 12, 31, tzinfo=timezone.utc)  # type: ignore[arg-type]
        expected: Timestamp = Timestamp('2020-12-31', tzinfo=timezone.utc)
        assert ts == expected

    @pytest.mark.parametrize('kwd', ['nanosecond', 'microsecond', 'second', 'minute'])
    def test_constructor_positional_keyword_mixed_with_tzinfo(self, kwd: str, request: Any) -> None:
        if kwd != 'nanosecond':
            mark = pytest.mark.xfail(reason='GH#45307')
            request.applymarker(mark)
        kwargs: Dict[str, int] = {kwd: 4}
        ts: Timestamp = Timestamp(2020, 12, 31, tzinfo=timezone.utc, **kwargs)  # type: ignore[arg-type]
        td_kwargs: Dict[str, int] = {kwd + 's': 4}
        td: Timedelta = Timedelta(**td_kwargs)
        expected: Timestamp = Timestamp('2020-12-31', tz=timezone.utc) + td
        assert ts == expected

class TestTimestampClassMethodConstructors:
    def test_utcnow_deprecated(self) -> None:
        msg: str = 'Timestamp.utcnow is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            Timestamp.utcnow()

    def test_utcfromtimestamp_deprecated(self) -> None:
        msg: str = 'Timestamp.utcfromtimestamp is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            Timestamp.utcfromtimestamp(43)

    def test_constructor_strptime(self) -> None:
        fmt: str = '%Y%m%d-%H%M%S-%f%z'
        ts: str = '20190129-235348-000001+0000'
        msg: str = 'Timestamp.strptime\\(\\) is not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            Timestamp.strptime(ts, fmt)

    def test_constructor_fromisocalendar(self) -> None:
        expected_timestamp: Timestamp = Timestamp('2000-01-03 00:00:00')
        expected_stdlib: datetime = datetime.fromisocalendar(2000, 1, 1)
        result: Timestamp = Timestamp.fromisocalendar(2000, 1, 1)
        assert result == expected_timestamp
        assert result == expected_stdlib
        assert isinstance(result, Timestamp)

    def test_constructor_fromordinal(self) -> None:
        base: datetime = datetime(2000, 1, 1)
        ts: Timestamp = Timestamp.fromordinal(base.toordinal())
        assert base == ts
        assert base.toordinal() == ts.toordinal()
        ts = Timestamp.fromordinal(base.toordinal(), tz='US/Eastern')
        assert Timestamp('2000-01-01', tz='US/Eastern') == ts
        assert base.toordinal() == ts.toordinal()
        dt: datetime = datetime(2011, 4, 16, 0, 0)
        ts = Timestamp.fromordinal(dt.toordinal())
        assert ts.to_pydatetime() == dt
        stamp: Timestamp = Timestamp('2011-4-16', tz='US/Eastern')
        dt_tz: datetime = stamp.to_pydatetime()
        ts = Timestamp.fromordinal(dt_tz.toordinal(), tz='US/Eastern')
        assert ts.to_pydatetime() == dt_tz

    def test_now(self) -> None:
        ts_from_string: Timestamp = Timestamp('now')
        ts_from_method: Timestamp = Timestamp.now()
        ts_datetime: datetime = datetime.now()
        ts_from_string_tz: Timestamp = Timestamp('now', tz='US/Eastern')
        ts_from_method_tz: Timestamp = Timestamp.now(tz='US/Eastern')
        delta: Timedelta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert abs(ts_from_string_tz.tz_localize(None) - ts_from_method_tz.tz_localize(None)) < delta

    def test_today(self) -> None:
        ts_from_string: Timestamp = Timestamp('today')
        ts_from_method: Timestamp = Timestamp.today()
        ts_datetime: datetime = datetime.today()
        ts_from_string_tz: Timestamp = Timestamp('today', tz='US/Eastern')
        ts_from_method_tz: Timestamp = Timestamp.today(tz='US/Eastern')
        delta: Timedelta = Timedelta(seconds=1)
        assert abs(ts_from_method - ts_from_string) < delta
        assert abs(ts_datetime - ts_from_method) < delta
        assert abs(ts_from_method_tz - ts_from_string_tz) < delta
        assert abs(ts_from_string_tz.tz_localize(None) - ts_from_method_tz.tz_localize(None)) < delta

class TestTimestampResolutionInference:
    def test_construct_from_time_unit(self) -> None:
        ts: Timestamp = Timestamp('01:01:01.111')
        assert ts.unit == 'ms'

    def test_constructor_str_infer_reso(self) -> None:
        ts: Timestamp = Timestamp('01/30/2023')
        assert ts.unit == 's'
        ts = Timestamp('2015Q1')
        assert ts.unit == 's'
        ts = Timestamp('2016-01-01 1:30:01 PM')
        assert ts.unit == 's'
        ts = Timestamp('2016 June 3 15:25:01.345')
        assert ts.unit == 'ms'
        ts = Timestamp('300-01-01')
        assert ts.unit == 's'
        ts = Timestamp('300 June 1:30:01.300')
        assert ts.unit == 'ms'
        ts = Timestamp('01-01-2013T00:00:00.000000000+0000')
        assert ts.unit == 'ns'
        ts = Timestamp('2016/01/02 03:04:05.001000 UTC')
        assert ts.unit == 'us'
        ts = Timestamp('01-01-2013T00:00:00.000000002100+0000')
        assert ts == Timestamp('01-01-2013T00:00:00.000000002+0000')
        assert ts.unit == 'ns'
        ts = Timestamp('2020-01-01 00:00+00:00')
        assert ts.unit == 's'
        ts = Timestamp('2020-01-01 00+00:00')
        assert ts.unit == 's'

    @pytest.mark.parametrize('method', ['now', 'today'])
    def test_now_today_unit(self, method: str) -> None:
        ts_from_method: Timestamp = getattr(Timestamp, method)()
        ts_from_string: Timestamp = Timestamp(method)
        assert ts_from_method.unit == ts_from_string.unit == 'us'

class TestTimestampConstructors:
    def test_weekday_but_no_day_raises(self) -> None:
        msg: str = 'Parsing datetimes with weekday but no day information is not supported'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2023 Sept Thu')

    def test_construct_from_string_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match='gives an invalid tzoffset'):
            Timestamp('200622-12-31')

    def test_constructor_from_iso8601_str_with_offset_reso(self) -> None:
        ts: Timestamp = Timestamp('2016-01-01 04:05:06-01:00')
        assert ts.unit == 's'
        ts = Timestamp('2016-01-01 04:05:06.000-01:00')
        assert ts.unit == 'ms'
        ts = Timestamp('2016-01-01 04:05:06.000000-01:00')
        assert ts.unit == 'us'
        ts = Timestamp('2016-01-01 04:05:06.000000001-01:00')
        assert ts.unit == 'ns'

    def test_constructor_from_date_second_reso(self) -> None:
        obj: date = date(2012, 9, 1)
        ts: Timestamp = Timestamp(obj)
        assert ts.unit == 's'

    def test_constructor_datetime64_with_tz(self) -> None:
        dt: np.datetime64 = np.datetime64('1970-01-01 05:00:00')
        tzstr: str = 'UTC+05:00'
        ts: Timestamp = Timestamp(dt, tz=tzstr)
        alt: Timestamp = Timestamp(dt).tz_localize(tzstr)
        assert ts == alt
        assert ts.hour == 5

    def test_constructor(self) -> None:
        base_str: str = '2014-07-01 09:00'
        base_dt: datetime = datetime(2014, 7, 1, 9)
        base_expected: int = 1404205200000000000
        assert calendar.timegm(base_dt.timetuple()) * 1000000000 == base_expected
        tests: list[tuple[Any, Any, int]] = [
            (base_str, base_dt, base_expected),
            ('2014-07-01 10:00', datetime(2014, 7, 1, 10), base_expected + 3600 * 1000000000),
            ('2014-07-01 09:00:00.000008000', datetime(2014, 7, 1, 9, 0, 0, 8), base_expected + 8000),
            ('2014-07-01 09:00:00.000000005', Timestamp('2014-07-01 09:00:00.000000005'), base_expected + 5)
        ]
        timezones: list[tuple[Optional[Any], int]] = [
            (None, 0), ('UTC', 0), (timezone.utc, 0), ('Asia/Tokyo', 9),
            ('US/Eastern', -4), ('dateutil/US/Pacific', -7),
            (timezone(timedelta(hours=-3)), -3),
            (dateutil.tz.tzoffset(None, 18000), 5)
        ]
        for date_str, date_obj, expected in tests:
            for result in [Timestamp(date_str), Timestamp(date_obj)]:
                result = result.as_unit('ns')
                assert result.as_unit('ns')._value == expected  # type: ignore[attr-defined]
                result = Timestamp(result)
                assert result.as_unit('ns')._value == expected  # type: ignore[attr-defined]
            for tz, offset in timezones:
                for result in [Timestamp(date_str, tz=tz), Timestamp(date_obj, tz=tz)]:
                    result = result.as_unit('ns')
                    expected_tz: int = expected - offset * 3600 * 1000000000
                    assert result.as_unit('ns')._value == expected_tz  # type: ignore[attr-defined]
                    result = Timestamp(result)
                    assert result.as_unit('ns')._value == expected_tz  # type: ignore[attr-defined]
                    if tz is not None:
                        result = Timestamp(result).tz_convert('UTC')
                    else:
                        result = Timestamp(result, tz='UTC')
                    expected_utc: int = expected - offset * 3600 * 1000000000
                    assert result.as_unit('ns')._value == expected_utc  # type: ignore[attr-defined]

    def test_constructor_with_stringoffset(self) -> None:
        base_str: str = '2014-07-01 11:00:00+02:00'
        base_dt: datetime = datetime(2014, 7, 1, 9)
        base_expected: int = 1404205200000000000
        assert calendar.timegm(base_dt.timetuple()) * 1000000000 == base_expected
        tests: list[tuple[str, int]] = [
            (base_str, base_expected),
            ('2014-07-01 12:00:00+02:00', base_expected + 3600 * 1000000000),
            ('2014-07-01 11:00:00.000008000+02:00', base_expected + 8000),
            ('2014-07-01 11:00:00.000000005+02:00', base_expected + 5)
        ]
        timezones: list[tuple[Optional[Any], int]] = [
            ('UTC', 0), (timezone.utc, 0), ('Asia/Tokyo', 9),
            ('US/Eastern', -4), ('dateutil/US/Pacific', -7),
            (timezone(timedelta(hours=-3)), -3), (dateutil.tz.tzoffset(None, 18000), 5)
        ]
        for date_str, expected in tests:
            for result in [Timestamp(date_str)]:
                assert result.as_unit('ns')._value == expected  # type: ignore[attr-defined]
                result = Timestamp(result)
                assert result.as_unit('ns')._value == expected  # type: ignore[attr-defined]
            for tz, offset in timezones:
                result = Timestamp(date_str, tz=tz)
                expected_tz: int = expected
                assert result.as_unit('ns')._value == expected_tz  # type: ignore[attr-defined]
                result = Timestamp(result)
                assert result.as_unit('ns')._value == expected_tz  # type: ignore[attr-defined]
                result = Timestamp(result).tz_convert('UTC')
                expected_utc: int = expected
                assert result.as_unit('ns')._value == expected_utc  # type: ignore[attr-defined]
        result = Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')
        assert result._value == Timestamp('2013-11-01 05:00')._value  # type: ignore[attr-defined]
        expected_repr: str = "Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')"
        assert repr(result) == expected_repr
        assert result == eval(repr(result))
        result = Timestamp('2013-11-01 00:00:00-0500', tz='Asia/Tokyo')
        assert result._value == Timestamp('2013-11-01 05:00')._value  # type: ignore[attr-defined]
        expected_repr = "Timestamp('2013-11-01 14:00:00+0900', tz='Asia/Tokyo')"
        assert repr(result) == expected_repr
        assert result == eval(repr(result))
        result = Timestamp('2015-11-18 15:45:00+05:45', tz='Asia/Katmandu')
        assert result._value == Timestamp('2015-11-18 10:00')._value  # type: ignore[attr-defined]
        expected_repr = "Timestamp('2015-11-18 15:45:00+0545', tz='Asia/Katmandu')"
        assert repr(result) == expected_repr
        assert result == eval(repr(result))
        result = Timestamp('2015-11-18 15:30:00+05:30', tz='Asia/Kolkata')
        assert result._value == Timestamp('2015-11-18 10:00')._value  # type: ignore[attr-defined]
        expected_repr = "Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')"
        assert repr(result) == expected_repr
        assert result == eval(repr(result))

    def test_constructor_invalid(self) -> None:
        msg: str = 'Cannot convert input'
        with pytest.raises(TypeError, match=msg):
            Timestamp(slice(2))  # type: ignore[arg-type]
        msg = 'Cannot convert Period'
        with pytest.raises(ValueError, match=msg):
            Timestamp(Period('1000-01-01'))

    def test_constructor_invalid_tz(self) -> None:
        msg: str = "Argument 'tzinfo' has incorrect type \\(expected datetime.tzinfo, got str\\)"
        with pytest.raises(TypeError, match=msg):
            Timestamp('2017-10-22', tzinfo='US/Eastern')  # type: ignore[arg-type]
        msg = 'at most one of'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2017-10-22', tzinfo=timezone.utc, tz='UTC')
        msg = 'Cannot pass a date attribute keyword argument when passing a date string'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2012-01-01', 'US/Pacific')  # type: ignore[arg-type]

    def test_constructor_tz_or_tzinfo(self) -> None:
        stamps: list[Timestamp] = [
            Timestamp(year=2017, month=10, day=22, tz='UTC'),
            Timestamp(year=2017, month=10, day=22, tzinfo=timezone.utc),
            Timestamp(year=2017, month=10, day=22, tz=timezone.utc),
            Timestamp(datetime(2017, 10, 22), tzinfo=timezone.utc),
            Timestamp(datetime(2017, 10, 22), tz='UTC'),
            Timestamp(datetime(2017, 10, 22), tz=timezone.utc)
        ]
        assert all((ts == stamps[0] for ts in stamps))

    @pytest.mark.parametrize('result', [
        Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), nanosecond=1),
        Timestamp(year=2000, month=1, day=2, hour=3, minute=4, second=5, microsecond=6, nanosecond=1),
        Timestamp(year=2000, month=1, day=2, hour=3, minute=4, second=5, microsecond=6, nanosecond=1, tz='UTC'),
        Timestamp(2000, 1, 2, 3, 4, 5, 6, None, nanosecond=1),
        Timestamp(2000, 1, 2, 3, 4, 5, 6, tz=timezone.utc, nanosecond=1)
    ])
    def test_constructor_nanosecond(self, result: Timestamp) -> None:
        expected: Timestamp = Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), tz=result.tz)
        expected = expected + Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize('z', ['Z0', 'Z00'])
    def test_constructor_invalid_Z0_isostring(self, z: str) -> None:
        msg: str = f'Unknown datetime string format, unable to parse: 2014-11-02 01:00{z}'
        with pytest.raises(ValueError, match=msg):
            Timestamp(f'2014-11-02 01:00{z}')

    def test_out_of_bounds_integer_value(self) -> None:
        msg: str = str(Timestamp.max._value * 2)  # type: ignore[attr-defined]
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.max._value * 2)  # type: ignore[attr-defined]
        msg = str(Timestamp.min._value * 2)  # type: ignore[attr-defined]
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.min._value * 2)  # type: ignore[attr-defined]

    def test_out_of_bounds_value(self) -> None:
        one_us: np.timedelta64 = np.timedelta64(1).astype('timedelta64[us]')
        min_ts_us: np.datetime64 = np.datetime64(Timestamp.min).astype('M8[us]') + one_us  # type: ignore[attr-defined]
        max_ts_us: np.datetime64 = np.datetime64(Timestamp.max).astype('M8[us]')  # type: ignore[attr-defined]
        Timestamp(min_ts_us)
        Timestamp(max_ts_us)
        us_val: int = NpyDatetimeUnit.NPY_FR_us.value
        assert Timestamp(min_ts_us - one_us)._creso == us_val  # type: ignore[attr-defined]
        assert Timestamp(max_ts_us + one_us)._creso == us_val  # type: ignore[attr-defined]
        too_low: np.datetime64 = np.datetime64('-292277022657-01-27T08:29', 'm')
        too_high: np.datetime64 = np.datetime64('292277026596-12-04T15:31', 'm')
        msg: str = 'Out of bounds'
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_low)
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_high)

    def test_out_of_bounds_string(self) -> None:
        msg: str = "Cannot cast .* to unit='ns' without overflow"
        with pytest.raises(ValueError, match=msg):
            Timestamp('1676-01-01').as_unit('ns')
        with pytest.raises(ValueError, match=msg):
            Timestamp('2263-01-01').as_unit('ns')
        ts: Timestamp = Timestamp('2263-01-01')
        assert ts.unit == 's'
        ts = Timestamp('1676-01-01')
        assert ts.unit == 's'

    def test_barely_out_of_bounds(self) -> None:
        msg: str = 'Out of bounds nanosecond timestamp: 2262-04-11 23:47:16'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp('2262-04-11 23:47:16.854775808')

    @pytest.mark.skip_ubsan
    def test_bounds_with_different_units(self) -> None:
        out_of_bounds_dates: tuple[str, str] = ('1677-09-21', '2262-04-12')
        time_units: tuple[str, ...] = ('D', 'h', 'm', 's', 'ms', 'us')
        for date_string in out_of_bounds_dates:
            for unit in time_units:
                dt64: np.datetime64 = np.datetime64(date_string, unit)
                ts: Timestamp = Timestamp(dt64)
                if unit in ['s', 'ms', 'us']:
                    assert ts._value == dt64.view('i8')  # type: ignore[attr-defined]
                else:
                    assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value  # type: ignore[attr-defined]
        info = np.iinfo(np.int64)
        msg: str = 'Out of bounds second timestamp:'
        for value in [info.min + 1, info.max]:
            for unit in ['D', 'h', 'm']:
                dt64: np.datetime64 = np.datetime64(value, unit)
                with pytest.raises(OutOfBoundsDatetime, match=msg):
                    Timestamp(dt64)
        in_bounds_dates: tuple[str, str] = ('1677-09-23', '2262-04-11')
        for date_string in in_bounds_dates:
            for unit in time_units:
                dt64: np.datetime64 = np.datetime64(date_string, unit)
                Timestamp(dt64)

    @pytest.mark.parametrize('arg', ['001-01-01', '0001-01-01'])
    def test_out_of_bounds_string_consistency(self, arg: str) -> None:
        msg: str = "Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(arg).as_unit('ns')
        ts: Timestamp = Timestamp(arg)
        assert ts.unit == 's'
        assert ts.year == ts.month == ts.day == 1

    def test_min_valid(self) -> None:
        Timestamp(Timestamp.min)  # type: ignore[attr-defined]

    def test_max_valid(self) -> None:
        Timestamp(Timestamp.max)  # type: ignore[attr-defined]

    @pytest.mark.parametrize('offset', ['+0300', '+0200'])
    def test_construct_timestamp_near_dst(self, offset: str) -> None:
        expected: Timestamp = Timestamp(f'2016-10-30 03:00:00{offset}', tz='Europe/Helsinki')
        result: Timestamp = Timestamp(expected).tz_convert('Europe/Helsinki')
        assert result == expected

    @pytest.mark.parametrize('arg', ['2013/01/01 00:00:00+09:00', '2013-01-01 00:00:00+09:00'])
    def test_construct_with_different_string_format(self, arg: str) -> None:
        result: Timestamp = Timestamp(arg)
        expected: Timestamp = Timestamp(datetime(2013, 1, 1), tz=timezone(timedelta(hours=9)))
        assert result == expected

    @pytest.mark.parametrize('box', [datetime, Timestamp])
    def test_raise_tz_and_tzinfo_in_datetime_input(self, box: Callable[..., Any]) -> None:
        kwargs: Dict[str, Any] = {'year': 2018, 'month': 1, 'day': 1, 'tzinfo': timezone.utc}
        msg: str = 'Cannot pass a datetime or Timestamp'
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tz='US/Pacific')
        msg = 'Cannot pass a datetime or Timestamp'
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tzinfo=zoneinfo.ZoneInfo('US/Pacific'))

    def test_dont_convert_dateutil_utc_to_default_utc(self) -> None:
        result: Timestamp = Timestamp(datetime(2018, 1, 1), tz=tzutc())
        expected: Timestamp = Timestamp(datetime(2018, 1, 1)).tz_localize(tzutc())
        assert result == expected

    def test_constructor_subclassed_datetime(self) -> None:
        class SubDatetime(datetime):
            pass
        data: datetime = SubDatetime(2000, 1, 1)
        result: Timestamp = Timestamp(data)
        expected: Timestamp = Timestamp(2000, 1, 1)
        assert result == expected

    def test_timestamp_constructor_tz_utc(self) -> None:
        utc_stamp: Timestamp = Timestamp('3/11/2012 05:00', tz='utc')
        assert utc_stamp.tzinfo is timezone.utc
        assert utc_stamp.hour == 5
        utc_stamp = Timestamp('3/11/2012 05:00').tz_localize('utc')
        assert utc_stamp.hour == 5

    def test_timestamp_to_datetime_tzoffset(self) -> None:
        tzinfo = tzoffset(None, 7200)
        expected: Timestamp = Timestamp('3/11/2012 04:00', tz=tzinfo)
        result: Timestamp = Timestamp(expected.to_pydatetime())
        assert expected == result

    def test_timestamp_constructor_near_dst_boundary(self) -> None:
        for tz in ['Europe/Brussels', 'Europe/Prague']:
            result: Timestamp = Timestamp('2015-10-25 01:00', tz=tz)
            expected: Timestamp = Timestamp('2015-10-25 01:00').tz_localize(tz)
            assert result == expected
            msg: str = 'Cannot infer dst time from 2015-10-25 02:00:00'
            with pytest.raises(ValueError, match=msg):
                Timestamp('2015-10-25 02:00', tz=tz)
        result = Timestamp('2017-03-26 01:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 01:00').tz_localize('Europe/Paris')
        assert result == expected
        msg = '2017-03-26 02:00'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2017-03-26 02:00', tz='Europe/Paris')
        naive: Timestamp = Timestamp('2015-11-18 10:00:00')
        result = naive.tz_localize('UTC').tz_convert('Asia/Kolkata')
        expected = Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')
        assert result == expected
        result = Timestamp('2017-03-26 00:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 00:00:00+0100', tz='Europe/Paris')
        assert result == expected
        result = Timestamp('2017-03-26 01:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 01:00:00+0100', tz='Europe/Paris')
        assert result == expected
        msg = '2017-03-26 02:00'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2017-03-26 02:00', tz='Europe/Paris')
        result = Timestamp('2017-03-26 02:00:00+0100', tz='Europe/Paris')
        naive = Timestamp(result.as_unit('ns')._value)  # type: ignore[attr-defined]
        expected = naive.tz_localize('UTC').tz_convert('Europe/Paris')
        assert result == expected
        result = Timestamp('2017-03-26 03:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 03:00:00+0200', tz='Europe/Paris')
        assert result == expected

    @pytest.mark.parametrize('tz', ['pytz/US/Eastern', gettz('US/Eastern'), 'US/Eastern', 'dateutil/US/Eastern'])
    def test_timestamp_constructed_by_date_and_tz(self, tz: Union[str, Any]) -> None:
        if isinstance(tz, str) and tz.startswith('pytz/'):
            pytz = pytest.importorskip('pytz')
            tz = pytz.timezone(tz.removeprefix('pytz/'))
        result: Timestamp = Timestamp(date(2012, 3, 11), tz=tz)
        expected: Timestamp = Timestamp('3/11/2012', tz=tz)
        assert result.hour == expected.hour
        assert result == expected

    def test_explicit_tz_none(self) -> None:
        msg: str = "Passed data is timezone-aware, incompatible with 'tz=None'"
        with pytest.raises(ValueError, match=msg):
            Timestamp(datetime(2022, 1, 1, tzinfo=timezone.utc), tz=None)
        with pytest.raises(ValueError, match=msg):
            Timestamp('2022-01-01 00:00:00', tzinfo=timezone.utc, tz=None)
        with pytest.raises(ValueError, match=msg):
            Timestamp('2022-01-01 00:00:00-0400', tz=None)

def test_constructor_ambiguous_dst() -> None:
    ts: Timestamp = Timestamp(1382835600000000000, tz='dateutil/Europe/London')
    expected: int = ts._value  # type: ignore[attr-defined]
    result: int = Timestamp(ts)._value  # type: ignore[attr-defined]
    assert result == expected

@pytest.mark.parametrize('epoch', [1552211999999999872, 1552211999999999999])
def test_constructor_before_dst_switch(epoch: int) -> None:
    ts: Timestamp = Timestamp(epoch, tz='dateutil/America/Los_Angeles')
    result: timedelta = ts.tz.dst(ts)
    expected: timedelta = timedelta(seconds=0)
    assert Timestamp(ts)._value == epoch  # type: ignore[attr-defined]
    assert result == expected

def test_timestamp_constructor_identity() -> None:
    expected: Timestamp = Timestamp('2017-01-01T12')
    result: Timestamp = Timestamp(expected)
    assert result is expected

@pytest.mark.parametrize('nano', [-1, 1000])
def test_timestamp_nano_range(nano: int) -> None:
    with pytest.raises(ValueError, match='nanosecond must be in 0..999'):
        Timestamp(year=2022, month=1, day=1, nanosecond=nano)

def test_non_nano_value() -> None:
    result: int = Timestamp('1800-01-01', unit='s').value  # type: ignore[attr-defined]
    assert result == -5364662400000000000
    msg: str = "Cannot convert Timestamp to nanoseconds without overflow. Use `.asm8.view\\('i8'\\)` to cast represent Timestamp in its own unit \\(here, s\\).$"
    ts: Timestamp = Timestamp('0300-01-01')
    with pytest.raises(OverflowError, match=msg):
        ts.value  # type: ignore[attr-defined]
    result = ts.asm8.view('i8')
    assert result == -52700112000

@pytest.mark.parametrize('na_value', [None, np.nan, np.datetime64('NaT'), NaT, NA])
def test_timestamp_constructor_na_value(na_value: Any) -> None:
    result: Any = Timestamp(na_value)
    expected: Any = NaT
    assert result is expected
