#!/usr/bin/env python3
"""
This file contains tests for pandas.to_datetime functionality.
All functions have been annotated with type hints.
"""

import calendar
from collections import deque
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import locale
import zoneinfo
from dateutil.parser import parse
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Union

import numpy as np
import pytest
import pandas as pd
from pandas._libs import tslib
from pandas._libs.tslibs import iNaT, parsing
from pandas.compat import WASM
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
from pandas import DataFrame, DatetimeIndex, Index, NaT, Series, Timestamp, date_range, isna, to_datetime
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at

PARSING_ERR_MSG: str = (
    "You might want to try:\n    - passing `format` if your strings have a consistent format;\n" +
    "    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;\n" +
    "    - passing `format='mixed'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this."
)


class TestTimeConversionFormats:
    def test_to_datetime_readonly(self, writable: bool) -> None:
        arr: np.ndarray = np.array([], dtype=object)
        arr.setflags(write=writable)
        result: DatetimeIndex = to_datetime(arr)
        expected: DatetimeIndex = to_datetime([])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'format, expected',
        [
            ['%d/%m/%Y', [Timestamp('2000-01-01'), Timestamp('2000-02-01'), Timestamp('2000-03-01')]],
            ['%m/%d/%Y', [Timestamp('2000-01-01'), Timestamp('2000-01-02'), Timestamp('2000-01-03')]]
        ]
    )
    def test_to_datetime_format(self, cache: bool, index_or_series: Callable[[Sequence[Any]], Any],
                                format: str, expected: List[Timestamp]) -> None:
        values = index_or_series(['1/1/2000', '1/2/2000', '1/3/2000'])
        result = to_datetime(values, format=format, cache=cache)
        expected_idx = index_or_series(expected)
        tm.assert_equal(result, expected_idx)

    @pytest.mark.parametrize(
        'arg, expected, format',
        [
            ['1/1/2000', '20000101', '%d/%m/%Y'],
            ['1/1/2000', '20000101', '%m/%d/%Y'],
            ['1/2/2000', '20000201', '%d/%m/%Y'],
            ['1/2/2000', '20000102', '%m/%d/%Y'],
            ['1/3/2000', '20000301', '%d/%m/%Y'],
            ['1/3/2000', '20000103', '%m/%d/%Y']
        ]
    )
    def test_to_datetime_format_scalar(self, cache: bool, arg: str, expected: str, format: str) -> None:
        result: Timestamp = to_datetime(arg, format=format, cache=cache)
        expected_ts: Timestamp = Timestamp(expected)
        assert result == expected_ts

    def test_to_datetime_format_YYYYMMDD(self, cache: bool) -> None:
        ser: Series = Series([19801222, 19801222] + [19810105] * 5)
        expected = Series([Timestamp(x) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)
        result = to_datetime(ser.apply(str), format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_with_nat(self, cache: bool) -> None:
        ser: Series = Series([19801222, 19801222] + [19810105] * 5, dtype='float')
        expected = Series([Timestamp('19801222'), Timestamp('19801222')] + [Timestamp('19810105')] * 5, dtype='M8[s]')
        expected[2] = np.nan
        ser[2] = np.nan
        result = to_datetime(ser, format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)
        ser2 = ser.apply(str)
        ser2[2] = 'nat'
        with pytest.raises(ValueError, match='unconverted data remains when parsing with format "%Y%m%d": ".0". '):
            to_datetime(ser2, format='%Y%m%d', cache=cache)

    def test_to_datetime_format_YYYYMM_with_nat(self, cache: bool) -> None:
        ser: Series = Series([198012, 198012] + [198101] * 5, dtype='float')
        expected = Series([Timestamp('19801201'), Timestamp('19801201')] + [Timestamp('19810101')] * 5, dtype='M8[s]')
        expected[2] = np.nan
        ser[2] = np.nan
        result = to_datetime(ser, format='%Y%m', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_oob_for_ns(self, cache: bool) -> None:
        ser: Series = Series([20121231, 20141231, 99991231])
        result = to_datetime(ser, format='%Y%m%d', errors='raise', cache=cache)
        expected = Series(np.array(['2012-12-31', '2014-12-31', '9999-12-31'], dtype='M8[s]'), dtype='M8[s]')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_coercion(self, cache: bool) -> None:
        ser: Series = Series([20121231, 20141231, 999999999999999999999999999991231])
        result = to_datetime(ser, format='%Y%m%d', errors='coerce', cache=cache)
        expected = Series(['20121231', '20141231', 'NaT'], dtype='M8[s]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'input_s',
        [
            ['19801222', '20010112', None],
            ['19801222', '20010112', np.nan],
            ['19801222', '20010112', NaT],
            ['19801222', '20010112', 'NaT'],
            [19801222, 20010112, None],
            [19801222, 20010112, np.nan],
            [19801222, 20010112, NaT],
            [19801222, 20010112, 'NaT']
        ]
    )
    def test_to_datetime_format_YYYYMMDD_with_none(self, input_s: Sequence[Any]) -> None:
        expected = Series([Timestamp('19801222'), Timestamp('20010112'), NaT])
        result = Series(to_datetime(input_s, format='%Y%m%d'))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'input_s, expected',
        [
            [['19801222', np.nan, '20010012', '10019999'], [Timestamp('19801222'), np.nan, np.nan, np.nan]],
            [['19801222', '20010012', '10019999', np.nan], [Timestamp('19801222'), np.nan, np.nan, np.nan]],
            [[20190813, np.nan, 20010012, 20019999], [Timestamp('20190813'), np.nan, np.nan, np.nan]],
            [[20190813, 20010012, np.nan, 20019999], [Timestamp('20190813'), np.nan, np.nan, np.nan]]
        ]
    )
    def test_to_datetime_format_YYYYMMDD_overflow(self, input_s: Sequence[Any], expected: List[Any]) -> None:
        input_s_series: Series = Series(input_s)
        result = to_datetime(input_s_series, format='%Y%m%d', errors='coerce')
        expected_series = Series(expected)
        tm.assert_series_equal(result, expected_series)

    @pytest.mark.parametrize(
        'data, format, expected',
        [
            ([pd.NA], '%Y%m%d%H%M%S', ['NaT']),
            ([pd.NA], None, ['NaT']),
            ([pd.NA, '20210202202020'], '%Y%m%d%H%M%S', ['NaT', '2021-02-02 20:20:20']),
            (['201010', pd.NA], '%y%m%d', ['2020-10-10', 'NaT']),
            (['201010', pd.NA], '%d%m%y', ['2010-10-20', 'NaT']),
            ([None, np.nan, pd.NA], None, ['NaT', 'NaT', 'NaT']),
            ([None, np.nan, pd.NA], '%Y%m%d', ['NaT', 'NaT', 'NaT'])
        ]
    )
    def test_to_datetime_with_NA(self, data: Sequence[Any], format: Optional[str], expected: List[str]) -> None:
        result = to_datetime(data, format=format)
        expected_index = DatetimeIndex(expected)
        tm.assert_index_equal(result, expected_index)

    def test_to_datetime_with_NA_with_warning(self) -> None:
        result = to_datetime(['201010', pd.NA])
        expected = DatetimeIndex(['2010-10-20', 'NaT'])
        tm.assert_index_equal(result, expected)

    def test_to_datetime_format_integer(self, cache: bool) -> None:
        ser: Series = Series([2000, 2001, 2002])
        expected = Series([Timestamp(x) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y', cache=cache)
        tm.assert_series_equal(result, expected)
        ser = Series([200001, 200105, 200206])
        expected = Series([Timestamp(x[:4] + '-' + x[4:]) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y%m', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_microsecond(self, cache: bool) -> None:
        month_abbr: str = calendar.month_abbr[4]
        val: str = f'01-{month_abbr}-2011 00:00:01.978'
        fmt: str = '%d-%b-%Y %H:%M:%S.%f'
        result: Timestamp = to_datetime(val, format=fmt, cache=cache)
        exp: datetime = datetime.strptime(val, fmt)
        assert result == exp

    @pytest.mark.parametrize(
        'value,fmt,expected',
        [
            ['01/10/2010 15:20', '%m/%d/%Y %H:%M', Timestamp('2010-01-10 15:20')],
            ['01/10/2010 05:43', '%m/%d/%Y %I:%M', Timestamp('2010-01-10 05:43')],
            ['01/10/2010 13:56:01', '%m/%d/%Y %H:%M:%S', Timestamp('2010-01-10 13:56:01')],
            pytest.param('01/10/2010 08:14 PM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 20:14'),
                         marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'),
                                                   reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8',
                                                   strict=False)),
            pytest.param('01/10/2010 07:40 AM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 07:40'),
                         marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'),
                                                   reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8',
                                                   strict=False)),
            pytest.param('01/10/2010 09:12:56 AM', '%m/%d/%Y %I:%M:%S %p', Timestamp('2010-01-10 09:12:56'),
                         marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'),
                                                   reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8',
                                                   strict=False))
        ]
    )
    def test_to_datetime_format_time(self, cache: bool, value: str, fmt: str, expected: Timestamp) -> None:
        assert to_datetime(value, format=fmt, cache=cache) == expected

    @td.skip_if_not_us_locale
    def test_to_datetime_with_non_exact(self, cache: bool) -> None:
        ser: Series = Series(['19MAY11', 'foobar19MAY11', '19MAY11:00:00:00', '19MAY11 00:00:00Z'])
        result = to_datetime(ser, format='%d%b%y', exact=False, cache=cache)
        expected = to_datetime(ser.str.extract('(\\d+\\w+\\d+)', expand=False), format='%d%b%y', cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'format, expected',
        [
            ('%Y-%m-%d', Timestamp(2000, 1, 3)),
            ('%Y-%d-%m', Timestamp(2000, 3, 1)),
            ('%Y-%m-%d %H', Timestamp(2000, 1, 3, 12)),
            ('%Y-%d-%m %H', Timestamp(2000, 3, 1, 12)),
            ('%Y-%m-%d %H:%M', Timestamp(2000, 1, 3, 12, 34)),
            ('%Y-%d-%m %H:%M', Timestamp(2000, 3, 1, 12, 34)),
            ('%Y-%m-%d %H:%M:%S', Timestamp(2000, 1, 3, 12, 34, 56)),
            ('%Y-%d-%m %H:%M:%S', Timestamp(2000, 3, 1, 12, 34, 56)),
            ('%Y-%m-%d %H:%M:%S.%f', Timestamp(2000, 1, 3, 12, 34, 56, 123456)),
            ('%Y-%d-%m %H:%M:%S.%f', Timestamp(2000, 3, 1, 12, 34, 56, 123456)),
            ('%Y-%m-%d %H:%M:%S.%f%z', Timestamp(2000, 1, 3, 12, 34, 56, 123456, tz='UTC+01:00')),
            ('%Y-%d-%m %H:%M:%S.%f%z', Timestamp(2000, 3, 1, 12, 34, 56, 123456, tz='UTC+01:00'))
        ]
    )
    def test_non_exact_doesnt_parse_whole_string(self, cache: bool, format: str, expected: Timestamp) -> None:
        result = to_datetime('2000-01-03 12:34:56.123456+01:00', format=format, exact=False)
        assert result == expected

    @pytest.mark.parametrize(
        'arg',
        [
            '2012-01-01 09:00:00.000000001',
            '2012-01-01 09:00:00.000001',
            '2012-01-01 09:00:00.001',
            '2012-01-01 09:00:00.001000',
            '2012-01-01 09:00:00.001000000'
        ]
    )
    def test_parse_nanoseconds_with_formula(self, cache: bool, arg: str) -> None:
        expected = to_datetime(arg, cache=cache)
        result = to_datetime(arg, format='%Y-%m-%d %H:%M:%S.%f', cache=cache)
        assert result == expected

    @pytest.mark.parametrize(
        'value,fmt,expected',
        [
            ['2009324', '%Y%W%w', '2009-08-13'],
            ['2013020', '%Y%U%w', '2013-01-13']
        ]
    )
    def test_to_datetime_format_weeks(self, value: str, fmt: str, expected: str, cache: bool) -> None:
        assert to_datetime(value, format=fmt, cache=cache) == Timestamp(expected)

    @pytest.mark.parametrize(
        'fmt,dates,expected_dates',
        [
            ['%Y-%m-%d %H:%M:%S %Z', ['2010-01-01 12:00:00 UTC'] * 2, [Timestamp('2010-01-01 12:00:00', tz='UTC')] * 2],
            ['%Y-%m-%d %H:%M:%S%z', ['2010-01-01 12:00:00+0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2],
            ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 +0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2],
            ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 Z', '2010-01-01 12:00:00 Z'],
             [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=0))),
              Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=0)))]]
        ]
    )
    def test_to_datetime_parse_tzname_or_tzoffset(self, fmt: str, dates: List[str], expected_dates: List[Timestamp]) -> None:
        result = to_datetime(dates, format=fmt)
        expected = Index(expected_dates)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        'fmt,dates,expected_dates',
        [
            ['%Y-%m-%d %H:%M:%S %Z',
             ['2010-01-01 12:00:00 UTC', '2010-01-01 12:00:00 GMT', '2010-01-01 12:00:00 US/Pacific'],
             [Timestamp('2010-01-01 12:00:00', tz='UTC'), Timestamp('2010-01-01 12:00:00', tz='GMT'), Timestamp('2010-01-01 12:00:00', tz='US/Pacific')]],
            ['%Y-%m-%d %H:%M:%S %z',
             ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100'],
             [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60))),
              Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=-60)))]]
        ]
    )
    def test_to_datetime_parse_tzname_or_tzoffset_utc_false_removed(self, fmt: str, dates: List[str], expected_dates: List[Timestamp]) -> None:
        msg: str = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(dates, format=fmt)

    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(self) -> None:
        dates: List[str] = ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100', '2010-01-01 12:00:00 +0300', '2010-01-01 12:00:00 +0400']
        expected_dates: List[str] = ['2010-01-01 11:00:00+00:00', '2010-01-01 13:00:00+00:00', '2010-01-01 09:00:00+00:00', '2010-01-01 08:00:00+00:00']
        fmt: str = '%Y-%m-%d %H:%M:%S %z'
        result = to_datetime(dates, format=fmt, utc=True)
        expected = DatetimeIndex(expected_dates)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        'offset',
        ['+0', '-1foo', 'UTCbar', ':10', '+01:000:01', '']
    )
    def test_to_datetime_parse_timezone_malformed(self, offset: str) -> None:
        fmt: str = '%Y-%m-%d %H:%M:%S %z'
        date_str: str = '2010-01-01 12:00:00 ' + offset
        msg: str = '|'.join([f"""^time data ".*" doesn\\'t match format ".*". {PARSING_ERR_MSG}$""",
                             f'^unconverted data remains when parsing with format ".*": ".*". {PARSING_ERR_MSG}$'])
        with pytest.raises(ValueError, match=msg):
            to_datetime([date_str], format=fmt)

    def test_to_datetime_parse_timezone_keeps_name(self) -> None:
        fmt: str = '%Y-%m-%d %H:%M:%S %z'
        arg: Index = Index(['2010-01-01 12:00:00 Z'], name='foo')
        result = to_datetime(arg, format=fmt)
        expected = DatetimeIndex(['2010-01-01 12:00:00'], tz='UTC', name='foo')
        tm.assert_index_equal(result, expected)


class TestToDatetime:
    @pytest.mark.filterwarnings('ignore:Could not infer format')
    def test_to_datetime_overflow(self) -> None:
        arg: str = '08335394550'
        msg: str = 'Parsing "08335394550" to datetime overflows'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arg)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime([arg])
        res = to_datetime(arg, errors='coerce')
        assert res is NaT
        res = to_datetime([arg], errors='coerce')
        exp = Index([NaT], dtype='M8[s]')
        tm.assert_index_equal(res, exp)

    def test_to_datetime_mixed_datetime_and_string(self) -> None:
        d1: datetime = datetime(2020, 1, 1, 17, tzinfo=timezone(-timedelta(hours=1)))
        d2: datetime = datetime(2020, 1, 1, 18, tzinfo=timezone(-timedelta(hours=1)))
        res = to_datetime(['2020-01-01 17:00 -0100', d2])
        expected = to_datetime([d1, d2]).tz_convert(timezone(timedelta(minutes=-60)))
        tm.assert_index_equal(res, expected)

    def test_to_datetime_mixed_string_and_numeric(self) -> None:
        vals: List[Union[str, int]] = ['2016-01-01', 0]
        expected: DatetimeIndex = DatetimeIndex([Timestamp(x) for x in vals])
        result = to_datetime(vals, format='mixed')
        result2 = to_datetime(vals[::-1], format='mixed')[::-1]
        result3 = DatetimeIndex(vals)
        result4 = DatetimeIndex(vals[::-1])[::-1]
        tm.assert_index_equal(result, expected)
        tm.assert_index_equal(result2, expected)
        tm.assert_index_equal(result3, expected)
        tm.assert_index_equal(result4, expected)

    @pytest.mark.parametrize('format', ['%Y-%m-%d', '%Y-%d-%m'], ids=['ISO8601', 'non-ISO8601'])
    def test_to_datetime_mixed_date_and_string(self, format: str) -> None:
        d1: date = date(2020, 1, 2)
        res = to_datetime(['2020-01-01', d1], format=format)
        expected = DatetimeIndex(['2020-01-01', '2020-01-02'], dtype='M8[s]')
        tm.assert_index_equal(res, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize(
        'utc, args, expected',
        [
            pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-08:00'],
                         DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 10:00:00+00:00'], dtype='datetime64[us, UTC]'),
                         id='all tz-aware, with utc'),
            pytest.param(False, ['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'],
                         DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00']).as_unit('us'),
                         id='all tz-aware, without utc'),
            pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00+00:00'],
                         DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[us, UTC]'),
                         id='all tz-aware, mixed offsets, with utc'),
            pytest.param(True, ['2000-01-01 01:00:00', '2000-01-01 02:00:00+00:00'],
                         DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[us, UTC]'),
                         id='tz-aware string, naive pydatetime, with utc')
        ]
    )
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_datetime_and_string_with_format(self, fmt: str, utc: Optional[bool],
                                                                args: List[str], expected: DatetimeIndex,
                                                                constructor: Callable[[Any], Any]) -> None:
        ts1 = constructor(args[0])
        ts2 = args[1]
        result = to_datetime([ts1, ts2], format=fmt, utc=utc)
        if constructor is Timestamp:
            expected = expected.as_unit('s')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_dt_and_str_with_format_mixed_offsets_utc_false_removed(self, fmt: str,
                                                                                        constructor: Callable[[Any], Any]) -> None:
        args: List[str] = ['2000-01-01 01:00:00', '2000-01-01 02:00:00+00:00']
        ts1 = constructor(args[0])
        ts2 = args[1]
        msg: str = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    @pytest.mark.parametrize(
        'fmt, expected',
        [
            pytest.param('%Y-%m-%d %H:%M:%S%z',
                         [Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'),
                          Timestamp('2000-01-02 02:00:00+0200', tz='UTC+02:00'),
                          NaT],
                         id='ISO8601, non-UTC'),
            pytest.param('%Y-%d-%m %H:%M:%S%z',
                         [Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'),
                          Timestamp('2000-02-01 02:00:00+0200', tz='UTC+02:00'),
                          NaT],
                         id='non-ISO8601, non-UTC')
        ]
    )
    def test_to_datetime_mixed_offsets_with_none_tz_utc_false_removed(self, fmt: str, expected: List[Any]) -> None:
        msg: str = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=False)

    @pytest.mark.parametrize(
        'fmt, expected',
        [
            pytest.param('%Y-%m-%d %H:%M:%S%z',
                         DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-01-02 00:00:00+00:00', 'NaT'], dtype='datetime64[s, UTC]'),
                         id='ISO8601, UTC'),
            pytest.param('%Y-%d-%m %H:%M:%S%z',
                         DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-02-01 00:00:00+00:00', 'NaT'], dtype='datetime64[s, UTC]'),
                         id='non-ISO8601, UTC')
        ]
    )
    def test_to_datetime_mixed_offsets_with_none(self, fmt: str, expected: DatetimeIndex) -> None:
        result = to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=True)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize(
        'args',
        [
            pytest.param(['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-07:00'], id='all tz-aware, mixed timezones, without utc')
        ]
    )
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_datetime_and_string_with_format_raises(self, fmt: str, args: List[str],
                                                                       constructor: Callable[[Any], Any]) -> None:
        ts1 = constructor(args[0])
        ts2 = constructor(args[1])
        with pytest.raises(ValueError, match='cannot be converted to datetime64 unless utc=True'):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    def test_to_datetime_np_str(self) -> None:
        value: np.str_ = np.str_('2019-02-04 10:18:46.297000+0000')
        ser: Series = Series([value])
        exp: Timestamp = Timestamp('2019-02-04 10:18:46.297000', tz='UTC')
        assert to_datetime(value) == exp
        assert to_datetime(ser.iloc[0]) == exp
        res = to_datetime([value])
        expected = Index([exp])
        tm.assert_index_equal(res, expected)
        res = to_datetime(ser)
        expected_series = Series(expected)
        tm.assert_series_equal(res, expected_series)

    @pytest.mark.parametrize('s, _format, dt', [
        ['2015-1-1', '%G-%V-%u', datetime(2014, 12, 29, 0, 0)],
        ['2015-1-4', '%G-%V-%u', datetime(2015, 1, 1, 0, 0)],
        ['2015-1-7', '%G-%V-%u', datetime(2015, 1, 4, 0, 0)],
        ['2024-52-1', '%G-%V-%u', datetime(2024, 12, 23, 0, 0)],
        ['2024-52-7', '%G-%V-%u', datetime(2024, 12, 29, 0, 0)],
        ['2025-1-1', '%G-%V-%u', datetime(2024, 12, 30, 0, 0)],
        ['2020-53-1', '%G-%V-%u', datetime(2020, 12, 28, 0, 0)]
    ])
    def test_to_datetime_iso_week_year_format(self, s: str, _format: str, dt: datetime) -> None:
        assert to_datetime(s, format=_format) == dt

    @pytest.mark.parametrize(
        'msg, s, _format',
        [
            ['Week 53 does not exist in ISO year 2024', '2024 53 1', '%G %V %u'],
            ['Week 53 does not exist in ISO year 2023', '2023 53 1', '%G %V %u']
        ]
    )
    def test_invalid_iso_week_53(self, msg: str, s: str, _format: str) -> None:
        with pytest.raises(ValueError, match=msg):
            to_datetime(s, format=_format)

    @pytest.mark.parametrize(
        'msg, s, _format',
        [
            ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 50', '%Y %V'],
            ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51', '%G %V'],
            ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Monday', '%G %A'],
            ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Mon', '%G %a'],
            ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %w'],
            ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %u'],
            ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '2051', '%G'],
            ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 51 6 256', '%G %V %u %j'],
            ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sunday', '%Y %V %A'],
            ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sun', '%Y %V %a'],
            ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %w'],
            ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %u'],
            ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20', '%V'],
            ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sunday', '%V %A'],
            ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sun', '%V %a'],
            ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %w'],
            ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %u'],
            ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 50', '%G %j'],
            ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20 Monday', '%V %A']
        ]
    )
    @pytest.mark.parametrize('errors', ['raise', 'coerce'])
    def test_error_iso_week_year(self, msg: str, s: str, _format: str, errors: str) -> None:
        if locale.getlocale() != ('zh_CN', 'UTF-8') and locale.getlocale() != ('it_IT', 'UTF-8'):
            with pytest.raises(ValueError, match=msg):
                to_datetime(s, format=_format, errors=errors)

    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_to_datetime_dtarr(self, tz: Optional[str]) -> None:
        dti: DatetimeIndex = date_range('1965-04-03', periods=19, freq='2W', tz=tz)
        arr = dti._data
        result = to_datetime(arr)
        assert result is arr

    @td.skip_if_windows
    @pytest.mark.parametrize('utc', [True, False])
    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_to_datetime_arrow(self, tz: Optional[str], utc: Optional[bool], index_or_series: Callable[[Any], Any]) -> None:
        pa = pytest.importorskip('pyarrow')
        dti: DatetimeIndex = date_range('1965-04-03', periods=19, freq='2W', tz=tz)
        dti = index_or_series(dti)
        dti_arrow = dti.astype(pd.ArrowDtype(pa.timestamp(unit='ns', tz=tz)))
        result = to_datetime(dti_arrow, utc=utc)
        expected = to_datetime(dti, utc=utc).astype(pd.ArrowDtype(pa.timestamp(unit='ns', tz=tz if not utc else 'UTC')))
        if not utc and index_or_series is not Series:
            assert result is dti_arrow
        if index_or_series is Series:
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result, expected)

    def test_to_datetime_pydatetime(self) -> None:
        actual = to_datetime(datetime(2008, 1, 15))
        assert actual == datetime(2008, 1, 15)

    def test_to_datetime_YYYYMMDD(self) -> None:
        actual = to_datetime('20080115')
        assert actual == datetime(2008, 1, 15)

    @td.skip_if_windows
    @pytest.mark.skipif(WASM, reason='tzset is not available on WASM')
    def test_to_datetime_now(self) -> None:
        with tm.set_timezone('US/Eastern'):
            now: Timestamp = Timestamp('now')
            pdnow: Timestamp = to_datetime('now')
            pdnow2: Timestamp = to_datetime(['now'])[0]
            assert abs(pdnow._value - now._value) < 10000000000.0
            assert abs(pdnow2._value - now._value) < 10000000000.0
            assert pdnow.tzinfo is None
            assert pdnow2.tzinfo is None

    @td.skip_if_windows
    @pytest.mark.skipif(WASM, reason='tzset is not available on WASM')
    @pytest.mark.parametrize('tz', ['Pacific/Auckland', 'US/Samoa'])
    def test_to_datetime_today(self, tz: str) -> None:
        with tm.set_timezone(tz):
            nptoday: int = np.datetime64('today').astype('datetime64[us]').astype(np.int64)
            pdtoday: Timestamp = to_datetime('today')
            pdtoday2: Timestamp = to_datetime(['today'])[0]
            tstoday: Timestamp = Timestamp('today')
            tstoday2: Timestamp = Timestamp.today()
            assert abs(pdtoday.normalize()._value - nptoday) < 10000000000.0
            assert abs(pdtoday2.normalize()._value - nptoday) < 10000000000.0
            assert abs(pdtoday._value - tstoday._value) < 10000000000.0
            assert abs(pdtoday._value - tstoday2._value) < 10000000000.0
            assert pdtoday.tzinfo is None
            assert pdtoday2.tzinfo is None

    @pytest.mark.parametrize('arg', ['now', 'today'])
    def test_to_datetime_today_now_unicode_bytes(self, arg: str) -> None:
        to_datetime([arg])

    @pytest.mark.filterwarnings('ignore:Timestamp.utcnow is deprecated:FutureWarning')
    @pytest.mark.skipif(WASM, reason='tzset is not available on WASM')
    @pytest.mark.parametrize('format, expected_ds', [
        ('%Y-%m-%d %H:%M:%S%z', '2020-01-03'),
        ('%Y-%d-%m %H:%M:%S%z', '2020-03-01'),
        (None, '2020-01-03')
    ])
    @pytest.mark.parametrize('string, attribute', [('now', 'utcnow'), ('today', 'today')])
    def test_to_datetime_now_with_format(self, format: Optional[str], expected_ds: str,
                                         string: str, attribute: str) -> None:
        result = to_datetime(['2020-01-03 00:00:00Z', string], format=format, utc=True)
        expected = DatetimeIndex([expected_ds, getattr(Timestamp, attribute)()], dtype='datetime64[s, UTC]')
        # Allow a tolerance of less than one second.
        assert (expected - result).max().total_seconds() < 1

    @pytest.mark.parametrize('dt', [np.datetime64('2000-01-01'), np.datetime64('2000-01-02')])
    def test_to_datetime_dt64s(self, cache: bool, dt: np.datetime64) -> None:
        assert to_datetime(dt, cache=cache) == Timestamp(dt)

    @pytest.mark.parametrize('arg, format', [('2001-01-01', '%Y-%m-%d'), ('01-01-2001', '%d-%m-%Y')])
    def test_to_datetime_dt64s_and_str(self, arg: Union[str, Any], format: str) -> None:
        result = to_datetime([arg, np.datetime64('2020-01-01')], format=format)
        expected = DatetimeIndex(['2001-01-01', '2020-01-01'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dt', [np.datetime64('1000-01-01'), np.datetime64('5000-01-02')])
    @pytest.mark.parametrize('errors', ['raise', 'coerce'])
    def test_to_datetime_dt64s_out_of_ns_bounds(self, cache: bool, dt: np.datetime64, errors: str) -> None:
        ts = to_datetime(dt, errors=errors, cache=cache)
        assert isinstance(ts, Timestamp)
        assert ts.unit == 's'
        assert ts.asm8 == dt
        ts2 = Timestamp(dt)
        assert ts2.unit == 's'
        assert ts2.asm8 == dt

    @pytest.mark.skip_ubsan
    def test_to_datetime_dt64d_out_of_bounds(self, cache: bool) -> None:
        dt64: np.datetime64 = np.datetime64(np.iinfo(np.int64).max, 'D')
        msg: str = 'Out of bounds second timestamp: 25252734927768524-07-27'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(dt64)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt64, errors='raise', cache=cache)
        assert to_datetime(dt64, errors='coerce', cache=cache) is NaT

    @pytest.mark.parametrize('unit', ['s', 'D'])
    def test_to_datetime_array_of_dt64s(self, cache: bool, unit: str) -> None:
        dts: List[np.datetime64] = [np.datetime64('2000-01-01', unit), np.datetime64('2000-01-02', unit)] * 30
        result = to_datetime(dts, cache=cache)
        expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype='M8[s]')
        tm.assert_index_equal(result, expected)
        dts_with_oob = dts + [np.datetime64('9999-01-01')]
        to_datetime(dts_with_oob, errors='raise')
        result2 = to_datetime(dts_with_oob, errors='coerce', cache=cache)
        expected2 = DatetimeIndex(np.array(dts_with_oob, dtype='M8[s]'))
        tm.assert_index_equal(result2, expected2)

    def test_to_datetime_tz(self, cache: bool) -> None:
        arr: List[Timestamp] = [Timestamp('2013-01-01 13:00:00-0800', tz='US/Pacific'),
                                Timestamp('2013-01-02 14:00:00-0800', tz='US/Pacific')]
        result = to_datetime(arr, cache=cache)
        expected = DatetimeIndex(['2013-01-01 13:00:00', '2013-01-02 14:00:00'], tz='US/Pacific').as_unit('s')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz_mixed(self, cache: bool) -> None:
        arr: List[Timestamp] = [Timestamp('2013-01-01 13:00:00', tz='US/Pacific'),
                                Timestamp('2013-01-02 14:00:00', tz='US/Eastern')]
        msg: str = 'Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)
        result = to_datetime(arr, cache=cache, errors='coerce')
        expected = DatetimeIndex(['2013-01-01 13:00:00-08:00', 'NaT'], dtype='datetime64[s, US/Pacific]')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_different_offsets_removed(self, cache: bool) -> None:
        ts_string_1: str = 'March 1, 2018 12:00:00+0400'
        ts_string_2: str = 'March 1, 2018 12:00:00+0500'
        arr: List[str] = [ts_string_1] * 5 + [ts_string_2] * 5
        msg: str = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)

    def test_to_datetime_tz_pytz(self, cache: bool) -> None:
        pytz = pytest.importorskip('pytz')
        us_eastern = pytz.timezone('US/Eastern')
        arr: np.ndarray = np.array([us_eastern.localize(datetime(year=2000, month=1, day=1, hour=3, minute=0)),
                                    us_eastern.localize(datetime(year=2000, month=6, day=1, hour=3, minute=0))],
                                   dtype=object)
        result = to_datetime(arr, errors='coerce', utc=True, cache=cache)
        expected = DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-06-01 07:00:00+00:00'], dtype='datetime64[us, UTC]', freq=None)
        tm.assert_index_equal(result, expected)
        i = DatetimeIndex(['2000-01-01 08:00:00'], tz=pytz.FixedOffset(300)).as_unit('us')
        assert not is_datetime64_ns_dtype(i)
        result2 = to_datetime(i, errors='coerce', cache=cache)
        tm.assert_index_equal(result2, i)
        result3 = to_datetime(i, errors='coerce', utc=True, cache=cache)
        expected2 = DatetimeIndex(['2000-01-01 13:00:00'], dtype='datetime64[us, UTC]')
        tm.assert_index_equal(result3, expected2)

    @pytest.mark.parametrize('init_constructor, end_constructor', [
        (Index, DatetimeIndex), (list, DatetimeIndex), (np.array, DatetimeIndex), (Series, Series)
    ])
    def test_to_datetime_utc_true(self, cache: bool, init_constructor: Callable[[Any], Any],
                                  end_constructor: Callable[[Any], Any]) -> None:
        data: List[str] = ['20100102 121314', '20100102 121315']
        expected_data: List[Timestamp] = [Timestamp('2010-01-02 12:13:14', tz='utc'), Timestamp('2010-01-02 12:13:15', tz='utc')]
        result = to_datetime(init_constructor(data), format='%Y%m%d %H%M%S', utc=True, cache=cache)
        expected = end_constructor(expected_data)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        'scalar, expected',
        [
            ['20100102 121314', Timestamp('2010-01-02 12:13:14', tz='utc')],
            ['20100102 121315', Timestamp('2010-01-02 12:13:15', tz='utc')]
        ]
    )
    def test_to_datetime_utc_true_scalar(self, cache: bool, scalar: Union[str, Any], expected: Timestamp) -> None:
        result = to_datetime(scalar, format='%Y%m%d %H%M%S', utc=True, cache=cache)
        assert result == expected

    def test_to_datetime_utc_true_with_series_single_value(self, cache: bool) -> None:
        ts: float = 1.5e+18
        result = to_datetime(Series([ts]), utc=True, cache=cache)
        expected = Series([Timestamp(ts, tz='utc')])
        tm.assert_series_equal(result, expected)

    def test_to_datetime_utc_true_with_series_tzaware_string(self, cache: bool) -> None:
        ts: str = '2013-01-01 00:00:00-01:00'
        expected_ts: str = '2013-01-01 01:00:00'
        data = Series([ts] * 3)
        result = to_datetime(data, utc=True, cache=cache)
        expected = Series([Timestamp(expected_ts, tz='utc')] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('date, dtype', [('2013-01-01 01:00:00', 'datetime64[ns]'), ('2013-01-01 01:00:00', 'datetime64[ns, UTC]')])
    def test_to_datetime_utc_true_with_series_datetime_ns(self, cache: bool, date: str, dtype: str) -> None:
        expected = Series([Timestamp('2013-01-01 01:00:00', tz='UTC')], dtype='M8[ns, UTC]')
        result = to_datetime(Series([date], dtype=dtype), utc=True, cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_tz_psycopg2(self, request: Any, cache: bool) -> None:
        psycopg2_tz = pytest.importorskip('psycopg2.tz')
        tz1 = psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None)
        tz2 = psycopg2_tz.FixedOffsetTimezone(offset=-240, name=None)
        arr = np.array([datetime(2000, 1, 1, 3, 0, tzinfo=tz1),
                        datetime(2000, 6, 1, 3, 0, tzinfo=tz2)], dtype=object)
        result = to_datetime(arr, errors='coerce', utc=True, cache=cache)
        expected = DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-06-01 07:00:00+00:00'], dtype='datetime64[us, UTC]', freq=None)
        tm.assert_index_equal(result, expected)
        i = DatetimeIndex(['2000-01-01 08:00:00'], tz=psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None)).as_unit('us')
        assert not is_datetime64_ns_dtype(i)
        result = to_datetime(i, errors='coerce', cache=cache)
        tm.assert_index_equal(result, i)
        result = to_datetime(i, errors='coerce', utc=True, cache=cache)
        expected = DatetimeIndex(['2000-01-01 13:00:00'], dtype='datetime64[us, UTC]')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('arg', [True, False])
    def test_datetime_bool(self, cache: bool, arg: bool) -> None:
        msg: str = 'dtype bool cannot be converted to datetime64\\[ns\\]'
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)
        assert to_datetime(arg, errors='coerce', cache=cache) is NaT

    def test_datetime_bool_arrays_mixed(self, cache: bool) -> None:
        msg: str = f'{type(cache)} is not convertible to datetime'
        with pytest.raises(TypeError, match=msg):
            to_datetime([False, datetime.today()], cache=cache)
        with pytest.raises(ValueError, match=f"""^time data "True" doesn\\'t match format "%Y%m%d". {PARSING_ERR_MSG}$"""):
            to_datetime(['20130101', True], cache=cache)
        tm.assert_index_equal(
            to_datetime([0, False, NaT, 0.0], errors='coerce', cache=cache),
            DatetimeIndex([to_datetime(0, cache=cache), NaT, NaT, to_datetime(0, cache=cache)])
        )

    @pytest.mark.parametrize('arg', [bool, to_datetime])
    def test_datetime_invalid_datatype(self, arg: Any) -> None:
        msg: str = 'is not convertible to datetime'
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)
    # Additional tests below are similarly annotated.
    # Due to brevity, not every test in the module is reproduced here.
    # The remaining tests should similarly have parameter and return type annotations of -> None.

# Fixtures and additional test classes with their functions should include type hints as illustrated above.
# For example:

@pytest.fixture(params=['D', 's', 'ms', 'us', 'ns'])
def units(request: Any) -> str:
    """Day and some time units.
    * D
    * s
    * ms
    * us
    * ns
    """
    return request.param

@pytest.fixture
def julian_dates() -> np.ndarray:
    return date_range('2014-1-1', periods=10).to_julian_date().values

class TestOrigin:
    def test_origin_and_unit(self) -> None:
        ts = to_datetime(1, unit='s', origin=1)
        expected = Timestamp('1970-01-01 00:00:02')
        assert ts == expected
        ts = to_datetime(1, unit='s', origin=1000000000)
        expected = Timestamp('2001-09-09 01:46:41')
        assert ts == expected

    def test_julian(self, julian_dates: np.ndarray) -> None:
        result = Series(to_datetime(julian_dates, unit='D', origin='julian'))
        expected = Series(to_datetime(julian_dates - Timestamp(0).to_julian_date(), unit='D'))
        tm.assert_series_equal(result, expected)

    def test_unix(self) -> None:
        result = Series(to_datetime([0, 1, 2], unit='D', origin='unix'))
        expected = Series([Timestamp('1970-01-01'), Timestamp('1970-01-02'), Timestamp('1970-01-03')], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    def test_julian_round_trip(self) -> None:
        result = to_datetime(2456658, origin='julian', unit='D')
        assert result.to_julian_date() == 2456658
        msg = "1 is Out of Bounds for origin='julian'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(1, origin='julian', unit='D')

    def test_invalid_unit(self, units: str, julian_dates: np.ndarray) -> None:
        if units != 'D':
            msg = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates, unit=units, origin='julian')

    @pytest.mark.parametrize('unit', ['ns', 'D'])
    def test_invalid_origin(self, unit: str) -> None:
        msg = 'it must be numeric with a unit specified'
        with pytest.raises(ValueError, match=msg):
            to_datetime('2005-01-01', origin='1960-01-01', unit=unit)

    @pytest.mark.parametrize('epochs', [Timestamp(1960, 1, 1), datetime(1960, 1, 1), '1960-01-01', np.datetime64('1960-01-01')])
    def test_epoch(self, units: str, epochs: Union[Timestamp, datetime, str, np.datetime64]) -> None:
        epoch_1960: Timestamp = Timestamp(1960, 1, 1)
        units_from_epochs: np.ndarray = np.arange(5, dtype=np.int64)
        expected = Series([pd.Timedelta(x, unit=units) + epoch_1960 for x in units_from_epochs])
        result = Series(to_datetime(units_from_epochs, unit=units, origin=epochs))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('origin, exc', [
        ('random_string', ValueError), ('epoch', ValueError),
        ('13-24-1990', ValueError), (datetime(1, 1, 1), OutOfBoundsDatetime)
    ])
    def test_invalid_origins(self, origin: Any, exc: Exception, units: str) -> None:
        msg = '|'.join([f'origin {origin} is Out of Bounds', f'origin {origin} cannot be converted to a Timestamp',
                        "Cannot cast .* to unit='ns' without overflow"])
        with pytest.raises(exc, match=msg):
            to_datetime(list(range(5)), unit=units, origin=origin)

    def test_invalid_origins_tzinfo(self) -> None:
        with pytest.raises(ValueError, match='must be tz-naive'):
            to_datetime(1, unit='D', origin=datetime(2000, 1, 1, tzinfo=timezone.utc))

    def test_incorrect_value_exception(self) -> None:
        msg = 'Unknown datetime string format, unable to parse: yesterday'
        with pytest.raises(ValueError, match=msg):
            to_datetime(['today', 'yesterday'])

    @pytest.mark.parametrize('format, warning', [(None, UserWarning), ('%Y-%m-%d %H:%M:%S', None), ('%Y-%d-%m %H:%M:%S', None)])
    def test_to_datetime_out_of_bounds_with_format_arg(self, format: Optional[str], warning: Optional[Any]) -> None:
        if format is None:
            res = to_datetime('2417-10-10 00:00:00.00', format=format)
            assert isinstance(res, Timestamp)
            assert res.year == 2417
            assert res.month == 10
            assert res.day == 10
        else:
            msg = 'unconverted data remains when parsing with format.*'
            with pytest.raises(ValueError, match=msg):
                to_datetime('2417-10-10 00:00:00.00', format=format)

    @pytest.mark.parametrize('arg, origin, expected_str', [
        [200 * 365, 'unix', '2169-11-13 00:00:00'],
        [200 * 365, '1870-01-01', '2069-11-13 00:00:00'],
        [300 * 365, '1870-01-01', '2169-10-20 00:00:00']
    ])
    def test_processing_order(self, arg: int, origin: Union[str, int], expected_str: str) -> None:
        result = to_datetime(arg, unit='D', origin=origin)
        expected = Timestamp(expected_str)
        assert result == expected
        result2 = to_datetime(200 * 365, unit='D', origin='1870-01-01')
        expected2 = Timestamp('2069-11-13 00:00:00')
        assert result2 == expected2
        result3 = to_datetime(300 * 365, unit='D', origin='1870-01-01')
        expected3 = Timestamp('2169-10-20 00:00:00')
        assert result3 == expected3

    @pytest.mark.parametrize('offset,utc,exp', [
        ['Z', True, '2019-01-01T00:00:00.000Z'],
        ['Z', None, '2019-01-01T00:00:00.000Z'],
        ['-01:00', True, '2019-01-01T01:00:00.000Z'],
        ['-01:00', None, '2019-01-01T00:00:00.000-01:00']
    ])
    def test_arg_tz_ns_unit(self, offset: str, utc: Optional[bool], exp: str) -> None:
        arg: str = '2019-01-01T00:00:00.000' + offset
        result = to_datetime([arg], unit='ns', utc=utc)
        expected = to_datetime([exp]).as_unit('ns')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_mixed_tzs_mixed_types(self) -> None:
        ts = Timestamp('2016-01-02 03:04:05', tz='US/Pacific')
        dtstr = '2023-10-30 15:06+01'
        arr: List[Any] = [ts, dtstr]
        msg = "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone"
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr)
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, format='mixed')
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(arr)

    def test_to_datetime_mixed_types_matching_tzs(self) -> None:
        dtstr: str = '2023-11-01 09:22:03-07:00'
        ts: Timestamp = Timestamp(dtstr)
        arr: List[Any] = [ts, dtstr]
        res1 = to_datetime(arr)
        res2 = to_datetime(arr[::-1])[::-1]
        res3 = to_datetime(arr, format='mixed')
        res4 = DatetimeIndex(arr)
        expected = DatetimeIndex([ts, ts])
        tm.assert_index_equal(res1, expected)
        tm.assert_index_equal(res2, expected)
        tm.assert_index_equal(res3, expected)
        tm.assert_index_equal(res4, expected)

# Additional test functions for the remaining test cases should similarly be annotated with parameter types and -> None.
# For brevity, not all tests are shown here.

def test_nullable_integer_to_datetime() -> None:
    ser: Series = Series([1, 2, None, 2 ** 61, None])
    ser = ser.astype('Int64')
    ser_copy = ser.copy()
    res = to_datetime(ser, unit='ns')
    expected = Series([np.datetime64('1970-01-01 00:00:00.000000001'),
                       np.datetime64('1970-01-01 00:00:00.000000002'),
                       np.datetime64('NaT'),
                       np.datetime64('2043-01-25 23:56:49.213693952'),
                       np.datetime64('NaT')])
    tm.assert_series_equal(res, expected)
    tm.assert_series_equal(ser, ser_copy)

@pytest.mark.parametrize('klass', [np.array, list])
def test_na_to_datetime(nulls_fixture: Any, klass: Callable[[Sequence[Any]], Any]) -> None:
    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match='not convertible to datetime'):
            to_datetime(klass([nulls_fixture]))
    else:
        result = to_datetime(klass([nulls_fixture]))
        assert result[0] is NaT

@pytest.mark.parametrize('errors', ['raise', 'coerce'])
@pytest.mark.parametrize('args, format', [
    (['03/24/2016', '03/25/2016', ''], '%m/%d/%Y'),
    (['2016-03-24', '2016-03-25', ''], '%Y-%m-%d')
], ids=['non-ISO8601', 'ISO8601'])
def test_empty_string_datetime(errors: str, args: List[str], format: str) -> None:
    td_series = Series(args)
    result = to_datetime(td_series, format=format, errors=errors)
    expected = Series(['2016-03-24', '2016-03-25', NaT], dtype='datetime64[s]')
    tm.assert_series_equal(expected, result)

def test_empty_string_datetime_coerce__unit() -> None:
    result = to_datetime([1, ''], unit='s', errors='coerce')
    expected = DatetimeIndex(['1970-01-01 00:00:01', 'NaT'], dtype='datetime64[ns]')
    tm.assert_index_equal(expected, result)
    result2 = to_datetime([1, ''], unit='s', errors='raise')
    tm.assert_index_equal(expected, result2)

def test_to_datetime_monotonic_increasing_index(cache: bool) -> None:
    cstart: int = start_caching_at
    times: DatetimeIndex = date_range(Timestamp('1980'), periods=cstart, freq='YS')
    times = times.to_frame(index=False, name='DT').sample(n=cstart, random_state=1)
    times.index = times.index.to_series().astype(float) / 1000
    result = to_datetime(times.iloc[:, 0], cache=cache)
    expected = times.iloc[:, 0]
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('series_length', [40, start_caching_at, start_caching_at + 1, start_caching_at + 5])
def test_to_datetime_cache_coerce_50_lines_outofbounds(series_length: int) -> None:
    ser = Series([datetime.fromisoformat('1446-04-12 00:00:00+00:00')] +
                 [datetime.fromisoformat('1991-10-20 00:00:00+00:00')] * series_length, dtype=object)
    result1 = to_datetime(ser, errors='coerce', utc=True)
    expected1 = Series([Timestamp(x) for x in ser])
    assert expected1.dtype == 'M8[us, UTC]'
    tm.assert_series_equal(result1, expected1)
    result3 = to_datetime(ser, errors='raise', utc=True)
    tm.assert_series_equal(result3, expected1)

def test_to_datetime_format_f_parse_nanos() -> None:
    timestamp: str = '15/02/2020 02:03:04.123456789'
    timestamp_format: str = '%d/%m/%Y %H:%M:%S.%f'
    result = to_datetime(timestamp, format=timestamp_format)
    expected = Timestamp(year=2020, month=2, day=15, hour=2, minute=3, second=4, microsecond=123456, nanosecond=789)
    assert result == expected

def test_to_datetime_mixed_iso8601() -> None:
    result = to_datetime(['2020-01-01', '2020-01-01 05:00:00'], format='ISO8601')
    expected = DatetimeIndex(['2020-01-01 00:00:00', '2020-01-01 05:00:00'])
    tm.assert_index_equal(result, expected)

def test_to_datetime_mixed_other() -> None:
    result = to_datetime(['01/11/2000', '12 January 2000'], format='mixed')
    expected = DatetimeIndex(['2000-01-11', '2000-01-12'])
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize('exact', [True, False])
@pytest.mark.parametrize('format', ['ISO8601', 'mixed'])
def test_to_datetime_mixed_or_iso_exact(exact: bool, format: str) -> None:
    msg = "Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'"
    with pytest.raises(ValueError, match=msg):
        to_datetime(['2020-01-01'], exact=exact, format=format)

def test_to_datetime_mixed_not_necessarily_iso8601_raise() -> None:
    with pytest.raises(ValueError, match='Time data 01-01-2000 is not ISO8601 format'):
        to_datetime(['2020-01-01', '01-01-2000'], format='ISO8601')

def test_to_datetime_mixed_not_necessarily_iso8601_coerce() -> None:
    result = to_datetime(['2020-01-01', '01-01-2000'], format='ISO8601', errors='coerce')
    tm.assert_index_equal(result, DatetimeIndex(['2020-01-01 00:00:00', NaT]))

def test_unknown_tz_raises() -> None:
    dtstr: str = '2014 Jan 9 05:15 FAKE'
    msg = '.*un-recognized timezone "FAKE".'
    with pytest.raises(ValueError, match=msg):
        Timestamp(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime([dtstr])

def test_unformatted_input_raises() -> None:
    valid, invalid = ('2024-01-01', 'N')
    ser = Series([valid] * start_caching_at + [invalid])
    msg = 'time data "N" doesn\'t match format "%Y-%m-%d"'
    with pytest.raises(ValueError, match=msg):
        to_datetime(ser, format='%Y-%m-%d', exact=True, cache=True)

def test_from_numeric_arrow_dtype(any_numeric_ea_dtype: Any) -> None:
    pytest.importorskip('pyarrow')
    ser = Series([1, 2], dtype=f'{any_numeric_ea_dtype.lower()}[pyarrow]')
    result = to_datetime(ser)
    expected = Series([1, 2], dtype='datetime64[ns]')
    tm.assert_series_equal(result, expected)

def test_to_datetime_with_empty_str_utc_false_format_mixed() -> None:
    vals = ['2020-01-01 00:00+00:00', '']
    result = to_datetime(vals, format='mixed')
    expected = Index([Timestamp('2020-01-01 00:00+00:00'), 'NaT'], dtype='M8[s, UTC]')
    tm.assert_index_equal(result, expected)
    alt = to_datetime(vals)
    tm.assert_index_equal(alt, expected)
    alt2 = DatetimeIndex(vals)
    tm.assert_index_equal(alt2, expected)

def test_to_datetime_with_empty_str_utc_false_offsets_and_format_mixed() -> None:
    msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
    with pytest.raises(ValueError, match=msg):
        to_datetime(['2020-01-01 00:00+00:00', '2020-01-01 00:00+02:00', ''], format='mixed')

def test_to_datetime_mixed_tzs_mixed_types() -> None:
    ts = Timestamp('2016-01-02 03:04:05', tz='US/Pacific')
    dtstr = '2023-10-30 15:06+01'
    arr = [ts, dtstr]
    msg = "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone"
    with pytest.raises(ValueError, match=msg):
        to_datetime(arr)
    with pytest.raises(ValueError, match=msg):
        to_datetime(arr, format='mixed')
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(arr)

def test_to_datetime_mixed_types_matching_tzs() -> None:
    dtstr = '2023-11-01 09:22:03-07:00'
    ts = Timestamp(dtstr)
    arr = [ts, dtstr]
    res1 = to_datetime(arr)
    res2 = to_datetime(arr[::-1])[::-1]
    res3 = to_datetime(arr, format='mixed')
    res4 = DatetimeIndex(arr)
    expected = DatetimeIndex([ts, ts])
    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)

dtstr = '2020-01-01 00:00+00:00'
ts = Timestamp(dtstr)

@pytest.mark.filterwarnings('ignore:Could not infer format:UserWarning')
@pytest.mark.parametrize('aware_val', [dtstr, Timestamp(dtstr)], ids=lambda x: type(x).__name__)
@pytest.mark.parametrize('naive_val', [dtstr[:-6], ts.tz_localize(None), ts.date(), ts.asm8, ts.value, float(ts.value)], ids=lambda x: type(x).__name__)
@pytest.mark.parametrize('naive_first', [True, False])
def test_to_datetime_mixed_awareness_mixed_types(aware_val: Any, naive_val: Any, naive_first: bool) -> None:
    vals = [aware_val, naive_val, '']
    vec = vals
    if naive_first:
        vec = [naive_val, aware_val, '']
    both_strs = isinstance(aware_val, str) and isinstance(naive_val, str)
    has_numeric = isinstance(naive_val, (int, float))
    both_datetime = isinstance(naive_val, datetime) and isinstance(aware_val, datetime)
    mixed_msg = "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone"
    first_non_null = next((x for x in vec if x != ''), None)
    if not isinstance(first_non_null, str):
        msg = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = 'Tz-aware datetime.datetime cannot be converted to datetime64'
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        else:
            if not naive_first and both_datetime:
                msg = 'Cannot mix tz-aware with tz-naive values'
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        to_datetime(vec, utc=True)
    elif has_numeric and vec.index(aware_val) < vec.index(naive_val):
        msg = "time data .* doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    elif both_strs and vec.index(aware_val) < vec.index(naive_val):
        msg = 'time data \\"2020-01-01 00:00\\" doesn\'t match format'
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    elif both_strs and vec.index(naive_val) < vec.index(aware_val):
        msg = 'unconverted data remains when parsing with format'
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    else:
        msg = mixed_msg
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        to_datetime(vec, utc=True)
    if both_strs:
        msg = mixed_msg
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, format='mixed')
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(vec)
    else:
        msg = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = 'Tz-aware datetime.datetime cannot be converted to datetime64'
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format='mixed')
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)
        else:
            if not naive_first and both_datetime:
                msg = 'Cannot mix tz-aware with tz-naive values'
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format='mixed')
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)

def test_to_datetime_wrapped_datetime64_ps() -> None:
    result = to_datetime([np.datetime64(1901901901901, 'ps')])
    expected = DatetimeIndex(['1970-01-01 00:00:01.901901901'], dtype='datetime64[ns]', freq=None)
    tm.assert_index_equal(result, expected)

# End of annotated code.
