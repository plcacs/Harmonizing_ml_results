"""test to_datetime"""
import calendar
from collections import deque
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import locale
import zoneinfo
from dateutil.parser import parse
import numpy as np
import pytest
from pandas._libs import tslib
from pandas._libs.tslibs import iNaT, parsing
from pandas.compat import WASM
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, NaT, Series, Timestamp, date_range, isna, to_datetime
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
PARSING_ERR_MSG = "You might want to try:\\n    - passing `format` if your strings have a consistent format;\\n    - passing `format=\\'ISO8601\\'` if your strings are all ISO8601 but not necessarily in exactly the same format;\\n    - passing `format=\\'mixed\\'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this."

class TestTimeConversionFormats:

    def test_to_datetime_readonly(self, writable: Union[str, list[str], int]) -> None:
        arr = np.array([], dtype=object)
        arr.setflags(write=writable)
        result = to_datetime(arr)
        expected = to_datetime([])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('format, expected', [['%d/%m/%Y', [Timestamp('20000101'), Timestamp('20000201'), Timestamp('20000301')]], ['%m/%d/%Y', [Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000103')]]])
    def test_to_datetime_format(self, cache: Union[bool, str, dict[str, datetime.datetime]], index_or_series: Union[typing.Sequence[str], None, pandas.DataFrame, typing.Mapping], format: Union[bool, str, dict[str, datetime.datetime]], expected: Union[bool, str, numpy.ndarray]) -> None:
        values = index_or_series(['1/1/2000', '1/2/2000', '1/3/2000'])
        result = to_datetime(values, format=format, cache=cache)
        expected = index_or_series(expected)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('arg, expected, format', [['1/1/2000', '20000101', '%d/%m/%Y'], ['1/1/2000', '20000101', '%m/%d/%Y'], ['1/2/2000', '20000201', '%d/%m/%Y'], ['1/2/2000', '20000102', '%m/%d/%Y'], ['1/3/2000', '20000301', '%d/%m/%Y'], ['1/3/2000', '20000103', '%m/%d/%Y']])
    def test_to_datetime_format_scalar(self, cache: Union[str, dict], arg: Union[str, dict], expected: Union[dict, datetime.datetime.date.time, int], format: Union[str, dict]) -> None:
        result = to_datetime(arg, format=format, cache=cache)
        expected = Timestamp(expected)
        assert result == expected

    def test_to_datetime_format_YYYYMMDD(self, cache: Union[datetime.datetime, common.ScanLoadFn, dict[str, typing.Any]]) -> None:
        ser = Series([19801222, 19801222] + [19810105] * 5)
        expected = Series([Timestamp(x) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)
        result = to_datetime(ser.apply(str), format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_with_nat(self, cache: Union[datetime.datetime, pandas.DataFrame, dict[int, datetime.datetime]]) -> None:
        ser = Series([19801222, 19801222] + [19810105] * 5, dtype='float')
        expected = Series([Timestamp('19801222'), Timestamp('19801222')] + [Timestamp('19810105')] * 5, dtype='M8[s]')
        expected[2] = np.nan
        ser[2] = np.nan
        result = to_datetime(ser, format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)
        ser2 = ser.apply(str)
        ser2[2] = 'nat'
        with pytest.raises(ValueError, match='unconverted data remains when parsing with format "%Y%m%d": ".0". '):
            to_datetime(ser2, format='%Y%m%d', cache=cache)

    def test_to_datetime_format_YYYYMM_with_nat(self, cache: Union[datetime.datetime, dict[int, datetime.datetime], pandas.DataFrame]) -> None:
        ser = Series([198012, 198012] + [198101] * 5, dtype='float')
        expected = Series([Timestamp('19801201'), Timestamp('19801201')] + [Timestamp('19810101')] * 5, dtype='M8[s]')
        expected[2] = np.nan
        ser[2] = np.nan
        result = to_datetime(ser, format='%Y%m', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_oob_for_ns(self, cache: Union[datetime.datetime, dict[int, datetime.datetime], dict[str, typing.Any]]) -> None:
        ser = Series([20121231, 20141231, 99991231])
        result = to_datetime(ser, format='%Y%m%d', errors='raise', cache=cache)
        expected = Series(np.array(['2012-12-31', '2014-12-31', '9999-12-31'], dtype='M8[s]'), dtype='M8[s]')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_coercion(self, cache: Union[common.ScanLoadFn, dict[str, typing.Any], dict[int, datetime.datetime]]) -> None:
        ser = Series([20121231, 20141231, 999999999999999999999999999991231])
        result = to_datetime(ser, format='%Y%m%d', errors='coerce', cache=cache)
        expected = Series(['20121231', '20141231', 'NaT'], dtype='M8[s]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input_s', [['19801222', '20010112', None], ['19801222', '20010112', np.nan], ['19801222', '20010112', NaT], ['19801222', '20010112', 'NaT'], [19801222, 20010112, None], [19801222, 20010112, np.nan], [19801222, 20010112, NaT], [19801222, 20010112, 'NaT']])
    def test_to_datetime_format_YYYYMMDD_with_none(self, input_s: Union[str, typing.TextIO]) -> None:
        expected = Series([Timestamp('19801222'), Timestamp('20010112'), NaT])
        result = Series(to_datetime(input_s, format='%Y%m%d'))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input_s, expected', [[['19801222', np.nan, '20010012', '10019999'], [Timestamp('19801222'), np.nan, np.nan, np.nan]], [['19801222', '20010012', '10019999', np.nan], [Timestamp('19801222'), np.nan, np.nan, np.nan]], [[20190813, np.nan, 20010012, 20019999], [Timestamp('20190813'), np.nan, np.nan, np.nan]], [[20190813, 20010012, np.nan, 20019999], [Timestamp('20190813'), np.nan, np.nan, np.nan]]])
    def test_to_datetime_format_YYYYMMDD_overflow(self, input_s: Union[str, bool, list[str]], expected: Union[str, int, typing.Sequence[int]]) -> None:
        input_s = Series(input_s)
        result = to_datetime(input_s, format='%Y%m%d', errors='coerce')
        expected = Series(expected)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('data, format, expected', [([pd.NA], '%Y%m%d%H%M%S', ['NaT']), ([pd.NA], None, ['NaT']), ([pd.NA, '20210202202020'], '%Y%m%d%H%M%S', ['NaT', '2021-02-02 20:20:20']), (['201010', pd.NA], '%y%m%d', ['2020-10-10', 'NaT']), (['201010', pd.NA], '%d%m%y', ['2010-10-20', 'NaT']), ([None, np.nan, pd.NA], None, ['NaT', 'NaT', 'NaT']), ([None, np.nan, pd.NA], '%Y%m%d', ['NaT', 'NaT', 'NaT'])])
    def test_to_datetime_with_NA(self, data: Union[str, pandas.DataFrame], format: Union[str, pandas.DataFrame], expected: Union[str, pandas.DataFrame, list[str]]) -> None:
        result = to_datetime(data, format=format)
        expected = DatetimeIndex(expected)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_with_NA_with_warning(self) -> None:
        result = to_datetime(['201010', pd.NA])
        expected = DatetimeIndex(['2010-10-20', 'NaT'])
        tm.assert_index_equal(result, expected)

    def test_to_datetime_format_integer(self, cache: Union[datetime.datetime, dict[int, datetime.datetime], dict[str, typing.Any]]) -> None:
        ser = Series([2000, 2001, 2002])
        expected = Series([Timestamp(x) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y', cache=cache)
        tm.assert_series_equal(result, expected)
        ser = Series([200001, 200105, 200206])
        expected = Series([Timestamp(x[:4] + '-' + x[4:]) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y%m', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_microsecond(self, cache: Union[bool, blurr.core.record.Record, str, None]) -> None:
        month_abbr = calendar.month_abbr[4]
        val = f'01-{month_abbr}-2011 00:00:01.978'
        format = '%d-%b-%Y %H:%M:%S.%f'
        result = to_datetime(val, format=format, cache=cache)
        exp = datetime.strptime(val, format)
        assert result == exp

    @pytest.mark.parametrize('value, format, dt', [['01/10/2010 15:20', '%m/%d/%Y %H:%M', Timestamp('2010-01-10 15:20')], ['01/10/2010 05:43', '%m/%d/%Y %I:%M', Timestamp('2010-01-10 05:43')], ['01/10/2010 13:56:01', '%m/%d/%Y %H:%M:%S', Timestamp('2010-01-10 13:56:01')], pytest.param('01/10/2010 08:14 PM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 20:14'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False)), pytest.param('01/10/2010 07:40 AM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 07:40'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False)), pytest.param('01/10/2010 09:12:56 AM', '%m/%d/%Y %I:%M:%S %p', Timestamp('2010-01-10 09:12:56'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False))])
    def test_to_datetime_format_time(self, cache: Union[datetime.datetime.datetime, str], value: Union[datetime.datetime.datetime, str], format: Union[datetime.datetime.datetime, str], dt: Union[datetime.datetime.datetime, str]) -> None:
        assert to_datetime(value, format=format, cache=cache) == dt

    @td.skip_if_not_us_locale
    def test_to_datetime_with_non_exact(self, cache: Union[common.ScanLoadFn, dict, datetime.datetime]) -> None:
        ser = Series(['19MAY11', 'foobar19MAY11', '19MAY11:00:00:00', '19MAY11 00:00:00Z'])
        result = to_datetime(ser, format='%d%b%y', exact=False, cache=cache)
        expected = to_datetime(ser.str.extract('(\\d+\\w+\\d+)', expand=False), format='%d%b%y', cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('format, expected', [('%Y-%m-%d', Timestamp(2000, 1, 3)), ('%Y-%d-%m', Timestamp(2000, 3, 1)), ('%Y-%m-%d %H', Timestamp(2000, 1, 3, 12)), ('%Y-%d-%m %H', Timestamp(2000, 3, 1, 12)), ('%Y-%m-%d %H:%M', Timestamp(2000, 1, 3, 12, 34)), ('%Y-%d-%m %H:%M', Timestamp(2000, 3, 1, 12, 34)), ('%Y-%m-%d %H:%M:%S', Timestamp(2000, 1, 3, 12, 34, 56)), ('%Y-%d-%m %H:%M:%S', Timestamp(2000, 3, 1, 12, 34, 56)), ('%Y-%m-%d %H:%M:%S.%f', Timestamp(2000, 1, 3, 12, 34, 56, 123456)), ('%Y-%d-%m %H:%M:%S.%f', Timestamp(2000, 3, 1, 12, 34, 56, 123456)), ('%Y-%m-%d %H:%M:%S.%f%z', Timestamp(2000, 1, 3, 12, 34, 56, 123456, tz='UTC+01:00')), ('%Y-%d-%m %H:%M:%S.%f%z', Timestamp(2000, 3, 1, 12, 34, 56, 123456, tz='UTC+01:00'))])
    def test_non_exact_doesnt_parse_whole_string(self, cache: Union[bool, typing.Sequence[str]], format: Union[str, bool, None], expected: Union[numpy.ndarray, int]) -> None:
        result = to_datetime('2000-01-03 12:34:56.123456+01:00', format=format, exact=False)
        assert result == expected

    @pytest.mark.parametrize('arg', ['2012-01-01 09:00:00.000000001', '2012-01-01 09:00:00.000001', '2012-01-01 09:00:00.001', '2012-01-01 09:00:00.001000', '2012-01-01 09:00:00.001000000'])
    def test_parse_nanoseconds_with_formula(self, cache: Any, arg: Any) -> None:
        expected = to_datetime(arg, cache=cache)
        result = to_datetime(arg, format='%Y-%m-%d %H:%M:%S.%f', cache=cache)
        assert result == expected

    @pytest.mark.parametrize('value,fmt,expected', [['2009324', '%Y%W%w', '2009-08-13'], ['2013020', '%Y%U%w', '2013-01-13']])
    def test_to_datetime_format_weeks(self, value: Union[str, None, datetime.datetime.datetime], fmt: Union[str, None, datetime.datetime.datetime], expected: Union[str, None, datetime.datetime.datetime], cache: Union[str, None, datetime.datetime.datetime]) -> None:
        assert to_datetime(value, format=fmt, cache=cache) == Timestamp(expected)

    @pytest.mark.parametrize('fmt,dates,expected_dates', [['%Y-%m-%d %H:%M:%S %Z', ['2010-01-01 12:00:00 UTC'] * 2, [Timestamp('2010-01-01 12:00:00', tz='UTC')] * 2], ['%Y-%m-%d %H:%M:%S%z', ['2010-01-01 12:00:00+0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 +0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 Z', '2010-01-01 12:00:00 Z'], [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=0))), Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=0)))]]])
    def test_to_datetime_parse_tzname_or_tzoffset(self, fmt: Union[str, datetime.datetime, float], dates: Union[str, datetime.datetime, float], expected_dates: Union[str, int, typing.Type]) -> None:
        result = to_datetime(dates, format=fmt)
        expected = Index(expected_dates)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('fmt,dates,expected_dates', [['%Y-%m-%d %H:%M:%S %Z', ['2010-01-01 12:00:00 UTC', '2010-01-01 12:00:00 GMT', '2010-01-01 12:00:00 US/Pacific'], [Timestamp('2010-01-01 12:00:00', tz='UTC'), Timestamp('2010-01-01 12:00:00', tz='GMT'), Timestamp('2010-01-01 12:00:00', tz='US/Pacific')]], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100'], [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60))), Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=-60)))]]])
    def test_to_datetime_parse_tzname_or_tzoffset_utc_false_removed(self, fmt: Union[str, datetime.date, datetime.datetime.datetime, None], dates: Union[str, datetime.date, datetime.datetime.datetime, None], expected_dates: Union[datetime.date, str]) -> None:
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(dates, format=fmt)

    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(self) -> None:
        dates = ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100', '2010-01-01 12:00:00 +0300', '2010-01-01 12:00:00 +0400']
        expected_dates = ['2010-01-01 11:00:00+00:00', '2010-01-01 13:00:00+00:00', '2010-01-01 09:00:00+00:00', '2010-01-01 08:00:00+00:00']
        fmt = '%Y-%m-%d %H:%M:%S %z'
        result = to_datetime(dates, format=fmt, utc=True)
        expected = DatetimeIndex(expected_dates)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('offset', ['+0', '-1foo', 'UTCbar', ':10', '+01:000:01', ''])
    def test_to_datetime_parse_timezone_malformed(self, offset: pandas.DatetimeIndex) -> None:
        fmt = '%Y-%m-%d %H:%M:%S %z'
        date = '2010-01-01 12:00:00 ' + offset
        msg = '|'.join([f"""^time data ".*" doesn\\'t match format ".*". {PARSING_ERR_MSG}$""", f'^unconverted data remains when parsing with format ".*": ".*". {PARSING_ERR_MSG}$'])
        with pytest.raises(ValueError, match=msg):
            to_datetime([date], format=fmt)

    def test_to_datetime_parse_timezone_keeps_name(self) -> None:
        fmt = '%Y-%m-%d %H:%M:%S %z'
        arg = Index(['2010-01-01 12:00:00 Z'], name='foo')
        result = to_datetime(arg, format=fmt)
        expected = DatetimeIndex(['2010-01-01 12:00:00'], tz='UTC', name='foo')
        tm.assert_index_equal(result, expected)

class TestToDatetime:

    @pytest.mark.filterwarnings('ignore:Could not infer format')
    def test_to_datetime_overflow(self) -> None:
        arg = '08335394550'
        msg = 'Parsing "08335394550" to datetime overflows'
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
        d1 = datetime(2020, 1, 1, 17, tzinfo=timezone(-timedelta(hours=1)))
        d2 = datetime(2020, 1, 1, 18, tzinfo=timezone(-timedelta(hours=1)))
        res = to_datetime(['2020-01-01 17:00 -0100', d2])
        expected = to_datetime([d1, d2]).tz_convert(timezone(timedelta(minutes=-60)))
        tm.assert_index_equal(res, expected)

    def test_to_datetime_mixed_string_and_numeric(self) -> None:
        vals = ['2016-01-01', 0]
        expected = DatetimeIndex([Timestamp(x) for x in vals])
        result = to_datetime(vals, format='mixed')
        result2 = to_datetime(vals[::-1], format='mixed')[::-1]
        result3 = DatetimeIndex(vals)
        result4 = DatetimeIndex(vals[::-1])[::-1]
        tm.assert_index_equal(result, expected)
        tm.assert_index_equal(result2, expected)
        tm.assert_index_equal(result3, expected)
        tm.assert_index_equal(result4, expected)

    @pytest.mark.parametrize('format', ['%Y-%m-%d', '%Y-%d-%m'], ids=['ISO8601', 'non-ISO8601'])
    def test_to_datetime_mixed_date_and_string(self, format: Union[str, None]) -> None:
        d1 = date(2020, 1, 2)
        res = to_datetime(['2020-01-01', d1], format=format)
        expected = DatetimeIndex(['2020-01-01', '2020-01-02'], dtype='M8[s]')
        tm.assert_index_equal(res, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize('utc, args, expected', [pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-08:00'], DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 10:00:00+00:00'], dtype='datetime64[us, UTC]'), id='all tz-aware, with utc'), pytest.param(False, ['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00']).as_unit('us'), id='all tz-aware, without utc'), pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[us, UTC]'), id='all tz-aware, mixed offsets, with utc'), pytest.param(True, ['2000-01-01 01:00:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[us, UTC]'), id='tz-aware string, naive pydatetime, with utc')])
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_datetime_and_string_with_format(self, fmt: Union[str, datetime.date.time.date.time, datetime.datetime.datetime], utc: Union[str, datetime.date.time.date.time, datetime.datetime.datetime], args: Any, expected: Union[str, int], constructor: Union[str, float]) -> None:
        ts1 = constructor(args[0])
        ts2 = args[1]
        result = to_datetime([ts1, ts2], format=fmt, utc=utc)
        if constructor is Timestamp:
            expected = expected.as_unit('s')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_dt_and_str_with_format_mixed_offsets_utc_false_removed(self, fmt: Union[str, datetime.datetime.datetime], constructor: Union[str, typing.Callable[T, T], T]) -> None:
        args = ['2000-01-01 01:00:00', '2000-01-01 02:00:00+00:00']
        ts1 = constructor(args[0])
        ts2 = args[1]
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    @pytest.mark.parametrize('fmt, expected', [pytest.param('%Y-%m-%d %H:%M:%S%z', [Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'), Timestamp('2000-01-02 02:00:00+0200', tz='UTC+02:00'), NaT], id='ISO8601, non-UTC'), pytest.param('%Y-%d-%m %H:%M:%S%z', [Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'), Timestamp('2000-02-01 02:00:00+0200', tz='UTC+02:00'), NaT], id='non-ISO8601, non-UTC')])
    def test_to_datetime_mixed_offsets_with_none_tz_utc_false_removed(self, fmt: Union[str, datetime.tzinfo, None, datetime.datetime.date.time], expected: Union[str, set[str]]) -> None:
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=False)

    @pytest.mark.parametrize('fmt, expected', [pytest.param('%Y-%m-%d %H:%M:%S%z', DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-01-02 00:00:00+00:00', 'NaT'], dtype='datetime64[s, UTC]'), id='ISO8601, UTC'), pytest.param('%Y-%d-%m %H:%M:%S%z', DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-02-01 00:00:00+00:00', 'NaT'], dtype='datetime64[s, UTC]'), id='non-ISO8601, UTC')])
    def test_to_datetime_mixed_offsets_with_none(self, fmt: Union[datetime.datetime.datetime, str, None], expected: Union[str, float]) -> None:
        result = to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=True)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize('args', [pytest.param(['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-07:00'], id='all tz-aware, mixed timezones, without utc')])
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_datetime_and_string_with_format_raises(self, fmt: Union[typing.Callable, datetime.datetime, str], args: Any, constructor: Union[typing.Callable, str, list[list[typing.Any]]]) -> None:
        ts1 = constructor(args[0])
        ts2 = constructor(args[1])
        with pytest.raises(ValueError, match='cannot be converted to datetime64 unless utc=True'):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    def test_to_datetime_np_str(self) -> None:
        value = np.str_('2019-02-04 10:18:46.297000+0000')
        ser = Series([value])
        exp = Timestamp('2019-02-04 10:18:46.297000', tz='UTC')
        assert to_datetime(value) == exp
        assert to_datetime(ser.iloc[0]) == exp
        res = to_datetime([value])
        expected = Index([exp])
        tm.assert_index_equal(res, expected)
        res = to_datetime(ser)
        expected = Series(expected)
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize('s, _format, dt', [['2015-1-1', '%G-%V-%u', datetime(2014, 12, 29, 0, 0)], ['2015-1-4', '%G-%V-%u', datetime(2015, 1, 1, 0, 0)], ['2015-1-7', '%G-%V-%u', datetime(2015, 1, 4, 0, 0)], ['2024-52-1', '%G-%V-%u', datetime(2024, 12, 23, 0, 0)], ['2024-52-7', '%G-%V-%u', datetime(2024, 12, 29, 0, 0)], ['2025-1-1', '%G-%V-%u', datetime(2024, 12, 30, 0, 0)], ['2020-53-1', '%G-%V-%u', datetime(2020, 12, 28, 0, 0)]])
    def test_to_datetime_iso_week_year_format(self, s: Union[datetime.datetime, datetime.date.time.date.time, str], _format: Union[datetime.datetime, datetime.date.time.date.time, str], dt: Union[datetime.datetime, datetime.date.time.date.time, str]) -> None:
        assert to_datetime(s, format=_format) == dt

    @pytest.mark.parametrize('msg, s, _format', [['Week 53 does not exist in ISO year 2024', '2024 53 1', '%G %V %u'], ['Week 53 does not exist in ISO year 2023', '2023 53 1', '%G %V %u']])
    def test_invalid_iso_week_53(self, msg: Union[str, typing.Callable], s: Union[str, float, bytes], _format: Union[str, float, bytes]) -> None:
        with pytest.raises(ValueError, match=msg):
            to_datetime(s, format=_format)

    @pytest.mark.parametrize('msg, s, _format', [["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 50', '%Y %V'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51', '%G %V'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Monday', '%G %A'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Mon', '%G %a'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %w'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %u'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '2051', '%G'], ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 51 6 256', '%G %V %u %j'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sunday', '%Y %V %A'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sun', '%Y %V %a'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %w'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %u'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20', '%V'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sunday', '%V %A'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sun', '%V %a'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %w'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %u'], ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 50', '%G %j'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20 Monday', '%V %A']])
    @pytest.mark.parametrize('errors', ['raise', 'coerce'])
    def test_error_iso_week_year(self, msg: Union[str, bool, BaseException], s: Union[str, datetime.datetime, None, dict], _format: Union[str, datetime.datetime, None, dict], errors: Union[str, datetime.datetime, None, dict]) -> None:
        if locale.getlocale() != ('zh_CN', 'UTF-8') and locale.getlocale() != ('it_IT', 'UTF-8'):
            with pytest.raises(ValueError, match=msg):
                to_datetime(s, format=_format, errors=errors)

    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_to_datetime_dtarr(self, tz: Union[datetime.datetime.datetime, str, None, datetime.timezone]) -> None:
        dti = date_range('1965-04-03', periods=19, freq='2W', tz=tz)
        arr = dti._data
        result = to_datetime(arr)
        assert result is arr

    @td.skip_if_windows
    @pytest.mark.parametrize('utc', [True, False])
    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_to_datetime_arrow(self, tz: Union[datetime.datetime.datetime, None], utc: Union[datetime.datetime.datetime, None], index_or_series: Union[pandas.Series, str, int]) -> None:
        pa = pytest.importorskip('pyarrow')
        dti = date_range('1965-04-03', periods=19, freq='2W', tz=tz)
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
            now = Timestamp('now')
            pdnow = to_datetime('now')
            pdnow2 = to_datetime(['now'])[0]
            assert abs(pdnow._value - now._value) < 10000000000.0
            assert abs(pdnow2._value - now._value) < 10000000000.0
            assert pdnow.tzinfo is None
            assert pdnow2.tzinfo is None

    @td.skip_if_windows
    @pytest.mark.skipif(WASM, reason='tzset is not available on WASM')
    @pytest.mark.parametrize('tz', ['Pacific/Auckland', 'US/Samoa'])
    def test_to_datetime_today(self, tz: Union[datetime.datetime, None, str]) -> None:
        with tm.set_timezone(tz):
            nptoday = np.datetime64('today').astype('datetime64[us]').astype(np.int64)
            pdtoday = to_datetime('today')
            pdtoday2 = to_datetime(['today'])[0]
            tstoday = Timestamp('today')
            tstoday2 = Timestamp.today()
            assert abs(pdtoday.normalize()._value - nptoday) < 10000000000.0
            assert abs(pdtoday2.normalize()._value - nptoday) < 10000000000.0
            assert abs(pdtoday._value - tstoday._value) < 10000000000.0
            assert abs(pdtoday._value - tstoday2._value) < 10000000000.0
            assert pdtoday.tzinfo is None
            assert pdtoday2.tzinfo is None

    @pytest.mark.parametrize('arg', ['now', 'today'])
    def test_to_datetime_today_now_unicode_bytes(self, arg: Union[str, bytes]) -> None:
        to_datetime([arg])

    @pytest.mark.filterwarnings('ignore:Timestamp.utcnow is deprecated:FutureWarning')
    @pytest.mark.skipif(WASM, reason='tzset is not available on WASM')
    @pytest.mark.parametrize('format, expected_ds', [('%Y-%m-%d %H:%M:%S%z', '2020-01-03'), ('%Y-%d-%m %H:%M:%S%z', '2020-03-01'), (None, '2020-01-03')])
    @pytest.mark.parametrize('string, attribute', [('now', 'utcnow'), ('today', 'today')])
    def test_to_datetime_now_with_format(self, format: Union[str, None, datetime.tzinfo], expected_ds: Union[int, str, None], string: Union[str, None, datetime.tzinfo], attribute: Union[int, str, None]) -> None:
        result = to_datetime(['2020-01-03 00:00:00Z', string], format=format, utc=True)
        expected = DatetimeIndex([expected_ds, getattr(Timestamp, attribute)()], dtype='datetime64[s, UTC]')
        assert (expected - result).max().total_seconds() < 1

    @pytest.mark.parametrize('dt', [np.datetime64('2000-01-01'), np.datetime64('2000-01-02')])
    def test_to_datetime_dt64s(self, cache: Union[datetime.datetime, datetime.date, str], dt: Union[datetime.datetime, datetime.date, str]) -> None:
        assert to_datetime(dt, cache=cache) == Timestamp(dt)

    @pytest.mark.parametrize('arg, format', [('2001-01-01', '%Y-%m-%d'), ('01-01-2001', '%d-%m-%Y')])
    def test_to_datetime_dt64s_and_str(self, arg: Union[str, bool], format: Union[str, bool]) -> None:
        result = to_datetime([arg, np.datetime64('2020-01-01')], format=format)
        expected = DatetimeIndex(['2001-01-01', '2020-01-01'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dt', [np.datetime64('1000-01-01'), np.datetime64('5000-01-02')])
    @pytest.mark.parametrize('errors', ['raise', 'coerce'])
    def test_to_datetime_dt64s_out_of_ns_bounds(self, cache: Union[datetime.datetime, None], dt: Union[datetime.datetime, bool], errors: Union[datetime.datetime, None]) -> None:
        ts = to_datetime(dt, errors=errors, cache=cache)
        assert isinstance(ts, Timestamp)
        assert ts.unit == 's'
        assert ts.asm8 == dt
        ts = Timestamp(dt)
        assert ts.unit == 's'
        assert ts.asm8 == dt

    @pytest.mark.skip_ubsan
    def test_to_datetime_dt64d_out_of_bounds(self, cache: Union[dict, datetime.date, datetime.datetime, dict[str, dict[int, str]]]) -> None:
        dt64 = np.datetime64(np.iinfo(np.int64).max, 'D')
        msg = 'Out of bounds second timestamp: 25252734927768524-07-27'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(dt64)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt64, errors='raise', cache=cache)
        assert to_datetime(dt64, errors='coerce', cache=cache) is NaT

    @pytest.mark.parametrize('unit', ['s', 'D'])
    def test_to_datetime_array_of_dt64s(self, cache: Union[datetime.datetime, datetime.date, bool], unit: Union[pandas.DataFrame, list, dict]) -> None:
        dts = [np.datetime64('2000-01-01', unit), np.datetime64('2000-01-02', unit)] * 30
        result = to_datetime(dts, cache=cache)
        expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype='M8[s]')
        tm.assert_index_equal(result, expected)
        dts_with_oob = dts + [np.datetime64('9999-01-01')]
        to_datetime(dts_with_oob, errors='raise')
        result = to_datetime(dts_with_oob, errors='coerce', cache=cache)
        expected = DatetimeIndex(np.array(dts_with_oob, dtype='M8[s]'))
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz(self, cache: Union[datetime.datetime, dict, pandas.DataFrame]) -> None:
        arr = [Timestamp('2013-01-01 13:00:00-0800', tz='US/Pacific'), Timestamp('2013-01-02 14:00:00-0800', tz='US/Pacific')]
        result = to_datetime(arr, cache=cache)
        expected = DatetimeIndex(['2013-01-01 13:00:00', '2013-01-02 14:00:00'], tz='US/Pacific').as_unit('s')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz_mixed(self, cache: Union[dict, recidiviz.common.ingest_metadata.IngestMetadata, list[dict]]) -> None:
        arr = [Timestamp('2013-01-01 13:00:00', tz='US/Pacific'), Timestamp('2013-01-02 14:00:00', tz='US/Eastern')]
        msg = 'Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)
        result = to_datetime(arr, cache=cache, errors='coerce')
        expected = DatetimeIndex(['2013-01-01 13:00:00-08:00', 'NaT'], dtype='datetime64[s, US/Pacific]')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_different_offsets_removed(self, cache: Union[bool, float]) -> None:
        ts_string_1 = 'March 1, 2018 12:00:00+0400'
        ts_string_2 = 'March 1, 2018 12:00:00+0500'
        arr = [ts_string_1] * 5 + [ts_string_2] * 5
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)

    def test_to_datetime_tz_pytz(self, cache: Union[datetime.datetime, dict[int, datetime.datetime], pandas.DataFrame]) -> None:
        pytz = pytest.importorskip('pytz')
        us_eastern = pytz.timezone('US/Eastern')
        arr = np.array([us_eastern.localize(datetime(year=2000, month=1, day=1, hour=3, minute=0)), us_eastern.localize(datetime(year=2000, month=6, day=1, hour=3, minute=0))], dtype=object)
        result = to_datetime(arr, utc=True, cache=cache)
        expected = DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-06-01 07:00:00+00:00'], dtype='datetime64[us, UTC]', freq=None)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('init_constructor, end_constructor', [(Index, DatetimeIndex), (list, DatetimeIndex), (np.array, DatetimeIndex), (Series, Series)])
    def test_to_datetime_utc_true(self, cache: Union[dict[str, object], float, bool], init_constructor: Union[dict[str, object], float, bool], end_constructor: dict[str, object]) -> None:
        data = ['20100102 121314', '20100102 121315']
        expected_data = [Timestamp('2010-01-02 12:13:14', tz='utc'), Timestamp('2010-01-02 12:13:15', tz='utc')]
        result = to_datetime(init_constructor(data), format='%Y%m%d %H%M%S', utc=True, cache=cache)
        expected = end_constructor(expected_data)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('scalar, expected', [['20100102 121314', Timestamp('2010-01-02 12:13:14', tz='utc')], ['20100102 121315', Timestamp('2010-01-02 12:13:15', tz='utc')]])
    def test_to_datetime_utc_true_scalar(self, cache: Union[bool, dict, list[typing.AnyStr]], scalar: Union[bool, dict, list[typing.AnyStr]], expected: Union[int, typing.Sequence[int], tuple[numpy.ndarray]]) -> None:
        result = to_datetime(scalar, format='%Y%m%d %H%M%S', utc=True, cache=cache)
        assert result == expected

    def test_to_datetime_utc_true_with_series_single_value(self, cache: Union[datetime.datetime, common.ScanLoadFn]) -> None:
        ts = 1.5e+18
        result = to_datetime(Series([ts]), utc=True, cache=cache)
        expected = Series([Timestamp(ts, tz='utc')])
        tm.assert_series_equal(result, expected)

    def test_to_datetime_utc_true_with_series_tzaware_string(self, cache: Any) -> None:
        ts = '2013-01-01 00:00:00-01:00'
        expected_ts = '2013-01-01 01:00:00'
        data = Series([ts] * 3)
        result = to_datetime(data, utc=True, cache=cache)
        expected = Series([Timestamp(expected_ts, tz='utc')] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('date, dtype', [('2013-01-01 01:00:00', 'datetime64[ns]'), ('2013-01-01 01:00:00', 'datetime64[ns, UTC]')])
    def test_to_datetime_utc_true_with_series_datetime_ns(self, cache: Union[datetime.datetime.datetime, datetime.date, None], date: Union[datetime.datetime.datetime, datetime.date, None], dtype: Union[str, int, pandas.DataFrame]) -> None:
        expected = Series([Timestamp('2013-01-01 01:00:00', tz='UTC')], dtype='M8[ns, UTC]')
        result = to_datetime(Series([date], dtype=dtype), utc=True, cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_tz_psycopg2(self, request: Any, cache: Union[datetime.datetime.datetime, float, int]) -> None:
        psycopg2_tz = pytest.importorskip('psycopg2.tz')
        tz1 = psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None)
        tz2 = psycopg2_tz.FixedOffsetTimezone(offset=-240, name=None)
        arr = np.array([datetime(2000, 1, 1, 3, 0, tzinfo=tz1), datetime(2000, 6, 1, 3, 0, tzinfo=tz2)], dtype=object)
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
    def test_datetime_bool(self, cache: Union[typing.Mapping, None, list], arg: Union[sustainerds.api.core.resource.ResourceContext, typing.Callable]) -> None:
        msg = 'dtype bool cannot be converted to datetime64\\[ns\\]'
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)
        assert to_datetime(arg, errors='coerce', cache=cache) is NaT

    def test_datetime_bool_arrays_mixed(self, cache: dict) -> None:
        msg = f'{type(cache)} is not convertible to datetime'
        with pytest.raises(TypeError, match=msg):
            to_datetime([False, datetime.today()], cache=cache)
        with pytest.raises(ValueError, match=f"""^time data "True" doesn\\'t match format "%Y%m%d". {PARSING_ERR_MSG}$"""):
            to_datetime(['20130101', True], cache=cache)
        tm.assert_index_equal(to_datetime([0, False, NaT, 0.0], errors='coerce', cache=cache), DatetimeIndex([to_datetime(0, cache=cache), NaT, NaT, to_datetime(0, cache=cache)]))

    @pytest.mark.parametrize('arg', [bool, to_datetime])
    def test_datetime_invalid_datatype(self, arg: str) -> None:
        msg = 'is not convertible to datetime'
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)

    @pytest.mark.parametrize('errors', ['coerce', 'raise'])
    def test_invalid_format_raises(self, errors: Union[Exception, dict, None]) -> None:
        with pytest.raises(ValueError, match="':' is a bad directive in format 'H%:M%:S%"):
            to_datetime(['00:00:00'], format='H%:M%:S%', errors=errors)

    @pytest.mark.parametrize('value', ['a', '00:01:99'])
    @pytest.mark.parametrize('format', [None, '%H:%M:%S'])
    def test_datetime_invalid_scalar(self, value: Union[str, None, datetime.datetime], format: Union[str, None, datetime.datetime]) -> None:
        res = to_datetime(value, errors='coerce', format=format)
        assert res is NaT
        msg = '|'.join([f"""^time data "a" doesn\\'t match format "%H:%M:%S". {PARSING_ERR_MSG}$""", '^Given date string "a" not likely a datetime$', f'^unconverted data remains when parsing with format "%H:%M:%S": "9". {PARSING_ERR_MSG}$', '^second must be in 0..59: 00:01:99$'])
        with pytest.raises(ValueError, match=msg):
            to_datetime(value, errors='raise', format=format)

    @pytest.mark.parametrize('value', ['3000/12/11 00:00:00'])
    @pytest.mark.parametrize('format', [None, '%H:%M:%S'])
    def test_datetime_outofbounds_scalar(self, value: Union[str, None, datetime.datetime.datetime], format: Union[str, None]) -> None:
        res = to_datetime(value, errors='coerce', format=format)
        if format is None:
            assert isinstance(res, Timestamp)
            assert res == Timestamp(value)
        else:
            assert res is NaT
        if format is not None:
            msg = '^time data ".*" doesn\\\'t match format ".*"'
            with pytest.raises(ValueError, match=msg):
                to_datetime(value, errors='raise', format=format)
        else:
            res = to_datetime(value, errors='raise', format=format)
            assert isinstance(res, Timestamp)
            assert res == Timestamp(value)

    @pytest.mark.parametrize('values', [['a'], ['00:01:99'], ['a', 'b', '99:00:00']])
    @pytest.mark.parametrize('format', [None, '%H:%M:%S'])
    def test_datetime_invalid_index(self, values: Union[str, datetime.datetime], format: Union[str, datetime.datetime.datetime, None]) -> None:
        if format is None and len(values) > 1:
            warn = UserWarning
        else:
            warn = None
        with tm.assert_produces_warning(warn, match='Could not infer format', raise_on_extra_warnings=False):
            res = to_datetime(values, errors='coerce', format=format)
        tm.assert_index_equal(res, DatetimeIndex([NaT] * len(values)))
        msg = '|'.join(['^Given date string "a" not likely a datetime$', f"""^time data "a" doesn\\'t match format "%H:%M:%S". {PARSING_ERR_MSG}$""", f'^unconverted data remains when parsing with format "%H:%M:%S": "9". {PARSING_ERR_MSG}$', '^second must be in 0..59: 00:01:99$'])
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(warn, match='Could not infer format', raise_on_extra_warnings=False):
                to_datetime(values, errors='raise', format=format)

    @pytest.mark.parametrize('utc', [True, None])
    @pytest.mark.parametrize('format', ['%Y%m%d %H:%M:%S', None])
    @pytest.mark.parametrize('constructor', [list, tuple, np.array, Index, deque])
    def test_to_datetime_cache(self, utc: Union[datetime.datetime.datetime, datetime.datetime.tzinfo], format: Union[datetime.datetime.datetime, datetime.datetime.tzinfo], constructor: Union[str, bool]) -> None:
        date = '20130101 00:00:00'
        test_dates = [date] * 10 ** 5
        data = constructor(test_dates)
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_from_deque(self) -> None:
        result = to_datetime(deque([Timestamp('2010-06-02 09:30:00')] * 51))
        expected = to_datetime([Timestamp('2010-06-02 09:30:00')] * 51)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('utc', [True, None])
    @pytest.mark.parametrize('format', ['%Y%m%d %H:%M:%S', None])
    def test_to_datetime_cache_series(self, utc: Union[str, datetime.datetime.datetime], format: Union[str, datetime.datetime.datetime]) -> None:
        date = '20130101 00:00:00'
        test_dates = [date] * 10 ** 5
        data = Series(test_dates)
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_cache_scalar(self) -> None:
        date = '20130101 00:00:00'
        result = to_datetime(date, cache=True)
        expected = Timestamp('20130101 00:00:00')
        assert result == expected

    @pytest.mark.parametrize('datetimelikes,expected_values,exp_unit', (((None, np.nan) + (NaT,) * start_caching_at, (NaT,) * (start_caching_at + 2), 's'), ((None, Timestamp('2012-07-26')) + (NaT,) * start_caching_at, (NaT, Timestamp('2012-07-26')) + (NaT,) * start_caching_at, 's'), ((None,) + (NaT,) * start_caching_at + ('2012 July 26', Timestamp('2012-07-26')), (NaT,) * (start_caching_at + 1) + (Timestamp('2012-07-26'), Timestamp('2012-07-26')), 's')))
    def test_convert_object_to_datetime_with_cache(self, datetimelikes: Union[bool, pandas.DataFrame, tuple[dict]], expected_values: Union[numpy.ndarray, str, list[dict]], exp_unit: Union[numpy.ndarray, str, list[dict]]) -> None:
        ser = Series(datetimelikes, dtype='object')
        result_series = to_datetime(ser, errors='coerce')
        expected_series = Series(expected_values, dtype=f'datetime64[{exp_unit}]')
        tm.assert_series_equal(result_series, expected_series)

    @pytest.mark.parametrize('input', [Series([NaT] * 20 + [None] * 20, dtype='object'), Series([NaT] * 60 + [None] * 60, dtype='object'), Series([None] * 20), Series([None] * 60), Series([''] * 20), Series([''] * 60), Series([pd.NA] * 20), Series([pd.NA] * 60), Series([np.nan] * 20), Series([np.nan] * 60)])
    def test_to_datetime_converts_null_like_to_nat(self, cache: Union[dict, recidiviz.common.ingest_metadata.IngestMetadata, bool], input: Union[dict[str, typing.Any], list[dict]]) -> None:
        expected = Series([NaT] * len(input), dtype='M8[s]')
        result = to_datetime(input, cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('date, format', [('2017-20', '%Y-%W'), ('20 Sunday', '%W %A'), ('20 Sun', '%W %a'), ('2017-21', '%Y-%U'), ('20 Sunday', '%U %A'), ('20 Sun', '%U %a')])
    def test_week_without_day_and_calendar_year(self, date: str, format: str) -> None:
        msg = "Cannot use '%W' or '%U' without day and year"
        with pytest.raises(ValueError, match=msg):
            to_datetime(date, format=format)

    def test_to_datetime_coerce(self) -> None:
        ts_strings = ['March 1, 2018 12:00:00+0400', 'March 1, 2018 12:00:00+0500', '20100240']
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(ts_strings, errors='coerce')

    @pytest.mark.parametrize('string_arg, format', [('March 1, 2018', '%B %d, %Y'), ('2018-03-01', '%Y-%m-%d')])
    @pytest.mark.parametrize('outofbounds', [datetime(9999, 1, 1), date(9999, 1, 1), np.datetime64('9999-01-01'), 'January 1, 9999', '9999-01-01'])
    def test_to_datetime_coerce_oob(self, string_arg: Union[str, int], format: Union[str, bytes], outofbounds: str) -> None:
        ts_strings = [string_arg, outofbounds]
        result = to_datetime(ts_strings, errors='coerce', format=format)
        if isinstance(outofbounds, str) and format.startswith('%B') ^ outofbounds.startswith('J'):
            expected = DatetimeIndex([datetime(2018, 3, 1), NaT], dtype='M8[s]')
        elif isinstance(outofbounds, datetime):
            expected = DatetimeIndex([datetime(2018, 3, 1), outofbounds], dtype='M8[us]')
        else:
            expected = DatetimeIndex([datetime(2018, 3, 1), outofbounds], dtype='M8[s]')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_malformed_no_raise(self) -> None:
        ts_strings = ['200622-12-31', '111111-24-11']
        with tm.assert_produces_warning(UserWarning, match='Could not infer format', raise_on_extra_warnings=False):
            result = to_datetime(ts_strings, errors='coerce')
        exp = Index([NaT, NaT], dtype='M8[s]')
        tm.assert_index_equal(result, exp)

    def test_to_datetime_malformed_raise(self) -> None:
        ts_strings = ['200622-12-31', '111111-24-11']
        msg = 'Parsed string "200622-12-31" gives an invalid tzoffset, which must be between -timedelta\\(hours=24\\) and timedelta\\(hours=24\\)'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(UserWarning, match='Could not infer format'):
                to_datetime(ts_strings, errors='raise')

    def test_iso_8601_strings_with_same_offset(self) -> None:
        ts_str = '2015-11-18 15:30:00+05:30'
        result = to_datetime(ts_str)
        expected = Timestamp(ts_str)
        assert result == expected
        expected = DatetimeIndex([Timestamp(ts_str)] * 2)
        result = to_datetime([ts_str] * 2)
        tm.assert_index_equal(result, expected)
        result = DatetimeIndex([ts_str] * 2)
        tm.assert_index_equal(result, expected)

    def test_iso_8601_strings_with_different_offsets_removed(self) -> None:
        ts_strings = ['2015-11-18 15:30:00+05:30', '2015-11-18 16:30:00+06:30', NaT]
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(ts_strings)

    def test_iso_8601_strings_with_different_offsets_utc(self) -> None:
        ts_strings = ['2015-11-18 15:30:00+05:30', '2015-11-18 16:30:00+06:30', NaT]
        result = to_datetime(ts_strings, utc=True)
        expected = DatetimeIndex([Timestamp(2015, 11, 18, 10), Timestamp(2015, 11, 18, 10), NaT], tz='UTC').as_unit('s')
        tm.assert_index_equal(result, expected)

    def test_mixed_offsets_with_native_datetime_utc_false_raises(self) -> None:
        vals = ['nan', Timestamp('1990-01-01'), '2015-03-14T16:15:14.123-08:00', '2019-03-04T21:56:32.620-07:00', None, 'today', 'now']
        ser = Series(vals)
        assert all((ser[i] is vals[i] for i in range(len(vals))))
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser)

    def test_non_iso_strings_with_tz_offset(self) -> None:
        result = to_datetime(['March 1, 2018 12:00:00+0400'] * 2)
        expected = DatetimeIndex([datetime(2018, 3, 1, 12, tzinfo=timezone(timedelta(minutes=240)))] * 2).as_unit('s')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('ts, expected', [(Timestamp('2018-01-01'), Timestamp('2018-01-01', tz='UTC')), (Timestamp('2018-01-01', tz='US/Pacific'), Timestamp('2018-01-01 08:00', tz='UTC'))])
    def test_timestamp_utc_true(self, ts: datetime.datetime.datetime, expected: Union[float, str, None, tuple[float]]) -> None:
        result = to_datetime(ts, utc=True)
        assert result == expected

    @pytest.mark.parametrize('dt_str', ['00010101', '13000101', '30000101', '99990101'])
    def test_to_datetime_with_format_out_of_bounds(self, dt_str: str) -> None:
        res = to_datetime(dt_str, format='%Y%m%d')
        dtobj = datetime.strptime(dt_str, '%Y%m%d')
        expected = Timestamp(dtobj).as_unit('s')
        assert res == expected
        assert res.unit == expected.unit

    def test_to_datetime_utc(self) -> None:
        arr = np.array([parse('2012-06-13T01:39:00Z')], dtype=object)
        result = to_datetime(arr, utc=True)
        assert result.tz is timezone.utc

    def test_to_datetime_fixed_offset(self) -> None:
        from pandas.tests.indexes.datetimes.test_timezones import FixedOffset
        fixed_off = FixedOffset(-420, '-07:00')
        dates = [datetime(2000, 1, 1, tzinfo=fixed_off), datetime(2000, 1, 2, tzinfo=fixed_off), datetime(2000, 1, 3, tzinfo=fixed_off)]
        result = to_datetime(dates)
        assert result.tz == fixed_off

    @pytest.mark.parametrize('date', [['2020-10-26 00:00:00+06:00', '2020-10-26 00:00:00+01:00'], ['2020-10-26 00:00:00+06:00', Timestamp('2018-01-01', tz='US/Pacific')], ['2020-10-26 00:00:00+06:00', datetime(2020, 1, 1, 18).astimezone(zoneinfo.ZoneInfo('Australia/Melbourne'))]])
    def test_to_datetime_mixed_offsets_with_utc_false_removed(self, date: Union[datetime.datetime, int]) -> None:
        msg = 'Mixed timezones detected. Pass utc=True in to_datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(date, utc=False)

class TestToDatetimeUnit:

    @pytest.mark.parametrize('unit', ['Y', 'M'])
    @pytest.mark.parametrize('item', [150, float(150)])
    def test_to_datetime_month_or_year_unit_int(self, cache: Union[datetime.datetime.date.time, dict[int, datetime.datetime], dict], unit: list[blurr.core.record.Record], item: list[blurr.core.record.Record], request: typing.Iterable[typing.Mapping]) -> None:
        ts = Timestamp(item, unit=unit)
        expected = DatetimeIndex([ts], dtype='M8[ns]')
        result = to_datetime([item], unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(np.array([item], dtype=object), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(np.array([item]), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(np.array([item, np.nan]), unit=unit, cache=cache)
        assert result.isna()[1]
        tm.assert_index_equal(result[:1], expected)

    @pytest.mark.parametrize('unit', ['Y', 'M'])
    def test_to_datetime_month_or_year_unit_non_round_float(self, cache: Union[str, typing.Mapping, dict[str, typing.Any]], unit: Union[ResponseType, dict]) -> None:
        msg = f'Conversion of non-round float with unit={unit} is ambiguous'
        with pytest.raises(ValueError, match=msg):
            to_datetime([1.5], unit=unit, errors='raise')
        with pytest.raises(ValueError, match=msg):
            to_datetime(np.array([1.5]), unit=unit, errors='raise')
        msg = 'Given date string \\"1.5\\" not likely a datetime'
        with pytest.raises(ValueError, match=msg):
            to_datetime(['1.5'], unit=unit, errors='raise')
        res = to_datetime([1.5], unit=unit, errors='coerce')
        expected = Index([NaT], dtype='M8[ns]')
        tm.assert_index_equal(res, expected)
        res = to_datetime(['1.5'], unit=unit, errors='coerce')
        expected = to_datetime([NaT]).as_unit('ns')
        tm.assert_index_equal(res, expected)
        res = to_datetime([1.0], unit=unit)
        expected = to_datetime([1], unit=unit)
        tm.assert_index_equal(res, expected)

    def test_unit(self, cache: Union[waterbutler.core.path.WaterButlerPath, dict[str, typing.Any], dict[str, dict[str, typing.Any]]]) -> None:
        msg = 'cannot specify both format and unit'
        with pytest.raises(ValueError, match=msg):
            to_datetime([1], unit='D', format='%Y%m%d', cache=cache)

    def test_unit_array_mixed_nans(self, cache: Union[dict, dict[str, typing.Any]]) -> None:
        values = [11111111111111111, 1, 1.0, iNaT, NaT, np.nan, 'NaT', '']
        result = to_datetime(values, unit='D', errors='coerce', cache=cache)
        expected = DatetimeIndex(['NaT', '1970-01-02', '1970-01-02', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        msg = "cannot convert input 11111111111111111 with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, unit='D', errors='raise', cache=cache)

    def test_unit_array_mixed_nans_large_int(self, cache: Union[datetime.datetime, list[blurr.core.record.Record]]) -> None:
        values = [1420043460000000000000000, iNaT, NaT, np.nan, 'NaT']
        result = to_datetime(values, errors='coerce', unit='s', cache=cache)
        expected = DatetimeIndex(['NaT', 'NaT', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        msg = "cannot convert input 1420043460000000000000000 with the unit 's'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, errors='raise', unit='s', cache=cache)

    def test_to_datetime_invalid_str_not_out_of_bounds_valuerror(self, cache: Union[str, waterbutler.core.path.WaterButlerPath]) -> None:
        msg = 'Unknown datetime string format, unable to parse: foo'
        with pytest.raises(ValueError, match=msg):
            to_datetime('foo', errors='raise', unit='s', cache=cache)

    @pytest.mark.parametrize('error', ['raise', 'coerce'])
    def test_unit_consistency(self, cache: Union[bool, Job], error: Union[bool, Job]) -> None:
        expected = Timestamp('1970-05-09 14:25:11')
        result = to_datetime(11111111, unit='s', errors=error, cache=cache)
        assert result == expected
        assert isinstance(result, Timestamp)

    @pytest.mark.parametrize('errors', ['raise', 'coerce'])
    @pytest.mark.parametrize('dtype', ['float64', 'int64'])
    def test_unit_with_numeric(self, cache: Union[datetime.datetime.datetime, float], errors: Union[datetime.datetime.datetime, float], dtype: Union[bool, numpy.dtype, typing.Iterable]) -> None:
        expected = DatetimeIndex(['2015-06-19 05:33:20', '2015-05-27 22:33:20'], dtype='M8[ns]')
        arr = np.array([1.434692e+18, 1.432766e+18]).astype(dtype)
        result = to_datetime(arr, errors=errors, cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('exp, arr, warning', [[['NaT', '2015-06-19 05:33:20', '2015-05-27 22:33:20'], ['foo', 1.434692e+18, 1.432766e+18], UserWarning], [['2015-06-19 05:33:20', '2015-05-27 22:33:20', 'NaT', 'NaT'], [1.434692e+18, 1.432766e+18, 'foo', 'NaT'], None]])
    def test_unit_with_numeric_coerce(self, cache: Union[datetime.datetime, int], exp: Union[int, numpy.ndarray, tuple[numpy.ndarray]], arr: Union[datetime.datetime, int], warning: Union[int, tuple[str], dict[str, object]]) -> None:
        expected = DatetimeIndex(exp, dtype='M8[ns]')
        with tm.assert_produces_warning(warning, match='Could not infer format'):
            result = to_datetime(arr, errors='coerce', cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('arr', [[Timestamp('20130101'), 1.434692e+18, 1.432766e+18], [1.434692e+18, 1.432766e+18, Timestamp('20130101')]])
    def test_unit_mixed(self, cache: Union[int, tuple[int], datetime.datetime.datetime], arr: dict) -> None:
        expected = Index([Timestamp(x) for x in arr], dtype='M8[ns]')
        result = to_datetime(arr, errors='coerce', cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(arr, errors='raise', cache=cache)
        tm.assert_index_equal(result, expected)
        result = DatetimeIndex(arr)
        tm.assert_index_equal(result, expected)

    def test_unit_rounding(self, cache: Union[common.ScanLoadFn, dict, pandas.DataFrame]) -> None:
        value = 1434743731.877
        result = to_datetime(value, unit='s', cache=cache)
        expected = Timestamp('2015-06-19 19:55:31.877000093')
        assert result == expected
        alt = Timestamp(value, unit='s')
        assert alt == result

    @pytest.mark.parametrize('dtype', [int, float])
    def test_to_datetime_unit(self, dtype: bool) -> None:
        epoch = 1370745748
        ser = Series([epoch + t for t in range(20)]).astype(dtype)
        result = to_datetime(ser, unit='s')
        expected = Series([Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t) for t in range(20)], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('null', [iNaT, np.nan])
    def test_to_datetime_unit_with_nulls(self, null: bool) -> None:
        epoch = 1370745748
        ser = Series([epoch + t for t in range(20)] + [null])
        result = to_datetime(ser, unit='s')
        expected = Series([Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t) for t in range(20)] + [NaT], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_unit_fractional_seconds(self) -> None:
        epoch = 1370745748
        ser = Series([epoch + t for t in np.arange(0, 2, 0.25)] + [iNaT]).astype(float)
        result = to_datetime(ser, unit='s')
        expected = Series([Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t) for t in np.arange(0, 2, 0.25)] + [NaT], dtype='M8[ns]')
        result = result.round('ms')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_unit_na_values(self) -> None:
        result = to_datetime([1, 2, 'NaT', NaT, np.nan], unit='D')
        expected = DatetimeIndex([Timestamp('1970-01-02'), Timestamp('1970-01-03')] + ['NaT'] * 3, dtype='M8[ns]')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('bad_val', ['foo', 111111111])
    def test_to_datetime_unit_invalid(self, bad_val: Union[str, int, datetime.datetime]) -> None:
        if bad_val == 'foo':
            msg = f'Unknown datetime string format, unable to parse: {bad_val}'
        else:
            msg = "cannot convert input 111111111 with the unit 'D'"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, 2, bad_val], unit='D')

    @pytest.mark.parametrize('bad_val', ['foo', 111111111])
    def test_to_timestamp_unit_coerce(self, bad_val: Union[str, datetime.datetime.datetime, None]) -> None:
        expected = DatetimeIndex([Timestamp('1970-01-02'), Timestamp('1970-01-03')] + ['NaT'] * 1, dtype='M8[ns]')
        result = to_datetime([1, 2, bad_val], unit='D', errors='coerce')
        tm.assert_index_equal(result, expected)

    def test_float_to_datetime_raise_near_bounds(self) -> None:
        msg = "cannot convert input with unit 'D'"
        oneday_in_ns = 1000000000.0 * 60 * 60 * 24
        tsmax_in_days = 2 ** 63 / oneday_in_ns
        should_succeed = Series([0, tsmax_in_days - 0.005, -tsmax_in_days + 0.005], dtype=float)
        expected = (should_succeed * oneday_in_ns).astype(np.int64)
        for error_mode in ['raise', 'coerce']:
            result1 = to_datetime(should_succeed, unit='D', errors=error_mode)
            tm.assert_almost_equal(result1.astype(np.int64).astype(np.float64), expected.astype(np.float64), rtol=1e-10)
        should_fail1 = Series([0, tsmax_in_days + 0.005], dtype=float)
        should_fail2 = Series([0, -tsmax_in_days - 0.005], dtype=float)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail1, unit='D', errors='raise')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail2, unit='D', errors='raise')

class TestToDatetimeDataFrame:

    @pytest.fixture
    def df(self) -> DataFrame:
        return DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5], 'hour': [6, 7], 'minute': [58, 59], 'second': [10, 11], 'ms': [1, 1], 'us': [2, 2], 'ns': [3, 3]})

    def test_dataframe(self, df: pandas.DataFrame, cache: pandas.DataFrame) -> None:
        result = to_datetime({'year': df['year'], 'month': df['month'], 'day': df['day']}, cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), Timestamp('20160305 00:0:00')])
        tm.assert_series_equal(result, expected)
        result = to_datetime(df[['year', 'month', 'day']].to_dict(), cache=cache)
        expected.index = Index([0, 1])
        tm.assert_series_equal(result, expected)

    def test_dataframe_dict_with_constructable(self, df: Union[pandas.DataFrame, bytes], cache: Union[pandas.DataFrame, dict[str, list[typing.Any]], core_lib.core.models.Venue]) -> None:
        df2 = df[['year', 'month', 'day']].to_dict()
        df2['month'] = 2
        result = to_datetime(df2, cache=cache)
        expected2 = Series([Timestamp('20150204 00:00:00'), Timestamp('20160205 00:0:00')], index=Index([0, 1]))
        tm.assert_series_equal(result, expected2)

    @pytest.mark.parametrize('unit', [{'year': 'years', 'month': 'months', 'day': 'days', 'hour': 'hours', 'minute': 'minutes', 'second': 'seconds'}, {'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour', 'minute': 'minute', 'second': 'second'}])
    def test_dataframe_field_aliases_column_subset(self, df: pandas.DataFrame, cache: Union[pandas.DataFrame, typing.Mapping], unit: pandas.DataFrame) -> None:
        result = to_datetime(df[list(unit.keys())].rename(columns=unit), cache=cache)
        expected = Series([Timestamp('20150204 06:58:10'), Timestamp('20160305 07:59:11')], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    def test_dataframe_field_aliases(self, df: Union[pandas.DataFrame, core_lib.core.models.Venue, dict[str, list[typing.Any]]], cache: Union[pandas.DataFrame, core_lib.core.models.Venue, dict[str, list[typing.Any]]]) -> None:
        d = {'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour', 'minute': 'minute', 'second': 'second', 'ms': 'ms', 'us': 'us', 'ns': 'ns'}
        result = to_datetime(df.rename(columns=d), cache=cache)
        expected = Series([Timestamp('20150204 06:58:10.001002003'), Timestamp('20160305 07:59:11.001002003')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_str_dtype(self, df: pandas.DataFrame, cache: pandas.DataFrame) -> None:
        result = to_datetime(df.astype(str), cache=cache)
        expected = Series([Timestamp('20150204 06:58:10.001002003'), Timestamp('20160305 07:59:11.001002003')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_float32_dtype(self, df: pandas.DataFrame, cache: pandas.DataFrame) -> None:
        result = to_datetime(df.astype(np.float32), cache=cache)
        expected = Series([Timestamp('20150204 06:58:10.001002003'), Timestamp('20160305 07:59:11.001002003')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_coerce(self, cache: Union[recidiviz.common.ingest_metadata.IngestMetadata, list[str], dict]) -> None:
        df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5]})
        msg = '^cannot assemble the datetimes: time data ".+" doesn\\\'t match format "%Y%m%d"\\.'
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
        result = to_datetime(df2, errors='coerce', cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), NaT])
        tm.assert_series_equal(result, expected)

    def test_dataframe_extra_keys_raises(self, df: pandas.DataFrame, cache: Union[pandas.DataFrame, core_lib.core.models.Venue]) -> None:
        msg = 'extra keys have been passed to the datetime assemblage: \\[foo\\]'
        df2 = df.copy()
        df2['foo'] = 1
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    @pytest.mark.parametrize('cols', [['year'], ['year', 'month'], ['year', 'month', 'second'], ['month', 'day'], ['year', 'day', 'second']])
    def test_dataframe_missing_keys_raises(self, df: Union[pandas.DataFrame, str, dict[str, str]], cache: Union[pandas.DataFrame, str, dict[str, str]], cols: Union[pandas.DataFrame, str, dict[str, str]]) -> None:
        msg = 'to assemble mappings requires at least that \\[year, month, day\\] be specified: \\[.+\\] is missing'
        with pytest.raises(ValueError, match=msg):
            to_datetime(df[cols], cache=cache)

    def test_dataframe_duplicate_columns_raises(self, cache: Union[recidiviz.common.ingest_metadata.IngestMetadata, bool, T]) -> None:
        msg = 'cannot assemble with duplicate keys'
        df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5]})
        df2.columns = ['year', 'year', 'day']
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
        df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5], 'hour': [4, 5]})
        df2.columns = ['year', 'month', 'day', 'day']
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)

    def test_dataframe_int16(self, cache: Union[datetime.datetime, pandas.DataFrame, dict]) -> None:
        df = DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
        result = to_datetime(df.astype('int16'), cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), Timestamp('20160305 00:00:00')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_mixed(self, cache: Union[bool, pandas.DataFrame, common.ScanLoadFn]) -> None:
        df = DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
        df['month'] = df['month'].astype('int8')
        df['day'] = df['day'].astype('int8')
        result = to_datetime(df, cache=cache)
        expected = Series([Timestamp('20150204 00:00:00'), Timestamp('20160305 00:00:00')])
        tm.assert_series_equal(result, expected)

    def test_dataframe_float(self, cache: Union[recidiviz.common.ingest_metadata.IngestMetadata, bool, pandas.DataFrame]) -> None:
        df = DataFrame({'year': [2000, 2001], 'month': [1.5, 1], 'day': [1, 1]})
        msg = '^cannot assemble the datetimes: unconverted data remains when parsing with format ".*": "1".'
        with pytest.raises(ValueError, match=msg):
            to_datetime(df, cache=cache)

    def test_dataframe_utc_true(self) -> None:
        df = DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
        result = to_datetime(df, utc=True)
        expected = Series(np.array(['2015-02-04', '2016-03-05'], dtype='datetime64[s]')).dt.tz_localize('UTC')
        tm.assert_series_equal(result, expected)

class TestToDatetimeMisc:

    def test_to_datetime_barely_out_of_bounds(self) -> None:
        arr = np.array(['2262-04-11 23:47:16.854775808'], dtype=object)
        msg = '^Out of bounds nanosecond timestamp: .*'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arr)

    @pytest.mark.parametrize('arg, exp_str', [['2012-01-01 00:00:00', '2012-01-01 00:00:00'], ['20121001', '2012-10-01']])
    def test_to_datetime_iso8601(self, cache: Union[str, dict], arg: Union[str, dict], exp_str: str) -> None:
        result = to_datetime([arg], cache=cache)
        exp = Timestamp(exp_str)
        assert result[0] == exp

    @pytest.mark.parametrize('input, format', [('2012', '%Y-%m'), ('2012-01', '%Y-%m-%d'), ('2012-01-01', '%Y-%m-%d %H'), ('2012-01-01 10', '%Y-%m-%d %H:%M'), ('2012-01-01 10:00', '%Y-%m-%d %H:%M:%S'), ('2012-01-01 10:00:00', '%Y-%m-%d %H:%M:%S.%f'), ('2012-01-01 10:00:00.123', '%Y-%m-%d %H:%M:%S.%f%z'), (0, '%Y-%m-%d')])
    @pytest.mark.parametrize('exact', [True, False])
    def test_to_datetime_iso8601_fails(self, input: str, format: str, exact: Union[str, int, None]) -> None:
        with pytest.raises(ValueError, match=f'''time data \\"{input}\\" doesn't match format \\"{format}\\"'''):
            to_datetime(input, format=format, exact=exact)

    @pytest.mark.parametrize('input, format', [('2012-01-01', '%Y-%m'), ('2012-01-01 10', '%Y-%m-%d'), ('2012-01-01 10:00', '%Y-%m-%d %H'), ('2012-01-01 10:00:00', '%Y-%m-%d %H:%M'), (0, '%Y-%m-%d')])
    def test_to_datetime_iso8601_exact_fails(self, input: str, format: str) -> None:
        msg = '|'.join([f'^unconverted data remains when parsing with format ".*": ".*". {PARSING_ERR_MSG}$', f"""^time data ".*" doesn't match format ".*". {PARSING_ERR_MSG}$"""])
        with pytest.raises(ValueError, match=msg):
            to_datetime(input, format=format)

    @pytest.mark.parametrize('input, format', [('2012-01-01', '%Y-%m'), ('2012-01-01 00', '%Y-%m-%d'), ('2012-01-01 00:00', '%Y-%m-%d %H'), ('2012-01-01 00:00:00', '%Y-%m-%d %H:%M')])
    def test_to_datetime_iso8601_non_exact(self, input: Union[str, pandas.DataFrame, pandas.core.series.Series], format: Union[str, pandas.DataFrame, pandas.core.series.Series]) -> None:
        expected = Timestamp(2012, 1, 1)
        result = to_datetime(input, format=format, exact=False)
        assert result == expected

    @pytest.mark.parametrize('input, format', [('2020-01', '%Y/%m'), ('2020-01-01', '%Y/%m/%d'), ('2020-01-01 00', '%Y/%m/%dT%H'), ('2020-01-01T00', '%Y/%m/%d %H'), ('2020-01-01 00:00', '%Y/%m/%dT%H:%M'), ('2020-01-01T00:00', '%Y/%m/%d %H:%M'), ('2020-01-01 00:00:00', '%Y/%m/%dT%H:%M:%S'), ('2020-01-01T00:00:00', '%Y/%m/%d %H:%M:%S')])
    def test_to_datetime_iso8601_separator(self, input: str, format: str) -> None:
        with pytest.raises(ValueError, match=f'''time data \\"{input}\\" doesn\\'t match format \\"{format}\\"'''):
            to_datetime(input, format=format)

    @pytest.mark.parametrize('input, format', [('2020-01', '%Y-%m'), ('2020-01-01', '%Y-%m-%d'), ('2020-01-01 00', '%Y-%m-%d %H'), ('2020-01-01T00', '%Y-%m-%dT%H'), ('2020-01-01 00:00', '%Y-%m-%d %H:%M'), ('2020-01-01T00:00', '%Y-%m-%dT%H:%M'), ('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), ('2020-01-01T00:00:00', '%Y-%m-%dT%H:%M:%S'), ('2020-01-01T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f'), ('2020-01-01T00:00:00.000000', '%Y-%m-%dT%H:%M:%S.%f'), ('2020-01-01T00:00:00.000000000', '%Y-%m-%dT%H:%M:%S.%f')])
    def test_to_datetime_iso8601_valid(self, input: Union[str, pandas.core.series.Series, datetime.timedelta], format: Union[str, pandas.core.series.Series, datetime.timedelta]) -> None:
        expected = Timestamp(2020, 1, 1)
        result = to_datetime(input, format=format)
        assert result == expected

    @pytest.mark.parametrize('input, format', [('2020-1', '%Y-%m'), ('2020-1-1', '%Y-%m-%d'), ('2020-1-1 0', '%Y-%m-%d %H'), ('2020-1-1T0', '%Y-%m-%dT%H'), ('2020-1-1 0:0', '%Y-%m-%d %H:%M'), ('2020-1-1T0:0', '%Y-%m-%dT%H:%M'), ('2020-1-1 0:0:0', '%Y-%m-%d %H:%M:%S'), ('2020-1-1T0:0:0', '%Y-%m-%dT%H:%M:%S'), ('2020-1-1T0:0:0.000', '%Y-%m-%dT%H:%M:%S.%f'), ('2020-1-1T0:0:0.000000', '%Y-%m-%dT%H:%M:%S.%f'), ('2020-1-1T0:0:0.000000000', '%Y-%m-%dT%H:%M:%S.%f')])
    def test_to_datetime_iso8601_non_padded(self, input: Union[str, pandas.core.series.Series, datetime.timedelta], format: Union[str, pandas.core.series.Series, datetime.timedelta]) -> None:
        expected = Timestamp(2020, 1, 1)
        result = to_datetime(input, format=format)
        assert result == expected

    @pytest.mark.parametrize('input, format', [('2020-01-01T00:00:00.000000000+00:00', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2020-01-01T00:00:00+00:00', '%Y-%m-%dT%H:%M:%S%z'), ('2020-01-01T00:00:00Z', '%Y-%m-%dT%H:%M:%S%z')])
    def test_to_datetime_iso8601_with_timezone_valid(self, input: Union[str, pandas.core.series.Series, datetime.datetime.datetime], format: Union[str, pandas.core.series.Series, datetime.datetime.datetime]) -> None:
        expected = Timestamp(2020, 1, 1, tzinfo=timezone.utc)
        result = to_datetime(input, format=format)
        assert result == expected

    def test_to_datetime_default(self, cache: Union[datetime.datetime, recidiviz.common.ingest_metadata.IngestMetadata]) -> None:
        rs = to_datetime('2001', cache=cache)
        xp = datetime(2001, 1, 1)
        assert rs == xp

    @pytest.mark.xfail(reason='fails to enforce dayfirst=True, which would raise')
    def test_to_datetime_respects_dayfirst(self, cache: Union[common.ScanLoadFn, dict, bool]) -> None:
        msg = 'Invalid date specified'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(UserWarning, match='Provide format'):
                to_datetime('01-13-2012', dayfirst=True, cache=cache)

    def test_to_datetime_on_datetime64_series(self, cache: Union[dict[int, datetime.datetime], pandas.DataFrame, datetime.datetime]) -> None:
        ser = Series(date_range('1/1/2000', periods=10))
        result = to_datetime(ser, cache=cache)
        assert result[0] == ser[0]

    def test_to_datetime_with_space_in_series(self, cache: Union[dict, pandas.DataFrame]) -> None:
        ser = Series(['10/18/2006', '10/18/2008', ' '])
        msg = f"""^time data " " doesn\\'t match format "%m/%d/%Y". {PARSING_ERR_MSG}$"""
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, errors='raise', cache=cache)
        result_coerce = to_datetime(ser, errors='coerce', cache=cache)
        expected_coerce = Series([datetime(2006, 10, 18), datetime(2008, 10, 18), NaT]).dt.as_unit('s')
        tm.assert_series_equal(result_coerce, expected_coerce)

    @td.skip_if_not_us_locale
    def test_to_datetime_with_apply(self, cache: datetime.datetime) -> None:
        td = Series(['May 04', 'Jun 02', 'Dec 11'], index=[1, 2, 3])
        expected = to_datetime(td, format='%b %y', cache=cache)
        result = td.apply(to_datetime, format='%b %y', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_timezone_name(self) -> None:
        result = to_datetime('2020-01-01 00:00:00UTC', format='%Y-%m-%d %H:%M:%S%Z')
        expected = Timestamp(2020, 1, 1).tz_localize('UTC')
        assert result == expected

    @td.skip_if_not_us_locale
    @pytest.mark.parametrize('errors', ['raise', 'coerce'])
    def test_to_datetime_with_apply_with_empty_str(self, cache: Union[Exception, str, None, datetime.datetime.datetime], errors: Union[Exception, str, None, datetime.datetime.datetime]) -> None:
        td = Series(['May 04', 'Jun 02', ''], index=[1, 2, 3])
        expected = to_datetime(td, format='%b %y', errors=errors, cache=cache)
        result = td.apply(lambda x: to_datetime(x, format='%b %y', errors='coerce', cache=cache))
        tm.assert_series_equal(result, expected)

    def test_to_datetime_empty_stt(self, cache: datetime.datetime) -> None:
        result = to_datetime('', cache=cache)
        assert result is NaT

    def test_to_datetime_empty_str_list(self, cache: dict) -> None:
        result = to_datetime(['', ''], cache=cache)
        assert isna(result).all()

    def test_to_datetime_zero(self, cache: pandas.DataFrame) -> None:
        result = Timestamp(0)
        expected = to_datetime(0, cache=cache)
        assert result == expected

    def test_to_datetime_strings(self, cache: common.ScanLoadFn) -> None:
        expected = to_datetime(['2012'], cache=cache)[0]
        result = to_datetime('2012', cache=cache)
        assert result == expected

    def test_to_datetime_strings_variation(self, cache: Union[datetime.datetime, dict[str, typing.Any], float]) -> None:
        array = ['2012', '20120101', '20120101 12:01:01']
        expected = [to_datetime(dt_str, cache=cache) for dt_str in array]
        result = [Timestamp(date_str) for date_str in array]
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize('result', [Timestamp('2012'), to_datetime('2012')])
    def test_to_datetime_strings_vs_constructor(self, result: Union[pandas.DataFrame, dict[str, typing.Any], int]) -> None:
        expected = Timestamp(2012, 1, 1)
        assert result == expected

    def test_to_datetime_unprocessable_input(self, cache: Union[str, dict]) -> None:
        msg = '^Given date string "1" not likely a datetime$'
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, '1'], errors='raise', cache=cache)

    def test_to_datetime_other_datetime64_units(self) -> None:
        scalar = np.int64(1337904000000000).view('M8[us]')
        as_obj = scalar.astype('O')
        index = DatetimeIndex([scalar])
        assert index[0] == scalar.astype('O')
        value = Timestamp(scalar)
        assert value == as_obj

    def test_to_datetime_list_of_integers(self) -> None:
        rng = date_range('1/1/2000', periods=20)
        rng = DatetimeIndex(rng.values)
        ints = list(rng.asi8)
        result = DatetimeIndex(ints)
        tm.assert_index_equal(rng, result)

    def test_to_datetime_overflow(self) -> None:
        msg = "Cannot cast 139999 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            date_range(start='1/1/1700', freq='B', periods=100000)

    def test_string_invalid_operation(self, cache: Union[dict[str, typing.Any], dict, str]) -> None:
        invalid = np.array(['87156549591102612381000001219H5'], dtype=object)
        with pytest.raises(ValueError, match='Unknown datetime string format'):
            to_datetime(invalid, errors='raise', cache=cache)

    def test_string_na_nat_conversion(self, cache: Union[str, list[str]]) -> None:
        strings = np.array(['1/1/2000', '1/2/2000', np.nan, '1/4/2000'], dtype=object)
        expected = np.empty(4, dtype='M8[s]')
        for i, val in enumerate(strings):
            if isna(val):
                expected[i] = iNaT
            else:
                expected[i] = parse(val)
        result = tslib.array_to_datetime(strings)[0]
        tm.assert_almost_equal(result, expected)
        result2 = to_datetime(strings, cache=cache)
        assert isinstance(result2, DatetimeIndex)
        tm.assert_numpy_array_equal(result, result2.values)

    def test_string_na_nat_conversion_malformed(self, cache: Union[datetime.datetime, dict, dict[str, typing.Any]]) -> None:
        malformed = np.array(['1/100/2000', np.nan], dtype=object)
        msg = 'Unknown datetime string format'
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors='raise', cache=cache)
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors='raise', cache=cache)

    def test_string_na_nat_conversion_with_name(self, cache: Union[dict[str, typing.Any], list[dict], blurr.core.record.Record]) -> None:
        idx = ['a', 'b', 'c', 'd', 'e']
        series = Series(['1/1/2000', np.nan, '1/3/2000', np.nan, '1/5/2000'], index=idx, name='foo')
        dseries = Series([to_datetime('1/1/2000', cache=cache), np.nan, to_datetime('1/3/2000', cache=cache), np.nan, to_datetime('1/5/2000', cache=cache)], index=idx, name='foo')
        result = to_datetime(series, cache=cache)
        dresult = to_datetime(dseries, cache=cache)
        expected = Series(np.empty(5, dtype='M8[s]'), index=idx)
        for i in range(5):
            x = series.iloc[i]
            if isna(x):
                expected.iloc[i] = NaT
            else:
                expected.iloc[i] = to_datetime(x, cache=cache)
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name == 'foo'
        tm.assert_series_equal(dresult, expected, check_names=False)
        assert dresult.name == 'foo'

    @pytest.mark.parametrize('unit', ['h', 'm', 's', 'ms', 'us', 'ns'])
    def test_dti_constructor_numpy_timeunits(self, cache: Union[pandas.DataFrame, core.D.Key], unit: pandas.DataFrame) -> None:
        dtype = np.dtype(f'M8[{unit}]')
        base = to_datetime(['2000-01-01T00:00', '2000-01-02T00:00', 'NaT'], cache=cache)
        values = base.values.astype(dtype)
        if unit in ['h', 'm']:
            unit = 's'
        exp_dtype = np.dtype(f'M8[{unit}]')
        expected = DatetimeIndex(base.astype(exp_dtype))
        assert expected.dtype == exp_dtype
        tm.assert_index_equal(DatetimeIndex(values), expected)
        tm.assert_index_equal(to_datetime(values, cache=cache), expected)

    def test_dayfirst(self, cache: Union[datetime.datetime, bool]) -> None:
        arr = ['10/02/2014', '11/02/2014', '12/02/2014']
        expected = DatetimeIndex([datetime(2014, 2, 10), datetime(2014, 2, 11), datetime(2014, 2, 12)]).as_unit('s')
        idx1 = DatetimeIndex(arr, dayfirst=True)
        idx2 = DatetimeIndex(np.array(arr), dayfirst=True)
        idx3 = to_datetime(arr, dayfirst=True, cache=cache)
        idx4 = to_datetime(np.array(arr), dayfirst=True, cache=cache)
        idx5 = DatetimeIndex(Index(arr), dayfirst=True)
        idx6 = DatetimeIndex(Series(arr), dayfirst=True)
        tm.assert_index_equal(expected, idx1)
        tm.assert_index_equal(expected, idx2)
        tm.assert_index_equal(expected, idx3)
        tm.assert_index_equal(expected, idx4)
        tm.assert_index_equal(expected, idx5)
        tm.assert_index_equal(expected, idx6)

    def test_dayfirst_warnings_valid_input(self) -> None:
        warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
        arr = ['31/12/2014', '10/03/2011']
        expected = DatetimeIndex(['2014-12-31', '2011-03-10'], dtype='datetime64[s]', freq=None)
        res1 = to_datetime(arr, dayfirst=True)
        tm.assert_index_equal(expected, res1)
        with tm.assert_produces_warning(UserWarning, match=warning_msg):
            res2 = to_datetime(arr, dayfirst=False)
        tm.assert_index_equal(expected, res2)

    def test_dayfirst_warnings_invalid_input(self) -> None:
        arr = ['31/12/2014', '03/30/2011']
        with pytest.raises(ValueError, match=f"""^time data "03/30/2011" doesn\\'t match format "%d/%m/%Y". {PARSING_ERR_MSG}$"""):
            to_datetime(arr, dayfirst=True)

    @pytest.mark.parametrize('klass', [DatetimeIndex, DatetimeArray._from_sequence])
    def test_to_datetime_dta_tz(self, klass: Union[str, typing.Type, list['NodeType']]) -> None:
        dti = date_range('2015-04-05', periods=3).rename('foo')
        expected = dti.tz_localize('UTC')
        obj = klass(dti)
        expected = klass(expected)
        result = to_datetime(obj, utc=True)
        tm.assert_equal(result, expected)

class TestGuessDatetimeFormat:

    @pytest.mark.parametrize('test_list', [['2011-12-30 00:00:00.000000', '2011-12-30 00:00:00.000000', '2011-12-30 00:00:00.000000'], [np.nan, np.nan, '2011-12-30 00:00:00.000000'], ['', '2011-12-30 00:00:00.000000'], ['NaT', '2011-12-30 00:00:00.000000'], ['2011-12-30 00:00:00.000000', 'random_string'], ['now', '2011-12-30 00:00:00.000000'], ['today', '2011-12-30 00:00:00.000000']])
    def test_guess_datetime_format_for_array(self, test_list: Union[typing.Callable[str, bool], list[tuple[str]], numpy.ndarray]) -> None:
        expected_format = '%Y-%m-%d %H:%M:%S.%f'
        test_array = np.array(test_list, dtype=object)
        assert tools._guess_datetime_format_for_array(test_array) == expected_format

    @td.skip_if_not_us_locale
    def test_guess_datetime_format_for_array_all_nans(self) -> None:
        format_for_string_of_nans = tools._guess_datetime_format_for_array(np.array([np.nan, np.nan, np.nan], dtype='O'))
        assert format_for_string_of_nans is None

class TestToDatetimeInferFormat:

    @pytest.mark.parametrize('test_format', ['%m-%d-%Y', '%m/%d/%Y %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f'])
    def test_to_datetime_infer_datetime_format_consistent_format(self, cache: Union[str, None, tuple[typing.Union[str,dict[str, str]]]], test_format: Union[bool, common.ScanLoadFn]) -> None:
        ser = Series(date_range('20000101', periods=50, freq='h'))
        s_as_dt_strings = ser.apply(lambda x: x.strftime(test_format))
        with_format = to_datetime(s_as_dt_strings, format=test_format, cache=cache)
        without_format = to_datetime(s_as_dt_strings, cache=cache)
        tm.assert_series_equal(with_format, without_format)

    def test_to_datetime_inconsistent_format(self, cache: Union[pandas.DataFrame, bool, recidiviz.common.ingest_metadata.IngestMetadata]) -> None:
        data = ['01/01/2011 00:00:00', '01-02-2011 00:00:00', '2011-01-03T00:00:00']
        ser = Series(np.array(data))
        msg = f"""^time data "01-02-2011 00:00:00" doesn\\'t match format "%m/%d/%Y %H:%M:%S". {PARSING_ERR_MSG}$"""
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, cache=cache)

    def test_to_datetime_consistent_format(self, cache: datetime.datetime) -> None:
        data = ['Jan/01/2011', 'Feb/01/2011', 'Mar/01/2011']
        ser = Series(np.array(data))
        result = to_datetime(ser, cache=cache)
        expected = Series(['2011-01-01', '2011-02-01', '2011-03-01'], dtype='datetime64[s]')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_series_with_nans(self, cache: Union[datetime.datetime, common.ScanLoadFn, dict[str, typing.Any]]) -> None:
        ser = Series(np.array(['01/01/2011 00:00:00', np.nan, '01/03/2011 00:00:00', np.nan], dtype=object))
        result = to_datetime(ser, cache=cache)
        expected = Series(['2011-01-01', NaT, '2011-01-03', NaT], dtype='datetime64[s]')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_series_start_with_nans(self, cache: Union[datetime.datetime, dict, common.ScanLoadFn]) -> None:
        ser = Series(np.array([np.nan, np.nan, '01/01/2011 00:00:00', '01/02/2011 00:00:00', '01/03/2011 00:00:00'], dtype=object))
        result = to_datetime(ser, cache=cache)
        expected = Series([NaT, NaT, '2011-01-01', '2011-01-02', '2011-01-03'], dtype='datetime64[s]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz_name, offset', [('UTC', 0), ('UTC-3', 180), ('UTC+3', -180)])
    def test_infer_datetime_format_tz_name(self, tz_name: Union[str, int, list[dict[str, typing.Any]]], offset: Union[datetime.datetime, None, bool, str]) -> None:
        ser = Series([f'2019-02-02 08:07:13 {tz_name}'])
        result = to_datetime(ser)
        tz = timezone(timedelta(minutes=offset))
        expected = Series([Timestamp('2019-02-02 08:07:13').tz_localize(tz)])
        expected = expected.dt.as_unit('s')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ts,zero_tz', [('2019-02-02 08:07:13', 'Z'), ('2019-02-02 08:07:13', ''), ('2019-02-02 08:07:13.012345', 'Z'), ('2019-02-02 08:07:13.012345', '')])
    def test_infer_datetime_format_zero_tz(self, ts: Union[datetime.datetime, int], zero_tz: Union[datetime.datetime, bool]) -> None:
        ser = Series([ts + zero_tz])
        result = to_datetime(ser)
        tz = timezone.utc if zero_tz == 'Z' else None
        expected = Series([Timestamp(ts, tz=tz)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('format', [None, '%Y-%m-%d'])
    def test_to_datetime_iso8601_noleading_0s(self, cache: Union[bool, datetime.date.time.date, datetime.datetime], format: Union[bool, datetime.date.time.date, datetime.datetime]) -> None:
        ser = Series(['2014-1-1', '2014-2-2', '2015-3-3'])
        expected = Series([Timestamp('2014-01-01'), Timestamp('2014-02-02'), Timestamp('2015-03-03')])
        result = to_datetime(ser, format=format, cache=cache)
        tm.assert_series_equal(result, expected)

class TestDaysInMonth:

    @pytest.mark.parametrize('arg, format', [['2015-02-29', None], ['2015-02-29', '%Y-%m-%d'], ['2015-02-32', '%Y-%m-%d'], ['2015-04-31', '%Y-%m-%d']])
    def test_day_not_in_month_coerce(self, cache: Union[dict, str, None, dict[str, typing.Any]], arg: Union[dict, str, None, dict[str, typing.Any]], format: Union[dict, str, None, dict[str, typing.Any]]) -> None:
        assert isna(to_datetime(arg, errors='coerce', format=format, cache=cache))

    def test_day_not_in_month_raise(self, cache: str) -> None:
        msg = 'day is out of range for month: 2015-02-29'
        with pytest.raises(ValueError, match=msg):
            to_datetime('2015-02-29', errors='raise', cache=cache)

    @pytest.mark.parametrize('arg, format, msg', [('2015-02-29', '%Y-%m-%d', f'^day is out of range for month. {PARSING_ERR_MSG}$'), ('2015-29-02', '%Y-%d-%m', f'^day is out of range for month. {PARSING_ERR_MSG}$'), ('2015-02-32', '%Y-%m-%d', f'^unconverted data remains when parsing with format "%Y-%m-%d": "2". {PARSING_ERR_MSG}$'), ('2015-32-02', '%Y-%d-%m', f"""^time data "2015-32-02" doesn't match format "%Y-%d-%m". {PARSING_ERR_MSG}$"""), ('2015-04-31', '%Y-%m-%d', f'^day is out of range for month. {PARSING_ERR_MSG}$'), ('2015-31-04', '%Y-%d-%m', f'^day is out of range for month. {PARSING_ERR_MSG}$')])
    def test_day_not_in_month_raise_value(self, cache: Union[str, dict[str, typing.Any], list[str]], arg: Union[str, dict[str, typing.Any], list[str]], format: Union[str, dict[str, typing.Any], list[str]], msg: Union[str, list[str], typing.Callable]) -> None:
        with pytest.raises(ValueError, match=msg):
            to_datetime(arg, errors='raise', format=format, cache=cache)

class TestDatetimeParsingWrappers:

    @pytest.mark.parametrize('date_str, expected', [('2011-01-01', datetime(2011, 1, 1)), ('2Q2005', datetime(2005, 4, 1)), ('2Q05', datetime(2005, 4, 1)), ('2005Q1', datetime(2005, 1, 1)), ('05Q1', datetime(2005, 1, 1)), ('2011Q3', datetime(2011, 7, 1)), ('11Q3', datetime(2011, 7, 1)), ('3Q2011', datetime(2011, 7, 1)), ('3Q11', datetime(2011, 7, 1)), ('2000Q4', datetime(2000, 10, 1)), ('00Q4', datetime(2000, 10, 1)), ('4Q2000', datetime(2000, 10, 1)), ('4Q00', datetime(2000, 10, 1)), ('2000q4', datetime(2000, 10, 1)), ('2000-Q4', datetime(2000, 10, 1)), ('00-Q4', datetime(2000, 10, 1)), ('4Q-2000', datetime(2000, 10, 1)), ('4Q-00', datetime(2000, 10, 1)), ('00q4', datetime(2000, 10, 1)), ('2005', datetime(2005, 1, 1)), ('2005-11', datetime(2005, 11, 1)), ('2005 11', datetime(2005, 11, 1)), ('11-2005', datetime(2005, 11, 1)), ('11 2005', datetime(2005, 11, 1)), ('200511', datetime(2020, 5, 11)), ('20051109', datetime(2005, 11, 9)), ('20051109 10:15', datetime(2005, 11, 9, 10, 15)), ('20051109 08H', datetime(2005, 11, 9, 8, 0)), ('2005-11-09 10:15', datetime(2005, 11, 9, 10, 15)), ('2005-11-09 08H', datetime(2005, 11, 9, 8, 0)), ('2005/11/09 10:15', datetime(2005, 11, 9, 10, 15)), ('2005/11/09 10:15:32', datetime(2005, 11, 9, 10, 15, 32)), ('2005/11/09 10:15:32 AM', datetime(2005, 11, 9, 10, 15, 32)), ('2005/11/09 10:15:32 PM', datetime(2005, 11, 9, 22, 15, 32)), ('2005/11/09 08H', datetime(2005, 11, 9, 8, 0)), ('Thu Sep 25 10:36:28 2003', datetime(2003, 9, 25, 10, 36, 28)), ('Thu Sep 25 2003', datetime(2003, 9, 25)), ('Sep 25 2003', datetime(2003, 9, 25)), ('January 1 2014', datetime(2014, 1, 1)), ('2014-06', datetime(2014, 6, 1)), ('06-2014', datetime(2014, 6, 1)), ('2014-6', datetime(2014, 6, 1)), ('6-2014', datetime(2014, 6, 1)), ('20010101 12', datetime(2001, 1, 1, 12)), ('20010101 1234', datetime(2001, 1, 1, 12, 34)), ('20010101 123456', datetime(2001, 1, 1, 12, 34, 56))])
    def test_parsers(self, date_str: Union[str, float], expected: Union[int, float, pandas.DataFrame], cache: Union[datetime.datetime, pandas.DatetimeIndex]) -> None:
        yearfirst = True
        result1, reso_attrname = parsing.parse_datetime_string_with_reso(date_str, yearfirst=yearfirst)
        reso = {'nanosecond': 'ns', 'microsecond': 'us', 'millisecond': 'ms', 'second': 's'}.get(reso_attrname, 's')
        result2 = to_datetime(date_str, yearfirst=yearfirst)
        result3 = to_datetime([date_str], yearfirst=yearfirst)
        result4 = to_datetime(np.array([date_str], dtype=object), yearfirst=yearfirst, cache=cache)
        result6 = DatetimeIndex([date_str], yearfirst=yearfirst)
        result8 = DatetimeIndex(Index([date_str]), yearfirst=yearfirst)
        result9 = DatetimeIndex(Series([date_str]), yearfirst=yearfirst)
        for res in [result1, result2]:
            assert res == expected
        for res in [result3, result4, result6, result8, result9]:
            exp = DatetimeIndex([Timestamp(expected)]).as_unit(reso)
            tm.assert_index_equal(res, exp)
        if not yearfirst:
            result5 = Timestamp(date_str)
            assert result5 == expected
            result7 = date_range(date_str, freq='S', periods=1, yearfirst=yearfirst)
            assert result7 == expected

    def test_na_values_with_cache(self, cache: Union[bool, list[str], dict], unique_nulls_fixture: Union[bool, list[str], dict], unique_nulls_fixture2: Union[bool, list[str], dict]) -> None:
        expected = Index([NaT, NaT], dtype='datetime64[s]')
        result = to_datetime([unique_nulls_fixture, unique_nulls_fixture2], cache=cache)
        tm.assert_index_equal(result, expected)

    def test_parsers_nat(self) -> None:
        result1, _ = parsing.parse_datetime_string_with_reso('NaT')
        result2 = to_datetime('NaT')
        result3 = Timestamp('NaT')
        result4 = DatetimeIndex(['NaT'])[0]
        assert result1 is NaT
        assert result2 is NaT
        assert result3 is NaT
        assert result4 is NaT

    @pytest.mark.parametrize('date_str, dayfirst, yearfirst, expected', [('10-11-12', False, False, datetime(2012, 10, 11)), ('10-11-12', True, False, datetime(2012, 11, 10)), ('10-11-12', False, True, datetime(2010, 11, 12)), ('10-11-12', True, True, datetime(2010, 12, 11)), ('20/12/21', False, False, datetime(2021, 12, 20)), ('20/12/21', True, False, datetime(2021, 12, 20)), ('20/12/21', False, True, datetime(2020, 12, 21)), ('20/12/21', True, True, datetime(2020, 12, 21)), ('20201012', True, False, datetime(2020, 12, 10))])
    def test_parsers_dayfirst_yearfirst(self, cache: str, date_str: Union[str, None], dayfirst: Union[str, None], yearfirst: Union[str, None], expected: str) -> None:
        dateutil_result = parse(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        assert dateutil_result == expected
        result1, _ = parsing.parse_datetime_string_with_reso(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        if not dayfirst and (not yearfirst):
            result2 = Timestamp(date_str)
            assert result2 == expected
        result3 = to_datetime(date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache)
        result4 = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]
        assert result1 == expected
        assert result3 == expected
        assert result4 == expected

    @pytest.mark.parametrize('date_str, exp_def', [['10:15', datetime(1, 1, 1, 10, 15)], ['9:05', datetime(1, 1, 1, 9, 5)]])
    def test_parsers_timestring(self, date_str: str, exp_def: Union[str, typing.Type, float]) -> None:
        exp_now = parse(date_str)
        result1, _ = parsing.parse_datetime_string_with_reso(date_str)
        result2 = to_datetime(date_str)
        result3 = to_datetime([date_str])
        result4 = Timestamp(date_str)
        result5 = DatetimeIndex([date_str])[0]
        assert result1 == exp_def
        assert result2 == exp_now
        assert result3 == exp_now
        assert result4 == exp_now
        assert result5 == exp_now

    @pytest.mark.parametrize('dt_string, tz, dt_string_repr', [('2013-01-01 05:45+0545', timezone(timedelta(minutes=345)), "Timestamp('2013-01-01 05:45:00+0545', tz='UTC+05:45')"), ('2013-01-01 05:30+0530', timezone(timedelta(minutes=330)), "Timestamp('2013-01-01 05:30:00+0530', tz='UTC+05:30')")])
    def test_parsers_timezone_minute_offsets_roundtrip(self, cache: Union[datetime, dict, dict[str, typing.Any]], dt_string: Union[str, int, dict], tz: Union[str, None, typing.Mapping], dt_string_repr: Union[str, int, dict]) -> None:
        base = to_datetime('2013-01-01 00:00:00', cache=cache)
        base = base.tz_localize('UTC').tz_convert(tz)
        dt_time = to_datetime(dt_string, cache=cache)
        assert base == dt_time
        assert dt_string_repr == repr(dt_time)

@pytest.fixture(params=['D', 's', 'ms', 'us', 'ns'])
def units(request: Any):
    """Day and some time units.

    * D
    * s
    * ms
    * us
    * ns
    """
    return request.param

@pytest.fixture
def julian_dates():
    return date_range('2014-1-1', periods=10).to_julian_date().values

class TestOrigin:

    def test_origin_and_unit(self) -> None:
        ts = to_datetime(1, unit='s', origin=1)
        expected = Timestamp('1970-01-01 00:00:02')
        assert ts == expected
        ts = to_datetime(1, unit='s', origin=1000000000)
        expected = Timestamp('2001-09-09 01:46:41')
        assert ts == expected

    def test_julian(self, julian_dates: Union[datetime.datetime, datetime.date]) -> None:
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

    def test_invalid_unit(self, units: Union[str, datetime.timezone, int], julian_dates: Union[str, datetime.timezone, int]) -> None:
        if units != 'D':
            msg = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates, unit=units, origin='julian')

    @pytest.mark.parametrize('unit', ['ns', 'D'])
    def test_invalid_origin(self, unit: Union[datetime.date, float, tuple[datetime.datetime]]) -> None:
        msg = 'it must be numeric with a unit specified'
        with pytest.raises(ValueError, match=msg):
            to_datetime('2005-01-01', origin='1960-01-01', unit=unit)

    @pytest.mark.parametrize('epochs', [Timestamp(1960, 1, 1), datetime(1960, 1, 1), '1960-01-01', np.datetime64('1960-01-01')])
    def test_epoch(self, units: Union[int, datetime.date, float, None], epochs: Union[float, int, datetime.datetime.datetime]) -> None:
        epoch_1960 = Timestamp(1960, 1, 1)
        units_from_epochs = np.arange(5, dtype=np.int64)
        expected = Series([pd.Timedelta(x, unit=units) + epoch_1960 for x in units_from_epochs])
        result = Series(to_datetime(units_from_epochs, unit=units, origin=epochs))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('origin, exc', [('random_string', ValueError), ('epoch', ValueError), ('13-24-1990', ValueError), (datetime(1, 1, 1), OutOfBoundsDatetime)])
    def test_invalid_origins(self, origin: Union[str, int, typing.Mapping], exc: Union[str, typing.Callable], units: Union[int, str, float]) -> None:
        msg = '|'.join([f'origin {origin} is Out of Bounds', f'origin {origin} cannot be converted to a Timestamp', "Cannot cast .* to unit='ns' without overflow"])
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
    def test_to_datetime_out_of_bounds_with_format_arg(self, format: Union[str, list[str]], warning: str) -> None:
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

    @pytest.mark.parametrize('arg, origin, expected_str', [[200 * 365, 'unix', '2169-11-13 00:00:00'], [200 * 365, '1870-01-01', '2069-11-13 00:00:00'], [300 * 365, '1870-01-01', '2169-10-20 00:00:00']])
    def test_processing_order(self, arg: Union[str, list[str]], origin: Union[str, list, int], expected_str: Union[str, float, datetime.datetime.datetime]) -> None:
        result = to_datetime(arg, unit='D', origin=origin)
        expected = Timestamp(expected_str)
        assert result == expected
        result = to_datetime(200 * 365, unit='D', origin='1870-01-01')
        expected = Timestamp('2069-11-13 00:00:00')
        assert result == expected
        result = to_datetime(300 * 365, unit='D', origin='1870-01-01')
        expected = Timestamp('2169-10-20 00:00:00')
        assert result == expected

    @pytest.mark.parametrize('offset,utc,exp', [['Z', True, '2019-01-01T00:00:00.000Z'], ['Z', None, '2019-01-01T00:00:00.000Z'], ['-01:00', True, '2019-01-01T01:00:00.000Z'], ['-01:00', None, '2019-01-01T00:00:00.000-01:00']])
    def test_arg_tz_ns_unit(self, offset: Any, utc: Union[int, datetime.timedelta, datetime.date.time.datetime], exp: Union[datetime.timedelta, int, datetime.datetime]) -> None:
        arg = '2019-01-01T00:00:00.000' + offset
        result = to_datetime([arg], unit='ns', utc=utc)
        expected = to_datetime([exp]).as_unit('ns')
        tm.assert_index_equal(result, expected)

class TestShouldCache:

    @pytest.mark.parametrize('listlike,do_caching', [([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], False), ([1, 1, 1, 1, 4, 5, 6, 7, 8, 9], True)])
    def test_should_cache(self, listlike: Any, do_caching: Any) -> None:
        assert tools.should_cache(listlike, check_count=len(listlike), unique_share=0.7) == do_caching

    @pytest.mark.parametrize('unique_share,check_count, err_message', [(0.5, 11, 'check_count must be in next bounds: \\[0; len\\(arg\\)\\]'), (10, 2, 'unique_share must be in next bounds: \\(0; 1\\)')])
    def test_should_cache_errors(self, unique_share: Union[bool, str, None, dict], check_count: Union[bool, str, None, dict], err_message: Union[str, bool]) -> None:
        arg = [5] * 10
        with pytest.raises(AssertionError, match=err_message):
            tools.should_cache(arg, unique_share, check_count)

    @pytest.mark.parametrize('listlike', [deque([Timestamp('2010-06-02 09:30:00')] * 51), [Timestamp('2010-06-02 09:30:00')] * 51, tuple([Timestamp('2010-06-02 09:30:00')] * 51)])
    def test_no_slicing_errors_in_should_cache(self, listlike: Union[bool, tests.basilisp.helpers.CompileFn]) -> None:
        assert tools.should_cache(listlike) is True

def test_nullable_integer_to_datetime() -> None:
    ser = Series([1, 2, None, 2 ** 61, None])
    ser = ser.astype('Int64')
    ser_copy = ser.copy()
    res = to_datetime(ser, unit='ns')
    expected = Series([np.datetime64('1970-01-01 00:00:00.000000001'), np.datetime64('1970-01-01 00:00:00.000000002'), np.datetime64('NaT'), np.datetime64('2043-01-25 23:56:49.213693952'), np.datetime64('NaT')])
    tm.assert_series_equal(res, expected)
    tm.assert_series_equal(ser, ser_copy)

@pytest.mark.parametrize('klass', [np.array, list])
def test_na_to_datetime(nulls_fixture: Union[str, list['RunnerState'], dict, None], klass: Union[str, dict]) -> None:
    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match='not convertible to datetime'):
            to_datetime(klass([nulls_fixture]))
    else:
        result = to_datetime(klass([nulls_fixture]))
        assert result[0] is NaT

@pytest.mark.parametrize('errors', ['raise', 'coerce'])
@pytest.mark.parametrize('args, format', [(['03/24/2016', '03/25/2016', ''], '%m/%d/%Y'), (['2016-03-24', '2016-03-25', ''], '%Y-%m-%d')], ids=['non-ISO8601', 'ISO8601'])
def test_empty_string_datetime(errors: Union[str, None, int, datetime.datetime], args: Any, format: Union[str, None, int, datetime.datetime]) -> None:
    td = Series(args)
    result = to_datetime(td, format=format, errors=errors)
    expected = Series(['2016-03-24', '2016-03-25', NaT], dtype='datetime64[s]')
    tm.assert_series_equal(expected, result)

def test_empty_string_datetime_coerce__unit() -> None:
    result = to_datetime([1, ''], unit='s', errors='coerce')
    expected = DatetimeIndex(['1970-01-01 00:00:01', 'NaT'], dtype='datetime64[ns]')
    tm.assert_index_equal(expected, result)
    result = to_datetime([1, ''], unit='s', errors='raise')
    tm.assert_index_equal(expected, result)

def test_to_datetime_monotonic_increasing_index(cache: Union[datetime.datetime, dict[str, typing.Any]]) -> None:
    cstart = start_caching_at
    times = date_range(Timestamp('1980'), periods=cstart, freq='YS')
    times = times.to_frame(index=False, name='DT').sample(n=cstart, random_state=1)
    times.index = times.index.to_series().astype(float) / 1000
    result = to_datetime(times.iloc[:, 0], cache=cache)
    expected = times.iloc[:, 0]
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('series_length', [40, start_caching_at, start_caching_at + 1, start_caching_at + 5])
def test_to_datetime_cache_coerce_50_lines_outofbounds(series_length: Union[int, datetime.datetime]) -> None:
    ser = Series([datetime.fromisoformat('1446-04-12 00:00:00+00:00')] + [datetime.fromisoformat('1991-10-20 00:00:00+00:00')] * series_length, dtype=object)
    result1 = to_datetime(ser, errors='coerce', utc=True)
    expected1 = Series([Timestamp(x) for x in ser])
    assert expected1.dtype == 'M8[us, UTC]'
    tm.assert_series_equal(result1, expected1)
    result3 = to_datetime(ser, errors='raise', utc=True)
    tm.assert_series_equal(result3, expected1)

def test_to_datetime_format_f_parse_nanos() -> None:
    timestamp = '15/02/2020 02:03:04.123456789'
    timestamp_format = '%d/%m/%Y %H:%M:%S.%f'
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
def test_to_datetime_mixed_or_iso_exact(exact: Union[str, typing.Pattern, None], format: Union[str, typing.Pattern, None]) -> None:
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
    dtstr = '2014 Jan 9 05:15 FAKE'
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
def test_to_datetime_mixed_awareness_mixed_types(aware_val: Union[float, None, str], naive_val: Union[bool, float, str], naive_first: Union[datetime.date, None, bool, datetime.timedelta]) -> None:
    vals = [aware_val, naive_val, '']
    vec = vals
    if naive_first:
        vec = [naive_val, aware_val, '']
    both_strs = isinstance(aware_val, str) and isinstance(naive_val, str)
    has_numeric = isinstance(naive_val, (int, float))
    both_datetime = isinstance(naive_val, datetime) and isinstance(aware_val, datetime)
    mixed_msg = "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone"
    first_non_null = next((x for x in vec if x != ''))
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