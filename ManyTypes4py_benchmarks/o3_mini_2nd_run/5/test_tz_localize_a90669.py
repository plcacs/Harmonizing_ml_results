from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas import DatetimeIndex, Timestamp, bdate_range, date_range, offsets, to_datetime
import pandas._testing as tm
from typing import Any, Union

@pytest.fixture(params=['pytz/US/Eastern', gettz('US/Eastern'), ZoneInfo('US/Eastern')])
def tz(request: pytest.FixtureRequest) -> tzinfo:
    if isinstance(request.param, str) and request.param.startswith('pytz/'):
        pytz = pytest.importorskip('pytz')
        return pytz.timezone(request.param.removeprefix('pytz/'))
    return request.param

class TestTZLocalize:
    def test_tz_localize_invalidates_freq(self) -> None:
        dti = date_range('2014-03-08 23:00', '2014-03-09 09:00', freq='h')
        assert dti.freq == 'h'
        result = dti.tz_localize(None)
        assert result.freq == 'h'
        result = dti.tz_localize('UTC')
        assert result.freq == 'h'
        result = dti.tz_localize('US/Eastern', nonexistent='shift_forward')
        assert result.freq is None
        assert result.inferred_freq is None
        dti2 = dti[:1]
        result = dti2.tz_localize('US/Eastern')
        assert result.freq == 'h'

    def test_tz_localize_utc_copies(self, utc_fixture: tzinfo) -> None:
        times = ['2015-03-08 01:00', '2015-03-08 02:00', '2015-03-08 03:00']
        index = DatetimeIndex(times)
        res = index.tz_localize(utc_fixture)
        assert not tm.shares_memory(res, index)
        res2 = index._data.tz_localize(utc_fixture)
        assert not tm.shares_memory(index._data, res2)

    def test_dti_tz_localize_nonexistent_raise_coerce(self) -> None:
        times = ['2015-03-08 01:00', '2015-03-08 02:00', '2015-03-08 03:00']
        index = DatetimeIndex(times)
        tz_str: str = 'US/Eastern'
        with pytest.raises(ValueError, match='|'.join(times)):
            index.tz_localize(tz=tz_str)
        with pytest.raises(ValueError, match='|'.join(times)):
            index.tz_localize(tz=tz_str, nonexistent='raise')
        result = index.tz_localize(tz=tz_str, nonexistent='NaT')
        test_times = ['2015-03-08 01:00-05:00', 'NaT', '2015-03-08 03:00-04:00']
        dti = to_datetime(test_times, utc=True)
        expected = dti.tz_convert('US/Eastern')
        tm.assert_index_equal(result, expected)

    def test_dti_tz_localize_ambiguous_infer(self, tz: tzinfo) -> None:
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour())
        with pytest.raises(ValueError, match='Cannot infer dst time'):
            dr.tz_localize(tz)

    def test_dti_tz_localize_ambiguous_infer2(self, tz: tzinfo, unit: str) -> None:
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit)
        times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
        di = DatetimeIndex(times).as_unit(unit)
        result = di.tz_localize(tz, ambiguous='infer')
        expected = dr._with_freq(None)
        tm.assert_index_equal(result, expected)
        result2 = DatetimeIndex(times, tz=tz, ambiguous='infer').as_unit(unit)
        tm.assert_index_equal(result2, expected)

    def test_dti_tz_localize_ambiguous_infer3(self, tz: tzinfo) -> None:
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        localized = dr.tz_localize(tz)
        localized_infer = dr.tz_localize(tz, ambiguous='infer')
        tm.assert_index_equal(localized, localized_infer)

    def test_dti_tz_localize_ambiguous_times(self, tz: tzinfo) -> None:
        dr = date_range(datetime(2011, 3, 13, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(ValueError, match='2011-03-13 02:30:00'):
            dr.tz_localize(tz)
        dr = date_range(datetime(2011, 3, 13, 3, 30), periods=3, freq=offsets.Hour(), tz=tz)
        dr = date_range(datetime(2011, 11, 6, 1, 30), periods=3, freq=offsets.Hour())
        with pytest.raises(ValueError, match='Cannot infer dst time'):
            dr.tz_localize(tz)
        dr = date_range(datetime(2011, 3, 13), periods=48, freq=offsets.Minute(30), tz=timezone.utc)

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_dti_tz_localize_pass_dates_to_utc(self, tzstr: str) -> None:
        strdates = ['1/1/2012', '3/1/2012', '4/1/2012']
        idx = DatetimeIndex(strdates)
        conv = idx.tz_localize(tzstr)
        fromdates = DatetimeIndex(strdates, tz=tzstr)
        assert conv.tz == fromdates.tz
        tm.assert_numpy_array_equal(conv.values, fromdates.values)

    @pytest.mark.parametrize('prefix', ['', 'dateutil/'])
    def test_dti_tz_localize(self, prefix: str) -> None:
        tzstr = prefix + 'US/Eastern'
        dti = date_range(start='1/1/2005', end='1/1/2005 0:00:30.256', freq='ms')
        dti2 = dti.tz_localize(tzstr)
        dti_utc = date_range(start='1/1/2005 05:00', end='1/1/2005 5:00:30.256', freq='ms', tz='utc')
        tm.assert_numpy_array_equal(dti2.values, dti_utc.values)
        dti3 = dti2.tz_convert(prefix + 'US/Pacific')
        tm.assert_numpy_array_equal(dti3.values, dti_utc.values)
        dti = date_range(start='11/6/2011 1:59', end='11/6/2011 2:00', freq='ms')
        with pytest.raises(ValueError, match='Cannot infer dst time'):
            dti.tz_localize(tzstr)
        dti = date_range(start='3/13/2011 1:59', end='3/13/2011 2:00', freq='ms')
        with pytest.raises(ValueError, match='2011-03-13 02:00:00'):
            dti.tz_localize(tzstr)

    def test_dti_tz_localize_utc_conversion(self, tz: tzinfo) -> None:
        rng = date_range('3/10/2012', '3/11/2012', freq='30min')
        converted = rng.tz_localize(tz)
        expected_naive = rng + offsets.Hour(5)
        tm.assert_numpy_array_equal(converted.asi8, expected_naive.asi8)
        rng = date_range('3/11/2012', '3/12/2012', freq='30min')
        with pytest.raises(ValueError, match='2012-03-11 02:00:00'):
            rng.tz_localize(tz)

    def test_dti_tz_localize_roundtrip(self, tz_aware_fixture: tzinfo) -> None:
        idx = date_range(start='2014-06-01', end='2014-08-30', freq='15min')
        tz_obj: tzinfo = tz_aware_fixture
        localized = idx.tz_localize(tz_obj)
        with pytest.raises(TypeError, match='Already tz-aware, use tz_convert to convert'):
            localized.tz_localize(tz_obj)
        reset = localized.tz_localize(None)
        assert reset.tzinfo is None
        expected = idx._with_freq(None)
        tm.assert_index_equal(reset, expected)

    def test_dti_tz_localize_naive(self) -> None:
        rng = date_range('1/1/2011', periods=100, freq='h')
        conv = rng.tz_localize('US/Pacific')
        exp = date_range('1/1/2011', periods=100, freq='h', tz='US/Pacific')
        tm.assert_index_equal(conv, exp._with_freq(None))

    def test_dti_tz_localize_tzlocal(self) -> None:
        offset = dateutil.tz.tzlocal().utcoffset(datetime(2011, 1, 1))
        offset_ns = int(offset.total_seconds() * 1000000000)
        dti = date_range(start='2001-01-01', end='2001-03-01')
        dti2 = dti.tz_localize(dateutil.tz.tzlocal())
        tm.assert_numpy_array_equal(dti2.asi8 + offset_ns, dti.asi8)
        dti = date_range(start='2001-01-01', end='2001-03-01', tz=dateutil.tz.tzlocal())
        dti2 = dti.tz_localize(None)
        tm.assert_numpy_array_equal(dti2.asi8 - offset_ns, dti.asi8)

    def test_dti_tz_localize_ambiguous_nat(self, tz: tzinfo) -> None:
        times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
        di = DatetimeIndex(times)
        localized = di.tz_localize(tz, ambiguous='NaT')
        times_test = ['11/06/2011 00:00', np.nan, np.nan, '11/06/2011 02:00', '11/06/2011 03:00']
        di_test = DatetimeIndex(times_test, tz='US/Eastern')
        tm.assert_numpy_array_equal(di_test.values, localized.values)

    def test_dti_tz_localize_ambiguous_flags(self, tz: tzinfo, unit: str) -> None:
        dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit)
        times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
        di = DatetimeIndex(times).as_unit(unit)
        is_dst = [1, 1, 0, 0, 0]
        localized = di.tz_localize(tz, ambiguous=is_dst)
        expected = dr._with_freq(None)
        tm.assert_index_equal(expected, localized)
        result = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(result, expected)
        localized = di.tz_localize(tz, ambiguous=np.array(is_dst))
        tm.assert_index_equal(dr, localized)
        localized = di.tz_localize(tz, ambiguous=np.array(is_dst).astype('bool'))
        tm.assert_index_equal(dr, localized)
        localized = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
        tm.assert_index_equal(dr, localized)
        times_extended = times + times
        di_extended = DatetimeIndex(times_extended).as_unit(unit)
        msg: str = 'Length of ambiguous bool-array must be the same size as vals'
        with pytest.raises(Exception, match=msg):
            di_extended.tz_localize(tz, ambiguous=is_dst)
        is_dst_extended = np.hstack((is_dst, is_dst))
        localized = di_extended.tz_localize(tz, ambiguous=is_dst_extended)
        dr_appended = dr.append(dr)
        tm.assert_index_equal(dr_appended, localized)

    def test_dti_tz_localize_ambiguous_flags2(self, tz: tzinfo) -> None:
        dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
        is_dst = np.array([1] * 10)
        localized = dr.tz_localize(tz)
        localized_is_dst = dr.tz_localize(tz, ambiguous=is_dst)
        tm.assert_index_equal(localized, localized_is_dst)

    def test_dti_tz_localize_bdate_range(self) -> None:
        dr = bdate_range('1/1/2009', '1/1/2010')
        dr_utc = bdate_range('1/1/2009', '1/1/2010', tz=timezone.utc)
        localized = dr.tz_localize(timezone.utc)
        tm.assert_index_equal(dr_utc, localized)

    @pytest.mark.parametrize(
        'start_ts, tz, end_ts, shift',
        [
            ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:00:00', 'forward'],
            ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:59:59.999999999', 'backward'],
            ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:20:00', timedelta(hours=1)],
            ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:20:00', timedelta(hours=-1)],
            ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:00:00', 'forward'],
            ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:59:59.999999999', 'backward'],
            ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:33:00', timedelta(hours=1)],
            ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:33:00', timedelta(hours=-1)]
        ]
    )
    @pytest.mark.parametrize('tz_type', ['', 'dateutil/'])
    def test_dti_tz_localize_nonexistent_shift(
        self,
        start_ts: str,
        tz: str,
        end_ts: str,
        shift: Union[str, timedelta],
        tz_type: str,
        unit: str
    ) -> None:
        tz_full: str = tz_type + tz
        if isinstance(shift, str):
            shift = 'shift_' + shift
        dti = DatetimeIndex([Timestamp(start_ts)]).as_unit(unit)
        result = dti.tz_localize(tz_full, nonexistent=shift)
        expected = DatetimeIndex([Timestamp(end_ts)]).tz_localize(tz_full).as_unit(unit)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('offset', [-1, 1])
    def test_dti_tz_localize_nonexistent_shift_invalid(self, offset: int, warsaw: tzinfo) -> None:
        tz_obj: tzinfo = warsaw
        dti = DatetimeIndex([Timestamp('2015-03-29 02:20:00')])
        msg: str = 'The provided timedelta will relocalize on a nonexistent time'
        with pytest.raises(ValueError, match=msg):
            dti.tz_localize(tz_obj, nonexistent=timedelta(seconds=offset))
