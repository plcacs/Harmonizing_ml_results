from __future__ import annotations
from datetime import timedelta
import re
import zoneinfo
from dateutil.tz import gettz
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import NaT, Timestamp
from typing import Union

Timezone = Union[str, zoneinfo.ZoneInfo]

class TestTimestampTZLocalize:

    @pytest.mark.skip_ubsan
    def test_tz_localize_pushes_out_of_bounds(self) -> None:
        pytz = pytest.importorskip('pytz')
        msg = f"Converting {Timestamp.min.strftime('%Y-%m-%d %H:%M:%S')} underflows past {Timestamp.min}"
        pac = Timestamp.min.tz_localize(pytz.timezone('US/Pacific'))
        assert pac._value > Timestamp.min._value
        pac.tz_convert('Asia/Tokyo')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.tz_localize(pytz.timezone('Asia/Tokyo'))
        msg = f"Converting {Timestamp.max.strftime('%Y-%m-%d %H:%M:%S')} overflows past {Timestamp.max}"
        tokyo = Timestamp.max.tz_localize(pytz.timezone('Asia/Tokyo'))
        assert tokyo._value < Timestamp.max._value
        tokyo.tz_convert('US/Pacific')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.tz_localize(pytz.timezone('US/Pacific'))

    @pytest.mark.parametrize('tz', [zoneinfo.ZoneInfo('US/Central'), 'dateutil/US/Central', 'pytz/US/Central'])
    def test_tz_localize_ambiguous_bool(self, unit: str, tz: Union[str, zoneinfo.ZoneInfo]) -> None:
        if isinstance(tz, str) and tz.startswith('pytz/'):
            pytz = pytest.importorskip('pytz')
            tz = pytz.timezone(tz.removeprefix('pytz/'))
        ts: Timestamp = Timestamp('2015-11-01 01:00:03').as_unit(unit)
        expected0: Timestamp = Timestamp('2015-11-01 01:00:03-0500', tz=tz)
        expected1: Timestamp = Timestamp('2015-11-01 01:00:03-0600', tz=tz)
        msg = 'Cannot infer dst time from 2015-11-01 01:00:03'
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz)
        result: Timestamp = ts.tz_localize(tz, ambiguous=True)
        assert result == expected0
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = ts.tz_localize(tz, ambiguous=False)
        assert result == expected1
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value

    def test_tz_localize_ambiguous(self) -> None:
        ts: Timestamp = Timestamp('2014-11-02 01:00')
        ts_dst: Timestamp = ts.tz_localize('US/Eastern', ambiguous=True)
        ts_no_dst: Timestamp = ts.tz_localize('US/Eastern', ambiguous=False)
        assert ts_no_dst._value - ts_dst._value == 3600
        msg = re.escape("'ambiguous' parameter must be one of: True, False, 'NaT', 'raise' (default)")
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize('US/Eastern', ambiguous='infer')
        msg = 'Cannot localize tz-aware Timestamp, use tz_convert for conversions'
        with pytest.raises(TypeError, match=msg):
            Timestamp('2011-01-01', tz='US/Eastern').tz_localize('Asia/Tokyo')
        msg = 'Cannot convert tz-naive Timestamp, use tz_localize to localize'
        with pytest.raises(TypeError, match=msg):
            Timestamp('2011-01-01').tz_convert('Asia/Tokyo')

    @pytest.mark.parametrize('stamp, tz', [
        ('2015-03-08 02:00', 'US/Eastern'),
        ('2015-03-08 02:30', 'US/Pacific'),
        ('2015-03-29 02:00', 'Europe/Paris'),
        ('2015-03-29 02:30', 'Europe/Belgrade')
    ])
    def test_tz_localize_nonexistent(self, stamp: str, tz: str) -> None:
        ts: Timestamp = Timestamp(stamp)
        with pytest.raises(ValueError, match=stamp):
            ts.tz_localize(tz)
        with pytest.raises(ValueError, match=stamp):
            ts.tz_localize(tz, nonexistent='raise')
        assert ts.tz_localize(tz, nonexistent='NaT') is NaT

    @pytest.mark.parametrize('stamp, tz, forward_expected, backward_expected', [
        ('2015-03-29 02:00:00', 'Europe/Warsaw', '2015-03-29 03:00:00', '2015-03-29 01:59:59'),
        ('2023-03-12 02:00:00', 'America/Los_Angeles', '2023-03-12 03:00:00', '2023-03-12 01:59:59'),
        ('2023-03-26 01:00:00', 'Europe/London', '2023-03-26 02:00:00', '2023-03-26 00:59:59'),
        ('2023-03-26 00:00:00', 'Atlantic/Azores', '2023-03-26 01:00:00', '2023-03-25 23:59:59')
    ])
    def test_tz_localize_nonexistent_shift(
        self, stamp: str, tz: str, forward_expected: str, backward_expected: str
    ) -> None:
        ts: Timestamp = Timestamp(stamp)
        forward_ts: Timestamp = ts.tz_localize(tz, nonexistent='shift_forward')
        assert forward_ts == Timestamp(forward_expected, tz=tz)
        backward_ts: Timestamp = ts.tz_localize(tz, nonexistent='shift_backward')
        assert backward_ts == Timestamp(backward_expected, tz=tz)

    def test_tz_localize_ambiguous_raise(self) -> None:
        ts: Timestamp = Timestamp('2015-11-1 01:00')
        msg = 'Cannot infer dst time from 2015-11-01 01:00:00,'
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize('US/Pacific', ambiguous='raise')

    def test_tz_localize_nonexistent_invalid_arg(self, warsaw: Timezone) -> None:
        tz: Timezone = warsaw
        ts: Timestamp = Timestamp('2015-03-29 02:00:00')
        msg = "The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent='foo')

    @pytest.mark.parametrize('stamp', ['2014-02-01 09:00', '2014-07-08 09:00', '2014-11-01 17:00', '2014-11-05 00:00'])
    def test_tz_localize_roundtrip(self, stamp: str, tz_aware_fixture: Timezone) -> None:
        tz: Timezone = tz_aware_fixture
        ts: Timestamp = Timestamp(stamp)
        localized: Timestamp = ts.tz_localize(tz)
        assert localized == Timestamp(stamp, tz=tz)
        msg = 'Cannot localize tz-aware Timestamp'
        with pytest.raises(TypeError, match=msg):
            localized.tz_localize(tz)
        reset: Timestamp = localized.tz_localize(None)
        assert reset == ts
        assert reset.tzinfo is None

    def test_tz_localize_ambiguous_compat(self) -> None:
        pytz = pytest.importorskip('pytz')
        naive: Timestamp = Timestamp('2013-10-27 01:00:00')
        pytz_zone = pytz.timezone('Europe/London')
        dateutil_zone: str = 'dateutil/Europe/London'
        result_pytz: Timestamp = naive.tz_localize(pytz_zone, ambiguous=False)
        result_dateutil: Timestamp = naive.tz_localize(dateutil_zone, ambiguous=False)
        assert result_pytz._value == result_dateutil._value
        assert result_pytz._value == 1382835600
        assert result_pytz.to_pydatetime().tzname() == 'GMT'
        assert result_dateutil.to_pydatetime().tzname() == 'GMT'
        assert str(result_pytz) == str(result_dateutil)
        result_pytz = naive.tz_localize(pytz_zone, ambiguous=True)
        result_dateutil = naive.tz_localize(dateutil_zone, ambiguous=True)
        assert result_pytz._value == result_dateutil._value
        assert result_pytz._value == 1382832000
        assert str(result_pytz) == str(result_dateutil)
        assert result_pytz.to_pydatetime().tzname() == result_dateutil.to_pydatetime().tzname()

    @pytest.mark.parametrize('tz', ['pytz/US/Eastern', gettz('US/Eastern'), zoneinfo.ZoneInfo('US/Eastern'), 'dateutil/US/Eastern'])
    def test_timestamp_tz_localize(self, tz: Union[str, zoneinfo.ZoneInfo]) -> None:
        if isinstance(tz, str) and tz.startswith('pytz/'):
            pytz = pytest.importorskip('pytz')
            tz = pytz.timezone(tz.removeprefix('pytz/'))
        stamp: Timestamp = Timestamp('3/11/2012 04:00')
        result: Timestamp = stamp.tz_localize(tz)
        expected: Timestamp = Timestamp('3/11/2012 04:00', tz=tz)
        assert result.hour == expected.hour
        assert result == expected

    @pytest.mark.parametrize('start_ts, tz, end_ts, shift', [
        ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:00:00', 'forward'],
        ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:59:59.999999999', 'backward'],
        ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 03:20:00', timedelta(hours=1)],
        ['2015-03-29 02:20:00', 'Europe/Warsaw', '2015-03-29 01:20:00', timedelta(hours=-1)],
        ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:00:00', 'forward'],
        ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:59:59.999999999', 'backward'],
        ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 03:33:00', timedelta(hours=1)],
        ['2018-03-11 02:33:00', 'US/Pacific', '2018-03-11 01:33:00', timedelta(hours=-1)]
    ])
    @pytest.mark.parametrize('tz_type', ['', 'dateutil/'])
    def test_timestamp_tz_localize_nonexistent_shift(
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
        ts: Timestamp = Timestamp(start_ts).as_unit(unit)
        result: Timestamp = ts.tz_localize(tz_full, nonexistent=shift)
        expected: Timestamp = Timestamp(end_ts).tz_localize(tz_full)
        if unit == 'us':
            assert result == expected.replace(nanosecond=0)
        elif unit == 'ms':
            micros = expected.microsecond - expected.microsecond % 1000
            assert result == expected.replace(microsecond=micros, nanosecond=0)
        elif unit == 's':
            assert result == expected.replace(microsecond=0, nanosecond=0)
        else:
            assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value

    @pytest.mark.parametrize('offset', [-1, 1])
    def test_timestamp_tz_localize_nonexistent_shift_invalid(self, offset: int, warsaw: Timezone) -> None:
        tz: Timezone = warsaw
        ts: Timestamp = Timestamp('2015-03-29 02:20:00')
        msg = 'The provided timedelta will relocalize on a nonexistent time'
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent=timedelta(seconds=offset))

    def test_timestamp_tz_localize_nonexistent_NaT(self, warsaw: Timezone, unit: str) -> None:
        tz: Timezone = warsaw
        ts: Timestamp = Timestamp('2015-03-29 02:20:00').as_unit(unit)
        result: Timestamp | type(NaT) = ts.tz_localize(tz, nonexistent='NaT')
        assert result is NaT

    def test_timestamp_tz_localize_nonexistent_raise(self, warsaw: Timezone, unit: str) -> None:
        tz: Timezone = warsaw
        ts: Timestamp = Timestamp('2015-03-29 02:20:00').as_unit(unit)
        msg = '2015-03-29 02:20:00'
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent='raise')
        msg = "The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
        with pytest.raises(ValueError, match=msg):
            ts.tz_localize(tz, nonexistent='foo')