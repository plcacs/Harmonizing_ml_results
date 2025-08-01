#!/usr/bin/env python3
"""
Tests for DateOffset additions over Daylight Savings Time
"""
from datetime import timedelta
from typing import Any, Optional, Tuple, List
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (BMonthBegin, BMonthEnd, BQuarterBegin, BQuarterEnd,
                                          BYearBegin, BYearEnd, CBMonthBegin, CBMonthEnd,
                                          CustomBusinessDay, DateOffset, Day, MonthBegin,
                                          MonthEnd, QuarterBegin, QuarterEnd, SemiMonthBegin,
                                          SemiMonthEnd, Week, YearBegin, YearEnd)
from pandas import DatetimeIndex
import pandas._testing as tm

pytz = pytest.importorskip('pytz')


def get_utc_offset_hours(ts: Timestamp) -> float:
    o = ts.utcoffset()
    return (o.days * 24 * 3600 + o.seconds) / 3600.0


class TestDST:
    ts_pre_fallback: str = '2013-11-03 01:59:59.999999'
    ts_pre_springfwd: str = '2013-03-10 01:59:59.999999'
    timezone_utc_offsets: Any = {
        pytz.timezone('US/Eastern'):
            {'utc_offset_daylight': -4, 'utc_offset_standard': -5},
        'dateutil/US/Pacific':
            {'utc_offset_daylight': -7, 'utc_offset_standard': -8},
    }
    valid_date_offsets_singular: List[str] = ['weekday', 'day', 'hour', 'minute', 'second', 'microsecond']
    valid_date_offsets_plural: List[str] = ['weeks', 'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds']

    def _test_all_offsets(self, n: int, performance_warning: Any, **kwds: Any) -> None:
        valid_offsets: List[str] = self.valid_date_offsets_plural if n > 1 else self.valid_date_offsets_singular
        for name in valid_offsets:
            self._test_offset(offset_name=name, offset_n=n, performance_warning=performance_warning, **kwds)

    def _test_offset(self, offset_name: str, offset_n: int, tstart: Timestamp,
                     expected_utc_offset: Optional[float], performance_warning: Any) -> None:
        offset = DateOffset(**{offset_name: offset_n})
        if (offset_name in ['hour', 'minute', 'second', 'microsecond'] and offset_n == 1 and
                (tstart == Timestamp('2013-11-03 01:59:59.999999-0500', tz=pytz.timezone('US/Eastern')))):
            err_msg = {
                'hour': '2013-11-03 01:59:59.999999',
                'minute': '2013-11-03 01:01:59.999999',
                'second': '2013-11-03 01:59:01.999999',
                'microsecond': '2013-11-03 01:59:59.000001'
            }[offset_name]
            with pytest.raises(ValueError, match=err_msg):
                tstart + offset
            dti: DatetimeIndex = DatetimeIndex([tstart])
            warn_msg = 'Non-vectorized DateOffset'
            with pytest.raises(ValueError, match=err_msg):
                with tm.assert_produces_warning(performance_warning, match=warn_msg):
                    dti + offset
            return
        t: Timestamp = tstart + offset
        if expected_utc_offset is not None:
            assert get_utc_offset_hours(t) == expected_utc_offset
        if offset_name == 'weeks':
            assert t.date() == timedelta(days=7 * offset.kwds['weeks']) + tstart.date()
            assert (t.dayofweek == tstart.dayofweek and t.hour == tstart.hour and
                    t.minute == tstart.minute and t.second == tstart.second)
        elif offset_name == 'days':
            assert timedelta(offset.kwds['days']) + tstart.date() == t.date()
            assert t.hour == tstart.hour and t.minute == tstart.minute and t.second == tstart.second
        elif offset_name in self.valid_date_offsets_singular:
            datepart_offset = getattr(t, offset_name if offset_name != 'weekday' else 'dayofweek')
            assert datepart_offset == offset.kwds[offset_name]
        else:
            assert t == (tstart.tz_convert('UTC') + offset).tz_convert(pytz.timezone('US/Pacific'))

    def _make_timestamp(self, string: str, hrs_offset: int, tz: Any) -> Timestamp:
        if hrs_offset >= 0:
            offset_string = f'{hrs_offset:02d}00'
        else:
            offset_string = f'-{hrs_offset * -1:02}00'
        return Timestamp(string + offset_string).tz_convert(tz)

    def test_springforward_plural(self, performance_warning: Any) -> None:
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre: int = utc_offsets['utc_offset_standard']
            hrs_post: int = utc_offsets['utc_offset_daylight']
            self._test_all_offsets(
                n=3,
                performance_warning=performance_warning,
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                expected_utc_offset=hrs_post
            )

    def test_fallback_singular(self, performance_warning: Any) -> None:
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre: int = utc_offsets['utc_offset_standard']
            self._test_all_offsets(
                n=1,
                performance_warning=performance_warning,
                tstart=self._make_timestamp(self.ts_pre_fallback, hrs_pre, tz),
                expected_utc_offset=None
            )

    def test_springforward_singular(self, performance_warning: Any) -> None:
        for tz, utc_offsets in self.timezone_utc_offsets.items():
            hrs_pre: int = utc_offsets['utc_offset_standard']
            self._test_all_offsets(
                n=1,
                performance_warning=performance_warning,
                tstart=self._make_timestamp(self.ts_pre_springfwd, hrs_pre, tz),
                expected_utc_offset=None
            )

    offset_classes = {
        MonthBegin: ['11/2/2012', '12/1/2012'],
        MonthEnd: ['11/2/2012', '11/30/2012'],
        BMonthBegin: ['11/2/2012', '12/3/2012'],
        BMonthEnd: ['11/2/2012', '11/30/2012'],
        CBMonthBegin: ['11/2/2012', '12/3/2012'],
        CBMonthEnd: ['11/2/2012', '11/30/2012'],
        SemiMonthBegin: ['11/2/2012', '11/15/2012'],
        SemiMonthEnd: ['11/2/2012', '11/15/2012'],
        Week: ['11/2/2012', '11/9/2012'],
        YearBegin: ['11/2/2012', '1/1/2013'],
        YearEnd: ['11/2/2012', '12/31/2012'],
        BYearBegin: ['11/2/2012', '1/1/2013'],
        BYearEnd: ['11/2/2012', '12/31/2012'],
        QuarterBegin: ['11/2/2012', '12/1/2012'],
        QuarterEnd: ['11/2/2012', '12/31/2012'],
        BQuarterBegin: ['11/2/2012', '12/3/2012'],
        BQuarterEnd: ['11/2/2012', '12/31/2012'],
        Day: ['11/4/2012', '11/4/2012 23:00']
    }.items()

    @pytest.mark.parametrize('tup', list(offset_classes))
    def test_all_offset_classes(self, tup: Tuple[Any, List[str]]) -> None:
        offset, test_values = tup
        first: Timestamp = Timestamp(test_values[0], tz='US/Eastern') + offset()
        second: Timestamp = Timestamp(test_values[1], tz='US/Eastern')
        assert first == second


@pytest.mark.parametrize('original_dt, target_dt, offset, tz', [
    (Timestamp('2021-10-01 01:15'), Timestamp('2021-10-31 01:15'), MonthEnd(1), 'Europe/London'),
    (Timestamp('2010-12-05 02:59'), Timestamp('2010-10-31 02:59'), SemiMonthEnd(-3), 'Europe/Paris'),
    (Timestamp('2021-10-31 01:20'), Timestamp('2021-11-07 01:20'), CustomBusinessDay(2, weekmask='Sun Mon'), 'US/Eastern'),
    (Timestamp('2020-04-03 01:30'), Timestamp('2020-11-01 01:30'), YearBegin(1, month=11), 'America/Chicago')
])
def test_nontick_offset_with_ambiguous_time_error(original_dt: Timestamp, target_dt: Timestamp,
                                                   offset: DateOffset, tz: str) -> None:
    localized_dt: Timestamp = original_dt.tz_localize(tz)
    msg: str = f"Cannot infer dst time from {target_dt}, try using the 'ambiguous' argument"
    with pytest.raises(ValueError, match=msg):
        localized_dt + offset
