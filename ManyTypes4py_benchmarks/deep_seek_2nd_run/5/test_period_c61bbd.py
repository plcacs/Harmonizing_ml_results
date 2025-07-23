from datetime import date, datetime, timedelta
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import DAYS, MONTHS
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import NaT, Period, Timedelta, Timestamp, offsets
import pandas._testing as tm

bday_msg: str = 'Period with BDay freq is deprecated'

class TestPeriodDisallowedFreqs:

    @pytest.mark.parametrize('freq, freq_msg', [(offsets.BYearBegin(), 'BYearBegin'), (offsets.YearBegin(2), 'YearBegin'), (offsets.QuarterBegin(startingMonth=12), 'QuarterBegin'), (offsets.BusinessMonthEnd(2), 'BusinessMonthEnd')])
    def test_offsets_not_supported(self, freq: offsets.BaseOffset, freq_msg: str) -> None:
        msg: str = re.escape(f'{freq} is not supported as period frequency')
        with pytest.raises(ValueError, match=msg):
            Period(year=2014, freq=freq)

    def test_custom_business_day_freq_raises(self) -> None:
        msg: str = 'C is not supported as period frequency'
        with pytest.raises(ValueError, match=msg):
            Period('2023-04-10', freq='C')
        msg = f'{offsets.CustomBusinessDay().base} is not supported as period frequency'
        with pytest.raises(ValueError, match=msg):
            Period('2023-04-10', freq=offsets.CustomBusinessDay())

    def test_invalid_frequency_error_message(self) -> None:
        msg: str = 'WOM-1MON is not supported as period frequency'
        with pytest.raises(ValueError, match=msg):
            Period('2012-01-02', freq='WOM-1MON')

    def test_invalid_frequency_period_error_message(self) -> None:
        msg: str = 'Invalid frequency: ME'
        with pytest.raises(ValueError, match=msg):
            Period('2012-01-02', freq='ME')

class TestPeriodConstruction:

    def test_from_td64nat_raises(self) -> None:
        td: np.ndarray = NaT.to_numpy('m8[ns]')
        msg: str = 'Value must be Period, string, integer, or datetime'
        with pytest.raises(ValueError, match=msg):
            Period(td)
        with pytest.raises(ValueError, match=msg):
            Period(td, freq='D')

    def test_construction(self) -> None:
        i1: Period = Period('1/1/2005', freq='M')
        i2: Period = Period('Jan 2005')
        assert i1 == i2
        i1 = Period('2005', freq='Y')
        i2 = Period('2005')
        assert i1 == i2
        i4: Period = Period('2005', freq='M')
        assert i1 != i4
        i1 = Period.now(freq='Q')
        i2 = Period(datetime.now(), freq='Q')
        assert i1 == i2
        i1 = Period.now(freq='D')
        i2 = Period(datetime.now(), freq='D')
        i3 = Period.now(offsets.Day())
        assert i1 == i2
        assert i1 == i3
        i1 = Period('1982', freq='min')
        msg: str = "'MIN' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            i2 = Period('1982', freq='MIN')
        assert i1 == i2
        i1 = Period(year=2005, month=3, day=1, freq='D')
        i2 = Period('3/1/2005', freq='D')
        assert i1 == i2
        msg = "'d' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            i3 = Period(year=2005, month=3, day=1, freq='d')
        assert i1 == i3
        i1 = Period('2007-01-01 09:00:00.001')
        expected: Period = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq='ms')
        assert i1 == expected
        expected = Period('2007-01-01 09:00:00.001', freq='ms')
        assert i1 == expected
        i1 = Period('2007-01-01 09:00:00.00101')
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq='us')
        assert i1 == expected
        expected = Period('2007-01-01 09:00:00.00101', freq='us')
        assert i1 == expected
        msg = 'Must supply freq for ordinal value'
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=200701)
        msg = 'Invalid frequency: X'
        with pytest.raises(ValueError, match=msg):
            Period('2007-1-1', freq='X')

    def test_tuple_freq_disallowed(self) -> None:
        with pytest.raises(TypeError, match='pass as a string instead'):
            Period('1982', freq=('Min', 1))
        with pytest.raises(TypeError, match='pass as a string instead'):
            Period('2006-12-31', ('w', 1))

    def test_construction_from_timestamp_nanos(self) -> None:
        ts: Timestamp = Timestamp('2022-04-20 09:23:24.123456789')
        per: Period = Period(ts, freq='ns')
        rt: Timestamp = per.to_timestamp()
        assert rt == ts
        dt64: np.ndarray = ts.asm8
        per2: Period = Period(dt64, freq='ns')
        rt2: Timestamp = per2.to_timestamp()
        assert rt2.asm8 == dt64

    def test_construction_bday(self) -> None:
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            i1: Period = Period('3/10/12', freq='B')
            i2: Period = Period('3/10/12', freq='D')
            assert i1 == i2.asfreq('B')
            i2 = Period('3/11/12', freq='D')
            assert i1 == i2.asfreq('B')
            i2 = Period('3/12/12', freq='D')
            assert i1 == i2.asfreq('B')
            i3: Period = Period('3/10/12', freq='b')
            assert i1 == i3
            i1 = Period(year=2012, month=3, day=10, freq='B')
            i2 = Period('3/12/12', freq='B')
            assert i1 == i2

    def test_construction_quarter(self) -> None:
        i1: Period = Period(year=2005, quarter=1, freq='Q')
        i2: Period = Period('1/1/2005', freq='Q')
        assert i1 == i2
        i1 = Period(year=2005, quarter=3, freq='Q')
        i2 = Period('9/1/2005', freq='Q')
        assert i1 == i2
        i1 = Period('2005Q1')
        i2 = Period(year=2005, quarter=1, freq='Q')
        i3: Period = Period('2005q1')
        assert i1 == i2
        assert i1 == i3
        i1 = Period('05Q1')
        assert i1 == i2
        lower: Period = Period('05q1')
        assert i1 == lower
        i1 = Period('1Q2005')
        assert i1 == i2
        lower = Period('1q2005')
        assert i1 == lower
        i1 = Period('1Q05')
        assert i1 == i2
        lower = Period('1q05')
        assert i1 == lower
        i1 = Period('4Q1984')
        assert i1.year == 1984
        lower = Period('4q1984')
        assert i1 == lower

    def test_construction_month(self) -> None:
        expected: Period = Period('2007-01', freq='M')
        i1: Period = Period('200701', freq='M')
        assert i1 == expected
        i1 = Period('200701', freq='M')
        assert i1 == expected
        i1 = Period(200701, freq='M')
        assert i1 == expected
        i1 = Period(ordinal=200701, freq='M')
        assert i1.year == 18695
        i1 = Period(datetime(2007, 1, 1), freq='M')
        i2: Period = Period('200701', freq='M')
        assert i1 == i2
        i1 = Period(date(2007, 1, 1), freq='M')
        i2 = Period(datetime(2007, 1, 1), freq='M')
        i3: Period = Period(np.datetime64('2007-01-01'), freq='M')
        i4: Period = Period('2007-01-01 00:00:00', freq='M')
        i5: Period = Period('2007-01-01 00:00:00.000', freq='M')
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

    def test_period_constructor_offsets(self) -> None:
        assert Period('1/1/2005', freq=offsets.MonthEnd()) == Period('1/1/2005', freq='M')
        assert Period('2005', freq=offsets.YearEnd()) == Period('2005', freq='Y')
        assert Period('2005', freq=offsets.MonthEnd()) == Period('2005', freq='M')
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period('3/10/12', freq=offsets.BusinessDay()) == Period('3/10/12', freq='B')
        assert Period('3/10/12', freq=offsets.Day()) == Period('3/10/12', freq='D')
        assert Period(year=2005, quarter=1, freq=offsets.QuarterEnd(startingMonth=12)) == Period(year=2005, quarter=1, freq='Q')
        assert Period(year=2005, quarter=2, freq=offsets.QuarterEnd(startingMonth=12)) == Period(year=2005, quarter=2, freq='Q')
        assert Period(year=2005, month=3, day=1, freq=offsets.Day()) == Period(year=2005, month=3, day=1, freq='D')
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay()) == Period(year=2012, month=3, day=10, freq='B')
        expected: Period = Period('2005-03-01', freq='3D')
        assert Period(year=2005, month=3, day=1, freq=offsets.Day(3)) == expected
        assert Period(year=2005, month=3, day=1, freq='3D') == expected
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay(3)) == Period(year=2012, month=3, day=10, freq='3B')
        assert Period(200701, freq=offsets.MonthEnd()) == Period(200701, freq='M')
        i1: Period = Period(ordinal=200701, freq=offsets.MonthEnd())
        i2: Period = Period(ordinal=200701, freq='M')
        assert i1 == i2
        assert i1.year == 18695
        assert i2.year == 18695
        i1 = Period(datetime(2007, 1, 1), freq='M')
        i2 = Period('200701', freq='M')
        assert i1 == i2
        i1 = Period(date(2007, 1, 1), freq='M')
        i2 = Period(datetime(2007, 1, 1), freq='M')
        i3: Period = Period(np.datetime64('2007-01-01'), freq='M')
        i4: Period = Period('2007-01-01 00:00:00', freq='M')
        i5: Period = Period('2007-01-01 00:00:00.000', freq='M')
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5
        i1 = Period('2007-01-01 09:00:00.001')
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq='ms')
        assert i1 == expected
        expected = Period('2007-01-01 09:00:00.001', freq='ms')
        assert i1 == expected
        i1 = Period('2007-01-01 09:00:00.00101')
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq='us')
        assert i1 == expected
        expected = Period('2007-01-01 09:00:00.00101', freq='us')
        assert i1 == expected

    def test_invalid_arguments(self) -> None:
        msg: str = 'Must supply freq for datetime value'
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now())
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now().date())
        msg = 'Value must be Period, string, integer, or datetime'
        with pytest.raises(ValueError, match=msg):
            Period(1.6, freq='D')
        msg = 'Ordinal must be an integer'
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1.6, freq='D')
        msg = 'Only value or ordinal but not both should be given but not both'
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=2, value=1, freq='D')
        msg = 'If value is None, freq cannot be None'
        with pytest.raises(ValueError, match=msg):
            Period(month=1)
        msg = '^Given date string "-2000" not likely a datetime$'
        with pytest.raises(ValueError, match=msg):
            Period('-2000', 'Y')
        msg = 'day is out of range for month'
        with pytest.raises(DateParseError, match=msg):
            Period('0', 'Y')
        msg = 'Unknown datetime string format, unable to parse'
        with pytest.raises(DateParseError, match=msg):
            Period('1/1/-2000', 'Y')

    def test_constructor_corner(self) -> None:
        expected: Period = Period('2007-01', freq='2M')
        assert Period(year=2007, month=1, freq='2M') == expected
        assert Period(None) is NaT
        p: Period = Period('2007-01-01', freq='D')
        result: Period = Period(p, freq='Y')
        exp: Period = Period('2007', freq='Y')
        assert result == exp

    def test_constructor_infer_freq(self) -> None:
        p: Period = Period('2007-01-01')
        assert p.freq == 'D'
        p = Period('2007-01-01 07')
        assert p.freq == 'h'
        p = Period('2007-01-01 07:10')
        assert p.freq == 'min'
        p = Period('2007-01-01 07:10:15')
        assert p.freq == 's'
        p = Period('2007-01-01 07:10:15.123')
        assert p.freq == 'ms'
        p = Period('2007-01-01 07:10:15.123000')
        assert p.freq == 'us'
        p = Period('2007-01-01 07:10:15.123400')
        assert p.freq == 'us'

    def test_multiples(self) -> None:
        result1: Period = Period('1989', freq='2Y')
        result2: Period = Period('1989', freq='Y')
        assert result1.ordinal == result2.ordinal
        assert result1.freqstr == '2Y-DEC'
        assert result2.freqstr == 'Y-DEC'
        assert result1.freq == offsets.YearEnd(2)
        assert result2.freq == offsets.YearEnd()
        assert (result1 + 1).ordinal == result1.ordinal + 2
        assert (1 + result1).ordinal == result1.ordinal + 2
        assert (result1 - 1).ordinal == result2.ordinal - 2