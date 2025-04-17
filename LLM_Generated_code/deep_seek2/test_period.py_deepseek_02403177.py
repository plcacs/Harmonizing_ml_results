from datetime import date, datetime, timedelta
import re
from typing import List, Tuple, Union

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import DAYS, MONTHS
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG

from pandas import NaT, Period, Timedelta, Timestamp, offsets
import pandas._testing as tm

bday_msg: str = "Period with BDay freq is deprecated"


class TestPeriodDisallowedFreqs:
    @pytest.mark.parametrize(
        "freq, freq_msg",
        [
            (offsets.BYearBegin(), "BYearBegin"),
            (offsets.YearBegin(2), "YearBegin"),
            (offsets.QuarterBegin(startingMonth=12), "QuarterBegin"),
            (offsets.BusinessMonthEnd(2), "BusinessMonthEnd"),
        ],
    )
    def test_offsets_not_supported(self, freq: offsets.BaseOffset, freq_msg: str) -> None:
        # GH#55785
        msg: str = re.escape(f"{freq} is not supported as period frequency")
        with pytest.raises(ValueError, match=msg):
            Period(year=2014, freq=freq)

    def test_custom_business_day_freq_raises(self) -> None:
        # GH#52534
        msg: str = "C is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq="C")
        msg = f"{offsets.CustomBusinessDay().base} is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2023-04-10", freq=offsets.CustomBusinessDay())

    def test_invalid_frequency_error_message(self) -> None:
        msg: str = "WOM-1MON is not supported as period frequency"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="WOM-1MON")

    def test_invalid_frequency_period_error_message(self) -> None:
        msg: str = "Invalid frequency: ME"
        with pytest.raises(ValueError, match=msg):
            Period("2012-01-02", freq="ME")


class TestPeriodConstruction:
    def test_from_td64nat_raises(self) -> None:
        # GH#44507
        td: np.timedelta64 = NaT.to_numpy("m8[ns]")

        msg: str = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(td)

        with pytest.raises(ValueError, match=msg):
            Period(td, freq="D")

    def test_construction(self) -> None:
        i1: Period = Period("1/1/2005", freq="M")
        i2: Period = Period("Jan 2005")

        assert i1 == i2

        # GH#54105 - Period can be confusingly instantiated with lowercase freq
        # TODO: raise in the future an error when passing lowercase freq
        i1 = Period("2005", freq="Y")
        i2 = Period("2005")

        assert i1 == i2

        i4: Period = Period("2005", freq="M")
        assert i1 != i4

        i1 = Period.now(freq="Q")
        i2 = Period(datetime.now(), freq="Q")

        assert i1 == i2

        # Pass in freq as a keyword argument sometimes as a test for
        # https://github.com/pandas-dev/pandas/issues/53369
        i1 = Period.now(freq="D")
        i2 = Period(datetime.now(), freq="D")
        i3 = Period.now(offsets.Day())

        assert i1 == i2
        assert i1 == i3

        i1 = Period("1982", freq="min")
        msg: str = "'MIN' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            i2 = Period("1982", freq="MIN")
        assert i1 == i2

        i1 = Period(year=2005, month=3, day=1, freq="D")
        i2 = Period("3/1/2005", freq="D")
        assert i1 == i2

        msg = "'d' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            i3 = Period(year=2005, month=3, day=1, freq="d")
        assert i1 == i3

        i1 = Period("2007-01-01 09:00:00.001")
        expected: Period = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected

        msg = "Must supply freq for ordinal value"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=200701)

        msg = "Invalid frequency: X"
        with pytest.raises(ValueError, match=msg):
            Period("2007-1-1", freq="X")

    def test_tuple_freq_disallowed(self) -> None:
        # GH#34703 tuple freq disallowed
        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("1982", freq=("Min", 1))

        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("2006-12-31", ("w", 1))

    def test_construction_from_timestamp_nanos(self) -> None:
        # GH#46811 don't drop nanos from Timestamp
        ts: Timestamp = Timestamp("2022-04-20 09:23:24.123456789")
        per: Period = Period(ts, freq="ns")

        # should losslessly round-trip, not lose the 789
        rt: Timestamp = per.to_timestamp()
        assert rt == ts

        # same thing but from a datetime64 object
        dt64: np.datetime64 = ts.asm8
        per2: Period = Period(dt64, freq="ns")
        rt2: Timestamp = per2.to_timestamp()
        assert rt2.asm8 == dt64

    def test_construction_bday(self) -> None:
        # Biz day construction, roll forward if non-weekday
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            i1: Period = Period("3/10/12", freq="B")
            i2: Period = Period("3/10/12", freq="D")
            assert i1 == i2.asfreq("B")
            i2 = Period("3/11/12", freq="D")
            assert i1 == i2.asfreq("B")
            i2 = Period("3/12/12", freq="D")
            assert i1 == i2.asfreq("B")

            i3: Period = Period("3/10/12", freq="b")
            assert i1 == i3

            i1 = Period(year=2012, month=3, day=10, freq="B")
            i2 = Period("3/12/12", freq="B")
            assert i1 == i2

    def test_construction_quarter(self) -> None:
        i1: Period = Period(year=2005, quarter=1, freq="Q")
        i2: Period = Period("1/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period(year=2005, quarter=3, freq="Q")
        i2 = Period("9/1/2005", freq="Q")
        assert i1 == i2

        i1 = Period("2005Q1")
        i2 = Period(year=2005, quarter=1, freq="Q")
        i3: Period = Period("2005q1")
        assert i1 == i2
        assert i1 == i3

        i1 = Period("05Q1")
        assert i1 == i2
        lower: Period = Period("05q1")
        assert i1 == lower

        i1 = Period("1Q2005")
        assert i1 == i2
        lower = Period("1q2005")
        assert i1 == lower

        i1 = Period("1Q05")
        assert i1 == i2
        lower = Period("1q05")
        assert i1 == lower

        i1 = Period("4Q1984")
        assert i1.year == 1984
        lower = Period("4q1984")
        assert i1 == lower

    def test_construction_month(self) -> None:
        expected: Period = Period("2007-01", freq="M")
        i1: Period = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period("200701", freq="M")
        assert i1 == expected

        i1 = Period(200701, freq="M")
        assert i1 == expected

        i1 = Period(ordinal=200701, freq="M")
        assert i1.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3: Period = Period(np.datetime64("2007-01-01"), freq="M")
        i4: Period = Period("2007-01-01 00:00:00", freq="M")
        i5: Period = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

    def test_period_constructor_offsets(self) -> None:
        assert Period("1/1/2005", freq=offsets.MonthEnd()) == Period(
            "1/1/2005", freq="M"
        )
        assert Period("2005", freq=offsets.YearEnd()) == Period("2005", freq="Y")
        assert Period("2005", freq=offsets.MonthEnd()) == Period("2005", freq="M")
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period("3/10/12", freq=offsets.BusinessDay()) == Period(
                "3/10/12", freq="B"
            )
        assert Period("3/10/12", freq=offsets.Day()) == Period("3/10/12", freq="D")

        assert Period(
            year=2005, quarter=1, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=1, freq="Q")
        assert Period(
            year=2005, quarter=2, freq=offsets.QuarterEnd(startingMonth=12)
        ) == Period(year=2005, quarter=2, freq="Q")

        assert Period(year=2005, month=3, day=1, freq=offsets.Day()) == Period(
            year=2005, month=3, day=1, freq="D"
        )
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay()) == Period(
                year=2012, month=3, day=10, freq="B"
            )

        expected: Period = Period("2005-03-01", freq="3D")
        assert Period(year=2005, month=3, day=1, freq=offsets.Day(3)) == expected
        assert Period(year=2005, month=3, day=1, freq="3D") == expected

        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            assert Period(year=2012, month=3, day=10, freq=offsets.BDay(3)) == Period(
                year=2012, month=3, day=10, freq="3B"
            )

        assert Period(200701, freq=offsets.MonthEnd()) == Period(200701, freq="M")

        i1: Period = Period(ordinal=200701, freq=offsets.MonthEnd())
        i2: Period = Period(ordinal=200701, freq="M")
        assert i1 == i2
        assert i1.year == 18695
        assert i2.year == 18695

        i1 = Period(datetime(2007, 1, 1), freq="M")
        i2 = Period("200701", freq="M")
        assert i1 == i2

        i1 = Period(date(2007, 1, 1), freq="M")
        i2 = Period(datetime(2007, 1, 1), freq="M")
        i3: Period = Period(np.datetime64("2007-01-01"), freq="M")
        i4: Period = Period("2007-01-01 00:00:00", freq="M")
        i5: Period = Period("2007-01-01 00:00:00.000", freq="M")
        assert i1 == i2
        assert i1 == i3
        assert i1 == i4
        assert i1 == i5

        i1 = Period("2007-01-01 09:00:00.001")
        expected: Period = Period(datetime(2007, 1, 1, 9, 0, 0, 1000), freq="ms")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.001", freq="ms")
        assert i1 == expected

        i1 = Period("2007-01-01 09:00:00.00101")
        expected = Period(datetime(2007, 1, 1, 9, 0, 0, 1010), freq="us")
        assert i1 == expected

        expected = Period("2007-01-01 09:00:00.00101", freq="us")
        assert i1 == expected

    def test_invalid_arguments(self) -> None:
        msg: str = "Must supply freq for datetime value"
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now())
        with pytest.raises(ValueError, match=msg):
            Period(datetime.now().date())

        msg = "Value must be Period, string, integer, or datetime"
        with pytest.raises(ValueError, match=msg):
            Period(1.6, freq="D")
        msg = "Ordinal must be an integer"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1.6, freq="D")
        msg = "Only value or ordinal but not both should be given but not both"
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=2, value=1, freq="D")

        msg = "If value is None, freq cannot be None"
        with pytest.raises(ValueError, match=msg):
            Period(month=1)

        msg = '^Given date string "-2000" not likely a datetime$'
        with pytest.raises(ValueError, match=msg):
            Period("-2000", "Y")
        msg = "day is out of range for month"
        with pytest.raises(DateParseError, match=msg):
            Period("0", "Y")
        msg = "Unknown datetime string format, unable to parse"
        with pytest.raises(DateParseError, match=msg):
            Period("1/1/-2000", "Y")

    def test_constructor_corner(self) -> None:
        expected: Period = Period("2007-01", freq="2M")
        assert Period(year=2007, month=1, freq="2M") == expected

        assert Period(None) is NaT

        p: Period = Period("2007-01-01", freq="D")

        result: Period = Period(p, freq="Y")
        exp: Period = Period("2007", freq="Y")
        assert result == exp

    def test_constructor_infer_freq(self) -> None:
        p: Period = Period("2007-01-01")
        assert p.freq == "D"

        p = Period("2007-01-01 07")
        assert p.freq == "h"

        p = Period("2007-01-01 07:10")
        assert p.freq == "min"

        p = Period("2007-01-01 07:10:15")
        assert p.freq == "s"

        p = Period("2007-01-01 07:10:15.123")
        assert p.freq == "ms"

        # We see that there are 6 digits after the decimal, so get microsecond
        #  even though they are all zeros.
        p = Period("2007-01-01 07:10:15.123000")
        assert p.freq == "us"

        p = Period("2007-01-01 07:10:15.123400")
        assert p.freq == "us"

    def test_multiples(self) -> None:
        result1: Period = Period("1989", freq="2Y")
        result2: Period = Period("1989", freq="Y")
        assert result1.ordinal == result2.ordinal
        assert result1.freqstr == "2Y-DEC"
        assert result2.freqstr == "Y-DEC"
        assert result1.freq == offsets.YearEnd(2)
        assert result2.freq == offsets.YearEnd()

        assert (result1 + 1).ordinal == result1.ordinal + 2
        assert (1 + result1).ordinal == result1.ordinal + 2
        assert (result1 - 1).ordinal == result2.ordinal - 2
        assert (-1 + result1).ordinal == result2.ordinal - 2

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_quarterly(self, month: str) -> None:
        # bugs in scikits.timeseries
        freq: str = f"Q-{month}"
        exp: Period = Period("1989Q3", freq=freq)
        assert "1989Q3" in str(exp)
        stamp: Timestamp = exp.to_timestamp("D", how="end")
        p: Period = Period(stamp, freq=freq)
        assert p == exp

        stamp = exp.to_timestamp("3D", how="end")
        p = Period(stamp, freq=freq)
        assert p == exp

    @pytest.mark.parametrize("month", MONTHS)
    def test_period_cons_annual(self, month: str) -> None:
        # bugs in scikits.timeseries
        freq: str = f"Y-{month}"
        exp: Period = Period("1989", freq=freq)
        stamp: Timestamp = exp.to_timestamp("D", how="end") + timedelta(days=30)
        p: Period = Period(stamp, freq=freq)

        assert p == exp + 1
        assert isinstance(p, Period)

    @pytest.mark.parametrize("day", DAYS)
    @pytest.mark.parametrize("num", range(10, 17))
    def test_period_cons_weekly(self, num: int, day: str) -> None:
        daystr: str = f"2011-02-{num}"
        freq: str = f"W-{day}"

        result: Period = Period(daystr, freq=freq)
        expected: Period = Period(daystr, freq="D").asfreq(freq)
        assert result == expected
        assert isinstance(result, Period)

    def test_parse_week_str_roundstrip(self) -> None:
        # GH#50803
        per: Period = Period("2017-01-23/2017-01-29")
        assert per.freq.freqstr == "W-SUN"

        per = Period("2017-01-24/2017-01-30")
        assert per.freq.freqstr == "W-MON"

        msg: str = "Could not parse as weekly-freq Period"
        with pytest.raises(ValueError, match=msg):
            # not 6 days apart
            Period("2016-01-23/2017-01-29")

    def test_period_from_ordinal(self) -> None:
        p: Period = Period("2011-01", freq="M")
        res: Period = Period._from_ordinal(p.ordinal, freq=p.freq)
        assert p == res
        assert isinstance(res, Period)

    @pytest.mark.parametrize("freq", ["Y", "M", "D", "h"])
    def test_construct_from_nat_string_and_freq(self, freq: str) -> None:
        per: Period = Period("NaT", freq=freq)
        assert per is NaT

        per = Period("NaT", freq="2" + freq)
        assert per is NaT

        per = Period("NaT", freq="3" + freq)
        assert per is NaT

    def test_period_cons_nat(self) -> None:
        p: Period = Period("nat", freq="W-SUN")
        assert p is NaT

        p = Period(iNaT, freq="D")
        assert p is NaT

        p = Period(iNaT, freq="3D")
        assert p is NaT

        p = Period(iNaT, freq="1D1h")
        assert p is NaT

        p = Period("NaT")
        assert p is NaT

        p = Period(iNaT)
        assert p is NaT

    def test_period_cons_mult(self) -> None:
        p1: Period = Period("2011-01", freq="3M")
        p2: Period = Period("2011-01", freq="M")
        assert p1.ordinal == p2.ordinal

        assert p1.freq == offsets.MonthEnd(3)
        assert p1.freqstr == "3M"

        assert p2.freq == offsets.MonthEnd()
        assert p2.freqstr == "M"

        result: Period = p1 + 1
        assert result.ordinal == (p2 + 3).ordinal

        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        result = p1 - 1
        assert result.ordinal == (p2 - 3).ordinal
        assert result.freq == p1.freq
        assert result.freqstr == "3M"

        msg: str = "Frequency must be positive, because it represents span: -3M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-3M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0M")

    def test_period_cons_combined(self) -> None:
        p: List[Tuple[Period, Period, Period]] = [
            (
                Period("2011-01", freq="1D1h"),
                Period("2011-01", freq="1h1D"),
                Period("2011-01", freq="h"),
            ),
            (
                Period(ordinal=1, freq="1D1h"),
                Period(ordinal=1, freq="1h1D"),
                Period(ordinal=1, freq="h"),
            ),
        ]

        for p1, p2, p3 in p:
            assert p1.ordinal == p3.ordinal
            assert p2.ordinal == p3.ordinal

            assert p1.freq == offsets.Hour(25)
            assert p1.freqstr == "25h"

            assert p2.freq == offsets.Hour(25)
            assert p2.freqstr == "25h"

            assert p3.freq == offsets.Hour()
            assert p3.freqstr == "h"

            result: Period = p1 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 + 1
            assert result.ordinal == (p3 + 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

            result = p1 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p1.freq
            assert result.freqstr == "25h"

            result = p2 - 1
            assert result.ordinal == (p3 - 25).ordinal
            assert result.freq == p2.freq
            assert result.freqstr == "25h"

        msg: str = "Frequency must be positive, because it represents span: -25h"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="-1h1D")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1D1h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="-1h1D")

        msg = "Frequency must be positive, because it represents span: 0D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="0D0h")
        with pytest.raises(ValueError, match=msg):
            Period(ordinal=1, freq="0D0h")

        # You can only combine together day and intraday offsets
        msg = "Invalid frequency: 1W1D"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1W1D")
        msg = "Invalid frequency: 1D1W"
        with pytest.raises(ValueError, match=msg):
            Period("2011-01", freq="1D1W")

    @pytest.mark.parametrize("day", ["1970/01/01 ", "2020-12-31 ", "1981/09/13 "])
    @pytest.mark.parametrize("hour", ["00:00:00", "00:00:01", "23:59:59", "12:00:59"])
    @pytest.mark.parametrize(
        "sec_float, expected",
        [
            (".000000001", 1),
            (".000000999", 999),
            (".123456789", 789),
            (".999999999", 999),
            (".999999000", 0),
            # Test femtoseconds, attoseconds, picoseconds are dropped like Timestamp
            (".999999001123", 1),
            (".999999001123456", 1),
            (".999999001123456789", 1),
        ],
    )
    def test_period_constructor_nanosecond(
        self, day: str, hour: str, sec_float: str, expected: int
    ) -> None:
        # GH 34621

        assert Period(day + hour + sec_float).start_time.nanosecond == expected

    @pytest.mark.parametrize("hour", range(24))
    def test_period_large_ordinal(self, hour: int) -> None:
        # Issue #36430
        # Integer overflow for Period over the maximum timestamp
        p: Period = Period(ordinal=2562048 + hour, freq="1h")
        assert p.hour == hour

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    @pytest.mark.parametrize(
        "freq,freq_depr",
        [("2W", "2w"), ("2W-FRI", "2w-fri"), ("2D", "2d"), ("2B", "2b")],
    )
    def test_period_deprecated_lowercase_freq(self, freq: str, freq_depr: str) -> None:
        # GH#58998
        msg: str = (
            f"'{freq_depr[1:]}' is deprecated and will be removed in a future version."
        )

        with tm.assert_produces_warning(FutureWarning, match=msg):
            result: Period = Period("2016-03-01 09:00", freq=freq_depr)

        expected: Period = Period("2016-03-01 09:00", freq=freq)
        assert result == expected


class TestPeriodMethods:
    def test_round_trip(self) -> None:
        p: Period = Period("2000Q1")
        new_p: Period = tm.round_trip_pickle(p)
        assert new_p == p

    def test_hash(self) -> None:
        assert hash(Period("2011-01", freq="M")) == hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01-01", freq="D")) != hash(Period("2011-01", freq="M"))

        assert hash(Period("2011-01", freq="3M")) != hash(Period("2011-01", freq="2M"))

        assert hash(Period("2011-01", freq="M")) != hash(Period("2011-02", freq="M"))

    # --------------------------------------------------------------
    # to_timestamp

    def test_to_timestamp_mult(self) -> None:
        p: Period = Period("2011-01", freq="M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected: Timestamp = Timestamp("2011-02-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

        p = Period("2011-01", freq="3M")
        assert p.to_timestamp(how="S") == Timestamp("2011-01-01")
        expected = Timestamp("2011-04-01") - Timedelta(1, "ns")
        assert p.to_timestamp(how="E") == expected

    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_to_timestamp(self) -> None:
        p: Period = Period("1982", freq="Y")
        start_ts: Timestamp = p.to_timestamp(how="S")
        aliases: List[str] = ["s", "StarT", "BEGIn"]
        for a in aliases:
            assert start_ts == p.to_timestamp("D", how=a)
            # freq with mult should not affect to the result
            assert start_ts == p.to_timestamp("3D", how=a)

        end_ts: Timestamp = p.to_timestamp(how="E")
        aliases = ["e", "end", "FINIsH"]
        for a in aliases:
            assert end_ts == p.to_timestamp("D", how=a)
            assert end_ts == p.to_timestamp("3D", how=a)

        from_lst: List[str] = ["Y", "Q", "M", "W", "B", "D", "h", "Min", "s"]

        def _ex(p: Period) -> Timestamp:
            if p.freq == "B":
                return p.start_time + Timedelta(days=1, nanoseconds=-1)
            return Timestamp((p + p.freq).start_time._value - 1)

        for fcode in from_lst:
            p = Period("1982", freq=fcode)
            result: Period = p.to_timestamp().to_period(fcode)
            assert result == p

            assert p.start_time == p.to_timestamp(how="S")

            assert p.end_time == _ex(p)

        # Frequency other than daily

        p = Period("1985", freq="Y")

        result = p.to_timestamp("h", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected
        result = p.to_timestamp("3h", how="end")
        assert result == expected

        result = p.to_timestamp("min", how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected
        result = p.to_timestamp("2min", how="end")
        assert result == expected

        result = p.to_timestamp(how="end")
        expected = Timestamp(1986, 1, 1) - Timedelta(1, "ns")
        assert result == expected

        expected = datetime(1985, 1, 1)
        result = p.to_timestamp("h", how="start")
        assert result == expected
        result = p.to_timestamp("min", how="start")
        assert result == expected
        result = p.to_timestamp("s", how="start")
        assert result == expected
        result = p.to_timestamp("3h", how="start")
        assert result == expected
        result = p.to_timestamp("5s", how="start")
        assert result == expected

    def test_to_timestamp_business_end(self) -> None:
        with tm.assert_produces_warning(FutureWarning, match=bday_msg):
            per: Period = Period("1990-01-05", "B")  # Friday
            result: Timestamp = per.to_timestamp("B", how="E")

        expected: Timestamp = Timestamp("1990-01-06") - Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize(
        "ts, expected",
        [
            ("1970-01-01 00:00:00", 0),
            ("1970-01-01 00:00:00.000001", 