import calendar
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
import zoneinfo
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import dateutil.tz
from dateutil.tz import (
    gettz,
    tzoffset,
    tzutc,
)
import numpy as np
import pytest

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime

from pandas import (
    NA,
    NaT,
    Period,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestTimestampConstructorUnitKeyword:
    @pytest.mark.parametrize("typ", [int, float])
    def test_constructor_int_float_with_YM_unit(self, typ: type) -> None:
        # GH#47266 avoid the conversions in cast_from_unit
        val = typ(150)

        ts = Timestamp(val, unit="Y")
        expected = Timestamp("2120-01-01")
        assert ts == expected

        ts = Timestamp(val, unit="M")
        expected = Timestamp("1982-07-01")
        assert ts == expected

    @pytest.mark.parametrize("typ", [int, float])
    def test_construct_from_int_float_with_unit_out_of_bound_raises(self, typ: type) -> None:
        # GH#50870  make sure we get a OutOfBoundsDatetime instead of OverflowError
        val = typ(150000000000000)

        msg = f"cannot convert input {val} with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(val, unit="D")

    def test_constructor_float_not_round_with_YM_unit_raises(self) -> None:
        # GH#47267 avoid the conversions in cast_from-unit

        msg = "Conversion of non-round float with unit=[MY] is ambiguous"
        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="Y")

        with pytest.raises(ValueError, match=msg):
            Timestamp(150.5, unit="M")

    @pytest.mark.parametrize(
        "value, check_kwargs",
        [
            [946688461000000000, {}],
            [946688461000000000 / 1000, {"unit": "us"}],
            [946688461000000000 / 1_000_000, {"unit": "ms"}],
            [946688461000000000 / 1_000_000_000, {"unit": "s"}],
            [10957, {"unit": "D", "h": 0}],
            [
                (946688461000000000 + 500000) / 1000000000,
                {"unit": "s", "us": 499, "ns": 964},
            ],
            [
                (946688461000000000 + 500000000) / 1000000000,
                {"unit": "s", "us": 500000},
            ],
            [(946688461000000000 + 500000) / 1000000, {"unit": "ms", "us": 500}],
            [(946688461000000000 + 500000) / 1000, {"unit": "us", "us": 500}],
            [(946688461000000000 + 500000000) / 1000000, {"unit": "ms", "us": 500000}],
            [946688461000000000 / 1000.0 + 5, {"unit": "us", "us": 5}],
            [946688461000000000 / 1000.0 + 5000, {"unit": "us", "us": 5000}],
            [946688461000000000 / 1000000.0 + 0.5, {"unit": "ms", "us": 500}],
            [946688461000000000 / 1000000.0 + 0.005, {"unit": "ms", "us": 5, "ns": 5}],
            [946688461000000000 / 1000000000.0 + 0.5, {"unit": "s", "us": 500000}],
            [10957 + 0.5, {"unit": "D", "h": 12}],
        ],
    )
    def test_construct_with_unit(self, value: Union[int, float], check_kwargs: Dict[str, Any]) -> None:
        def check(value: Union[int, float], unit: Optional[str] = None, h: int = 1, s: int = 1, us: int = 0, ns: int = 0) -> None:
            stamp = Timestamp(value, unit=unit)
            assert stamp.year == 2000
            assert stamp.month == 1
            assert stamp.day == 1
            assert stamp.hour == h
            if unit != "D":
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
        # Test for GH#25057
        # Valid fold values are only [None, 0, 1]
        msg = "Valid values for the fold argument are None, 0, or 1."
        with pytest.raises(ValueError, match=msg):
            Timestamp(123, fold=2)

    def test_timestamp_constructor_pytz_fold_raise(self) -> None:
        # Test for GH#25057
        # pytz doesn't support fold. Check that we raise
        # if fold is passed with pytz
        pytz = pytest.importorskip("pytz")
        msg = "pytz timezones do not support fold. Please use dateutil timezones."
        tz = pytz.timezone("Europe/London")
        with pytest.raises(ValueError, match=msg):
            Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)

    @pytest.mark.parametrize("fold", [0, 1])
    @pytest.mark.parametrize(
        "ts_input",
        [
            1572136200000000000,
            1572136200000000000.0,
            np.datetime64(1572136200000000000, "ns"),
            "2019-10-27 01:30:00+01:00",
            datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc),
        ],
    )
    def test_timestamp_constructor_fold_conflict(self, ts_input: Any, fold: int) -> None:
        # Test for GH#25057
        # Check that we raise on fold conflict
        msg = (
            "Cannot pass fold with possibly unambiguous input: int, float, "
            "numpy.datetime64, str, or timezone-aware datetime-like. "
            "Pass naive datetime-like or build Timestamp from components."
        )
        with pytest.raises(ValueError, match=msg):
            Timestamp(ts_input=ts_input, fold=fold)

    @pytest.mark.parametrize("tz", ["dateutil/Europe/London", None])
    @pytest.mark.parametrize("fold", [0, 1])
    def test_timestamp_constructor_retain_fold(self, tz: Optional[str], fold: int) -> None:
        # Test for GH#25057
        # Check that we retain fold
        ts = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
        result = ts.fold
        expected = fold
        assert result == expected

    @pytest.mark.parametrize(
        "tz",
        [
            "dateutil/Europe/London",
            zoneinfo.ZoneInfo("Europe/London"),
        ],
    )
    @pytest.mark.parametrize(
        "ts_input,fold_out",
        [
            (1572136200000000000, 0),
            (1572139800000000000, 1),
            ("2019-10-27 01:30:00+01:00", 0),
            ("2019-10-27 01:30:00+00:00", 1),
            (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0),
            (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1),
        ],
    )
    def test_timestamp_constructor_infer_fold_from_value(self, tz: Any, ts_input: Any, fold_out: int) -> None:
        # Test for GH#25057
        # Check that we infer fold correctly based on timestamps since utc
        # or strings
        ts = Timestamp(ts_input, tz=tz)
        result = ts.fold
        expected = fold_out
        assert result == expected

    @pytest.mark.parametrize("tz", ["dateutil/Europe/London"])
    @pytest.mark.parametrize(
        "fold,value_out",
        [
            (0, 1572136200000000),
            (1, 1572139800000000),
        ],
    )
    def test_timestamp_constructor_adjust_value_for_fold(self, tz: str, fold: int, value_out: int) -> None:
        # Test for GH#25057
        # Check that we adjust value for fold correctly
        # based on timestamps since utc
        ts_input = datetime(2019, 10, 27, 1, 30)
        ts = Timestamp(ts_input, tz=tz, fold=fold)
        result = ts._value
        expected = value_out
        assert result == expected


class TestTimestampConstructorPositionalAndKeywordSupport:
    def test_constructor_positional(self) -> None:
        # see GH#10758
        msg = "'NoneType' object cannot be interpreted as an integer"
        with pytest.raises(TypeError, match=msg):
            Timestamp(2000, 1)

        msg = "month must be in 1..12"
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 0, 1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 13, 1)

        msg = "day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(2000, 1, 32)

        # see gh-11630
        assert repr(Timestamp(2015, 11, 12)) == repr(Timestamp("20151112"))
        assert repr(Timestamp(2015, 11, 12, 1, 2, 3, 999999)) == repr(
            Timestamp("2015-11-12 01:02:03.999999")
        )

    def test_constructor_keyword(self) -> None:
        # GH#10758
        msg = "function missing required argument 'day'|Required argument 'day'"
        with pytest.raises(TypeError, match=msg):
            Timestamp(year=2000, month=1)

        msg = "month must be in 1..12"
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=0, day=1)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=13, day=1)

        msg = "day is out of range for month"
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=0)
        with pytest.raises(ValueError, match=msg):
            Timestamp(year=2000, month=1, day=32)

        assert repr(Timestamp(year=2015, month=11, day=12)) == repr(
            Timestamp("20151112")
        )

        assert repr(
            Timestamp(
                year=2015,
                month=11,
                day=12,
                hour=1,
                minute=2,
                second=3,
                microsecond=999999,
            )
        ) == repr(Timestamp("2015-11-12 01:02:03.999999"))

    @pytest.mark.parametrize(
        "arg",
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "nanosecond",
        ],
    )
    def test_invalid_date_kwarg_with_string_input(self, arg: str) -> None:
        kwarg = {arg: 1}
        msg = "Cannot pass a date attribute keyword argument"
        with pytest.raises(ValueError, match=msg):
            Timestamp("2010-10-10 12:59:59.999999999", **kwarg)

    @pytest.mark.parametrize("kwargs", [{}, {"year": 2020}, {"year": 2020, "month": 1}])
    def test_constructor_missing_keyword(self, kwargs: Dict[str, int]) -> None:
        # GH#31200

        # The exact error message of datetime() depends on its version
        msg1 = r"function missing required argument '(year|month|day)' \(pos [123]\)"
        msg2 = r"Required argument '(year|month|day)' \(pos [123]\) not found"
        msg = "|".join([msg1, msg2])

        with pytest.raises(TypeError, match=msg):
            Timestamp(**kwargs)

    def test_constructor_positional_with_tzinfo(self) -> None:
        # GH#31929
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc)
        expected = Timestamp("2020-12-31", tzinfo=timezone.utc)
        assert ts == expected

    @pytest.mark.parametrize("kwd", ["nanosecond", "microsecond", "second", "minute"])
    def test_constructor_positional_keyword_mixed_with_tzinfo(self, kwd: str, request: Any) -> None:
        # TODO: if we passed microsecond with a keyword we would mess up
        #  xref GH#45307
        if kwd != "nanosecond":
            # nanosecond is keyword-only as of 2.0, others are not
            mark = pytest.mark.xfail(reason="GH#45307")
            request.applymarker(mark)

        kwargs = {kwd: 4}
        ts = Timestamp(2020, 12, 31, tzinfo=timezone.utc, **kwargs)

        td_kwargs = {kwd + "s": 4}
        td = Timedelta(**td_kwargs)
        expected = Timestamp("2020-12-31", tz=timezone.utc) + td
        assert ts == expected


class TestTimestampClassMethodConstructors:
    # Timestamp constructors other than __new__

    def test_utcnow_deprecated(self) -> None:
        # GH#56680
        msg = "Timestamp.utcnow is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            Timestamp.utcnow()

    def test_utcfromtimestamp_deprecated(self) -> None:
        # GH#56680
        msg = "Timestamp.utcfromtimestamp is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            Timestamp.utcfromtimestamp(43)

    def test_constructor_strptime(self) -> None:
        # GH#25016
        # Test support for Timestamp.strptime
        fmt = "%Y%m%d-%H%M%S-%f%z"
        ts = "20190129-235348-000001+0000"
        msg = r"Timestamp.strptime\(\) is not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            Timestamp.strptime(ts, fmt)

    def test_constructor_fromisocalendar(self) -> None:
        # GH#30395
        expected_timestamp = Timestamp("2000-01-03 00:00:00")
        expected_stdlib = datetime.fromisocalendar(2000, 1, 1)
        result = Timestamp.fromisocalendar(2000, 1, 1)
        assert result == expected_timestamp
        assert result == expected_stdlib
        assert isinstance(result, Timestamp)

    def test_constructor_fromordinal(self) -> None:
        base = datetime(2000, 1, 1)

        ts = Timestamp.fromordinal(base.toordinal())
        assert base == ts
        assert base.toordinal() == ts.toordinal()

        ts = Timestamp.fromordinal(base.toordinal(), tz="US/Eastern")
        assert Timestamp("2000-01-01", tz="US/Eastern") == ts
        assert base.toordinal() == ts.toordinal()

        # GH#3042
        dt = datetime(2011, 4, 16, 0, 0)
        ts = Timestamp.fromordinal(dt.toordinal())
        assert ts.to_pydatetime() == dt

        # with a tzinfo
        stamp = Timestamp("2011-4-16", tz="US/Eastern")
        dt_tz = stamp.to_pydatetime()
        ts = Timestamp.fromordinal(dt_tz.toordinal