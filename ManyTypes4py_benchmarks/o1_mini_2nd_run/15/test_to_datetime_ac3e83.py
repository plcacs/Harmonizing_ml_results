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
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timestamp,
    date_range,
    isna,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
from typing import Any, List, Tuple, Union, Optional

PARSING_ERR_MSG = (
    "You might want to try:\\n    - passing `format` if your strings have a consistent format;"
    "\\n    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;"
    "\\n    - passing `format='mixed'`, and the format will be inferred for each element individually. "
    "You might want to use `dayfirst` alongside this."
)


class TestTimeConversionFormats:
    def test_to_datetime_readonly(self, writable: Any) -> None:
        arr: np.ndarray[Any, Any] = np.array([], dtype=object)
        arr.setflags(write=writable)
        result: DatetimeIndex = to_datetime(arr)
        expected: DatetimeIndex = to_datetime([])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "format, expected",
        [
            ["%d/%m/%Y", [Timestamp("20000101"), Timestamp("20000201"), Timestamp("20000301")]],
            ["%m/%d/%Y", [Timestamp("20000101"), Timestamp("20000102"), Timestamp("20000103")]],
        ],
    )
    def test_to_datetime_format(
        self, cache: Any, index_or_series: Any, format: str, expected: List[Timestamp]
    ) -> None:
        values = index_or_series(["1/1/2000", "1/2/2000", "1/3/2000"])
        result = to_datetime(values, format=format, cache=cache)
        expected_index_or_series = index_or_series(expected)
        tm.assert_equal(result, expected_index_or_series)

    @pytest.mark.parametrize(
        "arg, expected, format",
        [
            ["1/1/2000", "20000101", "%d/%m/%Y"],
            ["1/1/2000", "20000101", "%m/%d/%Y"],
            ["1/2/2000", "20000201", "%d/%m/%Y"],
            ["1/2/2000", "20000102", "%m/%d/%Y"],
            ["1/3/2000", "20000301", "%d/%m/%Y"],
            ["1/3/2000", "20000103", "%m/%d/%Y"],
        ],
    )
    def test_to_datetime_format_scalar(
        self,
        cache: Any,
        arg: str,
        expected: str,
        format: str,
    ) -> None:
        result: Timestamp = to_datetime(arg, format=format, cache=cache)
        expected_ts: Timestamp = Timestamp(expected)
        assert result == expected_ts

    def test_to_datetime_format_YYYYMMDD(self, cache: Any) -> None:
        ser: Series = Series([19801222, 19801222] + [19810105] * 5)
        expected: Series = Series([Timestamp(x) for x in ser.apply(str)])
        result: Series = to_datetime(ser, format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)
        result = to_datetime(ser.apply(str), format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_with_nat(self, cache: Any) -> None:
        ser: Series = Series([19801222, 19801222] + [19810105] * 5, dtype="float")
        expected: Series = Series(
            [Timestamp("19801222"), Timestamp("19801222")] + [Timestamp("19810105")] * 5,
            dtype="M8[s]",
        )
        expected[2] = np.nan
        ser[2] = np.nan
        result: Series = to_datetime(ser, format="%Y%m%d", cache=cache)
        tm.assert_series_equal(result, expected)
        ser2: Series = ser.apply(str)
        ser2[2] = "nat"
        with pytest.raises(
            ValueError,
            match=r'unconverted data remains when parsing with format "%Y%m%d": ".0". ',
        ):
            to_datetime(ser2, format="%Y%m%d", cache=cache)

    def test_to_datetime_format_YYYYMM(self, cache: Any) -> None:
        ser: Series = Series([198012, 198012] + [198101] * 5, dtype="float")
        expected: Series = Series(
            [Timestamp("19801201"), Timestamp("19801201")] + [Timestamp("19810101")] * 5,
            dtype="M8[s]",
        )
        expected[2] = np.nan
        ser[2] = np.nan
        result: Series = to_datetime(ser, format="%Y%m", cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_oob_for_ns(self, cache: Any) -> None:
        ser: Series = Series([20121231, 20141231, 99991231])
        result: Series = to_datetime(ser, format="%Y%m%d", errors="raise", cache=cache)
        expected: Series = Series(
            np.array(["2012-12-31", "2014-12-31", "9999-12-31"], dtype="M8[s]"),
            dtype="M8[s]",
        )
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_coercion(self, cache: Any) -> None:
        ser: Series = Series([20121231, 20141231, 999999999999999999999999999991231])
        result: Series = to_datetime(
            ser, format="%Y%m%d", errors="coerce", cache=cache
        )
        expected: Series = Series(
            ["20121231", "20141231", "NaT"], dtype="M8[s]"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_s",
        [
            ["19801222", "20010112", None],
            ["19801222", "20010112", np.nan],
            ["19801222", "20010112", NaT],
            ["19801222", "20010112", "NaT"],
            [19801222, 20010112, None],
            [19801222, 20010112, np.nan],
            [19801222, 20010112, NaT],
            [19801222, 20010112, "NaT"],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_with_none(
        self, input_s: List[Union[str, int, None, float]]
    ) -> None:
        expected: Series = Series(
            [Timestamp("19801222"), Timestamp("20010112"), NaT]
        )
        result: Series = Series(to_datetime(input_s, format="%Y%m%d"))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_s, expected",
        [
            [
                ["19801222", np.nan, "20010012", "10019999"],
                [Timestamp("19801222"), np.nan, np.nan, np.nan],
            ],
            [
                ["19801222", "20010012", "10019999", np.nan],
                [Timestamp("19801222"), np.nan, np.nan, np.nan],
            ],
            [
                [20190813, np.nan, 20010012, 20019999],
                [Timestamp("20190813"), np.nan, np.nan, np.nan],
            ],
            [
                [20190813, 20010012, np.nan, 20019999],
                [Timestamp("20190813"), np.nan, np.nan, np.nan],
            ],
        ],
    )
    def test_to_datetime_format_YYYYMMDD_overflow(
        self, input_s: List[Union[str, int, float]], expected: List[Union[Timestamp, float]]
    ) -> None:
        input_series: Series = Series(input_s)
        result: Series = to_datetime(input_series, format="%Y%m%d", errors="coerce")
        expected_series: Series = Series(expected)
        tm.assert_series_equal(result, expected_series)

    @pytest.mark.parametrize(
        "data, format, expected",
        [
            ([pd.NA], "%Y%m%d%H%M%S", ["NaT"]),
            ([pd.NA], None, ["NaT"]),
            ([pd.NA, "20210202202020"], "%Y%m%d%H%M%S", ["NaT", "2021-02-02 20:20:20"]),
            (["201010", pd.NA], "%y%m%d", ["2020-10-10", "NaT"]),
            (["201010", pd.NA], "%d%m%y", ["2010-10-20", "NaT"]),
            ([None, np.nan, pd.NA], None, ["NaT", "NaT", "NaT"]),
            ([None, np.nan, pd.NA], "%Y%m%d", ["NaT", "NaT", "NaT"]),
        ],
    )
    def test_to_datetime_with_NA(
        self,
        data: List[Union[str, None, float, pd._libs.missing.NAType]],
        format: Optional[str],
        expected: List[Union[str, pd.Timestamp]],
    ) -> None:
        result: DatetimeIndex = to_datetime(data, format=format)
        expected_index: DatetimeIndex = DatetimeIndex(expected)
        tm.assert_index_equal(result, expected_index)

    def test_to_datetime_with_NA_with_warning(self) -> None:
        result: DatetimeIndex = to_datetime(["201010", pd.NA])
        expected: DatetimeIndex = DatetimeIndex(["2010-10-20", "NaT"])
        tm.assert_index_equal(result, expected)

    def test_to_datetime_format_integer(self, cache: Any) -> None:
        ser: Series = Series([2000, 2001, 2002])
        expected: Series = Series([Timestamp(x) for x in ser.apply(str)])
        result: Series = to_datetime(ser, format="%Y", cache=cache)
        tm.assert_series_equal(result, expected)
        ser = Series([200001, 200105, 200206])
        expected = Series([Timestamp(x[:4] + "-" + x[4:]) for x in ser.apply(str)])
        result = to_datetime(ser, format="%Y%m", cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_microsecond(self, cache: Any) -> None:
        month_abbr: str = calendar.month_abbr[4]
        val: str = f"01-{month_abbr}-2011 00:00:01.978"
        format: str = "%d-%b-%Y %H:%M:%S.%f"
        result: Timestamp = to_datetime(val, format=format, cache=cache)
        exp: datetime = datetime.strptime(val, format)
        assert result == exp

    @pytest.mark.parametrize(
        "value, format, dt",
        [
            ["01/10/2010 15:20", "%m/%d/%Y %H:%M", Timestamp("2010-01-10 15:20")],
            ["01/10/2010 05:43", "%m/%d/%Y %I:%M", Timestamp("2010-01-10 05:43")],
            ["01/10/2010 13:56:01", "%m/%d/%Y %H:%M:%S", Timestamp("2010-01-10 13:56:01")],
            pytest.param(
                "01/10/2010 08:14 PM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 20:14"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 07:40 AM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 07:40"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 09:12:56 AM",
                "%m/%d/%Y %I:%M:%S %p",
                Timestamp("2010-01-10 09:12:56"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
        ],
    )
    def test_to_datetime_format_time(
        self, cache: Any, value: str, format: str, dt: Timestamp
    ) -> None:
        assert to_datetime(value, format=format, cache=cache) == dt

    @td.skip_if_not_us_locale
    def test_to_datetime_with_non_exact(self, cache: Any) -> None:
        ser: Series = Series(
            [
                "19MAY11",
                "foobar19MAY11",
                "19MAY11:00:00:00",
                "19MAY11 00:00:00Z",
            ]
        )
        result: Series = to_datetime(ser, format="%d%b%y", exact=False, cache=cache)
        expected: Series = to_datetime(ser.str.extract("(\\d+\\w+\\d+)", expand=False), format="%d%b%y", cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "format, expected",
        [
            ("%Y-%m-%d", Timestamp(2000, 1, 3)),
            ("%Y-%d-%m", Timestamp(2000, 3, 1)),
            ("%Y-%m-%d %H", Timestamp(2000, 1, 3, 12)),
            ("%Y-%d-%m %H", Timestamp(2000, 3, 1, 12)),
            ("%Y-%m-%d %H:%M", Timestamp(2000, 1, 3, 12, 34)),
            ("%Y-%d-%m %H:%M", Timestamp(2000, 3, 1, 12, 34)),
            ("%Y-%m-%d %H:%M:%S", Timestamp(2000, 1, 3, 12, 34, 56)),
            ("%Y-%d-%m %H:%M:%S", Timestamp(2000, 3, 1, 12, 34, 56)),
            (
                "%Y-%m-%d %H:%M:%S.%f",
                Timestamp(2000, 1, 3, 12, 34, 56, 123456),
            ),
            (
                "%Y-%d-%m %H:%M:%S.%f",
                Timestamp(2000, 3, 1, 12, 34, 56, 123456),
            ),
            (
                "%Y-%m-%d %H:%M:%S.%f%z",
                Timestamp("2000-01-03 12:34:56.123456+01:00", tz="UTC+01:00"),
            ),
            (
                "%Y-%d-%m %H:%M:%S.%f%z",
                Timestamp("2000-03-01 12:34:56.123456+01:00", tz="UTC+01:00"),
            ),
        ],
    )
    def test_non_exact_doesnt_parse_whole_string(
        self, cache: Any, format: str, expected: Timestamp
    ) -> None:
        result: Timestamp = to_datetime(
            "2000-01-03 12:34:56.123456+01:00", format=format, exact=False
        )
        assert result == expected

    @pytest.mark.parametrize(
        "value, fmt, expected",
        [
            [
                "2009324",
                "%Y%W%w",
                "2009-08-13",
            ],
            [
                "2013020",
                "%Y%U%w",
                "2013-01-13",
            ],
        ],
    )
    def test_to_datetime_format_weeks(
        self,
        value: str,
        fmt: str,
        expected: str,
        cache: Any,
    ) -> None:
        assert to_datetime(value, format=fmt, cache=cache) == Timestamp(expected)

    @pytest.mark.parametrize(
        "fmt, dates, expected_dates",
        [
            [
                "%Y-%m-%d %H:%M:%S %Z",
                ["2010-01-01 12:00:00 UTC"] * 2,
                [Timestamp("2010-01-01 12:00:00", tz="UTC")] * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S%z",
                ["2010-01-01 12:00:00+0100"] * 2,
                [Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60)))] * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100"] * 2,
                [Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60)))] * 2,
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 Z", "2010-01-01 12:00:00 Z"],
                [
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))),
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))),
                ],
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset(
        self,
        fmt: str,
        dates: List[str],
        expected_dates: List[Timestamp],
    ) -> None:
        result: DatetimeIndex = to_datetime(dates, format=fmt)
        expected: Index = Index(expected_dates)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "fmt, dates, expected_dates",
        [
            [
                "%Y-%m-%d %H:%M:%S %Z",
                [
                    "2010-01-01 12:00:00 UTC",
                    "2010-01-01 12:00:00 GMT",
                    "2010-01-01 12:00:00 US/Pacific",
                ],
                [
                    Timestamp("2010-01-01 12:00:00", tz="UTC"),
                    Timestamp("2010-01-01 12:00:00", tz="GMT"),
                    Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                ],
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],
                [
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))),
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=-60))),
                ],
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset_utc_false_removed(
        self,
        fmt: str,
        dates: List[str],
        expected_dates: List[Timestamp],
    ) -> None:
        msg = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(dates, format=fmt)

    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(
        self,
    ) -> None:
        dates: List[str] = [
            "2010-01-01 12:00:00 +0100",
            "2010-01-01 12:00:00 -0100",
            "2010-01-01 12:00:00 +0300",
            "2010-01-01 12:00:00 +0400",
        ]
        expected_dates: List[str] = [
            "2010-01-01 11:00:00+00:00",
            "2010-01-01 13:00:00+00:00",
            "2010-01-01 09:00:00+00:00",
            "2010-01-01 08:00:00+00:00",
        ]
        fmt: str = "%Y-%m-%d %H:%M:%S %z"
        result: DatetimeIndex = to_datetime(dates, format=fmt, utc=True)
        expected: DatetimeIndex = DatetimeIndex(expected_dates).tz_localize("UTC")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "offset",
        [
            "+0",
            "-1foo",
            "UTCbar",
            ":10",
            "+01:000:01",
            "",
        ],
    )
    def test_to_datetime_parse_timezone_malformed(self, offset: str) -> None:
        fmt: str = "%Y-%m-%d %H:%M:%S %z"
        date: str = f"2010-01-01 12:00:00 {offset}"
        msg = "|".join(
            [
                f'^time data ".*" doesn\'t match format ".*". {PARSING_ERR_MSG}$',
                f'^unconverted data remains when parsing with format ".*": ".*". {PARSING_ERR_MSG}$',
            ]
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime([date], format=fmt)

    def test_to_datetime_parse_timezone_keeps_name(self) -> None:
        fmt: str = "%Y-%m-%d %H:%M:%S %z"
        arg: Index = Index(["2010-01-01 12:00:00 Z"], name="foo")
        result: DatetimeIndex = to_datetime(arg, format=fmt)
        expected: DatetimeIndex = DatetimeIndex(["2010-01-01 12:00:00"], tz="UTC", name="foo")
        tm.assert_index_equal(result, expected)

    class TestToDatetime:
        @pytest.mark.filterwarnings("ignore:Could not infer format")
        def test_to_datetime_overflow(self) -> None:
            arg: str = "08335394550"
            msg: str = 'Parsing "08335394550" to datetime overflows'
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(arg)
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime([arg])
            res: Timestamp = to_datetime(arg, errors="coerce")
            assert res is NaT
            res: DatetimeIndex = to_datetime([arg], errors="coerce")
            exp: DatetimeIndex = Index([NaT], dtype="M8[s]")
            tm.assert_index_equal(res, exp)

        def test_to_datetime_mixed_datetime_and_string(self) -> None:
            d1: datetime = datetime(
                2020, 1, 1, 17, tzinfo=timezone(-timedelta(hours=1))
            )
            d2: datetime = datetime(
                2020, 1, 1, 18, tzinfo=timezone(-timedelta(hours=1))
            )
            res: DatetimeIndex = to_datetime(["2020-01-01 17:00 -0100", d2])
            expected: DatetimeIndex = to_datetime([d1, d2]).tz_convert(timezone(timedelta(minutes=-60)))
            tm.assert_index_equal(res, expected)

        def test_to_datetime_mixed_string_and_numeric(self) -> None:
            vals: List[Union[str, int]] = ["2016-01-01", 0]
            expected: DatetimeIndex = DatetimeIndex([Timestamp(x) for x in vals])
            result: DatetimeIndex = to_datetime(vals, format="mixed")
            result2: DatetimeIndex = to_datetime(vals[::-1], format="mixed")[::-1]
            result3: DatetimeIndex = DatetimeIndex(vals)
            result4: DatetimeIndex = DatetimeIndex(vals[::-1])[::-1]
            tm.assert_index_equal(result, expected)
            tm.assert_index_equal(result2, expected)
            tm.assert_index_equal(result3, expected)
            tm.assert_index_equal(result4, expected)

        @pytest.mark.parametrize(
            "format",
            ["%Y-%m-%d", "%Y-%d-%m"],
            ids=["ISO8601", "non-ISO8601"],
        )
        def test_to_datetime_mixed_date_and_string(
            self, format: str
        ) -> None:
            d1: date = date(2020, 1, 2)
            res: DatetimeIndex = to_datetime(["2020-01-01", d1], format=format)
            expected: DatetimeIndex = DatetimeIndex(["2020-01-01", "2020-01-02"], dtype="M8[s]")
            tm.assert_index_equal(res, expected)

        @pytest.mark.parametrize(
            "fmt, expected",
            [
                ("%Y-%m-%d %H:%M:%S%z", DatetimeIndex(["2000-01-01 09:00:00+00:00", "2000-01-02 00:00:00+00:00", "NaT"], dtype="datetime64[us, UTC]")),
                ("%Y-%d-%m %H:%M:%S%z", DatetimeIndex(["2000-03-01 09:00:00+00:00", "2000-01-02 00:00:00+00:00", "NaT"], dtype="datetime64[us, UTC]")),
            ],
        )
        @pytest.mark.parametrize(
            "utc, args, expected",
            [
                pytest.param(
                    True,
                    ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00-08:00"],
                    DatetimeIndex(
                        ["2000-01-01 09:00:00+00:00", "2000-01-01 10:00:00+00:00"],
                        dtype="datetime64[us, UTC]",
                    ),
                    id="all tz-aware, with utc",
                ),
                pytest.param(
                    False,
                    ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],
                    DatetimeIndex(
                        ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"]
                    ).as_unit("us"),
                    id="all tz-aware, without utc",
                ),
                pytest.param(
                    True,
                    ["2000-01-01 01:00:00-08:00", "2000-01-01 02:00:00+00:00"],
                    DatetimeIndex(
                        ["2000-01-01 09:00:00+00:00", "2000-01-01 02:00:00+00:00"],
                        dtype="datetime64[us, UTC]",
                    ),
                    id="all tz-aware, mixed offsets, with utc",
                ),
                pytest.param(
                    True,
                    ["2000-01-01 01:00:00", "2000-01-01 02:00:00+00:00"],
                    DatetimeIndex(
                        ["2000-01-01 01:00:00+00:00", "2000-01-01 02:00:00+00:00"],
                        dtype="datetime64[us, UTC]",
                    ),
                    id="tz-aware string, naive pydatetime, with utc",
                ),
            ],
        )
        @pytest.mark.parametrize(
            "constructor",
            [Timestamp, lambda x: Timestamp(x).to_pydatetime()],
        )
        def test_to_datetime_mixed_datetime_and_string_with_format(
            self,
            fmt: str,
            utc: Optional[bool],
            args: List[str],
            expected: DatetimeIndex,
            constructor: Any,
        ) -> None:
            ts1 = constructor(args[0])
            ts2 = args[1]
            result: DatetimeIndex = to_datetime([ts1, ts2], format=fmt, utc=utc)
            if constructor is Timestamp:
                expected = expected.as_unit("s")
            tm.assert_index_equal(result, expected)

        @pytest.mark.parametrize(
            "fmt, expected",
            [
                ("%Y-%m-%d %H:%M:%S%z", [Timestamp("2000-01-01 08:00:00+00:00"), Timestamp("2000-01-02 00:00:00+00:00"), NaT]),
                ("%Y-%d-%m %H:%M:%S%z", [Timestamp("2000-03-01 08:00:00+00:00"), Timestamp("2000-01-02 00:00:00+00:00"), NaT]),
            ],
        )
        def test_to_datetime_mixed_offsets_with_none_tz_utc_false_removed(
            self, fmt: str, expected: List[Union[Timestamp, float]]
        ) -> None:
            msg = "Mixed timezones detected. Pass utc=True in to_datetime"
            with pytest.raises(ValueError, match=msg):
                to_datetime(
                    ["2000-01-01 08:00:00+01:00", "2000-01-02 00:00:00+02:00", None],
                    format=fmt,
                    utc=False,
                )

        @pytest.mark.parametrize(
            "fmt, expected",
            [
                (
                    "%Y-%m-%d %H:%M:%S%z",
                    DatetimeIndex(
                        ["2000-01-01 08:00:00+00:00", "2000-01-02 00:00:00+00:00", "NaT"],
                        dtype="datetime64[s, UTC]",
                    ),
                ),
                (
                    "%Y-%d-%m %H:%M:%S%z",
                    DatetimeIndex(
                        ["2000-03-01 08:00:00+00:00", "2000-01-02 00:00:00+00:00", "NaT"],
                        dtype="datetime64[s, UTC]",
                    ),
                ),
            ],
        )
        def test_to_datetime_mixed_offsets_with_none(
            self,
            fmt: str,
            expected: DatetimeIndex,
        ) -> None:
            result: DatetimeIndex = to_datetime(
                ["2000-01-01 08:00:00+01:00", "2000-01-02 00:00:00+02:00", None],
                format=fmt,
                utc=True,
            )
            tm.assert_index_equal(result, expected)

        @pytest.mark.parametrize(
            "fmt, dates, expected_dates",
            [
                [
                    "%Y-%m-%d %H:%M:%S %Z",
                    ["2010-01-01 12:00:00 UTC", "2010-01-01 12:00:00 GMT", "2010-01-01 12:00:00 US/Pacific"],
                    [
                        Timestamp("2010-01-01 12:00:00", tz="UTC"),
                        Timestamp("2010-01-01 12:00:00", tz="GMT"),
                        Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                    ],
                ],
                [
                    "%Y-%m-%d %H:%M:%S %z",
                    ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],
                    [
                        Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))),
                        Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=-60))),
                    ],
                ],
            ],
        )
    def test_to_datetime_parse_tzname_or_tzoffset(
        self,
        fmt: str,
        dates: List[str],
        expected_dates: List[Timestamp],
    ) -> None:
        result: DatetimeIndex = to_datetime(dates, format=fmt)
        expected: Index = Index(expected_dates)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "fmt, dates, expected_dates",
        [
            [
                "%Y-%m-%d %H:%M:%S %Z",
                ["2010-01-01 12:00:00 UTC", "2010-01-01 12:00:00 GMT", "2010-01-01 12:00:00 US/Pacific"],
                [
                    Timestamp("2010-01-01 12:00:00", tz="UTC"),
                    Timestamp("2010-01-01 12:00:00", tz="GMT"),
                    Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                ],
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],
                [
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))),
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=-60))),
                ],
            ],
        ],
    )
    @pytest.mark.parametrize(
        "fmt, dates, expected_dates",
        [
            [
                "%Y-%m-%d %H:%M:%S %Z",
                ["2010-01-01 12:00:00 UTC", "2010-01-01 12:00:00 GMT", "2010-01-01 12:00:00 US/Pacific"],
                [
                    Timestamp("2010-01-01 12:00:00", tz="UTC"),
                    Timestamp("2010-01-01 12:00:00", tz="GMT"),
                    Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                ],
            ],
            [
                "%Y-%m-%d %H:%M:%S %z",
                ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],
                [
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))),
                    Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=-60))),
                ],
            ],
        ],
    )
    def test_to_datetime_parse_tzname_or_tzoffset_utc_false_removed(
        self,
        fmt: str,
        dates: List[str],
        expected_dates: List[Timestamp],
    ) -> None:
        msg: str = "Mixed timezones detected. Pass utc=True in to_datetime"
        with pytest.raises(ValueError, match=msg):
            to_datetime(dates, format=fmt)

    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(
        self,
    ) -> None:
        dates: List[str] = [
            "2010-01-01 12:00:00 +0100",
            "2010-01-01 12:00:00 -0100",
            "2010-01-01 12:00:00 +0300",
            "2010-01-01 12:00:00 +0400",
        ]
        expected_dates: List[str] = [
            "2010-01-01 11:00:00+00:00",
            "2010-01-01 13:00:00+00:00",
            "2010-01-01 09:00:00+00:00",
            "2010-01-01 08:00:00+00:00",
        ]
        fmt: str = "%Y-%m-%d %H:%M:%S %z"
        result: DatetimeIndex = to_datetime(dates, format=fmt, utc=True)
        expected: DatetimeIndex = DatetimeIndex(expected_dates).tz_localize("UTC")
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "offset",
        [
            "+0",
            "-1foo",
            "UTCbar",
            ":10",
            "+01:000:01",
            "",
        ],
    )
    def test_to_datetime_parse_timezone_malformed(self, offset: str) -> None:
        fmt: str = "%Y-%m-%d %H:%M:%S %z"
        date: str = f"2010-01-01 12:00:00 {offset}"
        msg: str = "|".join(
            [
                f'^time data ".*" doesn\'t match format ".*". {PARSING_ERR_MSG}$',
                f'^unconverted data remains when parsing with format ".*": ".*". {PARSING_ERR_MSG}$',
            ]
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime([date], format=fmt)

    def test_to_datetime_parse_timezone_keeps_name(self) -> None:
        fmt: str = "%Y-%m-%d %H:%M:%S %z"
        arg: Index = Index(["2010-01-01 12:00:00 Z"], name="foo")
        result: DatetimeIndex = to_datetime(arg, format=fmt)
        expected: DatetimeIndex = DatetimeIndex(
            ["2010-01-01 12:00:00"], tz="UTC", name="foo"
        )
        tm.assert_index_equal(result, expected)

    class TestToDatetimeUnit:
        @pytest.mark.parametrize(
            "unit",
            ["Y", "M"],
        )
        @pytest.mark.parametrize(
            "item",
            [150, float(150)],
        )
        def test_to_datetime_month_or_year_unit_int(
            self, cache: Any, unit: str, item: Union[int, float], request: Any
        ) -> None:
            ts: Timestamp = Timestamp(item, unit=unit)
            expected: DatetimeIndex = DatetimeIndex([ts], dtype="M8[ns]")
            result: DatetimeIndex = to_datetime([item], unit=unit, cache=cache)
            tm.assert_index_equal(result, expected)
            result = to_datetime(np.array([item], dtype=object), unit=unit, cache=cache)
            tm.assert_index_equal(result, expected)
            result = to_datetime(np.array([item]), unit=unit, cache=cache)
            tm.assert_index_equal(result, expected)
            result = to_datetime(np.array([item, np.nan]), unit=unit, cache=cache)
            assert result.isna()[1]
            tm.assert_index_equal(result[:1], expected)

        @pytest.mark.parametrize(
            "unit",
            ["Y", "M"],
        )
        def test_to_datetime_month_or_year_unit_non_round_float(
            self, cache: Any, unit: str
        ) -> None:
            msg: str = f"Conversion of non-round float with unit={unit} is ambiguous"
            with pytest.raises(ValueError, match=msg):
                to_datetime([1.5], unit=unit, errors="raise")
            msg = 'Given date string \\"1.5\\" not likely a datetime'
            with pytest.raises(ValueError, match=msg):
                to_datetime(["1.5"], unit=unit, errors="raise")
            res: DatetimeIndex = to_datetime([1.5], unit=unit, errors="coerce")
            expected: DatetimeIndex = DatetimeIndex([NaT], dtype="M8[ns]")
            tm.assert_index_equal(res, expected)
            res = to_datetime(["1.5"], unit=unit, errors="coerce")
            expected = to_datetime([NaT]).as_unit("ns")
            tm.assert_index_equal(res, expected)
            res = to_datetime([1.0], unit=unit)
            expected = to_datetime([1], unit=unit)
            tm.assert_index_equal(res, expected)

        def test_unit(self, cache: Any) -> None:
            msg: str = "cannot specify both format and unit"
            with pytest.raises(ValueError, match=msg):
                to_datetime([1], unit="D", format="%Y%m%d", cache=cache)

        def test_unit_array_mixed_nans(self, cache: Any) -> None:
            values: List[Union[str, int]] = [11111111111111111, 1, 1.0, iNaT, NaT, np.nan, "NaT", ""]
            result: DatetimeIndex = to_datetime(values, unit="D", errors="coerce", cache=cache)
            expected: DatetimeIndex = DatetimeIndex(
                ["NaT", "1970-01-02", "1970-01-02", "NaT", "NaT", "NaT", "NaT", "NaT"],
                dtype="M8[ns]",
            )
            tm.assert_index_equal(result, expected)
            msg: str = "cannot convert input 11111111111111111 with the unit 'D'"
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(values, unit="D", errors="raise", cache=cache)

        def test_unit_array_mixed_nans_large_int(self, cache: Any) -> None:
            values: List[Union[str, int, float]] = [
                1420043460000000000000000,
                iNaT,
                NaT,
                np.nan,
                "NaT",
            ]
            result: DatetimeIndex = to_datetime(values, errors="coerce", unit="s", cache=cache)
            expected: DatetimeIndex = DatetimeIndex(["NaT", "NaT", "NaT", "NaT", "NaT"], dtype="M8[ns]")
            tm.assert_index_equal(result, expected)
            msg: str = "cannot convert input 1420043460000000000000000 with the unit 's'"
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(values, errors="raise", unit="s", cache=cache)

        def test_to_datetime_invalid_str_not_out_of_bounds_valuerror(
            self, cache: Any
        ) -> None:
            msg: str = "Unknown datetime string format, unable to parse: foo"
            with pytest.raises(ValueError, match=msg):
                to_datetime("foo", errors="raise", unit="s", cache=cache)

        @pytest.mark.parametrize(
            "unit",
            ["s", "D"],
        )
        def test_unit_consistency(self, cache: Any, unit: str) -> None:
            expected: Timestamp = Timestamp("1970-01-01 00:00:01")
            result: Timestamp = to_datetime(1, unit=unit, errors="raise", cache=cache)
            assert result == expected
            assert isinstance(result, Timestamp)

        @pytest.mark.parametrize(
            "errors, arr, warning",
            [
                (
                    "coerce",
                    ["2015-06-19 05:33:20", "2015-05-27 22:33:20"],
                    None,
                ),
                (
                    "raise",
                    ["2015-06-19 05:33:20", "2015-05-27 22:33:20"],
                    None,
                ),
            ],
        )
        def test_unit_with_numeric(
            self, cache: Any, errors: str, dtype: str
        ) -> None:
            expected: DatetimeIndex = DatetimeIndex(
                ["2015-06-19 05:33:20", "2015-05-27 22:33:20"], dtype="M8[ns]"
            )
            arr: np.ndarray[Any, Any] = np.array([1.434692e+18, 1.432766e+18], dtype=dtype)
            result: DatetimeIndex = to_datetime(arr, errors=errors, cache=cache)
            tm.assert_index_equal(result, expected)

        @pytest.mark.parametrize(
            "exp, arr, warning",
            [
                (
                    ["NaT", "2015-06-19 05:33:20", "2015-05-27 22:33:20"],
                    ["foo", 1.434692e+18, 1.432766e+18],
                    UserWarning,
                ),
                (
                    ["2015-06-19 05:33:20", "2015-05-27 22:33:20", "NaT", "NaT"],
                    [1.434692e+18, 1.432766e+18, "foo", "NaT"],
                    None,
                ),
            ],
        )
        def test_unit_with_numeric_coerce(
            self,
            cache: Any,
            exp: List[Union[str, pd.Timestamp]],
            arr: List[Union[str, float]],
            warning: Optional[Any],
        ) -> None:
            expected: DatetimeIndex = DatetimeIndex(exp, dtype="M8[ns]")
            with tm.assert_produces_warning(
                warning, match="Could not infer format"
            ):
                result: DatetimeIndex = to_datetime(arr, errors="coerce", cache=cache)
            tm.assert_index_equal(result, expected)

        @pytest.mark.parametrize(
            "arr",
            [
                [Timestamp("20130101"), 1.434692e+18, 1.432766e+18],
                [1.434692e+18, 1.432766e+18, Timestamp("20130101")],
            ],
        )
        def test_unit_mixed(
            self, cache: Any, arr: List[Union[Timestamp, float]]
        ) -> None:
            expected: DatetimeIndex = DatetimeIndex(
                [Timestamp(x) for x in arr], dtype="M8[ns]"
            )
            result: DatetimeIndex = to_datetime(arr, errors="coerce", cache=cache)
            tm.assert_index_equal(result, expected)
            result = to_datetime(arr, errors="raise", cache=cache)
            tm.assert_index_equal(result, expected)
            result = DatetimeIndex(arr)
            tm.assert_index_equal(result, expected)

        def test_unit_rounding(self, cache: Any) -> None:
            value: float = 1434743731.877
            result: Timestamp = to_datetime(value, unit="s", cache=cache)
            expected: Timestamp = Timestamp("2015-06-19 19:55:31.877000093")
            assert result == expected
            alt: Timestamp = Timestamp(value, unit="s")
            assert alt == result

        @pytest.mark.parametrize(
            "dtype",
            ["float64", "int64"],
        )
        def test_to_datetime_unit(self, cache: Any, dtype: str) -> None:
            epoch: int = 1370745748
            ser: Series = Series([epoch + t for t in range(20)]).astype(dtype)
            result: DatetimeIndex = to_datetime(ser, unit="s")
            expected: Series = Series(
                [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)],
                dtype="M8[ns]",
            )
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "null",
            [iNaT, np.nan],
        )
        def test_to_datetime_unit_with_nulls(self, null: Union[int, float, str]) -> None:
            epoch: int = 1370745748
            ser: Series = Series([epoch + t for t in range(20)] + [null])
            result: DatetimeIndex = to_datetime(ser, unit="s")
            expected: Series = Series(
                [Timestamp("2013-06-09 02:42:28") + timedelta(seconds=t) for t in range(20)]
                + [NaT],
                dtype="M8[ns]",
            )
            tm.assert_series_equal(result, expected)

        def test_float_to_datetime_raise_near_bounds(self) -> None:
            msg: str = "cannot convert input with unit 'D'"
            oneday_in_ns: float = 1000000000.0 * 60 * 60 * 24
            tsmax_in_days: float = 2**63 / oneday_in_ns
            should_succeed: Series = Series(
                [0, tsmax_in_days - 0.005, -tsmax_in_days + 0.005], dtype=float
            )
            expected: np.ndarray[Any, Any] = (should_succeed * oneday_in_ns).astype(np.int64)
            for error_mode in ["raise", "coerce"]:
                result1: DatetimeIndex = to_datetime(
                    should_succeed, unit="D", errors=error_mode
                )
                tm.assert_almost_equal(
                    result1.astype(np.int64).astype(np.float64),
                    expected.astype(np.float64),
                    rtol=1e-10,
                )
            should_fail1: Series = Series([0, tsmax_in_days + 0.005], dtype=float)
            should_fail2: Series = Series([0, -tsmax_in_days - 0.005], dtype=float)
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(should_fail1, unit="D", errors="raise")
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(should_fail2, unit="D", errors="raise")

    class TestToDatetimeDataFrame:
        @pytest.fixture
        def df(self) -> DataFrame:
            return DataFrame(
                {
                    "year": [2015, 2016],
                    "month": [2, 3],
                    "day": [4, 5],
                    "hour": [6, 7],
                    "minute": [58, 59],
                    "second": [10, 11],
                    "ms": [1, 1],
                    "us": [2, 2],
                    "ns": [3, 3],
                }
            )

        def test_dataframe(self, df: DataFrame, cache: Any) -> None:
            result: Series = to_datetime(
                {"year": df["year"], "month": df["month"], "day": df["day"]},
                cache=cache,
            )
            expected: Series = Series(
                [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
            )
            tm.assert_series_equal(result, expected)
            result = to_datetime(
                df[["year", "month", "day"]].to_dict(), cache=cache
            )
            expected.index = Index([0, 1])
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "unit",
            [{"year": "years", "month": "months", "day": "days", "hour": "hours", "minute": "minutes", "second": "seconds"}, {"year": "year", "month": "month", "day": "day", "hour": "hour", "minute": "minute", "second": "second"}],
        )
        def test_dataframe_field_aliases_column_subset(
            self, df: DataFrame, cache: Any, unit: dict
        ) -> None:
            result: Series = to_datetime(
                df[list(unit.keys())].rename(columns=unit), cache=cache
            )
            expected: Series = Series(
                [Timestamp("20150204 06:58:10"), Timestamp("20160305 07:59:11")],
                dtype="M8[ns]",
            )
            tm.assert_series_equal(result, expected)

        def test_dataframe_field_aliases(self, df: DataFrame, cache: Any) -> None:
            d: dict = {
                "year": "year",
                "month": "month",
                "day": "day",
                "hour": "hour",
                "minute": "minute",
                "second": "second",
                "ms": "ms",
                "us": "us",
                "ns": "ns",
            }
            result: Series = to_datetime(df.rename(columns=d), cache=cache)
            expected: Series = Series(
                [Timestamp("20150204 06:58:10.001002003"), Timestamp("20160305 07:59:11.001002003")]
            )
            tm.assert_series_equal(result, expected)

        def test_dataframe_str_dtype(self, df: DataFrame, cache: Any) -> None:
            result: Series = to_datetime(df.astype(str), cache=cache)
            expected: Series = Series(
                [Timestamp("20150204 06:58:10.001002003"), Timestamp("20160305 07:59:11.001002003")]
            )
            tm.assert_series_equal(result, expected)

        def test_dataframe_float32_dtype(self, df: DataFrame, cache: Any) -> None:
            result: Series = to_datetime(df.astype(np.float32), cache=cache)
            expected: Series = Series(
                [Timestamp("20150204 06:58:10.001002003"), Timestamp("20160305 07:59:11.001002003")]
            )
            tm.assert_series_equal(result, expected)

        def test_dataframe_coerce(self, cache: Any) -> None:
            df2: DataFrame = DataFrame(
                {"year": [2015, 2016], "month": [2, 20], "day": [4, 5]}
            )
            msg: str = "^cannot assemble the datetimes: time data \".+\" doesn't match format \"%Y%m%d\"\\.$"
            with pytest.raises(ValueError, match=msg):
                to_datetime(df2, cache=cache)
            result: Series = to_datetime(df2, errors="coerce", cache=cache)
            expected: Series = Series(
                [Timestamp("20150204 00:00:00"), NaT],
                dtype="M8[ns]",
            )
            tm.assert_series_equal(result, expected)

        def test_dataframe_extra_keys_raises(self, cache: Any) -> None:
            msg: str = "extra keys have been passed to the datetime assemblage: \\[foo\\]"
            df2: DataFrame = DataFrame(
                {"year": [2015, 2016], "month": [2, 20], "day": [4, 5]}
            )
            df2["foo"] = 1
            with pytest.raises(ValueError, match=msg):
                to_datetime(df2, cache=cache)

        @pytest.mark.parametrize(
            "cols",
            [
                ["year"],
                ["year", "month"],
                ["year", "month", "second"],
                ["month", "day"],
                ["year", "day", "second"],
            ],
        )
        def test_dataframe_missing_keys_raises(
            self, df: DataFrame, cache: Any, cols: List[str]
        ) -> None:
            msg: str = "to assemble mappings requires at least that \\[year, month, day\\] be specified: \\[.+\\] is missing"
            with pytest.raises(ValueError, match=msg):
                to_datetime(df[cols], cache=cache)

        def test_dataframe_duplicate_columns_raises(self, cache: Any) -> None:
            msg: str = "cannot assemble with duplicate keys"
            df2: DataFrame = DataFrame(
                {"year": [2015, 2016], "month": [2, 20], "day": [4, 5]}
            )
            df2.columns = ["year", "year", "day"]
            with pytest.raises(ValueError, match=msg):
                to_datetime(df2, cache=cache)
            df2 = DataFrame(
                {
                    "year": [2015, 2016],
                    "month": [2, 20],
                    "day": [4, 5],
                    "hour": [4, 5],
                }
            )
            df2.columns = ["year", "month", "day", "day"]
            with pytest.raises(ValueError, match=msg):
                to_datetime(df2, cache=cache)

        def test_dataframe_int16(self, cache: Any) -> None:
            df: DataFrame = DataFrame(
                {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}
            )
            result: Series = to_datetime(df.astype("int16"), cache=cache)
            expected: Series = Series(
                [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")]
            )
            tm.assert_series_equal(result, expected)

        def test_dataframe_mixed(self, cache: Any) -> None:
            df: DataFrame = DataFrame(
                {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}
            )
            df["month"] = df["month"].astype("int8")
            df["day"] = df["day"].astype("int8")
            result: Series = to_datetime(df, cache=cache)
            expected: Series = Series(
                [Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")],
                dtype="M8[ns]",
            )
            tm.assert_series_equal(result, expected)

        def test_dataframe_float(self, cache: Any) -> None:
            df: DataFrame = DataFrame(
                {"year": [2000, 2001], "month": [1.5, 1], "day": [1, 1]}
            )
            msg: str = '^unconverted data remains when parsing with format ".*": "1". .+$'
            with pytest.raises(ValueError, match=msg):
                to_datetime(df, cache=cache)

        def test_dataframe_utc_true(self) -> None:
            df: DataFrame = DataFrame(
                {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}
            )
            result: Series = to_datetime(df, utc=True)
            expected: Series = Series(
                ["2015-02-04", "2016-03-05"], dtype="datetime64[s]"
            ).dt.tz_localize("UTC")
            tm.assert_series_equal(result, expected)

    class TestOrigin:
        def test_origin_and_unit(self) -> None:
            ts: Timestamp = to_datetime(1, unit="s", origin=1)
            expected: Timestamp = Timestamp("1970-01-01 00:00:02")
            assert ts == expected
            ts = to_datetime(1, unit="s", origin=1000000000)
            expected = Timestamp("2001-09-09 01:46:41")
            assert ts == expected

        def test_julian(self, julian_dates: np.ndarray[Any, Any]) -> None:
            result: Series = Series(to_datetime(julian_dates, unit="D", origin="julian"))
            expected: Series = Series(
                to_datetime(julian_dates - Timestamp(0).to_julian_date(), unit="D")
            )
            tm.assert_series_equal(result, expected)

        def test_unix(self) -> None:
            result: Series = Series(to_datetime([0, 1, 2], unit="D", origin="unix"))
            expected: Series = Series(
                [
                    Timestamp("1970-01-01"),
                    Timestamp("1970-01-02"),
                    Timestamp("1970-01-03"),
                ],
                dtype="M8[ns]",
            )
            tm.assert_series_equal(result, expected)

        def test_julian_round_trip(self) -> None:
            result: Timestamp = to_datetime(2456658, origin="julian", unit="D")
            assert result.to_julian_date() == 2456658
            msg: str = "1 is Out of Bounds for origin='julian'"
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(1, origin="julian", unit="D")

        @pytest.mark.parametrize(
            "unit",
            ["ns", "D"],
        )
        def test_invalid_unit(self, unit: str, julian_dates: np.ndarray[Any, Any]) -> None:
            if unit != "D":
                msg: str = "unit must be 'D' for origin='julian'"
                with pytest.raises(ValueError, match=msg):
                    to_datetime(julian_dates, unit=unit, origin="julian")

        @pytest.mark.parametrize(
            "epochs",
            [
                Timestamp(1960, 1, 1),
                datetime(1960, 1, 1),
                "1960-01-01",
                np.datetime64("1960-01-01"),
            ],
        )
        def test_epoch(self, unit: str, epochs: Any, cache: Any) -> None:
            epoch_1960: Timestamp = Timestamp(1960, 1, 1)
            units_from_epochs: np.ndarray[Any, Any] = np.arange(5, dtype=np.int64)
            expected: Series = Series(
                [pd.Timedelta(x, unit=unit) + epoch_1960 for x in units_from_epochs]
            )
            result: Series = to_datetime(units_from_epochs, unit=unit, origin=epochs)
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "origin, exc",
            [
                ("random_string", ValueError),
                ("epoch", ValueError),
                ("13-24-1990", ValueError),
                (datetime(1, 1, 1), OutOfBoundsDatetime),
            ],
        )
        def test_invalid_origins(
            self,
            origin: Union[str, datetime],
            exc: type,
            units: str,
        ) -> None:
            msg: str = "|".join(
                [
                    f"origin {origin} is Out of Bounds",
                    f"origin {origin} cannot be converted to a Timestamp",
                    "Cannot cast .* to unit='ns' without overflow",
                ]
            )
            with pytest.raises(exc, match=msg):
                to_datetime(list(range(5)), unit=units, origin=origin)

        def test_invalid_origins_tzinfo(self) -> None:
            msg: str = "must be tz-naive"
            with pytest.raises(ValueError, match=msg):
                to_datetime(1, unit="D", origin=datetime(2000, 1, 1, tzinfo=timezone.utc))

        def test_incorrect_value_exception(self) -> None:
            msg: str = 'Unknown datetime string format, unable to parse: yesterday'
            with pytest.raises(ValueError, match=msg):
                to_datetime(["today", "yesterday"])

        @pytest.mark.parametrize(
            "fmt, warning",
            [
                (None, UserWarning),
                ("%Y-%m-%d %H:%M:%S", None),
                ("%Y-%d-%m %H:%M:%S", None),
            ],
        )
        def test_to_datetime_out_of_bounds_with_format_arg(
            self, fmt: Optional[str], warning: Optional[Any]
        ) -> None:
            if fmt is None:
                res: Timestamp = to_datetime("2417-10-10 00:00:00.00", format=fmt)
                assert isinstance(res, Timestamp)
                assert res.year == 2417
                assert res.month == 10
                assert res.day == 10
            else:
                msg: str = 'unconverted data remains when parsing with format ".*": ".*". .*'
                with pytest.raises(ValueError, match=msg):
                    to_datetime("2417-10-10 00:00:00.00", format=fmt)

    class TestGuessDatetimeFormat:
        @pytest.mark.parametrize(
            "test_list",
            [
                [
                    "2011-12-30 00:00:00.000000",
                    "2011-12-30 00:00:00.000000",
                    "2011-12-30 00:00:00.000000",
                ],
                [np.nan, np.nan, "2011-12-30 00:00:00.000000"],
                ["", "2011-12-30 00:00:00.000000"],
                ["NaT", "2011-12-30 00:00:00.000000"],
                ["2011-12-30 00:00:00.000000", "random_string"],
                ["now", "2011-12-30 00:00:00.000000"],
                ["today", "2011-12-30 00:00:00.000000"],
            ],
        )
        def test_guess_datetime_format_for_array(self, test_list: List[str]) -> None:
            expected_format: str = "%Y-%m-%d %H:%M:%S.%f"
            test_array: np.ndarray[Any, Any] = np.array(test_list, dtype=object)
            assert tools._guess_datetime_format_for_array(test_array) == expected_format

        @td.skip_if_not_us_locale
        def test_guess_datetime_format_for_array_all_nans(self) -> None:
            format_for_string_of_nans: Optional[str] = tools._guess_datetime_format_for_array(
                np.array([np.nan, np.nan, np.nan], dtype="O")
            )
            assert format_for_string_of_nans is None

    class TestToDatetimeInferFormat:
        @pytest.mark.parametrize(
            "test_format",
            ["%m-%d-%Y", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"],
            ids=["consistent_format", "consistent_format", "consistent_format"],
        )
        def test_to_datetime_infer_datetime_format_consistent_format(
            self, cache: Any, test_format: str
        ) -> None:
            ser: Series = Series(date_range("20000101", periods=50, freq="h"))
            s_as_dt_strings: Series = ser.apply(lambda x: x.strftime(test_format))
            with_format: Series = to_datetime(s_as_dt_strings, format=test_format, cache=cache)
            without_format: Series = to_datetime(s_as_dt_strings, cache=cache)
            tm.assert_series_equal(with_format, without_format)

        def test_to_datetime_inconsistent_format(self, cache: Any) -> None:
            data: List[str] = [
                "01/01/2011 00:00:00",
                "01-02-2011 00:00:00",
                "2011-01-03T00:00:00",
            ]
            ser: Series = Series(np.array(data))
            msg: str = r'^time data "01-02-2011 00:00:00" doesn\'t match format "%m/%d/%Y %H:%M:%S". .*'
            with pytest.raises(ValueError, match=msg):
                to_datetime(ser, cache=cache)

        def test_to_datetime_consistent_format(self, cache: Any) -> None:
            data: List[str] = ["Jan/01/2011", "Feb/01/2011", "Mar/01/2011"]
            ser: Series = Series(np.array(data))
            result: DatetimeIndex = to_datetime(ser, cache=cache)
            expected: DatetimeIndex = DatetimeIndex(["2011-01-01", "2011-02-01", "2011-03-01"], dtype="datetime64[s]")
            tm.assert_series_equal(result, expected)

        def test_to_datetime_series_with_nans(self, cache: Any) -> None:
            ser: Series = Series(
                np.array(
                    ["01/01/2011 00:00:00", np.nan, "01/03/2011 00:00:00", np.nan],
                    dtype=object,
                )
            )
            result: Series = to_datetime(ser, cache=cache)
            expected: Series = Series(
                ["2011-01-01", NaT, "2011-01-03", NaT], dtype="datetime64[s]"
            )
            tm.assert_series_equal(result, expected)

        def test_to_datetime_series_start_with_nans(self, cache: Any) -> None:
            ser: Series = Series(
                np.array(
                    [np.nan, np.nan, "01/01/2011 00:00:00", "01/02/2011 00:00:00", "01/03/2011 00:00:00"],
                    dtype=object,
                )
            )
            result: Series = to_datetime(ser, cache=cache)
            expected: Series = Series(
                [NaT, NaT, "2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[s]"
            )
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "tz_name, offset, dt_string_repr",
            [
                ("UTC", 0, "Timestamp('2019-01-01T00:00:00.000Z', tz='UTC')"),
                (
                    "-01:00",
                    180,
                    "Timestamp('2019-01-01T01:00:00.000Z', tz='UTC')",
                ),
                (
                    "+01:00",
                    -180,
                    "Timestamp('2019-01-01T00:00:00.000Z', tz='UTC')",
                ),
            ],
        )
        def test_infer_datetime_format_tz_name(
            self,
            tz_name: str,
            offset: int,
            dt_string_repr: str,
        ) -> None:
            ser: Series = Series([f"2019-01-01 00:00:00{tz_name}"])
            result: DatetimeIndex = to_datetime(ser)
            tz: timezone = timezone(timedelta(minutes=offset))
            expected: Series = Series([Timestamp("2019-01-01 00:00:00", tz=tz)])
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "ts, zero_tz",
            [
                ("2020-01-01 00:00:00Z", "Z"),
                ("2020-01-01 00:00:00", ""),
                ("2020-01-01 00:00:00.000", "Z"),
                ("2020-01-01 00:00:00.000", ""),
            ],
        )
        def test_infer_datetime_format_zero_tz(
            self, ts: str, zero_tz: str
        ) -> None:
            ser: Series = Series([ts + zero_tz])
            result: DatetimeIndex = to_datetime(ser)
            tz: Optional[timezone] = timezone.utc if zero_tz == "Z" else None
            expected: Series = Series([Timestamp(ts, tz=tz)])
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "date_str, dayfirst, yearfirst, expected",
            [
                ("10-11-12", False, False, datetime(2012, 10, 11)),
                ("10-11-12", True, False, datetime(2012, 11, 10)),
                ("10-11-12", False, True, datetime(2010, 11, 12)),
                ("10-11-12", True, True, datetime(2010, 12, 11)),
                ("20/12/21", False, False, datetime(2021, 12, 20)),
                ("20/12/21", True, False, datetime(2021, 12, 20)),
                ("20/12/21", False, True, datetime(2020, 12, 21)),
                ("20/12/21", True, True, datetime(2020, 12, 21)),
                ("20201012", True, False, datetime(2020, 12, 10)),
            ],
        )
        def test_parsers_dayfirst_yearfirst(
            self,
            cache: Any,
            date_str: str,
            dayfirst: bool,
            yearfirst: bool,
            expected: datetime,
        ) -> None:
            dateutil_result: datetime = parse(
                date_str, dayfirst=dayfirst, yearfirst=yearfirst
            )
            assert dateutil_result == expected
            result1, _ = parsing.parse_datetime_string_with_reso(
                date_str, dayfirst=dayfirst, yearfirst=yearfirst
            )
            if not dayfirst and not yearfirst:
                result2: Timestamp = Timestamp(date_str)
                assert result2 == expected
            result3: Timestamp = to_datetime(
                date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache
            )
            result4: Timestamp = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]
            assert result1 == expected
            assert result3 == expected
            assert result4 == expected

        @pytest.mark.parametrize(
            "date_str, dayfirst, yearfirst, expected",
            [
                ("10-11-12", False, False, datetime(2012, 10, 11)),
                ("10-11-12", True, False, datetime(2012, 11, 10)),
                ("10-11-12", False, True, datetime(2010, 11, 12)),
                ("10-11-12", True, True, datetime(2010, 12, 11)),
                ("20/12/21", False, False, datetime(2021, 12, 20)),
                ("20/12/21", True, False, datetime(2021, 12, 20)),
                ("20/12/21", False, True, datetime(2020, 12, 21)),
                ("20/12/21", True, True, datetime(2020, 12, 21)),
                ("20201012", True, False, datetime(2020, 12, 10)),
            ],
        )
        def test_parsers_dayfirst_yearfirst(
            self,
            cache: Any,
            date_str: str,
            dayfirst: bool,
            yearfirst: bool,
            expected: datetime,
        ) -> None:
            dateutil_result: datetime = parse(
                date_str, dayfirst=dayfirst, yearfirst=yearfirst
            )
            assert dateutil_result == expected
            result1, _ = parsing.parse_datetime_string_with_reso(
                date_str, dayfirst=dayfirst, yearfirst=yearfirst
            )
            if not dayfirst and not yearfirst:
                result2: Timestamp = Timestamp(date_str)
                assert result2 == expected
            result3: Timestamp = to_datetime(
                date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache
            )
            result4: Timestamp = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]
            assert result1 == expected
            assert result3 == expected
            assert result4 == expected

        @pytest.mark.parametrize(
            "value, fmt, dt",
            [
                ["01/10/2010 15:20", "%m/%d/%Y %H:%M", Timestamp("2010-01-10 15:20")],
                ["01/10/2010 05:43", "%m/%d/%Y %I:%M", Timestamp("2010-01-10 05:43")],
                ["01/10/2010 13:56:01", "%m/%d/%Y %H:%M:%S", Timestamp("2010-01-10 13:56:01")],
                pytest.param(
                    "01/10/2010 08:14 PM",
                    "%m/%d/%Y %I:%M %p",
                    Timestamp("2010-01-10 20:14"),
                    marks=pytest.mark.xfail(
                        locale.getlocale()[0] in ("zh_CN", "it_IT"),
                        reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                        strict=False,
                    ),
                ),
                pytest.param(
                    "01/10/2010 07:40 AM",
                    "%m/%d/%Y %I:%M %p",
                    Timestamp("2010-01-10 07:40"),
                    marks=pytest.mark.xfail(
                        locale.getlocale()[0] in ("zh_CN", "it_IT"),
                        reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                        strict=False,
                    ),
                ),
                pytest.param(
                    "01/10/2010 09:12:56 AM",
                    "%m/%d/%Y %I:%M:%S %p",
                    Timestamp("2010-01-10 09:12:56"),
                    marks=pytest.mark.xfail(
                        locale.getlocale()[0] in ("zh_CN", "it_IT"),
                        reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                        strict=False,
                    ),
                ),
            ],
        )
        def test_to_datetime_format_time(
            self, cache: Any, value: str, fmt: str, dt: Timestamp
        ) -> None:
            assert to_datetime(value, format=fmt, cache=cache) == dt

        @td.skip_if_not_us_locale
        def test_to_datetime_with_non_exact(self, cache: Any) -> None:
            ser: Series = Series(
                [
                    "19MAY11",
                    "foobar19MAY11",
                    "19MAY11:00:00:00",
                    "19MAY11 00:00:00Z",
                ]
            )
            result: Series = to_datetime(ser, format="%d%b%y", exact=False, cache=cache)
            expected: Series = to_datetime(ser.str.extract("(\\d+\\w+\\d+)", expand=False), format="%d%b%y", cache=cache)
            tm.assert_series_equal(result, expected)

        @pytest.mark.parametrize(
            "format, expected",
            [
                ("%Y-%m-%d", Timestamp(2000, 1, 3)),
                ("%Y-%d-%m", Timestamp(2000, 3, 1)),
                ("%Y-%m-%d %H", Timestamp(2000, 1, 3, 12)),
                ("%Y-%d-%m %H", Timestamp(2000, 3, 1, 12)),
                ("%Y-%m-%d %H:%M", Timestamp(2000, 1, 3, 12, 34)),
                ("%Y-%d-%m %H:%M", Timestamp(2000, 3, 1, 12, 34)),
                ("%Y-%m-%d %H:%M:%S", Timestamp(2000, 1, 3, 12, 34, 56)),
                ("%Y-%d-%m %H:%M:%S", Timestamp(2000, 3, 1, 12, 34, 56)),
                ("%Y-%m-%d %H:%M:%S.%f", Timestamp(2000, 1, 3, 12, 34, 56, 123456)),
                ("%Y-%d-%m %H:%M:%S.%f", Timestamp(2000, 3, 1, 12, 34, 56, 123456)),
                (
                    "%Y-%m-%d %H:%M:%S.%f%z",
                    Timestamp("2000-01-03 12:34:56.123456+01:00", tz="UTC+01:00"),
                ),
                (
                    "%Y-%d-%m %H:%M:%S.%f%z",
                    Timestamp("2000-03-01 12:34:56.123456+01:00", tz="UTC+01:00"),
                ),
            ],
        )
        def test_non_exact_doesnt_parse_whole_string(
            self,
            cache: Any,
            format: str,
            expected: Timestamp,
        ) -> None:
            result: Timestamp = to_datetime(
                "2000-01-03 12:34:56.123456+01:00", format=format, exact=False
            )
            assert result == expected

        @pytest.mark.parametrize(
            "arg",
            [
                "2012-01-01 09:00:00.000000001",
                "2012-01-01 09:00:00.000001",
                "2012-01-01 09:00:00.001",
                "2012-01-01 09:00:00.001000",
                "2012-01-01 09:00:00.001000000",
            ],
        )
        def test_parse_nanoseconds_with_formula(
            self, cache: Any, arg: str
        ) -> None:
            expected: Timestamp = to_datetime(arg, cache=cache)
            result: Timestamp = to_datetime(arg, format="%Y-%m-%d %H:%M:%S.%f", cache=cache)
            assert result == expected

        @pytest.mark.parametrize(
            "value, fmt, expected",
            [
                ["2009324", "%Y%W%w", "2009-08-13"],
                ["2013020", "%Y%U%w", "2013-01-13"],
            ],
        )
        def test_to_datetime_format_weeks(
            self,
            value: str,
            fmt: str,
            expected: str,
            cache: Any,
        ) -> None:
            assert to_datetime(value, format=fmt, cache=cache) == Timestamp(expected)

        @pytest.mark.parametrize(
            "fmt, dates, expected_dates",
            [
                [
                    "%Y-%m-%d %H:%M:%S %Z",
                    ["2010-01-01 12:00:00 UTC"] * 2,
                    [Timestamp("2010-01-01 12:00:00", tz="UTC")] * 2,
                ],
                [
                    "%Y-%m-%d %H:%M:%S%z",
                    ["2010-01-01 12:00:00+0100"] * 2,
                    [Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60)))] * 2,
                ],
                [
                    "%Y-%m-%d %H:%M:%S %z",
                    ["2010-01-01 12:00:00 +0100"] * 2,
                    [Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60)))] * 2,
                ],
                [
                    "%Y-%m-%d %H:%M:%S %z",
                    ["2010-01-01 12:00:00 Z", "2010-01-01 12:00:00 Z"],
                    [
                        Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))),
                        Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=0))),
                    ],
                ],
            ],
        )
        @pytest.mark.parametrize(
            "fmt, dates, expected_dates",
            [
                [
                    "%Y-%m-%d %H:%M:%S %Z",
                    ["2010-01-01 12:00:00 UTC", "2010-01-01 12:00:00 GMT", "2010-01-01 12:00:00 US/Pacific"],
                    [
                        Timestamp("2010-01-01 12:00:00", tz="UTC"),
                        Timestamp("2010-01-01 12:00:00", tz="GMT"),
                        Timestamp("2010-01-01 12:00:00", tz="US/Pacific"),
                    ],
                ],
                [
                    "%Y-%m-%d %H:%M:%S %z",
                    ["2010-01-01 12:00:00 +0100", "2010-01-01 12:00:00 -0100"],
                    [
                        Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=60))),
                        Timestamp("2010-01-01 12:00:00", tzinfo=timezone(timedelta(minutes=-60))),
                    ],
                ],
            ],
        )
        def test_to_datetime_parse_tzname_or_tzoffset_utc_false_removed(
            self,
            fmt: str,
            dates: List[str],
            expected_dates: List[Timestamp],
        ) -> None:
            msg: str = "Mixed timezones detected. Pass utc=True in to_datetime"
            with pytest.raises(ValueError, match=msg):
                to_datetime(dates, format=fmt)

        def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(
            self,
        ) -> None:
            dates: List[str] = [
                "2010-01-01 12:00:00 +0100",
                "2010-01-01 12:00:00 -0100",
                "2010-01-01 12:00:00 +0300",
                "2010-01-01 12:00:00 +0400",
            ]
            expected_dates: List[str] = [
                "2010-01-01 11:00:00+00:00",
                "2010-01-01 13:00:00+00:00",
                "2010-01-01 09:00:00+00:00",
                "2010-01-01 08:00:00+00:00",
            ]
            fmt: str = "%Y-%m-%d %H:%M:%S %z"
            result: DatetimeIndex = to_datetime(dates, format=fmt, utc=True)
            expected: DatetimeIndex = DatetimeIndex(expected_dates).tz_localize("UTC")
            tm.assert_index_equal(result, expected)


class TestToDatetimeMisc:
    def test_to_datetime_barely_out_of_bounds(self) -> None:
        arr: np.ndarray = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)
        msg: str = "^Out of bounds nanosecond timestamp: .*"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arr)

    @pytest.mark.parametrize(
        "arg, expected",
        [
            ("2011-01-01", datetime(2011, 1, 1)),
            ("2Q2005", datetime(2005, 4, 1)),
            ("2Q05", datetime(2005, 4, 1)),
            ("2005Q1", datetime(2005, 1, 1)),
            ("05Q1", datetime(2005, 1, 1)),
            ("2011Q3", datetime(2011, 7, 1)),
            ("11Q3", datetime(2011, 7, 1)),
            ("3Q2011", datetime(2011, 7, 1)),
            ("3Q11", datetime(2011, 7, 1)),
            ("2000Q4", datetime(2000, 10, 1)),
            ("00Q4", datetime(2000, 10, 1)),
            ("4Q2000", datetime(2000, 10, 1)),
            ("4Q00", datetime(2000, 10, 1)),
            ("2000q4", datetime(2000, 10, 1)),
            ("2000-Q4", datetime(2000, 10, 1)),
            ("00-Q4", datetime(2000, 10, 1)),
            ("4Q-2000", datetime(2000, 10, 1)),
            ("4Q-00", datetime(2000, 10, 1)),
            ("00q4", datetime(2000, 10, 1)),
            ("2005", datetime(2005, 1, 1)),
            ("2005-11", datetime(2005, 11, 1)),
            ("2005 11", datetime(2005, 11, 1)),
            ("11-2005", datetime(2005, 11, 1)),
            ("11 2005", datetime(2005, 11, 1)),
            ("200511", datetime(2020, 5, 11)),
            ("20051109", datetime(2005, 11, 9)),
            ("20051109 10:15", datetime(2005, 11, 9, 10, 15)),
            ("20051109 08H", datetime(2005, 11, 9, 8, 0)),
            ("2005-11-09 10:15", datetime(2005, 11, 9, 10, 15)),
            ("2005-11-09 08H", datetime(2005, 11, 9, 8, 0)),
            ("2005/11/09 10:15", datetime(2005, 11, 9, 10, 15)),
            ("2005/11/09 10:15:32", datetime(2005, 11, 9, 10, 15, 32)),
            ("2005/11/09 10:15:32 AM", datetime(2005, 11, 9, 10, 15, 32)),
            ("2005/11/09 10:15:32 PM", datetime(2005, 11, 9, 22, 15, 32)),
            ("2005/11/09 08H", datetime(2005, 11, 9, 8, 0)),
            ("Thu Sep 25 10:36:28 2003", datetime(2003, 9, 25, 10, 36, 28)),
            ("Thu Sep 25 2003", datetime(2003, 9, 25)),
            ("Sep 25 2003", datetime(2003, 9, 25)),
            ("January 1 2014", datetime(2014, 1, 1)),
            ("2014-06", datetime(2014, 6, 1)),
            ("06-2014", datetime(2014, 6, 1)),
            ("2014-6", datetime(2014, 6, 1)),
            ("6-2014", datetime(2014, 6, 1)),
            ("20010101 12", datetime(2001, 1, 1, 12)),
            ("20010101 1234", datetime(2001, 1, 1, 12, 34)),
            ("20010101 123456", datetime(2001, 1, 1, 12, 34, 56)),
        ],
    )
    def test_parsers(
        self,
        date_str: str,
        expected: datetime,
        cache: Any,
    ) -> None:
        yearfirst: bool = True
        result1, reso_attrname: Tuple[Optional[datetime], Optional[str]] = parsing.parse_datetime_string_with_reso(
            date_str, yearfirst=yearfirst
        )
        reso: str = (
            {"nanosecond": "ns", "microsecond": "us", "millisecond": "ms", "second": "s"}.get(reso_attrname, "s")
        )
        result2: Timestamp = to_datetime(date_str, yearfirst=yearfirst, cache=cache)
        result3: DatetimeIndex = to_datetime([date_str], yearfirst=yearfirst, cache=cache)
        result4: Timestamp = Timestamp(date_str)
        result5: Timestamp = DatetimeIndex([date_str], yearfirst=yearfirst)[0]
        assert result1 == expected
        assert result2 == expected
        tm.assert_index_equal(
            result3,
            DatetimeIndex([Timestamp(expected)], dtype=f"M8[{reso}]"),
        )
        assert result4 == expected
        assert result5 == expected

        if not yearfirst:
            result5 = Timestamp(date_str)
            assert result5 == expected
            result7 = date_range(date_str, freq="S", periods=1, yearfirst=yearfirst)
            assert result7 == expected

    def test_na_values_with_cache(
        self, cache: Any, unique_nulls_fixture: Any, unique_nulls_fixture2: Any
    ) -> None:
        expected: Index = Index([NaT, NaT], dtype="datetime64[s]")
        result: DatetimeIndex = to_datetime(
            [unique_nulls_fixture, unique_nulls_fixture2], cache=cache
        )
        tm.assert_index_equal(result, expected)

    def test_parsers_nat(self) -> None:
        result1, _ = parsing.parse_datetime_string_with_reso("NaT")
        result2: Timestamp = to_datetime("NaT")
        result3: Timestamp = Timestamp("NaT")
        result4: Timestamp = DatetimeIndex(["NaT"])[0]
        assert result1 is NaT
        assert result2 is NaT
        assert result3 is NaT
        assert result4 is NaT

    @pytest.mark.parametrize(
        "date_str, dayfirst, yearfirst, expected",
        [
            ("10-11-12", False, False, datetime(2012, 10, 11)),
            ("10-11-12", True, False, datetime(2012, 11, 10)),
            ("10-11-12", False, True, datetime(2010, 11, 12)),
            ("10-11-12", True, True, datetime(2010, 12, 11)),
            ("20/12/21", False, False, datetime(2021, 12, 20)),
            ("20/12/21", True, False, datetime(2021, 12, 20)),
            ("20/12/21", False, True, datetime(2020, 12, 21)),
            ("20/12/21", True, True, datetime(2020, 12, 21)),
            ("20201012", True, False, datetime(2020, 12, 10)),
        ],
    )
    def test_parsers_dayfirst_yearfirst(
        self,
        cache: Any,
        date_str: str,
        dayfirst: bool,
        yearfirst: bool,
        expected: datetime,
    ) -> None:
        dateutil_result: datetime = parse(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst
        )
        assert dateutil_result == expected
        result1, _ = parsing.parse_datetime_string_with_reso(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst
        )
        if not dayfirst and not yearfirst:
            result2: Timestamp = Timestamp(date_str)
            assert result2 == expected
        result3: Timestamp = to_datetime(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache
        )
        result4: Timestamp = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]
        assert result1 == expected
        assert result3 == expected
        assert result4 == expected

    @pytest.mark.parametrize(
        "ts, zero_tz, expected_repr",
        [
            ("2019-01-01 00:00:00Z", "Z", "Timestamp('2019-01-01T00:00:00.000Z', tz='UTC')"),
            ("2019-01-01 00:00:00", "", "Timestamp('2019-01-01T00:00:00.000', tz=None)"),
            ("2019-01-01 00:00:00.000", "Z", "Timestamp('2019-01-01T00:00:00.000Z', tz='UTC')"),
            ("2019-01-01 00:00:00.000", "", "Timestamp('2019-01-01T00:00:00.000', tz=None)"),
        ],
    )
    def test_infer_datetime_format_zero_tz(
        self, ts: str, zero_tz: str, expected_repr: str
    ) -> None:
        ser: Series = Series([ts + zero_tz])
        result: DatetimeIndex = to_datetime(ser)
        tz: Optional[timezone] = timezone.utc if zero_tz == "Z" else None
        expected: Series = Series([Timestamp(ts, tz=tz)])
        tm.assert_series_equal(result, expected)

    def test_timezone_name_raises_on_invalid(self) -> None:
        dtstr: str = "2014 Jan 9 05:15 FAKE"
        msg: str = '.*un-recognized timezone "FAKE".'
        with pytest.raises(ValueError, match=msg):
            Timestamp(dtstr)
        with pytest.raises(ValueError, match=msg):
            to_datetime(dtstr)
        with pytest.raises(ValueError, match=msg):
            to_datetime([dtstr])

    def test_unformatted_input_raises(self) -> None:
        valid: str = "2012-01-01 00:00:00"
        invalid: str = "N"
        ser: Series = Series([valid, invalid])
        with pytest.raises(ValueError, match='^Given date string "1" not likely a datetime$'):
            to_datetime([1, "1"], errors="raise", cache=True)

    def test_from_numeric_arrow_dtype(self, any_numeric_ea_dtype: str) -> None:
        pytest.importorskip("pyarrow")
        ser: Series = Series([1, 2], dtype=f"{any_numeric_ea_dtype.lower()}[pyarrow]")
        result: Series = to_datetime(ser)
        expected: Series = Series([1, 2], dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

    def test_to_datetime_with_format_f_parse_nanos(self, cache: Any) -> None:
        timestamp: str = "15/02/2020 02:03:04.123456789"
        timestamp_format: str = "%d/%m/%Y %H:%M:%S.%f"
        result: Timestamp = to_datetime(timestamp, format=timestamp_format)
        expected: Timestamp = Timestamp(year=2020, month=2, day=15, hour=2, minute=3, second=4, microsecond=123456, nanosecond=789)
        assert result == expected

    def test_to_datetime_mixed_iso8601(self) -> None:
        result: DatetimeIndex = to_datetime(["2020-01-01", "2020-01-01 05:00:00"], format="ISO8601")
        expected: DatetimeIndex = DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 05:00:00"])
        tm.assert_index_equal(result, expected)

    def test_to_datetime_mixed_other(self) -> None:
        result: DatetimeIndex = to_datetime(["01/11/2000", "12 January 2000"], format="mixed")
        expected: DatetimeIndex = DatetimeIndex(["2000-01-11", "2000-01-12"])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "exact, format",
        [
            (True, "mixed"),
            (True, "ISO8601"),
        ],
    )
    def test_to_datetime_mixed_or_iso_exact(
        self, exact: bool, format: str
    ) -> None:
        msg: str = "Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(["2020-01-01"], exact=exact, format=format)

    def test_to_datetime_mixed_not_necessarily_iso8601_raise(self) -> None:
        with pytest.raises(ValueError, match='Time data 01-01-2000 is not ISO8601 format'):
            to_datetime(["2020-01-01", "01-01-2000"], format="ISO8601")

    def test_to_datetime_mixed_not_necessarily_iso8601_coerce(self) -> None:
        result: DatetimeIndex = to_datetime(
            ["2020-01-01", "01-01-2000"], format="ISO8601", errors="coerce"
        )
        expected: DatetimeIndex = DatetimeIndex(["2020-01-01 00:00:00", NaT])
        tm.assert_index_equal(result, expected)

    def test_unknown_tz_raises(self) -> None:
        dtstr: str = "2014 Jan 9 05:15 FAKE"
        msg: str = '.*un-recognized timezone "FAKE".'
        with pytest.raises(ValueError, match=msg):
            Timestamp(dtstr)
        with pytest.raises(ValueError, match=msg):
            to_datetime(dtstr)
        with pytest.raises(ValueError, match=msg):
            to_datetime([dtstr])

    def test_unformatted_input_raises(self) -> None:
        valid: str = "2012-01-01 00:00:00"
        invalid: str = "1"
        ser: Series = Series([1, "1"])
        msg: str = "^Given date string \"1\" not likely a datetime$"
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, errors="raise")

    def test_string_invalid_operation(self, cache: Any) -> None:
        invalid: np.ndarray = np.array(["87156549591102612381000001219H5"], dtype=object)
        with pytest.raises(ValueError, match="Unknown datetime string format"):
            to_datetime(invalid, errors="raise", cache=cache)

    def test_string_na_nat_conversion(self, cache: Any) -> None:
        strings: np.ndarray = np.array(
            ["1/1/2000", "1/2/2000", np.nan, "1/4/2000"], dtype=object
        )
        expected: np.ndarray = np.empty(4, dtype="M8[s]")
        for i, val in enumerate(strings):
            if isna(val):
                expected[i] = iNaT
            else:
                expected[i] = parse(val)
        result: DatetimeIndex = tslib.array_to_datetime(strings)[0]
        tm.assert_almost_equal(result, expected)
        result2: DatetimeIndex = to_datetime(strings, cache=cache)
        assert isinstance(result2, DatetimeIndex)
        tm.assert_numpy_array_equal(result, result2.values)

    def test_string_na_nat_conversion_malformed(
        self, cache: Any
    ) -> None:
        malformed: np.ndarray = np.array(["1/100/2000", np.nan], dtype=object)
        msg: str = "Unknown datetime string format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

    def test_string_na_nat_conversion_with_name(
        self, cache: Any
    ) -> None:
        idx: List[str] = ["a", "b", "c", "d", "e"]
        series: Series = Series(
            ["1/1/2000", np.nan, "1/3/2000", np.nan, "1/5/2000"], index=idx, name="foo"
        )
        dseries: Series = Series(
            [
                to_datetime("1/1/2000", cache=cache),
                np.nan,
                to_datetime("1/3/2000", cache=cache),
                np.nan,
                to_datetime("1/5/2000", cache=cache),
            ],
            index=idx,
            name="foo",
        )
        result: Series = to_datetime(series, cache=cache)
        expected: Series = Series(
            [to_datetime("1/1/2000", cache=cache), NaT, to_datetime("1/3/2000", cache=cache), NaT, to_datetime("1/5/2000", cache=cache)],
            index=idx,
            dtype="M8[s]",
        )
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name == "foo"
        tm.assert_series_equal(dresult, expected, check_names=False)
        assert dresult.name == "foo"

    def test_to_datetime_wrapped_datetime64_ps(self) -> None:
        result: DatetimeIndex = to_datetime([np.datetime64(1901901901901, "ps")])
        expected: DatetimeIndex = DatetimeIndex(["1970-01-01 00:00:01.901901901"], dtype="datetime64[ns]", freq=None)
        tm.assert_index_equal(result, expected)
    
class TestDaysInMonth:
    @pytest.mark.parametrize(
        "arg, format",
        [
            ("2015-02-29", None),
            ("2015-02-29", "%Y-%m-%d"),
            ("2015-02-32", "%Y-%m-%d"),
            ("2015-04-31", "%Y-%m-%d"),
        ],
    )
    def test_day_not_in_month_coerce(
        self, cache: Any, arg: str, format: Optional[str]
    ) -> None:
        assert isna(to_datetime(arg, errors="coerce", format=format, cache=cache))

    @pytest.mark.parametrize(
        "arg, format, msg",
        [
            ("2015-02-29", "%Y-%m-%d", r"^day is out of range for month. .*"),
            ("2015-29-02", "%Y-%d-%m", r"^day is out of range for month. .*"),
            ("2015-02-32", "%Y-%m-%d", r"^unconverted data remains when parsing with format \".*\": \"2\".*"),
            ("2015-32-02", "%Y-%d-%m", r"^time data \"2015-32-02\" doesn't match format \".*\".*"),
            ("2015-04-31", "%Y-%m-%d", r"^day is out of range for month. .*"),
            ("2015-31-04", "%Y-%d-%m", r"^day is out of range for month. .*"),
        ],
    )
    def test_day_not_in_month_raise_value(
        self, cache: Any, arg: str, format: str, msg: str
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            to_datetime(arg, errors="raise", format=format, cache=cache)

class TestDatetimeParsingWrappers:
    @pytest.mark.parametrize(
        "date_str, yearfirst, dayfirst, expected",
        [
            ("10-11-12", False, False, datetime(2012, 10, 11)),
            ("10-11-12", True, False, datetime(2012, 11, 10)),
            ("10-11-12", False, True, datetime(2010, 11, 12)),
            ("10-11-12", True, True, datetime(2010, 12, 11)),
            ("20/12/21", False, False, datetime(2021, 12, 20)),
            ("20/12/21", True, False, datetime(2021, 12, 20)),
            ("20/12/21", False, True, datetime(2020, 12, 21)),
            ("20/12/21", True, True, datetime(2020, 12, 21)),
            ("20201012", True, False, datetime(2020, 12, 10)),
        ],
    )
    def test_parsers_dayfirst_yearfirst(
        self,
        cache: Any,
        date_str: str,
        yearfirst: bool,
        dayfirst: bool,
        expected: datetime,
    ) -> None:
        dateutil_result: datetime = parse(
            date_str, dayfirst=dayfirst, yearfirst=yearfirst
        )
        assert dateutil_result == expected
        result1, reso_attrname: Tuple[Optional[datetime], Optional[str]] = parsing.parse_datetime_string_with_reso(
            date_str, yearfirst=yearfirst, dayfirst=dayfirst
        )
        if not dayfirst and not yearfirst:
            result2: Timestamp = Timestamp(date_str)
            assert result2 == expected
        result3: Timestamp = to_datetime(
            date_str, yearfirst=yearfirst, dayfirst=dayfirst, cache=cache
        )
        result4: Timestamp = DatetimeIndex([date_str], yearfirst=yearfirst, dayfirst=dayfirst)[0]
        assert result1 == expected
        assert result3 == expected
        assert result4 == expected

    @pytest.mark.parametrize(
        "value, fmt, dt",
        [
            ["01/10/2010 15:20", "%m/%d/%Y %H:%M", Timestamp("2010-01-10 15:20")],
            ["01/10/2010 05:43", "%m/%d/%Y %I:%M", Timestamp("2010-01-10 05:43")],
            ["01/10/2010 13:56:01", "%m/%d/%Y %H:%M:%S", Timestamp("2010-01-10 13:56:01")],
            pytest.param(
                "01/10/2010 08:14 PM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 20:14"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 07:40 AM",
                "%m/%d/%Y %I:%M %p",
                Timestamp("2010-01-10 07:40"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
            pytest.param(
                "01/10/2010 09:12:56 AM",
                "%m/%d/%Y %I:%M:%S %p",
                Timestamp("2010-01-10 09:12:56"),
                marks=pytest.mark.xfail(
                    locale.getlocale()[0] in ("zh_CN", "it_IT"),
                    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
                    strict=False,
                ),
            ),
        ],
    )
    def test_to_datetime_format_time(
        self, cache: Any, value: str, fmt: str, dt: Timestamp
    ) -> None:
        assert to_datetime(value, format=fmt, cache=cache) == dt

    @pytest.mark.parametrize(
        "format, warning",
        [
            (None, UserWarning),
            ("%Y-%m-%d %H:%M:%S", None),
            ("%Y-%d-%m %H:%M:%S", None),
        ],
    )
    def test_to_datetime_out_of_bounds_with_format_arg(
        self, format: Optional[str], warning: Optional[Any]
    ) -> None:
        if format is None:
            res: Timestamp = to_datetime("2417-10-10 00:00:00.00", format=format)
            assert isinstance(res, Timestamp)
            assert res.year == 2417
            assert res.month == 10
            assert res.day == 10
        else:
            msg: str = 'unconverted data remains when parsing with format ".*": ".*". .*'
            with pytest.raises(ValueError, match=msg):
                to_datetime("2417-10-10 00:00:00.00", format=format)

    def test_to_datetime_mixed_iso8601_raise(self) -> None:
        with pytest.raises(ValueError, match='Time data 01-01-2000 is not ISO8601 format'):
            to_datetime(["2020-01-01", "01-01-2000"], format="ISO8601")

    def test_to_datetime_mixed_iso8601_coerce(self) -> None:
        result: DatetimeIndex = to_datetime(
            ["2020-01-01", "01-01-2000"], format="ISO8601", errors="coerce"
        )
        expected: DatetimeIndex = DatetimeIndex(["2020-01-01 00:00:00", NaT])
        tm.assert_index_equal(result, expected)

    def test_unknown_tz_raises(self) -> None:
        dtstr: str = "2014 Jan 9 05:15 FAKE"
        msg: str = '.*un-recognized timezone "FAKE".'
        with pytest.raises(ValueError, match=msg):
            Timestamp(dtstr)
        with pytest.raises(ValueError, match=msg):
            to_datetime(dtstr)
        with pytest.raises(ValueError, match=msg):
            to_datetime([dtstr])

    def test_unformatted_input_raises(self) -> None:
        valid: str = "2012-01-01 00:00:00"
        invalid: str = "1"
        ser: Series = Series([1, "1"])
        msg: str = "^Given date string \"1\" not likely a datetime$"
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, errors="raise")

    def test_string_invalid_operation(self, cache: Any) -> None:
        invalid: np.ndarray = np.array(["87156549591102612381000001219H5"], dtype=object)
        with pytest.raises(ValueError, match="Unknown datetime string format"):
            to_datetime(invalid, errors="raise", cache=cache)

    def test_string_na_nat_conversion(self, cache: Any) -> None:
        strings: np.ndarray = np.array(
            ["1/1/2000", "1/2/2000", np.nan, "1/4/2000"], dtype=object
        )
        expected: np.ndarray = np.empty(4, dtype="M8[s]")
        for i, val in enumerate(strings):
            if isna(val):
                expected[i] = iNaT
            else:
                expected[i] = parse(val)
        result: DatetimeIndex = tslib.array_to_datetime(strings)[0]
        tm.assert_almost_equal(result, expected)
        result2: DatetimeIndex = to_datetime(strings, cache=cache)
        assert isinstance(result2, DatetimeIndex)
        tm.assert_numpy_array_equal(result, result2.values)

    def test_string_na_nat_conversion_malformed(self, cache: Any) -> None:
        malformed: np.ndarray = np.array(["1/100/2000", np.nan], dtype=object)
        msg: str = "Unknown datetime string format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)

    def test_string_na_nat_conversion_with_name(self, cache: Any) -> None:
        idx: List[str] = ["a", "b", "c", "d", "e"]
        series: Series = Series(
            ["1/1/2000", np.nan, "1/3/2000", np.nan, "1/5/2000"], index=idx, name="foo"
        )
        dseries: Series = Series(
            [
                to_datetime("1/1/2000", cache=cache),
                np.nan,
                to_datetime("1/3/2000", cache=cache),
                np.nan,
                to_datetime("1/5/2000", cache=cache),
            ],
            index=idx,
            name="foo",
        )
        result: Series = to_datetime(series, cache=cache)
        expected: Series = Series(
            [to_datetime("1/1/2000", cache=cache), NaT, to_datetime("1/3/2000", cache=cache), NaT, to_datetime("1/5/2000", cache=cache)],
            index=idx,
            dtype="M8[s]",
        )
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name == "foo"
        tm.assert_series_equal(dresult, expected, check_names=False)
        assert dresult.name == "foo"

    def test_to_datetime_wrapped_datetime64_ps(self) -> None:
        result: DatetimeIndex = to_datetime([np.datetime64(1901901901901, "ps")])
        expected: DatetimeIndex = DatetimeIndex(
            ["1970-01-01 00:00:01.901901901"], dtype="datetime64[ns]", freq=None
        )
        tm.assert_index_equal(result, expected)


@pytest.fixture(params=["D", "s", "ms", "us", "ns"])
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
    return date_range("2014-1-1", periods=10).to_julian_date().values


class TestShouldCache:
    @pytest.mark.parametrize(
        "listlike,do_caching",
        [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], False),
            ([1, 1, 1, 1, 4, 5, 6, 7, 8, 9], True),
        ],
    )
    def test_should_cache(
        self, listlike: List[Any], do_caching: bool
    ) -> None:
        assert tools.should_cache(listlike, check_count=len(listlike), unique_share=0.7) == do_caching

    @pytest.mark.parametrize(
        "unique_share,check_count, err_message",
        [
            (0.5, 11, r"check_count must be in next bounds: \[0; len\(arg\)\]"),
            (10, 2, r"unique_share must be in next bounds: \(0; 1\)"),
        ],
    )
    def test_should_cache_errors(
        self, unique_share: float, check_count: int, err_message: str
    ) -> None:
        arg: List[Any] = [5] * 10
        with pytest.raises(AssertionError, match=err_message):
            tools.should_cache(arg, unique_share, check_count)

    @pytest.mark.parametrize(
        "listlike",
        [
            deque([Timestamp("2010-06-02 09:30:00")] * 51),
            [Timestamp("2010-06-02 09:30:00")] * 51,
            tuple([Timestamp("2010-06-02 09:30:00")] * 51),
        ],
    )
    def test_no_slicing_errors_in_should_cache(
        self, listlike: Union[List[Timestamp], deque, tuple]
    ) -> None:
        assert tools.should_cache(listlike) is True


def test_nullable_integer_to_datetime() -> None:
    ser: Series = Series([1, 2, None, 2**61, None])
    ser = ser.astype("Int64")
    ser_copy: Series = ser.copy()
    res: DatetimeIndex = to_datetime(ser, unit="ns")
    expected: DatetimeIndex = DatetimeIndex(
        [
            np.datetime64("1970-01-01 00:00:00.000000001"),
            np.datetime64("1970-01-01 00:00:00.000000002"),
            np.datetime64("NaT"),
            np.datetime64("2043-01-25 23:56:49.213693952"),
            np.datetime64("NaT"),
        ]
    )
    tm.assert_series_equal(res, expected)
    tm.assert_series_equal(ser, ser_copy)


@pytest.mark.parametrize(
    "klass",
    [np.array, list],
)
def test_na_to_datetime(nulls_fixture: Any, klass: Any) -> None:
    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match="not convertible to datetime"):
            to_datetime(klass([nulls_fixture]))
    else:
        result: DatetimeIndex = to_datetime(klass([nulls_fixture]))
        assert result[0] is NaT


@pytest.mark.parametrize(
    "errors, args, format",
    [
        (["raise", "coerce"],
         [["03/24/2016", "03/25/2016", ""], "%m/%d/%Y"],
         [["2016-03-24", "2016-03-25", "NaT"], ["2016-03-24", "2016-03-25", "NaT"]]),
        (["raise", "coerce"],
         [["2016-03-24", "2016-03-25", ""], "%Y-%m-%d"],
         [["2016-03-24", "2016-03-25", "NaT"], ["2016-03-24", "2016-03-25", "NaT"]]),
    ],
    ids=["non-ISO8601", "ISO8601"],
)
def test_empty_string_datetime(
    errors: str,
    args: List[Any],
    format: Optional[str],
) -> None:
    ser: Series = Series(args[0])
    result: DatetimeIndex = to_datetime(ser, format=format, errors=errors)
    expected: Series = Series(
        ["2016-03-24", "2016-03-25", NaT], dtype="datetime64[s]"
    )
    tm.assert_series_equal(expected, result)

def test_empty_string_datetime_coerce__unit() -> None:
    result: DatetimeIndex = to_datetime([1, ""], unit="s", errors="coerce")
    expected: DatetimeIndex = DatetimeIndex(["1970-01-01 00:00:01", "NaT"], dtype="datetime64[ns]")
    tm.assert_index_equal(expected, result)
    result = to_datetime([1, ""], unit="s", errors="raise")
    tm.assert_index_equal(expected, result)

def test_to_datetime_monotonic_increasing_index(cache: Any) -> None:
    cstart: int = start_caching_at
    times: pd.DatetimeIndex = date_range(Timestamp("1980"), periods=cstart, freq="YS")
    times = times.to_frame(index=False, name="DT").sample(n=cstart, random_state=1)
    times.index = times.index.to_series().astype(float) / 1000
    result: Series = to_datetime(times.iloc[:, 0], cache=cache)
    expected: Series = times.iloc[:, 0]
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "series_length",
    [40, start_caching_at, start_caching_at + 1, start_caching_at + 5],
)
def test_to_datetime_cache_coerce_50_lines_outofbounds(series_length: int) -> None:
    ser: Series = Series(
        [datetime.fromisoformat("1446-04-12 00:00:00+00:00")] + [datetime.fromisoformat("1991-10-20 00:00:00+00:00")] * series_length,
        dtype=object,
    )
    result1: DatetimeIndex = to_datetime(
        ser, errors="coerce", utc=True, cache=True
    )
    expected1: Series = Series([Timestamp(x) for x in ser])
    assert expected1.dtype == "datetime64[us, UTC]"
    tm.assert_series_equal(result1, expected1)

def test_to_datetime_format_f_parse_nanos():
    timestamp: str = "15/02/2020 02:03:04.123456789"
    timestamp_format: str = "%d/%m/%Y %H:%M:%S.%f"
    result: Timestamp = to_datetime(timestamp, format=timestamp_format)
    expected: Timestamp = Timestamp(year=2020, month=2, day=15, hour=2, minute=3, second=4, microsecond=123456, nanosecond=789)
    assert result == expected

def test_to_datetime_mixed_iso8601() -> None:
    result: DatetimeIndex = to_datetime(["2020-01-01", "2020-01-01 05:00:00"], format="ISO8601")
    expected: DatetimeIndex = DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 05:00:00"])
    tm.assert_index_equal(result, expected)

def test_to_datetime_mixed_other() -> None:
    result: DatetimeIndex = to_datetime(["01/11/2000", "12 January 2000"], format="mixed")
    expected: DatetimeIndex = DatetimeIndex(["2000-01-11", "2000-01-12"])
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize(
    "exact, format",
    [
        (True, "mixed"),
        (True, "ISO8601"),
    ],
)
def test_to_datetime_mixed_or_iso_exact(exact: bool, format: str) -> None:
    msg: str = "Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'"
    with pytest.raises(ValueError, match=msg):
        to_datetime(["2020-01-01"], exact=exact, format=format)

def test_to_datetime_mixed_not_necessarily_iso8601_raise() -> None:
    with pytest.raises(ValueError, match='Time data 01-01-2000 is not ISO8601 format'):
        to_datetime(["2020-01-01", "01-01-2000"], format="ISO8601")

def test_to_datetime_mixed_not_necessarily_iso8601_coerce() -> None:
    result: DatetimeIndex = to_datetime(
        ["2020-01-01", "01-01-2000"], format="ISO8601", errors="coerce"
    )
    expected: DatetimeIndex = DatetimeIndex(["2020-01-01 00:00:00", NaT])
    tm.assert_index_equal(result, expected)

def test_unknown_tz_raises() -> None:
    dtstr: str = "2014 Jan 9 05:15 FAKE"
    msg: str = '.*un-recognized timezone "FAKE".'
    with pytest.raises(ValueError, match=msg):
        Timestamp(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime([dtstr])

def test_unformatted_input_raises() -> None:
    valid: str = "2012-01-01 00:00:00"
    invalid: str = "1"
    ser: Series = Series([1, "1"])
    msg: str = "^Given date string \"1\" not likely a datetime$"
    with pytest.raises(ValueError, match=msg):
        to_datetime(ser, errors="raise")

def test_string_invalid_operation(cache: Any) -> None:
    invalid: np.ndarray = np.array(["87156549591102612381000001219H5"], dtype=object)
    with pytest.raises(ValueError, match="Unknown datetime string format"):
        to_datetime(invalid, errors="raise", cache=cache)

def test_string_na_nat_conversion(cache: Any) -> None:
    strings: np.ndarray = np.array(
        ["1/1/2000", "1/2/2000", np.nan, "1/4/2000"], dtype=object
    )
    expected: np.ndarray = np.empty(4, dtype="M8[s]")
    for i, val in enumerate(strings):
        if isna(val):
            expected[i] = iNaT
        else:
            expected[i] = parse(val)
    result: DatetimeIndex = tslib.array_to_datetime(strings)[0]
    tm.assert_almost_equal(result, expected)
    result2: DatetimeIndex = to_datetime(strings, cache=cache)
    assert isinstance(result2, DatetimeIndex)
    tm.assert_numpy_array_equal(result, result2.values)

def test_string_na_nat_conversion_malformed(cache: Any) -> None:
    malformed: np.ndarray = np.array(["1/100/2000", np.nan], dtype=object)
    msg: str = "Unknown datetime string format"
    with pytest.raises(ValueError, match=msg):
        to_datetime(malformed, errors="raise", cache=cache)
    with pytest.raises(ValueError, match=msg):
        to_datetime(malformed, errors="raise", cache=cache)

def test_string_na_nat_conversion_with_name(cache: Any) -> None:
    idx: List[str] = ["a", "b", "c", "d", "e"]
    series: Series = Series(
        ["1/1/2000", np.nan, "1/3/2000", np.nan, "1/5/2000"], index=idx, name="foo"
    )
    dseries: Series = Series(
        [
            to_datetime("1/1/2000", cache=cache),
            np.nan,
            to_datetime("1/3/2000", cache=cache),
            np.nan,
            to_datetime("1/5/2000", cache=cache),
        ],
        index=idx,
        name="foo",
    )
    result: Series = to_datetime(series, cache=cache)
    expected: Series = Series(
        [to_datetime("1/1/2000", cache=cache), NaT, to_datetime("1/3/2000", cache=cache), NaT, to_datetime("1/5/2000", cache=cache)],
        index=idx,
        dtype="M8[s]",
    )
    tm.assert_series_equal(result, expected, check_names=False)
    assert result.name == "foo"
    tm.assert_series_equal(dresult, expected, check_names=False)
    assert dresult.name == "foo"

def test_to_datetime_wrapped_datetime64_ps() -> None:
    result: DatetimeIndex = to_datetime([np.datetime64(1901901901901, "ps")])
    expected: DatetimeIndex = DatetimeIndex(
        ["1970-01-01 00:00:01.901901901"], dtype="datetime64[ns]", freq=None
    )
    tm.assert_index_equal(result, expected)
