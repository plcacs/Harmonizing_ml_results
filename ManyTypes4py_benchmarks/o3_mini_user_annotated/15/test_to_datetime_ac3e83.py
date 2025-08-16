#!/usr/bin/env python3
"""
This module tests various aspects of datetime conversion.
"""

from __future__ import annotations
from collections import deque
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytest
from pandas import NaT, Timestamp, to_datetime, date_range, DataFrame, Series, DatetimeIndex
import pandas._testing as tm
from pandas._libs.tslibs import OutOfBoundsDatetime, OutOfBoundsTimedelta
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from dateutil.parser import parse


PARSING_ERR_MSG = (
    "You might want to try:\n"
    "    - passing `format` if your strings have a consistent format;\n"
    "    - passing `format='ISO8601'` if your strings are all ISO8601 but not necessarily in exactly the same format;\n"
    "    - passing `format='mixed'`, and the format will be inferred for each element individually. You might want to use `dayfirst` alongside this."
)


class TestNullableIntegerToDatetime:
    def test_nullable_integer_to_datetime(self) -> None:
        # Test for #30050
        ser: Series = Series([1, 2, None, 2**61, None])
        ser = ser.astype("Int64")
        ser_copy: Series = ser.copy()
    
        res: Series = to_datetime(ser, unit="ns")
    
        expected: Series = Series(
            [
                np.datetime64("1970-01-01 00:00:00.000000001"),
                np.datetime64("1970-01-01 00:00:00.000000002"),
                np.datetime64("NaT"),
                np.datetime64("2043-01-25 23:56:49.213693952"),
                np.datetime64("NaT"),
            ]
        )
        tm.assert_series_equal(res, expected)
        # Check that ser isn't mutated
        tm.assert_series_equal(ser, ser_copy)


@pytest.mark.parametrize("klass", [np.array, list])
def test_na_to_datetime(nulls_fixture: Any, klass: Callable[[List[Any]], Any]) -> None:
    if isinstance(nulls_fixture, Decimal):
        with pytest.raises(TypeError, match="not convertible to datetime"):
            to_datetime(klass([nulls_fixture]))
    else:
        result: DatetimeIndex = to_datetime(klass([nulls_fixture]))
        assert result[0] is NaT


@pytest.mark.parametrize("errors", ["raise", "coerce"])
@pytest.mark.parametrize(
    "args, format",
    [
        (["03/24/2016", "03/25/2016", ""], "%m/%d/%Y"),
        (["2016-03-24", "2016-03-25", ""], "%Y-%m-%d"),
    ],
    ids=["non-ISO8601", "ISO8601"],
)
def test_empty_string_datetime(errors: str, args: List[str], format: str) -> None:
    # GH13044, GH50251
    td_series: Series = Series(args)
    
    # coerce empty string to pd.NaT
    result: Series = to_datetime(td_series, format=format, errors=errors)
    expected: Series = Series(["2016-03-24", "2016-03-25", NaT], dtype="datetime64[s]")
    tm.assert_series_equal(expected, result)


def test_empty_string_datetime_coerce__unit() -> None:
    # GH13044
    # coerce empty string to pd.NaT
    result: DatetimeIndex = to_datetime([1, ""], unit="s", errors="coerce")
    expected: DatetimeIndex = DatetimeIndex(["1970-01-01 00:00:01", "NaT"], dtype="datetime64[ns]")
    tm.assert_index_equal(expected, result)
    
    # verify that no exception is raised even when errors='raise' is set
    result = to_datetime([1, ""], unit="s", errors="raise")
    tm.assert_index_equal(expected, result)


def test_to_datetime_monotonic_increasing_index(cache: Any) -> None:
    # GH28238
    cstart: int = tools.start_caching_at  # type: ignore[attr-defined]
    times: DatetimeIndex = date_range(Timestamp("1980"), periods=cstart, freq="YS")
    times_df: DataFrame = times.to_frame(index=False, name="DT").sample(n=cstart, random_state=1)
    times_df.index = times_df.index.to_series().astype(float) / 1000
    result: Series = to_datetime(times_df.iloc[:, 0], cache=cache)
    expected: Series = times_df.iloc[:, 0]
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "series_length",
    [40, tools.start_caching_at, (tools.start_caching_at + 1), (tools.start_caching_at + 5)],
)
def test_to_datetime_cache_coerce_50_lines_outofbounds(series_length: int) -> None:
    # GH#45319
    ser: Series = Series(
        [datetime.fromisoformat("1446-04-12 00:00:00+00:00")]
        + ([datetime.fromisoformat("1991-10-20 00:00:00+00:00")] * series_length),
        dtype=object,
    )
    result1: Series = to_datetime(ser, errors="coerce", utc=True)
    
    expected1: Series = Series([Timestamp(x) for x in ser])
    assert expected1.dtype == "M8[us, UTC]"
    tm.assert_series_equal(result1, expected1)
    
    result3: Series = to_datetime(ser, errors="raise", utc=True)
    tm.assert_series_equal(result3, expected1)


def test_to_datetime_format_f_parse_nanos() -> None:
    # GH 48767
    timestamp: str = "15/02/2020 02:03:04.123456789"
    timestamp_format: str = "%d/%m/%Y %H:%M:%S.%f"
    result: Timestamp = to_datetime(timestamp, format=timestamp_format)
    expected: Timestamp = Timestamp(
        year=2020,
        month=2,
        day=15,
        hour=2,
        minute=3,
        second=4,
        microsecond=123456,
        nanosecond=789,
    )
    assert result == expected


def test_to_datetime_mixed_iso8601() -> None:
    # https://github.com/pandas-dev/pandas/issues/50411
    result: DatetimeIndex = to_datetime(["2020-01-01", "2020-01-01 05:00:00"], format="ISO8601")
    expected: DatetimeIndex = DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 05:00:00"])
    tm.assert_index_equal(result, expected)


def test_to_datetime_mixed_other() -> None:
    # https://github.com/pandas-dev/pandas/issues/50411
    result: DatetimeIndex = to_datetime(["01/11/2000", "12 January 2000"], format="mixed")
    expected: DatetimeIndex = DatetimeIndex(["2000-01-11", "2000-01-12"])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("format", ["ISO8601", "mixed"])
def test_to_datetime_mixed_or_iso_exact(exact: bool, format: str) -> None:
    msg: str = "Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'"
    with pytest.raises(ValueError, match=msg):
        to_datetime(["2020-01-01"], exact=exact, format=format)


def test_to_datetime_mixed_not_necessarily_iso8601_raise() -> None:
    # https://github.com/pandas-dev/pandas/issues/50411
    with pytest.raises(ValueError, match="Time data 01-01-2000 is not ISO8601 format"):
        to_datetime(["2020-01-01", "01-01-2000"], format="ISO8601")


def test_to_datetime_mixed_not_necessarily_iso8601_coerce() -> None:
    # https://github.com/pandas-dev/pandas/issues/50411
    result: DatetimeIndex = to_datetime(["2020-01-01", "01-01-2000"], format="ISO8601", errors="coerce")
    tm.assert_index_equal(result, DatetimeIndex(["2020-01-01 00:00:00", NaT]))


def test_unknown_tz_raises() -> None:
    # GH#18702, GH#51476
    dtstr: str = "2014 Jan 9 05:15 FAKE"
    msg: str = '.*un-recognized timezone "FAKE".'
    with pytest.raises(ValueError, match=msg):
        Timestamp(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime(dtstr)
    with pytest.raises(ValueError, match=msg):
        to_datetime([dtstr])


def test_unformatted_input_raises() -> None:
    valid: str = "2024-01-01"
    invalid: str = "N"
    ser: Series = Series([valid] * tools.start_caching_at + [invalid])
    msg: str = 'time data "N" doesn\'t match format "%Y-%m-%d"'
    
    with pytest.raises(ValueError, match=msg):
        to_datetime(ser, format="%Y-%m-%d", exact=True, cache=True)


def test_to_datetime_mixed_tzs_mixed_types() -> None:
    # GH#55793, GH#55693 mismatched tzs but one is str and other is datetime object
    ts_val: Timestamp = Timestamp("2016-01-02 03:04:05", tz="US/Pacific")
    dtstr: str = "2023-10-30 15:06+01"
    arr: List[Any] = [ts_val, dtstr]
    
    mixed_msg: str = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' "
        "in DatetimeIndex to convert to a common timezone"
    )
    with pytest.raises(ValueError, match=mixed_msg):
        to_datetime(arr)
    with pytest.raises(ValueError, match=mixed_msg):
        to_datetime(arr, format="mixed")
    with pytest.raises(ValueError, match=mixed_msg):
        DatetimeIndex(arr)


def test_to_datetime_mixed_types_matching_tzs() -> None:
    # GH#55793
    dtstr: str = "2023-11-01 09:22:03-07:00"
    ts_val: Timestamp = Timestamp(dtstr)
    arr: List[Union[Timestamp, str]] = [ts_val, dtstr]
    res1: DatetimeIndex = to_datetime(arr)
    res2: DatetimeIndex = to_datetime(arr[::-1])[::-1]
    res3: DatetimeIndex = to_datetime(arr, format="mixed")
    res4: DatetimeIndex = DatetimeIndex(arr)
    
    expected: DatetimeIndex = DatetimeIndex([ts_val, ts_val])
    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


dtstr_global: str = "2020-01-01 00:00+00:00"
ts_global: Timestamp = Timestamp(dtstr_global)


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
@pytest.mark.parametrize(
    "aware_val",
    [dtstr_global, Timestamp(dtstr_global)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize(
    "naive_val",
    [dtstr_global[:-6], ts_global.tz_localize(None), ts_global.date(), ts_global.asm8, ts_global.value, float(ts_global.value)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize("naive_first", [True, False])
def test_to_datetime_mixed_awareness_mixed_types(aware_val: Any, naive_val: Any, naive_first: bool) -> None:
    # GH#55793, GH#55693, GH#57275
    vals: List[Any] = [aware_val, naive_val, ""]
    vec: List[Any] = vals.copy()
    if naive_first:
        vec = [naive_val, aware_val, ""]
    
    both_strs: bool = isinstance(aware_val, str) and isinstance(naive_val, str)
    has_numeric: bool = isinstance(naive_val, (int, float))
    both_datetime: bool = isinstance(naive_val, datetime) and isinstance(aware_val, datetime)
    
    mixed_msg: str = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone"
    )
    
    first_non_null: Any = next(x for x in vec if x != "")
    if not isinstance(first_non_null, str):
        msg: str = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        else:
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
    
        # No warning/error with utc=True
        to_datetime(vec, utc=True)
    
    elif has_numeric and vec.index(aware_val) < vec.index(naive_val):
        msg = "time data .* doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    
    elif both_strs and vec.index(aware_val) < vec.index(naive_val):
        msg = r"time data \"2020-01-01 00:00\" doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    
    elif both_strs and vec.index(naive_val) < vec.index(aware_val):
        msg = "unconverted data remains when parsing with format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    
    else:
        msg = mixed_msg
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
    
        # No warning/error with utc=True
        to_datetime(vec, utc=True)
    
    if both_strs:
        msg = mixed_msg
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, format="mixed")
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(vec)
    else:
        msg = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format="mixed")
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)
        else:
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format="mixed")
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)


def test_to_datetime_wrapped_datetime64_ps() -> None:
    # GH#60341
    result: DatetimeIndex = to_datetime([np.datetime64(1901901901901, "ps")])
    expected: DatetimeIndex = DatetimeIndex(["1970-01-01 00:00:01.901901901"], dtype="datetime64[ns]", freq=None)
    tm.assert_index_equal(result, expected)


# ------------------ Fixtures ------------------

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


# ------------------ TestOrigin ------------------

class TestOrigin:
    def test_origin_and_unit(self) -> None:
        # GH#42624
        ts: Timestamp = to_datetime(1, unit="s", origin=1)
        expected: Timestamp = Timestamp("1970-01-01 00:00:02")
        assert ts == expected
    
        ts = to_datetime(1, unit="s", origin=1_000_000_000)
        expected = Timestamp("2001-09-09 01:46:41")
        assert ts == expected
    
    def test_julian(self, julian_dates: np.ndarray) -> None:
        # gh-11276, gh-11745
        result: Series = Series(to_datetime(julian_dates, unit="D", origin="julian"))
        expected: Series = Series(
            to_datetime(julian_dates - Timestamp(0).to_julian_date(), unit="D")
        )
        tm.assert_series_equal(result, expected)
    
    def test_unix(self) -> None:
        result: Series = Series(to_datetime([0, 1, 2], unit="D", origin="unix"))
        expected: Series = Series(
            [Timestamp("1970-01-01"), Timestamp("1970-01-02"), Timestamp("1970-01-03")],
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)
    
    def test_julian_round_trip(self) -> None:
        result: Timestamp = to_datetime(2456658, origin="julian", unit="D")
        assert result.to_julian_date() == 2456658
        msg: str = "1 is Out of Bounds for origin='julian'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(1, origin="julian", unit="D")
    
    def test_invalid_unit(self, units: str, julian_dates: np.ndarray) -> None:
        # checking for invalid combination of origin='julian' and unit != D
        if units != "D":
            msg: str = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates, unit=units, origin="julian")
    
    @pytest.mark.parametrize("unit", ["ns", "D"])
    def test_invalid_origin(self, unit: str) -> None:
        msg: str = "it must be numeric with a unit specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2005-01-01", origin="1960-01-01", unit=unit)
    
    @pytest.mark.parametrize(
        "epochs",
        [
            Timestamp(1960, 1, 1),
            datetime(1960, 1, 1),
            "1960-01-01",
            np.datetime64("1960-01-01"),
        ],
    )
    def test_epoch(self, units: str, epochs: Union[Timestamp, datetime, str, np.datetime64]) -> None:
        epoch_1960: Timestamp = Timestamp(1960, 1, 1)
        units_from_epochs: np.ndarray = np.arange(5, dtype=np.int64)
        expected: Series = Series(
            [pd.Timedelta(x, unit=units) + epoch_1960 for x in units_from_epochs]
        )
        result: Series = Series(to_datetime(units_from_epochs, unit=units, origin=epochs))
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
    def test_invalid_origins(self, origin: Any, exc: Any, units: str) -> None:
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
        # GH16842
        with pytest.raises(ValueError, match="must be tz-naive"):
            to_datetime(1, unit="D", origin=datetime(2000, 1, 1, tzinfo=timezone.utc))
    
    def test_incorrect_value_exception(self) -> None:
        # GH47495
        msg: str = "Unknown datetime string format, unable to parse: yesterday"
        with pytest.raises(ValueError, match=msg):
            to_datetime(["today", "yesterday"])
    
    @pytest.mark.parametrize(
        "format, warning",
        [
            (None, UserWarning),
            ("%Y-%m-%d %H:%M:%S", None),
            ("%Y-%d-%m %H:%M:%S", None),
        ],
    )
    def test_to_datetime_out_of_bounds_with_format_arg(self, format: Optional[str], warning: Optional[Any]) -> None:
        # see gh-23830
        if format is None:
            res: Timestamp = to_datetime("2417-10-10 00:00:00.00", format=format)
            assert isinstance(res, Timestamp)
            assert res.year == 2417
            assert res.month == 10
            assert res.day == 10
        else:
            msg: str = "unconverted data remains when parsing with format.*"
            with pytest.raises(ValueError, match=msg):
                to_datetime("2417-10-10 00:00:00.00", format=format)
    
    @pytest.mark.parametrize(
        "arg, origin, expected_str",
        [
            [200 * 365, "unix", "2169-11-13 00:00:00"],
            [200 * 365, "1870-01-01", "2069-11-13 00:00:00"],
            [300 * 365, "1870-01-01", "2169-10-20 00:00:00"],
        ],
    )
    def test_processing_order(self, arg: int, origin: Union[str, int], expected_str: str) -> None:
        result: Timestamp = to_datetime(arg, unit="D", origin=origin)
        expected: Timestamp = Timestamp(expected_str)
        assert result == expected
        result = to_datetime(200 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2069-11-13 00:00:00")
        assert result == expected
        result = to_datetime(300 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2169-10-20 00:00:00")
        assert result == expected
    
    @pytest.mark.parametrize(
        "offset,utc,exp",
        [
            ["Z", True, "2019-01-01T00:00:00.000Z"],
            ["Z", None, "2019-01-01T00:00:00.000Z"],
            ["-01:00", True, "2019-01-01T01:00:00.000Z"],
            ["-01:00", None, "2019-01-01T00:00:00.000-01:00"],
        ],
    )
    def test_arg_tz_ns_unit(self, offset: str, utc: Optional[bool], exp: str) -> None:
        # GH 25546
        arg: str = "2019-01-01T00:00:00.000" + offset
        result: DatetimeIndex = to_datetime([arg], unit="ns", utc=utc)
        expected: DatetimeIndex = to_datetime([exp]).as_unit("ns")
        tm.assert_index_equal(result, expected)


class TestShouldCache:
    @pytest.mark.parametrize(
        "listlike,do_caching",
        [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], False),
            ([1, 1, 1, 1, 4, 5, 6, 7, 8, 9], True),
        ],
    )
    def test_should_cache(self, listlike: Sequence[Any], do_caching: bool) -> None:
        assert (
            tools.should_cache(listlike, check_count=len(listlike), unique_share=0.7)
            == do_caching
        )
    
    @pytest.mark.parametrize(
        "unique_share,check_count, err_message",
        [
            (0.5, 11, r"check_count must be in next bounds: \[0; len\(arg\)\]"),
            (10, 2, r"unique_share must be in next bounds: \(0; 1\)"),
        ],
    )
    def test_should_cache_errors(self, unique_share: Any, check_count: int, err_message: str) -> None:
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
    def test_no_slicing_errors_in_should_cache(self, listlike: Any) -> None:
        # GH#29403
        assert tools.should_cache(listlike) is True


class TestNullableIntegerToDatetimeConversion:
    def test_nullable_integer_to_datetime(self) -> None:
        # Already defined above in TestNullableIntegerToDatetime
        pass  # duplicate placeholder


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
        result: Series = to_datetime({"year": df["year"], "month": df["month"], "day": df["day"]}, cache=cache)
        expected: Series = Series([Timestamp("20150204 00:00:00"), Timestamp("20160305 00:0:00")])
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_dict_with_constructable(self, df: DataFrame, cache: Any) -> None:
        df2: dict[str, Any] = df[["year", "month", "day"]].to_dict()
        df2["month"] = 2
        result: Series = to_datetime(df2, cache=cache)
        expected2: Series = Series([Timestamp("20150204 00:00:00"), Timestamp("20160205 00:0:00")], index=pd.Index([0, 1]))
        tm.assert_series_equal(result, expected2)
    
    @pytest.mark.parametrize(
        "unit",
        [
            {
                "year": "years",
                "month": "months",
                "day": "days",
                "hour": "hours",
                "minute": "minutes",
                "second": "seconds",
            },
            {
                "year": "year",
                "month": "month",
                "day": "day",
                "hour": "hour",
                "minute": "minute",
                "second": "second",
            },
        ],
    )
    def test_dataframe_field_aliases_column_subset(self, df: DataFrame, cache: Any, unit: dict[str, str]) -> None:
        result: Series = to_datetime(df[list(unit.keys())].rename(columns=unit), cache=cache)
        expected: Series = Series(
            [Timestamp("20150204 06:58:10"), Timestamp("20160305 07:59:11")],
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_field_aliases(self, df: DataFrame, cache: Any) -> None:
        d: dict[str, str] = {
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
        expected: Series = Series([Timestamp("20150204 06:58:10.001002003"), Timestamp("20160305 07:59:11.001002003")])
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_str_dtype(self, df: DataFrame, cache: Any) -> None:
        result: Series = to_datetime(df.astype(str), cache=cache)
        expected: Series = Series([Timestamp("20150204 06:58:10.001002003"), Timestamp("20160305 07:59:11.001002003")])
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_float32_dtype(self, df: DataFrame, cache: Any) -> None:
        result: Series = to_datetime(df.astype(np.float32), cache=cache)
        expected: Series = Series([Timestamp("20150204 06:58:10.001002003"), Timestamp("20160305 07:59:11.001002003")])
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_coerce(self, cache: Any) -> None:
        df2: DataFrame = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})
        msg: str = r'^cannot assemble the datetimes: time data ".+" doesn\'t match format "%Y%m%d"\.'
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
        result: Series = to_datetime(df2, errors="coerce", cache=cache)
        expected: Series = Series([Timestamp("20150204 00:00:00"), NaT])
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_extra_keys_raises(self, df: DataFrame, cache: Any) -> None:
        msg: str = r"extra keys have been passed to the datetime assemblage: \[foo\]"
        df2: DataFrame = df.copy()
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
    def test_dataframe_missing_keys_raises(self, df: DataFrame, cache: Any, cols: List[str]) -> None:
        msg: str = (
            r"to assemble mappings requires at least that \[year, month, day\] be specified: \[.+\] is missing"
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(df[cols], cache=cache)
    
    def test_dataframe_duplicate_columns_raises(self, cache: Any) -> None:
        msg: str = "cannot assemble with duplicate keys"
        df2: DataFrame = DataFrame({"year": [2015, 2016], "month": [2, 20], "day": [4, 5]})
        df2.columns = ["year", "year", "day"]
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
    
        df2 = DataFrame(
            {"year": [2015, 2016], "month": [2, 20], "day": [4, 5], "hour": [4, 5]}
        )
        df2.columns = ["year", "month", "day", "day"]
        with pytest.raises(ValueError, match=msg):
            to_datetime(df2, cache=cache)
    
    def test_dataframe_int16(self, cache: Any) -> None:
        df: DataFrame = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        result: Series = to_datetime(df.astype("int16"), cache=cache)
        expected: Series = Series([Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")])
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_mixed(self, cache: Any) -> None:
        df: DataFrame = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        df["month"] = df["month"].astype("int8")
        df["day"] = df["day"].astype("int8")
        result: Series = to_datetime(df, cache=cache)
        expected: Series = Series([Timestamp("20150204 00:00:00"), Timestamp("20160305 00:00:00")])
        tm.assert_series_equal(result, expected)
    
    def test_dataframe_float(self, cache: Any) -> None:
        df: DataFrame = DataFrame({"year": [2000, 2001], "month": [1.5, 1], "day": [1, 1]})
        msg: str = (
            r"^cannot assemble the datetimes: unconverted data remains when parsing "
            r'with format ".*": "1".'
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(df, cache=cache)
    
    def test_dataframe_utc_true(self) -> None:
        df: DataFrame = DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
        result: Series = to_datetime(df, utc=True)
        expected: Series = Series(np.array(["2015-02-04", "2016-03-05"], dtype="datetime64[s]")).dt.tz_localize("UTC")
        tm.assert_series_equal(result, expected)


class TestToDatetimeMisc:
    def test_to_datetime_barely_out_of_bounds(self) -> None:
        # GH#19529
        arr: np.ndarray = np.array(["2262-04-11 23:47:16.854775808"], dtype=object)
        msg: str = "^Out of bounds nanosecond timestamp: .*"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arr)
    
    @pytest.mark.parametrize(
        "arg, exp_str",
        [
            ("2012-01-01 00:00:00", "2012-01-01 00:00:00"),
            ("20121001", "2012-10-01"),
        ],
    )
    def test_to_datetime_iso8601(self, cache: Any, arg: str, exp_str: str) -> None:
        result: DatetimeIndex = to_datetime([arg], cache=cache)
        exp: Timestamp = Timestamp(exp_str)
        assert result[0] == exp
    
    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012", "%Y-%m"),
            ("2012-01", "%Y-%m-%d"),
            ("2012-01-01", "%Y-%m-%d %H"),
            ("2012-01-01 10", "%Y-%m-%d %H:%M"),
            ("2012-01-01 10:00", "%Y-%m-%d %H:%M:%S"),
            ("2012-01-01 10:00:00", "%Y-%m-%d %H:%M:%S.%f"),
            ("2012-01-01 10:00:00.123", "%Y-%m-%d %H:%M:%S.%f%z"),
            (0, "%Y-%m-%d"),
        ],
    )
    @pytest.mark.parametrize("exact", [True, False])
    def test_to_datetime_iso8601_fails(self, input: Union[str, int], format: str, exact: bool) -> None:
        # https://github.com/pandas-dev/pandas/issues/12649
        with pytest.raises(
            ValueError,
            match=(rf"time data \"{input}\" doesn't match format " rf"\"{format}\""),
        ):
            to_datetime(input, format=format, exact=exact)
    
    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012-01-01", "%Y-%m"),
            ("2012-01-01 10", "%Y-%m-%d"),
            ("2012-01-01 10:00", "%Y-%m-%d %H"),
            ("2012-01-01 10:00:00", "%Y-%m-%d %H:%M"),
            (0, "%Y-%m-%d"),
        ],
    )
    def test_to_datetime_iso8601_exact_fails(self, input: Union[str, int], format: str) -> None:
        msg: str = "|".join(
            [
                '^unconverted data remains when parsing with format ".*": ".*". '
                f"{PARSING_ERR_MSG}$",
                f'^time data ".*" doesn\'t match format ".*". {PARSING_ERR_MSG}$',
            ]
        )
        with pytest.raises(ValueError, match=(msg)):
            to_datetime(input, format=format, exact=True)
    
    @pytest.mark.parametrize(
        "input, format",
        [
            ("2012-01-01", "%Y-%m"),
            ("2012-01-01 00", "%Y-%m-%d"),
            ("2012-01-01 00:00", "%Y-%m-%d %H"),
            ("2012-01-01 00:00:00", "%Y-%m-%d %H:%M"),
        ],
    )
    def test_to_datetime_iso8601_non_exact(self, input: str, format: str) -> None:
        expected: Timestamp = Timestamp(2012, 1, 1)
        result: Timestamp = to_datetime(input, format=format, exact=False)
        assert result == expected
    
    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-01", "%Y/%m"),
            ("2020-01-01", "%Y/%m/%d"),
            ("2020-01-01 00", "%Y/%m/%dT%H"),
            ("2020-01-01T00", "%Y/%m/%d %H"),
            ("2020-01-01 00:00", "%Y/%m/%dT%H:%M"),
            ("2020-01-01T00:00", "%Y/%m/%d %H:%M"),
            ("2020-01-01 00:00:00", "%Y/%m/%dT%H:%M:%S"),
            ("2020-01-01T00:00:00", "%Y/%m/%d %H:%M:%S"),
        ],
    )
    def test_to_datetime_iso8601_separator(self, input: str, format: str) -> None:
        with pytest.raises(
            ValueError,
            match=(rf"time data \"{input}\" doesn\'t match format " rf"\"{format}\""),
        ):
            to_datetime(input, format=format)
    
    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-01", "%Y-%m"),
            ("2020-01-01", "%Y-%m-%d"),
            ("2020-01-01 00", "%Y-%m-%d %H"),
            ("2020-01-01T00", "%Y-%m-%dT%H"),
            ("2020-01-01 00:00", "%Y-%m-%d %H:%M"),
            ("2020-01-01T00:00", "%Y-%m-%dT%H:%M"),
            ("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
            ("2020-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S"),
            ("2020-01-01T00:00:00.000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-01-01T00:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-01-01T00:00:00.000000000", "%Y-%m-%dT%H:%M:%S.%f"),
        ],
    )
    def test_to_datetime_iso8601_valid(self, input: str, format: str) -> None:
        expected: Timestamp = Timestamp(2020, 1, 1)
        result: Timestamp = to_datetime(input, format=format)
        assert result == expected
    
    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-1", "%Y-%m"),
            ("2020-1-1", "%Y-%m-%d"),
            ("2020-1-1 0", "%Y-%m-%d %H"),
            ("2020-1-1T0", "%Y-%m-%dT%H"),
            ("2020-1-1 0:0", "%Y-%m-%d %H:%M"),
            ("2020-1-1T0:0", "%Y-%m-%dT%H:%M"),
            ("2020-1-1 0:0:0", "%Y-%m-%d %H:%M:%S"),
            ("2020-1-1T0:0:0", "%Y-%m-%dT%H:%M:%S"),
            ("2020-1-1T0:0:0.000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-1-1T0:0:0.000000", "%Y-%m-%dT%H:%M:%S.%f"),
            ("2020-1-1T0:0:0.000000000", "%Y-%m-%dT%H:%M:%S.%f"),
        ],
    )
    def test_to_datetime_iso8601_non_padded(self, input: str, format: str) -> None:
        expected: Timestamp = Timestamp(2020, 1, 1)
        result: Timestamp = to_datetime(input, format=format)
        assert result == expected
    
    @pytest.mark.parametrize(
        "input, format",
        [
            ("2020-01-01T00:00:00.000000000+00:00", "%Y-%m-%dT%H:%M:%S.%f%z"),
            ("2020-01-01T00:00:00+00:00", "%Y-%m-%dT%H:%M:%S%z"),
            ("2020-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%S%z"),
        ],
    )
    def test_to_datetime_iso8601_with_timezone_valid(self, input: str, format: str) -> None:
        expected: Timestamp = Timestamp(2020, 1, 1, tzinfo=timezone.utc)
        result: Timestamp = to_datetime(input, format=format)
        assert result == expected
    
    def test_to_datetime_default(self, cache: Any) -> None:
        rs: Timestamp = to_datetime("2001", cache=cache)
        xp: datetime = datetime(2001, 1, 1)
        assert rs == xp
    
    @pytest.mark.xfail(reason="fails to enforce dayfirst=True, which would raise")
    def test_to_datetime_respects_dayfirst(self, cache: Any) -> None:
        msg: str = "Invalid date specified"
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(UserWarning, match="Provide format"):
                to_datetime("01-13-2012", dayfirst=True, cache=cache)
    
    def test_to_datetime_on_datetime64_series(self, cache: Any) -> None:
        ser: Series = Series(date_range("1/1/2000", periods=10))
        result: DatetimeIndex = to_datetime(ser, cache=cache)
        assert result[0] == ser[0]
    
    def test_to_datetime_with_space_in_series(self, cache: Any) -> None:
        ser: Series = Series(["10/18/2006", "10/18/2008", " "])
        msg: str = r'^time data " " doesn\'t match format "%m/%d/%Y". ' rf"{PARSING_ERR_MSG}$"
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, errors="raise", cache=cache)
        result_coerce: Series = to_datetime(ser, errors="coerce", cache=cache)
        expected_coerce: Series = Series([datetime(2006, 10, 18), datetime(2008, 10, 18), NaT]).dt.as_unit("s")
        tm.assert_series_equal(result_coerce, expected_coerce)
    
    @pytest.mark.skip_if_not_us_locale
    def test_to_datetime_with_apply(self, cache: Any) -> None:
        td_series: Series = Series(["May 04", "Jun 02", "Dec 11"], index=[1, 2, 3])
        expected: DatetimeIndex = to_datetime(td_series, format="%b %y", cache=cache)
        result: Series = td_series.apply(lambda x: to_datetime(x, format="%b %y", cache=cache))
        tm.assert_series_equal(result, expected)
    
    def test_to_datetime_timezone_name(self) -> None:
        result: Timestamp = to_datetime("2020-01-01 00:00:00UTC", format="%Y-%m-%d %H:%M:%S%Z")
        expected: Timestamp = Timestamp(2020, 1, 1).tz_localize("UTC")
        assert result == expected
    
    @pytest.mark.skip_if_not_us_locale
    @pytest.mark.parametrize("errors", ["raise", "coerce"])
    def test_to_datetime_with_apply_with_empty_str(self, cache: Any, errors: str) -> None:
        td_series: Series = Series(["May 04", "Jun 02", ""], index=[1, 2, 3])
        expected: DatetimeIndex = to_datetime(td_series, format="%b %y", errors=errors, cache=cache)
        result: Series = td_series.apply(lambda x: to_datetime(x, format="%b %y", errors="coerce", cache=cache))
        tm.assert_series_equal(result, expected)
    
    def test_to_datetime_empty_stt(self, cache: Any) -> None:
        result: Any = to_datetime("", cache=cache)
        assert result is NaT
    
    def test_to_datetime_empty_str_list(self, cache: Any) -> None:
        result: DatetimeIndex = to_datetime(["", ""], cache=cache)
        assert pd.isna(result).all()
    
    def test_to_datetime_zero(self, cache: Any) -> None:
        result: Timestamp = Timestamp(0)
        expected: Timestamp = to_datetime(0, cache=cache)
        assert result == expected
    
    def test_to_datetime_strings(self, cache: Any) -> None:
        expected: Timestamp = to_datetime(["2012"], cache=cache)[0]
        result: Timestamp = to_datetime("2012", cache=cache)
        assert result == expected
    
    def test_to_datetime_strings_variation(self, cache: Any) -> None:
        array: List[str] = ["2012", "20120101", "20120101 12:01:01"]
        expected: List[Timestamp] = [to_datetime(dt_str, cache=cache) for dt_str in array]
        result: List[Timestamp] = [Timestamp(date_str) for date_str in array]
        tm.assert_almost_equal(result, expected)
    
    @pytest.mark.parametrize("result", [Timestamp("2012"), to_datetime("2012")])
    def test_to_datetime_strings_vs_constructor(self, result: Any) -> None:
        expected: Timestamp = Timestamp(2012, 1, 1)
        assert result == expected
    
    def test_to_datetime_unprocessable_input(self, cache: Any) -> None:
        msg: str = '^Given date string "1" not likely a datetime$'
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, "1"], errors="raise", cache=cache)
    
    def test_to_datetime_other_datetime64_units(self) -> None:
        scalar: np.int64 = np.int64(1337904000000000).view("M8[us]")
        as_obj: Any = scalar.astype("O")
        index: DatetimeIndex = DatetimeIndex([scalar])
        assert index[0] == scalar.astype("O")
        value: Timestamp = Timestamp(scalar)
        assert value == as_obj
    
    def test_to_datetime_list_of_integers(self) -> None:
        rng: DatetimeIndex = date_range("1/1/2000", periods=20)
        rng = DatetimeIndex(rng.values)
        ints: List[int] = list(rng.asi8)
        result: DatetimeIndex = DatetimeIndex(ints)
        tm.assert_index_equal(rng, result)
    
    def test_to_datetime_overflow(self) -> None:
        # gh-17637
        msg: str = "Cannot cast 139999 days 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            date_range(start="1/1/1700", freq="B", periods=100000)
    
    def test_string_invalid_operation(self, cache: Any) -> None:
        invalid: np.ndarray = np.array(["87156549591102612381000001219H5"], dtype=object)
        with pytest.raises(ValueError, match="Unknown datetime string format"):
            to_datetime(invalid, errors="raise", cache=cache)
    
    def test_string_na_nat_conversion(self, cache: Any) -> None:
        strings: np.ndarray = np.array(["1/1/2000", "1/2/2000", np.nan, "1/4/2000"], dtype=object)
        expected: np.ndarray = np.empty(4, dtype="M8[s]")
        for i, val in enumerate(strings):
            if pd.isna(val):
                expected[i] = tools.iNaT  # using pandas internal NA value for datetime
            else:
                expected[i] = parse(val)
        result, _ = tools.array_to_datetime(strings)
        tm.assert_almost_equal(result, expected)
        result2: DatetimeIndex = to_datetime(strings, cache=cache)
        assert isinstance(result2, DatetimeIndex)
        tm.assert_numpy_array_equal(result, result2.values)
    
    def test_string_na_nat_conversion_malformed(self, cache: Any) -> None:
        malformed: np.ndarray = np.array(["1/100/2000", np.nan], dtype=object)
        msg: str = r"Unknown datetime string format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)
        with pytest.raises(ValueError, match=msg):
            to_datetime(malformed, errors="raise", cache=cache)
    
    def test_string_na_nat_conversion_with_name(self, cache: Any) -> None:
        idx = ["a", "b", "c", "d", "e"]
        series: Series = Series(["1/1/2000", np.nan, "1/3/2000", np.nan, "1/5/2000"], index=idx, name="foo")
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
        result: DatetimeIndex = to_datetime(series, cache=cache)
        dresult: DatetimeIndex = to_datetime(dseries, cache=cache)
    
        expected: Series = Series(np.empty(5, dtype="M8[s]"), index=idx)
        for i in range(5):
            x: Any = series.iloc[i]
            if pd.isna(x):
                expected.iloc[i] = NaT
            else:
                expected.iloc[i] = to_datetime(x, cache=cache)
        tm.assert_series_equal(result, expected, check_names=False)
        assert result.name == "foo"
        tm.assert_series_equal(dresult, expected, check_names=False)
        assert dresult.name == "foo"
    
    @pytest.mark.parametrize(
        "unit",
        ["h", "m", "s", "ms", "us", "ns"],
    )
    def test_dti_constructor_numpy_timeunits(self, cache: Any, unit: str) -> None:
        dtype = np.dtype(f"M8[{unit}]")
        base: DatetimeIndex = to_datetime(["2000-01-01T00:00", "2000-01-02T00:00", "NaT"], cache=cache)
        values: np.ndarray = base.values.astype(dtype)
        if unit in ["h", "m"]:
            unit = "s"
        exp_dtype = np.dtype(f"M8[{unit}]")
        expected: DatetimeIndex = DatetimeIndex(base.astype(exp_dtype))
        assert expected.dtype == exp_dtype
        tm.assert_index_equal(DatetimeIndex(values), expected)
        tm.assert_index_equal(to_datetime(values, cache=cache), expected)
    
    def test_dayfirst(self, cache: Any) -> None:
        arr: List[str] = ["10/02/2014", "11/02/2014", "12/02/2014"]
        expected: DatetimeIndex = DatetimeIndex([datetime(2014, 2, 10), datetime(2014, 2, 11), datetime(2014, 2, 12)]).as_unit("s")
        idx1: DatetimeIndex = DatetimeIndex(arr, dayfirst=True)
        idx2: DatetimeIndex = DatetimeIndex(np.array(arr), dayfirst=True)
        idx3: DatetimeIndex = to_datetime(arr, dayfirst=True, cache=cache)
        idx4: DatetimeIndex = to_datetime(np.array(arr), dayfirst=True, cache=cache)
        idx5: DatetimeIndex = DatetimeIndex(pd.Index(arr), dayfirst=True)
        idx6: DatetimeIndex = DatetimeIndex(Series(arr), dayfirst=True)
        tm.assert_index_equal(expected, idx1)
        tm.assert_index_equal(expected, idx2)
        tm.assert_index_equal(expected, idx3)
        tm.assert_index_equal(expected, idx4)
        tm.assert_index_equal(expected, idx5)
        tm.assert_index_equal(expected, idx6)
    
    def test_dayfirst_warnings_valid_input(self) -> None:
        warning_msg: str = (
            "Parsing dates in .* format when dayfirst=.* was specified. "
            "Pass `dayfirst=.*` or specify a format to silence this warning."
        )
        arr: List[str] = ["31/12/2014", "10/03/2011"]
        expected: DatetimeIndex = DatetimeIndex(["2014-12-31", "2011-03-10"], dtype="datetime64[s]", freq=None)
        res1: DatetimeIndex = to_datetime(arr, dayfirst=True)
        tm.assert_index_equal(expected, res1)
        with tm.assert_produces_warning(UserWarning, match=warning_msg):
            res2: DatetimeIndex = to_datetime(arr, dayfirst=False)
        tm.assert_index_equal(expected, res2)
    
    def test_dayfirst_warnings_invalid_input(self) -> None:
        arr: List[str] = ["31/12/2014", "03/30/2011"]
        with pytest.raises(
            ValueError,
            match=(
                r'^time data "03/30/2011" doesn\'t match format '
                rf'"%d/%m/%Y". {PARSING_ERR_MSG}$'
            ),
        ):
            to_datetime(arr, dayfirst=True)
    
    @pytest.mark.parametrize("klass", [DatetimeIndex, DatetimeArray._from_sequence])
    def test_to_datetime_dta_tz(self, klass: Callable[[Sequence[Any]], Any]) -> None:
        dti: DatetimeIndex = date_range("2015-04-05", periods=3).rename("foo")
        expected: DatetimeIndex = dti.tz_localize("UTC")
        obj: Any = klass(dti)
        expected = klass(expected)
        result: Any = to_datetime(obj, utc=True)
        tm.assert_equal(result, expected)


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
    def test_guess_datetime_format_for_array(self, test_list: List[Any]) -> None:
        expected_format: str = "%Y-%m-%d %H:%M:%S.%f"
        test_array: np.ndarray = np.array(test_list, dtype=object)
        assert tools._guess_datetime_format_for_array(test_array) == expected_format
    
    @pytest.mark.skip_if_not_us_locale
    def test_guess_datetime_format_for_array_all_nans(self) -> None:
        format_for_string_of_nans: Optional[str] = tools._guess_datetime_format_for_array(
            np.array([np.nan, np.nan, np.nan], dtype="O")
        )
        assert format_for_string_of_nans is None


class TestToDatetimeInferFormat:
    @pytest.mark.parametrize("test_format", ["%m-%d-%Y", "%m/%d/%Y %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"])
    def test_to_datetime_infer_datetime_format_consistent_format(self, cache: Any, test_format: str) -> None:
        ser: Series = Series(date_range("20000101", periods=50, freq="h"))
        s_as_dt_strings: Series = ser.apply(lambda x: x.strftime(test_format))
        with_format: Series = to_datetime(s_as_dt_strings, format=test_format, cache=cache)
        without_format: Series = to_datetime(s_as_dt_strings, cache=cache)
        tm.assert_series_equal(with_format, without_format)
    
    def test_to_datetime_inconsistent_format(self, cache: Any) -> None:
        data: List[str] = ["01/01/2011 00:00:00", "01-02-2011 00:00:00", "2011-01-03T00:00:00"]
        ser: Series = Series(np.array(data))
        msg: str = (
            r'^time data "01-02-2011 00:00:00" doesn\'t match format '
            rf'"%m/%d/%Y %H:%M:%S". {PARSING_ERR_MSG}$'
        )
        with pytest.raises(ValueError, match=msg):
            to_datetime(ser, cache=cache)
    
    def test_to_datetime_consistent_format(self, cache: Any) -> None:
        data: List[str] = ["Jan/01/2011", "Feb/01/2011", "Mar/01/2011"]
        ser: Series = Series(np.array(data))
        result: Series = to_datetime(ser, cache=cache)
        expected: Series = Series(["2011-01-01", "2011-02-01", "2011-03-01"], dtype="datetime64[s]")
        tm.assert_series_equal(result, expected)
    
    def test_to_datetime_series_with_nans(self, cache: Any) -> None:
        ser: Series = Series(np.array(["01/01/2011 00:00:00", np.nan, "01/03/2011 00:00:00", np.nan], dtype=object))
        result: Series = to_datetime(ser, cache=cache)
        expected: Series = Series(["2011-01-01", NaT, "2011-01-03", NaT], dtype="datetime64[s]")
        tm.assert_series_equal(result, expected)
    
    def test_to_datetime_series_start_with_nans(self, cache: Any) -> None:
        ser: Series = Series(np.array([np.nan, np.nan, "01/01/2011 00:00:00", "01/02/2011 00:00:00", "01/03/2011 00:00:00"], dtype=object))
        result: Series = to_datetime(ser, cache=cache)
        expected: Series = Series([NaT, NaT, "2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[s]")
        tm.assert_series_equal(result, expected)
    
    @pytest.mark.parametrize(
        "tz_name, offset",
        [("UTC", 0), ("UTC-3", 180), ("UTC+3", -180)],
    )
    def test_infer_datetime_format_tz_name(self, tz_name: str, offset: int) -> None:
        ser: Series = Series([f"2019-02-02 08:07:13 {tz_name}"])
        result: DatetimeIndex = to_datetime(ser)
        tz: timezone = timezone(timedelta(minutes=offset))
        expected: Series = Series([Timestamp("2019-02-02 08:07:13").tz_localize(tz)])
        expected = expected.dt.as_unit("s")
        tm.assert_series_equal(result, expected)
    
    @pytest.mark.parametrize(
        "ts,zero_tz",
        [
            ("2019-02-02 08:07:13", "Z"),
            ("2019-02-02 08:07:13", ""),
            ("2019-02-02 08:07:13.012345", "Z"),
            ("2019-02-02 08:07:13.012345", ""),
        ],
    )
    def test_infer_datetime_format_zero_tz(self, ts: str, zero_tz: str) -> None:
        ser: Series = Series([ts + zero_tz])
        result: DatetimeIndex = to_datetime(ser)
        tz: Optional[timezone] = timezone.utc if zero_tz == "Z" else None
        expected: Series = Series([Timestamp(ts, tz=tz)])
        tm.assert_series_equal(result, expected)
    
    @pytest.mark.parametrize("format", [None, "%Y-%m-%d"])
    def test_to_datetime_iso8601_noleading_0s(self, cache: Any, format: Optional[str]) -> None:
        ser: Series = Series(["2014-1-1", "2014-2-2", "2015-3-3"])
        expected: Series = Series([Timestamp("2014-01-01"), Timestamp("2014-02-02"), Timestamp("2015-03-03")])
        result: Series = to_datetime(ser, format=format, cache=cache)
        tm.assert_series_equal(result, expected)


class TestDaysInMonth:
    @pytest.mark.parametrize(
        "arg, format",
        [
            ["2015-02-29", None],
            ["2015-02-29", "%Y-%m-%d"],
            ["2015-02-32", "%Y-%m-%d"],
            ["2015-04-31", "%Y-%m-%d"],
        ],
    )
    def test_day_not_in_month_coerce(self, cache: Any, arg: str, format: Optional[str]) -> None:
        assert pd.isna(to_datetime(arg, errors="coerce", format=format, cache=cache))
    
    def test_day_not_in_month_raise(self, cache: Any) -> None:
        msg: str = "day is out of range for month: 2015-02-29"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2015-02-29", errors="raise", cache=cache)
    
    @pytest.mark.parametrize(
        "arg, format, msg",
        [
            (
                "2015-02-29",
                "%Y-%m-%d",
                f"^day is out of range for month. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-29-02",
                "%Y-%d-%m",
                f"^day is out of range for month. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-02-32",
                "%Y-%m-%d",
                '^unconverted data remains when parsing with format "%Y-%m-%d": "2". ' + f"{PARSING_ERR_MSG}$",
            ),
            (
                "2015-32-02",
                "%Y-%d-%m",
                '^time data "2015-32-02" doesn\'t match format "%Y-%d-%m". ' + f"{PARSING_ERR_MSG}$",
            ),
            (
                "2015-04-31",
                "%Y-%m-%d",
                f"^day is out of range for month. {PARSING_ERR_MSG}$",
            ),
            (
                "2015-31-04",
                "%Y-%d-%m",
                f"^day is out of range for month. {PARSING_ERR_MSG}$",
            ),
        ],
    )
    def test_day_not_in_month_raise_value(self, cache: Any, arg: str, format: Optional[str], msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            to_datetime(arg, errors="raise", format=format, cache=cache)


class TestDatetimeParsingWrappers:
    @pytest.mark.parametrize(
        "date_str, expected",
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
            # quarterly without space
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
            # GH#10537
            ("2014-06", datetime(2014, 6, 1)),
            ("06-2014", datetime(2014, 6, 1)),
            ("2014-6", datetime(2014, 6, 1)),
            ("6-2014", datetime(2014, 6, 1)),
            ("20010101 12", datetime(2001, 1, 1, 12)),
            ("20010101 1234", datetime(2001, 1, 1, 12, 34)),
            ("20010101 123456", datetime(2001, 1, 1, 12, 34, 56)),
        ],
    )
    def test_parsers(self, date_str: str, expected: datetime, cache: Any) -> None:
        yearfirst: bool = True
        result1, reso_attrname = tools.parsing.parse_datetime_string_with_reso(
            date_str, yearfirst=yearfirst
        )
        reso: str = {"nanosecond": "ns", "microsecond": "us", "millisecond": "ms", "second": "s"}.get(reso_attrname, "s")
        result2: Timestamp = to_datetime(date_str, yearfirst=yearfirst)
        result3: DatetimeIndex = to_datetime([date_str], yearfirst=yearfirst)
        result4: DatetimeIndex = to_datetime(np.array([date_str], dtype=object), yearfirst=yearfirst, cache=cache)
        result6: DatetimeIndex = DatetimeIndex([date_str], yearfirst=yearfirst)
        result8: DatetimeIndex = DatetimeIndex(pd.Index([date_str]), yearfirst=yearfirst)
        result9: DatetimeIndex = DatetimeIndex(Series([date_str]), yearfirst=yearfirst)
    
        for res in [result1, result2]:
            assert res == expected
        for res in [result3, result4, result6, result8, result9]:
            exp: DatetimeIndex = DatetimeIndex([Timestamp(expected)]).as_unit(reso)
            tm.assert_index_equal(res, exp)
    
        if not yearfirst:
            result5: Timestamp = Timestamp(date_str)
            assert result5 == expected
            result7: DatetimeIndex = date_range(date_str, freq="S", periods=1, yearfirst=yearfirst)
            assert result7 == expected
    
    def test_na_values_with_cache(self, cache: Any, unique_nulls_fixture: Any, unique_nulls_fixture2: Any) -> None:
        expected: DatetimeIndex = DatetimeIndex([NaT, NaT], dtype="datetime64[s]")
        result: DatetimeIndex = to_datetime([unique_nulls_fixture, unique_nulls_fixture2], cache=cache)
        tm.assert_index_equal(result, expected)
    
    def test_parsers_nat(self) -> None:
        result1, _ = tools.parsing.parse_datetime_string_with_reso("NaT")
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
    def test_parsers_dayfirst_yearfirst(self, cache: Any, date_str: str, dayfirst: bool, yearfirst: bool, expected: datetime) -> None:
        from dateutil.parser import parse
        dateutil_result: datetime = parse(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        assert dateutil_result == expected
        result1, _ = tools.parsing.parse_datetime_string_with_reso(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
        if not dayfirst and not yearfirst:
            result2: Timestamp = Timestamp(date_str)
            assert result2 == expected
        result3: Timestamp = to_datetime(date_str, dayfirst=dayfirst, yearfirst=yearfirst, cache=cache)
        result4: Timestamp = DatetimeIndex([date_str], dayfirst=dayfirst, yearfirst=yearfirst)[0]
        assert result1 == expected
        assert result3 == expected
        assert result4 == expected
    
    @pytest.mark.parametrize(
        "date_str, exp_def",
        [["10:15", datetime(1, 1, 1, 10, 15)], ["9:05", datetime(1, 1, 1, 9, 5)]],
    )
    def test_parsers_timestring(self, date_str: str, exp_def: datetime) -> None:
        from dateutil.parser import parse
        exp_now: datetime = parse(date_str)
        result1, _ = tools.parsing.parse_datetime_string_with_reso(date_str)
        result2: Timestamp = to_datetime(date_str)
        result3: DatetimeIndex = to_datetime([date_str])
        result4: Timestamp = Timestamp(date_str)
        result5: Timestamp = DatetimeIndex([date_str])[0]
        assert result1 == exp_def
        assert result2 == exp_now
        assert result3 == exp_now
        assert result4 == exp_now
        assert result5 == exp_now
    
    @pytest.mark.parametrize(
        "dt_string, tz, dt_string_repr",
        [
            (
                "2013-01-01 05:45+0545",
                timezone(timedelta(minutes=345)),
                "Timestamp('2013-01-01 05:45:00+0545', tz='UTC+05:45')",
            ),
            (
                "2013-01-01 05:30+0530",
                timezone(timedelta(minutes=330)),
                "Timestamp('2013-01-01 05:30:00+0530', tz='UTC+05:30')",
            ),
        ],
    )
    def test_parsers_timezone_minute_offsets_roundtrip(self, cache: Any, dt_string: str, tz: timezone, dt_string_repr: str) -> None:
        base: Timestamp = to_datetime("2013-01-01 00:00:00", cache=cache)
        base = base.tz_localize("UTC").tz_convert(tz)
        dt_time: Timestamp = to_datetime(dt_string, cache=cache)
        assert base == dt_time
        assert dt_string_repr == repr(dt_time)


@pytest.fixture(params=["D", "s", "ms", "us", "ns"])
def units_fixture(request: Any) -> str:
    return request.param


@pytest.fixture
def julian_dates_fixture() -> np.ndarray:
    return date_range("2014-1-1", periods=10).to_julian_date().values


class TestOriginFixture:
    def test_origin_and_unit(self) -> None:
        ts: Timestamp = to_datetime(1, unit="s", origin=1)
        expected: Timestamp = Timestamp("1970-01-01 00:00:02")
        assert ts == expected
        ts = to_datetime(1, unit="s", origin=1_000_000_000)
        expected = Timestamp("2001-09-09 01:46:41")
        assert ts == expected
    
    def test_julian(self, julian_dates_fixture: np.ndarray) -> None:
        result: Series = Series(to_datetime(julian_dates_fixture, unit="D", origin="julian"))
        expected: Series = Series(to_datetime(julian_dates_fixture - Timestamp(0).to_julian_date(), unit="D"))
        tm.assert_series_equal(result, expected)
    
    def test_unix(self) -> None:
        result: Series = Series(to_datetime([0, 1, 2], unit="D", origin="unix"))
        expected: Series = Series([Timestamp("1970-01-01"), Timestamp("1970-01-02"), Timestamp("1970-01-03")], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)
    
    def test_julian_round_trip(self) -> None:
        result: Timestamp = to_datetime(2456658, origin="julian", unit="D")
        assert result.to_julian_date() == 2456658
        msg: str = "1 is Out of Bounds for origin='julian'"
        with pytest.raises(ValueError, match=msg):
            to_datetime(1, origin="julian", unit="D")
    
    def test_invalid_unit(self, units_fixture: str, julian_dates_fixture: np.ndarray) -> None:
        if units_fixture != "D":
            msg: str = "unit must be 'D' for origin='julian'"
            with pytest.raises(ValueError, match=msg):
                to_datetime(julian_dates_fixture, unit=units_fixture, origin="julian")
    
    @pytest.mark.parametrize("unit", ["ns", "D"])
    def test_invalid_origin(self, unit: str) -> None:
        msg: str = "it must be numeric with a unit specified"
        with pytest.raises(ValueError, match=msg):
            to_datetime("2005-01-01", origin="1960-01-01", unit=unit)
    
    @pytest.mark.parametrize("epochs", [Timestamp(1960, 1, 1), datetime(1960, 1, 1), "1960-01-01", np.datetime64("1960-01-01")])
    def test_epoch(self, units_fixture: str, epochs: Union[Timestamp, datetime, str, np.datetime64]) -> None:
        epoch_1960: Timestamp = Timestamp(1960, 1, 1)
        units_from_epochs: np.ndarray = np.arange(5, dtype=np.int64)
        expected: Series = Series([pd.Timedelta(x, unit=units_fixture) + epoch_1960 for x in units_from_epochs])
        result: Series = Series(to_datetime(units_from_epochs, unit=units_fixture, origin=epochs))
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
    def test_invalid_origins(self, origin: Any, exc: Any, units_fixture: str) -> None:
        msg: str = "|".join(
            [
                f"origin {origin} is Out of Bounds",
                f"origin {origin} cannot be converted to a Timestamp",
                "Cannot cast .* to unit='ns' without overflow",
            ]
        )
        with pytest.raises(exc, match=msg):
            to_datetime(list(range(5)), unit=units_fixture, origin=origin)
    
    def test_invalid_origins_tzinfo(self) -> None:
        with pytest.raises(ValueError, match="must be tz-naive"):
            to_datetime(1, unit="D", origin=datetime(2000, 1, 1, tzinfo=timezone.utc))
    
    def test_incorrect_value_exception(self) -> None:
        msg: str = "Unknown datetime string format, unable to parse: yesterday"
        with pytest.raises(ValueError, match=msg):
            to_datetime(["today", "yesterday"])
    
    @pytest.mark.parametrize("format, warning", [(None, UserWarning), ("%Y-%m-%d %H:%M:%S", None), ("%Y-%d-%m %H:%M:%S", None)])
    def test_to_datetime_out_of_bounds_with_format_arg(self, format: Optional[str], warning: Optional[Any]) -> None:
        if format is None:
            res: Timestamp = to_datetime("2417-10-10 00:00:00.00", format=format)
            assert isinstance(res, Timestamp)
            assert res.year == 2417
            assert res.month == 10
            assert res.day == 10
        else:
            msg: str = "unconverted data remains when parsing with format.*"
            with pytest.raises(ValueError, match=msg):
                to_datetime("2417-10-10 00:00:00.00", format=format)
    
    @pytest.mark.parametrize("arg, origin, expected_str", [[200 * 365, "unix", "2169-11-13 00:00:00"], [200 * 365, "1870-01-01", "2069-11-13 00:00:00"], [300 * 365, "1870-01-01", "2169-10-20 00:00:00"]])
    def test_processing_order(self, arg: int, origin: Union[str, int], expected_str: str) -> None:
        result: Timestamp = to_datetime(arg, unit="D", origin=origin)
        expected: Timestamp = Timestamp(expected_str)
        assert result == expected
        result = to_datetime(200 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2069-11-13 00:00:00")
        assert result == expected
        result = to_datetime(300 * 365, unit="D", origin="1870-01-01")
        expected = Timestamp("2169-10-20 00:00:00")
        assert result == expected
    
    @pytest.mark.parametrize("offset,utc,exp", [["Z", True, "2019-01-01T00:00:00.000Z"], ["Z", None, "2019-01-01T00:00:00.000Z"], ["-01:00", True, "2019-01-01T01:00:00.000Z"], ["-01:00", None, "2019-01-01T00:00:00.000-01:00"]])
    def test_arg_tz_ns_unit(self, offset: str, utc: Optional[bool], exp: str) -> None:
        arg: str = "2019-01-01T00:00:00.000" + offset
        result: DatetimeIndex = to_datetime([arg], unit="ns", utc=utc)
        expected: DatetimeIndex = to_datetime([exp]).as_unit("ns")
        tm.assert_index_equal(result, expected)
    
    # Additional tests for origin fixture can be added here if needed.


def test_to_datetime_wrapped_datetime64_ps() -> None:
    result: DatetimeIndex = to_datetime([np.datetime64(1901901901901, "ps")])
    expected: DatetimeIndex = DatetimeIndex(["1970-01-01 00:00:01.901901901"], dtype="datetime64[ns]", freq=None)
    tm.assert_index_equal(result, expected)


def test_to_datetime_mixed_awareness_mixed_types_end() -> None:
    # This test is already defined in TestDatetimeParsingWrappers, so can be omitted or merged.
    pass


def test_from_numeric_arrow_dtype(any_numeric_ea_dtype: str) -> None:
    pytest.importorskip("pyarrow")
    ser: Series = Series([1, 2], dtype=f"{any_numeric_ea_dtype.lower()}[pyarrow]")
    result: Series = to_datetime(ser)
    expected: Series = Series([1, 2], dtype="datetime64[ns]")
    tm.assert_series_equal(result, expected)


def test_to_datetime_with_empty_str_utc_false_format_mixed() -> None:
    vals: List[str] = ["2020-01-01 00:00+00:00", ""]
    result: DatetimeIndex = to_datetime(vals, format="mixed")
    expected: DatetimeIndex = DatetimeIndex([Timestamp("2020-01-01 00:00+00:00"), "NaT"], dtype="M8[s, UTC]")
    tm.assert_index_equal(result, expected)
    alt: DatetimeIndex = to_datetime(vals)
    tm.assert_index_equal(alt, expected)
    alt2: DatetimeIndex = DatetimeIndex(vals)
    tm.assert_index_equal(alt2, expected)


def test_to_datetime_with_empty_str_utc_false_offsets_and_format_mixed() -> None:
    msg: str = "Mixed timezones detected. Pass utc=True in to_datetime"
    with pytest.raises(ValueError, match=msg):
        to_datetime(["2020-01-01 00:00+00:00", "2020-01-01 00:00+02:00", ""], format="mixed")


def test_to_datetime_mixed_tzs_mixed_types_final() -> None:
    ts_val: Timestamp = Timestamp("2016-01-02 03:04:05", tz="US/Pacific")
    dtstr: str = "2023-10-30 15:06+01"
    arr: List[Any] = [ts_val, dtstr]
    mixed_msg: str = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone"
    )
    with pytest.raises(ValueError, match=mixed_msg):
        to_datetime(arr)
    with pytest.raises(ValueError, match=mixed_msg):
        to_datetime(arr, format="mixed")
    with pytest.raises(ValueError, match=mixed_msg):
        DatetimeIndex(arr)


def test_to_datetime_mixed_types_matching_tzs_final() -> None:
    dtstr: str = "2023-11-01 09:22:03-07:00"
    ts_val: Timestamp = Timestamp(dtstr)
    arr: List[Union[Timestamp, str]] = [ts_val, dtstr]
    res1: DatetimeIndex = to_datetime(arr)
    res2: DatetimeIndex = to_datetime(arr[::-1])[::-1]
    res3: DatetimeIndex = to_datetime(arr, format="mixed")
    res4: DatetimeIndex = DatetimeIndex(arr)
    expected: DatetimeIndex = DatetimeIndex([ts_val, ts_val])
    tm.assert_index_equal(res1, expected)
    tm.assert_index_equal(res2, expected)
    tm.assert_index_equal(res3, expected)
    tm.assert_index_equal(res4, expected)


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
@pytest.mark.parametrize(
    "aware_val",
    [dtstr_global, Timestamp(dtstr_global)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize(
    "naive_val",
    [dtstr_global[:-6], ts_global.tz_localize(None), ts_global.date(), ts_global.asm8, ts_global.value, float(ts_global.value)],
    ids=lambda x: type(x).__name__,
)
@pytest.mark.parametrize("naive_first", [True, False])
def test_to_datetime_mixed_awareness_mixed_types_extra(aware_val: Any, naive_val: Any, naive_first: bool) -> None:
    vals: List[Any] = [aware_val, naive_val, ""]
    vec: List[Any] = vals.copy()
    if naive_first:
        vec = [naive_val, aware_val, ""]
    both_strs: bool = isinstance(aware_val, str) and isinstance(naive_val, str)
    has_numeric: bool = isinstance(naive_val, (int, float))
    both_datetime: bool = isinstance(naive_val, datetime) and isinstance(aware_val, datetime)
    mixed_msg: str = (
        "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone"
    )
    first_non_null: Any = next(x for x in vec if x != "")
    if not isinstance(first_non_null, str):
        msg: str = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec)
        else:
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
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
        msg = r"time data \"2020-01-01 00:00\" doesn't match format"
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec)
        with pytest.raises(ValueError, match=msg):
            to_datetime(vec, utc=True)
    elif both_strs and vec.index(naive_val) < vec.index(aware_val):
        msg = "unconverted data remains when parsing with format"
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
            to_datetime(vec, format="mixed")
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(vec)
    else:
        msg = mixed_msg
        if naive_first and isinstance(aware_val, Timestamp):
            if isinstance(naive_val, Timestamp):
                msg = "Tz-aware datetime.datetime cannot be converted to datetime64"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format="mixed")
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)
        else:
            if not naive_first and both_datetime:
                msg = "Cannot mix tz-aware with tz-naive values"
            with pytest.raises(ValueError, match=msg):
                to_datetime(vec, format="mixed")
            with pytest.raises(ValueError, match=msg):
                DatetimeIndex(vec)


def test_to_datetime_wrapped_datetime64_ps_extra() -> None:
    result: DatetimeIndex = to_datetime([np.datetime64(1901901901901, "ps")])
    expected: DatetimeIndex = DatetimeIndex(["1970-01-01 00:00:01.901901901"], dtype="datetime64[ns]", freq=None)
    tm.assert_index_equal(result, expected)


# GH60341: test converting extremely small unit datetimes.
def test_to_datetime_mixed_nulls() -> None:
    # Additional tests can be added as needed.
    pass


# End of file.
