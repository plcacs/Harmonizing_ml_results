from datetime import datetime, timedelta
from typing import Union, Sequence, Any, Tuple
import numpy as np
import pytest
from pandas.compat import WASM
from pandas.errors import OutOfBoundsDatetime
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)


def assert_dtype(obj: Union[Series, Index, DataFrame], expected_dtype: str) -> None:
    """
    Helper to check the dtype for a Series, Index, or single-column DataFrame.
    """
    dtype = tm.get_dtype(obj)
    assert dtype == expected_dtype


def get_expected_name(box: type, names: Sequence[str]) -> str:
    if box is DataFrame:
        exname = names[0]
    elif box in [tm.to_array, pd.array]:
        exname = names[1]
    else:
        exname = names[2]
    return exname


class TestTimedelta64ArrayLikeComparisons:
    def test_compare_timedelta64_zerodim(self, box_with_array: type) -> None:
        box = box_with_array
        xbox = box_with_array if box_with_array not in [Index, pd.array] else np.ndarray
        tdi = timedelta_range("2h", periods=4)
        other = np.array(tdi.to_numpy()[0])
        tdi = tm.box_expected(tdi, box)
        res = tdi <= other
        expected = np.array([True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize(
        "td_scalar",
        [
            timedelta(days=1),
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            offsets.Hour(24),
        ],
    )
    def test_compare_timedeltalike_scalar(
        self, box_with_array: type, td_scalar: Union[timedelta, Timedelta, np.timedelta64, offsets.Hour]
    ) -> None:
        box = box_with_array
        xbox = box if box_with_array not in [Index, pd.array] else np.ndarray
        ser = Series([timedelta(days=1), timedelta(days=2)])
        ser = tm.box_expected(ser, box)
        actual = ser > td_scalar
        expected = Series([False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "invalid",
        [
            345600000000000,
            "a",
            Timestamp("2021-01-01"),
            Timestamp("2021-01-01").now("UTC"),
            Timestamp("2021-01-01").now().to_datetime64(),
            Timestamp("2021-01-01").now().to_pydatetime(),
            Timestamp("2021-01-01").date(),
            np.array(4),
        ],
    )
    def test_td64_comparisons_invalid(
        self, box_with_array: type, invalid: Any
    ) -> None:
        box = box_with_array
        rng = timedelta_range("1 days", periods=10)
        obj = tm.box_expected(rng, box)
        assert_invalid_comparison(obj, invalid, box)

    @pytest.mark.parametrize(
        "other",
        [
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.date_range("1970-01-01", periods=10, tz="UTC").array,
            np.array(pd.date_range("1970-01-01", periods=10)),
            list(pd.date_range("1970-01-01", periods=10)),
            pd.date_range("1970-01-01", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_td64arr_cmp_arraylike_invalid(
        self, other: Any, box_with_array: type
    ) -> None:
        rng = timedelta_range("1 days", periods=10)._data
        rng = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(rng, other, box_with_array)

    def test_td64arr_cmp_mixed_invalid(self) -> None:
        rng = timedelta_range("1 days", periods=5)._data
        other = np.array([0, 1, 2, rng[3], Timestamp("2021-01-01")])
        result = rng == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = rng != other
        tm.assert_numpy_array_equal(result, ~expected)
        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            rng < other
        with pytest.raises(TypeError, match=msg):
            rng > other
        with pytest.raises(TypeError, match=msg):
            rng <= other
        with pytest.raises(TypeError, match=msg):
            rng >= other


class TestTimedelta64ArrayComparisons:
    @pytest.mark.parametrize("dtype", [None, object])
    def test_comp_nat(self, dtype: Union[None, type]) -> None:
        left = TimedeltaIndex([Timedelta("1 days"), NaT, Timedelta("3 days")])
        right = TimedeltaIndex([NaT, NaT, Timedelta("3 days")])
        lhs, rhs = (left, right)
        if dtype is object:
            lhs, rhs = (left.astype(object), right.astype(object))
        result = rhs == lhs
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = rhs != lhs
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)
        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    @pytest.mark.parametrize(
        "idx2",
        [
            TimedeltaIndex(
                [
                    "2 day",
                    "2 day",
                    NaT,
                    NaT,
                    "1 day 00:00:02",
                    "5 days 00:00:03",
                ]
            ),
            np.array(
                [
                    np.timedelta64(2, "D"),
                    np.timedelta64(2, "D"),
                    np.timedelta64("nat"),
                    np.timedelta64("nat"),
                    np.timedelta64(1, "D") + np.timedelta64(2, "s"),
                    np.timedelta64(5, "D") + np.timedelta64(3, "s"),
                ]
            ),
        ],
    )
    def test_comparisons_nat(self, idx2: Any) -> None:
        idx1 = TimedeltaIndex(["1 day", NaT, "1 day 00:00:01", NaT, "1 day 00:00:01", "5 day 00:00:03"])
        result = idx1 < idx2
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = idx2 > idx1
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 <= idx2
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx2 >= idx1
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 == idx2
        expected = np.array([False, False, False, False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 != idx2
        expected = np.array([True, True, True, True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_comparisons_coverage(self) -> None:
        rng = timedelta_range("1 days", periods=10)
        result = rng < rng[3]
        expected = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)
        result = rng == list(rng)
        exp = rng == rng
        tm.assert_numpy_array_equal(result, exp)


class TestTimedelta64ArithmeticUnsorted:
    def test_ufunc_coercions(self) -> None:
        idx = TimedeltaIndex(["2h", "4h", "6h", "8h", "10h"], freq="2h", name="x")
        for result in [idx * 2, np.multiply(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["4h", "8h", "12h", "16h", "20h"], freq="4h", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "4h"
        for result in [idx / 2, np.divide(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["1h", "2h", "3h", "4h", "5h"], freq="h", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "h"
        for result in [-idx, np.negative(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["-2h", "-4h", "-6h", "-8h", "-10h"], freq="-2h", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "-2h"
        idx = TimedeltaIndex(["-2h", "-1h", "0h", "1h", "2h"], freq="h", name="x")
        for result in [abs(idx), np.absolute(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["2h", "1h", "0h", "1h", "2h"], freq=None, name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq is None

    def test_subtraction_ops(self) -> None:
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")
        msg = "cannot subtract a datelike from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        with pytest.raises(TypeError, match=msg):
            tdi - dti
        msg = "unsupported operand type\\(s\\) for -"
        with pytest.raises(TypeError, match=msg):
            td - dt
        msg = "(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            td - dti
        result = dt - dti
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"], name="bar")
        tm.assert_index_equal(result, expected)
        result = dti - dt
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"], name="bar")
        tm.assert_index_equal(result, expected)
        result = tdi - td
        expected = TimedeltaIndex(["0 days", NaT, "1 days"], name="foo")
        tm.assert_index_equal(result, expected)
        result = td - tdi
        expected = TimedeltaIndex(["0 days", NaT, "-1 days"], name="foo")
        tm.assert_index_equal(result, expected)
        result = dti - td
        expected = DatetimeIndex(["20121231", "20130101", "20130102"], dtype="M8[ns]", freq="D", name="bar")
        tm.assert_index_equal(result, expected)
        result = dt - tdi
        expected = DatetimeIndex(["20121231", NaT, "20121230"], dtype="M8[ns]", name="foo")
        tm.assert_index_equal(result, expected)

    def test_subtraction_ops_with_tz(self, box_with_array: type) -> None:
        dti = pd.date_range("20130101", periods=3)
        dti = tm.box_expected(dti, box_with_array)
        ts = Timestamp("20130101")
        dt = ts.to_pydatetime()
        dti_tz = pd.date_range("20130101", periods=3).tz_localize("US/Eastern")
        dti_tz = tm.box_expected(dti_tz, box_with_array)
        ts_tz = Timestamp("20130101").tz_localize("US/Eastern")
        ts_tz2 = Timestamp("20130101").tz_localize("CET")
        dt_tz = ts_tz.to_pydatetime()
        td = Timedelta("1 days")

        def _check(result: Any, expected: Timedelta) -> None:
            assert result == expected
            assert isinstance(result, Timedelta)

        result = ts - ts
        expected = Timedelta("0 days")
        _check(result, expected)
        result = dt_tz - ts_tz
        expected = Timedelta("0 days")
        _check(result, expected)
        result = ts_tz - dt_tz
        expected = Timedelta("0 days")
        _check(result, expected)
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects."
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt_tz - dt
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt - dt_tz
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects."
        with pytest.raises(TypeError, match=msg):
            ts - dt_tz
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - ts
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - dt
        msg = "Cannot subtract tz-naive and tz-aware"
        with pytest.raises(TypeError, match=msg):
            dti - ts_tz
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts
        result = dti_tz - dt_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = dt_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = dti_tz - ts_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = ts_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = td - td
        expected = Timedelta("0 days")
        _check(result, expected)
        result = dti_tz - td
        expected = DatetimeIndex(["20121231", "20130101", "20130102"], tz="US/Eastern").as_unit("ns")
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dti_tdi_numeric_ops(self) -> None:
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        result = tdi - tdi
        expected = TimedeltaIndex(["0 days", NaT, "0 days"], name="foo")
        tm.assert_index_equal(result, expected)
        result = tdi + tdi
        expected = TimedeltaIndex(["2 days", NaT, "4 days"], name="foo")
        tm.assert_index_equal(result, expected)
        result = dti - tdi
        expected = DatetimeIndex(["20121231", NaT, "20130101"], dtype="M8[ns]")
        tm.assert_index_equal(result, expected)


    def test_addition_ops(self) -> None:
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")
        result = tdi + dt
        expected = DatetimeIndex(["20130102", NaT, "20130103"], dtype="M8[ns]", name="foo")
        tm.assert_index_equal(result, expected)
        result = dt + tdi
        expected = DatetimeIndex(["20130102", NaT, "20130103"], dtype="M8[ns]", name="foo")
        tm.assert_index_equal(result, expected)
        result = td + tdi
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)
        result = tdi + td
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            tdi + dti[0:1]
        with pytest.raises(ValueError, match=msg):
            tdi[0:1] + dti
        msg = "Addition/subtraction of integers and integer-arrays"
        with pytest.raises(TypeError, match=msg):
            tdi + Index([1, 2, 3], dtype=np.int64)
        result = tdi + dti
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[ns]")
        tm.assert_index_equal(result, expected)
        result = dti + tdi
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[ns]")
        tm.assert_index_equal(result, expected)
        result = dt + td
        expected = Timestamp("20130102")
        assert result == expected
        result = td + dt
        expected = Timestamp("20130102")
        assert result == expected
        @pytest.mark.parametrize("freq", ["D", "B"])
        def test_timedelta(self, freq: str) -> None:
            index = pd.date_range("1/1/2000", periods=50, freq=freq)
            shifted = index + timedelta(1)
            back = shifted + timedelta(-1)
            back = back._with_freq("infer")
            tm.assert_index_equal(index, back)
            if freq == "D":
                expected = pd.tseries.offsets.Day(1)
                assert index.freq == expected
                assert shifted.freq == expected
                assert back.freq == expected
            else:
                assert index.freq == pd.tseries.offsets.BusinessDay(1)
                assert shifted.freq is None
                assert back.freq == pd.tseries.offsets.BusinessDay(1)
            result = index - timedelta(1)
            expected = index + timedelta(-1)
            tm.assert_index_equal(result, expected)

    def test_timedelta_tick_arithmetic(self) -> None:
        rng = pd.date_range("2013", "2014")
        s = Series(rng)
        result1 = rng - offsets.Hour(1)
        result2 = DatetimeIndex(s - np.timedelta64(100000000))
        result3 = rng - np.timedelta64(100000000)
        result4 = DatetimeIndex(s - offsets.Hour(1))
        assert result1.freq == rng.freq
        tm.assert_index_equal(result1._with_freq(None), result4)
        assert result3.freq == rng.freq
        tm.assert_index_equal(result2, result3._with_freq(None))

    def test_tda_add_sub_index(self, box_with_array: type) -> None:
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"])
        tda = tdi.array
        dti = pd.date_range("1999-12-31", periods=3, freq="D")
        result = tda + dti
        expected = tdi + dti
        tm.assert_index_equal(result, expected)
        result = tda + tdi
        expected = tdi + tdi
        tm.assert_index_equal(result, expected)
        result = tda - tdi
        expected = tdi - tdi
        tm.assert_index_equal(result, expected)

    def test_tda_add_dt64_object_array(
        self,
        performance_warning: Any,
        box_with_array: type,
        tz_naive_fixture: Union[None, str],
    ) -> None:
        box = box_with_array
        dti = pd.date_range("2016-01-01", periods=3, tz=tz_naive_fixture)
        dti = dti._with_freq(None)
        tdi = dti - dti
        obj = tm.box_expected(tdi, box)
        other = tm.box_expected(dti, box)
        with tm.assert_produces_warning(performance_warning):
            result = obj + other.astype(object)
        tm.assert_equal(result, other.astype(object))

    def test_tdi_iadd_timedeltalike(
        self, two_hours: Timedelta, box_with_array: type
    ) -> None:
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        orig_rng = rng
        rng += two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            tm.assert_equal(orig_rng, expected)

    def test_tdi_isub_timedeltalike(
        self, two_hours: Timedelta, box_with_array: type
    ) -> None:
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        orig_rng = rng
        rng -= two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            tm.assert_equal(orig_rng, expected)

    def test_tdi_ops_attributes(self) -> None:
        rng = timedelta_range("2 days", periods=5, freq="2D", name="x")
        result = rng + 1 * rng.freq
        exp = timedelta_range("4 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"
        result = rng - 2 * rng.freq
        exp = timedelta_range("-2 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"
        result = rng * 2
        exp = timedelta_range("4 days", periods=5, freq="4D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "4D"
        result = rng / 2
        exp = timedelta_range("1 days", periods=5, freq="D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "D"
        result = -rng
        exp = timedelta_range("-2 days", periods=5, freq="-2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "-2D"
        rng = timedelta_range("-2 days", periods=5, freq="D", name="x")
        result = abs(rng)
        exp = TimedeltaIndex(["2 days", "1 days", "0 days", "1 days", "2 days"], name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq is None


class TestAddSubNaTMasking:
    @pytest.mark.parametrize("str_ts", ["1950-01-01", "1980-01-01"])
    def test_tdarr_add_timestamp_nat_masking(
        self, box_with_array: type, str_ts: str
    ) -> None:
        tdinat = pd.to_timedelta(["24658 days 11:15:00", "NaT"])
        tdobj = tm.box_expected(tdinat, box_with_array)
        ts = Timestamp(str_ts)
        ts_variants: Tuple[
            Timestamp, datetime, np.datetime64, np.datetime64
        ] = (
            ts,
            ts.to_pydatetime(),
            ts.to_datetime64().astype("datetime64[ns]"),
            ts.to_datetime64().astype("datetime64[D]"),
        )
        for variant in ts_variants:
            res = tdobj + variant
            if box_with_array is DataFrame:
                assert res.iloc[1, 1] is NaT
            else:
                assert res[1] is NaT

    def test_tdi_add_overflow(self) -> None:
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            pd.to_timedelta(106580, "D") + Timestamp("2000")
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            Timestamp("2000") + pd.to_timedelta(106580, "D")
        _NaT = NaT._value + 1
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([106580], "D") + Timestamp("2000")
        with pytest.raises(OverflowError, match=msg):
            Timestamp("2000") + pd.to_timedelta([106580], "D")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta(["5 days", _NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT, "5 days", "1 hours"]) - pd.to_timedelta(
                ["7 seconds", _NaT, "4 hours"]
            )
        exp = TimedeltaIndex([NaT])
        result = pd.to_timedelta([NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)
        exp = TimedeltaIndex(["4 days", NaT])
        result = pd.to_timedelta(["5 days", NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)
        exp = TimedeltaIndex([NaT, NaT, "5 hours"])
        result = pd.to_timedelta([NaT, "5 days", "1 hours"]) + pd.to_timedelta(
            ["7 seconds", NaT, "4 hours"]
        )
        tm.assert_index_equal(result, exp)


class TestTimedeltaArraylikeAddSubOps:
    def test_sub_nat_retain_unit(self) -> None:
        ser = pd.to_timedelta(Series(["00:00:01"])).astype("m8[s]")
        result = ser - NaT
        expected = Series([NaT], dtype="m8[s]")
        tm.assert_series_equal(result, expected)

    def test_timedelta_ops_with_missing_values(self) -> None:
        s1 = pd.to_timedelta(Series(["00:00:01"]))
        s2 = pd.to_timedelta(Series(["00:00:02"]))
        sn = pd.to_timedelta(Series([NaT], dtype="m8[ns]"))
        df1 = DataFrame(["00:00:01"]).apply(pd.to_timedelta)
        df2 = DataFrame(["00:00:02"]).apply(pd.to_timedelta)
        dfn = DataFrame([NaT._value]).apply(pd.to_timedelta)
        scalar1 = pd.to_timedelta("00:00:01")
        scalar2 = pd.to_timedelta("00:00:02")
        timedelta_NaT = pd.to_timedelta("NaT")
        actual = scalar1 + scalar1
        assert actual == scalar2
        actual = scalar2 - scalar1
        assert actual == scalar1
        actual = s1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - s1
        tm.assert_series_equal(actual, s1)
        actual = s1 + scalar1
        tm.assert_series_equal(actual, s2)
        actual = scalar1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - scalar1
        tm.assert_series_equal(actual, s1)
        actual = -scalar1 + s2
        tm.assert_series_equal(actual, s1)
        actual = s1 + timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        actual = s1 - timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = -timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            s1 + np.nan
        with pytest.raises(TypeError, match=msg):
            np.nan + s1
        with pytest.raises(TypeError, match=msg):
            s1 - np.nan
        with pytest.raises(TypeError, match=msg):
            -np.nan + s1
        actual = s1 + NaT
        tm.assert_series_equal(actual, sn)
        actual = s2 - NaT
        tm.assert_series_equal(actual, sn)
        actual = s1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = s2 - df1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + s1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - s1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - df1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + scalar1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - scalar1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        msg = "cannot subtract a datelike from|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            df1 + np.nan
        with pytest.raises(TypeError, match=msg):
            df1 - np.nan
        actual = df1 + NaT
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - NaT
        tm.assert_frame_equal(actual, dfn)

    def test_operators_timedelta64(self) -> None:
        v1 = pd.date_range("2012-1-1", periods=3, freq="D")
        v2 = pd.date_range("2012-1-2", periods=3, freq="D")
        rs = Series(v2) - Series(v1)
        xp = Series(1000000000.0 * 3600 * 24, rs.index).astype("int64").astype("timedelta64[ns]")
        tm.assert_series_equal(rs, xp)
        assert rs.dtype == "timedelta64[ns]"
        df = DataFrame({"A": v1})
        td = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == "timedelta64[ns]"
        result = df["A"] - df["A"].shift()
        assert result.dtype == "timedelta64[ns]"
        result = df["A"] + td
        assert result.dtype == "M8[ns]"
        maxa = df["A"].max()
        assert isinstance(maxa, Timestamp)
        resultb = df["A"] - df["A"].max()
        assert resultb.dtype == "timedelta64[ns]"
        result = resultb + df["A"]
        values = [Timestamp("20111230"), Timestamp("20120101"), Timestamp("20120103")]
        expected = Series(values, dtype="M8[ns]", name="A")
        tm.assert_series_equal(result, expected)
        result = df["A"] - datetime(2001, 1, 1)
        expected = Series([timedelta(days=4017 + i) for i in range(3)], name="A")
        tm.assert_series_equal(result, expected)
        assert result.dtype == "m8[ns]"
        d = datetime(2001, 1, 1, 3, 4)
        resulta = df["A"] - d
        assert resulta.dtype == "m8[ns]"
        resultb = resulta + d
        tm.assert_series_equal(resultb, df["A"])
        td = timedelta(days=1)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(resultb, df["A"])
        assert resultb.dtype == "M8[ns]"
        td = timedelta(minutes=5, seconds=3)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(resultb, df["A"])
        assert resultb.dtype == "M8[ns]"
        value = rs[2] + np.timedelta64(timedelta(minutes=5, seconds=1))
        rs[2] += np.timedelta64(timedelta(minutes=5, seconds=1))
        assert rs[2] == value

    def test_timedelta64_ops_nat(self) -> None:
        timedelta_series = Series([NaT, Timedelta("1s")])
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype="timedelta64[ns]")
        single_nat_dtype_timedelta = Series([NaT], dtype="timedelta64[ns]")
        tm.assert_series_equal(timedelta_series - NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(-NaT + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series - single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(-single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta * 1.0, nat_series_dtype_timedelta)
        tm.assert_series_equal(1.0 * nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series * 1, timedelta_series)
        tm.assert_series_equal(1 * timedelta_series, timedelta_series)
        tm.assert_series_equal(timedelta_series * 1.5, Series([NaT, Timedelta("1.5s")]))
        tm.assert_series_equal(1.5 * timedelta_series, Series([NaT, Timedelta("1.5s")]))
        tm.assert_series_equal(timedelta_series * np.nan, nat_series_dtype_timedelta)
        tm.assert_series_equal(np.nan * timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series / 2, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / 2.0, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / np.nan, nat_series_dtype_timedelta)

    @pytest.mark.parametrize("cls", [Timestamp, datetime, np.datetime64])
    def test_td64arr_add_sub_datetimelike_scalar(
        self,
        cls: type,
        box_with_array: type,
        tz_naive_fixture: Union[None, str],
    ) -> None:
        tz = tz_naive_fixture
        dt_scalar = Timestamp("2012-01-01", tz=tz)
        if cls is datetime:
            ts = dt_scalar.to_pydatetime()
        elif cls is np.datetime64:
            if tz_naive_fixture is not None:
                pytest.skip(f"{cls} doesn support {tz_naive_fixture}")
            ts = dt_scalar.to_datetime64()
        else:
            ts = dt_scalar
        tdi = timedelta_range("1 day", periods=3)
        expected = pd.date_range("2012-01-02", periods=3, tz=tz)
        tdarr = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(ts + tdarr, expected)
        tm.assert_equal(tdarr + ts, expected)
        expected2 = pd.date_range("2011-12-31", periods=3, freq="-1D", tz=tz)
        expected2 = tm.box_expected(expected2, box_with_array)
        tm.assert_equal(ts - tdarr, expected2)
        tm.assert_equal(ts + -tdarr, expected2)
        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            tdarr - ts

    def test_td64arr_add_datetime64_nat(self, box_with_array: type) -> None:
        other = np.datetime64("NaT")
        tdi = timedelta_range("1 day", periods=3)
        expected = DatetimeIndex(["NaT", "NaT", "NaT"], dtype="M8[ns]")
        tdser = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(tdser + other, expected)
        tm.assert_equal(other + tdser, expected)

    def test_td64arr_sub_dt64_array(self, box_with_array: type) -> None:
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) - tdi
        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        msg = "cannot subtract a datelike from"
        with pytest.raises(TypeError, match=msg):
            tdi - dtarr
        result = dtarr - tdi
        tm.assert_equal(result, expected)

    def test_td64arr_add_dt64_array(self, box_with_array: type) -> None:
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) + tdi
        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = tdi + dtarr
        tm.assert_equal(result, expected)
        result = dtarr + tdi
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("tdi_freq", [None, "h"])
    def test_td64arr_sub_periodlike(
        self,
        box_with_array: type,
        box_with_array2: type,
        tdi_freq: Union[None, str],
        pi_freq: str,
    ) -> None:
        tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
        dti = Timestamp("2018-03-07 17:16:40") + tdi
        pi = dti.to_period(pi_freq)
        per = pi[0]
        tdi = tm.box_expected(tdi, box_with_array)
        pi = tm.box_expected(pi, box_with_array2)
        msg = "cannot subtract|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            tdi - pi
        with pytest.raises(TypeError, match=msg):
            tdi - per

    @pytest.mark.parametrize("other", ["a", 1, 1.5, np.array(2)])
    def test_td64arr_addsub_numeric_scalar_invalid(
        self, box_with_array: type, other: Any
    ) -> None:
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)
        assert_invalid_addsub_type(tdarr, other)

    @pytest.mark.parametrize(
        "vec",
        [
            np.array([1, 2, 3]),
            Index([1, 2, 3]),
            Series([1, 2, 3]),
            DataFrame([[1, 2, 3]]),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_addsub_numeric_arr_invalid(
        self,
        box_with_array: type,
        vec: Any,
        any_real_numpy_dtype: Any,
    ) -> None:
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)
        vector = vec.astype(any_real_numpy_dtype)
        assert_invalid_addsub_type(tdarr, vector)

    def test_td64arr_add_sub_int(
        self, box_with_array: type, one: int
    ) -> None:
        rng = timedelta_range("1 days 09:00:00", freq="h", periods=10)
        tdarr = tm.box_expected(rng, box_with_array)
        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, one, msg)
        with pytest.raises(TypeError, match=msg):
            tdarr += one
        with pytest.raises(TypeError, match=msg):
            tdarr -= one

    def test_td64arr_add_sub_integer_array(self, box_with_array: type) -> None:
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        rng = timedelta_range("1 days 09:00:00", freq="h", periods=3)
        tdarr = tm.box_expected(rng, box)
        other = tm.box_expected([4, 3, 2], xbox)
        msg = "Addition/subtraction of integers and integer-arrays"
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_addsub_integer_array_no_freq(
        self, box_with_array: type
    ) -> None:
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        tdi = TimedeltaIndex(["1 Day", "NaT", "3 Hours"])
        tdarr = tm.box_expected(tdi, box)
        other = tm.box_expected([14, -1, 16], xbox)
        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_add_sub_td64_array(self, box_with_array: type) -> None:
        box = box_with_array
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = dti - dti.shift(1)
        tdarr = tdi.values
        expected = 2 * tdi
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)
        result = tdi + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + tdi
        tm.assert_equal(result, expected)
        expected_sub = 0 * tdi
        result = tdi - tdarr
        tm.assert_equal(result, expected_sub)
        result = tdarr - tdi
        tm.assert_equal(result, expected_sub)

    def test_td64arr_add_sub_tdi(
        self,
        box_with_array: type,
        names: Sequence[str],
    ) -> None:
        box = box_with_array
        exname = get_expected_name(box, names)
        tdi = TimedeltaIndex(["0 days", "1 day"], name=names[1])
        tdi = np.array(tdi) if box in [tm.to_array, pd.array] else tdi
        ser = Series([Timedelta(hours=3), Timedelta(hours=4)], name=names[0])
        expected = Series(
            [Timedelta(hours=3), Timedelta(days=1, hours=4)], name=exname
        )
        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)
        result = tdi + ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")
        result = ser + tdi
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")
        expected = Series(
            [Timedelta(hours=-3), Timedelta(days=1, hours=-4)], name=exname
        )
        expected = tm.box_expected(expected, box)
        result = tdi - ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")
        result = ser - tdi
        tm.assert_equal(result, -expected)
        assert_dtype(result, "timedelta64[ns]")

    @pytest.mark.parametrize("tdnat", [np.timedelta64("NaT"), NaT])
    def test_td64arr_add_sub_td64_nat(
        self,
        box_with_array: type,
        tdnat: Union[np.timedelta64, Timedelta],
    ) -> None:
        box = box_with_array
        tdi = TimedeltaIndex([NaT, Timedelta("1s")])
        expected = TimedeltaIndex(["NaT"] * 2)
        obj = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)
        result = obj + tdnat
        tm.assert_equal(result, expected)
        result = tdnat + obj
        tm.assert_equal(result, expected)
        result = obj - tdnat
        tm.assert_equal(result, expected)
        result = tdnat - obj
        tm.assert_equal(result, expected)

    def test_td64arr_add_timedeltalike(self, two_hours: Timedelta, box_with_array: type) -> None:
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)
        result = rng + two_hours
        tm.assert_equal(result, expected)
        result = two_hours + rng
        tm.assert_equal(result, expected)

    def test_td64arr_sub_timedeltalike(
        self, two_hours: Timedelta, box_with_array: type
    ) -> None:
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)
        result = rng - two_hours
        tm.assert_equal(result, expected)
        result = two_hours - rng
        tm.assert_equal(result, -expected)

    def test_td64arr_add_sub_offset_index(
        self,
        performance_warning: Any,
        names: Sequence[str],
        box_with_array: type,
    ) -> None:
        box = box_with_array
        exname = get_expected_name(box, names)
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        other = np.array(other) if box in [tm.to_array, pd.array] else other
        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))],
            freq="infer",
            name=exname,
        )
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))],
            freq="infer",
            name=exname,
        )
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object)
        expected_sub = tm.box_expected(expected_sub, box).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res = tdi + other
        tm.assert_equal(res, expected)
        with tm.assert_produces_warning(performance_warning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)
        with tm.assert_produces_warning(performance_warning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_add_sub_offset_array(
        self, performance_warning: Any, box_with_array: type
    ) -> None:
        box = box_with_array
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])
        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer"
        )
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer"
        )
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res = tdi + other
        tm.assert_equal(res, expected)
        with tm.assert_produces_warning(performance_warning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)
        expected_sub = tm.box_expected(expected_sub, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res_sub = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_with_offset_series(
        self,
        performance_warning: Any,
        names: Sequence[str],
        box_with_array: type,
    ) -> None:
        box = box_with_array
        box2 = Series if box in [Index, tm.to_array, pd.array] else box
        exname = get_expected_name(box, names)
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        expected_add = Series(
            [tdi[n] + other[n] for n in range(len(tdi))],
            name=exname,
            dtype=object,
        )
        obj = tm.box_expected(tdi, box)
        expected_add = tm.box_expected(expected_add, box2).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res = obj + other
        tm.assert_equal(res, expected_add)
        with tm.assert_produces_warning(performance_warning):
            res2 = other + obj
        tm.assert_equal(res2, expected_add)
        expected_sub = Series(
            [tdi[n] - other[n] for n in range(len(tdi))],
            name=exname,
            dtype=object,
        )
        expected_sub = tm.box_expected(expected_sub, box2).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res3 = obj - other
        tm.assert_equal(res3, expected_sub)

    @pytest.mark.parametrize("obox", [np.array, Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(
        self,
        performance_warning: Any,
        obox: type,
        box_with_array: type,
    ) -> None:
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        tdi = tm.box_expected(tdi, box_with_array)
        anchored = obox([offsets.MonthEnd(), offsets.Day(n=2)])
        msg = "has incorrect type|cannot add the type MonthEnd"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdi + anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                anchored + tdi
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdi - anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                anchored - tdi

    def test_td64arr_add_sub_object_array(
        self,
        performance_warning: Any,
        box_with_array: type,
    ) -> None:
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box
        tdi = timedelta_range("1 day", periods=3, freq="D")
        tdarr = tm.box_expected(tdi, box)
        other = np.array([Timedelta(days=1), offsets.Day(2), Timestamp("2000-01-04")])
        with tm.assert_produces_warning(performance_warning):
            result = tdarr + other
        expected = Index([Timedelta(days=2), Timedelta(days=4), Timestamp("2000-01-07")])
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)
        msg = "unsupported operand type|cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdarr - other
        with tm.assert_produces_warning(performance_warning):
            result = other - tdarr
        expected = Index([Timedelta(0), Timedelta(0), Timestamp("2000-01-01")])
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)


class TestTimedelta64ArrayLikeArithmetic:
    def test_td64arr_pow_invalid(
        self, scalar_td: Timedelta, box_with_array: type
    ) -> None:
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan
        td1 = tm.box_expected(td1, box_with_array)
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            scalar_td ** td1
        with pytest.raises(TypeError, match=pattern):
            td1 ** scalar_td


def test_add_timestamp_to_timedelta() -> None:
    timestamp = Timestamp("2021-01-01")
    result = timestamp + timedelta_range("0s", "1s", periods=31)
    expected = DatetimeIndex(
        [
            timestamp
            + (
                pd.to_timedelta("0.033333333s") * i
                + pd.to_timedelta("0.000000001s") * divmod(i, 3)[0]
            )
            for i in range(31)
        ]
    )
    tm.assert_index_equal(result, expected)
