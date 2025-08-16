from datetime import (
    datetime,
    time,
    timedelta,
    timezone,
)
from itertools import (
    product,
    starmap,
)
import operator

import numpy as np
import pytest

from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months

import pandas as pd
from pandas import (
    DateOffset,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
    assert_cannot_add,
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)

# ------------------------------------------------------------------
# Comparisons


class TestDatetime64ArrayLikeComparisons:
    # Comparison tests for datetime64 vectors fully parametrized over
    #  DataFrame/Series/DatetimeIndex/DatetimeArray.  Ideally all comparison
    #  tests will eventually end up here.

    def test_compare_zerodim(self, tz_naive_fixture: str, box_with_array: str) -> None:
        # Test comparison with zero-dimensional array is unboxed
        tz = tz_naive_fixture
        box = box_with_array
        dti = date_range("20130101", periods=3, tz=tz)

        other = np.array(dti.to_numpy()[0])

        dtarr = tm.box_expected(dti, box)
        xbox = get_upcast_box(dtarr, other, True)
        result = dtarr <= other
        expected = np.array([True, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            "foo",
            -1,
            99,
            4.0,
            object(),
            timedelta(days=2),
            # GH#19800, GH#19301 datetime.date comparison raises to
            #  match DatetimeIndex/Timestamp.  This also matches the behavior
            #  of stdlib datetime.datetime
            datetime(2001, 1, 1).date(),
            # GH#19301 None and NaN are *not* cast to NaT for comparisons
            None,
            np.nan,
        ],
    )
    def test_dt64arr_cmp_scalar_invalid(self, other: object, tz_naive_fixture: str, box_with_array: str) -> None:
        # GH#22074, GH#15966
        tz = tz_naive_fixture

        rng = date_range("1/1/2000", periods=10, tz=tz)
        dtarr = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(dtarr, other, box_with_array)

    @pytest.mark.parametrize(
        "other",
        [
            # GH#4968 invalid date/int comparisons
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.timedelta_range("1ns", periods=10).array,
            np.array(pd.timedelta_range("1ns", periods=10)),
            list(pd.timedelta_range("1ns", periods=10)),
            pd.timedelta_range("1 Day", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_dt64arr_cmp_arraylike_invalid(
        self, other: object, tz_naive_fixture: str, box_with_array: str
    ) -> None:
        tz = tz_naive_fixture

        dta = date_range("1970-01-01", freq="ns", periods=10, tz=tz)._data
        obj = tm.box_expected(dta, box_with_array)
        assert_invalid_comparison(obj, other, box_with_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture: str) -> None:
        tz = tz_naive_fixture

        dta = date_range("1970-01-01", freq="h", periods=5, tz=tz)._data

        other = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        result = dta == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dta != other
        tm.assert_numpy_array_equal(result, ~expected)

        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            dta < other
        with pytest.raises(TypeError, match=msg):
            dta > other
        with pytest.raises(TypeError, match=msg):
            dta <= other
        with pytest.raises(TypeError, match=msg):
            dta >= other

    def test_dt64arr_nat_comparison(self, tz_naive_fixture: str, box_with_array: str) -> None:
        # GH#22242, GH#22163 DataFrame considered NaT == ts incorrectly
        tz = tz_naive_fixture
        box = box_with_array

        ts = Timestamp("2021-01-01", tz=tz)
        ser = Series([ts, NaT])

        obj = tm.box_expected(ser, box)
        xbox = get_upcast_box(obj, ts, True)

        expected = Series([True, False], dtype=np.bool_)
        expected = tm.box_expected(expected, xbox)

        result = obj == ts
        tm.assert_equal(result, expected)


class TestDatetime64SeriesComparison:
    # TODO: moved from tests.series.test_operators; needs cleanup

    @pytest.mark.parametrize(
        "pair",
        [
            (
                [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
                [NaT, NaT, Timestamp("2011-01-03")],
            ),
            (
                [Timedelta("1 days"), NaT, Timedelta("3 days")],
                [NaT, NaT, Timedelta("3 days")],
            ),
            (
                [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
                [NaT, NaT, Period("2011-03", freq="M")],
            ),
        ],
    )
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("dtype", [None, object])
    @pytest.mark.parametrize(
        "op, expected",
        [
            (operator.eq, [False, False, True]),
            (operator.ne, [True, True, False]),
            (operator.lt, [False, False, False]),
            (operator.gt, [False, False, False]),
            (operator.ge, [False, False, True]),
            (operator.le, [False, False, True]),
        ],
    )
    def test_nat_comparisons(
        self,
        dtype: str,
        index_or_series: type,
        reverse: bool,
        pair: tuple[list, list],
        op: operator,
        expected: list[bool],
    ) -> None:
        box = index_or_series
        lhs, rhs = pair
        if reverse:
            # add lhs / rhs switched data
            lhs, rhs = rhs, lhs

        left = Series(lhs, dtype=dtype)
        right = box(rhs, dtype=dtype)

        result = op(left, right)

        tm.assert_series_equal(result, Series(expected))

    @pytest.mark.parametrize(
        "data",
        [
            [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
            [Timedelta("1 days"), NaT, Timedelta("3 days")],
            [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
        ],
    )
    @pytest.mark.parametrize("dtype", [None, object])
    def test_nat_comparisons_scalar(self, dtype: str, data: list, box_with_array: str) -> None:
        box = box_with_array

        left = Series(data, dtype=dtype)
        left = tm.box_expected(left, box)
        xbox = get_upcast_box(left, NaT, True)

        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")

        tm.assert_equal(left == NaT, expected)
        tm.assert_equal(NaT == left, expected)

        expected = [True, True, True]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        tm.assert_equal(left != NaT, expected)
        tm.assert_equal(NaT != left, expected)

        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        tm.assert_equal(left < NaT, expected)
        tm.assert_equal(NaT > left, expected)
        tm.assert_equal(left <= NaT, expected)
        tm.assert_equal(NaT >= left, expected)

        tm.assert_equal(left > NaT, expected)
        tm.assert_equal(NaT < left, expected)
        tm.assert_equal(left >= NaT, expected)
        tm.assert_equal(NaT <= left, expected)

    @pytest.mark.parametrize("val", [datetime(2000, 1, 4), datetime(2000, 1, 5)])
    def test_series_comparison_scalars(self, val: datetime) -> None:
        series = Series(date_range("1/1/2000", periods=10))

        result = series > val
        expected = Series([x > val for x in series])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "left,right", [("lt", "gt"), ("le", "ge"), ("eq", "eq"), ("ne", "ne")]
    )
    def test_timestamp_compare_series(self, left: str, right: str) -> None:
        # see gh-4982
        # Make sure we can compare Timestamps on the right AND left hand side.
        ser = Series(date_range("20010101", periods=10), name="dates")
        s_nat = ser.copy(deep=True)

        ser[0] = Timestamp("nat")
        ser[3] = Timestamp("nat")

        left_f = getattr(operator, left)
        right_f = getattr(operator, right)

        # No NaT
        expected = left_f(ser, Timestamp("20010109"))
        result = right_f(Timestamp("20010109"), ser)
        tm.assert_series_equal(result, expected)

        # NaT
        expected = left_f(ser, Timestamp("nat"))
        result = right_f(Timestamp("nat"), ser)
        tm.assert_series_equal(result, expected)

        # Compare to Timestamp with series containing NaT
        expected = left_f(s_nat, Timestamp("20010109"))
        result = right_f(Timestamp("20010109"), s_nat)
        tm.assert_series_equal(result, expected)

        # Compare to NaT with series containing NaT
        expected = left_f(s_nat, NaT)
        result = right_f(NaT, s_nat)
        tm.assert_series_equal(result, expected)

    def test_dt64arr_timestamp_equality(self, box_with_array: str) -> None:
        # GH#11034
        box = box_with_array

        ser = Series([Timestamp("2000-01-29 01:59:00"), Timestamp("2000-01-30"), NaT])
        ser = tm.box_expected(ser, box)
        xbox = get_upcast_box(ser, ser, True)

        result = ser != ser
        expected = tm.box_expected([False, False, True], xbox)
        tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser != ser[0]

        else:
            result = ser != ser[0]
            expected = tm.box_expected([False, True, True], xbox)
            tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser != ser[2]
        else:
            result = ser != ser[2]
            expected = tm.box_expected([True, True, True], xbox)
            tm.assert_equal(result, expected)

        result = ser == ser
        expected = tm.box_expected([True, True, False], xbox)
        tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser == ser[0]
        else:
            result = ser == ser[0]
            expected = tm.box_expected([True, False, False], xbox)
            tm.assert_equal(result, expected)

        if box is pd.DataFrame:
            # alignment for frame vs series comparisons deprecated
            #  in GH#46795 enforced 2.0
            with pytest.raises(ValueError, match="not aligned"):
                ser == ser[2]
        else:
            result = ser == ser[2]
            expected = tm.box_expected([False, False, False], xbox)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "datetimelike",
        [
            Timestamp("20130101"),
            datetime(2013, 1, 1),
            np.datetime64("2013-01-01T00:00", "ns"),
        ],
    )
    @pytest.mark.parametrize(
        "op,expected",
        [
            (operator.lt, [True, False, False, False]),
            (operator.le, [True, True, False, False]),
            (operator.eq, [False, True, False, False]),
            (operator.gt, [False, False, False, True]),
        ],
    )
    def test_dt64_compare_datetime_scalar(self, datetimelike: object, op: operator, expected: list[bool]) -> None:
        # GH#17965, test for ability to compare datetime64[ns] columns
        #  to datetimelike
        ser = Series(
            [
                Timestamp("20120101"),
                Timestamp("20130101"),
                np.nan,
                Timestamp("20130103"),
            ],
            name="A",
        )
        result = op(ser, datetimelike)
        expected = Series(expected, name="A")
        tm.assert_series_equal(result, expected)

    def test_ts_series_numpy_maximum(self) -> None:
        # GH#50864, test numpy.maximum does not fail
        # given a TimeStamp and Series(with dtype datetime64) comparison
        ts = Timestamp("2024-07-01")
        ts_series = Series(
            ["2024-06-01", "2024-07-01", "2024-08-01"],
            dtype="datetime64[us]",
        )

        expected = Series(
            ["2024-07-01", "2024-07-01", "2024-08-01"],
            dtype="datetime64[us]",
        )

        tm.assert_series_equal(expected, np.maximum(ts, ts_series))


class TestDatetimeIndexComparisons:
    # TODO: moved from tests.indexes.test_base; parametrize and de-duplicate
    def test_comparators(self, comparison_op: operator) -> None:
        index = date_range("2020-01-01", periods=10)
        element = index[len(index) // 2]
        element = Timestamp(element).to_datetime64()

        arr = np.array(index)
        arr_result = comparison_op(arr, element)
        index_result = comparison_op(index, element)

        assert isinstance(index_result, np.ndarray)
        tm.assert_numpy_array_equal(arr_result, index_result)

    @pytest.mark.parametrize(
        "other",
        [datetime(2016, 1, 1), Timestamp("2016-01-01"), np.datetime64("2016-01-01")],
    )
    def test_dti_cmp_datetimelike(self, other: object, tz_naive_fixture: str) -> None:
        tz = tz_naive_fixture
        dti = date_range("2016-01-01", periods=2, tz=tz)
        if tz is not None:
            if isinstance(other, np.datetime64):
                pytest.skip(f"{type(other).__name__} is not tz aware")
            other = localize_pydatetime(other, dti.tzinfo)

        result = dti == other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dti > other
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = dti >= other
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        result = dti < other
        expected = np.array([False