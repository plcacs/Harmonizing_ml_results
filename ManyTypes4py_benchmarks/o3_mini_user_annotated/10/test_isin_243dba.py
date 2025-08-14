from typing import Any, List, Union, Sequence
import numpy as np
import pytest

import pandas as pd
from pandas import Series, date_range
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray


class TestSeriesIsIn:
    def test_isin(self) -> None:
        s: Series = Series(["A", "B", "C", "a", "B", "B", "A", "C"])

        result: Series = s.isin(["A", "C"])
        expected: Series = Series([True, False, True, False, False, False, True, True])
        tm.assert_series_equal(result, expected)

        # GH#16012
        # This specific issue has to have a series over 1e6 in len, but the
        # comparison array (in_list) must be large enough so that numpy doesn't
        # do a manual masking trick that will avoid this issue altogether
        s = Series(list("abcdefghijk" * 10**5))
        # If numpy doesn't do the manual comparison/mask, these
        # unorderable mixed types are what cause the exception in numpy
        in_list: List[Any] = [-1, "a", "b", "G", "Y", "Z", "E", "K", "E", "S", "I", "R", "R"] * 6

        assert s.isin(in_list).sum() == 200000

    def test_isin_with_string_scalar(self) -> None:
        # GH#4763
        s: Series = Series(["A", "B", "C", "a", "B", "B", "A", "C"])
        msg: str = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a `str`"
        )
        with pytest.raises(TypeError, match=msg):
            s.isin("a")  # type: ignore

        s = Series(["aaa", "b", "c"])
        with pytest.raises(TypeError, match=msg):
            s.isin("aaa")  # type: ignore

    def test_isin_datetimelike_mismatched_reso(self) -> None:
        expected: Series = Series([True, True, False, False, False])

        ser: Series = Series(date_range("jan-01-2013", "jan-05-2013"))

        # fails on dtype conversion in the first place
        day_values: np.ndarray = np.asarray(ser[0:2].values).astype("datetime64[D]")
        result: Series = ser.isin(day_values)
        tm.assert_series_equal(result, expected)

        dta: np.ndarray = ser[:2]._values.astype("M8[s]")
        result = ser.isin(dta)
        tm.assert_series_equal(result, expected)

    def test_isin_datetimelike_mismatched_reso_list(self) -> None:
        expected: Series = Series([True, True, False, False, False])

        ser: Series = Series(date_range("jan-01-2013", "jan-05-2013"))

        dta: np.ndarray = ser[:2]._values.astype("M8[s]")
        result: Series = ser.isin(list(dta))
        tm.assert_series_equal(result, expected)

    def test_isin_with_i8(self) -> None:
        # GH#5021

        expected: Series = Series([True, True, False, False, False])
        expected2: Series = Series([False, True, False, False, False])

        # datetime64[ns]
        s: Series = Series(date_range("jan-01-2013", "jan-05-2013"))

        result: Series = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

        result = s.isin(s[0:2].values)
        tm.assert_series_equal(result, expected)

        result = s.isin([s[1]])
        tm.assert_series_equal(result, expected2)

        result = s.isin([np.datetime64(s[1])])
        tm.assert_series_equal(result, expected2)

        result = s.isin(set(s[0:2]))
        tm.assert_series_equal(result, expected)

        # timedelta64[ns]
        s = Series(pd.to_timedelta(range(5), unit="D"))
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty: Union[List[Any], Series, np.ndarray]) -> None:
        # see GH#16991
        s: Series = Series(["a", "b"])
        expected: Series = Series([False, False])

        result: Series = s.isin(empty)
        tm.assert_series_equal(expected, result)

    def test_isin_read_only(self) -> None:
        # https://github.com/pandas-dev/pandas/issues/37174
        arr: np.ndarray = np.array([1, 2, 3])
        arr.setflags(write=False)
        s: Series = Series([1, 2, 3])
        result: Series = s.isin(arr)
        expected: Series = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [object, None])
    def test_isin_dt64_values_vs_ints(self, dtype: Any) -> None:
        # GH#36621 dont cast integers to datetimes for isin
        dti: pd.DatetimeIndex = date_range("2013-01-01", "2013-01-05")
        ser: Series = Series(dti)

        comps: np.ndarray = np.asarray([1356998400000000000], dtype=dtype)

        res: np.ndarray = dti.isin(comps)
        expected: np.ndarray = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(comps)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, comps)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_tzawareness_mismatch(self) -> None:
        dti: pd.DatetimeIndex = date_range("2013-01-01", "2013-01-05")
        ser: Series = Series(dti)

        other: pd.DatetimeIndex = dti.tz_localize("UTC")

        res: np.ndarray = dti.isin(other)
        expected: np.ndarray = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_period_freq_mismatch(self) -> None:
        dti: pd.DatetimeIndex = date_range("2013-01-01", "2013-01-05")
        pi = dti.to_period("M")
        ser: Series = Series(pi)

        # We construct another PeriodIndex with the same i8 values
        #  but different dtype
        dtype = dti.to_period("Y").dtype
        other = PeriodArray._simple_new(pi.asi8, dtype=dtype)

        res: np.ndarray = pi.isin(other)
        expected: np.ndarray = np.array([False] * len(pi), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)

    @pytest.mark.parametrize("values", [[-9.0, 0.0], [-9, 0]])
    def test_isin_float_in_int_series(self, values: List[Union[int, float]]) -> None:
        # GH#19356 GH#21804
        ser: Series = Series(values)
        result: Series = ser.isin([-9, -0.5])
        expected: Series = Series([True, False])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    @pytest.mark.parametrize(
        "data,values,expected",
        [
            ([0, 1, 0], [1], [False, True, False]),
            ([0, 1, 0], [1, pd.NA], [False, True, False]),
            ([0, pd.NA, 0], [1, 0], [True, False, True]),
            ([0, 1, pd.NA], [1, pd.NA], [False, True, True]),
            ([0, 1, pd.NA], [1, np.nan], [False, True, False]),
            ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False]),
        ],
    )
    def test_isin_masked_types(
        self,
        dtype: Union[str, None],
        data: List[Any],
        values: List[Any],
        expected: List[Any],
    ) -> None:
        # GH#42405
        ser: Series = Series(data, dtype=dtype)

        result: Series = ser.isin(values)
        expected_series: Series = Series(expected, dtype="boolean")

        tm.assert_series_equal(result, expected_series)


def test_isin_large_series_mixed_dtypes_and_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    # https://github.com/pandas-dev/pandas/issues/37094
    # combination of object dtype for the values
    # and > _MINIMUM_COMP_ARR_LEN elements
    min_isin_comp: int = 5
    ser: Series = Series([1, 2, np.nan] * min_isin_comp)
    with monkeypatch.context() as m:
        m.setattr(algorithms, "_MINIMUM_COMP_ARR_LEN", min_isin_comp)
        result: Series = ser.isin({"foo", "bar"})
    expected: Series = Series([False] * 3 * min_isin_comp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype, data, values, expected",
    [
        ("boolean", [pd.NA, False, True], [False, pd.NA], [True, True, False]),
        ("Int64", [pd.NA, 2, 1], [1, pd.NA], [True, False, True]),
        ("boolean", [pd.NA, False, True], [pd.NA, True, "a", 20], [True, False, True]),
        ("boolean", [pd.NA, False, True], [], [False, False, False]),
        ("Float64", [20.0, 30.0, pd.NA], [pd.NA], [False, False, True]),
    ],
)
def test_isin_large_series_and_pdNA(
    dtype: str, data: List[Any], values: List[Any], expected: List[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # https://github.com/pandas-dev/pandas/issues/60678
    # combination of  large series (> _MINIMUM_COMP_ARR_LEN elements) and
    # values contains pdNA
    min_isin_comp: int = 2
    ser: Series = Series(data, dtype=dtype)
    expected_series: Series = Series(expected, dtype="boolean")

    with monkeypatch.context() as m:
        m.setattr(algorithms, "_MINIMUM_COMP_ARR_LEN", min_isin_comp)
        result: Series = ser.isin(values)
    tm.assert_series_equal(result, expected_series)


def test_isin_complex_numbers() -> None:
    # GH 17927
    array: List[complex] = [0, 1j, 1j, 1, 1 + 1j, 1 + 2j, 1 + 1j]
    result: Series = Series(array).isin([1j, 1 + 1j, 1 + 2j])
    expected: Series = Series([False, True, True, False, True, True, True], dtype=bool)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data,is_in",
    [([1, [2]], [1]), (["simple str", [{"values": 3}]], ["simple str"])],
)
def test_isin_filtering_with_mixed_object_types(
    data: List[Any], is_in: List[Any]
) -> None:
    # GH 20883
    ser: Series = Series(data)
    result: Series = ser.isin(is_in)
    expected: Series = Series([True, False])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize("isin", [[1, 2], [1.0, 2.0]])
def test_isin_filtering_on_iterable(
    data: List[Union[int, float]], isin: List[Union[int, float]]
) -> None:
    # GH 50234
    ser: Series = Series(data)
    result: Series = ser.isin(i for i in isin)
    expected_result: Series = Series([True, True, False])
    tm.assert_series_equal(result, expected_result)