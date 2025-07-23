from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pytest
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    cut,
    date_range,
    interval_range,
    isna,
    qcut,
    timedelta_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
from pandas.core.arrays import DatetimeArray, IntegerArray
from pandas.core.dtypes.common import is_datetime64_dtype


def test_simple() -> None:
    data = np.ones(5, dtype="int64")
    result = cut(data, 4, labels=False)
    expected = np.array([1, 1, 1, 1, 1])
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("func", [list, np.array])
def test_bins(func: Callable) -> None:
    data = func([0.2, 1.4, 2.5, 6.2, 9.7, 2.1])
    result, bins = cut(data, 3, retbins=True)
    intervals = IntervalIndex.from_breaks(bins.round(3))
    intervals = intervals.take([0, 0, 0, 1, 2, 0])
    expected = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.1905, 3.36666667, 6.53333333, 9.7]))


def test_right() -> None:
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    result, bins = cut(data, 4, right=True, retbins=True)
    intervals = IntervalIndex.from_breaks(bins.round(3))
    expected = Categorical(intervals, ordered=True)
    expected = expected.take([0, 0, 0, 2, 3, 0, 0])
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.1905, 2.575, 4.95, 7.325, 9.7]))


def test_no_right() -> None:
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    result, bins = cut(data, 4, right=False, retbins=True)
    intervals = IntervalIndex.from_breaks(bins.round(3), closed="left")
    intervals = intervals.take([0, 0, 0, 2, 3, 0, 1])
    expected = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.2, 2.575, 4.95, 7.325, 9.7095]))


def test_bins_from_interval_index() -> None:
    c = cut(range(5), 3)
    expected = c
    result = cut(range(5), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)
    expected = Categorical.from_codes(
        np.append(c.codes, -1), categories=c.categories, ordered=True
    )
    result = cut(range(6), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)


def test_bins_from_interval_index_doc_example() -> None:
    ages = np.array([10, 15, 13, 12, 23, 25, 28, 59, 60])
    c = cut(ages, bins=[0, 18, 35, 70])
    expected = IntervalIndex.from_tuples([(0, 18), (18, 35), (35, 70)])
    tm.assert_index_equal(c.categories, expected)
    result = cut([25, 20, 50], bins=c.categories)
    tm.assert_index_equal(result.categories, expected)
    tm.assert_numpy_array_equal(result.codes, np.array([1, 1, 2], dtype="int8"))


def test_bins_not_overlapping_from_interval_index() -> None:
    msg = "Overlapping IntervalIndex is not accepted"
    ii = IntervalIndex.from_tuples([(0, 10), (2, 12), (4, 14)])
    with pytest.raises(ValueError, match=msg):
        cut([5, 6], bins=ii)


def test_bins_not_monotonic() -> None:
    msg = "bins must increase monotonically"
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]
    with pytest.raises(ValueError, match=msg):
        cut(data, [0.1, 1.5, 1, 10])


@pytest.mark.parametrize(
    "x, bins, expected",
    [
        (
            date_range("2017-12-31", periods=3),
            [Timestamp.min, Timestamp("2018-01-01"), Timestamp.max],
            IntervalIndex.from_tuples(
                [
                    (Timestamp.min, Timestamp("2018-01-01")),
                    (Timestamp("2018-01-01"), Timestamp.max),
                ]
            ),
        ),
        (
            [-1, 0, 1],
            np.array(
                [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype="int64"
            ),
            IntervalIndex.from_tuples(
                [(np.iinfo(np.int64).min, 0), (0, np.iinfo(np.int64).max)]
            ),
        ),
        (
            [
                np.timedelta64(-1, "ns"),
                np.timedelta64(0, "ns"),
                np.timedelta64(1, "ns"),
            ],
            np.array(
                [
                    np.timedelta64(-np.iinfo(np.int64).max, "ns"),
                    np.timedelta64(0, "ns"),
                    np.timedelta64(np.iinfo(np.int64).max, "ns"),
                ]
            ),
            IntervalIndex.from_tuples(
                [
                    (np.timedelta64(-np.iinfo(np.int64).max, "ns"), np.timedelta64(0, "ns")),
                    (np.timedelta64(0, "ns"), np.timedelta64(np.iinfo(np.int64).max, "ns")),
                ]
            ),
        ),
    ],
)
def test_bins_monotonic_not_overflowing(
    x: Union[List[Timestamp], List[int], List[np.timedelta64]],
    bins: Union[List[Timestamp], np.ndarray],
    expected: IntervalIndex,
) -> None:
    result = cut(x, bins)
    tm.assert_index_equal(result.categories, expected)


def test_wrong_num_labels() -> None:
    msg = "Bin labels must be one fewer than the number of bin edges"
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]
    with pytest.raises(ValueError, match=msg):
        cut(data, [0, 1, 10], labels=["foo", "bar", "baz"])


@pytest.mark.parametrize(
    "x,bins,msg",
    [
        ([], 2, "Cannot cut empty array"),
        ([1, 2, 3], 0.5, "`bins` should be a positive integer"),
    ],
)
def test_cut_corner(x: List[float], bins: Union[int, float], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        cut(x, bins)


@pytest.mark.parametrize("arg", [2, np.eye(2), DataFrame(np.eye(2))])
@pytest.mark.parametrize("cut_func", [cut, qcut])
def test_cut_not_1d_arg(arg: Any, cut_func: Callable) -> None:
    msg = "Input array must be 1 dimensional"
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3, 4, np.inf],
        [-np.inf, 0, 1, 2, 3, 4],
        [-np.inf, 0, 1, 2, 3, 4, np.inf],
    ],
)
def test_int_bins_with_inf(data: List[Union[float, np.float64]]) -> None:
    msg = "cannot specify integer `bins` when input data contains infinity"
    with pytest.raises(ValueError, match=msg):
        cut(data, bins=3)


def test_cut_out_of_range_more() -> None:
    name = "x"
    ser = Series([0, -1, 0, 1, -3], name=name)
    ind = cut(ser, [0, 1], labels=False)
    exp = Series([np.nan, np.nan, np.nan, 0, np.nan], name=name)
    tm.assert_series_equal(ind, exp)


@pytest.mark.parametrize(
    "right,breaks,closed",
    [
        (True, [-0.001, 0.25, 0.5, 0.75, 1], "right"),
        (False, [0, 0.25, 0.5, 0.75, 1 + 0.001], "left"),
    ],
)
def test_labels(right: bool, breaks: List[float], closed: str) -> None:
    arr = np.tile(np.arange(0, 1.01, 0.1), 4)
    result, bins = cut(arr, 4, retbins=True, right=right)
    ex_levels = IntervalIndex.from_breaks(breaks, closed=closed)
    tm.assert_index_equal(result.categories, ex_levels)


def test_cut_pass_series_name_to_factor() -> None:
    name = "foo"
    ser = Series(np.random.default_rng(2).standard_normal(100), name=name)
    factor = cut(ser, 4)
    assert factor.name == name


def test_label_precision() -> None:
    arr = np.arange(0, 0.73, 0.01)
    result = cut(arr, 4, precision=2)
    ex_levels = IntervalIndex.from_breaks([-0.00072, 0.18, 0.36, 0.54, 0.72])
    tm.assert_index_equal(result.categories, ex_levels)


@pytest.mark.parametrize("labels", [None, False])
def test_na_handling(labels: Optional[bool]) -> None:
    arr = np.arange(0, 0.75, 0.01)
    arr[::3] = np.nan
    result = cut(arr, 4, labels=labels)
    result = np.asarray(result)
    expected = np.where(isna(arr), np.nan, result)
    tm.assert_almost_equal(result, expected)


def test_inf_handling() -> None:
    data = np.arange(6)
    data_ser = Series(data, dtype="int64")
    bins = [-np.inf, 2, 4, np.inf]
    result = cut(data, bins)
    result_ser = cut(data_ser, bins)
    ex_uniques = IntervalIndex.from_breaks(bins)
    tm.assert_index_equal(result.categories, ex_uniques)
    assert result[5] == Interval(4, np.inf)
    assert result[0] == Interval(-np.inf, 2)
    assert result_ser[5] == Interval(4, np.inf)
    assert result_ser[0] == Interval(-np.inf, 2)


def test_cut_out_of_bounds() -> None:
    arr = np.random.default_rng(2).standard_normal(100)
    result = cut(arr, [-1, 0, 1])
    mask = isna(result)
    ex_mask = (arr < -1) | (arr > 1)
    tm.assert_numpy_array_equal(mask, ex_mask)


@pytest.mark.parametrize(
    "get_labels,get_expected",
    [
        (
            lambda labels: labels,
            lambda labels: Categorical(
                ["Medium"] + 4 * ["Small"] + ["Medium", "Large"],
                categories=labels,
                ordered=True,
            ),
        ),
        (
            lambda labels: Categorical.from_codes([0, 1, 2], labels),
            lambda labels: Categorical.from_codes([1] + 4 * [0] + [1, 2], labels),
        ),
    ],
)
def test_cut_pass_labels(
    get_labels: Callable, get_expected: Callable
) -> None:
    bins = [0, 25, 50, 100]
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ["Small", "Medium", "Large"]
    result = cut(arr, bins, labels=get_labels(labels))
    tm.assert_categorical_equal(result, get_expected(labels))


def test_cut_pass_labels_compat() -> None:
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ["Good", "Medium", "Bad"]
    result = cut(arr, 3, labels=labels)
    exp = cut(arr, 3, labels=Categorical(labels, categories=labels, ordered=True))
    tm.assert_categorical_equal(result, exp)


@pytest.mark.parametrize("x", [np.arange(11.0), np.arange(11.0) / 10000000000.0])
def test_round_frac_just_works(x: np.ndarray) -> None:
    cut(x, 2)


@pytest.mark.parametrize(
    "val,precision,expected",
    [
        (-117.9998, 3, -118),
        (117.9998, 3, 118),
        (117.9998, 2, 118),
        (0.000123456, 2, 0.00012),
    ],
)
def test_round_frac(val: float, precision: int, expected: float) -> None:
    result = tmod._round_frac(val, precision=precision)
    assert result == expected


def test_cut_return_intervals() -> None:
    ser = Series([0, 1, 2, 3, 4, 5, 6, 7, 8])
    result = cut(ser, 3)
    exp_bins = np.linspace(0, 8, num=4).round(3)
    exp_bins[0] -= 0.008
    expected = Series(
        IntervalIndex.from_breaks(exp_bins, closed="right").take([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


def test_series_ret_bins() -> None:
    ser = Series(np.arange(4))
    result, bins = cut(ser, 2, retbins=True)
    expected = Series(
        IntervalIndex.from_breaks([-0.003, 1.5, 3], closed="right").repeat(2)
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"duplicates": "drop"}, None),
        ({}, "Bin edges must be unique"),
        ({"duplicates": "raise"}, "Bin edges must be unique"),
        ({"duplicates": "foo"}, "invalid value for 'duplicates' parameter"),
    ],
)
def test_cut_duplicates_bin(
    kwargs: Dict[str, str], msg: Optional[str]
) -> None:
    bins = [0, 2, 4, 6, 10, 10]
    values = Series(np.array([1, 3, 5, 7, 9]), index=["a", "b", "c", "d", "e"])
    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            cut(values, bins, **kwargs)
    else:
        result = cut(values, bins, **kwargs)
        expected = cut(values, pd.unique(np.asarray(bins)))
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("data", [9.0, -9.0, 0.0])
@pytest.mark.parametrize("length", [1, 2])
def test_single_bin(data: float, length: int) -> None:
    ser = Series([data] * length)
    result = cut(ser, 1, labels=False)
    expected = Series([0] * length, dtype=np.intp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "array_1_writeable,array_2_writeable", [(True, True), (True, False), (False, False)]
)
def test_cut_read_only(array_1_writeable: bool, array_2_writeable: bool) -> None:
    array_1 = np.arange(0, 100, 10)
    array_1.flags.writeable = array_1_writeable
    array_2 = np.arange(0, 100, 10)
    array_2.flags.writeable = array_2_writeable
    hundred_elements = np.arange(100)
    tm.assert_categorical_equal(
        cut(hundred_elements, array_1), cut(hundred_elements, array_2)
    )


@pytest.mark.parametri