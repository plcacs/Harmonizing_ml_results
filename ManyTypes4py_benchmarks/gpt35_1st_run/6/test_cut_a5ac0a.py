from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, DataFrame, Series, cut
import pandas._testing as tm
from pandas.api.types import CategoricalDtype

def test_simple() -> None:
    data: np.ndarray = np.ones(5, dtype='int64')
    result = cut(data, 4, labels=False)
    expected = np.array([1, 1, 1, 1, 1])
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)

def test_bins(func: callable) -> None:
    data = func([0.2, 1.4, 2.5, 6.2, 9.7, 2.1])
    result, bins = cut(data, 3, retbins=True)
    intervals = pd.IntervalIndex.from_breaks(bins.round(3))
    intervals = intervals.take([0, 0, 0, 1, 2, 0])
    expected = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.1905, 3.36666667, 6.53333333, 9.7]))

def test_right() -> None:
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    result, bins = cut(data, 4, right=True, retbins=True)
    intervals = pd.IntervalIndex.from_breaks(bins.round(3))
    expected = Categorical(intervals, ordered=True)
    expected = expected.take([0, 0, 0, 2, 3, 0, 0])
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.1905, 2.575, 4.95, 7.325, 9.7]))

def test_no_right() -> None:
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    result, bins = cut(data, 4, right=False, retbins=True)
    intervals = pd.IntervalIndex.from_breaks(bins.round(3), closed='left')
    intervals = intervals.take([0, 0, 0, 2, 3, 0, 1])
    expected = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)
    tm.assert_almost_equal(bins, np.array([0.2, 2.575, 4.95, 7.325, 9.7095]))

def test_bins_from_interval_index() -> None:
    c = cut(range(5), 3)
    expected = c
    result = cut(range(5), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)
    expected = Categorical.from_codes(np.append(c.codes, -1), categories=c.categories, ordered=True)
    result = cut(range(6), bins=expected.categories)
    tm.assert_categorical_equal(result, expected)

def test_bins_from_interval_index_doc_example() -> None:
    ages = np.array([10, 15, 13, 12, 23, 25, 28, 59, 60])
    c = cut(ages, bins=[0, 18, 35, 70])
    expected = pd.IntervalIndex.from_tuples([(0, 18), (18, 35), (35, 70)])
    tm.assert_index_equal(c.categories, expected)
    result = cut([25, 20, 50], bins=c.categories)
    tm.assert_index_equal(result.categories, expected)
    tm.assert_numpy_array_equal(result.codes, np.array([1, 1, 2], dtype='int8'))

def test_bins_not_overlapping_from_interval_index() -> None:
    msg = 'Overlapping IntervalIndex is not accepted'
    ii = pd.IntervalIndex.from_tuples([(0, 10), (2, 12), (4, 14)])
    with pytest.raises(ValueError, match=msg):
        cut([5, 6], bins=ii)

def test_bins_not_monotonic() -> None:
    msg = 'bins must increase monotonically'
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]
    with pytest.raises(ValueError, match=msg):
        cut(data, [0.1, 1.5, 1, 10])

def test_bins_monotonic_not_overflowing(x, bins, expected) -> None:
    result = cut(x, bins)
    tm.assert_index_equal(result.categories, expected)

def test_wrong_num_labels() -> None:
    msg = 'Bin labels must be one fewer than the number of bin edges'
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]
    with pytest.raises(ValueError, match=msg):
        cut(data, [0, 1, 10], labels=['foo', 'bar', 'baz'])

def test_cut_corner(x, bins, msg) -> None:
    with pytest.raises(ValueError, match=msg):
        cut(x, bins)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_int_bins_with_inf(data) -> None:
    msg = 'cannot specify integer `bins` when input data contains infinity'
    with pytest.raises(ValueError, match=msg):
        cut(data, bins=3)

def test_cut_out_of_range_more() -> None:
    name = 'x'
    ser = Series([0, -1, 0, 1, -3], name=name)
    ind = cut(ser, [0, 1], labels=False)
    exp = Series([np.nan, np.nan, np.nan, 0, np.nan], name=name)
    tm.assert_series_equal(ind, exp)

def test_labels(right, breaks, closed) -> None:
    arr = np.tile(np.arange(0, 1.01, 0.1), 4)
    result, bins = cut(arr, 4, retbins=True, right=right)
    ex_levels = pd.IntervalIndex.from_breaks(breaks, closed=closed)
    tm.assert_index_equal(result.categories, ex_levels)

def test_cut_pass_series_name_to_factor() -> None:
    name = 'foo'
    ser = Series(np.random.default_rng(2).standard_normal(100), name=name)
    factor = cut(ser, 4)
    assert factor.name == name

def test_label_precision() -> None:
    arr = np.arange(0, 0.73, 0.01)
    result = cut(arr, 4, precision=2)
    ex_levels = pd.IntervalIndex.from_breaks([-0.00072, 0.18, 0.36, 0.54, 0.72])
    tm.assert_index_equal(result.categories, ex_levels)

def test_na_handling(labels) -> None:
    arr = np.arange(0, 0.75, 0.01)
    arr[::3] = np.nan
    result = cut(arr, 4, labels=labels)
    result = np.asarray(result)
    expected = np.where(pd.isna(arr), np.nan, result)
    tm.assert_almost_equal(result, expected)

def test_inf_handling() -> None:
    data = np.arange(6)
    data_ser = Series(data, dtype='int64')
    bins = [-np.inf, 2, 4, np.inf]
    result = cut(data, bins)
    result_ser = cut(data_ser, bins)
    ex_uniques = pd.IntervalIndex.from_breaks(bins)
    tm.assert_index_equal(result.categories, ex_uniques)
    assert result[5] == pd.Interval(4, np.inf)
    assert result[0] == pd.Interval(-np.inf, 2)
    assert result_ser[5] == pd.Interval(4, np.inf)
    assert result_ser[0] == pd.Interval(-np.inf, 2)

def test_cut_out_of_bounds() -> None:
    arr = np.random.default_rng(2).standard_normal(100)
    result = cut(arr, [-1, 0, 1])
    mask = pd.isna(result)
    ex_mask = (arr < -1) | (arr > 1)
    tm.assert_numpy_array_equal(mask, ex_mask)

def test_cut_pass_labels(get_labels, get_expected) -> None:
    bins = [0, 25, 50, 100]
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ['Small', 'Medium', 'Large']
    result = cut(arr, bins, labels=get_labels(labels))
    tm.assert_categorical_equal(result, get_expected(labels))

def test_cut_pass_labels_compat() -> None:
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ['Good', 'Medium', 'Bad']
    result = cut(arr, 3, labels=labels)
    exp = cut(arr, 3, labels=Categorical(labels, categories=labels, ordered=True))
    tm.assert_categorical_equal(result, exp)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with pytest.raises(ValueError, match=msg):
        cut_func(arg, 2)

def test_cut_not_1d_arg(arg, cut_func) -> None:
    msg = 'Input array must be 1 dimensional'
    with