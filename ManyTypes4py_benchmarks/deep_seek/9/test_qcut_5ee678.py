import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, DatetimeIndex, Interval, IntervalIndex, NaT, Series, Timedelta, TimedeltaIndex, Timestamp, cut, date_range, isna, qcut, timedelta_range
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
from pandas.core.dtypes.common import is_scalar
from numpy.typing import NDArray

def test_qcut() -> None:
    arr: NDArray[np.float64] = np.random.default_rng(2).standard_normal(1000)
    labels, _ = qcut(arr, 4, retbins=True)
    ex_bins: NDArray[np.float64] = np.quantile(arr, [0, 0.25, 0.5, 0.75, 1.0])
    result: NDArray[np.float64] = labels.categories.left.values
    assert np.allclose(result, ex_bins[:-1], atol=0.01)
    result = labels.categories.right.values
    assert np.allclose(result, ex_bins[1:], atol=0.01)
    ex_levels: Categorical = cut(arr, ex_bins, include_lowest=True)
    tm.assert_categorical_equal(labels, ex_levels)

def test_qcut_bounds() -> None:
    arr: NDArray[np.float64] = np.random.default_rng(2).standard_normal(1000)
    factor: NDArray[np.int64] = qcut(arr, 10, labels=False)
    assert len(np.unique(factor)) == 10

def test_qcut_specify_quantiles() -> None:
    arr: NDArray[np.float64] = np.random.default_rng(2).standard_normal(100)
    factor: Categorical = qcut(arr, [0, 0.25, 0.5, 0.75, 1.0])
    expected: Categorical = qcut(arr, 4)
    tm.assert_categorical_equal(factor, expected)

def test_qcut_all_bins_same() -> None:
    with pytest.raises(ValueError, match='edges.*unique'):
        qcut([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3)

def test_qcut_include_lowest() -> None:
    values: NDArray[np.int64] = np.arange(10)
    ii: Categorical = qcut(values, 4)
    ex_levels: IntervalIndex = IntervalIndex([Interval(-0.001, 2.25), Interval(2.25, 4.5), Interval(4.5, 6.75), Interval(6.75, 9)])
    tm.assert_index_equal(ii.categories, ex_levels)

def test_qcut_nas() -> None:
    arr: NDArray[np.float64] = np.random.default_rng(2).standard_normal(100)
    arr[:20] = np.nan
    result: Categorical = qcut(arr, 4)
    assert isna(result[:20]).all()

def test_qcut_index() -> None:
    result: Categorical = qcut([0, 2], 2)
    intervals: List[Interval[float]] = [Interval(-0.001, 1), Interval(1, 2)]
    expected: Categorical = Categorical(intervals, ordered=True)
    tm.assert_categorical_equal(result, expected)

def test_qcut_binning_issues(datapath: str) -> None:
    cut_file: str = datapath(os.path.join('reshape', 'data', 'cut_data.csv'))
    arr: NDArray[np.float64] = np.loadtxt(cut_file)
    result: Categorical = qcut(arr, 20)
    starts: List[float] = []
    ends: List[float] = []
    for lev in np.unique(result):
        s: float = lev.left
        e: float = lev.right
        assert s != e
        starts.append(float(s))
        ends.append(float(e))
    for (sp, sn), (ep, en) in zip(zip(starts[:-1], starts[1:]), zip(ends[:-1], ends[1:])):
        assert sp < sn
        assert ep < en
        assert ep <= sn

def test_qcut_return_intervals() -> None:
    ser: Series[int] = Series([0, 1, 2, 3, 4, 5, 6, 7, 8])
    res: Categorical = qcut(ser, [0, 0.333, 0.666, 1])
    exp_levels: NDArray[Interval[float]] = np.array([Interval(-0.001, 2.664), Interval(2.664, 5.328), Interval(5.328, 8)])
    exp: Series[Categorical] = Series(exp_levels.take([0, 0, 0, 1, 1, 1, 2, 2, 2])).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(res, exp)

@pytest.mark.parametrize('labels', ['foo', 1, True])
def test_qcut_incorrect_labels(labels: Union[str, int, bool]) -> None:
    values: range = range(5)
    msg: str = 'Bin labels must either be False, None or passed in as a list-like argument'
    with pytest.raises(ValueError, match=msg):
        qcut(values, 4, labels=labels)

@pytest.mark.parametrize('labels', [['a', 'b', 'c'], list(range(3))])
def test_qcut_wrong_length_labels(labels: Union[List[str], List[int]]) -> None:
    values: range = range(10)
    msg: str = 'Bin labels must be one fewer than the number of bin edges'
    with pytest.raises(ValueError, match=msg):
        qcut(values, 4, labels=labels)

@pytest.mark.parametrize('labels, expected', [(['a', 'b', 'c'], ['a', 'b', 'c']), (list(range(3)), [0, 1, 2])])
def test_qcut_list_like_labels(labels: Union[List[str], List[int]], expected: Union[List[str], List[int]]) -> None:
    values: range = range(3)
    result: Categorical = qcut(values, 3, labels=labels)
    expected_cat: Categorical = Categorical(expected, ordered=True)
    tm.assert_categorical_equal(result, expected_cat)

@pytest.mark.parametrize('kwargs,msg', [({'duplicates': 'drop'}, None), ({}, 'Bin edges must be unique'), ({'duplicates': 'raise'}, 'Bin edges must be unique'), ({'duplicates': 'foo'}, "invalid value for 'duplicates' parameter")])
def test_qcut_duplicates_bin(kwargs: Dict[str, Any], msg: Optional[str]) -> None:
    values: List[int] = [0, 0, 0, 0, 1, 2, 3]
    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            qcut(values, 3, **kwargs)
    else:
        result: Categorical = qcut(values, 3, **kwargs)
        expected: IntervalIndex = IntervalIndex([Interval(-0.001, 1), Interval(1, 3)])
        tm.assert_index_equal(result.categories, expected)

@pytest.mark.parametrize('data,start,end', [(9.0, 8.999, 9.0), (0.0, -0.001, 0.0), (-9.0, -9.001, -9.0)])
@pytest.mark.parametrize('length', [1, 2])
@pytest.mark.parametrize('labels', [None, False])
def test_single_quantile(data: float, start: float, end: float, length: int, labels: Optional[bool]) -> None:
    ser: Series[float] = Series([data] * length)
    result: Union[Series[Categorical], Series[int]] = qcut(ser, 1, labels=labels)
    if labels is None:
        intervals: IntervalIndex = IntervalIndex([Interval(start, end)] * length, closed='right')
        expected: Series[Categorical] = Series(intervals).astype(CategoricalDtype(ordered=True))
    else:
        expected: Series[int] = Series([0] * length, dtype=np.intp)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ser', [DatetimeIndex(['20180101', NaT, '20180103']), TimedeltaIndex(['0 days', NaT, '2 days'])], ids=lambda x: str(x.dtype))
def test_qcut_nat(ser: Union[DatetimeIndex, TimedeltaIndex], unit: str) -> None:
    ser_series: Series[Union[Timestamp, Timedelta]] = Series(ser)
    ser_series = ser_series.dt.as_unit(unit)
    td: Timedelta = Timedelta(1, unit=unit).as_unit(unit)
    left: Series[Union[Timestamp, Timedelta]] = Series([ser_series[0] - td, np.nan, ser_series[2] - Day()], dtype=ser_series.dtype)
    right: Series[Union[Timestamp, Timedelta]] = Series([ser_series[2] - Day(), np.nan, ser_series[2]], dtype=ser_series.dtype)
    intervals: IntervalIndex = IntervalIndex.from_arrays(left, right)
    expected: Series[Categorical] = Series(Categorical(intervals, ordered=True))
    result: Categorical = qcut(ser_series, 2)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('bins', [3, np.linspace(0, 1, 4)])
def test_datetime_tz_qcut(bins: Union[int, NDArray[np.float64]]) -> None:
    tz: str = 'US/Eastern'
    ser: Series[Timestamp] = Series(date_range('20130101', periods=3, tz=tz))
    result: Categorical = qcut(ser, bins)
    expected: Series[Categorical] = Series(IntervalIndex([Interval(Timestamp('2012-12-31 23:59:59.999999999', tz=tz), Timestamp('2013-01-01 16:00:00', tz=tz)), Interval(Timestamp('2013-01-01 16:00:00', tz=tz), Timestamp('2013-01-02 08:00:00', tz=tz)), Interval(Timestamp('2013-01-02 08:00:00', tz=tz), Timestamp('2013-01-03 00:00:00', tz=tz))])).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('arg,expected_bins', [[timedelta_range('1day', periods=3), TimedeltaIndex(['1 days', '2 days', '3 days'])], [date_range('20180101', periods=3), DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'])]])
def test_date_like_qcut_bins(arg: Union[TimedeltaIndex, DatetimeIndex], expected_bins: Union[TimedeltaIndex, DatetimeIndex], unit: str) -> None:
    arg = arg.as_unit(unit)
    expected_bins = expected_bins.as_unit(unit)
    ser: Series[Union[Timedelta, Timestamp]] = Series(arg)
    result: Categorical
    result_bins: Union[TimedeltaIndex, DatetimeIndex]
    result, result_bins = qcut(ser, 2, retbins=True)
    tm.assert_index_equal(result_bins, expected_bins)

@pytest.mark.parametrize('bins', [6, 7])
@pytest.mark.parametrize('box, compare', [(Series, tm.assert_series_equal), (np.array, tm.assert_categorical_equal), (list, tm.assert_equal)])
def test_qcut_bool_coercion_to_int(bins: int, box: Any, compare: Any) -> None:
    data_expected: Union[Series[int], NDArray[np.int64], List[int]] = box([0, 1, 1, 0, 1] * 10)
    data_result: Union[Series[bool], NDArray[np.bool_], List[bool]] = box([False, True, True, False, True] * 10)
    expected: Categorical = qcut(data_expected, bins, duplicates='drop')
    result: Categorical = qcut(data_result, bins, duplicates='drop')
    compare(result, expected)

@pytest.mark.parametrize('q', [2, 5, 10])
def test_qcut_nullable_integer(q: int, any_numeric_ea_dtype: str) -> None:
    arr: pd.arrays.IntegerArray = pd.array(np.arange(100), dtype=any_numeric_ea_dtype)
    arr[::2] = pd.NA
    result: Categorical = qcut(arr, q)
    expected: Categorical = qcut(arr.astype(float), q)
    tm.assert_categorical_equal(result, expected)

@pytest.mark.parametrize('scale', [1.0, 1 / 3, 17.0])
@pytest.mark.parametrize('q', [3, 7, 9])
@pytest.mark.parametrize('precision', [1, 3, 16])
def test_qcut_contains(scale: float, q: int, precision: int) -> None:
    arr: NDArray[np.float64] = (scale * np.arange(q + 1)).round(precision)
    result: Categorical = qcut(arr, q, precision=precision)
    for value, bucket in zip(arr, result):
        assert value in bucket
