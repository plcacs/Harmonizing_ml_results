import numpy as np
import pytest
from pandas import Categorical, DataFrame, Series
import pandas._testing as tm
from typing import Any, Callable, List, Optional, Union, Literal, Tuple, cast, overload

class TestSeriesSortValues:

    def test_sort_values(self, datetime_series: Series) -> None:
        ser = Series([3, 2, 4, 1], ['A', 'B', 'C', 'D'])
        expected = Series([1, 2, 3, 4], ['D', 'B', 'A', 'C'])
        result = ser.sort_values()
        tm.assert_series_equal(expected, result)
        ts = datetime_series.copy()
        ts[:5] = np.nan
        vals = ts.values
        result = ts.sort_values()
        assert np.isnan(result[-5:]).all()
        tm.assert_numpy_array_equal(result[:-5].values, np.sort(vals[5:]))
        result = ts.sort_values(na_position='first')
        assert np.isnan(result[:5]).all()
        tm.assert_numpy_array_equal(result[5:].values, np.sort(vals[5:]))
        ser = Series(['A', 'B'], [1, 2])
        ser.sort_values()
        ordered = ts.sort_values(ascending=False)
        expected = np.sort(ts.dropna().values)[::-1]
        tm.assert_almost_equal(expected, ordered.dropna().values)
        ordered = ts.sort_values(ascending=False, na_position='first')
        tm.assert_almost_equal(expected, ordered.dropna().values)
        ordered = ts.sort_values(ascending=[False])
        expected = ts.sort_values(ascending=False)
        tm.assert_series_equal(expected, ordered)
        ordered = ts.sort_values(ascending=[False], na_position='first')
        expected = ts.sort_values(ascending=False, na_position='first')
        tm.assert_series_equal(expected, ordered)
        msg = 'For argument "ascending" expected type bool, received type NoneType.'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=None)  # type: ignore
        msg = 'Length of ascending \\(0\\) must be 1 for Series'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[])
        msg = 'Length of ascending \\(3\\) must be 1 for Series'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[1, 2, 3])  # type: ignore
        msg = 'Length of ascending \\(2\\) must be 1 for Series'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[False, False])
        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending='foobar')  # type: ignore
        ts = datetime_series.copy()
        return_value = ts.sort_values(ascending=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(ts, datetime_series.sort_values(ascending=False))
        tm.assert_index_equal(ts.index, datetime_series.sort_values(ascending=False).index)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        s = df.iloc[:, 0]
        s.sort_values(inplace=True)
        tm.assert_series_equal(s, df.iloc[:, 0].sort_values())

    def test_sort_values_categorical(self) -> None:
        cat = Series(Categorical(['a', 'b', 'b', 'a'], ordered=False))
        expected = Series(Categorical(['a', 'a', 'b', 'b'], ordered=False), index=[0, 3, 1, 2])
        result = cat.sort_values()
        tm.assert_series_equal(result, expected)
        cat = Series(Categorical(['a', 'c', 'b', 'd'], ordered=True))
        res = cat.sort_values()
        exp = np.array(['a', 'b', 'c', 'd'], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        cat = Series(Categorical(['a', 'c', 'b', 'd'], categories=['a', 'b', 'c', 'd'], ordered=True))
        res = cat.sort_values()
        exp = np.array(['a', 'b', 'c', 'd'], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        res = cat.sort_values(ascending=False)
        exp = np.array(['d', 'c', 'b', 'a'], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        raw_cat1 = Categorical(['a', 'b', 'c', 'd'], categories=['a', 'b', 'c', 'd'], ordered=False)
        raw_cat2 = Categorical(['a', 'b', 'c', 'd'], categories=['d', 'c', 'b', 'a'], ordered=True)
        s = ['a', 'b', 'c', 'd']
        df = DataFrame({'unsort': raw_cat1, 'sort': raw_cat2, 'string': s, 'values': [1, 2, 3, 4]})
        res = df.sort_values(by=['string'], ascending=False)
        exp = np.array(['d', 'c', 'b', 'a'], dtype=np.object_)
        tm.assert_numpy_array_equal(res['sort'].values.__array__(), exp)
        assert res['sort'].dtype == 'category'
        res = df.sort_values(by=['sort'], ascending=False)
        exp = df.sort_values(by=['string'], ascending=True)
        tm.assert_series_equal(res['values'], exp['values'])
        assert res['sort'].dtype == 'category'
        assert res['unsort'].dtype == 'category'
        df.sort_values(by=['unsort'], ascending=False)
        df = DataFrame({'id': [6, 5, 4, 3, 2, 1], 'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']})
        df['grade'] = Categorical(df['raw_grade'], ordered=True)
        df['grade'] = df['grade'].cat.set_categories(['b', 'e', 'a'])
        result = df.sort_values(by=['grade'])
        expected = df.iloc[[1, 2, 5, 0, 3, 4]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['grade', 'id'])
        expected = df.iloc[[2, 1, 5, 4, 3, 0]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_list, sorted_list, ignore_index, output_index', [([2, 3, 6, 1], [6, 3, 2, 1], True, [0, 1, 2, 3]), ([2, 3, 6, 1], [6, 3, 2, 1], False, [2, 1, 0, 3])])
    def test_sort_values_ignore_index(self, inplace: bool, original_list: List[int], sorted_list: List[int], ignore_index: bool, output_index: List[int]) -> None:
        ser = Series(original_list)
        expected = Series(sorted_list, index=output_index)
        kwargs = {'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_ser = ser.copy()
            result_ser.sort_values(ascending=False, **kwargs)
        else:
            result_ser = ser.sort_values(ascending=False, **kwargs)
        tm.assert_series_equal(result_ser, expected)
        tm.assert_series_equal(ser, Series(original_list))

    def test_mergesort_descending_stability(self) -> None:
        s = Series([1, 2, 1, 3], ['first', 'b', 'second', 'c'])
        result = s.sort_values(ascending=False, kind='mergesort')
        expected = Series([3, 2, 1, 1], ['c', 'b', 'first', 'second'])
        tm.assert_series_equal(result, expected)

    def test_sort_values_validate_ascending_for_value_error(self) -> None:
        ser = Series([23, 7, 21])
        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            ser.sort_values(ascending='False')  # type: ignore

    def test_sort_values_validate_ascending_functional(self, ascending: bool) -> None:
        ser = Series([23, 7, 21])
        expected = np.sort(ser.values)
        sorted_ser = ser.sort_values(ascending=ascending)
        if not ascending:
            expected = expected[::-1]
        result = sorted_ser.values
        tm.assert_numpy_array_equal(result, expected)

class TestSeriesSortingKey:

    def test_sort_values_key(self) -> None:
        series = Series(np.array(['Hello', 'goodbye']))
        result = series.sort_values(axis=0)
        expected = series
        tm.assert_series_equal(result, expected)
        result = series.sort_values(axis=0, key=lambda x: x.str.lower())
        expected = series[::-1]
        tm.assert_series_equal(result, expected)

    def test_sort_values_key_nan(self) -> None:
        series = Series(np.array([0, 5, np.nan, 3, 2, np.nan]))
        result = series.sort_values(axis=0)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_series_equal(result, expected)
        result = series.sort_values(axis=0, key=lambda x: x + 5)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_series_equal(result, expected)
        result = series.sort_values(axis=0, key=lambda x: -x, ascending=False)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_series_equal(result, expected)
