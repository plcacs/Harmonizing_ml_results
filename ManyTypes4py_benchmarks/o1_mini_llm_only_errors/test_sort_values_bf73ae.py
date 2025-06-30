import numpy as np
import pytest
from pandas import Categorical, DataFrame, Series
import pandas._testing as tm
from typing import List, Callable, Any, Optional


class TestSeriesSortValues:

    def test_sort_values(self, datetime_series: Series) -> None:
        ser: Series = Series([3, 2, 4, 1], index=['A', 'B', 'C', 'D'])
        expected: Series = Series([1, 2, 3, 4], index=['D', 'B', 'A', 'C'])
        result: Series = ser.sort_values()
        tm.assert_series_equal(expected, result)
        ts: Series = datetime_series.copy()
        ts.iloc[:5] = np.nan
        vals: np.ndarray = ts.values
        result = ts.sort_values()
        assert np.isnan(result[-5:]).all()
        tm.assert_numpy_array_equal(result[:-5].values, np.sort(vals[5:]))
        result = ts.sort_values(na_position='first')
        assert np.isnan(result[:5]).all()
        tm.assert_numpy_array_equal(result[5:].values, np.sort(vals[5:]))
        ser = Series(['A', 'B'], index=[1, 2])
        ser.sort_values()
        ordered: Series = ts.sort_values(ascending=False)
        expected_sorted: np.ndarray = np.sort(ts.dropna().values)[::-1]
        tm.assert_almost_equal(expected_sorted, ordered.dropna().values)
        ordered = ts.sort_values(ascending=False, na_position='first')
        tm.assert_almost_equal(expected_sorted, ordered.dropna().values)
        ordered = ts.sort_values(ascending=[False])
        expected_ordered: Series = ts.sort_values(ascending=False)
        tm.assert_series_equal(expected_ordered, ordered)
        ordered = ts.sort_values(ascending=[False], na_position='first')
        expected_ordered = ts.sort_values(ascending=False, na_position='first')
        tm.assert_series_equal(expected_ordered, ordered)
        msg: str = 'For argument "ascending" expected type bool, received type NoneType.'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=None)
        msg = 'Length of ascending \\(0\\) must be 1 for Series'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[])
        msg = 'Length of ascending \\(3\\) must be 1 for Series'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[1, 2, 3])
        msg = 'Length of ascending \\(2\\) must be 1 for Series'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending=[False, False])
        msg = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            ts.sort_values(ascending='foobar')
        ts = datetime_series.copy()
        return_value: Optional[None] = ts.sort_values(ascending=False, inplace=True)
        assert return_value is None
        tm.assert_series_equal(ts, datetime_series.sort_values(ascending=False))
        tm.assert_index_equal(ts.index, datetime_series.sort_values(ascending=False).index)
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        s: Series = df.iloc[:, 0]
        s.sort_values(inplace=True)
        tm.assert_series_equal(s, df.iloc[:, 0].sort_values())

    def test_sort_values_categorical(self) -> None:
        cat: Series = Series(Categorical(['a', 'b', 'b', 'a'], ordered=False))
        expected: Series = Series(
            Categorical(['a', 'a', 'b', 'b'], ordered=False), index=[0, 3, 1, 2]
        )
        result: Series = cat.sort_values()
        tm.assert_series_equal(result, expected)
        cat = Series(Categorical(['a', 'c', 'b', 'd'], ordered=True))
        res: Series = cat.sort_values()
        exp: np.ndarray = np.array(['a', 'b', 'c', 'd'], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        cat = Series(
            Categorical(['a', 'c', 'b', 'd'], categories=['a', 'b', 'c', 'd'], ordered=True)
        )
        res = cat.sort_values()
        exp = np.array(['a', 'b', 'c', 'd'], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        res = cat.sort_values(ascending=False)
        exp = np.array(['d', 'c', 'b', 'a'], dtype=object)
        tm.assert_numpy_array_equal(res.__array__(), exp)
        raw_cat1: Categorical = Categorical(
            ['a', 'b', 'c', 'd'],
            categories=['a', 'b', 'c', 'd'],
            ordered=False
        )
        raw_cat2: Categorical = Categorical(
            ['a', 'b', 'c', 'd'],
            categories=['d', 'c', 'b', 'a'],
            ordered=True
        )
        s: List[str] = ['a', 'b', 'c', 'd']
        df: DataFrame = DataFrame({
            'unsort': raw_cat1,
            'sort': raw_cat2,
            'string': s,
            'values': [1, 2, 3, 4],
        })
        res = df.sort_values(by=['string'], ascending=False)
        exp = np.array(['d', 'c', 'b', 'a'], dtype=object)
        tm.assert_numpy_array_equal(res['sort'].values.__array__(), exp)
        assert res['sort'].dtype == 'category'
        res = df.sort_values(by=['sort'], ascending=False)
        exp = df.sort_values(by=['string'], ascending=True)
        tm.assert_series_equal(res['values'], exp['values'])
        assert res['sort'].dtype == 'category'
        assert res['unsort'].dtype == 'category'
        df.sort_values(by=['unsort'], ascending=False)
        df = DataFrame({
            'id': [6, 5, 4, 3, 2, 1],
            'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e'],
        })
        df['grade'] = Categorical(df['raw_grade'], ordered=True)
        df['grade'] = df['grade'].cat.set_categories(['b', 'e', 'a'])
        result: DataFrame = df.sort_values(by=['grade'])
        expected = df.iloc[[1, 2, 5, 0, 3, 4]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_values(by=['grade', 'id'])
        expected = df.iloc[[2, 1, 5, 4, 3, 0]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize(
        'original_list, sorted_list, ignore_index, output_index',
        [
            (
                [2, 3, 6, 1],
                [6, 3, 2, 1],
                True,
                [0, 1, 2, 3],
            ),
            (
                [2, 3, 6, 1],
                [6, 3, 2, 1],
                False,
                [2, 1, 0, 3],
            ),
        ],
    )
    def test_sort_values_ignore_index(
        self,
        inplace: bool,
        original_list: List[int],
        sorted_list: List[int],
        ignore_index: bool,
        output_index: List[int],
    ) -> None:
        ser: Series = Series(original_list)
        expected: Series = Series(sorted_list, index=output_index)
        kwargs: dict[str, Any] = {'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_ser: Series = ser.copy()
            result_ser.sort_values(ascending=False, **kwargs)
        else:
            result_ser: Series = ser.sort_values(ascending=False, **kwargs)
        tm.assert_series_equal(result_ser, expected)
        tm.assert_series_equal(ser, Series(original_list))

    def test_mergesort_descending_stability(self) -> None:
        s: Series = Series([1, 2, 1, 3], index=['first', 'b', 'second', 'c'])
        result: Series = s.sort_values(ascending=False, kind='mergesort')
        expected: Series = Series([3, 2, 1, 1], index=['c', 'b', 'first', 'second'])
        tm.assert_series_equal(result, expected)

    def test_sort_values_validate_ascending_for_value_error(self) -> None:
        ser: Series = Series([23, 7, 21])
        msg: str = 'For argument "ascending" expected type bool, received type str.'
        with pytest.raises(ValueError, match=msg):
            ser.sort_values(ascending='False')

    @pytest.mark.parametrize('ascending', [True, False])
    def test_sort_values_validate_ascending_functional(self, ascending: bool) -> None:
        ser: Series = Series([23, 7, 21])
        expected: np.ndarray = np.sort(ser.values)
        sorted_ser: Series = ser.sort_values(ascending=ascending)
        if not ascending:
            expected = expected[::-1]
        result: np.ndarray = sorted_ser.values
        tm.assert_numpy_array_equal(result, expected)


class TestSeriesSortingKey:

    def test_sort_values_key(self) -> None:
        series: Series = Series(np.array(['Hello', 'goodbye']))
        result: Series = series.sort_values(axis=0)
        expected: Series = series
        tm.assert_series_equal(result, expected)
        result = series.sort_values(axis=0, key=lambda x: x.str.lower())
        expected = series[::-1]
        tm.assert_series_equal(result, expected)

    def test_sort_values_key_nan(self) -> None:
        series: Series = Series(np.array([0, 5, np.nan, 3, 2, np.nan]))
        result: Series = series.sort_values(axis=0)
        expected: Series = series.iloc[[0, 4, 3, 1, 2, 5]]
        tm.assert_series_equal(result, expected)
        result = series.sort_values(axis=0, key=lambda x: x + 5)
        tm.assert_series_equal(result, expected)
        result = series.sort_values(axis=0, key=lambda x: -x, ascending=False)
        tm.assert_series_equal(result, expected)
