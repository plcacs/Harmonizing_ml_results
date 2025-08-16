import numpy as np
import pytest
from pandas import Categorical, DataFrame, Series
import pandas._testing as tm

class TestSeriesSortValues:

    def test_sort_values(self, datetime_series: Series) -> None:
        ser: Series = Series([3, 2, 4, 1], ['A', 'B', 'C', 'D'])
        expected: Series = Series([1, 2, 3, 4], ['D', 'B', 'A', 'C'])
        result: Series = ser.sort_values()
        ts: Series = datetime_series.copy()
        ts[:5] = np.nan
        vals: np.ndarray = ts.values
        result = ts.sort_values()
        result = ts.sort_values(na_position='first')
        ser = Series(['A', 'B'], [1, 2])
        ordered = ts.sort_values(ascending=False)
        ordered = ts.sort_values(ascending=False, na_position='first')
        ordered = ts.sort_values(ascending=[False])
        ordered = ts.sort_values(ascending=[False], na_position='first')
        ts.sort_values(ascending=None)
        ts.sort_values(ascending=[])
        ts.sort_values(ascending=[1, 2, 3])
        ts.sort_values(ascending=[False, False])
        ts.sort_values(ascending='foobar')
        ts.sort_values(ascending=False, inplace=True)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        s = df.iloc[:, 0]
        s.sort_values(inplace=True)

    def test_sort_values_categorical(self) -> None:
        cat: Series = Series(Categorical(['a', 'b', 'b', 'a'], ordered=False))
        expected: Series = Series(Categorical(['a', 'a', 'b', 'b'], ordered=False), index=[0, 3, 1, 2])
        result: Series = cat.sort_values()
        cat = Series(Categorical(['a', 'c', 'b', 'd'], ordered=True)
        res = cat.sort_values()
        cat = Series(Categorical(['a', 'c', 'b', 'd'], categories=['a', 'b', 'c', 'd'], ordered=True))
        res = cat.sort_values()
        res = cat.sort_values(ascending=False)
        raw_cat1 = Categorical(['a', 'b', 'c', 'd'], categories=['a', 'b', 'c', 'd'], ordered=False)
        raw_cat2 = Categorical(['a', 'b', 'c', 'd'], categories=['d', 'c', 'b', 'a'], ordered=True)
        s = ['a', 'b', 'c', 'd']
        df = DataFrame({'unsort': raw_cat1, 'sort': raw_cat2, 'string': s, 'values': [1, 2, 3, 4]})
        res = df.sort_values(by=['string'], ascending=False)
        res = df.sort_values(by=['sort'], ascending=False)
        df.sort_values(by=['unsort'], ascending=False)
        df = DataFrame({'id': [6, 5, 4, 3, 2, 1], 'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']})
        df['grade'] = Categorical(df['raw_grade'], ordered=True)
        df['grade'] = df['grade'].cat.set_categories(['b', 'e', 'a'])
        result = df.sort_values(by=['grade'])
        result = df.sort_values(by=['grade', 'id'])

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_list, sorted_list, ignore_index, output_index', [([2, 3, 6, 1], [6, 3, 2, 1], True, [0, 1, 2, 3]), ([2, 3, 6, 1], [6, 3, 2, 1], False, [2, 1, 0, 3])])
    def test_sort_values_ignore_index(self, inplace: bool, original_list: list, sorted_list: list, ignore_index: bool, output_index: list) -> None:
        ser: Series = Series(original_list)
        expected: Series = Series(sorted_list, index=output_index)
        kwargs: dict = {'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_ser: Series = ser.copy()
            result_ser.sort_values(ascending=False, **kwargs)
        else:
            result_ser: Series = ser.sort_values(ascending=False, **kwargs)

    def test_mergesort_descending_stability(self) -> None:
        s: Series = Series([1, 2, 1, 3], ['first', 'b', 'second', 'c'])
        result: Series = s.sort_values(ascending=False, kind='mergesort')
        expected: Series = Series([3, 2, 1, 1], ['c', 'b', 'first', 'second'])

    def test_sort_values_validate_ascending_for_value_error(self) -> None:
        ser: Series = Series([23, 7, 21])
        ser.sort_values(ascending='False')

    def test_sort_values_validate_ascending_functional(self, ascending: bool) -> None:
        ser: Series = Series([23, 7, 21])
        expected: np.ndarray = np.sort(ser.values)
        sorted_ser: Series = ser.sort_values(ascending=ascending)
        if not ascending:
            expected = expected[::-1]
        result: np.ndarray = sorted_ser.values

class TestSeriesSortingKey:

    def test_sort_values_key(self) -> None:
        series: Series = Series(np.array(['Hello', 'goodbye']))
        result: Series = series.sort_values(axis=0)
        expected: Series = series
        result: Series = series.sort_values(axis=0, key=lambda x: x.str.lower())
        expected: Series = series[::-1]

    def test_sort_values_key_nan(self) -> None:
        series: Series = Series(np.array([0, 5, np.nan, 3, 2, np.nan]))
        result: Series = series.sort_values(axis=0)
        expected: Series = series.iloc[[0, 4, 3, 1, 2, 5]]
        result: Series = series.sort_values(axis=0, key=lambda x: x + 5)
        expected: Series = series.iloc[[0, 4, 3, 1, 2, 5]]
        result: Series = series.sort_values(axis=0, key=lambda x: -x, ascending=False)
        expected: Series = series.iloc[[0, 4, 3, 1, 2, 5]]
