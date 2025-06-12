import numpy as np
import pytest
import pandas as pd
from pandas import CategoricalDtype, CategoricalIndex, DataFrame, IntervalIndex, MultiIndex, RangeIndex, Series, Timestamp
import pandas._testing as tm
from typing import Any, Dict, List, Optional, Tuple, Union

class TestDataFrameSortIndex:

    def test_sort_index_and_reconstruction_doc_example(self) -> None:
        df = DataFrame({'value': [1, 2, 3, 4]}, index=MultiIndex(levels=[['a', 'b'], ['bb', 'aa']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]))
        assert df.index._is_lexsorted()
        assert not df.index.is_monotonic_increasing
        expected = DataFrame({'value': [2, 1, 4, 3]}, index=MultiIndex(levels=[['a', 'b'], ['aa', 'bb']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]))
        result = df.sort_index()
        assert result.index.is_monotonic_increasing
        tm.assert_frame_equal(result, expected)
        result = df.sort_index().copy()
        result.index = result.index._sort_levels_monotonic()
        assert result.index.is_monotonic_increasing
        tm.assert_frame_equal(result, expected)

    def test_sort_index_non_existent_label_multiindex(self) -> None:
        df = DataFrame(0, columns=[], index=MultiIndex.from_product([[], []]))
        with tm.assert_produces_warning(None):
            df.loc['b', '2'] = 1
            df.loc['a', '3'] = 1
        result = df.sort_index().index.is_monotonic_increasing
        assert result is True

    def test_sort_index_reorder_on_ops(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((8, 2)), index=MultiIndex.from_product([['a', 'b'], ['big', 'small'], ['red', 'blu']], names=['letter', 'size', 'color']), columns=['near', 'far'])
        df = df.sort_index()

        def my_func(group: DataFrame) -> DataFrame:
            group.index = ['newz', 'newa']
            return group
        result = df.groupby(level=['letter', 'size']).apply(my_func).sort_index()
        expected = MultiIndex.from_product([['a', 'b'], ['big', 'small'], ['newa', 'newz']], names=['letter', 'size', None])
        tm.assert_index_equal(result.index, expected)

    def test_sort_index_nan_multiindex(self) -> None:
        tuples = [[12, 13], [np.nan, np.nan], [np.nan, 3], [1, 2]]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(np.arange(16).reshape(4, 4), index=mi, columns=list('ABCD'))
        s = Series(np.arange(4), index=mi)
        df2 = DataFrame({'date': pd.DatetimeIndex(['20121002', '20121007', '20130130', '20130202', '20130305', '20121002', '20121207', '20130130', '20130202', '20130305', '20130202', '20130305']), 'user_id': [1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5], 'whole_cost': [1790, np.nan, 280, 259, np.nan, 623, 90, 312, np.nan, 301, 359, 801], 'cost': [12, 15, 10, 24, 39, 1, 0, np.nan, 45, 34, 1, 12]}).set_index(['date', 'user_id'])
        result = df.sort_index()
        expected = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(na_position='last')
        expected = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(na_position='first')
        expected = df.iloc[[1, 2, 3, 0], :]
        tm.assert_frame_equal(result, expected)
        result = df2.dropna().sort_index()
        expected = df2.sort_index().dropna()
        tm.assert_frame_equal(result, expected)
        result = s.sort_index()
        expected = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(na_position='last')
        expected = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)
        result = s.sort_index(na_position='first')
        expected = s.iloc[[1, 2, 3, 0]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_nan(self) -> None:
        df = DataFrame({'A': [1, 2, np.nan, 1, 6, 8, 4], 'B': [9, np.nan, 5, 2, 5, 4, 5]}, index=[1, 2, 3, 4, 5, 6, np.nan])
        sorted_df = df.sort_index(kind='quicksort', ascending=True, na_position='last')
        expected = DataFrame({'A': [1, 2, np.nan, 1, 6, 8, 4], 'B': [9, np.nan, 5, 2, 5, 4, 5]}, index=[1, 2, 3, 4, 5, 6, np.nan])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = df.sort_index(na_position='first')
        expected = DataFrame({'A': [4, 1, 2, np.nan, 1, 6, 8], 'B': [5, 9, np.nan, 5, 2, 5, 4]}, index=[np.nan, 1, 2, 3, 4, 5, 6])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = df.sort_index(kind='quicksort', ascending=False)
        expected = DataFrame({'A': [8, 6, 1, np.nan, 2, 1, 4], 'B': [4, 5, 2, 5, np.nan, 9, 5]}, index=[6, 5, 4, 3, 2, 1, np.nan])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df = df.sort_index(kind='quicksort', ascending=False, na_position='first')
        expected = DataFrame({'A': [4, 8, 6, 1, np.nan, 2, 1], 'B': [5, 4, 5, 2, 5, np.nan, 9]}, index=[np.nan, 6, 5, 4, 3, 2, 1])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_index_multi_index(self) -> None:
        df = DataFrame({'a': [3, 1, 2], 'b': [0, 0, 0], 'c': [0, 1, 2], 'd': list('abc')})
        result = df.set_index(list('abc')).sort_index(level=list('ba'))
        expected = DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0], 'c': [1, 2, 0], 'd': list('bca')})
        expected = expected.set_index(list('abc'))
        tm.assert_frame_equal(result, expected)

    def test_sort_index_inplace(self) -> None:
        frame = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
        unordered = frame.loc[[3, 2, 4, 1]]
        a_values = unordered['A']
        df = unordered.copy()
        return_value = df.sort_index(inplace=True)
        assert return_value is None
        expected = frame
        tm.assert_frame_equal(df, expected)
        assert a_values is not df['A']
        df = unordered.copy()
        return_value = df.sort_index(ascending=False, inplace=True)
        assert return_value is None
        expected = frame[::-1]
        tm.assert_frame_equal(df, expected)
        unordered = frame.loc[:, ['D', 'B', 'C', 'A']]
        df = unordered.copy()
        return_value = df.sort_index(axis=1, inplace=True)
        assert return_value is None
        expected = frame
        tm.assert_frame_equal(df, expected)
        df = unordered.copy()
        return_value = df.sort_index(axis=1, ascending=False, inplace=True)
        assert return_value is None
        expected = frame.iloc[:, ::-1]
        tm.assert_frame_equal(df, expected)

    def test_sort_index_different_sortorder(self) -> None:
        A = np.arange(20).repeat(5)
        B = np.tile(np.arange(5), 20)
        indexer = np.random.default_rng(2).permutation(100)
        A = A.take(indexer)
        B = B.take(indexer)
        df = DataFrame({'A': A, 'B': B, 'C': np.random.default_rng(2).standard_normal(100)})
        ex_indexer = np.lexsort((df.B.max() - df.B, df.A))
        expected = df.take(ex_indexer)
        idf = df.set_index(['A', 'B'])
        result = idf.sort_index(ascending=[1, 0])
        expected = idf.take(ex_indexer)
        tm.assert_frame_equal(result, expected)
        result = idf['C'].sort_index(ascending=[1, 0])
        tm.assert_series_equal(result, expected['C'])

    def test_sort_index_level(self) -> None:
        mi = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        df = DataFrame([[1, 2], [3, 4]], mi)
        result = df.sort_index(level='A', sort_remaining=False)
        expected = df
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(level=['A', 'B'], sort_remaining=False)
        expected = df
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(level=['C', 'B', 'A'])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(level=['B', 'C', 'A'])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(level=['C', 'A'])
        expected = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index_categorical_index(self) -> None:
        df = DataFrame({'A': np.arange(6, dtype='int64'), 'B': Series(list('aabbca')).astype(CategoricalDtype(list('cab')))}).set_index('B')
        result = df.sort_index()
        expected = df.iloc[[4, 0, 1, 5, 2, 3]]
        tm.assert_frame_equal(result, expected)
        result = df.sort_index(ascending=False)
        expected = df.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index(self) -> None:
        frame = DataFrame(np.arange(16).reshape(4, 4), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
        unordered = frame.loc[[3, 2, 4, 1]]
        result = unordered.sort_index(axis=0)
        expected = frame
        tm.assert_frame_equal(result, expected)
        result = unordered.sort_index(ascending=False)
        expected = frame[::-1]
        tm.assert_frame_equal(result, expected)
        unordered = frame.iloc[:, [2, 1, 3, 0]]
        result = unordered.sort_index(axis=1)
        tm.assert_frame_equal(result, frame)
        result = unordered.sort_index(axis=1, ascending=False)
        expected = frame.iloc[:, ::-1]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('level', ['A', 0])
    def test_sort_index_multiindex(self, level: Union[str, int]) -> None:
        mi = MultiIndex.from_tuples([[2, 1, 3], [2, 1, 2], [1, 1, 1]], names=list('ABC'))
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mi)
        expected_mi = MultiIndex.from_tuples([[1, 1, 1], [2, 1, 2], [2, 1, 3]], names=list('ABC'))
        expected = DataFrame([[5, 6], [3, 4], [1, 2]], index=expected_mi)
        result = df.sort_index(level=level)
        tm.assert_frame_equal(result, expected)
        expected_mi = MultiIndex.from_tuples([[1, 1, 1], [2, 1, 3], [2, 1, 2]], names=list('ABC'))
        expected = DataFrame([[5, 6], [1, 2], [3, 4]], index=expected_mi)
        result = df.sort_index(level=level, sort_remaining=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_index_intervalindex(self) -> None:
        y = Series(np.random.default_rng(2).standard_normal(100))
        x1 = Series(np.sign(np.random.default_rng(2).standard_normal(100)))
        x2 = pd.cut(Series(np.random.default_rng(2).standard_normal(100)), bins=[-3, -0.5, 0, 0.5, 3])
        model = pd.concat([y, x1, x2], axis=1, keys=['Y', 'X1', 'X2'])
        result = model.groupby(['X1', 'X2'], observed=True).mean().unstack()
        expected = IntervalIndex.from_tuples([(-3.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 3.0)], closed='right')
        result = result.columns.levels[1].categories
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_dict, sorted_dict, ascending, ignore_index, output_index', [({'A': [1, 2, 3]}, {'A': [2, 3, 1]}, False, True, [0, 1, 2]), ({'A': [1, 2, 3]}, {'A': [1, 3, 2]}, True, True, [0, 1, 2]), ({'A': [1, 2, 3]}, {'A': [2, 3, 1]}, False, False, [5, 3, 2]), ({'A': [1, 2, 3]}, {'A': [1, 3, 2]}, True, False, [2, 3, 5])])
    def test_sort_index_ignore_index(self, inplace: bool, original_dict: Dict[str, List[int]], sorted_dict: Dict[str, List[int]], ascending: bool, ignore_index: bool, output_index: List[int]) -> None:
        original_index = [2, 5, 3]
        df = DataFrame(original_dict, index=original_index)
        expected_df = DataFrame(sorted_dict, index=output_index)
        kwargs = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_df = df.copy()
            result_df.sort_index(**kwargs)
        else:
            result_df = df.sort_index(**kwargs)
        tm.assert_frame_equal(result_df, expected_df)
        tm.assert_frame_equal(df, DataFrame(original_dict, index=original_index))

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('ignore_index', [True, False])
    def test_respect_ignore_index(self, inplace: bool, ignore_index: bool) -> None:
        df = DataFrame({'a': [1, 2, 3]}, index=RangeIndex(4, -1, -2))
        result = df.sort_index(ascending=False, ignore_index=ignore_index, inplace=inplace)
        if inplace:
            result = df
        if ignore_index:
            expected = DataFrame({'a': [1, 2, 3]})
        else:
            expected = DataFrame({'a': [1, 2, 3]}, index=RangeIndex(4, -1, -2))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('original_dict, sorted_dict, ascending, ignore_index, output_index', [({'M1': [1, 2], 'M2': [3, 4]}, {'M1': [1, 2], 'M2': [3, 4]}, True, True, [0, 1]), ({'M1': [1, 2], 'M2': [