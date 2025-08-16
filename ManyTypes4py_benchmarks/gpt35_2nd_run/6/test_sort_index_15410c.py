import numpy as np
import pytest
import pandas as pd
from pandas import CategoricalDtype, CategoricalIndex, DataFrame, IntervalIndex, MultiIndex, RangeIndex, Series, Timestamp
import pandas._testing as tm

class TestDataFrameSortIndex:

    def test_sort_index_and_reconstruction_doc_example(self) -> None:
        df: DataFrame = DataFrame({'value': [1, 2, 3, 4]}, index=MultiIndex(levels=[['a', 'b'], ['bb', 'aa']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        assert df.index._is_lexsorted()
        assert not df.index.is_monotonic_increasing
        expected: DataFrame = DataFrame({'value': [2, 1, 4, 3]}, index=MultiIndex(levels=[['a', 'b'], ['aa', 'bb']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        result: DataFrame = df.sort_index()
        assert result.index.is_monotonic_increasing
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index().copy()
        result.index = result.index._sort_levels_monotonic()
        assert result.index.is_monotonic_increasing
        tm.assert_frame_equal(result, expected)

    def test_sort_index_non_existent_label_multiindex(self) -> None:
        df: DataFrame = DataFrame(0, columns=[], index=MultiIndex.from_product([[], []]))
        with tm.assert_produces_warning(None):
            df.loc['b', '2'] = 1
            df.loc['a', '3'] = 1
        result: bool = df.sort_index().index.is_monotonic_increasing
        assert result is True

    def test_sort_index_reorder_on_ops(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((8, 2)), index=MultiIndex.from_product([['a', 'b'], ['big', 'small'], ['red', 'blu']], names=['letter', 'size', 'color']), columns=['near', 'far'])
        df = df.sort_index()

        def my_func(group):
            group.index = ['newz', 'newa']
            return group
        result: DataFrame = df.groupby(level=['letter', 'size']).apply(my_func).sort_index()
        expected: MultiIndex = MultiIndex.from_product([['a', 'b'], ['big', 'small'], ['newa', 'newz']], names=['letter', 'size', None])
        tm.assert_index_equal(result.index, expected)

    def test_sort_index_nan_multiindex(self) -> None:
        tuples: list = [[12, 13], [np.nan, np.nan], [np.nan, 3], [1, 2]]
        mi: MultiIndex = MultiIndex.from_tuples(tuples)
        df: DataFrame = DataFrame(np.arange(16).reshape(4, 4), index=mi, columns=list('ABCD'))
        s: Series = Series(np.arange(4), index=mi)
        df2: DataFrame = DataFrame({'date': pd.DatetimeIndex(['20121002', '20121007', '20130130', '20130202', '20130305', '20121002', '20121207', '20130130', '20130202', '20130305', '20130202', '20130305']), 'user_id': [1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5], 'whole_cost': [1790, np.nan, 280, 259, np.nan, 623, 90, 312, np.nan, 301, 359, 801], 'cost': [12, 15, 10, 24, 39, 1, 0, np.nan, 45, 34, 1, 12]}).set_index(['date', 'user_id'])
        result: DataFrame = df.sort_index()
        expected: DataFrame = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index(na_position='last')
        expected: DataFrame = df.iloc[[3, 0, 2, 1], :]
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index(na_position='first')
        expected: DataFrame = df.iloc[[1, 2, 3, 0], :]
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df2.dropna().sort_index()
        expected: DataFrame = df2.sort_index().dropna()
        tm.assert_frame_equal(result, expected)
        result: Series = s.sort_index()
        expected: Series = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)
        result: Series = s.sort_index(na_position='last')
        expected: Series = s.iloc[[3, 0, 2, 1]]
        tm.assert_series_equal(result, expected)
        result: Series = s.sort_index(na_position='first')
        expected: Series = s.iloc[[1, 2, 3, 0]]
        tm.assert_series_equal(result, expected)

    def test_sort_index_nan(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, np.nan, 1, 6, 8, 4], 'B': [9, np.nan, 5, 2, 5, 4, 5]}, index=[1, 2, 3, 4, 5, 6, np.nan])
        sorted_df: DataFrame = df.sort_index(kind='quicksort', ascending=True, na_position='last')
        expected: DataFrame = DataFrame({'A': [1, 2, np.nan, 1, 6, 8, 4], 'B': [9, np.nan, 5, 2, 5, 4, 5]}, index=[1, 2, 3, 4, 5, 6, np.nan])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df: DataFrame = df.sort_index(na_position='first')
        expected: DataFrame = DataFrame({'A': [4, 1, 2, np.nan, 1, 6, 8], 'B': [5, 9, np.nan, 5, 2, 5, 4]}, index=[np.nan, 1, 2, 3, 4, 5, 6])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df: DataFrame = df.sort_index(kind='quicksort', ascending=False)
        expected: DataFrame = DataFrame({'A': [8, 6, 1, np.nan, 2, 1, 4], 'B': [4, 5, 2, 5, np.nan, 9, 5]}, index=[6, 5, 4, 3, 2, 1, np.nan])
        tm.assert_frame_equal(sorted_df, expected)
        sorted_df: DataFrame = df.sort_index(kind='quicksort', ascending=False, na_position='first')
        expected: DataFrame = DataFrame({'A': [4, 8, 6, 1, np.nan, 2, 1], 'B': [5, 4, 5, 2, 5, np.nan, 9]}, index=[np.nan, 6, 5, 4, 3, 2, 1])
        tm.assert_frame_equal(sorted_df, expected)

    def test_sort_index_multi_index(self) -> None:
        df: DataFrame = DataFrame({'a': [3, 1, 2], 'b': [0, 0, 0], 'c': [0, 1, 2], 'd': list('abc')})
        result: DataFrame = df.set_index(list('abc')).sort_index(level=list('ba'))
        expected: DataFrame = DataFrame({'a': [1, 2, 3], 'b': [0, 0, 0], 'c': [1, 2, 0], 'd': list('bca')})
        expected: DataFrame = expected.set_index(list('abc'))
        tm.assert_frame_equal(result, expected)

    def test_sort_index_inplace(self) -> None:
        frame: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
        unordered: DataFrame = frame.loc[[3, 2, 4, 1]]
        a_values: Series = unordered['A']
        df: DataFrame = unordered.copy()
        return_value: None = df.sort_index(inplace=True)
        assert return_value is None
        expected: DataFrame = frame
        tm.assert_frame_equal(df, expected)
        assert a_values is not df['A']
        df: DataFrame = unordered.copy()
        return_value: None = df.sort_index(ascending=False, inplace=True)
        assert return_value is None
        expected: DataFrame = frame[::-1]
        tm.assert_frame_equal(df, expected)
        unordered: DataFrame = frame.loc[:, ['D', 'B', 'C', 'A']]
        df: DataFrame = unordered.copy()
        return_value: None = df.sort_index(axis=1, inplace=True)
        assert return_value is None
        expected: DataFrame = frame
        tm.assert_frame_equal(df, expected)
        df: DataFrame = unordered.copy()
        return_value: None = df.sort_index(axis=1, ascending=False, inplace=True)
        assert return_value is None
        expected: DataFrame = frame.iloc[:, ::-1]
        tm.assert_frame_equal(df, expected)

    def test_sort_index_different_sortorder(self) -> None:
        A: np.ndarray = np.arange(20).repeat(5)
        B: np.ndarray = np.tile(np.arange(5), 20)
        indexer: np.ndarray = np.random.default_rng(2).permutation(100)
        A: np.ndarray = A.take(indexer)
        B: np.ndarray = B.take(indexer)
        df: DataFrame = DataFrame({'A': A, 'B': B, 'C': np.random.default_rng(2).standard_normal(100)})
        ex_indexer: np.ndarray = np.lexsort((df.B.max() - df.B, df.A))
        expected: DataFrame = df.take(ex_indexer)
        idf: DataFrame = df.set_index(['A', 'B'])
        result: DataFrame = idf.sort_index(ascending=[1, 0])
        expected: DataFrame = idf.take(ex_indexer)
        tm.assert_frame_equal(result, expected)
        result: Series = idf['C'].sort_index(ascending=[1, 0])
        tm.assert_series_equal(result, expected['C'])

    def test_sort_index_level(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 3], [1, 1, 1]], names=list('ABC'))
        df: DataFrame = DataFrame([[1, 2], [3, 4]], mi)
        result: DataFrame = df.sort_index(level='A', sort_remaining=False)
        expected: DataFrame = df
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index(level=['A', 'B'], sort_remaining=False)
        expected: DataFrame = df
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index(level=['C', 'B', 'A'])
        expected: DataFrame = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index(level=['B', 'C', 'A'])
        expected: DataFrame = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index(level=['C', 'A'])
        expected: DataFrame = df.iloc[[1, 0]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index_categorical_index(self) -> None:
        df: DataFrame = DataFrame({'A': np.arange(6, dtype='int64'), 'B': Series(list('aabbca')).astype(CategoricalDtype(list('cab'))}).set_index('B')
        result: DataFrame = df.sort_index()
        expected: DataFrame = df.iloc[[4, 0, 1, 5, 2, 3]]
        tm.assert_frame_equal(result, expected)
        result: DataFrame = df.sort_index(ascending=False)
        expected: DataFrame = df.iloc[[2, 3, 0, 1, 5, 4]]
        tm.assert_frame_equal(result, expected)

    def test_sort_index(self) -> None:
        frame: DataFrame = DataFrame(np.arange(16).reshape(4, 4), index=[1, 2, 3, 4], columns=['A', 'B', 'C', 'D'])
        unordered: DataFrame = frame.loc[[3, 2, 4, 1]]
        result: DataFrame = unordered.sort_index(axis=0)
        expected: DataFrame = frame
        tm.assert_frame_equal(result, expected)
        result: DataFrame = unordered.sort_index(ascending=False)
        expected: DataFrame = frame[::-1]
        tm.assert_frame_equal(result, expected)
        unordered: DataFrame = frame.iloc[:, [2, 1, 3, 0]]
        result: DataFrame = unordered.sort_index(axis=1)
        tm.assert_frame_equal(result, frame)
        result: DataFrame = unordered.sort_index(axis=1, ascending=False)
        expected: DataFrame = frame.iloc[:, ::-1]
        tm.assert_frame_equal(result, expected)

    def test_sort_index_multiindex(self, level: str) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([[2, 1, 3], [2, 1, 2], [1, 1, 1]], names=list('ABC'))
        df: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6]], index=mi)
        expected_mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 1], [2, 1, 2], [2, 1, 3]], names=list('ABC'))
        expected: DataFrame = DataFrame([[5, 6], [3, 4], [1, 2]], index=expected_mi)
        result: DataFrame = df.sort_index(level=level)
        tm.assert_frame_equal(result, expected)
        expected_mi: MultiIndex = MultiIndex.from_tuples([[1, 1, 1], [2, 1, 3], [2, 1, 2]], names=list('ABC'))
        expected: DataFrame = DataFrame([[5, 6], [1, 2], [3, 4]], index=expected_mi)
        result: DataFrame = df.sort_index(level=level, sort_remaining=False)
        tm.assert_frame_equal(result, expected)

    def test_sort_index_intervalindex(self) -> None:
        y: Series = Series(np.random.default_rng(2).standard_normal(100))
        x1: Series = Series(np.sign(np.random.default_rng(2).standard_normal(100)))
        x2: Series = pd.cut(Series(np.random.default_rng(2).standard_normal(100)), bins=[-3, -0.5, 0, 0.5, 3])
        model: DataFrame = pd.concat([y, x1, x2], axis=1, keys=['Y', 'X1', 'X2'])
        result: DataFrame = model.groupby(['X1', 'X2'], observed=True).mean().unstack()
        expected: IntervalIndex = IntervalIndex.from_tuples([(-3.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 3.0)], closed='right')
        result: Index = result.columns.levels[1].categories
        tm.assert_index_equal(result, expected)

    def test_sort_index_ignore_index(self, original_dict: dict, sorted_dict: dict, ascending: bool, ignore_index: bool, output_index: list) -> None:
        original_index: list = [2, 5, 3]
        df: DataFrame = DataFrame(original_dict, index=original_index)
        expected_df: DataFrame = DataFrame(sorted_dict, index=output_index)
        kwargs: dict = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': True}
        if inplace:
            result_df: DataFrame = df.copy()
            result_df.sort_index(**kwargs)
        else:
            result_df: DataFrame = df.sort_index(**kwargs)
        tm.assert_frame_equal(result_df, expected_df)
        tm.assert_frame_equal(df, DataFrame(original_dict, index=original_index))

    def test_respect_ignore_index(self, inplace: bool, ignore_index: bool) -> None:
        df: DataFrame = DataFrame({'a': [1, 2, 3]}, index=RangeIndex(4, -1, -2))
        result: DataFrame = df.sort_index(ascending=False, ignore_index=ignore_index, inplace=inplace)
        if inplace:
            result: DataFrame = df
        if ignore_index:
            expected: DataFrame = DataFrame({'a': [1, 2, 3]})
        else:
            expected: DataFrame = DataFrame({'a': [1, 2, 3]}, index=RangeIndex(4, -1, -2))
        tm.assert_frame_equal(result, expected)

    def test_sort_index_ignore_index_multi_index(self, inplace: bool, original_dict: dict, sorted_dict: dict, ascending: bool, ignore_index: bool, output_index: MultiIndex) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([(2, 1), (3, 4)], names=list('AB'))
        df: DataFrame = DataFrame(original_dict, index=mi)
        expected_df: DataFrame = DataFrame(sorted_dict, index=output_index)
        kwargs: dict = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': inplace}
        if inplace:
            result_df: DataFrame = df.copy()
            result_df.sort_index(**kwargs)
        else:
            result_df: DataFrame = df.sort_index(**kwargs)
        tm.assert_frame_equal(result_df, expected_df)
        tm.assert_frame_equal(df, DataFrame(original_dict, index=mi))

    def test_sort_index_categorical_multiindex(self) -> None:
        df: DataFrame = DataFrame({'a': range(6), 'l1': pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c'], categories=['c', 'a', 'b'], ordered=True), 'l2': [0, 1, 0, 1, 0, 1]})
        result: DataFrame = df.set_index(['l1', 'l2']).sort_index()
        expected: DataFrame = DataFrame([4, 5, 0, 1, 2, 3], columns=['a'], index