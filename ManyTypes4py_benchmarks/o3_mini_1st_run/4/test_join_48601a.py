#!/usr/bin/env python
from typing import List, Tuple, Union, Any, Iterable
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Categorical, DataFrame, Index, MultiIndex, Series, Timestamp, bdate_range, concat, merge, option_context
import pandas._testing as tm

def get_test_data(ngroups: int = 8, n: int = 50) -> np.ndarray:
    unique_groups: List[int] = list(range(ngroups))
    arr: np.ndarray = np.asarray(np.tile(unique_groups, n // ngroups))
    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])
    np.random.default_rng(2).shuffle(arr)
    return arr

class TestJoin:
    @pytest.fixture
    def df(self) -> DataFrame:
        df: DataFrame = DataFrame({
            'key1': get_test_data(),
            'key2': get_test_data(),
            'data1': np.random.default_rng(2).standard_normal(50),
            'data2': np.random.default_rng(2).standard_normal(50)
        })
        df = df[df['key2'] > 1]
        return df

    @pytest.fixture
    def df2(self) -> DataFrame:
        return DataFrame({
            'key1': get_test_data(n=10),
            'key2': get_test_data(ngroups=4, n=10),
            'value': np.random.default_rng(2).standard_normal(10)
        })

    @pytest.fixture
    def target_source(self) -> Tuple[DataFrame, DataFrame]:
        data = {
            'A': [0.0, 1.0, 2.0, 3.0, 4.0],
            'B': [0.0, 1.0, 0.0, 1.0, 0.0],
            'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'],
            'D': bdate_range('1/1/2009', periods=5)
        }
        target: DataFrame = DataFrame(data, index=Index(['a', 'b', 'c', 'd', 'e'], dtype=object))
        source: DataFrame = DataFrame({'MergedA': data['A'], 'MergedD': data['D']}, index=data['C'])
        return (target, source)

    def test_left_outer_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2: DataFrame = merge(df, df2, on='key2')
        _check_join(df, df2, joined_key2, ['key2'], how='left')
        joined_both: DataFrame = merge(df, df2)
        _check_join(df, df2, joined_both, ['key1', 'key2'], how='left')

    def test_right_outer_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2: DataFrame = merge(df, df2, on='key2', how='right')
        _check_join(df, df2, joined_key2, ['key2'], how='right')
        joined_both: DataFrame = merge(df, df2, how='right')
        _check_join(df, df2, joined_both, ['key1', 'key2'], how='right')

    def test_full_outer_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2: DataFrame = merge(df, df2, on='key2', how='outer')
        _check_join(df, df2, joined_key2, ['key2'], how='outer')
        joined_both: DataFrame = merge(df, df2, how='outer')
        _check_join(df, df2, joined_both, ['key1', 'key2'], how='outer')

    def test_inner_join(self, df: DataFrame, df2: DataFrame) -> None:
        joined_key2: DataFrame = merge(df, df2, on='key2', how='inner')
        _check_join(df, df2, joined_key2, ['key2'], how='inner')
        joined_both: DataFrame = merge(df, df2, how='inner')
        _check_join(df, df2, joined_both, ['key1', 'key2'], how='inner')

    def test_handle_overlap(self, df: DataFrame, df2: DataFrame) -> None:
        joined: DataFrame = merge(df, df2, on='key2', suffixes=('.foo', '.bar'))
        assert 'key1.foo' in joined
        assert 'key1.bar' in joined

    def test_handle_overlap_arbitrary_key(self, df: DataFrame, df2: DataFrame) -> None:
        joined: DataFrame = merge(df, df2, left_on='key2', right_on='key1', suffixes=('.foo', '.bar'))
        assert 'key1.foo' in joined
        assert 'key2.bar' in joined

    @pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
    def test_join_on(self, target_source: Tuple[DataFrame, DataFrame], infer_string: bool) -> None:
        target, source = target_source
        merged: DataFrame = target.join(source, on='C')
        tm.assert_series_equal(merged['MergedA'], target['A'], check_names=False)
        tm.assert_series_equal(merged['MergedD'], target['D'], check_names=False)
        df: DataFrame = DataFrame({'key': ['a', 'a', 'b', 'b', 'c']})
        df2: DataFrame = DataFrame({'value': [0, 1, 2]}, index=['a', 'b', 'c'])
        joined: DataFrame = df.join(df2, on='key')
        expected: DataFrame = DataFrame({'key': ['a', 'a', 'b', 'b', 'c'], 'value': [0, 0, 1, 1, 2]})
        tm.assert_frame_equal(joined, expected)
        df_a: DataFrame = DataFrame([[1], [2], [3]], index=['a', 'b', 'c'], columns=['one'])
        df_b: DataFrame = DataFrame([['foo'], ['bar']], index=[1, 2], columns=['two'])
        df_c: DataFrame = DataFrame([[1], [2]], index=[1, 2], columns=['three'])
        joined = df_a.join(df_b, on='one')
        joined = joined.join(df_c, on='one')
        assert np.isnan(joined['two']['c'])
        assert np.isnan(joined['three']['c'])
        with pytest.raises(KeyError, match="^'E'$"):
            target.join(source, on='E')
        msg: str = "You are trying to merge on float64 and object|str columns for key 'A'. If you wish to proceed you should use pd.concat"
        with pytest.raises(ValueError, match=msg):
            target.join(source, on='A')

    def test_join_on_fails_with_different_right_index(self) -> None:
        df: DataFrame = DataFrame({
            'a': np.random.default_rng(2).choice(['m', 'f'], size=3),
            'b': np.random.default_rng(2).standard_normal(3)
        })
        df2: DataFrame = DataFrame({
            'a': np.random.default_rng(2).choice(['m', 'f'], size=10),
            'b': np.random.default_rng(2).standard_normal(10)
        }, index=MultiIndex.from_product([range(5), ['A', 'B']]))
        msg: str = 'len\\(left_on\\) must equal the number of levels in the index of "right"'
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on='a', right_index=True)

    def test_join_on_fails_with_different_left_index(self) -> None:
        df: DataFrame = DataFrame({
            'a': np.random.default_rng(2).choice(['m', 'f'], size=3),
            'b': np.random.default_rng(2).standard_normal(3)
        }, index=MultiIndex.from_arrays([range(3), list('abc')]))
        df2: DataFrame = DataFrame({
            'a': np.random.default_rng(2).choice(['m', 'f'], size=10),
            'b': np.random.default_rng(2).standard_normal(10)
        })
        msg: str = 'len\\(right_on\\) must equal the number of levels in the index of "left"'
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, right_on='b', left_index=True)

    def test_join_on_fails_with_different_column_counts(self) -> None:
        df: DataFrame = DataFrame({
            'a': np.random.default_rng(2).choice(['m', 'f'], size=3),
            'b': np.random.default_rng(2).standard_normal(3)
        })
        df2: DataFrame = DataFrame({
            'a': np.random.default_rng(2).choice(['m', 'f'], size=10),
            'b': np.random.default_rng(2).standard_normal(10)
        }, index=MultiIndex.from_product([range(5), ['A', 'B']]))
        msg: str = 'len\\(right_on\\) must equal len\\(left_on\\)'
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, right_on='a', left_on=['a', 'b'])

    @pytest.mark.parametrize('wrong_type', [2, 'str', None, np.array([0, 1])])
    def test_join_on_fails_with_wrong_object_type(self, wrong_type: Any) -> None:
        df: DataFrame = DataFrame({'a': [1, 1]})
        msg: str = f'Can only merge Series or DataFrame objects, a {type(wrong_type)} was passed'
        with pytest.raises(TypeError, match=msg):
            merge(wrong_type, df, left_on='a', right_on='a')
        with pytest.raises(TypeError, match=msg):
            merge(df, wrong_type, left_on='a', right_on='a')

    def test_join_on_pass_vector(self, target_source: Tuple[DataFrame, DataFrame]) -> None:
        target, source = target_source
        expected: DataFrame = target.join(source, on='C')
        expected = expected.rename(columns={'C': 'key_0'})
        expected = expected[['key_0', 'A', 'B', 'D', 'MergedA', 'MergedD']]
        join_col: Series = target.pop('C')
        result: DataFrame = target.join(source, on=join_col)
        tm.assert_frame_equal(result, expected)

    def test_join_with_len0(self, target_source: Tuple[DataFrame, DataFrame]) -> None:
        target, source = target_source
        merged: DataFrame = target.join(source.reindex([]), on='C')
        for col in source:
            assert col in merged
            assert merged[col].isna().all()
        merged2: DataFrame = target.join(source.reindex([]), on='C', how='inner')
        tm.assert_index_equal(merged2.columns, merged.columns)
        assert len(merged2) == 0

    def test_join_on_inner(self) -> None:
        df: DataFrame = DataFrame({'key': ['a', 'a', 'd', 'b', 'b', 'c']})
        df2: DataFrame = DataFrame({'value': [0, 1]}, index=['a', 'b'])
        joined: DataFrame = df.join(df2, on='key', how='inner')
        expected: DataFrame = df.join(df2, on='key')
        expected = expected[expected['value'].notna()]
        tm.assert_series_equal(joined['key'], expected['key'])
        tm.assert_series_equal(joined['value'], expected['value'], check_dtype=False)
        tm.assert_index_equal(joined.index, expected.index)

    def test_join_on_singlekey_list(self) -> None:
        df: DataFrame = DataFrame({'key': ['a', 'a', 'b', 'b', 'c']})
        df2: DataFrame = DataFrame({'value': [0, 1, 2]}, index=['a', 'b', 'c'])
        joined: DataFrame = df.join(df2, on=['key'])
        expected: DataFrame = df.join(df2, on='key')
        tm.assert_frame_equal(joined, expected)

    def test_join_on_series(self, target_source: Tuple[DataFrame, DataFrame]) -> None:
        target, source = target_source
        result: DataFrame = target.join(source['MergedA'], on='C')
        expected: DataFrame = target.join(source[['MergedA']], on='C')
        tm.assert_frame_equal(result, expected)

    def test_join_on_series_buglet(self) -> None:
        df: DataFrame = DataFrame({'a': [1, 1]})
        ds: Series = Series([2], index=[1], name='b')
        result: DataFrame = df.join(ds, on='a')
        expected: DataFrame = DataFrame({'a': [1, 1], 'b': [2, 2]}, index=df.index)
        tm.assert_frame_equal(result, expected)

    def test_join_index_mixed(self, join_type: str) -> None:
        df1: DataFrame = DataFrame(index=np.arange(10))
        df1['bool'] = True
        df1['string'] = 'foo'
        df2: DataFrame = DataFrame(index=np.arange(5, 15))
        df2['int'] = 1
        df2['float'] = 1.0
        joined: DataFrame = df1.join(df2, how=join_type)
        expected: DataFrame = _join_by_hand(df1, df2, how=join_type)
        tm.assert_frame_equal(joined, expected)
        joined = df2.join(df1, how=join_type)
        expected = _join_by_hand(df2, df1, how=join_type)
        tm.assert_frame_equal(joined, expected)

    def test_join_index_mixed_overlap(self) -> None:
        df1: DataFrame = DataFrame({'A': 1.0, 'B': 2, 'C': 'foo', 'D': True}, index=np.arange(10), columns=['A', 'B', 'C', 'D'])
        assert df1['B'].dtype == np.int64
        assert df1['D'].dtype == np.bool_
        df2: DataFrame = DataFrame({'A': 1.0, 'B': 2, 'C': 'foo', 'D': True}, index=np.arange(0, 10, 2), columns=['A', 'B', 'C', 'D'])
        joined: DataFrame = df1.join(df2, lsuffix='_one', rsuffix='_two')
        expected_columns: List[str] = ['A_one', 'B_one', 'C_one', 'D_one', 'A_two', 'B_two', 'C_two', 'D_two']
        df1.columns = expected_columns[:4]
        df2.columns = expected_columns[4:]
        expected: DataFrame = _join_by_hand(df1, df2)
        tm.assert_frame_equal(joined, expected)

    def test_join_empty_bug(self) -> None:
        x: DataFrame = DataFrame()
        x.join(DataFrame([3], index=[0], columns=['A']), how='outer')

    def test_join_unconsolidated(self) -> None:
        a: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((30, 2)), columns=['a', 'b'])
        c: Series = Series(np.random.default_rng(2).standard_normal(30))
        a['c'] = c
        d: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((30, 1)), columns=['q'])
        a.join(d)
        d.join(a)

    def test_join_multiindex(self) -> None:
        index1: MultiIndex = MultiIndex.from_arrays([['a', 'a', 'a', 'b', 'b', 'b'], [1, 2, 3, 1, 2, 3]], names=['first', 'second'])
        index2: MultiIndex = MultiIndex.from_arrays([['b', 'b', 'b', 'c', 'c', 'c'], [1, 2, 3, 1, 2, 3]], names=['first', 'second'])
        df1: DataFrame = DataFrame(data=np.random.default_rng(2).standard_normal(6), index=index1, columns=['var X'])
        df2: DataFrame = DataFrame(data=np.random.default_rng(2).standard_normal(6), index=index2, columns=['var Y'])
        df1 = df1.sort_index(level=0)
        df2 = df2.sort_index(level=0)
        joined: DataFrame = df1.join(df2, how='outer')
        ex_index = Index(index1.values).union(Index(index2.values))
        expected: DataFrame = df1.reindex(ex_index).join(df2.reindex(ex_index))
        expected.index.names = index1.names
        tm.assert_frame_equal(joined, expected)
        assert joined.index.names == index1.names
        df1 = df1.sort_index(level=1)
        df2 = df2.sort_index(level=1)
        joined = df1.join(df2, how='outer').sort_index(level=0)
        ex_index = Index(index1.values).union(Index(index2.values))
        expected = df1.reindex(ex_index).join(df2.reindex(ex_index))
        expected.index.names = index1.names
        tm.assert_frame_equal(joined, expected)
        assert joined.index.names == index1.names

    def test_join_inner_multiindex(self, lexsorted_two_level_string_multiindex: MultiIndex) -> None:
        key1: List[str] = ['bar', 'bar', 'bar', 'foo', 'foo', 'baz', 'baz', 'qux', 'qux', 'snap']
        key2: List[str] = ['two', 'one', 'three', 'one', 'two', 'one', 'two', 'two', 'three', 'one']
        data_vals: np.ndarray = np.random.default_rng(2).standard_normal(len(key1))
        data: DataFrame = DataFrame({'key1': key1, 'key2': key2, 'data': data_vals})
        index: MultiIndex = lexsorted_two_level_string_multiindex
        to_join: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), index=index, columns=['j_one', 'j_two', 'j_three'])
        joined: DataFrame = data.join(to_join, on=['key1', 'key2'], how='inner')
        expected: DataFrame = merge(data, to_join.reset_index(), left_on=['key1', 'key2'], right_on=['first', 'second'], how='inner', sort=False)
        expected2: DataFrame = merge(to_join, data, right_on=['key1', 'key2'], left_index=True, how='inner', sort=False)
        tm.assert_frame_equal(joined, expected2.reindex_like(joined))
        expected2 = merge(to_join, data, right_on=['key1', 'key2'], left_index=True, how='inner', sort=False)
        expected = expected.drop(['first', 'second'], axis=1)
        expected.index = joined.index
        assert joined.index.is_monotonic_increasing
        tm.assert_frame_equal(joined, expected)

    def test_join_hierarchical_mixed_raises(self) -> None:
        df: DataFrame = DataFrame([(1, 2, 3), (4, 5, 6)], columns=['a', 'b', 'c'])
        new_df: DataFrame = df.groupby(['a']).agg({'b': ['mean', 'sum']})
        other_df: DataFrame = DataFrame([(1, 2, 3), (7, 10, 6)], columns=['a', 'b', 'd'])
        other_df.set_index('a', inplace=True)
        with pytest.raises(pd.errors.MergeError, match='Not allowed to merge between different levels'):
            merge(new_df, other_df, left_index=True, right_index=True)

    def test_join_float64_float32(self) -> None:
        a: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=['a', 'b'], dtype=np.float64)
        b: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=['c'], dtype=np.float32)
        joined: DataFrame = a.join(b)
        assert joined.dtypes['a'] == 'float64'
        assert joined.dtypes['b'] == 'float64'
        assert joined.dtypes['c'] == 'float32'
        a_arr: np.ndarray = np.random.default_rng(2).integers(0, 5, 100).astype('int64')
        b_arr: np.ndarray = np.random.default_rng(2).random(100).astype('float64')
        c_arr: np.ndarray = np.random.default_rng(2).random(100).astype('float32')
        df: DataFrame = DataFrame({'a': a_arr, 'b': b_arr, 'c': c_arr})
        xpdf: DataFrame = DataFrame({'a': a_arr, 'b': b_arr, 'c': c_arr})
        s: DataFrame = DataFrame(np.random.default_rng(2).random(5).astype('float32'), columns=['md'])
        rs: DataFrame = df.merge(s, left_on='a', right_index=True)
        assert rs.dtypes['a'] == 'int64'
        assert rs.dtypes['b'] == 'float64'
        assert rs.dtypes['c'] == 'float32'
        assert rs.dtypes['md'] == 'float32'
        xp: DataFrame = xpdf.merge(s, left_on='a', right_index=True)
        tm.assert_frame_equal(rs, xp)

    def test_join_many_non_unique_index(self) -> None:
        df1: DataFrame = DataFrame({'a': [1, 1], 'b': [1, 1], 'c': [10, 20]})
        df2: DataFrame = DataFrame({'a': [1, 1], 'b': [1, 2], 'd': [100, 200]})
        df3: DataFrame = DataFrame({'a': [1, 1], 'b': [1, 2], 'e': [1000, 2000]})
        idf1: DataFrame = df1.set_index(['a', 'b'])
        idf2: DataFrame = df2.set_index(['a', 'b'])
        idf3: DataFrame = df3.set_index(['a', 'b'])
        result: DataFrame = idf1.join([idf2, idf3], how='outer')
        df_partially_merged: DataFrame = merge(df1, df2, on=['a', 'b'], how='outer')
        expected: DataFrame = merge(df_partially_merged, df3, on=['a', 'b'], how='outer')
        result = result.reset_index()
        expected = expected[result.columns]
        expected['a'] = expected.a.astype('int64')
        expected['b'] = expected.b.astype('int64')
        tm.assert_frame_equal(result, expected)
        df1 = DataFrame({'a': [1, 1, 1], 'b': [1, 1, 1], 'c': [10, 20, 30]})
        df2 = DataFrame({'a': [1, 1, 1], 'b': [1, 1, 2], 'd': [100, 200, 300]})
        df3 = DataFrame({'a': [1, 1, 1], 'b': [1, 1, 2], 'e': [1000, 2000, 3000]})
        idf1 = df1.set_index(['a', 'b'])
        idf2 = df2.set_index(['a', 'b'])
        idf3 = df3.set_index(['a', 'b'])
        result = idf1.join([idf2, idf3], how='inner')
        df_partially_merged = merge(df1, df2, on=['a', 'b'], how='inner')
        expected = merge(df_partially_merged, df3, on=['a', 'b'], how='inner')
        result = result.reset_index()
        tm.assert_frame_equal(result, expected.loc[:, result.columns])
        df = DataFrame({
            'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
            'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
            'C': np.random.default_rng(2).standard_normal(8),
            'D': np.random.default_rng(2).standard_normal(8)
        })
        s: Series = Series(np.repeat(np.arange(8), 2), index=np.repeat(np.arange(8), 2), name='TEST')
        inner: DataFrame = df.join(s, how='inner')
        outer: DataFrame = df.join(s, how='outer')
        left: DataFrame = df.join(s, how='left')
        right: DataFrame = df.join(s, how='right')
        tm.assert_frame_equal(inner, outer)
        tm.assert_frame_equal(inner, left)
        tm.assert_frame_equal(inner, right)

    @pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
    def test_join_sort(self, infer_string: bool) -> None:
        with option_context('future.infer_string', infer_string):
            left: DataFrame = DataFrame({'key': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 4]})
            right: DataFrame = DataFrame({'value2': ['a', 'b', 'c']}, index=['bar', 'baz', 'foo'])
            joined: DataFrame = left.join(right, on='key', sort=True)
            expected: DataFrame = DataFrame({
                'key': ['bar', 'baz', 'foo', 'foo'],
                'value': [2, 3, 1, 4],
                'value2': ['a', 'b', 'c', 'c']
            }, index=[1, 2, 0, 3])
            tm.assert_frame_equal(joined, expected)
            joined = left.join(right, on='key', sort=False)
            tm.assert_index_equal(joined.index, Index(range(4)), exact=True)

    def test_join_mixed_non_unique_index(self) -> None:
        df1: DataFrame = DataFrame({'a': [1, 2, 3, 4]}, index=[1, 2, 3, 'a'])
        df2: DataFrame = DataFrame({'b': [5, 6, 7, 8]}, index=[1, 3, 3, 4])
        result: DataFrame = df1.join(df2)
        expected: DataFrame = DataFrame({'a': [1, 2, 3, 3, 4], 'b': [5, np.nan, 6, 7, np.nan]}, index=[1, 2, 3, 3, 'a'])
        tm.assert_frame_equal(result, expected)
        df3: DataFrame = DataFrame({'a': [1, 2, 3, 4]}, index=[1, 2, 2, 'a'])
        df4: DataFrame = DataFrame({'b': [5, 6, 7, 8]}, index=[1, 2, 3, 4])
        result = df3.join(df4)
        expected = DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 6, np.nan]}, index=[1, 2, 2, 'a'])
        tm.assert_frame_equal(result, expected)

    def test_join_non_unique_period_index(self) -> None:
        index = pd.period_range('2016-01-01', periods=16, freq='M')
        df: DataFrame = DataFrame(list(range(len(index))), index=index, columns=['pnum'])
        df2: DataFrame = concat([df, df])
        result: DataFrame = df.join(df2, how='inner', rsuffix='_df2')
        expected: DataFrame = DataFrame(
            np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
            columns=['pnum', 'pnum_df2'],
            index=df2.sort_index().index
        )
        tm.assert_frame_equal(result, expected)

    def test_mixed_type_join_with_suffix(self, using_infer_string: bool) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((20, 6)), columns=['a', 'b', 'c', 'd', 'e', 'f'])
        df.insert(0, 'id', 0)
        df.insert(5, 'dt', 'foo')
        grouped = df.groupby('id')
        msg: str = re.escape('agg function failed [how->mean,dtype->')
        if using_infer_string:
            msg = "dtype 'str' does not support operation 'mean'"
        with pytest.raises(TypeError, match=msg):
            grouped.mean()
        mn = grouped.mean(numeric_only=True)
        cn = grouped.count()
        mn.join(cn, rsuffix='_right')

    def test_join_many(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 6)), columns=list('abcdef'))
        df_list: List[DataFrame] = [df[['a', 'b']], df[['c', 'd']], df[['e', 'f']]]
        joined: DataFrame = df_list[0].join(df_list[1:])
        tm.assert_frame_equal(joined, df)
        df_list = [df[['a', 'b']][:-2], df[['c', 'd']][2:], df[['e', 'f']][1:9]]
        def _check_diff_index(df_list: List[DataFrame], result: DataFrame, exp_index: Index) -> None:
            reindexed: List[DataFrame] = [x.reindex(exp_index) for x in df_list]
            expected: DataFrame = reindexed[0].join(reindexed[1:])
            tm.assert_frame_equal(result, expected)
        joined = df_list[0].join(df_list[1:], how='outer')
        _check_diff_index(df_list, joined, df.index)
        joined = df_list[0].join(df_list[1:])
        _check_diff_index(df_list, joined, df_list[0].index)
        joined = df_list[0].join(df_list[1:], how='inner')
        _check_diff_index(df_list, joined, df.index[2:8])
        msg: str = 'Joining multiple DataFrames only supported for joining on index'
        with pytest.raises(ValueError, match=msg):
            df_list[0].join(df_list[1:], on='a')

    def test_join_many_mixed(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((8, 4)), columns=['A', 'B', 'C', 'D'])
        df['key'] = ['foo', 'bar'] * 4
        df1: DataFrame = df.loc[:, ['A', 'B']]
        df2: DataFrame = df.loc[:, ['C', 'D']]
        df3: DataFrame = df.loc[:, ['key']]
        result: DataFrame = df1.join([df2, df3])
        tm.assert_frame_equal(result, df)

    def test_join_dups(self) -> None:
        df: DataFrame = concat([
            DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B']),
            DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=['A', 'C'])
        ], axis=1)
        expected: DataFrame = concat([df, df], axis=1)
        result: DataFrame = df.join(df, rsuffix='_2')
        result.columns = expected.columns
        tm.assert_frame_equal(result, expected)
        w: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
        x: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
        y: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
        z: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), columns=['x', 'y'])
        dta: DataFrame = x.merge(y, left_index=True, right_index=True).merge(z, left_index=True, right_index=True, how='outer')
        with pytest.raises(pd.errors.MergeError, match="Passing 'suffixes' which cause duplicate columns"):
            dta.merge(w, left_index=True, right_index=True)

    def test_join_multi_to_multi(self, join_type: str) -> None:
        leftindex: MultiIndex = MultiIndex.from_product([list('abc'), list('xy'), [1, 2]], names=['abc', 'xy', 'num'])
        left: DataFrame = DataFrame({'v1': range(12)}, index=leftindex)
        rightindex: MultiIndex = MultiIndex.from_product([list('abc'), list('xy')], names=['abc', 'xy'])
        right: DataFrame = DataFrame({'v2': [100 * i for i in range(1, 7)]}, index=rightindex)
        result: DataFrame = left.join(right, on=['abc', 'xy'], how=join_type)
        expected: DataFrame = left.reset_index().merge(right.reset_index(), on=['abc', 'xy'], how=join_type).set_index(['abc', 'xy', 'num'])
        tm.assert_frame_equal(expected, result)
        msg: str = 'len\\(left_on\\) must equal the number of levels in the index of "right"'
        with pytest.raises(ValueError, match=msg):
            left.join(right, on='xy', how=join_type)
        with pytest.raises(ValueError, match=msg):
            right.join(left, on=['abc', 'xy'], how=join_type)

    def test_join_on_tz_aware_datetimeindex(self) -> None:
        df1: DataFrame = DataFrame({
            'date': pd.date_range(start='2018-01-01', periods=5, tz='America/Chicago'),
            'vals': list('abcde')
        })
        df2: DataFrame = DataFrame({
            'date': pd.date_range(start='2018-01-03', periods=5, tz='America/Chicago'),
            'vals_2': list('tuvwx')
        })
        result: DataFrame = df1.join(df2.set_index('date'), on='date')
        expected: DataFrame = df1.copy()
        expected['vals_2'] = Series([np.nan] * 2 + list('tuv'))
        tm.assert_frame_equal(result, expected)

    def test_join_datetime_string(self) -> None:
        dfa: DataFrame = DataFrame([
            ['2012-08-02', 'L', 10],
            ['2012-08-02', 'J', 15],
            ['2013-04-06', 'L', 20],
            ['2013-04-06', 'J', 25]
        ], columns=['x', 'y', 'a'])
        dfa['x'] = pd.to_datetime(dfa['x']).astype('M8[ns]')
        dfb: DataFrame = DataFrame([
            ['2012-08-02', 'J', 1],
            ['2013-04-06', 'L', 2]
        ], columns=['x', 'y', 'z'], index=[2, 4])
        dfb['x'] = pd.to_datetime(dfb['x']).astype('M8[ns]')
        result: DataFrame = dfb.join(dfa.set_index(['x', 'y']), on=['x', 'y'])
        expected: DataFrame = DataFrame([
            [Timestamp('2012-08-02 00:00:00'), 'J', 1, 15],
            [Timestamp('2013-04-06 00:00:00'), 'L', 2, 20]
        ], index=[2, 4], columns=['x', 'y', 'z', 'a'])
        expected['x'] = expected['x'].astype('M8[ns]')
        tm.assert_frame_equal(result, expected)

    def test_join_with_categorical_index(self) -> None:
        ix: List[str] = ['a', 'b']
        id1: pd.CategoricalIndex = pd.CategoricalIndex(ix, categories=ix)
        id2: pd.CategoricalIndex = pd.CategoricalIndex(list(reversed(ix)), categories=list(reversed(ix)))
        df1: DataFrame = DataFrame({'c1': ix}, index=id1)
        df2: DataFrame = DataFrame({'c2': list(reversed(ix))}, index=id2)
        result: DataFrame = df1.join(df2)
        expected: DataFrame = DataFrame({'c1': ['a', 'b'], 'c2': ['a', 'b']}, index=pd.CategoricalIndex(['a', 'b'], categories=['a', 'b']))
        tm.assert_frame_equal(result, expected)

def _check_join(left: DataFrame, right: DataFrame, result: DataFrame, join_col: List[str], how: str = 'left', lsuffix: str = '_x', rsuffix: str = '_y') -> None:
    for c in join_col:
        assert result[c].notna().all()
    left_grouped = left.groupby(join_col)
    right_grouped = right.groupby(join_col)
    for group_key, group in result.groupby(join_col):
        l_joined: DataFrame = _restrict_to_columns(group, list(left.columns), lsuffix)
        r_joined: DataFrame = _restrict_to_columns(group, list(right.columns), rsuffix)
        try:
            lgroup: DataFrame = left_grouped.get_group(group_key)
        except KeyError as err:
            if how in ('left', 'inner'):
                raise AssertionError(f'key {group_key} should not have been in the join') from err
            _assert_all_na(l_joined, list(left.columns), join_col)
        else:
            _assert_same_contents(l_joined, lgroup)
        try:
            rgroup: DataFrame = right_grouped.get_group(group_key)
        except KeyError as err:
            if how in ('right', 'inner'):
                raise AssertionError(f'key {group_key} should not have been in the join') from err
            _assert_all_na(r_joined, list(right.columns), join_col)
        else:
            _assert_same_contents(r_joined, rgroup)

def _restrict_to_columns(group: DataFrame, columns: Iterable[str], suffix: str) -> DataFrame:
    found: List[str] = [c for c in group.columns if c in columns or c.replace(suffix, '') in columns]
    group = group.loc[:, found]
    group = group.rename(columns=lambda x: x.replace(suffix, ''))
    group = group.loc[:, columns]
    return group

def _assert_same_contents(join_chunk: DataFrame, source: DataFrame) -> None:
    NA_SENTINEL = -1234567
    jvalues = join_chunk.fillna(NA_SENTINEL).drop_duplicates().values
    svalues = source.fillna(NA_SENTINEL).drop_duplicates().values
    rows = {tuple(row) for row in jvalues}
    assert len(rows) == len(source)
    assert all((tuple(row) in rows for row in svalues))

def _assert_all_na(join_chunk: DataFrame, source_columns: List[str], join_col: List[str]) -> None:
    for c in source_columns:
        if c in join_col:
            continue
        assert join_chunk[c].isna().all()

def _join_by_hand(a: DataFrame, b: DataFrame, how: str = 'left') -> DataFrame:
    join_index: Index = a.index.join(b.index, how=how)
    a_re: DataFrame = a.reindex(join_index)
    b_re: DataFrame = b.reindex(join_index)
    result_columns = a.columns.append(b.columns)
    for col, s in b_re.items():
        a_re[col] = s
    return a_re.reindex(columns=result_columns)

def test_join_inner_multiindex_deterministic_order() -> None:
    left: DataFrame = DataFrame(data={'e': 5}, index=MultiIndex.from_tuples([(1, 2, 4)], names=('a', 'b', 'd')))
    right: DataFrame = DataFrame(data={'f': 6}, index=MultiIndex.from_tuples([(2, 3)], names=('b', 'c')))
    result: DataFrame = left.join(right, how='inner')
    expected: DataFrame = DataFrame({'e': [5], 'f': [6]}, index=MultiIndex.from_tuples([(1, 2, 4, 3)], names=('a', 'b', 'd', 'c')))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize(('input_col', 'output_cols'), [('b', ['a', 'b']), ('a', ['a_x', 'a_y'])])
def test_join_cross(input_col: str, output_cols: List[str]) -> None:
    left: DataFrame = DataFrame({'a': [1, 3]})
    right: DataFrame = DataFrame({input_col: [3, 4]})
    result: DataFrame = left.join(right, how='cross', lsuffix='_x', rsuffix='_y')
    expected: DataFrame = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    tm.assert_frame_equal(result, expected)

def test_join_multiindex_one_level(join_type: str) -> None:
    left: DataFrame = DataFrame(data={'c': 3}, index=MultiIndex.from_tuples([(1, 2)], names=('a', 'b')))
    right: DataFrame = DataFrame(data={'d': 4}, index=MultiIndex.from_tuples([(2,)], names=('b',)))
    result: DataFrame = left.join(right, how=join_type)
    if join_type == 'right':
        expected: DataFrame = DataFrame({'c': [3], 'd': [4]}, index=MultiIndex.from_tuples([(2, 1)], names=['b', 'a']))
    else:
        expected = DataFrame({'c': [3], 'd': [4]}, index=MultiIndex.from_tuples([(1, 2)], names=['a', 'b']))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('categories, values', [
    (['Y', 'X'], ['Y', 'X', 'X']),
    ([2, 1], [2, 1, 1]),
    ([2.5, 1.5], [2.5, 1.5, 1.5]),
    ([Timestamp('2020-12-31'), Timestamp('2019-12-31')], [Timestamp('2020-12-31'), Timestamp('2019-12-31'), Timestamp('2019-12-31')])
])
def test_join_multiindex_not_alphabetical_categorical(categories: List[Any], values: List[Any]) -> None:
    left: DataFrame = DataFrame({
        'first': ['A', 'A'],
        'second': pd.Categorical(categories, categories=categories),
        'value': [1, 2]
    }).set_index(['first', 'second'])
    right: DataFrame = DataFrame({
        'first': ['A', 'A', 'B'],
        'second': pd.Categorical(values, categories=categories),
        'value': [3, 4, 5]
    }).set_index(['first', 'second'])
    result: DataFrame = left.join(right, lsuffix='_left', rsuffix='_right')
    expected: DataFrame = DataFrame({
        'first': ['A', 'A'],
        'second': pd.Categorical(categories, categories=categories),
        'value_left': [1, 2],
        'value_right': [3, 4]
    }).set_index(['first', 'second'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('left_empty, how, exp', [
    (False, 'left', 'left'),
    (False, 'right', 'empty'),
    (False, 'inner', 'empty'),
    (False, 'outer', 'left'),
    (False, 'cross', 'empty'),
    (True, 'left', 'empty'),
    (True, 'right', 'right'),
    (True, 'inner', 'empty'),
    (True, 'outer', 'right'),
    (True, 'cross', 'empty')
])
def test_join_empty(left_empty: bool, how: str, exp: str) -> None:
    left: DataFrame = DataFrame({'A': [2, 1], 'B': [3, 4]}, dtype='int64').set_index('A')
    right: DataFrame = DataFrame({'A': [1], 'C': [5]}, dtype='int64').set_index('A')
    if left_empty:
        left = left.head(0)
    else:
        right = right.head(0)
    result: DataFrame = left.join(right, how=how)
    if exp == 'left':
        expected: DataFrame = DataFrame({'A': [2, 1], 'B': [3, 4], 'C': [np.nan, np.nan]})
        expected = expected.set_index('A')
    elif exp == 'right':
        expected = DataFrame({'B': [np.nan], 'A': [1], 'C': [5]})
        expected = expected.set_index('A')
    elif exp == 'empty':
        expected = DataFrame(columns=['B', 'C'], dtype='int64')
        if how != 'cross':
            expected = expected.rename_axis('A')
    if how == 'outer':
        expected = expected.sort_index()
    tm.assert_frame_equal(result, expected)

def test_join_empty_uncomparable_columns() -> None:
    df1: DataFrame = DataFrame()
    df2: DataFrame = DataFrame(columns=['test'])
    df3: DataFrame = DataFrame(columns=['foo', ('bar', 'baz')])
    result: DataFrame = df1 + df2
    expected: DataFrame = DataFrame(columns=['test'])
    tm.assert_frame_equal(result, expected)
    result = df2 + df3
    expected = DataFrame(columns=[('bar', 'baz'), 'foo', 'test'])
    tm.assert_frame_equal(result, expected)
    result = df1 + df3
    expected = DataFrame(columns=[('bar', 'baz'), 'foo'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('how, values', [
    ('inner', [0, 1, 2]),
    ('outer', [0, 1, 2]),
    ('left', [0, 1, 2]),
    ('right', [0, 2, 1])
])
def test_join_multiindex_categorical_output_index_dtype(how: str, values: List[int]) -> None:
    df1: DataFrame = DataFrame({
        'a': pd.Categorical([0, 1, 2]),
        'b': pd.Categorical([0, 1, 2]),
        'c': [0, 1, 2]
    }).set_index(['a', 'b'])
    df2: DataFrame = DataFrame({
        'a': pd.Categorical([0, 2, 1]),
        'b': pd.Categorical([0, 2, 1]),
        'd': [0, 2, 1]
    }).set_index(['a', 'b'])
    expected: DataFrame = DataFrame({
        'a': pd.Categorical(values),
        'b': pd.Categorical(values),
        'c': values,
        'd': values
    }).set_index(['a', 'b'])
    result: DataFrame = df1.join(df2, how=how)
    tm.assert_frame_equal(result, expected)

def test_join_multiindex_with_none_as_label() -> None:
    df1: DataFrame = DataFrame({'A': [1]}, index=MultiIndex.from_tuples([(3, 3)], names=['X', None]))
    df2: DataFrame = DataFrame({'B': [2]}, index=MultiIndex.from_tuples([(3, 3)], names=[None, 'X']))
    result12: DataFrame = df1.join(df2)
    expected12: DataFrame = DataFrame({'A': [1], 'B': [2]}, index=MultiIndex.from_tuples([(3, 3)], names=['X', None]))
    tm.assert_frame_equal(result12, expected12)
    result21: DataFrame = df2.join(df1)
    expected21: DataFrame = DataFrame({'B': [2], 'A': [1]}, index=MultiIndex.from_tuples([(3, 3)], names=[None, 'X']))
    tm.assert_frame_equal(result21, expected21)