from datetime import date, datetime, timedelta
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
from pandas import Categorical, DataFrame, Grouper, Index, MultiIndex, Series, concat, date_range
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
from typing import Any, Dict, List, Optional, Tuple, Union

class TestPivotTable:

    @pytest.fixture
    def data(self) -> DataFrame:
        return DataFrame({'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'], 'B': ['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'], 'C': ['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny', 'shiny', 'dull', 'shiny', 'shiny', 'shiny'], 'D': np.random.default_rng(2).standard_normal(11), 'E': np.random.default_rng(2).standard_normal(11), 'F': np.random.default_rng(2).standard_normal(11)})

    def test_pivot_table(self, observed: bool, data: DataFrame) -> None:
        index = ['A', 'B']
        columns = 'C'
        table = pivot_table(data, values='D', index=index, columns=columns, observed=observed)
        table2 = data.pivot_table(values='D', index=index, columns=columns, observed=observed)
        tm.assert_frame_equal(table, table2)
        pivot_table(data, values='D', index=index, observed=observed)
        if len(index) > 1:
            assert table.index.names == tuple(index)
        else:
            assert table.index.name == index[0]
        if len(columns) > 1:
            assert table.columns.names == columns
        else:
            assert table.columns.name == columns[0]
        expected = data.groupby(index + [columns])['D'].agg('mean').unstack()
        tm.assert_frame_equal(table, expected)

    def test_pivot_table_categorical_observed_equal(self, observed: bool) -> None:
        df = DataFrame({'col1': list('abcde'), 'col2': list('fghij'), 'col3': [1, 2, 3, 4, 5]})
        expected = df.pivot_table(index='col1', values='col3', columns='col2', aggfunc='sum', fill_value=0)
        expected.index = expected.index.astype('category')
        expected.columns = expected.columns.astype('category')
        df.col1 = df.col1.astype('category')
        df.col2 = df.col2.astype('category')
        result = df.pivot_table(index='col1', values='col3', columns='col2', aggfunc='sum', fill_value=0, observed=observed)
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_nocols(self) -> None:
        df = DataFrame({'rows': ['a', 'b', 'c'], 'cols': ['x', 'y', 'z'], 'values': [1, 2, 3]})
        rs = df.pivot_table(columns='cols', aggfunc='sum')
        xp = df.pivot_table(index='cols', aggfunc='sum').T
        tm.assert_frame_equal(rs, xp)
        rs = df.pivot_table(columns='cols', aggfunc={'values': 'mean'})
        xp = df.pivot_table(index='cols', aggfunc={'values': 'mean'}).T
        tm.assert_frame_equal(rs, xp)

    def test_pivot_table_dropna(self) -> None:
        df = DataFrame({'amount': {0: 60000, 1: 100000, 2: 50000, 3: 30000}, 'customer': {0: 'A', 1: 'A', 2: 'B', 3: 'C'}, 'month': {0: 201307, 1: 201309, 2: 201308, 3: 201310}, 'product': {0: 'a', 1: 'b', 2: 'c', 3: 'd'}, 'quantity': {0: 2000000, 1: 500000, 2: 1000000, 3: 1000000}})
        pv_col = df.pivot_table('quantity', 'month', ['customer', 'product'], dropna=False)
        pv_ind = df.pivot_table('quantity', ['customer', 'product'], 'month', dropna=False)
        m = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('A', 'c'), ('A', 'd'), ('B', 'a'), ('B', 'b'), ('B', 'c'), ('B', 'd'), ('C', 'a'), ('C', 'b'), ('C', 'c'), ('C', 'd')], names=['customer', 'product'])
        tm.assert_index_equal(pv_col.columns, m)
        tm.assert_index_equal(pv_ind.index, m)

    def test_pivot_table_categorical(self) -> None:
        cat1 = Categorical(['a', 'a', 'b', 'b'], categories=['a', 'b', 'z'], ordered=True)
        cat2 = Categorical(['c', 'd', 'c', 'd'], categories=['c', 'd', 'y'], ordered=True)
        df = DataFrame({'A': cat1, 'B': cat2, 'values': [1, 2, 3, 4]})
        result = pivot_table(df, values='values', index=['A', 'B'], dropna=True, observed=False)
        exp_index = MultiIndex.from_arrays([cat1, cat2], names=['A', 'B'])
        expected = DataFrame({'values': [1.0, 2.0, 3.0, 4.0]}, index=exp_index)
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_dropna_categoricals(self, dropna: bool) -> None:
        categories = ['a', 'b', 'c', 'd']
        df = DataFrame({'A': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], 'B': [1, 2, 3, 1, 2, 3, 1, 2, 3], 'C': range(9)})
        df['A'] = df['A'].astype(CategoricalDtype(categories, ordered=False))
        result = df.pivot_table(index='B', columns='A', values='C', dropna=dropna, observed=False)
        expected_columns = Series(['a', 'b', 'c'], name='A')
        expected_columns = expected_columns.astype(CategoricalDtype(categories, ordered=False))
        expected_index = Series([1, 2, 3], name='B')
        expected = DataFrame([[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]], index=expected_index, columns=expected_columns)
        if not dropna:
            expected = expected.reindex(columns=Categorical(categories)).astype('float')
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_non_observable_dropna(self, dropna: bool) -> None:
        df = DataFrame({'A': Categorical([np.nan, 'low', 'high', 'low', 'high'], categories=['low', 'high'], ordered=True), 'B': [0.0, 1.0, 2.0, 3.0, 4.0]})
        result = df.pivot_table(index='A', values='B', dropna=dropna, observed=False)
        if dropna:
            values = [2.0, 3.0]
            codes = [0, 1]
        else:
            values = [2.0, 3.0, 0.0]
            codes = [0, 1, -1]
        expected = DataFrame({'B': values}, index=Index(Categorical.from_codes(codes, categories=['low', 'high'], ordered=True), name='A'))
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_non_observable_dropna_multi_cat(self, dropna: bool) -> None:
        df = DataFrame({'A': Categorical(['left', 'low', 'high', 'low', 'high'], categories=['low', 'high', 'left'], ordered=True), 'B': range(5)})
        result = df.pivot_table(index='A', values='B', dropna=dropna, observed=False)
        expected = DataFrame({'B': [2.0, 3.0, 0.0]}, index=Index(Categorical.from_codes([0, 1, 2], categories=['low', 'high', 'left'], ordered=True), name='A'))
        if not dropna:
            expected['B'] = expected['B'].astype(float)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('left_right', [([0] * 4, [1] * 4), (range(3), range(1, 4))])
    def test_pivot_with_interval_index(self, left_right: Tuple[List[int], List[int]], dropna: bool, closed: str) -> None:
        left, right = left_right
        interval_values = Categorical(pd.IntervalIndex.from_arrays(left, right, closed))
        df = DataFrame({'A': interval_values, 'B': 1})
        result = df.pivot_table(index='A', values='B', dropna=dropna, observed=False)
        expected = DataFrame({'B': 1.0}, index=Index(interval_values.unique(), name='A'))
        if not dropna:
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)

    def test_pivot_with_interval_index_margins(self) -> None:
        ordered_cat = pd.IntervalIndex.from_arrays([0, 0, 1, 1], [1, 1, 2, 2])
        df = DataFrame({'A': np.arange(4, 0, -1, dtype=np.intp), 'B': ['a', 'b', 'a', 'b'], 'C': Categorical(ordered_cat, ordered=True).sort_values(ascending=False)})
        pivot_tab = pivot_table(df, index='C', columns='B', values='A', aggfunc='sum', margins=True, observed=False)
        result = pivot_tab['All']
        expected = Series([3, 7, 10], index=Index([pd.Interval(0, 1), pd.Interval(1, 2), 'All'], name='C'), name='All', dtype=np.intp)
        tm.assert_series_equal(result, expected)

    def test_pass_array(self, data: DataFrame) -> None:
        result = data.pivot_table('D', index=data.A, columns=data.C)
        expected = data.pivot_table('D', index='A', columns='C')
        tm.assert_frame_equal(result, expected)

    def test_pass_function(self, data: DataFrame) -> None:
        result = data.pivot_table('D', index=lambda x: x // 5, columns=data.C)
        expected = data.pivot_table('D', index=data.index // 5, columns='C')
        tm.assert_frame_equal(result, expected)

    def test_pivot_table_multiple(self, data: DataFrame) -> None:
        index = ['A', 'B']
        columns = 'C'
        table = pivot_table(data, index=index, columns=columns)
        expected = data.groupby(index + [columns]).agg('mean').unstack()
        tm.assert_frame_equal(table, expected)

    def test_pivot_dtypes(self) -> None:
        f = DataFrame({'a': ['cat', 'bat', 'cat', 'bat'], 'v': [1, 2, 3, 4], 'i': ['a', 'b', 'a', 'b']})
        assert f.dtypes['v'] == 'int64'
        z = pivot_table(f, values='v', index=['a'], columns=['i'], fill_value=0, aggfunc='sum')
        result = z.dtypes
        expected = Series([np.dtype('int64')] * 2, index=Index(list('ab'), name='i'))
        tm.assert_series_equal(result, expected)
        f = DataFrame({'a': ['cat', 'bat', 'cat', 'bat'], 'v': [1.5, 2.5, 3.5, 4.5], 'i': ['a', 'b', 'a', 'b']})
        assert f.dtypes['v'] == 'float64'
        z = pivot_table(f, values='v', index=['a'], columns=['i'], fill_value=0, aggfunc='mean')
        result = z.dtypes
        expected = Series([np.dtype('float64')] * 2, index=Index(list('ab'), name='i'))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('columns,values', [('bool1', ['float1', 'float2']), ('bool1', ['float1', 'float2', 'bool1']), ('bool2', ['float1', 'float2', 'bool1'])])
    def test_pivot_preserve_dtypes(self, columns: str, values: List[str]) -> None:
        v = np.arange(5, dtype=np.float64)
        df = DataFrame({'float1': v, 'float2': v + 2.0, 'bool1': v <= 2, 'bool2': v <= 3})
        df_res = df.reset_index().pivot_table(index='index', columns=columns, values=values)
        result = dict(df_res.dtypes)
        expected = {col: np.dtype('float64') for col in df_res}
        assert result == expected

    def test_pivot_no_values(self) -> None:
        idx = pd.DatetimeIndex(['2011-01-01', '2011-02-01', '2011-01-02', '2011-01-01', '2011-01-02'])
        df = DataFrame({'A': [1, 2, 3, 4, 5]}, index=idx)
        res = df.pivot_table(index=df.index.month, columns=df.index.day)
        exp_columns = MultiIndex.from_tuples([('A', 1), ('A', 2)])
        exp_columns = exp_columns.set_levels(exp_columns.levels[1].astype(np.int32), level=1)
        exp = DataFrame([[2.5, 4.0], [2.0, np.nan]], index=Index([1, 2], dtype=np.int32), columns=exp_columns)
        tm.assert_frame_equal(res, exp)
        df = DataFrame({'A': [1, 2, 3, 4, 5], 'dt': date_range('2011-01-01', freq='D', periods=5)}, index=idx)
        res = df.pivot_table(index=df.index.month, columns=Grouper(key='dt', freq='ME'))
        exp_columns = MultiIndex.from_arrays([['A'], pd.DatetimeIndex(['2011-01-31'], dtype='M8[ns]')], names=[None, 'dt'])
        exp = DataFrame([3.25, 2.0], index=Index([1, 2], dtype=np.int32), columns=exp_columns)
        tm.assert_frame_equal(res, exp)
        res = df.pivot_table(index=Grouper(freq='YE'), columns=Grouper(key='dt', freq='ME'))
        exp = DataFrame([3.0], index=pd.DatetimeIndex(['2011-12-31'], freq='YE'), columns=exp_columns)
        tm.assert_frame_equal(res, exp)

    def test_pivot_multi_values(self, data: DataFrame) -> None:
        result = pivot_table(data, values=['D', 'E'], index='A', columns=['B', 'C'], fill_value=0)
        expected = pivot_table(data.drop(['F'], axis=1), index='A', columns=['B', 'C'], fill_value=0)
        tm.assert_frame_equal(result, expected)

    def test_pivot_multi_functions(self, data: DataFrame) -> None:
        f = lambda func: pivot_table(data, values=['D', 'E'], index=['A', 'B'], columns='C', aggfunc=func)
        result = f(['mean', 'std'])
        means = f('mean')
        stds = f('std')
        expected = concat([means, stds], keys=['mean', 'std'], axis=1)
        tm.assert_frame_equal(result, expected)
        f = lambda func: pivot_table(data, values=['D', 'E'], index=['A', 'B'], columns='C', aggfunc=func, margins=True)
        result = f(['mean', 'std'])
        means = f('mean')
        stds = f('std')
        expected = concat([means, stds], keys=['mean', 'std'], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_index_with_nan(self, method: bool) -> None:
        nan = np.nan
        df = DataFrame({'a': ['R1', 'R2', nan, 'R4'], 'b': ['C1', 'C2', 'C3', 'C4'], 'c': [10, 15, 17, 20]})
        if method:
            result = df.pivot(index='a', columns='b', values='c')
        else:
            result = pd.pivot(df, index='a', columns='b', values='c')
        expected = DataFrame([[nan