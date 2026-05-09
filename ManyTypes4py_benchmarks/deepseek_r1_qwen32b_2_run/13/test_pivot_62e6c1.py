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
    def test_pivot_with_interval_index(self, left_right: tuple[list[int], list[int]], dropna: bool, closed: str) -> None:
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
    def test_pivot_preserve_dtypes(self, columns: str, values: list[str]) -> None:
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
        expected = DataFrame([[nan, nan, 17, nan], [10, nan, nan, nan], [nan, 15, nan, nan], [nan, nan, nan, 20]], index=Index([nan, 'R1', 'R2', 'R4'], name='a'), columns=Index(['C1', 'C2', 'C3', 'C4'], name='b'))
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(df.pivot(index='b', columns='a', values='c'), expected.T)

    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_index_with_nan_dates(self, method: bool) -> None:
        df = DataFrame({'a': date_range('2014-02-01', periods=6, freq='D'), 'c': 100 + np.arange(6)})
        df['b'] = df['a'] - pd.Timestamp('2014-02-02')
        df.loc[1, 'a'] = df.loc[3, 'a'] = np.nan
        df.loc[1, 'b'] = df.loc[4, 'b'] = np.nan
        if method:
            pv = df.pivot(index='a', columns='b', values='c')
        else:
            pv = pd.pivot(df, index='a', columns='b', values='c')
        assert pv.notna().values.sum() == len(df)
        for _, row in df.iterrows():
            assert pv.loc[row['a'], row['b']] == row['c']
        if method:
            result = df.pivot(index='b', columns='a', values='c')
        else:
            result = pd.pivot(df, index='b', columns='a', values='c')
        tm.assert_frame_equal(result, pv.T)

    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_with_tz(self, method: bool, unit: str) -> None:
        df = DataFrame({'dt1': pd.DatetimeIndex([datetime(2013, 1, 1, 9, 0), datetime(2013, 1, 2, 9, 0), datetime(2013, 1, 1, 9, 0), datetime(2013, 1, 2, 9, 0)], dtype=f'M8[{unit}, US/Pacific]'), 'dt2': pd.DatetimeIndex([datetime(2014, 1, 1, 9, 0), datetime(2014, 1, 1, 9, 0), datetime(2014, 1, 2, 9, 0), datetime(2014, 1, 2, 9, 0)], dtype=f'M8[{unit}, Asia/Tokyo]'), 'data1': np.arange(4, dtype='int64'), 'data2': np.arange(4, dtype='int64')})
        exp_col1 = Index(['data1', 'data1', 'data2', 'data2'])
        exp_col2 = pd.DatetimeIndex(['2014/01/01 09:00', '2014/01/02 09:00'] * 2, name='dt2', dtype=f'M8[{unit}, Asia/Tokyo]')
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        exp_idx = pd.DatetimeIndex(['2013/01/01 09:00', '2013/01/02 09:00'], name='dt1', dtype=f'M8[{unit}, US/Pacific]')
        expected = DataFrame([[0, 2, 0, 2], [1, 3, 1, 3]], index=exp_idx, columns=exp_col)
        if method:
            pv = df.pivot(index='dt1', columns='dt2')
        else:
            pv = pd.pivot(df, index='dt1', columns='dt2')
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame([[0, 2], [1, 3]], index=exp_idx, columns=exp_col2[:2])
        if method:
            pv = df.pivot(index='dt1', columns='dt2', values='data1')
        else:
            pv = pd.pivot(df, index='dt1', columns='dt2', values='data1')
        tm.assert_frame_equal(pv, expected)

    def test_pivot_tz_in_values(self) -> None:
        df = DataFrame([{'uid': 'aa', 'ts': pd.Timestamp('2016-08-12 13:00:00-0700', tz='US/Pacific')}, {'uid': 'aa', 'ts': pd.Timestamp('2016-08-12 08:00:00-0700', tz='US/Pacific')}, {'uid': 'aa', 'ts': pd.Timestamp('2016-08-12 14:00:00-0700', tz='US/Pacific')}, {'uid': 'aa', 'ts': pd.Timestamp('2016-08-25 11:00:00-0700', tz='US/Pacific')}, {'uid': 'aa', 'ts': pd.Timestamp('2016-08-25 13:00:00-0700', tz='US/Pacific')}])
        df = df.set_index('ts').reset_index()
        mins = df.ts.map(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
        result = pivot_table(df.set_index('ts').reset_index(), values='ts', index=['uid'], columns=[mins], aggfunc='min')
        expected = DataFrame([[pd.Timestamp('2016-08-12 08:00:00-0700', tz='US/Pacific'), pd.Timestamp('2016-08-25 11:00:00-0700', tz='US/Pacific')]], index=Index(['aa'], name='uid'), columns=pd.DatetimeIndex([pd.Timestamp('2016-08-12 00:00:00', tz='US/Pacific'), pd.Timestamp('2016-08-25 00:00:00', tz='US/Pacific')], name='ts'))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_periods(self, method: bool) -> None:
        df = DataFrame({'p1': [pd.Period('2013-01-01', 'D'), pd.Period('2013-01-02', 'D'), pd.Period('2013-01-01', 'D'), pd.Period('2013-01-02', 'D')], 'p2': [pd.Period('2013-01', 'M'), pd.Period('2013-01', 'M'), pd.Period('2013-02', 'M'), pd.Period('2013-02', 'M')], 'data1': np.arange(4, dtype='int64'), 'data2': np.arange(4, dtype='int64')})
        exp_col1 = Index(['data1', 'data1', 'data2', 'data2'])
        exp_col2 = pd.PeriodIndex(['2013-01', '2013-02'] * 2, name='p2', freq='M')
        exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
        expected = DataFrame([[0, 2, 0, 2], [1, 3, 1, 3]], index=pd.PeriodIndex(['2013-01-01', '2013-01-02'], name='p1', freq='D'), columns=exp_col)
        if method:
            pv = df.pivot(index='p1', columns='p2')
        else:
            pv = pd.pivot(df, index='p1', columns='p2')
        tm.assert_frame_equal(pv, expected)
        expected = DataFrame([[0, 2], [1, 3]], index=pd.PeriodIndex(['2013-01-01', '2013-01-02'], name='p1', freq='D'), columns=pd.PeriodIndex(['2013-01', '2013-02'], name='p2', freq='M'))
        if method:
            pv = df.pivot(index='p1', columns='p2', values='data1')
        else:
            pv = pd.pivot(df, index='p1', columns='p2', values='data1')
        tm.assert_frame_equal(pv, expected)

    def test_pivot_periods_with_margins(self) -> None:
        df = DataFrame({'a': [1, 1, 2, 2], 'b': [pd.Period('2019Q1'), pd.Period('2019Q2'), pd.Period('2019Q1'), pd.Period('2019Q2')], 'x': 1.0})
        expected = DataFrame(data=1.0, index=Index([1, 2, 'All'], name='a'), columns=Index([pd.Period('2019Q1'), pd.Period('2019Q2'), 'All'], name='b'))
        result = df.pivot_table(index='a', columns='b', values='x', margins=True)
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize('box', [list, np.array, Series, Index])
    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_with_list_like_values(self, box: type, method: bool) -> None:
        values = box(['baz', 'zoo'])
        df = DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        if method:
            result = df.pivot(index='foo', columns='bar', values=values)
        else:
            result = pd.pivot(df, index='foo', columns='bar', values=values)
        data = [[1, 2, 3, 'x', 'y', 'z'], [4, 5, 6, 'q', 'w', 't']]
        index = Index(data=['one', 'two'], name='foo')
        columns = MultiIndex(levels=[['baz', 'zoo'], ['A', 'B', 'C']], codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], names=[None, 'bar'])
        expected = DataFrame(data=data, index=index, columns=columns)
        expected['baz'] = expected['baz'].astype(object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('values', [['bar', 'baz'], np.array(['bar', 'baz']), Series(['bar', 'baz']), Index(['bar', 'baz'])])
    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_with_list_like_values_nans(self, values: list[str], method: bool) -> None:
        df = DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        if method:
            result = df.pivot(index='zoo', columns='foo', values=values)
        else:
            result = pd.pivot(df, index='zoo', columns='foo', values=values)
        data = [[np.nan, 'A', np.nan, 4], [np.nan, 'C', np.nan, 6], [np.nan, 'B', np.nan, 5], ['A', np.nan, 1, np.nan], ['B', np.nan, 2, np.nan], ['C', np.nan, 3, np.nan]]
        index = Index(data=['q', 't', 'w', 'x', 'y', 'z'], name='zoo')
        columns = MultiIndex(levels=[['bar', 'baz'], ['one', 'two']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]], names=[None, 'foo'])
        expected = DataFrame(data=data, index=index, columns=columns)
        expected['baz'] = expected['baz'].astype(object)
        tm.assert_frame_equal(result, expected)

    def test_pivot_columns_none_raise_error(self) -> None:
        df = DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3], 'col3': [1, 2, 3]})
        msg = "pivot\\(\\) missing 1 required keyword-only argument: 'columns'"
        with pytest.raises(TypeError, match=msg):
            df.pivot(index='col1', values='col3')

    @pytest.mark.xfail(reason='MultiIndexed unstack with tuple names fails with KeyError GH#19966')
    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_with_multiindex(self, method: bool) -> None:
        index = Index(data=[0, 1, 2, 3, 4, 5])
        data = [['one', 'A', 1, 'x'], ['one', 'B', 2, 'y'], ['one', 'C', 3, 'z'], ['two', 'A', 4, 'q'], ['two', 'B', 5, 'w'], ['two', 'C', 6, 't']]
        columns = MultiIndex(levels=[['bar', 'baz'], ['first', 'second']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        df = DataFrame(data=data, index=index, columns=columns, dtype='object')
        if method:
            result = df.pivot(index=('bar', 'first'), columns=('bar', 'second'), values=('baz', 'first'))
        else:
            result = pd.pivot(df, index=('bar', 'first'), columns=('bar', 'second'), values=('baz', 'first'))
        data = {'A': Series([1, 4], index=['one', 'two']), 'B': Series([2, 5], index=['one', 'two']), 'C': Series([3, 6], index=['one', 'two'])}
        expected = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('method', [True, False])
    def test_pivot_with_tuple_of_values(self, method: bool) -> None:
        df = DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'], 'bar': ['A', 'B', 'C', 'A', 'B', 'C'], 'baz': [1, 2, 3, 4, 5, 6], 'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
        with pytest.raises(KeyError, match="^\\('bar', 'baz'\\)$"):
            if method:
                df.pivot(index='zoo', columns='foo', values=('bar', 'baz'))
            else:
                pd.pivot(df, index='zoo', columns='foo', values=('bar', 'baz'))

    def _check_output(self, result: DataFrame, values_col: str, data: DataFrame, index: list[str] = None, columns: list[str] = None, margins_col: str = 'All') -> None:
        if index is None:
            index = ['A', 'B']
        if columns is None:
            columns = ['C']
        col_margins = result.loc[result.index[:-1], margins_col]
        expected_col_margins = data.groupby(index)[values_col].mean()
        tm.assert_series_equal(col_margins, expected_col_margins, check_names=False)
        assert col_margins.name == margins_col
        result = result.sort_index()
        index_margins = result.loc[margins_col, ''].iloc[:-1]
        expected_ix_margins = data.groupby(columns)[values_col].mean()
        tm.assert_series_equal(index_margins, expected_ix_margins, check_names=False)
        assert index_margins.name == (margins_col, '')
        grand_total_margins = result.loc[(margins_col, ''), margins_col]
        expected_total_margins = data[values_col].mean()
        assert grand_total_margins == expected_total_margins

    def test_margins(self, data: DataFrame) -> None:
        result = data.pivot_table(values='D', index=['A', 'B'], columns='C', margins=True, aggfunc='mean')
        self._check_output(result, 'D', data)
        result = data.pivot_table(values='D', index=['A', 'B'], columns='C', margins=True, aggfunc='mean', margins_name='Totals')
        self._check_output(result, 'D', data, margins_col='Totals')
        table = data.pivot_table(index=['A', 'B'], columns='C', margins=True, aggfunc='mean')
        for value_col in table.columns.levels[0]:
            self._check_output(table[value_col], value_col, data)

    def test_no_col(self, data: DataFrame, using_infer_string: bool) -> None:
        data.columns = [k * 2 for k in data.columns]
        msg = re.escape('agg function failed [how->mean,dtype->')
        if using_infer_string:
            msg = "dtype 'str' does not support operation 'mean'"
        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
        table = data.drop(columns='CC').pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
        for value_col in table.columns:
            totals = table.loc[('All', ''), value_col]
            assert totals == data[value_col].mean()
        with pytest.raises(TypeError, match=msg):
            data.pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
        table = data.drop(columns='CC').pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
        for item in ['DD', 'EE', 'FF']:
            totals = table.loc[('All', ''), item]
            assert totals == data[item].mean()

    @pytest.mark.parametrize('columns, aggfunc, values, expected_columns', [('A', 'mean', [[5.5, 5.5, 2.2, 2.2], [8.0, 8.0, 4.4, 4.4]], Index(['bar', 'All', 'foo', 'All'], name='A')), (['A', 'B'], 'sum', [[9, 13, 22, 5, 6, 11], [14, 18, 32, 11, 11, 22]], MultiIndex.from_tuples([('bar', 'one'), ('bar', 'two'), ('bar', 'All'), ('foo', 'one'), ('foo', 'two'), ('foo', 'All')], names=['A', 'B']))])
    def test_margin_with_only_columns_defined(self, columns: list[str], aggfunc: str, values: list[list[float]], expected_columns: Index, using_infer_string: bool) -> None:
        df = DataFrame({'A': ['foo', 'foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar'], 'B': ['one', 'one', 'one', 'two', 'two', 'one', 'one', 'two', 'two'], 'C': ['small', 'large', 'large', 'small', 'small', 'large', 'small', 'small', 'large'], 'D': [1, 2, 2, 3, 3, 4, 5, 6, 7], 'E': [2, 4, 5, 5, 6, 6, 8, 9, 9]})
        if aggfunc != 'sum':
            msg = re.escape('agg function failed [how->mean,dtype->')
            if using_infer_string:
                msg = "dtype 'str' does not support operation 'mean'"
            with pytest.raises(TypeError, match=msg):
                df.pivot_table(columns=columns, margins=True, aggfunc=aggfunc)
        if 'B' not in columns:
            df = df.drop(columns='B')
        result = df.drop(columns='C').pivot_table(columns=columns, margins=True, aggfunc=aggfunc)
        expected = DataFrame(values, index=Index(['D', 'E']), columns=expected_columns)
        tm.assert_frame_equal(result, expected)

    def test_margins_dtype(self, data: DataFrame) -> None:
        df = data.copy()
        df[['D', 'E