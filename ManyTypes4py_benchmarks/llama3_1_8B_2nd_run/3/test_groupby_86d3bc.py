import numpy as np
import pytest
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, NamedAgg, Series, Timestamp, date_range, to_datetime
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby

@pytest.fixture
def times_frame() -> DataFrame:
    """Frame for testing times argument in EWM groupby."""
    return DataFrame({'A': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'], 'B': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3], 'C': to_datetime(['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-02', '2020-01-10', '2020-01-22', '2020-01-03', '2020-01-23', '2020-01-23', '2020-01-04'])})

@pytest.fixture
def roll_frame() -> DataFrame:
    return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40)})

class TestRolling:

    def test_groupby_unsupported_argument(self, roll_frame: DataFrame) -> None:
        msg = "groupby\\(\\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            roll_frame.groupby('A', foo=1)

    def test_getitem(self, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A')
        g_mutated = get_groupby(roll_frame, by='A')
        expected = g_mutated.B.apply(lambda x: x.rolling(2).mean())
        result = g.rolling(2).mean().B
        tm.assert_series_equal(result, expected)
        result = g.rolling(2).B.mean()
        tm.assert_series_equal(result, expected)
        result = g.B.rolling(2).mean()
        tm.assert_series_equal(result, expected)
        result = roll_frame.B.groupby(roll_frame.A).rolling(2).mean()
        tm.assert_series_equal(result, expected)

    def test_getitem_multiple(self, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A')
        r = g.rolling(2, min_periods=0)
        g_mutated = get_groupby(roll_frame, by='A')
        expected = g_mutated.B.apply(lambda x: x.rolling(2, min_periods=0).count())
        result = r.B.count()
        tm.assert_series_equal(result, expected)
        result = r.B.count()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('f', ['sum', 'mean', 'min', 'max', 'first', 'last', 'count', 'kurt', 'skew'])
    def test_rolling(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A', group_keys=False)
        r = g.rolling(window=4)
        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.rolling(4), f)())
        expected_index = MultiIndex.from_arrays([roll_frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['std', 'var'])
    def test_rolling_ddof(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A', group_keys=False)
        r = g.rolling(window=4)
        result = getattr(r, f)(ddof=1)
        expected = g.apply(lambda x: getattr(x.rolling(4), f)(ddof=1))
        expected_index = MultiIndex.from_arrays([roll_frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'midpoint', 'nearest'])
    def test_rolling_quantile(self, interpolation: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A', group_keys=False)
        r = g.rolling(window=4)
        result = r.quantile(0.4, interpolation=interpolation)
        expected = g.apply(lambda x: x.rolling(4).quantile(0.4, interpolation=interpolation))
        expected_index = MultiIndex.from_arrays([roll_frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f, expected_val', [['corr', 1], ['cov', 0.5]])
    def test_rolling_corr_cov_other_same_size_as_groups(self, f: str, expected_val: float, df: DataFrame, other: DataFrame) -> None:
        result = getattr(df.groupby(level=0).rolling(2), f)(other)
        expected_data = ([np.nan] + [expected_val] * 4) * 2
        expected = DataFrame(expected_data, columns=['value'], index=MultiIndex.from_arrays([[1] * 5 + [2] * 5, [1] * 5 + [2] * 5, list(range(1, 6)) * 2], names=['idx1', 'idx1', 'idx2']))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['corr', 'cov'])
    def test_rolling_corr_cov_other_diff_size_as_groups(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A')
        r = g.rolling(window=4)
        result = getattr(r, f)(roll_frame)

        def func(x):
            return getattr(x.rolling(4), f)(roll_frame)
        expected = g.apply(func)
        expected['A'] = np.nan
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['corr', 'cov'])
    def test_rolling_corr_cov_pairwise(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A')
        r = g.rolling(window=4)
        result = getattr(r.B, f)(pairwise=True)

        def func(x):
            return getattr(x.B.rolling(4), f)(pairwise=True)
        expected = g.apply(func)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('func, expected_values', [('cov', [[1.0, 1.0], [1.0, 4.0]]), ('corr', [[1.0, 0.5], [0.5, 1.0]])])
    def test_rolling_corr_cov_unordered(self, func: str, expected_values: list, df: DataFrame) -> None:
        result = getattr(rol, func)()
        expected = DataFrame({'b': 4 * [np.nan] + expected_values[0] + 2 * [np.nan], 'c': 4 * [np.nan] + expected_values[1] + 2 * [np.nan]}, index=MultiIndex.from_tuples([('g1', 0, 'b'), ('g1', 0, 'c'), ('g1', 2, 'b'), ('g1', 2, 'c'), ('g1', 3, 'b'), ('g1', 3, 'c'), ('g2', 1, 'b'), ('g2', 1, 'c')], names=['a', None, None]))
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply(self, raw: bool, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A', group_keys=False)
        r = g.rolling(window=4)
        result = r.apply(lambda x: x.sum(), raw=raw)
        expected = g.apply(lambda x: x.rolling(4).apply(lambda y: y.sum(), raw=raw))
        expected_index = MultiIndex.from_arrays([roll_frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply_mutability(self, df: DataFrame) -> None:
        g = df.groupby('A')
        mi = MultiIndex.from_tuples([('bar', 3), ('bar', 4), ('bar', 5), ('foo', 0), ('foo', 1), ('foo', 2)])
        mi.names = ['A', None]
        expected = DataFrame([np.nan, 2.0, 2.0] * 2, columns=['B'], index=mi)
        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)
        g.sum()
        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('expected_value,raw_value', [[1.0, True], [0.0, False]])
    def test_groupby_rolling(self, expected_value: float, raw_value: bool) -> None:
        df = DataFrame({'id': [1, 1, 1], 'value': [1, 2, 3]})
        result = df.groupby('id').value.rolling(1).apply(isnumpyarray, raw=raw_value)
        expected = Series([expected_value] * 3, index=MultiIndex.from_tuples(((1, 0), (1, 1), (1, 2)), names=['id', None]), name='value')
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_center_center(self, series: Series) -> None:
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series([np.nan] * 5, index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))))
        tm.assert_series_equal(result, expected)
        series = Series(range(1, 5))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series([np.nan] * 4, index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3))))
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': ['a'] * 5 + ['b'] * 6, 'b': range(11)})
        result = df.groupby('a').rolling(center=True, window=3).mean()
        expected = DataFrame([np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan], index=MultiIndex.from_tuples((('a', 0), ('a', 1), ('a', 2), ('a', 3), ('a', 4), ('b', 5), ('b', 6), ('b', 7), ('b', 8), ('b', 9), ('b', 10)), names=['a', None]), columns=['b'])
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'a': ['a'] * 5 + ['b'] * 5, 'b': range(10)})
        result = df.groupby('a').rolling(center=True, window=3).mean()
        expected = DataFrame([np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, np.nan], index=MultiIndex.from_tuples((('a', 0), ('a', 1), ('a', 2), ('a', 3), ('a', 4), ('b', 5), ('b', 6), ('b', 7), ('b', 8), ('b', 9)), names=['a', None]), columns=['b'])
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_center_on(self, df: DataFrame) -> None:
        result = df.groupby('gb').rolling(6, on='Date', center=True, min_periods=1).value.mean()
        mi = MultiIndex.from_arrays([df['gb'], df['Date']], names=['gb', 'Date'])
        expected = Series([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5], name='value', index=mi)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('min_periods', [5, 4, 3])
    def test_groupby_rolling_center_min_periods(self, min_periods: int, df: DataFrame) -> None:
        result = df.groupby('group').rolling(window=5, center=True, min_periods=min_periods).mean()
        result = result.reset_index()[['group', 'data']]
        grp_A_mean = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
        grp_B_mean = [x + 10.0 for x in grp_A_mean]
        num_nans = max(0, min_periods - 3)
        nans = [np.nan] * num_nans
        grp_A_expected = nans + grp_A_mean[num_nans:10 - num_nans] + nans
        grp_B_expected = nans + grp_B_mean[num_nans:10 - num_nans] + nans
        expected = DataFrame({'group': ['A'] * 10 + ['B'] * 10, 'data': grp_A_expected + grp_B_expected})
        tm.assert_frame_equal(result, expected)

    def test_groupby_subselect_rolling(self, df: DataFrame) -> None:
        result = df.groupby('a')[['b']].rolling(2).max()
        expected = DataFrame([np.nan, np.nan, 2.0, np.nan], columns=['b'], index=MultiIndex.from_tuples(((1, 0), (2, 1), (2, 3), (3, 2)), names=['a', None]))
        tm.assert_frame_equal(result, expected)
        result = df.groupby('a')['b'].rolling(2).max()
        expected = Series([np.nan, np.nan, 2.0, np.nan], index=MultiIndex.from_tuples(((1, 0), (2, 1), (2, 3), (3, 2)), names=['a', None]), name='b')
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_custom_indexer(self, df: DataFrame) -> None:
        result = df.groupby(df.index).rolling(SimpleIndexer(window_size=3), min_periods=1).sum()
        expected = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_subset_with_closed(self, df: DataFrame) -> None:
        result = df.groupby('group').rolling('1D', on='date', closed='left')['column1'].sum()
        expected = Series([np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0], index=MultiIndex.from_frame(df[['group', 'date']], names=['group', 'date']), name='column1')
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_agg_namedagg(self, df: DataFrame) -> None:
        result = df.groupby('kind').rolling(2).agg(total_weight=NamedAgg(column='weight', aggfunc=sum), min_height=NamedAgg(column='height', aggfunc=min))
        expected = DataFrame({'total_weight': [np.nan, 17.8, 19.9, np.nan, 205.5, 240.0], 'min_height': [np.nan, 9.1, 9.5, np.nan, 6.0, 8.0]}, index=MultiIndex([['cat', 'dog'], [0, 1, 2, 3, 4, 5]], [[0, 0, 0, 1, 1, 1], [0, 2, 4, 1, 3, 5]], names=['kind', None]))
        tm.assert_frame_equal(result, expected)

    def test_groupby_subset_rolling_subset_with_closed(self, df: DataFrame) -> None:
        result = df.groupby('group')[['column1', 'date']].rolling('1D', on='date', closed='left')['column1'].sum()
        expected = Series([np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0], index=MultiIndex.from_frame(df[['group', 'date']], names=['group', 'date']), name='column1')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('func', ['max', 'min'])
    def test_groupby_rolling_index_changed(self, func: str, ds: Series) -> None:
        result = getattr(ds.groupby(ds).rolling(2), func)()
        expected = Series([np.nan, np.nan, 2.0], index=MultiIndex.from_tuples([(1, 'a', 'x'), (2, 'a', 'y'), (2, 'c', 'z')], names=['a', '1', '2']), name='a')
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_empty_frame(self, expected: DataFrame) -> None:
        result = expected.groupby('s1').rolling(window=1).sum()
        expected = expected.drop(columns='s1')
        expected.index = MultiIndex.from_product([Index([], dtype='float64'), Index([], dtype='int64')], names=['s1', None])
        tm.assert_frame_equal(result, expected)
        expected = DataFrame({'s1': [], 's2': []})
        result = expected.groupby(['s1', 's2']).rolling(window=1).sum()
        expected = expected.drop(columns=['s1', 's2'])
        expected.index = MultiIndex.from_product([Index([], dtype='float64'), Index([], dtype='float64'), Index([], dtype='int64')], names=['s1', 's2', None])
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_string_index(self, df: DataFrame) -> None:
        groups = df.groupby('group')
        df['count_to_date'] = groups.cumcount()
        rolling_groups = groups.rolling('10D', on='eventTime')
        result = rolling_groups.apply(lambda df: df.shape[0])
        expected = DataFrame([['A', 'group_1', Timestamp(2019, 1, 1, 9), 1.0], ['B', 'group_1', Timestamp(2019, 1, 2, 9), 2.0], ['H', 'group_1', Timestamp(2019, 1, 6, 9), 3.0], ['Z', 'group_2', Timestamp(2019, 1, 3, 9), 1.0], ['E', 'group_2', Timestamp(2019, 1, 20, 9), 1.0]], columns=['index', 'group', 'eventTime', 'count_to_date']).set_index(['group', 'index'])
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_no_sort(self, df: DataFrame) -> None:
        result = DataFrame({'foo': [2, 1], 'bar': [2, 1]}).groupby('foo', sort=False).rolling(1).min()
        expected = DataFrame(np.array([[2.0, 2.0], [1.0, 1.0]]), columns=['foo', 'bar'], index=MultiIndex.from_tuples([(2, 0), (1, 1)], names=['foo', None]))
        expected = expected.drop(columns='foo')
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_count_closed_on(self, df: DataFrame, unit: str) -> None:
        result = df.groupby('group').rolling('3d', on='date', closed='left')['column1'].count()
        dti = DatetimeIndex(['2018-12-27', '2018-12-29', '2018-12-31', '2018-12-28', '2018-12-30', '2019-01-01'], dtype=f'M8[{unit}]')
        mi = MultiIndex.from_arrays([['A', 'A', 'A', 'B', 'B', 'B'], dti], names=['group', 'date'])
        expected = Series([np.nan, 1.0, 1.0, np.nan, 1.0, 1.0], name='column1', index=mi)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(('func', 'kwargs'), [('rolling', {'window': 2, 'min_periods': 1}), ('expanding', {})])
    def test_groupby_rolling_sem(self, func: str, kwargs: dict, df: DataFrame) -> None:
        result = getattr(df.groupby('a'), func)(**kwargs).sem()
        expected = DataFrame({'a': [np.nan] * 5, 'b': [np.nan, 0.70711, np.nan, 0.70711, 0.70711]}, index=MultiIndex.from_tuples([('a', 0), ('a', 1), ('b', 2), ('b', 3), ('b', 4)], names=['a', None]))
        expected = expected.drop(columns='a')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(('rollings', 'key'), [({'on': 'a'}, 'a'), ({'on': None}, 'index')])
    def test_groupby_rolling_nans_in_index(self, rollings: dict, key: str, df: DataFrame) -> None:
        with pytest.raises(ValueError, match=f'{key} values must not have NaT'):
            df.groupby('c').rolling('60min', **rollings)

    @pytest.mark.parametrize('group_keys', [True, False])
    def test_groupby_rolling_group_keys(self, group_keys: bool, s: Series) -> None:
        result = s.groupby(['idx1', 'idx2'], group_keys=group_keys).rolling(1).mean()
        expected = Series([1.0, 2.0, 3.0], index=MultiIndex.from_tuples([('val1', 'val1', 'val1', 'val1'), ('val1', 'val1', 'val1', 'val1'), ('val2', 'val2', 'val2', 'val2')], names=['idx1', 'idx2', 'idx1', 'idx2']))
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_index_level_and_column_label(self, df: DataFrame) -> None:
        result = df.groupby(['idx1', 'A']).rolling(1).mean()
        expected = DataFrame({'B': [0.0, 1.0, 2.0]}, index=MultiIndex.from_tuples([('val1', 1, 'val1', 'val1'), ('val1', 1, 'val1', 'val1'), ('val2', 2, 'val2', 'val2')], names=['idx1', 'A', 'idx1', 'idx2']))
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_resulting_multiindex(self, df: DataFrame) -> None:
        result = df.groupby('b').rolling(3).mean()
        expected_index = MultiIndex.from_tuples([(1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)], names=['b', None])
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex2(self, df: DataFrame) -> None:
        result = df.groupby(['b', 'c']).rolling(2).sum()
        expected_index = MultiIndex.from_tuples([(1, 1, 0), (1, 1, 4), (1, 1, 8), (1, 3, 2), (1, 3, 6), (1, 3, 10), (2, 2, 1), (2, 2, 5), (2, 2, 9), (2, 4, 3), (2, 4, 7), (2, 4, 11)], names=['b', 'c', None])
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex3(self, df: DataFrame) -> None:
        df = df.set_index('c', append=True)
        result = df.groupby('b').rolling(3).mean()
        expected_index = MultiIndex.from_tuples([(1, 0, 1), (1, 2, 3), (1, 4, 1), (1, 6, 3), (2, 1, 2), (2, 3, 4), (2, 5, 2), (2, 7, 4)], names=['b', None, 'c'])
        tm.assert_index_equal(result.index, expected_index, exact='equiv')

    def test_groupby_rolling_object_doesnt_affect_groupby_apply(self, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A', group_keys=False)
        expected = g.apply(lambda x: x.rolling(4).sum()).index
        _ = g.rolling(window=4)
        result = g.apply(lambda x: x.rolling(4).sum()).index
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(('window', 'min_periods', 'closed', 'expected'), [(2, 0, 'left', [None, 0.0, 1.0, 1.0, None, 0.0, 1.0, 1.0]), (2, 2, 'left', [None, None, 1.0, 1.0, None, None, 1.0, 1.0]), (4, 4, 'left', [None, None, None, None, None, None, None, None]), (4, 4, 'right', [None, None, None, 5.0, None, None, None, 5.0])])
    def test_groupby_rolling_var(self, window: int, min_periods: int, closed: str, expected: list, df: DataFrame) -> None:
        result = df.groupby([1, 2, 1, 2, 1, 2, 1, 2]).rolling(window=window, min_periods=min_periods, closed=closed).var(0)
        expected_result = DataFrame(np.array(expected, dtype='float64'), index=MultiIndex(levels=[np.array([1, 2]), [0, 1, 2, 3, 4, 5, 6, 7]], codes=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 2, 4, 6, 1, 3, 5, 7]]))
        tm.assert_frame_equal(result, expected_result)

    @pytest.mark.parametrize('columns', [MultiIndex.from_tuples([('A', ''), ('B', 'C')]), ['A', 'B']])
    def test_by_column_not_in_values(self, columns: MultiIndex, df: DataFrame) -> None:
        g = df.groupby('A')
        original_obj = g.obj.copy(deep=True)
        r = g.rolling(4)
        result = r.sum()
        assert 'A' not in result.columns
        tm.assert_frame_equal(g.obj, original_obj)

    def test_groupby_level(self, df: DataFrame) -> None:
        result = df.groupby(level=0)['Max Speed'].rolling(2).sum()
        expected = Series([np.nan, 740.0, np.nan, 50.0], index=MultiIndex.from_tuples([('Falcon', 'Falcon', 'Captive'), ('Falcon', 'Falcon', 'Wild'), ('Parrot', 'Parrot', 'Captive'), ('Parrot', 'Parrot', 'Wild')], names=['Animal', 'Animal', 'Type']), name='Max Speed')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('by, expected_data', [[['id'], {'num': [100.0, 150.0, 150.0, 200.0]}], [['id', 'index'], {'date': [Timestamp('2018-01-01'), Timestamp('2018-01-02'), Timestamp('2018-01-01'), Timestamp('2018-01-02')], 'num': [100.0, 200.0, 150.0, 250.0]}]])
    def test_as_index_false(self, by: list, expected_data: dict, unit: str) -> None:
        data = [['A', '2018-01-01', 100.0], ['A', '2018-01-02', 200.0], ['B', '2018-01-01', 150.0], ['B', '2018-01-02', 250.0]]
        df = DataFrame(data, columns=['id', 'date', 'num'])
        df['date'] = df['date'].astype(f'M8[{unit}]')
        df = df.set_index(['date'])
        gp_by = [getattr(df, attr) for attr in by]
        result = df.groupby(gp_by, as_index=False).rolling(window=2, min_periods=1).mean()
        expected = {'id': ['A', 'A', 'B', 'B']}
        expected.update(expected_data)
        expected = DataFrame(expected, index=df.index)
        if 'date' in expected_data:
            expected['date'] = expected['date'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(result, expected)

    def test_nan_and_zero_endpoints(self, any_int_numpy_dtype: str, typ: type, arr: np.ndarray, size: int, val: float, df: DataFrame) -> None:
        result = df.groupby('index')['adl2'].rolling(window=10, min_periods=1).mean()
        expected = Series(arr, name='adl2', index=MultiIndex.from_arrays([Index([0] * 999 + [1], dtype=typ, name='index'), Index([0] * 999 + [1], dtype=typ, name='index')]))
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_non_monotonic(self, df: DataFrame) -> None:
        with pytest.raises(ValueError, match='.* must be monotonic'):
            df.groupby('c').rolling('365D', on='B')

    def test_groupby_monotonic(self, df: DataFrame) -> None:
        expected = df.set_index('date').groupby('name').apply(lambda x: x.rolling('180D')['amount'].sum())
        result = df.groupby('name').rolling('180D', on='date')['amount'].sum()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_monotonic_within_each_group(self, df: DataFrame) -> None:
        expected = df.set_index('B').groupby('A').apply(lambda x: x.rolling('4s')['C'].mean())
        result = df.groupby('A').rolling('4s', on='B').C.mean()
        tm.assert_series_equal(result, expected)

    def test_datelike_on_not_monotonic_within_each_group(self, df: DataFrame) -> None:
        with pytest.raises(ValueError, match='Each group within B must be monotonic.'):
            df.groupby('A').rolling('365D', on='B')

class TestExpanding:

    @pytest.fixture
    def frame(self) -> DataFrame:
        return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40)})

    @pytest.mark.parametrize('f', ['sum', 'mean', 'min', 'max', 'first', 'last', 'count', 'kurt', 'skew'])
    def test_expanding(self, f: str, frame: DataFrame) -> None:
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.expanding(), f)())
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['std', 'var'])
    def test_expanding_ddof(self, f: str, frame: DataFrame) -> None:
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = getattr(r, f)(ddof=0)
        expected = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'midpoint', 'nearest'])
    def test_expanding_quantile(self, interpolation: str, frame: DataFrame) -> None:
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = r.quantile(0.4, interpolation=interpolation)
        expected = g.apply(lambda x: x.expanding().quantile(0.4, interpolation=interpolation))
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['corr', 'cov'])
    def test_expanding_corr_cov(self, f: str, frame: DataFrame) -> None:
        g = frame.groupby('A')
        r = g.expanding()
        result = getattr(r, f)(frame)

        def func_0(x):
            return getattr(x.expanding(), f)(frame)
        expected = g.apply(func_0)
        null_idx = list(range(20, 61)) + list(range(72, 113))
        expected.iloc[null_idx, 1] = np.nan
        expected['A'] = np.nan
        tm.assert_frame_equal(result, expected)
        result = getattr(r.B, f)(pairwise=True)

        def func_1(x):
            return getattr(x.B.expanding(), f)(pairwise=True)
        expected = g.apply(func_1)
        tm.assert_series_equal(result, expected)

    def test_expanding_apply(self, raw: bool, frame: DataFrame) -> None:
        g = frame.groupby('A', group_keys=False)
        r = g.expanding()
        result = r.apply(lambda x: x.sum(), raw=raw)
        expected = g.apply(lambda x: x.expanding().apply(lambda y: y.sum(), raw=raw))
        expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_groupby_expanding_agg_namedagg(self, df: DataFrame) -> None:
        result = df.groupby('kind').expanding(1).agg(total_weight=NamedAgg(column='weight', aggfunc=sum), min_height=NamedAgg(column='height', aggfunc=min))
        expected = DataFrame({'total_weight': [7.9, 17.8, 27.8, 7.5, 205.5, 247.5], 'min_height': [9.1, 9.1, 9.1, 6.0, 6.0, 6.0]}, index=MultiIndex([['cat', 'dog'], [0, 1, 2, 3, 4, 5]], [[0, 0, 0, 1, 1, 1], [0, 2, 4, 1, 3, 5]], names=['kind', None]))
        tm.assert_frame_equal(result, expected)

class TestEWM:

    @pytest.mark.parametrize('method, expected_data', [['mean', [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666]], ['std', [np.nan, 0.707107, 0.963624, 1.177164]], ['var', [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857]]])
    def test_methods(self, method: str, expected_data: list, df: DataFrame) -> None:
        result = getattr(df.groupby('A').ewm(com=1.0), method)()
        expected = DataFrame({'B': expected_data}, index=MultiIndex.from_tuples([('a', 0), ('a', 1), ('a', 2), ('a', 3)], names=['A', None]))
        tm.assert_frame_equal(result, expected)

    def test_groupby_ewm_agg_namedagg(self, df: DataFrame) -> None:
        result = df.groupby('A').ewm(com=1.0).agg(B_mean=NamedAgg(column='B', aggfunc='mean'), B_std=NamedAgg(column='B', aggfunc='std'), B_var=NamedAgg(column='B', aggfunc='var'))
        expected = DataFrame({'B_mean': [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666], 'B_std': [np.nan, 0.707107, 0.963624, 1.177164], 'B_var': [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857]}, index=MultiIndex.from_tuples([('a', 0), ('a', 1), ('a', 2), ('a', 3)], names=['A', None]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('method, expected_data', [['corr', [np.nan, 1.0, 1.0, 1]], ['cov', [np.nan, 0.5, 0.928571, 1.385714]]])
    def test_pairwise_methods(self, method: str, expected_data: list, df: DataFrame) -> None:
        result = getattr(df.groupby('A').ewm(com=1.0), method)()
        expected = DataFrame({'B': expected_data}, index=MultiIndex.from_tuples([('a', 0, 'B'), ('a', 1, 'B'), ('a', 2, 'B'), ('a', 3, 'B')], names=['A', None, None]))
        tm.assert_frame_equal(result, expected)
        expected = df.groupby('A')[['B']].apply(lambda x: getattr(x.ewm(com=1.0), method)())
        tm.assert_frame_equal(result, expected)

    def test_times(self, times_frame: DataFrame) -> None:
        halflife = '23 days'
        times = times_frame.pop('C')
        result = times_frame.groupby('A').ewm(halflife=halflife, times=times).mean()
        expected = DataFrame({'B': [0.0, 0.507534, 1.020088, 1.537661, 0.0, 0.567395, 1.221209, 0.0, 0.653141, 1.195003]}, index=MultiIndex.from_tuples([('a', 0), ('a', 3), ('a', 6), ('a', 9), ('b', 1), ('b', 4), ('b', 7), ('c', 2), ('c', 5), ('c', 8)], names=['A', None]))
        tm.assert_frame_equal(result, expected)

    def test_times_array(self, times_frame: DataFrame) -> None:
        halflife = '23 days'
        times = times_frame.pop('C')
        gb = times_frame.groupby('A')
        result = gb.ewm(halflife=halflife, times=times).mean()
        expected = gb.ewm(halflife=halflife, times=times.values).mean()
        tm.assert_frame_equal(result, expected)

    def test_dont_mutate_obj_after_slicing(self, df: DataFrame) -> None:
        grp = df.groupby('id').rolling('1h', on='timestamp')
        result = grp.count()
        expected_df = DataFrame({'timestamp': date_range('2021-9-1', periods=5, freq='h'), 'y': [1.0] * 5}, index=MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'b'], list(range(5))], names=['id', None]))
        tm.assert_frame_equal(result, expected_df)
        result = grp['y'].count()
        expected_series = Series([1.0] * 5, index=MultiIndex.from_arrays([['a', 'a', 'b', 'b', 'b'], date_range('2021-9-1', periods=5, freq='h')], names=['id', 'timestamp']), name='y')
        tm.assert_series_equal(result, expected_series)
        result = grp.count()
        tm.assert_frame_equal(result, expected_df)

def test_rolling_corr_with_single_integer_in_index() -> None:
    df = DataFrame({'a': [(1,), (1,), (1,)], 'b': [4, 5, 6]})
    gb = df.groupby(['a'])
    result = gb.rolling(2).corr(other=df)
    index = MultiIndex.from_tuples([((1,), 0), ((1,), 1), ((1,), 2)], names=['a', None])
    expected = DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [np.nan, 1.0, 1.0]}, index=index)
    tm.assert_frame_equal(result, expected)

def test_rolling_corr_with_tuples_in_index() -> None:
    df = DataFrame({'a': [(1, 2), (1, 2), (1, 2)], 'b': [4, 5, 6]})
    gb = df.groupby(['a'])
    result = gb.rolling(2).corr(other=df)
    index = MultiIndex.from_tuples([((1, 2), 0), ((1, 2), 1), ((1, 2), 2)], names=['a', None])
    expected = DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [np.nan, 1.0, 1.0]}, index=index)
    tm.assert_frame_equal(result, expected)
