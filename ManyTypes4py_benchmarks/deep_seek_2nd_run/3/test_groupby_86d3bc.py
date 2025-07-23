import numpy as np
import pytest
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, NamedAgg, Series, Timestamp, date_range, to_datetime
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
from typing import Any, Dict, List, Optional, Tuple, Union

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
    def test_rolling_corr_cov_other_same_size_as_groups(self, f: str, expected_val: float) -> None:
        df = DataFrame({'value': range(10), 'idx1': [1] * 5 + [2] * 5, 'idx2': [1, 2, 3, 4, 5] * 2}).set_index(['idx1', 'idx2'])
        other = DataFrame({'value': range(5), 'idx2': [1, 2, 3, 4, 5]}).set_index('idx2')
        result = getattr(df.groupby(level=0).rolling(2), f)(other)
        expected_data = ([np.nan] + [expected_val] * 4) * 2
        expected = DataFrame(expected_data, columns=['value'], index=MultiIndex.from_arrays([[1] * 5 + [2] * 5, [1] * 5 + [2] * 5, list(range(1, 6)) * 2], names=['idx1', 'idx1', 'idx2']))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['corr', 'cov'])
    def test_rolling_corr_cov_other_diff_size_as_groups(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A')
        r = g.rolling(window=4)
        result = getattr(r, f)(roll_frame)

        def func(x: DataFrame) -> DataFrame:
            return getattr(x.rolling(4), f)(roll_frame)
        expected = g.apply(func)
        expected['A'] = np.nan
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('f', ['corr', 'cov'])
    def test_rolling_corr_cov_pairwise(self, f: str, roll_frame: DataFrame) -> None:
        g = roll_frame.groupby('A')
        r = g.rolling(window=4)
        result = getattr(r.B, f)(pairwise=True)

        def func(x: DataFrame) -> Series:
            return getattr(x.B.rolling(4), f)(pairwise=True)
        expected = g.apply(func)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('func, expected_values', [('cov', [[1.0, 1.0], [1.0, 4.0]]), ('corr', [[1.0, 0.5], [0.5, 1.0]])])
    def test_rolling_corr_cov_unordered(self, func: str, expected_values: List[List[float]]) -> None:
        df = DataFrame({'a': ['g1', 'g2', 'g1', 'g1'], 'b': [0, 0, 1, 2], 'c': [2, 0, 6, 4]})
        rol = df.groupby('a').rolling(3)
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

    def test_rolling_apply_mutability(self) -> None:
        df = DataFrame({'A': ['foo'] * 3 + ['bar'] * 3, 'B': [1] * 6})
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

        def isnumpyarray(x: np.ndarray) -> int:
            return int(isinstance(x, np.ndarray))
        df = DataFrame({'id': [1, 1, 1], 'value': [1, 2, 3]})
        result = df.groupby('id').value.rolling(1).apply(isnumpyarray, raw=raw_value)
        expected = Series([expected_value] * 3, index=MultiIndex.from_tuples(((1, 0), (1, 1), (1, 2)), names=['id', None]), name='value')
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_center_center(self) -> None:
        series = Series(range(1, 6))
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

    def test_groupby_rolling_center_on(self) -> None:
        df = DataFrame(data={'Date': date_range('2020-01-01', '2020-01-10'), 'gb': ['group_1'] * 6 + ['group_2'] * 4, 'value': range(10)})
        result = df.groupby('gb').rolling(6, on='Date', center=True, min_periods=1).value.mean()
        mi = MultiIndex.from_arrays([df['gb'], df['Date']], names=['gb', 'Date'])
        expected = Series([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5], name='value', index=mi)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('min_periods', [5, 4, 3])
    def test_groupby_rolling_center_min_periods(self, min_periods: int) -> None:
        df = DataFrame({'group': ['A'] * 10 + ['B'] * 10, 'data': range(20)})
        window_size = 5
        result = df.groupby('group').rolling(window_size, center=True, min_periods=min_periods).mean()
        result = result.reset_index()[['group', 'data']]
        grp_A_mean = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
        grp_B_mean = [x + 10.0 for x in grp_A_mean]
        num_nans = max(0, min_periods - 3)
        nans = [np.nan] * num_nans
        grp_A_expected = nans + grp_A_mean[num_nans:10 - num_nans] + nans
        grp_B_expected = nans + grp_B_mean[num_nans:10 - num_nans] + nans
        expected = DataFrame({'group': ['A'] * 10 + ['B'] * 10, 'data': grp_A_expected + grp_B_expected})
        tm.assert_frame_equal(result, expected)

    def test_groupby_subselect_rolling(self) -> None:
        df = DataFrame({'a': [1, 2, 3, 2], 'b': [4.0, 2.0, 3.0, 1.0], 'c': [10, 20, 30, 20]})
        result = df.groupby('a')[['b']].rolling(2).max()
        expected = DataFrame([np.nan, np.nan, 2.0, np.nan], columns=['b'], index=MultiIndex.from_tuples(((1, 0), (2, 1), (2, 3), (3, 2)), names=['a', None]))
        tm.assert_frame_equal(result, expected)
        result = df.groupby('a')['b'].rolling(2).max()
        expected = Series([np.nan, np.nan, 2.0, np.nan], index=MultiIndex.from_tuples(((1, 0), (2, 1), (2, 3), (3, 2)), names=['a', None]), name='b')
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_custom_indexer(self) -> None:

        class SimpleIndexer(BaseIndexer):

            def get_window_bounds(self, num_values: int = 0, min_periods: Optional[int] = None, center: Optional[bool] = None, closed: Optional[str] = None, step: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
                min_periods = self.window_size if min_periods is None else 0
                end = np.arange(num_values, dtype=np.int64) + 1
                start = end - self.window_size
                start[start < 0] = min_periods
                return (start, end)
        df = DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0] * 3}, index=[0] * 5 + [1] * 5 + [2] * 5)
        result = df.groupby(df.index).rolling(SimpleIndexer(window_size=3), min_periods=1).sum()
        expected = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_sub