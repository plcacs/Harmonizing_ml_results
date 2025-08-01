from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pytest
from pandas._libs.algos import Infinity, NegInfinity
from pandas import DataFrame, Index, Series
import pandas._testing as tm

class TestRank:
    s: Series = Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])
    df: DataFrame = DataFrame({'A': s, 'B': s})
    results: Dict[str, np.ndarray] = {
        'average': np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5]),
        'min': np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5]),
        'max': np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6]),
        'first': np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6]),
        'dense': np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])
    }

    def test_rank(self, float_frame: DataFrame) -> None:
        sp_stats = pytest.importorskip('scipy.stats')
        float_frame.loc[::2, 'A'] = np.nan
        float_frame.loc[::3, 'B'] = np.nan
        float_frame.loc[::4, 'C'] = np.nan
        float_frame.loc[::5, 'D'] = np.nan
        ranks0 = float_frame.rank()
        ranks1 = float_frame.rank(1)
        mask = np.isnan(float_frame.values)
        fvals = float_frame.fillna(np.inf).values
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        exp0[mask] = np.nan
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)
        exp1[mask] = np.nan
        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)
        df = DataFrame(np.random.default_rng(2).integers(0, 5, size=40).reshape((10, 4)))
        result = df.rank()
        exp = df.astype(float).rank()
        tm.assert_frame_equal(result, exp)
        result = df.rank(1)
        exp = df.astype(float).rank(1)
        tm.assert_frame_equal(result, exp)

    def test_rank2(self) -> None:
        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected = DataFrame([[1.0, 3.0, 2.0], [1, 2, 3]]) / 3.0
        result = df.rank(1, pct=True)
        tm.assert_frame_equal(result, expected)
        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected = df.rank(0) / 2.0
        result = df.rank(0, pct=True)
        tm.assert_frame_equal(result, expected)
        df = DataFrame([['b', 'c', 'a'], ['a', 'c', 'b']])
        expected = DataFrame([[2.0, 3.0, 1.0], [1, 3, 2]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([[2.0, 1.5, 1.0], [1, 1.5, 2]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)
        df = DataFrame([['b', np.nan, 'a'], ['a', 'c', 'b']])
        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 3.0, 2.0]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 1.0, 2.0]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)
        data = [[datetime(2001, 1, 5), np.nan, datetime(2001, 1, 2)], [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 1)]]
        df = DataFrame(data)
        expected = DataFrame([[2.0, np.nan, 1.0], [2.0, 3.0, 1.0]])
        result = df.rank(1, numeric_only=False, ascending=True)
        tm.assert_frame_equal(result, expected)
        expected = DataFrame([[1.0, np.nan, 2.0], [2.0, 1.0, 3.0]])
        result = df.rank(1, numeric_only=False, ascending=False)
        tm.assert_frame_equal(result, expected)
        df = DataFrame({'a': [1e-20, -5, 1e-20 + 1e-40, 10, 1e+60, 1e+80, 1e-30]})
        exp = DataFrame({'a': [3.5, 1.0, 3.5, 5.0, 6.0, 7.0, 2.0]})
        tm.assert_frame_equal(df.rank(), exp)

    def test_rank_does_not_mutate(self) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), dtype='float64')
        expected = df.copy()
        df.rank()
        result = df
        tm.assert_frame_equal(result, expected)

    def test_rank_mixed_frame(self, float_string_frame: DataFrame) -> None:
        float_string_frame['datetime'] = datetime.now()
        float_string_frame['timedelta'] = timedelta(days=1, seconds=1)
        float_string_frame.rank(numeric_only=False)
        with pytest.raises(TypeError, match='not supported between instances of'):
            float_string_frame.rank(axis=1)

    def test_rank_na_option(self, float_frame: DataFrame) -> None:
        sp_stats = pytest.importorskip('scipy.stats')
        float_frame.loc[::2, 'A'] = np.nan
        float_frame.loc[::3, 'B'] = np.nan
        float_frame.loc[::4, 'C'] = np.nan
        float_frame.loc[::5, 'D'] = np.nan
        ranks0 = float_frame.rank(na_option='bottom')
        ranks1 = float_frame.rank(1, na_option='bottom')
        fvals = float_frame.fillna(np.inf).values
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)
        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)
        ranks0 = float_frame.rank(na_option='top')
        ranks1 = float_frame.rank(1, na_option='top')
        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fval0)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fval1)
        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)
        ranks0 = float_frame.rank(na_option='top', ascending=False)
        ranks1 = float_frame.rank(1, na_option='top', ascending=False)
        fvals = float_frame.fillna(np.inf).values
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fvals)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fvals)
        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)
        ranks0 = float_frame.rank(na_option='bottom', ascending=False)
        ranks1 = float_frame.rank(1, na_option='bottom', ascending=False)
        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values
        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fval0)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fval1)
        tm.assert_numpy_array_equal(ranks0.values, exp0)
        tm.assert_numpy_array_equal(ranks1.values, exp1)
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"
        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option='bad', ascending=False)
        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option=True, ascending=False)

    def test_rank_axis(self) -> None:
        df = DataFrame([[2, 1], [4, 3]])
        tm.assert_frame_equal(df.rank(axis=0), df.rank(axis='index'))
        tm.assert_frame_equal(df.rank(axis=1), df.rank(axis='columns'))

    @pytest.mark.parametrize('ax', [0, 1])
    def test_rank_methods_frame(self, ax: int, rank_method: str) -> None:
        sp_stats = pytest.importorskip('scipy.stats')
        xs = np.random.default_rng(2).integers(0, 21, (100, 26))
        xs = (xs - 10.0) / 10.0
        cols = [chr(ord('z') - i) for i in range(xs.shape[1])]
        for vals in [xs, xs + 1000000.0, xs * 1e-06]:
            df = DataFrame(vals, columns=cols)
            result = df.rank(axis=ax, method=rank_method)
            sprank = np.apply_along_axis(sp_stats.rankdata, ax, vals, rank_method if rank_method != 'first' else 'ordinal')
            sprank = sprank.astype(np.float64)
            expected = DataFrame(sprank, columns=cols).astype('float64')
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['O', 'f8', 'i8'])
    def test_rank_descending(self, rank_method: str, dtype: str) -> None:
        if 'i' in dtype:
            df = self.df.dropna().astype(dtype)
        else:
            df = self.df.astype(dtype)
        res = df.rank(ascending=False)
        expected = (df.max() - df).rank()
        tm.assert_frame_equal(res, expected)
        expected = (df.max() - df).rank(method=rank_method)
        if dtype != 'O':
            res2 = df.rank(method=rank_method, ascending=False, numeric_only=True)
            tm.assert_frame_equal(res2, expected)
        res3 = df.rank(method=rank_method, ascending=False, numeric_only=False)
        tm.assert_frame_equal(res3, expected)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_rank_2d_tie_methods(self, rank_method: str, axis: int, dtype: Optional[str]) -> None:
        df = self.df

        def _check2d(df: DataFrame, expected: np.ndarray, method: str = 'average', axis: int = 0) -> None:
            exp_df = DataFrame({'A': expected, 'B': expected})
            if axis == 1:
                df = df.T
                exp_df = exp_df.T
            result = df.rank(method=rank_method, axis=axis)
            tm.assert_frame_equal(result, exp_df)
        frame = df if dtype is None else df.astype(dtype)
        _check2d(frame, self.results[rank_method], method=rank_method, axis=axis)

    @pytest.mark.parametrize('rank_method,exp', [('dense', [[1.0, 1.0, 1.0], [1.0, 0.5, 2.0 / 3], [1.0, 0.5, 1.0 / 3]]), ('min', [[1.0 / 3, 1.0, 1.0], [1.0 / 3, 1.0 / 3, 2.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3]]), ('max', [[1.0, 1.0, 1.0], [1.0, 2.0 / 3, 2.0 / 3], [1.0, 2.0 / 3, 1.0 / 3]]), ('average', [[2.0 / 3, 1.0, 1.0], [2.0 / 3, 0.5, 2.0 / 3], [2.0 / 3, 0.5, 1.0 / 3]]), ('first', [[1.0 / 3, 1.0, 1.0], [2.0 / 3, 1.0 / 3, 2.0 / 3], [3.0 / 3, 2.0 / 3, 1.0 / 3]])])
    def test_rank_pct_true(self, rank_method: str, exp: List[List[float]]) -> None:
        df = DataFrame([[2012, 66, 3], [2012, 65, 2], [2012, 65, 1]])
        result = df.rank(method=rank_method, pct=True)
        expected = DataFrame(exp)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self) -> None:
        df = DataFrame({'A': np.arange(2 ** 24 + 1), 'B': np.arange(2 ** 24 + 1, 0, -1)})
        result = df.rank(pct=True).max()
        assert (result == 1).all()

    @pytest.mark.parametrize('contents,dtype', [([-np.inf, -50, -1, -1e-20, -1e-25, -1e-50, 0, 1e-40, 1e-20, 1e-10, 2, 40, np.inf], 'float64'), ([-np.inf, -50, -1, -1e-20, -1e-25, -1e-45, 0, 1e-40, 1e-20, 1e-10, 2, 40, np.inf], 'float32'), ([np.iinfo(np.uint8).min, 1, 2, 100, np.iinfo(np.uint8).max], 'uint8'), ([np.iinfo(np.int64).min, -100, 0, 1, 9999, 100000, 10000000000.0, np.iinfo(np.int64).max], 'int64'), ([NegInfinity(), '1', 'A', 'BA', 'Ba', 'C', Infinity()], 'object'), ([datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 5)], 'datetime64')])
    def test_rank_inf_and_nan(self, contents: List[Any], dtype: str, frame_or_series: Any) -> None:
        dtype_na_map = {'float64': np.nan, 'float32': np.nan, 'object': None, 'datetime64': np.datetime64('nat')}
        values = np.array(contents, dtype=dtype)
        exp_order = np.array(range(len(values)), dtype='float64') + 1.0
        if dtype in dtype_na_map:
            na_value = dtype_na_map[dtype]
            nan_indices = np.random.default_rng(2).choice(range(len(values)), 5)
            values = np.insert(values, nan_indices, na_value)
            exp_order = np.insert(exp_order, nan_indices, np.nan)
        random_order = np.random.default_rng(2).permutation(len(values))
        obj = frame_or_series(values[random_order])
        expected = frame_or_series(exp_order[random_order], dtype='float64')
        result = obj.rank()
        tm.assert_equal(result, expected)

    def test_df_series_inf_nan_consistency(self) -> None:
        index = [5, 4, 3, 2,