from itertools import chain
import operator
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
from pandas._libs.algos import Infinity, NegInfinity
import pandas.util._test_decorators as td
from pandas import NA, NaT, Series, Timestamp, date_range
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.dtypes import ExtensionDtype

@pytest.fixture
def ser() -> Series:
    return Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])

@pytest.fixture(params=[['average', np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5])], ['min', np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5])], ['max', np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6])], ['first', np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6])], ['dense', np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])]], ids=lambda x: x[0])
def results(request: pytest.FixtureRequest) -> Tuple[str, np.ndarray]:
    return request.param

@pytest.fixture(params=['object', 'float64', 'int64', 'Float64', 'Int64', pytest.param('float64[pyarrow]', marks=td.skip_if_no('pyarrow')), pytest.param('int64[pyarrow]', marks=td.skip_if_no('pyarrow')), pytest.param('string[pyarrow]', marks=td.skip_if_no('pyarrow')), 'string[python]', 'str'])
def dtype(request: pytest.FixtureRequest) -> str:
    return request.param

def expected_dtype(dtype: str, method: str, pct: bool = False) -> str:
    exp_dtype = 'float64'
    if dtype in ['string[pyarrow]']:
        exp_dtype = 'Float64'
    elif dtype in ['float64[pyarrow]', 'int64[pyarrow]']:
        if method == 'average' or pct:
            exp_dtype = 'double[pyarrow]'
        else:
            exp_dtype = 'uint64[pyarrow]'
    return exp_dtype

class TestSeriesRank:

    def test_rank(self, datetime_series: Series) -> None:
        sp_stats = pytest.importorskip('scipy.stats')
        datetime_series[::2] = np.nan
        datetime_series[:10:3] = 4.0
        ranks = datetime_series.rank()
        oranks = datetime_series.astype('O').rank()
        tm.assert_series_equal(ranks, oranks)
        mask = np.isnan(datetime_series)
        filled = datetime_series.fillna(np.inf)
        exp = Series(sp_stats.rankdata(filled), index=filled.index, name='ts')
        exp[mask] = np.nan
        tm.assert_series_equal(ranks, exp)
        iseries = Series(np.arange(5).repeat(2))
        iranks = iseries.rank()
        exp = iseries.astype(float).rank()
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.arange(5)) + 1.0
        exp = iseries / 5.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.repeat(1, 100))
        exp = Series(np.repeat(0.505, 100))
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)
        iseries = iseries.astype('float')
        iseries[1] = np.nan
        exp = Series(np.repeat(50.0 / 99.0, 100))
        exp[1] = np.nan
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.arange(5)) + 1.0
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.repeat(np.nan, 100))
        exp = iseries
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)
        iseries = Series(np.arange(5), dtype='float') + 1
        iseries[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)
        rng = date_range('1/1/1990', periods=5)
        iseries = Series(np.arange(5), rng, dtype='float') + 1
        iseries.iloc[4] = np.nan
        exp = iseries / 4.0
        iranks = iseries.rank(pct=True)
        tm.assert_series_equal(iranks, exp)
        iseries = Series([1e-50, 1e-100, 1e-20, 0.01, 1e-20 + 1e-30, 0.1])
        exp = Series([2, 1, 3, 5, 4, 6.0])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)
        iseries = Series(['3 day', '1 day 10m', '-2 day', NaT], dtype='m8[ns]')
        exp = Series([3, 2, 1, np.nan])
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)
        values = np.array([-50, -1, -1e-20, -1e-25, -1e-50, 0, 1e-40, 1e-20, 1e-10, 2, 40], dtype='float64')
        random_order = np.random.default_rng(2).permutation(len(values))
        iseries = Series(values[random_order])
        exp = Series(random_order + 1.0, dtype='float64')
        iranks = iseries.rank()
        tm.assert_series_equal(iranks, exp)

    def test_rank_categorical(self) -> None:
        exp = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        exp_desc = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        ordered = Series(['first', 'second', 'third', 'fourth', 'fifth', 'sixth']).astype(CategoricalDtype(categories=['first', 'second', 'third', 'fourth', 'fifth', 'sixth'], ordered=True))
        tm.assert_series_equal(ordered.rank(), exp)
        tm.assert_series_equal(ordered.rank(ascending=False), exp_desc)
        unordered = Series(['first', 'second', 'third', 'fourth', 'fifth', 'sixth']).astype(CategoricalDtype(categories=['first', 'second', 'third', 'fourth', 'fifth', 'sixth'], ordered=False))
        exp_unordered = Series([2.0, 4.0, 6.0, 3.0, 1.0, 5.0])
        res = unordered.rank()
        tm.assert_series_equal(res, exp_unordered)
        unordered1 = Series([1, 2, 3, 4, 5, 6]).astype(CategoricalDtype([1, 2, 3, 4, 5, 6], False))
        exp_unordered1 = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res1 = unordered1.rank()
        tm.assert_series_equal(res1, exp_unordered1)
        na_ser = Series(['first', 'second', 'third', 'fourth', 'fifth', 'sixth', np.nan]).astype(CategoricalDtype(['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'], True))
        exp_top = Series([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0])
        exp_bot = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        exp_keep = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
        tm.assert_series_equal(na_ser.rank(na_option='top'), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option='bottom'), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option='keep'), exp_keep)
        exp_top = Series([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        exp_bot = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.0])
        exp_keep = Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, np.nan])
        tm.assert_series_equal(na_ser.rank(na_option='top', ascending=False), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option='bottom', ascending=False), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option='keep', ascending=False), exp_keep)
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"
        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option='bad', ascending=False)
        with pytest.raises(ValueError, match=msg):
            na_ser.rank(na_option=True, ascending=False)
        na_ser = Series(['first', 'second', 'third', 'fourth', np.nan]).astype(CategoricalDtype(['first', 'second', 'third', 'fourth'], True))
        exp_top = Series([0.4, 0.6, 0.8, 1.0, 0.2])
        exp_bot = Series([0.2, 0.4, 0.6, 0.8, 1.0])
        exp_keep = Series([0.25, 0.5, 0.75, 1.0, np.nan])
        tm.assert_series_equal(na_ser.rank(na_option='top', pct=True), exp_top)
        tm.assert_series_equal(na_ser.rank(na_option='bottom', pct=True), exp_bot)
        tm.assert_series_equal(na_ser.rank(na_option='keep', pct=True), exp_keep)

    def test_rank_nullable_integer(self) -> None:
        exp = Series([np.nan, 2, np.nan, 3, 3, 2, 3, 1])
        exp = exp.astype('Int64')
        result = exp.rank(na_option='keep')
        expected = Series([np.nan, 2.5, np.nan, 5.0, 5.0, 2.5, 5.0, 1.0])
        tm.assert_series_equal(result, expected)

    def test_rank_signature(self) -> None:
        s = Series([0, 1])
        s.rank(method='average')
        msg = 'No axis named average for object type Series'
        with pytest.raises(ValueError, match=msg):
            s.rank('average')

    def test_rank_tie_methods(self, ser: Series, results: Tuple[str, np.ndarray], dtype: str, using_infer_string: bool) -> None:
        method, exp = results
        if dtype == 'int64' or (not using_infer_string and dtype == 'str'):
            pytest.skip('int64/str does not support NaN')
        ser = ser if dtype is None else ser.astype(dtype)
        result = ser.rank(method=method)
        tm.assert_series_equal(result, Series(exp, dtype=expected_dtype(dtype, method)))

    @pytest.mark.parametrize('na_option', ['top', 'bottom', 'keep'])
    @pytest.mark.parametrize('dtype, na_value, pos_inf, neg_inf', [('object', None, Infinity(), NegInfinity()), ('float64', np.nan, np.inf, -np.inf), ('Float64', NA, np.inf, -np.inf), pytest.param('float64[pyarrow]', NA, np.inf, -np.inf, marks=td.skip_if_no('pyarrow'))])
    def test_rank_tie_methods_on_infs_nans(self, rank_method: str, na_option: str, ascending: bool, dtype: str, na_value: Any, pos_inf: Any, neg_inf: Any) -> None:
        pytest.importorskip('scipy')
        if dtype == 'float64[pyarrow]':
            if rank_method == 'average':
                exp_dtype = 'float64[pyarrow]'
            else:
                exp_dtype = 'uint64[pyarrow]'
        else:
            exp_dtype = 'float64'
        chunk = 3
        in_arr = [neg_inf] * chunk + [na_value] * chunk + [pos_inf] * chunk
        iseries = Series(in_arr, dtype=dtype)
        exp_ranks = {'average': ([2, 2, 2], [5, 5, 5], [8, 8, 8]), 'min': ([1, 1, 1], [4, 4, 4], [7, 7, 7]), 'max': ([3, 3, 3], [6, 6, 6], [9, 9, 9]), 'first': ([1, 2, 3], [4, 5, 6], [7, 8, 9]), 'dense': ([1, 1, 1], [2, 2, 2], [3, 3, 3])}
        ranks = exp_ranks[rank_method]
        if na_option == 'top':
            order = [ranks[1], ranks[0], ranks[2]]
        elif na_option == 'bottom':
            order = [ranks[0], ranks[2], ranks[1]]
        else:
            order = [ranks[0], [np.nan] * chunk, ranks[1]]
        expected = order if ascending else order[::-1]
        expected = list(chain.from_iterable(expected))
        result = iseries.rank(method=rank_method, na_option=na_option, ascending=ascending)
        tm.assert_series_equal(result, Series(expected, dtype=exp_dtype))

    def test_rank_desc_mix_nans_infs(self) -> None:
        iseries = Series([1, np.nan, np.inf, -np.inf, 25])
        result = iseries.rank(ascending=False)
        exp = Series([3, np.nan, 1, 4, 2], dtype='float64')
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize('op, value', [[operator.add, 0], [operator.add, 1000000.0], [operator.mul, 1e-06]])
    def test_rank_methods_series(self, rank_method: str, op: operator, value: float) -> None:
        sp_stats = pytest.importorskip('scipy.stats')
        xs = np.random.default_rng(2).standard_normal(9)
        xs = np.concatenate([xs[i:] for i in range(0, 9, 2)])
        np.random.default_rng(2).shuffle(xs)
        index = [chr(ord('a') + i) for i in range(len(xs))]
        vals = op(xs, value)
        ts = Series(vals, index=index)
        result = ts.rank(method=rank_method)
        sprank = sp_stats.rankdata(vals, rank_method if rank_method != 'first' else 'ordinal')
        expected = Series(sprank, index=index).astype('float64')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ser, exp', [([1], [1]), ([2], [1]), ([0], [1]), ([2, 2], [1, 1]), ([1, 2, 3], [1, 2, 3]), ([4, 2, 1], [3, 2, 1]), ([1, 1, 5, 5, 3], [1, 1, 3, 3, 2]), ([-5, -4, -3, -2, -1], [1, 2, 3, 4, 5])])
    def test_rank_dense_method(self, dtype: str,