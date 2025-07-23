from datetime import datetime
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, Timestamp, date_range
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
from pandas.util.version import Version

@pytest.fixture
def int_frame_const_col() -> DataFrame:
    """
    Fixture for DataFrame of ints which are constant per column

    Columns are ['A', 'B', 'C'], with values (per column): [1, 2, 3]
    """
    df = DataFrame(np.tile(np.arange(3, dtype='int64'), 6).reshape(6, -1) + 1, columns=['A', 'B', 'C'])
    return df

@pytest.fixture(params=['python', pytest.param('numba', marks=pytest.mark.single_cpu)])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == 'numba':
        pytest.importorskip('numba')
    return request.param

def test_apply(float_frame: DataFrame, engine: str, request: pytest.FixtureRequest) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason='numba engine not supporting numpy ufunc yet')
        request.node.add_marker(mark)
    with np.errstate(all='ignore'):
        result = np.sqrt(float_frame['A'])
        expected = float_frame.apply(np.sqrt, engine=engine)['A']
        tm.assert_series_equal(result, expected)
        result = float_frame.apply(np.mean, engine=engine)['A']
        expected = np.mean(float_frame['A'])
        assert result == expected
        d = float_frame.index[0]
        result = float_frame.apply(np.mean, axis=1, engine=engine)
        expected = np.mean(float_frame.xs(d))
        assert result[d] == expected
        assert result.index is float_frame.index

@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('raw', [True, False])
@pytest.mark.parametrize('nopython', [True, False])
def test_apply_args(float_frame: DataFrame, axis: int, raw: bool, engine: str, nopython: bool) -> None:
    numba = pytest.importorskip('numba')
    if engine == 'numba' and Version(numba.__version__) == Version('0.61') and is_platform_arm():
        pytest.skip(f'Segfaults on ARM platforms with numba {numba.__version__}')
    engine_kwargs = {'nopython': nopython}
    result = float_frame.apply(lambda x, y: x + y, axis, args=(1,), raw=raw, engine=engine, engine_kwargs=engine_kwargs)
    expected = float_frame + 1
    tm.assert_frame_equal(result, expected)
    result = float_frame.apply(lambda x, a, b: x + a + b, args=(1,), b=2, raw=raw, engine=engine, engine_kwargs=engine_kwargs)
    expected = float_frame + 3
    tm.assert_frame_equal(result, expected)
    if engine == 'numba':
        with pytest.raises(TypeError, match="missing a required argument: 'a'"):
            float_frame.apply(lambda x, a: x + a, b=2, raw=raw, engine=engine, engine_kwargs=engine_kwargs)
        with pytest.raises(pd.errors.NumbaUtilError, match='numba does not support keyword-only arguments'):
            float_frame.apply(lambda x, a, *, b: x + a + b, args=(1,), b=2, raw=raw, engine=engine, engine_kwargs=engine_kwargs)
        with pytest.raises(pd.errors.NumbaUtilError, match='numba does not support keyword-only arguments'):
            float_frame.apply(lambda *x, b: x[0] + x[1] + b, args=(1,), b=2, raw=raw, engine=engine, engine_kwargs=engine_kwargs)

def test_apply_categorical_func() -> None:
    df = DataFrame({'c0': ['A', 'A', 'B', 'B'], 'c1': ['C', 'C', 'D', 'D']})
    result = df.apply(lambda ts: ts.astype('category'))
    assert result.shape == (4, 2)
    assert isinstance(result['c0'].dtype, CategoricalDtype)
    assert isinstance(result['c1'].dtype, CategoricalDtype)

def test_apply_axis1_with_ea() -> None:
    expected = DataFrame({'A': [Timestamp('2013-01-01', tz='UTC')]})
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data, dtype', [(1, None), (1, CategoricalDtype([1])), (Timestamp('2013-01-01', tz='UTC'), None)])
def test_agg_axis1_duplicate_index(data: Any, dtype: Any) -> None:
    expected = DataFrame([[data], [data]], index=['a', 'a'], dtype=dtype)
    result = expected.agg(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)

def test_apply_mixed_datetimelike() -> None:
    expected = DataFrame({'A': date_range('20130101', periods=3), 'B': pd.to_timedelta(np.arange(3), unit='s')})
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func', [np.sqrt, np.mean])
def test_apply_empty(func: Callable, engine: str) -> None:
    empty_frame = DataFrame()
    result = empty_frame.apply(func, engine=engine)
    assert result.empty

def test_apply_float_frame(float_frame: DataFrame, engine: str) -> None:
    no_rows = float_frame[:0]
    result = no_rows.apply(lambda x: x.mean(), engine=engine)
    expected = Series(np.nan, index=float_frame.columns)
    tm.assert_series_equal(result, expected)
    no_cols = float_frame.loc[:, []]
    result = no_cols.apply(lambda x: x.mean(), axis=1, engine=engine)
    expected = Series(np.nan, index=float_frame.index)
    tm.assert_series_equal(result, expected)

def test_apply_empty_except_index(engine: str) -> None:
    expected = DataFrame(index=['a'])
    result = expected.apply(lambda x: x['a'], axis=1, engine=engine)
    tm.assert_frame_equal(result, expected)

def test_apply_with_reduce_empty() -> None:
    empty_frame = DataFrame()
    x = []
    result = empty_frame.apply(x.append, axis=1, result_type='expand')
    tm.assert_frame_equal(result, empty_frame)
    result = empty_frame.apply(x.append, axis=1, result_type='reduce')
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)
    empty_with_cols = DataFrame(columns=['a', 'b', 'c'])
    result = empty_with_cols.apply(x.append, axis=1, result_type='expand')
    tm.assert_frame_equal(result, empty_with_cols)
    result = empty_with_cols.apply(x.append, axis=1, result_type='reduce')
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)
    assert x == []

@pytest.mark.parametrize('func', ['sum', 'prod', 'any', 'all'])
def test_apply_funcs_over_empty(func: str) -> None:
    df = DataFrame(columns=['a', 'b', 'c'])
    result = df.apply(getattr(np, func))
    expected = getattr(df, func)()
    if func in ('sum', 'prod'):
        expected = expected.astype(float)
    tm.assert_series_equal(result, expected)

def test_nunique_empty() -> None:
    df = DataFrame(columns=['a', 'b', 'c'])
    result = df.nunique()
    expected = Series(0, index=df.columns)
    tm.assert_series_equal(result, expected)
    result = df.T.nunique()
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)

def test_apply_standard_nonunique() -> None:
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'a', 'c'])
    result = df.apply(lambda s: s[0], axis=1)
    expected = Series([1, 4, 7], ['a', 'a', 'c'])
    tm.assert_series_equal(result, expected)
    result = df.T.apply(lambda s: s[0], axis=0)
    tm.assert_series_equal(result, expected)

def test_apply_broadcast_scalars(float_frame: DataFrame) -> None:
    result = float_frame.apply(np.mean, result_type='broadcast')
    expected = DataFrame([float_frame.mean()], index=float_frame.index)
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_scalars_axis1(float_frame: DataFrame) -> None:
    result = float_frame.apply(np.mean, axis=1, result_type='broadcast')
    m = float_frame.mean(axis=1)
    expected = DataFrame({c: m for c in float_frame.columns})
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_lists_columns(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: list(range(len(float_frame.columns))), axis=1, result_type='broadcast')
    m = list(range(len(float_frame.columns)))
    expected = DataFrame([m] * len(float_frame.index), dtype='float64', index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_lists_index(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: list(range(len(float_frame.index))), result_type='broadcast')
    m = list(range(len(float_frame.index)))
    expected = DataFrame({c: m for c in float_frame.columns}, dtype='float64', index=float_frame.index)
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_list_lambda_func(int_frame_const_col: DataFrame) -> None:
    df = int_frame_const_col
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type='broadcast')
    tm.assert_frame_equal(result, df)

def test_apply_broadcast_series_lambda_func(int_frame_const_col: DataFrame) -> None:
    df = int_frame_const_col
    result = df.apply(lambda x: Series([1, 2, 3], index=list('abc')), axis=1, result_type='broadcast')
    expected = df.copy()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_float_frame(float_frame: DataFrame, axis: int, engine: str) -> None:
    if engine == 'numba':
        pytest.skip("numba can't handle when UDF returns None.")

    def _assert_raw(x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1
    float_frame.apply(_assert_raw, axis=axis, engine=engine, raw=True)

@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_float_frame_lambda(float_frame: DataFrame, axis: int, engine: str) -> None:
    result = float_frame.apply(np.mean, axis=axis, engine=engine, raw=True)
    expected = float_frame.apply(lambda x: x.values.mean(), axis=axis)
    tm.assert_series_equal(result, expected)

def test_apply_raw_float_frame_no_reduction(float_frame: DataFrame, engine: str) -> None:
    result = float_frame.apply(lambda x: x * 2, engine=engine, raw=True)
    expected = float_frame * 2
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_mixed_type_frame(axis: int, engine: str) -> None:
    if engine == 'numba':
        pytest.skip("isinstance check doesn't work with numba")

    def _assert_raw(x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1
    df = DataFrame({'a': 1.0, 'b': 2, 'c': 'foo', 'float32': np.array([1.0] * 10, dtype='float32'), 'int32': np.array([1] * 10, dtype='int32')}, index=np.arange(10))
    df.apply(_assert_raw, axis=axis, engine=engine, raw=True)

def test_apply_axis1(float_frame: DataFrame) -> None:
    d = float_frame.index[0]
    result = float_frame.apply(np.mean, axis=1)[d]
    expected = np.mean(float_frame.xs(d))
    assert result == expected

def test_apply_mixed_dtype_corner() -> None:
    df = DataFrame({'A': ['foo'], 'B': [1.0]})
    result = df[:0].apply(np.mean, axis=1)
    expected = Series(dtype=np.float64)
    tm.assert_series_equal(result, expected)

def test_apply_mixed_dtype_corner_indexing() -> None:
    df = DataFrame({'A': ['foo'], 'B': [1.0]})
    result = df.apply(lambda x: x['A'], axis=1)
    expected = Series(['foo'], index=range(1))
    tm.assert_series_equal(result, expected)
    result = df.apply(lambda x: x['B'], axis=1)
    expected = Series([1.0], index=range(1))
    tm.assert_series_equal(result, expected)

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('ax', ['index', 'columns'])
@pytest.mark.parametrize('func', [lambda x: x, lambda x: x.mean()], ids=['identity', 'mean'])
@pytest.mark.parametrize('raw', [True, False])
@pytest.mark.parametrize('axis', [0, 1])
def test_apply_empty_infer_type(ax: str, func: Callable, raw: bool, axis: int, engine: str, request: pytest.FixtureRequest) -> None:
    df = DataFrame(**{ax: ['a', 'b', 'c']})
    with np.errstate(all='ignore'):
        test_res = func(np.array([], dtype='f8'))
        is_reduction = not isinstance(test_res, np.ndarray)
        result = df.apply(func, axis=axis, engine=engine, raw=raw)
        if is_reduction:
            agg_axis = df._get_agg_axis(axis)
            assert isinstance(result, Series)
            assert result.index is agg_axis
        else:
            assert isinstance(result, DataFrame)

def test_apply_empty_infer_type_broadcast() -> None:
    no_cols = DataFrame(index=['a', 'b', 'c'])
    result = no_cols.apply(lambda x: x.mean(), result_type='broadcast')
    assert isinstance(result, DataFrame)

def test_apply_with_args_kwds_add_some(float_frame: DataFrame) -> None:

    def add_some(x: Series, howmuch: int = 0) -> Series:
        return x + howmuch
    result = float_frame.apply(add_some, howmuch=2)
    expected = float_frame.apply(lambda x: x + 2)
    tm.assert_frame_equal(result, expected)

def test_apply_with_args_kwds_agg_and_add(float_frame: DataFrame) -> None:

    def agg_and_add(x: Series, howmuch: int = 0) -> float:
        return x.mean() + howmuch
    result = float_frame.apply(agg_and_add, howmuch=2)
    expected = float_frame.apply(lambda x: x.mean() + 2)
    tm.assert_series_equal(result, expected)

def test_apply_with_args_kwds_subtract_and_divide(float_frame: DataFrame) -> None:

    def subtract_and_divide(x: Series, sub: int, divide: int = 1) -> Series:
        return (x - sub) / divide
    result = float_frame.apply(subtract_and_divide, args=(2,), divide=2)
    expected = float_frame.apply(lambda x: (x - 2.0) / 2.0)
    tm.assert_frame_equal(result, expected)

def test_apply_yield_list(float_frame: DataFrame) -> None:
    result = float_frame.apply(list)
    tm.assert_frame_equal(result, float_frame)

def test_apply_reduce_Series(float_frame: DataFrame) -> None:
    float_frame.iloc[::2, float_frame.columns.get_loc('A')] = np.nan
    expected = float_frame.mean(axis=1)
    result = float_frame.apply(np.mean, axis=1)
    tm.assert_series_equal(result, expected)

def test_apply_reduce_to_dict() -> None:
    data = DataFrame([[1, 2], [3, 4]], columns=['c0', 'c1'], index=['i0', 'i1'])
    result = data.apply(dict, axis=0)
    expected = Series([{'i0': 1, 'i1': 3}, {'i0': 2, 'i1': 4}], index=data.columns)
    tm.assert_series_equal(result, expected)
    result = data.apply(dict, axis=1)
    expected = Series([{'c0': 1, 'c1': 2}, {'c0': 3, 'c1': 4}], index=data.index)
    tm.assert_series_equal(result, expected)

def test_apply_differently_indexed() -> None:
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 10)))
    result = df.apply(Series.describe, axis=