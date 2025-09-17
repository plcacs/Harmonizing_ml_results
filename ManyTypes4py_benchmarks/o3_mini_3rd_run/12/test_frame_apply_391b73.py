from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, Timestamp, date_range
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
from pandas.util.version import Version
from typing import Any, Callable, Dict, List, Optional, Union

@pytest.fixture
def int_frame_const_col() -> DataFrame:
    """
    Fixture for DataFrame of ints which are constant per column

    Columns are ['A', 'B', 'C'], with values (per column): [1, 2, 3]
    """
    df: DataFrame = DataFrame(np.tile(np.arange(3, dtype='int64'), 6).reshape(6, -1) + 1, columns=['A', 'B', 'C'])
    return df

@pytest.fixture(params=['python', pytest.param('numba', marks=pytest.mark.single_cpu)])
def engine(request: Any) -> str:
    if request.param == 'numba':
        pytest.importorskip('numba')
    return request.param

def test_apply(float_frame: DataFrame, engine: str, request: Any) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason='numba engine not supporting numpy ufunc yet')
        request.node.add_marker(mark)
    with np.errstate(all='ignore'):
        result: Series = np.sqrt(float_frame['A'])
        expected: Series = float_frame.apply(np.sqrt, engine=engine)['A']
        tm.assert_series_equal(result, expected)
        result = float_frame.apply(np.mean, engine=engine)['A']
        expected_scalar = np.mean(float_frame['A'])
        assert result == expected_scalar
        d = float_frame.index[0]
        result = float_frame.apply(np.mean, axis=1, engine=engine)
        expected_scalar = np.mean(float_frame.xs(d))
        assert result[d] == expected_scalar
        assert result.index is float_frame.index

@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('raw', [True, False])
@pytest.mark.parametrize('nopython', [True, False])
def test_apply_args(float_frame: DataFrame, axis: Union[int, str], raw: bool, engine: str, nopython: bool) -> None:
    numba = pytest.importorskip('numba')
    if engine == 'numba' and Version(numba.__version__) == Version('0.61') and is_platform_arm():
        pytest.skip(f'Segfaults on ARM platforms with numba {numba.__version__}')
    engine_kwargs: Dict[str, bool] = {'nopython': nopython}
    result: DataFrame = float_frame.apply(lambda x, y: x + y, axis, args=(1,), raw=raw, engine=engine, engine_kwargs=engine_kwargs)
    expected: DataFrame = float_frame + 1
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
    df: DataFrame = DataFrame({'c0': ['A', 'A', 'B', 'B'], 'c1': ['C', 'C', 'D', 'D']})
    result: DataFrame = df.apply(lambda ts: ts.astype('category'))
    assert result.shape == (4, 2)
    assert isinstance(result['c0'].dtype, CategoricalDtype)
    assert isinstance(result['c1'].dtype, CategoricalDtype)

def test_apply_axis1_with_ea() -> None:
    expected: DataFrame = DataFrame({'A': [Timestamp('2013-01-01', tz='UTC')]})
    result: DataFrame = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data, dtype', [(1, None), (1, CategoricalDtype([1])), (Timestamp('2013-01-01', tz='UTC'), None)])
def test_agg_axis1_duplicate_index(data: Any, dtype: Any) -> None:
    expected: DataFrame = DataFrame([[data], [data]], index=['a', 'a'], dtype=dtype)
    result: DataFrame = expected.agg(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)

def test_apply_mixed_datetimelike() -> None:
    expected: DataFrame = DataFrame({'A': date_range('20130101', periods=3), 'B': pd.to_timedelta(np.arange(3), unit='s')})
    result: DataFrame = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func', [np.sqrt, np.mean])
def test_apply_empty(func: Callable[[Any], Any], engine: str) -> None:
    empty_frame: DataFrame = DataFrame()
    result = empty_frame.apply(func, engine=engine)
    assert result.empty

def test_apply_float_frame(float_frame: DataFrame, engine: str) -> None:
    no_rows: DataFrame = float_frame[:0]
    result: Series = no_rows.apply(lambda x: x.mean(), engine=engine)
    expected: Series = Series(np.nan, index=float_frame.columns)
    tm.assert_series_equal(result, expected)
    no_cols: DataFrame = float_frame.loc[:, []]
    result = no_cols.apply(lambda x: x.mean(), axis=1, engine=engine)
    expected = Series(np.nan, index=float_frame.index)
    tm.assert_series_equal(result, expected)

def test_apply_empty_except_index(engine: str) -> None:
    expected: DataFrame = DataFrame(index=['a'])
    result: DataFrame = expected.apply(lambda x: x['a'], axis=1, engine=engine)
    tm.assert_frame_equal(result, expected)

def test_apply_with_reduce_empty() -> None:
    empty_frame: DataFrame = DataFrame()
    x: List[Any] = []
    result = empty_frame.apply(x.append, axis=1, result_type='expand')
    tm.assert_frame_equal(result, empty_frame)
    result = empty_frame.apply(x.append, axis=1, result_type='reduce')
    expected: Series = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)
    empty_with_cols: DataFrame = DataFrame(columns=['a', 'b', 'c'])
    result = empty_with_cols.apply(x.append, axis=1, result_type='expand')
    tm.assert_frame_equal(result, empty_with_cols)
    result = empty_with_cols.apply(x.append, axis=1, result_type='reduce')
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)
    assert x == []

@pytest.mark.parametrize('func', ['sum', 'prod', 'any', 'all'])
def test_apply_funcs_over_empty(func: str) -> None:
    df: DataFrame = DataFrame(columns=['a', 'b', 'c'])
    result = df.apply(getattr(np, func))
    expected = getattr(df, func)()
    if func in ('sum', 'prod'):
        expected = expected.astype(float)
    tm.assert_series_equal(result, expected)

def test_nunique_empty() -> None:
    df: DataFrame = DataFrame(columns=['a', 'b', 'c'])
    result = df.nunique()
    expected: Series = Series(0, index=df.columns)
    tm.assert_series_equal(result, expected)
    result = df.T.nunique()
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)

def test_apply_standard_nonunique() -> None:
    df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'a', 'c'])
    result = df.apply(lambda s: s[0], axis=1)
    expected = Series([1, 4, 7], index=['a', 'a', 'c'])
    tm.assert_series_equal(result, expected)
    result = df.T.apply(lambda s: s[0], axis=0)
    tm.assert_series_equal(result, expected)

def test_apply_broadcast_scalars(float_frame: DataFrame) -> None:
    result = float_frame.apply(np.mean, result_type='broadcast')
    expected = DataFrame([float_frame.mean()], index=float_frame.index)
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_scalars_axis1(float_frame: DataFrame) -> None:
    result = float_frame.apply(np.mean, axis=1, result_type='broadcast')
    m: Series = float_frame.mean(axis=1)
    expected = DataFrame({c: m for c in float_frame.columns})
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_lists_columns(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: list(range(len(float_frame.columns))), axis=1, result_type='broadcast')
    m: List[int] = list(range(len(float_frame.columns)))
    expected = DataFrame([m] * len(float_frame.index), dtype='float64', index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_lists_index(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: list(range(len(float_frame.index))), result_type='broadcast')
    m: List[int] = list(range(len(float_frame.index)))
    expected = DataFrame({c: m for c in float_frame.columns}, dtype='float64', index=float_frame.index)
    tm.assert_frame_equal(result, expected)

def test_apply_broadcast_list_lambda_func(int_frame_const_col: DataFrame) -> None:
    df: DataFrame = int_frame_const_col
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type='broadcast')
    tm.assert_frame_equal(result, df)

def test_apply_broadcast_series_lambda_func(int_frame_const_col: DataFrame) -> None:
    df: DataFrame = int_frame_const_col
    result = df.apply(lambda x: Series([1, 2, 3], index=list('abc')), axis=1, result_type='broadcast')
    expected = df.copy()
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_float_frame(float_frame: DataFrame, axis: Union[int, str], engine: str) -> None:
    if engine == 'numba':
        pytest.skip("numba can't handle when UDF returns None.")

    def _assert_raw(x: Any) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1
    float_frame.apply(_assert_raw, axis=axis, engine=engine, raw=True)

@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_float_frame_lambda(float_frame: DataFrame, axis: Union[int, str], engine: str) -> None:
    result = float_frame.apply(np.mean, axis=axis, engine=engine, raw=True)
    expected = float_frame.apply(lambda x: x.values.mean(), axis=axis)
    tm.assert_series_equal(result, expected)

def test_apply_raw_float_frame_no_reduction(float_frame: DataFrame, engine: str) -> None:
    result = float_frame.apply(lambda x: x * 2, engine=engine, raw=True)
    expected = float_frame * 2
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_mixed_type_frame(axis: Union[int, str], engine: str) -> None:
    if engine == 'numba':
        pytest.skip("isinstance check doesn't work with numba")

    def _assert_raw(x: Any) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1
    df: DataFrame = DataFrame({
        'a': 1.0,
        'b': 2,
        'c': 'foo',
        'float32': np.array([1.0] * 10, dtype='float32'),
        'int32': np.array([1] * 10, dtype='int32')
    }, index=np.arange(10))
    df.apply(_assert_raw, axis=axis, engine=engine, raw=True)

def test_apply_axis1(float_frame: DataFrame) -> None:
    d = float_frame.index[0]
    result = float_frame.apply(np.mean, axis=1)[d]
    expected = np.mean(float_frame.xs(d))
    assert result == expected

def test_apply_mixed_dtype_corner() -> None:
    df: DataFrame = DataFrame({'A': ['foo'], 'B': [1.0]})
    result = df[:0].apply(np.mean, axis=1)
    expected = Series(dtype=np.float64)
    tm.assert_series_equal(result, expected)

def test_apply_mixed_dtype_corner_indexing() -> None:
    df: DataFrame = DataFrame({'A': ['foo'], 'B': [1.0]})
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
def test_apply_empty_infer_type(ax: str, func: Callable[[np.ndarray], Any], raw: bool, axis: Union[int, str], engine: str, request: Any) -> None:
    df: DataFrame = DataFrame(**{ax: ['a', 'b', 'c']})
    with np.errstate(all='ignore'):
        test_res = func(np.array([], dtype='f8'))
        is_reduction: bool = not isinstance(test_res, np.ndarray)
        result = df.apply(func, axis=axis, engine=engine, raw=raw)
        if is_reduction:
            agg_axis = df._get_agg_axis(axis)
            assert isinstance(result, Series)
            assert result.index is agg_axis
        else:
            assert isinstance(result, DataFrame)

def test_apply_empty_infer_type_broadcast() -> None:
    no_cols: DataFrame = DataFrame(index=['a', 'b', 'c'])
    result = no_cols.apply(lambda x: x.mean(), result_type='broadcast')
    assert isinstance(result, DataFrame)

def test_apply_with_args_kwds_add_some(float_frame: DataFrame) -> None:
    def add_some(x: Any, howmuch: int = 0) -> Any:
        return x + howmuch
    result = float_frame.apply(add_some, howmuch=2)
    expected = float_frame.apply(lambda x: x + 2)
    tm.assert_frame_equal(result, expected)

def test_apply_with_args_kwds_agg_and_add(float_frame: DataFrame) -> None:
    def agg_and_add(x: Any, howmuch: int = 0) -> Any:
        return x.mean() + howmuch
    result = float_frame.apply(agg_and_add, howmuch=2)
    expected = float_frame.apply(lambda x: x.mean() + 2)
    tm.assert_series_equal(result, expected)

def test_apply_with_args_kwds_subtract_and_divide(float_frame: DataFrame) -> None:
    def subtract_and_divide(x: Any, sub: Union[int, float], divide: Union[int, float] = 1) -> Any:
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
    data: DataFrame = DataFrame([[1, 2], [3, 4]], columns=['c0', 'c1'], index=['i0', 'i1'])
    result = data.apply(dict, axis=0)
    expected = Series([{'i0': 1, 'i1': 3}, {'i0': 2, 'i1': 4}], index=data.columns)
    tm.assert_series_equal(result, expected)
    result = data.apply(dict, axis=1)
    expected = Series([{'c0': 1, 'c1': 2}, {'c0': 3, 'c1': 4}], index=data.index)
    tm.assert_series_equal(result, expected)

def test_apply_differently_indexed() -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((20, 10)))
    result = df.apply(Series.describe, axis=0)
    expected = DataFrame({i: v.describe() for i, v in df.items()}, columns=df.columns)
    tm.assert_frame_equal(result, expected)
    result = df.apply(Series.describe, axis=1)
    expected = DataFrame({i: v.describe() for i, v in df.T.items()}, columns=df.index).T
    tm.assert_frame_equal(result, expected)

def test_apply_bug() -> None:
    positions: DataFrame = DataFrame([[1, 'ABC0', 50], [1, 'YUM0', 20], [1, 'DEF0', 20], [2, 'ABC1', 50], [2, 'YUM1', 20], [2, 'DEF1', 20]], columns=['a', 'market', 'position'])
    def f(r: Series) -> Any:
        return r['market']
    expected = positions.apply(f, axis=1)
    positions = DataFrame([[datetime(2013, 1, 1), 'ABC0', 50],
                           [datetime(2013, 1, 2), 'YUM0', 20],
                           [datetime(2013, 1, 3), 'DEF0', 20],
                           [datetime(2013, 1, 4), 'ABC1', 50],
                           [datetime(2013, 1, 5), 'YUM1', 20],
                           [datetime(2013, 1, 6), 'DEF1', 20]],
                          columns=['a', 'market', 'position'])
    result = positions.apply(f, axis=1)
    tm.assert_series_equal(result, expected)

def test_apply_convert_objects() -> None:
    expected: DataFrame = DataFrame({
        'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'],
        'C': ['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny', 'shiny', 'dull', 'shiny', 'shiny', 'shiny'],
        'D': np.random.default_rng(2).standard_normal(11),
        'E': np.random.default_rng(2).standard_normal(11),
        'F': np.random.default_rng(2).standard_normal(11)
    })
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)

def test_apply_attach_name(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: x.name)
    expected = Series(float_frame.columns, index=float_frame.columns)
    tm.assert_series_equal(result, expected)

def test_apply_attach_name_axis1(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: x.name, axis=1)
    expected = Series(float_frame.index, index=float_frame.index)
    tm.assert_series_equal(result, expected)

def test_apply_attach_name_non_reduction(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)))
    expected = DataFrame(np.tile(float_frame.columns, (len(float_frame.index), 1)),
                         index=float_frame.index, columns=float_frame.columns)
    tm.assert_frame_equal(result, expected)

def test_apply_attach_name_non_reduction_axis1(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)), axis=1)
    expected = Series((np.repeat(t[0], len(float_frame.columns)) for t in float_frame.itertuples()))
    expected.index = float_frame.index
    tm.assert_series_equal(result, expected)

def test_apply_multi_index() -> None:
    index: MultiIndex = MultiIndex.from_arrays([['a', 'a', 'b'], ['c', 'd', 'd']])
    s: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=['col1', 'col2'])
    result = s.apply(lambda x: Series({'min': min(x), 'max': max(x)}), 1)
    expected = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=['min', 'max'])
    tm.assert_frame_equal(result, expected, check_like=True)

@pytest.mark.parametrize('df, dicts', [
    [DataFrame([['foo', 'bar'], ['spam', 'eggs']]), Series([{0: 'foo', 1: 'spam'}, {0: 'bar', 1: 'eggs'}])],
    [DataFrame([[0, 1], [2, 3]]), Series([{0: 0, 1: 2}, {0: 1, 1: 3}])]
])
def test_apply_dict(df: DataFrame, dicts: Series) -> None:
    fn: Callable[[Series], Dict[Any, Any]] = lambda x: x.to_dict()
    reduce_true = df.apply(fn, result_type='reduce')
    reduce_false = df.apply(fn, result_type='expand')
    reduce_none = df.apply(fn)
    tm.assert_series_equal(reduce_true, dicts)
    tm.assert_frame_equal(reduce_false, df)
    tm.assert_series_equal(reduce_none, dicts)

def test_apply_non_numpy_dtype() -> None:
    df: DataFrame = DataFrame({'dt': date_range('2015-01-01', periods=3, tz='Europe/Brussels')})
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)
    result = df.apply(lambda x: x + pd.Timedelta('1day'))
    expected = DataFrame({'dt': date_range('2015-01-02', periods=3, tz='Europe/Brussels')})
    tm.assert_frame_equal(result, expected)

def test_apply_non_numpy_dtype_category() -> None:
    df: DataFrame = DataFrame({'dt': ['a', 'b', 'c', 'a']}, dtype='category')
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)

def test_apply_dup_names_multi_agg() -> None:
    df: DataFrame = DataFrame([[0, 1], [2, 3]], columns=['a', 'a'])
    expected = DataFrame([[0, 1]], columns=['a', 'a'], index=['min'])
    result = df.agg(['min'])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('op', ['apply', 'agg'])
def test_apply_nested_result_axis_1(op: str) -> None:
    def apply_list(row: Series) -> List[Any]:
        return [2 * row['A'], 2 * row['C'], 2 * row['B']]
    df: DataFrame = DataFrame(np.zeros((4, 4)), columns=list('ABCD'))
    result = getattr(df, op)(apply_list, axis=1)
    expected = Series([[0.0, 0.0, 0.0] for _ in range(4)])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('df', [DataFrame({'A': ['a', None], 'B': ['c', 'd']})])
@pytest.mark.parametrize('method', ['min', 'max', 'sum'])
def test_mixed_column_raises(df: DataFrame, method: str, using_infer_string: Any) -> None:
    if method == 'sum':
        msg = 'can only concatenate str \\(not "int"\\) to str|does not support'
    else:
        msg = "not supported between instances of 'str' and 'float'"
    if not using_infer_string:
        with pytest.raises(TypeError, match=msg):
            getattr(df, method)()
    else:
        getattr(df, method)()

@pytest.mark.parametrize('col', [1, 1.0, True, 'a', np.nan])
def test_apply_dtype(col: Any) -> None:
    df: DataFrame = DataFrame([[1.0, col]], columns=['a', 'b'])
    result = df.apply(lambda x: x.dtype)
    expected = df.dtypes
    tm.assert_series_equal(result, expected)

def test_apply_mutating() -> None:
    df: DataFrame = DataFrame({'a': list(range(10)), 'b': list(range(10, 20))})
    df_orig: DataFrame = df.copy()

    def func(row: Series) -> Series:
        mgr = row._mgr
        row.loc['a'] += 1
        assert row._mgr is not mgr
        return row
    expected: DataFrame = df.copy()
    expected['a'] += 1
    result = df.apply(func, axis=1)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, df_orig)

def test_apply_empty_list_reduce() -> None:
    df: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], columns=['a', 'b'])
    result = df.apply(lambda x: [], result_type='reduce')
    expected = Series({'a': [], 'b': []}, dtype=object)
    tm.assert_series_equal(result, expected)

def test_apply_no_suffix_index(engine: str, request: Any) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason="numba engine doesn't support list-likes/dict-like callables")
        request.node.add_marker(mark)
    pdf: DataFrame = DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    result = pdf.apply(['sum', lambda x: x.sum(), lambda x: x.sum()], engine=engine)
    expected = DataFrame({'A': [12, 12, 12], 'B': [27, 27, 27]}, index=['sum', '<lambda>', '<lambda>'])
    tm.assert_frame_equal(result, expected)

def test_apply_raw_returns_string(engine: str) -> None:
    if engine == 'numba':
        pytest.skip('No object dtype support in numba')
    df: DataFrame = DataFrame({'A': ['aa', 'bbb']})
    result = df.apply(lambda x: x[0], engine=engine, axis=1, raw=True)
    expected = Series(['aa', 'bbb'])
    tm.assert_series_equal(result, expected)

def test_aggregation_func_column_order() -> None:
    df: DataFrame = DataFrame([(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 5, 4), (5, 6, 6), (6, 7, 7)],
                                columns=('att1', 'att2', 'att3'))
    def sum_div2(s: Series) -> float:
        return s.sum() / 2
    aggs: List[Union[str, Callable[[Series], Any]]] = ['sum', sum_div2, 'count', 'min']
    result = df.agg(aggs)
    expected = DataFrame({
        'att1': [21.0, 10.5, 6.0, 1.0],
        'att2': [18.0, 9.0, 6.0, 0.0],
        'att3': [17.0, 8.5, 6.0, 0.0]
    }, index=['sum', 'sum_div2', 'count', 'min'])
    tm.assert_frame_equal(result, expected)

def test_apply_getitem_axis_1(engine: str, request: Any) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason='numba engine not supporting duplicate index values')
        request.node.add_marker(mark)
    df: DataFrame = DataFrame({'a': [0, 1, 2], 'b': [1, 2, 3]})
    result = df[['a', 'a']].apply(lambda x: x.iloc[0] + x.iloc[1], axis=1, engine=engine)
    expected = Series([0, 2, 4])
    tm.assert_series_equal(result, expected)

def test_nuiscance_depr_passes_through_warnings() -> None:
    def expected_warning(x: Any) -> Any:
        warnings.warn('Hello, World!')
        return x.sum()
    df: DataFrame = DataFrame({'a': [1, 2, 3]})
    with tm.assert_produces_warning(UserWarning, match='Hello, World!'):
        df.agg([expected_warning])

def test_apply_type() -> None:
    df: DataFrame = DataFrame({'col1': [3, 'string', float],
                               'col2': [0.25, datetime(2020, 1, 1), np.nan]},
                              index=['a', 'b', 'c'])
    result = df.apply(type, axis=0)
    expected = Series({'col1': Series, 'col2': Series})
    tm.assert_series_equal(result, expected)
    result = df.apply(type, axis=1)
    expected = Series({'a': Series, 'b': Series, 'c': Series})
    tm.assert_series_equal(result, expected)

def test_apply_on_empty_dataframe(engine: str) -> None:
    df: DataFrame = DataFrame({'a': [1, 2], 'b': [3, 0]})
    result = df.head(0).apply(lambda x: max(x['a'], x['b']), axis=1, engine=engine)
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)

def test_apply_return_list() -> None:
    df: DataFrame = DataFrame({'a': [1, 2], 'b': [2, 3]})
    result = df.apply(lambda x: [x.values])
    expected = DataFrame({'a': [[1, 2]], 'b': [[2, 3]]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('test, constant', [
    ({'a': [1, 2, 3], 'b': [1, 1, 1]}, {'a': [1, 2, 3], 'b': [1]}),
    ({'a': [2, 2, 2], 'b': [1, 1, 1]}, {'a': [2], 'b': [1]})
])
def test_unique_agg_type_is_series(test: Dict[str, List[Any]], constant: Dict[str, List[Any]]) -> None:
    df1: DataFrame = DataFrame(test)
    expected = Series(data=constant, index=['a', 'b'], dtype='object')
    aggregation: Dict[str, str] = {'a': 'unique', 'b': 'unique'}
    result = df1.agg(aggregation)
    tm.assert_series_equal(result, expected)

def test_any_apply_keyword_non_zero_axis_regression() -> None:
    df: DataFrame = DataFrame({'A': [1, 2, 0], 'B': [0, 2, 0], 'C': [0, 0, 0]})
    expected = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)
    result = df.apply('any', axis=1)
    tm.assert_series_equal(result, expected)
    result = df.apply('any', 1)
    tm.assert_series_equal(result, expected)

def test_agg_mapping_func_deprecated() -> None:
    df: DataFrame = DataFrame({'x': [1, 2, 3]})
    def foo1(x: Series, a: int = 1, c: int = 0) -> Series:
        return x + a + c
    def foo2(x: Series, b: int = 2, c: int = 0) -> Series:
        return x + b + c
    result = df.agg(foo1, 0, 3, c=4)
    expected = df + 7
    tm.assert_frame_equal(result, expected)
    result = df.agg([foo1, foo2], 0, 3, c=4)
    expected = DataFrame([[8, 8], [9, 9], [10, 10]], columns=[['x', 'x'], ['foo1', 'foo2']])
    tm.assert_frame_equal(result, expected)
    result = df.agg({'x': foo1}, 0, 3, c=4)
    expected = DataFrame([2, 3, 4], columns=['x'])
    tm.assert_frame_equal(result, expected)

def test_agg_std() -> None:
    df: DataFrame = DataFrame(np.arange(6).reshape(3, 2), columns=['A', 'B'])
    result = df.agg(np.std, ddof=1)
    expected = Series({'A': 2.0, 'B': 2.0}, dtype=float)
    tm.assert_series_equal(result, expected)
    result = df.agg([np.std], ddof=1)
    expected = DataFrame({'A': 2.0, 'B': 2.0}, index=['std'])
    tm.assert_frame_equal(result, expected)

def test_agg_dist_like_and_nonunique_columns() -> None:
    df: DataFrame = DataFrame({'A': [None, 2, 3], 'B': [1.0, np.nan, 3.0], 'C': ['foo', None, 'bar']})
    df.columns = ['A', 'A', 'C']
    result = df.agg({'A': 'count'})
    expected = df['A'].count()
    tm.assert_series_equal(result, expected)