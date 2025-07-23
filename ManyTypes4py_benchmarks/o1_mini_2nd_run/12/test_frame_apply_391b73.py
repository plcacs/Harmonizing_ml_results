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
from typing import Callable, Any, List, Dict, Optional, Tuple, Union


@pytest.fixture
def int_frame_const_col() -> DataFrame:
    """
    Fixture for DataFrame of ints which are constant per column

    Columns are ['A', 'B', 'C'], with values (per column): [1, 2, 3]
    """
    df = DataFrame(
        np.tile(np.arange(3, dtype='int64'), 6).reshape(6, -1) + 1, columns=['A', 'B', 'C']
    )
    return df


@pytest.fixture(params=['python', pytest.param('numba', marks=pytest.mark.single_cpu)])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == 'numba':
        pytest.importorskip('numba')
    return request.param


def test_apply(
    float_frame: DataFrame,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
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
def test_apply_args(
    float_frame: DataFrame,
    axis: int,
    raw: bool,
    engine: str,
    nopython: bool
) -> None:
    numba = pytest.importorskip('numba')
    if (
        engine == 'numba'
        and Version(numba.__version__) == Version('0.61')
        and is_platform_arm()
    ):
        pytest.skip(f'Segfaults on ARM platforms with numba {numba.__version__}')
    engine_kwargs: Dict[str, Any] = {'nopython': nopython}
    result = float_frame.apply(
        lambda x, y: x + y, axis, args=(1,), raw=raw, engine=engine, engine_kwargs=engine_kwargs
    )
    expected = float_frame + 1
    tm.assert_frame_equal(result, expected)
    result = float_frame.apply(
        lambda x, a, b: x + a + b, args=(1,), b=2, raw=raw, engine=engine, engine_kwargs=engine_kwargs
    )
    expected = float_frame + 3
    tm.assert_frame_equal(result, expected)
    if engine == 'numba':
        with pytest.raises(TypeError, match="missing a required argument: 'a'"):
            float_frame.apply(
                lambda x, a: x + a,
                b=2,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs
            )
        with pytest.raises(
            pd.errors.NumbaUtilError,
            match='numba does not support keyword-only arguments'
        ):
            float_frame.apply(
                lambda x, a, *, b: x + a + b,
                args=(1,),
                b=2,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs
            )
        with pytest.raises(
            pd.errors.NumbaUtilError,
            match='numba does not support keyword-only arguments'
        ):
            float_frame.apply(
                lambda *x, b: x[0] + x[1] + b,
                args=(1,),
                b=2,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs
            )


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


@pytest.mark.parametrize(
    'data, dtype',
    [
        (1, None),
        (1, CategoricalDtype([1])),
        (Timestamp('2013-01-01', tz='UTC'), None)
    ]
)
def test_agg_axis1_duplicate_index(
    data: Union[int, Timestamp],
    dtype: Optional[CategoricalDtype]
) -> None:
    expected = DataFrame([[data], [data]], index=['a', 'a'], dtype=dtype)
    result = expected.agg(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


def test_apply_mixed_datetimelike() -> None:
    expected = DataFrame({
        'A': date_range('20130101', periods=3),
        'B': pd.to_timedelta(np.arange(3), unit='s')
    })
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('func', [np.sqrt, np.mean])
def test_apply_empty(
    func: Callable[[Any], Any],
    engine: str
) -> None:
    empty_frame = DataFrame()
    result = empty_frame.apply(func, engine=engine)
    assert result.empty


def test_apply_float_frame(
    float_frame: DataFrame,
    engine: str
) -> None:
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
    x: List[Any] = []
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
    df = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=['a', 'a', 'c']
    )
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
    result = float_frame.apply(
        lambda x: list(range(len(float_frame.columns))),
        axis=1,
        result_type='broadcast'
    )
    m = list(range(len(float_frame.columns)))
    expected = DataFrame(
        [m] * len(float_frame.index),
        dtype='float64',
        index=float_frame.index,
        columns=float_frame.columns
    )
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_lists_index(float_frame: DataFrame) -> None:
    result = float_frame.apply(
        lambda x: list(range(len(float_frame.index))),
        result_type='broadcast'
    )
    m = list(range(len(float_frame.index)))
    expected = DataFrame(
        {c: m for c in float_frame.columns},
        dtype='float64',
        index=float_frame.index
    )
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_list_lambda_func(int_frame_const_col: DataFrame) -> None:
    df = int_frame_const_col
    result = df.apply(
        lambda x: [1, 2, 3], axis=1, result_type='broadcast'
    )
    tm.assert_frame_equal(result, df)


def test_apply_broadcast_series_lambda_func(int_frame_const_col: DataFrame) -> None:
    df = int_frame_const_col
    result = df.apply(
        lambda x: Series([1, 2, 3], index=list('abc')),
        axis=1,
        result_type='broadcast'
    )
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_float_frame(
    float_frame: DataFrame,
    axis: int,
    engine: str
) -> None:
    if engine == 'numba':
        pytest.skip("numba can't handle when UDF returns None.")

    def _assert_raw(x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1

    float_frame.apply(_assert_raw, axis=axis, engine=engine, raw=True)


@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_float_frame_lambda(
    float_frame: DataFrame,
    axis: int,
    engine: str
) -> None:
    result = float_frame.apply(
        np.mean, axis=axis, engine=engine, raw=True
    )
    expected = float_frame.apply(
        lambda x: x.values.mean(), axis=axis
    )
    tm.assert_series_equal(result, expected)


def test_apply_raw_float_frame_no_reduction(
    float_frame: DataFrame,
    engine: str
) -> None:
    result = float_frame.apply(
        lambda x: x * 2, engine=engine, raw=True
    )
    expected = float_frame * 2
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('axis', [0, 1])
def test_apply_raw_mixed_type_frame(
    axis: int,
    engine: str
) -> None:
    if engine == 'numba':
        pytest.skip("isinstance check doesn't work with numba")

    def _assert_raw(x: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1

    df = DataFrame({
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
@pytest.mark.parametrize(
    'func',
    [lambda x: x, lambda x: x.mean()],
    ids=['identity', 'mean']
)
@pytest.mark.parametrize('raw', [True, False])
@pytest.mark.parametrize('axis', [0, 1])
def test_apply_empty_infer_type(
    ax: str,
    func: Callable[[Any], Any],
    raw: bool,
    axis: int,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    df = DataFrame(**{ax: ['a', 'b', 'c']})
    with np.errstate(all='ignore'):
        test_res = func(np.array([], dtype='f8'))
        is_reduction = not isinstance(test_res, np.ndarray)
        result = df.apply(
            func, axis=axis, engine=engine, raw=raw
        )
        if is_reduction:
            agg_axis = df._get_agg_axis(axis)
            assert isinstance(result, Series)
            assert result.index is agg_axis
        else:
            assert isinstance(result, DataFrame)


def test_apply_empty_infer_type_broadcast() -> None:
    no_cols = DataFrame(index=['a', 'b', 'c'])
    result = no_cols.apply(
        lambda x: x.mean(), result_type='broadcast'
    )
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

    def subtract_and_divide(x: Series, sub: float, divide: float = 1) -> Series:
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
    data = DataFrame(
        [[1, 2], [3, 4]],
        columns=['c0', 'c1'],
        index=['i0', 'i1']
    )
    result = data.apply(dict, axis=0)
    expected = Series(
        [{'i0': 1, 'i1': 3}, {'i0': 2, 'i1': 4}],
        index=data.columns
    )
    tm.assert_series_equal(result, expected)
    result = data.apply(dict, axis=1)
    expected = Series(
        [{'c0': 1, 'c1': 2}, {'c0': 3, 'c1': 4}],
        index=data.index
    )
    tm.assert_series_equal(result, expected)


def test_apply_differently_indexed() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((20, 10))
    )
    result = df.apply(Series.describe, axis=0)
    expected = DataFrame(
        {i: v.describe() for i, v in df.items()},
        columns=df.columns
    )
    tm.assert_frame_equal(result, expected)
    result = df.apply(Series.describe, axis=1)
    expected = DataFrame(
        {i: v.describe() for i, v in df.T.items()},
        columns=df.index
    ).T
    tm.assert_frame_equal(result, expected)


def test_apply_bug() -> None:
    positions = DataFrame(
        [
            [1, 'ABC0', 50],
            [1, 'YUM0', 20],
            [1, 'DEF0', 20],
            [2, 'ABC1', 50],
            [2, 'YUM1', 20],
            [2, 'DEF1', 20]
        ],
        columns=['a', 'market', 'position']
    )

    def f(r: Series) -> Any:
        return r['market']

    expected = positions.apply(f, axis=1)
    positions = DataFrame(
        [
            [datetime(2013, 1, 1), 'ABC0', 50],
            [datetime(2013, 1, 2), 'YUM0', 20],
            [datetime(2013, 1, 3), 'DEF0', 20],
            [datetime(2013, 1, 4), 'ABC1', 50],
            [datetime(2013, 1, 5), 'YUM1', 20],
            [datetime(2013, 1, 6), 'DEF1', 20]
        ],
        columns=['a', 'market', 'position']
    )
    result = positions.apply(f, axis=1)
    tm.assert_series_equal(result, expected)


def test_apply_convert_objects() -> None:
    expected = DataFrame({
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
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)), axis=1)
    expected = DataFrame(
        np.tile(float_frame.columns, (len(float_frame.index), 1)),
        index=float_frame.index,
        columns=float_frame.columns
    )
    tm.assert_frame_equal(result, expected)


def test_apply_attach_name_non_reduction_axis1(float_frame: DataFrame) -> None:
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)), axis=1)
    expected = Series(
        (np.repeat(t[0], len(float_frame.columns)) for t in float_frame.itertuples()),
        index=float_frame.index
    )
    tm.assert_series_equal(result, expected)


def test_apply_multi_index() -> None:
    index = MultiIndex.from_arrays([['a', 'a', 'b'], ['c', 'd', 'd']])
    s = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=index,
        columns=['col1', 'col2']
    )
    result = s.apply(
        lambda x: Series({'min': min(x), 'max': max(x)}),
        1
    )
    expected = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=index,
        columns=['min', 'max']
    )
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    'df, dicts',
    [
        (
            DataFrame([['foo', 'bar'], ['spam', 'eggs']]),
            Series([
                {0: 'foo', 1: 'spam'},
                {0: 'bar', 1: 'eggs'}
            ])
        ),
        (
            DataFrame([[0, 1], [2, 3]]),
            Series([
                {0: 0, 1: 2},
                {0: 1, 1: 3}
            ])
        )
    ]
)
def test_apply_dict(
    df: DataFrame,
    dicts: Series
) -> None:
    fn: Callable[[Series], Dict[Any, Any]] = lambda x: x.to_dict()
    reduce_true = df.apply(fn, result_type='reduce')
    reduce_false = df.apply(fn, result_type='expand')
    reduce_none = df.apply(fn)
    tm.assert_series_equal(reduce_true, dicts)
    tm.assert_frame_equal(reduce_false, df)
    tm.assert_series_equal(reduce_none, dicts)


def test_apply_non_numpy_dtype() -> None:
    df = DataFrame({'dt': date_range('2015-01-01', periods=3, tz='Europe/Brussels')})
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)
    result = df.apply(lambda x: x + pd.Timedelta('1day'))
    expected = DataFrame({'dt': date_range('2015-01-02', periods=3, tz='Europe/Brussels')})
    tm.assert_frame_equal(result, expected)


def test_apply_non_numpy_dtype_category() -> None:
    df = DataFrame({'dt': ['a', 'b', 'c', 'a']}, dtype='category')
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)


def test_apply_dup_names_multi_agg() -> None:
    df = DataFrame([[0, 1], [2, 3]], columns=['a', 'a'])
    expected = DataFrame([[0, 1]], columns=['a', 'a'], index=['min'])
    result = df.agg(['min'])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('op', ['apply', 'agg'])
def test_apply_nested_result_axis_1(op: str) -> None:

    def apply_list(row: Series) -> List[int]:
        return [2 * row['A'], 2 * row['C'], 2 * row['B']]

    df = DataFrame(
        np.zeros((4, 4)),
        columns=list('ABCD')
    )
    result = getattr(df, op)(apply_list, axis=1)
    expected = Series([[0.0, 0.0, 0.0]] * 4)
    tm.assert_series_equal(result, expected)


def test_apply_noreduction_tzaware_object() -> None:
    expected = DataFrame({'foo': [Timestamp('2020', tz='UTC')]}, dtype='datetime64[ns, UTC]')
    result = expected.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    result = expected.apply(lambda x: x.copy())
    tm.assert_frame_equal(result, expected)


def test_apply_function_runs_once() -> None:
    df = DataFrame({'a': [1, 2, 3]})
    names: List[Any] = []

    def reducing_function(row: Series) -> None:
        names.append(row.name)

    def non_reducing_function(row: Series) -> Series:
        names.append(row.name)
        return row

    for func in [reducing_function, non_reducing_function]:
        del names[:]
        df.apply(func, axis=1)
        assert names == list(df.index)


def test_apply_raw_function_runs_once(engine: str) -> None:
    if engine == 'numba':
        pytest.skip('appending to list outside of numba func is not supported')
    df = DataFrame({'a': [1, 2, 3]})
    values: List[Any] = []

    def reducing_function(row: np.ndarray) -> None:
        values.extend(row)

    def non_reducing_function(row: np.ndarray) -> np.ndarray:
        values.extend(row)
        return row

    for func in [reducing_function, non_reducing_function]:
        del values[:]
        df.apply(
            func, engine=engine, raw=True, axis=1
        )
        assert values == list(df.a.to_list())


def test_apply_with_byte_string() -> None:
    df = DataFrame(np.array([b'abcd', b'efgh']), columns=['col'])
    expected = DataFrame(
        np.array([b'abcd', b'efgh']),
        columns=['col'],
        dtype=object
    )
    result = df.apply(lambda x: x.astype('object'))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('val', ['asd', 12, None, np.nan])
def test_apply_category_equalness(val: Union[str, int, None, float]) -> None:
    df_values: List[Union[str, int, None, float]] = ['asd', None, 12, 'asd', 'cde', np.nan]
    df = DataFrame({'a': df_values}, dtype='category')
    result = df.a.apply(lambda x: x == val)
    expected = Series(
        [False if pd.isnull(x) else x == val for x in df_values],
        name='a'
    )
    tm.assert_series_equal(result, expected)


def test_infer_row_shape() -> None:
    df = DataFrame(
        np.random.default_rng(2).random((10, 2))
    )
    result = df.apply(np.fft.fft, axis=0).shape
    assert result == (10, 2)
    result = df.apply(np.fft.rfft, axis=0).shape
    assert result == (6, 2)


@pytest.mark.parametrize(
    'ops, by_row, expected',
    [
        (
            {'a': lambda x: x + 1},
            'compat',
            DataFrame({'a': [2, 3]})
        ),
        (
            {'a': lambda x: x + 1},
            False,
            DataFrame({'a': [2, 3]})
        ),
        (
            {'a': lambda x: x.sum()},
            'compat',
            Series({'a': 3})
        ),
        (
            {'a': lambda x: x.sum()},
            False,
            Series({'a': 3})
        ),
        (
            {'a': ['sum', np.sum, lambda x: x.sum()]},
            'compat',
            DataFrame({'a': [3, 3, 3]}, index=['sum', 'sum', '<lambda>'])
        ),
        (
            {'a': ['sum', np.sum, lambda x: x.sum()]},
            False,
            DataFrame({'a': [3, 3, 3]}, index=['sum', 'sum', '<lambda>'])
        ),
        (
            {'a': lambda x: 1},
            'compat',
            DataFrame({'a': [1, 1]})
        ),
        (
            {'a': lambda x: 1},
            False,
            Series({'a': 1})
        )
    ]
)
def test_dictlike_lambda(
    ops: Dict[str, Union[Callable[[Series], Any], List[Callable[[Series], Any]]]],
    by_row: Union[str, bool],
    expected: Union[DataFrame, Series]
) -> None:
    df = DataFrame({'a': [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    'ops',
    [
        {'a': lambda x: x + 1},
        {'a': lambda x: x.sum()},
        {'a': ['sum', np.sum, lambda x: x.sum()]},
        {'a': lambda x: 1}
    ]
)
def test_dictlike_lambda_raises(ops: Dict[str, Union[Callable[[Series], Any], List[Callable[[Series], Any]]]]) -> None:
    df = DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError, match='by_row=True not allowed'):
        df.apply(ops, by_row=True)


def test_with_dictlike_columns() -> None:
    df = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    result = df.apply(
        lambda x: {'s': x['a'] + x['b']},
        axis=1
    )
    expected = Series([{'s': 3}, {'s': 3}])
    tm.assert_series_equal(result, expected)
    df['tm'] = [
        Timestamp('2017-05-01 00:00:00'),
        Timestamp('2017-05-02 00:00:00')
    ]
    result = df.apply(
        lambda x: {'s': x['a'] + x['b']},
        axis=1
    )
    tm.assert_series_equal(result, expected)
    result = (df['a'] + df['b']).apply(
        lambda x: {'s': x}
    )
    expected = Series([{'s': 3}, {'s': 3}])
    tm.assert_series_equal(result, expected)


def test_with_dictlike_columns_with_datetime() -> None:
    df = DataFrame()
    df['author'] = ['X', 'Y', 'Z']
    df['publisher'] = ['BBC', 'NBC', 'N24']
    df['date'] = pd.to_datetime([
        '17-10-2010 07:15:30',
        '13-05-2011 08:20:35',
        '15-01-2013 09:09:09'
    ], dayfirst=True)
    result = df.apply(lambda x: {}, axis=1)
    expected = Series([{}, {}, {}])
    tm.assert_series_equal(result, expected)


def test_with_dictlike_columns_with_infer() -> None:
    df = DataFrame([[1, 2], [1, 2]], columns=['a', 'b'])
    result = df.apply(
        lambda x: {'s': x['a'] + x['b']},
        axis=1,
        result_type='expand'
    )
    expected = DataFrame({'s': [3, 3]})
    tm.assert_frame_equal(result, expected)
    df['tm'] = [
        Timestamp('2017-05-01 00:00:00'),
        Timestamp('2017-05-02 00:00:00')
    ]
    result = df.apply(
        lambda x: {'s': x['a'] + x['b']},
        axis=1,
        result_type='expand'
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'ops, by_row, expected',
    [
        (
            [lambda x: x + 1],
            'compat',
            DataFrame({('a', '<lambda>'): [2, 3]})
        ),
        (
            [lambda x: x + 1],
            False,
            DataFrame({('a', '<lambda>'): [2, 3]})
        ),
        (
            [lambda x: x.sum()],
            'compat',
            DataFrame({'a': [3]}, index=['<lambda>'])
        ),
        (
            [lambda x: x.sum()],
            False,
            DataFrame({'a': [3]}, index=['<lambda>'])
        ),
        (
            ['sum', np.sum, lambda x: x.sum()],
            'compat',
            DataFrame(
                {'a': [3, 3, 3]},
                index=['sum', 'sum', '<lambda>']
            )
        ),
        (
            ['sum', np.sum, lambda x: x.sum()],
            False,
            DataFrame(
                {'a': [3, 3, 3]},
                index=['sum', 'sum', '<lambda>']
            )
        ),
        (
            [lambda x: x + 1, lambda x: 3],
            'compat',
            DataFrame(
                [[2, 3], [3, 3]],
                columns=[['a', 'a'], ['<lambda>', '<lambda>']]
            )
        ),
        (
            [lambda x: 2, lambda x: 3],
            False,
            DataFrame({'a': [2, 3]}, ['<lambda>', '<lambda>'])
        )
    ]
)
def test_listlike_lambda(
    ops: List[Callable[[Series], Any]],
    by_row: Union[str, bool],
    expected: Union[DataFrame, Series]
) -> None:
    df = DataFrame({'a': [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    'ops',
    [
        [lambda x: x + 1],
        [lambda x: x.sum()],
        ['sum', np.sum, lambda x: x.sum()],
        [lambda x: x + 1, lambda x: 3]
    ]
)
def test_listlike_lambda_raises(ops: List[Union[Callable[[Series], Any], str]]) -> None:
    df = DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError, match='by_row=True not allowed'):
        df.apply(ops, by_row=True)


def test_with_listlike_columns() -> None:
    df = DataFrame({
        'a': Series(np.random.default_rng(2).standard_normal(4)),
        'b': ['a', 'list', 'of', 'words'],
        'ts': date_range('2016-10-01', periods=4, freq='h')
    })
    result = df[['a', 'b']].apply(tuple, axis=1)
    expected = Series([t[1:] for t in df[['a', 'b']].itertuples()])
    tm.assert_series_equal(result, expected)
    result = df[['a', 'ts']].apply(tuple, axis=1)
    expected = Series([t[1:] for t in df[['a', 'ts']].itertuples()])
    tm.assert_series_equal(result, expected)


def test_with_listlike_columns_returning_list() -> None:
    df = DataFrame({
        'x': Series([['a', 'b'], ['q']]),
        'y': Series([['z'], ['q', 't']])
    })
    df.index = MultiIndex.from_tuples([('i0', 'j0'), ('i1', 'j1')])
    result = df.apply(
        lambda row: [el for el in row['x'] if el in row['y']],
        axis=1
    )
    expected = Series([[], ['q']], index=df.index)
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_columns() -> None:
    df = DataFrame({
        'number': [1.0, 2.0],
        'string': ['foo', 'bar'],
        'datetime': [
            Timestamp('2017-11-29 03:30:00'),
            Timestamp('2017-11-29 03:45:00')
        ]
    })
    result = df.apply(
        lambda row: (row.number, row.string),
        axis=1
    )
    expected = Series([
        (t.number, t.string) for t in df.itertuples()
    ])
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_listlike_columns() -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((6, 3)),
        columns=['A', 'B', 'C']
    )
    result = df.apply(
        lambda x: [1, 2, 3],
        axis=1
    )
    expected = Series([[1, 2, 3] for _ in df.itertuples()], index=df.index)
    tm.assert_series_equal(result, expected)
    result = df.apply(
        lambda x: [1, 2],
        axis=1
    )
    expected = Series([[1, 2] for _ in df.itertuples()], index=df.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('val', [1, 2])
def test_infer_output_shape_listlike_columns_np_func(val: int) -> None:
    df = DataFrame({'a': [1, 2, 3]}, index=list('abc'))
    result = df.apply(
        lambda row: np.ones(val),
        axis=1
    )
    expected = Series(
        [np.ones(val) for _ in df.itertuples()],
        index=df.index
    )
    tm.assert_series_equal(result, expected)


def test_infer_output_shape_listlike_columns_with_timestamp() -> None:
    df = DataFrame({
        'a': [
            Timestamp('2010-02-01'),
            Timestamp('2010-02-04'),
            Timestamp('2010-02-05'),
            Timestamp('2010-02-06')
        ],
        'b': [9, 5, 4, 3],
        'c': [5, 3, 4, 2],
        'd': [1, 2, 3, 4]
    })

    def fun(x: Series) -> Tuple[int, int]:
        return (1, 2)

    result = df.apply(fun, axis=1)
    expected = Series([(1, 2) for _ in df.itertuples()], index=df.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('lst', [[1, 2, 3], [1, 2]])
def test_consistent_coerce_for_shapes(lst: List[int]) -> None:
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)),
        columns=['A', 'B', 'C']
    )
    result = df.apply(
        lambda x: lst,
        axis=1
    )
    expected = Series([lst for _ in df.itertuples()], index=df.index)
    tm.assert_series_equal(result, expected)


def test_consistent_names(int_frame_const_col: DataFrame) -> None:
    df = int_frame_const_col
    result = df.apply(
        lambda x: Series([1, 2, 3], index=['test', 'other', 'cols']),
        axis=1
    )
    expected = int_frame_const_col.rename(columns={'A': 'test', 'B': 'other', 'C': 'cols'})
    tm.assert_frame_equal(result, expected)
    result = df.apply(
        lambda x: Series([1, 2], index=['test', 'other']),
        axis=1
    )
    expected = expected[['test', 'other']]
    tm.assert_frame_equal(result, expected)


def test_result_type(int_frame_const_col: DataFrame) -> None:
    df = int_frame_const_col
    result = df.apply(
        lambda x: [1, 2, 3],
        axis=1,
        result_type='expand'
    )
    expected = df.copy()
    expected.columns = range(3)
    tm.assert_frame_equal(result, expected)


def test_result_type_shorter_list(int_frame_const_col: DataFrame) -> None:
    df = int_frame_const_col
    result = df.apply(
        lambda x: [1, 2],
        axis=1,
        result_type='expand'
    )
    expected = df[['A', 'B']].copy()
    expected.columns = range(2)
    tm.assert_frame_equal(result, expected)


def test_result_type_broadcast(
    int_frame_const_col: DataFrame,
    request: pytest.FixtureRequest,
    engine: str
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason="numba engine doesn't support list return")
        request.node.add_marker(mark)
    df = int_frame_const_col
    result = df.apply(
        lambda x: [1, 2, 3],
        axis=1,
        result_type='broadcast',
        engine=engine
    )
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_broadcast_series_func(
    int_frame_const_col: DataFrame,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason='numba Series constructor only support ndarrays not list data'
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    columns = ['other', 'col', 'names']
    result = df.apply(
        lambda x: Series([1, 2, 3], index=columns),
        axis=1,
        result_type='broadcast',
        engine=engine
    )
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_series_result(
    int_frame_const_col: DataFrame,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason='numba Series constructor only support ndarrays not list data'
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    result = df.apply(
        lambda x: Series([1, 2, 3], index=x.index),
        axis=1,
        engine=engine
    )
    expected = df.copy()
    tm.assert_frame_equal(result, expected)


def test_result_type_series_result_other_index(
    int_frame_const_col: DataFrame,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason='no support in numba Series constructor for list of columns'
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    columns = ['other', 'col', 'names']
    result = df.apply(
        lambda x: Series([1, 2, 3], index=columns),
        axis=1,
        engine=engine
    )
    expected = df.copy()
    expected.columns = columns
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'box',
    [
        lambda x: list(x),
        lambda x: tuple(x),
        lambda x: np.array(x, dtype='int64')
    ],
    ids=['list', 'tuple', 'array']
)
def test_consistency_for_boxed(
    box: Callable[[List[int]], Any],
    int_frame_const_col: DataFrame
) -> None:
    df = int_frame_const_col
    result = df.apply(
        lambda x: box([1, 2]),
        axis=1
    )
    expected = Series([box([1, 2]) for _ in df.itertuples()], index=df.index)
    tm.assert_series_equal(result, expected)
    result = df.apply(
        lambda x: box([1, 2]),
        axis=1,
        result_type='expand'
    )
    expected = int_frame_const_col[['A', 'B']].rename(columns={'A': 0, 'B': 1})
    tm.assert_frame_equal(result, expected)


def test_agg_transform(float_frame: DataFrame) -> None:
    other_axis: Union[int, str] = 1 if 0 == 0 else 0  # Dummy to satisfy mypy
    with np.errstate(all='ignore'):
        f_abs = np.abs(float_frame)
        f_sqrt = np.sqrt(float_frame)
        expected = f_sqrt.copy()
        result = float_frame.apply(np.sqrt, axis=0)
        tm.assert_frame_equal(result, expected)
        result = float_frame.apply([np.sqrt], axis=0)
        expected = f_sqrt.copy()
        expected.columns = MultiIndex.from_product([float_frame.columns, ['sqrt']])
        tm.assert_frame_equal(result, expected)
        result = float_frame.apply([np.abs, np.sqrt], axis=0)
        expected = zip_frames([f_abs, f_sqrt], axis=1)
        expected.columns = MultiIndex.from_product([float_frame.columns, ['absolute', 'sqrt']])
        tm.assert_frame_equal(result, expected)


def test_demo() -> None:
    df = DataFrame({'A': range(5), 'B': 5})
    result = df.agg(['min', 'max'])
    expected = DataFrame(
        {'A': [0, 4], 'B': [5, 5]},
        columns=['A', 'B'],
        index=['min', 'max']
    )
    tm.assert_frame_equal(result, expected)


def test_demo_dict_agg() -> None:
    df = DataFrame({'A': range(5), 'B': 5})
    result = df.agg({'A': ['min', 'max'], 'B': ['sum', 'max']})
    expected = DataFrame(
        {'A': [4.0, 0.0, np.nan], 'B': [5.0, np.nan, 25.0]},
        columns=['A', 'B'],
        index=['max', 'min', 'sum']
    )
    tm.assert_frame_equal(result.reindex_like(expected), expected)


def test_agg_with_name_as_column_name() -> None:
    data = {'name': ['foo', 'bar']}
    df = DataFrame(data)
    result = df.agg({'name': 'count'})
    expected = Series({'name': 2})
    tm.assert_series_equal(result, expected)
    result = df['name'].agg({'name': 'count'})
    expected = Series({'name': 2}, name='name')
    tm.assert_series_equal(result, expected)


def test_agg_multiple_mixed() -> None:
    mdf = DataFrame({
        'A': [1, 2, 3],
        'B': [1.0, 2.0, 3.0],
        'C': ['foo', 'bar', 'baz']
    })
    expected = DataFrame({
        'A': [1, 6],
        'B': [1.0, 6.0],
        'C': ['bar', 'foobarbaz']
    }, index=['min', 'sum'])
    result = mdf.agg(['min', 'sum'])
    tm.assert_frame_equal(result, expected)
    result = mdf[['C', 'B', 'A']].agg(['sum', 'min'])
    expected = expected[['C', 'B', 'A']].reindex(['sum', 'min'])
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_mixed_raises() -> None:
    mdf = DataFrame({
        'A': [1, 2, 3],
        'B': [1.0, 2.0, 3.0],
        'C': ['foo', 'bar', 'baz'],
        'D': date_range('20130101', periods=3)
    })
    msg = 'does not support operation'
    with pytest.raises(TypeError, match=msg):
        mdf.agg(['min', 'sum'])
    with pytest.raises(TypeError, match=msg):
        mdf[['D', 'C', 'B', 'A']].agg(['sum', 'min'])


def test_agg_reduce(float_frame: DataFrame) -> None:
    other_axis: Union[int, str] = 1 if 0 == 0 else 0  # Dummy to satisfy mypy
    name1, name2 = float_frame.axes[other_axis].unique()[:2].sort_values()
    expected = pd.concat(
        [float_frame.mean(axis=0), float_frame.max(axis=0), float_frame.sum(axis=0)],
        axis=1
    )
    expected.columns = ['mean', 'max', 'sum']
    result = float_frame.agg(['mean', 'max', 'sum'], axis=0)
    tm.assert_frame_equal(result, expected)
    func: Dict[Any, str] = {name1: 'mean', name2: 'sum'}
    result = float_frame.agg(func, axis=0)
    expected = Series(
        [float_frame.loc[other_axis][name1].mean(), float_frame.loc[other_axis][name2].sum()],
        index=[name1, name2]
    )
    tm.assert_series_equal(result, expected)
    func = {name1: ['mean'], name2: ['sum']}
    result = float_frame.agg(func, axis=0)
    expected = DataFrame({
        name1: Series([float_frame.loc[other_axis][name1].mean()], index=['mean']),
        name2: Series([float_frame.loc[other_axis][name2].sum()], index=['sum'])
    })
    tm.assert_frame_equal(result, expected)
    func = {name1: ['mean', 'sum'], name2: ['sum', 'max']}
    result = float_frame.agg(func, axis=0)
    expected = pd.concat({
        name1: Series(
            [float_frame.loc[other_axis][name1].mean(), float_frame.loc[other_axis][name1].sum()],
            index=['mean', 'sum']
        ),
        name2: Series(
            [float_frame.loc[other_axis][name2].sum(), float_frame.loc[other_axis][name2].max()],
            index=['sum', 'max']
        )
    }, axis=1)
    tm.assert_frame_equal(result, expected)


def test_named_agg_reduce_axis1_raises(float_frame: DataFrame) -> None:
    name1, name2 = float_frame.axes[0].unique()[:2].sort_values()
    msg = 'Named aggregation is not supported when axis=1.'
    for axis in [1, 'columns']:
        with pytest.raises(NotImplementedError, match=msg):
            float_frame.agg(row1=(name1, 'sum'), row2=(name2, 'max'), axis=axis)


def test_nuiscance_columns() -> None:
    df = DataFrame({
        'A': [1, 2, 3],
        'B': [1.0, 2.0, 3.0],
        'C': ['foo', 'bar', 'baz'],
        'D': date_range('20130101', periods=3)
    })
    result = df.agg('min')
    expected = Series(
        [1, 1.0, 'bar', Timestamp('20130101')],
        index=df.columns
    )
    tm.assert_series_equal(result, expected)
    result = df.agg(['min'])
    expected = DataFrame(
        [[1, 1.0, 'bar', Timestamp('20130101').as_unit('ns')]],
        index=['min'],
        columns=df.columns
    )
    tm.assert_frame_equal(result, expected)
    msg = 'does not support operation'
    with pytest.raises(TypeError, match=msg):
        df.agg('sum')
    result = df[['A', 'B', 'C']].agg('sum')
    expected = Series(
        [6, 6.0, 'foobarbaz'],
        index=['A', 'B', 'C']
    )
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match=msg):
        df.agg(['sum'])


@pytest.mark.parametrize(
    'how',
    ['agg', 'apply']
)
def test_non_callable_aggregates(
    how: str
) -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    result = getattr(df, how)({'A': 'count'})
    expected = Series({'A': 2})
    tm.assert_series_equal(result, expected)
    result = getattr(df, how)({'A': 'size'})
    expected = Series({'A': 3})
    tm.assert_series_equal(result, expected)
    result1 = getattr(df, how)(['count', 'size'])
    result2 = getattr(df, how)(
        {'A': ['count', 'size'], 'B': ['count', 'size'], 'C': ['count', 'size']}
    )
    expected = DataFrame({
        'A': {'count': 2, 'size': 3},
        'B': {'count': 2, 'size': 3},
        'C': {'count': 2, 'size': 3}
    })
    tm.assert_frame_equal(result1, result2, check_like=True)
    tm.assert_frame_equal(result2, expected, check_like=True)
    result = getattr(df, how)('count')
    expected = df.count()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'how',
    ['agg', 'apply']
)
def test_size_as_str(
    how: str,
    axis: int
) -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    result = getattr(df, how)('size', axis=axis)
    if axis in (0, 'index'):
        expected = Series(df.shape[0], index=df.columns)
    else:
        expected = Series(df.shape[1], index=df.index)
    tm.assert_series_equal(result, expected)


def test_agg_listlike_result() -> None:
    df = DataFrame({
        'A': [2, 2, 3],
        'B': [1.5, np.nan, 1.5],
        'C': ['foo', None, 'bar']
    })

    def func(group_col: Series) -> List[Any]:
        return list(group_col.dropna().unique())

    result = df.agg(func)
    expected = Series(
        [[2, 3], [1.5], ['foo', 'bar']],
        index=['A', 'B', 'C']
    )
    tm.assert_series_equal(result, expected)
    result = df.agg([func])
    expected = expected.to_frame('func').T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'axis, args, kwargs',
    [
        (0, (1, 2, 3), {}),
        (0, (8, 7, 15), {}),
        (0, (1, 2), {}),
        (0, (1,), {'b': 2}),
        (0, (), {'a': 1, 'b': 2}),
        (0, (), {'a': 2, 'b': 1}),
        (0, (), {'a': 1, 'b': 2, 'c': 3})
    ]
)
def test_agg_args_kwargs(
    axis: int,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> None:

    def f(x: Series, a: int, b: int, c: int = 3) -> float:
        return x.sum() + (a + b) / c

    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        expected = Series([5.0, 7.0])
    else:
        expected = Series([4.0, 8.0])
    result = df.agg(f, axis, *args, **kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'num_cols',
    [2, 3, 5]
)
def test_frequency_is_original(
    num_cols: int,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason='numba engine only supports numeric indices')
        request.node.add_marker(mark)
    index = pd.DatetimeIndex(['1950-06-30', '1952-10-24', '1953-05-29'])
    original = index.copy()
    df = DataFrame(1, index=index, columns=range(num_cols))
    df.apply(lambda x: x, engine=engine)
    assert index.freq == original.freq


def test_apply_datetime_tz_issue(
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support non-numeric indexes"
        )
        request.node.add_marker(mark)
    timestamps = [
        Timestamp('2019-03-15 12:34:31.909000+0000', tz='UTC'),
        Timestamp('2019-03-15 12:34:34.359000+0000', tz='UTC'),
        Timestamp('2019-03-15 12:34:34.660000+0000', tz='UTC')
    ]
    df = DataFrame(data=[0, 1, 2], index=timestamps)
    result = df.apply(
        lambda x: x.name,
        axis=1,
        engine=engine
    )
    expected = Series(timestamps, index=timestamps)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'df, method',
    [
        (DataFrame({'A': ['a', None], 'B': ['c', 'd']}), 'min'),
        (DataFrame({'A': ['a', None], 'B': ['c', 'd']}), 'max'),
        (DataFrame({'A': ['a', None], 'B': ['c', 'd']}), 'sum')
    ]
)
def test_mixed_column_raises(
    df: DataFrame,
    method: str,
    using_infer_string: bool
) -> None:
    if method == 'sum':
        msg = 'can only concatenate str \\(not "int"\\) to str|does not support'
    else:
        msg = "not supported between instances of 'str' and 'float'"
    if not using_infer_string:
        with pytest.raises(TypeError, match=msg):
            getattr(df, method)()
    else:
        getattr(df, method)()


@pytest.mark.parametrize(
    'col',
    [1, 1.0, True, 'a', np.nan]
)
def test_apply_dtype(
    col: Union[int, float, bool, str, None]
) -> None:
    df = DataFrame([[1.0, col]], columns=['a', 'b'])
    result = df.apply(lambda x: x.dtype)
    expected = df.dtypes
    tm.assert_series_equal(result, expected)


def test_apply_mutating() -> None:
    df = DataFrame({'a': range(10), 'b': range(10, 20)})
    df_orig = df.copy()

    def func(row: Series) -> Series:
        mgr = row._mgr
        row.loc['a'] += 1
        assert row._mgr is not mgr
        return row

    expected = df.copy()
    expected['a'] += 1
    result = df.apply(func, axis=1)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, df_orig)


def test_apply_empty_list_reduce() -> None:
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        columns=['a', 'b']
    )
    result = df.apply(
        lambda x: [],
        result_type='reduce'
    )
    expected = Series({'a': [], 'b': []}, dtype=object)
    tm.assert_series_equal(result, expected)


def test_apply_no_suffix_index(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support list-likes/dict-like callables"
        )
        request.node.add_marker(mark)
    pdf = DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    result = pdf.apply(
        ['sum', lambda x: x.sum(), lambda x: x.sum()],
        engine=engine
    )
    expected = DataFrame(
        {'A': [12, 12, 12], 'B': [27, 27, 27]},
        index=['sum', '<lambda>', '<lambda>']
    )
    tm.assert_frame_equal(result, expected)


def test_apply_raw_returns_string(engine: str) -> None:
    if engine == 'numba':
        pytest.skip('No object dtype support in numba')
    df = DataFrame({'A': ['aa', 'bbb']})
    result = df.apply(
        lambda x: x[0],
        engine=engine,
        axis=1,
        raw=True
    )
    expected = Series(['aa', 'bbb'])
    tm.assert_series_equal(result, expected)


def test_aggregation_func_column_order() -> None:
    df = DataFrame(
        [
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 5, 4),
            (5, 6, 6),
            (6, 7, 7)
        ],
        columns=('att1', 'att2', 'att3')
    )

    def sum_div2(s: Series) -> float:
        return s.sum() / 2

    aggs = ['sum', sum_div2, 'count', 'min']
    result = df.agg(aggs)
    expected = DataFrame({
        'att1': [21.0, 10.5, 6.0, 1.0],
        'att2': [18.0, 9.0, 6.0, 0.0],
        'att3': [17.0, 8.5, 6.0, 0.0]
    }, index=['sum', 'sum_div2', 'count', 'min'])
    tm.assert_frame_equal(result, expected)


def test_apply_getitem_axis_1(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason='numba engine not supporting duplicate index values'
        )
        request.node.add_marker(mark)
    df = DataFrame({'a': [0, 1, 2], 'b': [1, 2, 3]})
    result = df[['a', 'a']].apply(
        lambda x: x.iloc[0] + x.iloc[1],
        axis=1,
        engine=engine
    )
    expected = Series([0, 2, 4])
    tm.assert_series_equal(result, expected)


def test_nuisance_depr_passes_through_warnings() -> None:

    def expected_warning(x: Series) -> float:
        warnings.warn('Hello, World!', UserWarning)
        return x.sum()

    df = DataFrame({'a': [1, 2, 3]})
    with tm.assert_produces_warning(UserWarning, match='Hello, World!'):
        df.agg([expected_warning])


def test_apply_type() -> None:
    df = DataFrame({
        'col1': [3, 'string', float],
        'col2': [
            0.25,
            datetime(2020, 1, 1),
            np.nan
        ]
    }, index=['a', 'b', 'c'])
    result = df.apply(type, axis=0)
    expected = Series({'col1': Series, 'col2': Series})
    tm.assert_series_equal(result, expected)
    result = df.apply(type, axis=1)
    expected = Series({'a': Series, 'b': Series, 'c': Series})
    tm.assert_series_equal(result, expected)


def test_apply_on_empty_dataframe(engine: str) -> None:
    df = DataFrame({'a': [1, 2], 'b': [3, 0]})
    result = df.head(0).apply(
        lambda x: max(x['a'], x['b']),
        axis=1,
        engine=engine
    )
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)


def test_apply_return_list() -> None:
    df = DataFrame({'a': [1, 2], 'b': [2, 3]})
    result = df.apply(lambda x: [x.values], axis=0)
    expected = DataFrame({'a': [[1, 2]], 'b': [[2, 3]]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'test, constant',
    [
        (
            {'a': [1, 2, 3], 'b': [1, 1, 1]},
            {'a': [1, 2, 3], 'b': [1]}
        ),
        (
            {'a': [2, 2, 2], 'b': [1, 1, 1]},
            {'a': [2], 'b': [1]}
        )
    ]
)
def test_unique_agg_type_is_series(
    test: Dict[str, List[Union[int]]],
    constant: Dict[str, List[Union[int]]]
) -> None:
    df1 = DataFrame(test)
    expected = Series(data=constant, index=['a', 'b'], dtype='object')
    aggregation = {'a': 'unique', 'b': 'unique'}
    result = df1.agg(aggregation)
    tm.assert_series_equal(result, expected)


def test_any_apply_keyword_non_zero_axis_regression() -> None:
    df = DataFrame({
        'A': [1, 2, 0],
        'B': [0, 2, 0],
        'C': [0, 0, 0]
    })
    expected = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)
    result = df.apply('any', axis=1)
    tm.assert_series_equal(result, expected)
    result = df.apply('any', 1)
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated() -> None:
    df = DataFrame({'x': [1, 2, 3]})

    def foo1(x: Series, a: int = 1, c: int = 0) -> Series:
        return x + a + c

    def foo2(x: Series, b: int = 2, c: int = 0) -> Series:
        return x + b + c

    result = df.agg(foo1, 0, 3, c=4)
    expected = df + 7
    tm.assert_frame_equal(result, expected)
    result = df.agg([foo1, foo2], 0, 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        columns=[['x', 'x'], ['foo1', 'foo2']]
    )
    tm.assert_frame_equal(result, expected)
    result = df.agg({'x': foo1}, 0, 3, c=4)
    expected = DataFrame([2, 3, 4], columns=['x'])
    tm.assert_frame_equal(result, expected)


def test_agg_std() -> None:
    df = DataFrame(
        np.arange(6).reshape(3, 2),
        columns=['A', 'B']
    )
    result = df.agg(np.std, ddof=1)
    expected = Series({'A': 2.0, 'B': 2.0}, dtype=float)
    tm.assert_series_equal(result, expected)
    result = df.agg([np.std], ddof=1)
    expected = DataFrame(
        {'A': 2.0, 'B': 2.0},
        index=['std']
    )
    tm.assert_frame_equal(result, expected)


def test_agg_dist_like_and_nonunique_columns() -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    df.columns = ['A', 'A', 'C']
    result = df.agg({'A': 'count'})
    expected = df['A'].count()
    tm.assert_series_equal(result, expected)


def test_any_apply_keyword_non_zero_axis_regression() -> None:
    df = DataFrame({
        'A': [1, 2, 0],
        'B': [0, 2, 0],
        'C': [0, 0, 0]
    })
    expected = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)
    result = df.apply('any', axis=1)
    tm.assert_series_equal(result, expected)
    result = df.apply('any', 1)
    tm.assert_series_equal(result, expected)


def test_agg_reduce(float_frame: DataFrame) -> None:
    other_axis: Union[int, str] = 1 if 0 == 0 else 0  # Dummy to satisfy mypy
    name1, name2 = float_frame.axes[other_axis].unique()[:2].sort_values()
    expected = pd.concat(
        [float_frame.mean(axis=axis), float_frame.max(axis=axis), float_frame.sum(axis=axis)],
        axis=1
    )
    expected.columns = ['mean', 'max', 'sum']
    expected = expected.T if axis == 0 else expected
    result = float_frame.agg(['mean', 'max', 'sum'], axis=0)
    tm.assert_frame_equal(result, expected)
    func = {name1: 'mean', name2: 'sum'}
    result = float_frame.agg(func, axis=0)
    expected = Series(
        [float_frame.loc[other_axis][name1].mean(), float_frame.loc[other_axis][name2].sum()],
        index=[name1, name2]
    )
    tm.assert_series_equal(result, expected)
    func = {name1: ['mean'], name2: ['sum']}
    result = float_frame.agg(func, axis=0)
    expected = DataFrame({
        name1: Series([float_frame.loc[other_axis][name1].mean()], index=['mean']),
        name2: Series([float_frame.loc[other_axis][name2].sum()], index=['sum'])
    })
    expected = expected.T if axis == 1 else expected
    tm.assert_frame_equal(result, expected)
    func = {name1: ['mean', 'sum'], name2: ['sum', 'max']}
    result = float_frame.agg(func, axis=0)
    expected = pd.concat({
        name1: Series(
            [float_frame.loc[other_axis][name1].mean(), float_frame.loc[other_axis][name1].sum()],
            index=['mean', 'sum']
        ),
        name2: Series(
            [float_frame.loc[other_axis][name2].sum(), float_frame.loc[other_axis][name2].max()],
            index=['sum', 'max']
        )
    }, axis=1)
    tm.assert_frame_equal(result, expected)


def test_named_agg_reduce_axis1_raises(float_frame: DataFrame) -> None:
    name1, name2 = float_frame.axes[0].unique()[:2].sort_values()
    msg = 'Named aggregation is not supported when axis=1.'
    for axis in [1, 'columns']:
        with pytest.raises(NotImplementedError, match=msg):
            float_frame.agg(
                row1=(name1, 'sum'),
                row2=(name2, 'max'),
                axis=axis
            )


def test_nuiscance_columns() -> None:
    df = DataFrame({
        'A': [1, 2, 3],
        'B': [1.0, 2.0, 3.0],
        'C': ['foo', 'bar', 'baz'],
        'D': date_range('20130101', periods=3)
    })
    result = df.agg('min')
    expected = Series(
        [1, 1.0, 'bar', Timestamp('20130101')],
        index=df.columns
    )
    tm.assert_series_equal(result, expected)
    result = df.agg(['min'])
    expected = DataFrame(
        [[1, 1.0, 'bar', Timestamp('20130101').as_unit('ns')]],
        index=['min'],
        columns=df.columns
    )
    tm.assert_frame_equal(result, expected)
    msg = 'does not support operation'
    with pytest.raises(TypeError, match=msg):
        df.agg('sum')
    result = df[['A', 'B', 'C']].agg('sum')
    expected = Series(
        [6, 6.0, 'foobarbaz'],
        index=['A', 'B', 'C']
    )
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match=msg):
        df.agg(['sum'])


@pytest.mark.parametrize(
    'how',
    ['agg', 'apply']
)
def test_non_callable_aggregates(
    how: str
) -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    result = getattr(df, how)({'A': 'count'})
    expected = Series({'A': 2})
    tm.assert_series_equal(result, expected)
    result = getattr(df, how)({'A': 'size'})
    expected = Series({'A': 3})
    tm.assert_series_equal(result, expected)
    result1 = getattr(df, how)(['count', 'size'])
    result2 = getattr(df, how)({
        'A': ['count', 'size'],
        'B': ['count', 'size'],
        'C': ['count', 'size']
    })
    expected = DataFrame({
        'A': {'count': 2, 'size': 3},
        'B': {'count': 2, 'size': 3},
        'C': {'count': 2, 'size': 3}
    })
    tm.assert_frame_equal(result1, result2, check_like=True)
    tm.assert_frame_equal(result2, expected, check_like=True)
    result = getattr(df, how)('count')
    expected = df.count()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'how',
    ['agg', 'apply']
)
def test_size_as_str(
    how: str,
    axis: int
) -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    result = getattr(df, how)('size', axis=axis)
    if axis in (0, 'index'):
        expected = Series(df.shape[0], index=df.columns)
    else:
        expected = Series(df.shape[1], index=df.index)
    tm.assert_series_equal(result, expected)


def test_agg_listlike_result() -> None:
    df = DataFrame({
        'A': [2, 2, 3],
        'B': [1.5, np.nan, 1.5],
        'C': ['foo', None, 'bar']
    })

    def func(group_col: Series) -> List[Any]:
        return list(group_col.dropna().unique())

    result = df.agg(func)
    expected = Series(
        [[2, 3], [1.5], ['foo', 'bar']],
        index=['A', 'B', 'C']
    )
    tm.assert_series_equal(result, expected)
    result = df.agg([func])
    expected = expected.to_frame('func').T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'axis, args, kwargs',
    [
        (0, (1, 2, 3), {}),
        (0, (8, 7, 15), {}),
        (0, (1, 2), {}),
        (0, (1,), {'b': 2}),
        (0, (), {'a': 1, 'b': 2}),
        (0, (), {'a': 2, 'b': 1}),
        (0, (), {'a': 1, 'b': 2, 'c': 3})
    ]
)
def test_agg_args_kwargs(
    axis: int,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> None:

    def f(x: Series, a: int, b: int, c: int = 3) -> float:
        return x.sum() + (a + b) / c

    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        expected = Series([5.0, 7.0])
    else:
        expected = Series([4.0, 8.0])
    result = df.agg(f, axis, *args, **kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'num_cols',
    [2, 3, 5]
)
def test_frequency_is_original(
    num_cols: int,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason='numba engine only supports numeric indices')
        request.node.add_marker(mark)
    index = pd.DatetimeIndex(['1950-06-30', '1952-10-24', '1953-05-29'])
    original = index.copy()
    df = DataFrame(1, index=index, columns=range(num_cols))
    df.apply(lambda x: x, engine=engine)
    assert index.freq == original.freq


def test_apply_datetime_tz_issue(
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support non-numeric indexes"
        )
        request.node.add_marker(mark)
    timestamps = [
        Timestamp('2019-03-15 12:34:31.909000+0000', tz='UTC'),
        Timestamp('2019-03-15 12:34:34.359000+0000', tz='UTC'),
        Timestamp('2019-03-15 12:34:34.660000+0000', tz='UTC')
    ]
    df = DataFrame(data=[0, 1, 2], index=timestamps)
    result = df.apply(
        lambda x: x.name,
        axis=1,
        engine=engine
    )
    expected = Series(timestamps, index=timestamps)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'df, method',
    [
        (DataFrame({'A': ['a', None], 'B': ['c', 'd']}), 'min'),
        (DataFrame({'A': ['a', None], 'B': ['c', 'd']}), 'max'),
        (DataFrame({'A': ['a', None], 'B': ['c', 'd']}), 'sum')
    ]
)
def test_mixed_column_raises(
    df: DataFrame,
    method: str,
    using_infer_string: bool
) -> None:
    if method == 'sum':
        msg = 'can only concatenate str \\(not "int"\\) to str|does not support'
    else:
        msg = "not supported between instances of 'str' and 'float'"
    if not using_infer_string:
        with pytest.raises(TypeError, match=msg):
            getattr(df, method)()
    else:
        getattr(df, method)()


@pytest.mark.parametrize(
    'col',
    [1, 1.0, True, 'a', np.nan]
)
def test_apply_dtype(col: Union[int, float, bool, str, None]) -> None:
    df = DataFrame([[1.0, col]], columns=['a', 'b'])
    result = df.apply(lambda x: x.dtype)
    expected = df.dtypes
    tm.assert_series_equal(result, expected)


def test_apply_mutating() -> None:
    df = DataFrame({'a': range(10), 'b': range(10, 20)})
    df_orig = df.copy()

    def func(row: Series) -> Series:
        mgr = row._mgr
        row.loc['a'] += 1
        assert row._mgr is not mgr
        return row

    expected = df.copy()
    expected['a'] += 1
    result = df.apply(func, axis=1)
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, df_orig)


def test_apply_empty_list_reduce() -> None:
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        columns=['a', 'b']
    )
    result = df.apply(
        lambda x: [],
        result_type='reduce'
    )
    expected = Series({'a': [], 'b': []}, dtype=object)
    tm.assert_series_equal(result, expected)


def test_apply_no_suffix_index(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support list-likes/dict-like callables"
        )
        request.node.add_marker(mark)
    pdf = DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    result = pdf.apply(
        ['sum', lambda x: x.sum(), lambda x: x.sum()],
        engine=engine
    )
    expected = DataFrame(
        {'A': [12, 12, 12], 'B': [27, 27, 27]},
        index=['sum', '<lambda>', '<lambda>']
    )
    tm.assert_frame_equal(result, expected)


def test_apply_raw_returns_string(engine: str) -> None:
    if engine == 'numba':
        pytest.skip('No object dtype support in numba')
    df = DataFrame({'A': ['aa', 'bbb']})
    result = df.apply(
        lambda x: x[0],
        engine=engine,
        axis=1,
        raw=True
    )
    expected = Series(['aa', 'bbb'])
    tm.assert_series_equal(result, expected)


def test_aggregation_func_column_order() -> None:
    df = DataFrame(
        [
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 5, 4),
            (5, 6, 6),
            (6, 7, 7)
        ],
        columns=('att1', 'att2', 'att3')
    )

    def sum_div2(s: Series) -> float:
        return s.sum() / 2

    aggs = ['sum', sum_div2, 'count', 'min']
    result = df.agg(aggs)
    expected = DataFrame({
        'att1': [21.0, 10.5, 6.0, 1.0],
        'att2': [18.0, 9.0, 6.0, 0.0],
        'att3': [17.0, 8.5, 6.0, 0.0]
    }, index=['sum', 'sum_div2', 'count', 'min'])
    tm.assert_frame_equal(result, expected)


def test_apply_getitem_axis_1(engine: str, request: pytest.FixtureRequest) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(
            reason='numba engine not supporting duplicate index values'
        )
        request.node.add_marker(mark)
    df = DataFrame({'a': [0, 1, 2], 'b': [1, 2, 3]})
    result = df[['a', 'a']].apply(
        lambda x: x.iloc[0] + x.iloc[1],
        axis=1,
        engine=engine
    )
    expected = Series([0, 2, 4])
    tm.assert_series_equal(result, expected)


def test_nuiscance_depr_passes_through_warnings() -> None:

    def expected_warning(x: Series) -> float:
        warnings.warn('Hello, World!')
        return x.sum()

    df = DataFrame({'a': [1, 2, 3]})
    with tm.assert_produces_warning(UserWarning, match='Hello, World!'):
        df.agg([expected_warning])


def test_apply_type() -> None:
    df = DataFrame({
        'col1': [3, 'string', float],
        'col2': [
            0.25,
            datetime(2020, 1, 1),
            np.nan
        ]
    }, index=['a', 'b', 'c'])
    result = df.apply(type, axis=0)
    expected = Series({'col1': Series, 'col2': Series})
    tm.assert_series_equal(result, expected)
    result = df.apply(type, axis=1)
    expected = Series({'a': Series, 'b': Series, 'c': Series})
    tm.assert_series_equal(result, expected)


def test_apply_on_empty_dataframe(engine: str) -> None:
    df = DataFrame({'a': [1, 2], 'b': [3, 0]})
    result = df.head(0).apply(
        lambda x: max(x['a'], x['b']),
        axis=1,
        engine=engine
    )
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)


def test_apply_return_list() -> None:
    df = DataFrame({'a': [1, 2], 'b': [2, 3]})
    result = df.apply(
        lambda x: [x.values],
        axis=1
    )
    expected = DataFrame({'a': [[1, 2]], 'b': [[2, 3]]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'test, constant',
    [
        (
            {'a': [1, 2, 3], 'b': [1, 1, 1]},
            {'a': [1, 2, 3], 'b': [1]}
        ),
        (
            {'a': [2, 2, 2], 'b': [1, 1, 1]},
            {'a': [2], 'b': [1]}
        )
    ]
)
def test_unique_agg_type_is_series(
    test: Dict[str, List[Union[int]]],
    constant: Dict[str, List[Union[int]]]
) -> None:
    df1 = DataFrame(test)
    expected = Series(data=constant, index=['a', 'b'], dtype='object')
    aggregation = {'a': 'unique', 'b': 'unique'}
    result = df1.agg(aggregation)
    tm.assert_series_equal(result, expected)


def test_any_apply_keyword_non_zero_axis_regression() -> None:
    df = DataFrame({
        'A': [1, 2, 0],
        'B': [0, 2, 0],
        'C': [0, 0, 0]
    })
    expected = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)
    result = df.apply('any', axis=1)
    tm.assert_series_equal(result, expected)
    result = df.apply('any', 1)
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated() -> None:
    df = DataFrame({'x': [1, 2, 3]})

    def foo1(x: Series, a: int = 1, c: int = 0) -> Series:
        return x + a + c

    def foo2(x: Series, b: int = 2, c: int = 0) -> Series:
        return x + b + c

    result = df.agg(foo1, 0, 3, c=4)
    expected = df + 7
    tm.assert_frame_equal(result, expected)
    result = df.agg([foo1, foo2], 0, 3, c=4)
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],
        columns=[['x', 'x'], ['foo1', 'foo2']]
    )
    tm.assert_frame_equal(result, expected)
    result = df.agg({'x': foo1}, 0, 3, c=4)
    expected = DataFrame([2, 3, 4], columns=['x'])
    tm.assert_frame_equal(result, expected)


def test_agg_std() -> None:
    df = DataFrame(
        np.arange(6).reshape(3, 2),
        columns=['A', 'B']
    )
    result = df.agg(np.std, ddof=1)
    expected = Series({'A': 2.0, 'B': 2.0}, dtype=float)
    tm.assert_series_equal(result, expected)
    result = df.agg([np.std], ddof=1)
    expected = DataFrame(
        {'A': 2.0, 'B': 2.0},
        index=['std']
    )
    tm.assert_frame_equal(result, expected)


def test_agg_dist_like_and_nonunique_columns() -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    df.columns = ['A', 'A', 'C']
    result = df.agg({'A': 'count'})
    expected = df['A'].count()
    tm.assert_series_equal(result, expected)


def test_any_apply_keyword_non_zero_axis_regression() -> None:
    df = DataFrame({
        'A': [1, 2, 0],
        'B': [0, 2, 0],
        'C': [0, 0, 0]
    })
    expected = Series([True, True, False])
    tm.assert_series_equal(df.any(axis=1), expected)
    result = df.apply('any', axis=1)
    tm.assert_series_equal(result, expected)
    result = df.apply('any', 1)
    tm.assert_series_equal(result, expected)


def test_agg_reduce(float_frame: DataFrame) -> None:
    other_axis: Union[int, str] = 1 if 0 == 0 else 0  # Dummy to satisfy mypy
    name1, name2 = float_frame.axes[other_axis].unique()[:2].sort_values()
    expected = pd.concat(
        [float_frame.mean(axis=axis), float_frame.max(axis=axis), float_frame.sum(axis=axis)],
        axis=1
    )
    expected.columns = ['mean', 'max', 'sum']
    expected = expected.T if axis == 0 else expected
    result = float_frame.agg(['mean', 'max', 'sum'], axis=0)
    tm.assert_frame_equal(result, expected)
    func = {name1: 'mean', name2: 'sum'}
    result = float_frame.agg(func, axis=0)
    expected = Series(
        [float_frame.loc[other_axis][name1].mean(), float_frame.loc[other_axis][name2].sum()],
        index=[name1, name2]
    )
    tm.assert_series_equal(result, expected)
    func = {name1: ['mean'], name2: ['sum']}
    result = float_frame.agg(func, axis=0)
    expected = DataFrame({
        name1: Series([float_frame.loc[other_axis][name1].mean()], index=['mean']),
        name2: Series([float_frame.loc[other_axis][name2].sum()], index=['sum'])
    })
    tm.assert_frame_equal(result, expected)
    func = {name1: ['mean', 'sum'], name2: ['sum', 'max']}
    result = float_frame.agg(func, axis=0)
    expected = pd.concat({
        name1: Series(
            [float_frame.loc[other_axis][name1].mean(), float_frame.loc[other_axis][name1].sum()],
            index=['mean', 'sum']
        ),
        name2: Series(
            [float_frame.loc[other_axis][name2].sum(), float_frame.loc[other_axis][name2].max()],
            index=['sum', 'max']
        )
    }, axis=1)
    tm.assert_frame_equal(result, expected)


def test_named_agg_reduce_axis1_raises(float_frame: DataFrame) -> None:
    name1, name2 = float_frame.axes[0].unique()[:2].sort_values()
    msg = 'Named aggregation is not supported when axis=1.'
    for axis in [1, 'columns']:
        with pytest.raises(NotImplementedError, match=msg):
            float_frame.agg(
                row1=(name1, 'sum'),
                row2=(name2, 'max'),
                axis=axis
            )


def test_nuiscance_columns() -> None:
    df = DataFrame({
        'A': [1, 2, 3],
        'B': [1.0, 2.0, 3.0],
        'C': ['foo', 'bar', 'baz'],
        'D': date_range('20130101', periods=3)
    })
    result = df.agg('min')
    expected = Series(
        [1, 1.0, 'bar', Timestamp('20130101')],
        index=df.columns
    )
    tm.assert_series_equal(result, expected)
    result = df.agg(['min'])
    expected = DataFrame(
        [[1, 1.0, 'bar', Timestamp('20130101').as_unit('ns')]],
        index=['min'],
        columns=df.columns
    )
    tm.assert_frame_equal(result, expected)
    msg = 'does not support operation'
    with pytest.raises(TypeError, match=msg):
        df.agg('sum')
    result = df[['A', 'B', 'C']].agg('sum')
    expected = Series(
        [6, 6.0, 'foobarbaz'],
        index=['A', 'B', 'C']
    )
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match=msg):
        df.agg(['sum'])


@pytest.mark.parametrize(
    'how',
    ['agg', 'apply']
)
def test_non_callable_aggregates(
    how: str
) -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    result = getattr(df, how)({'A': 'count'})
    expected = Series({'A': 2})
    tm.assert_series_equal(result, expected)
    result = getattr(df, how)({'A': 'size'})
    expected = Series({'A': 3})
    tm.assert_series_equal(result, expected)
    result1 = getattr(df, how)(['count', 'size'])
    result2 = getattr(df, how)({
        'A': ['count', 'size'],
        'B': ['count', 'size'],
        'C': ['count', 'size']
    })
    expected = DataFrame({
        'A': {'count': 2, 'size': 3},
        'B': {'count': 2, 'size': 3},
        'C': {'count': 2, 'size': 3}
    })
    tm.assert_frame_equal(result1, result2, check_like=True)
    tm.assert_frame_equal(result2, expected, check_like=True)
    result = getattr(df, how)('count')
    expected = df.count()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'how',
    ['agg', 'apply']
)
def test_size_as_str(
    how: str,
    axis: int
) -> None:
    df = DataFrame({
        'A': [None, 2, 3],
        'B': [1.0, np.nan, 3.0],
        'C': ['foo', None, 'bar']
    })
    result = getattr(df, how)('size', axis=axis)
    if axis in (0, 'index'):
        expected = Series(df.shape[0], index=df.columns)
    else:
        expected = Series(df.shape[1], index=df.index)
    tm.assert_series_equal(result, expected)


def test_agg_listlike_result() -> None:
    df = DataFrame({
        'A': [2, 2, 3],
        'B': [1.5, np.nan, 1.5],
        'C': ['foo', None, 'bar']
    })

    def func(group_col: Series) -> List[Any]:
        return list(group_col.dropna().unique())

    result = df.agg(func)
    expected = Series(
        [[2, 3], [1.5], ['foo', 'bar']],
        index=['A', 'B', 'C']
    )
    tm.assert_series_equal(result, expected)
    result = df.agg([func])
    expected = expected.to_frame('func').T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'axis, args, kwargs',
    [
        (0, (1, 2, 3), {}),
        (0, (8, 7, 15), {}),
        (0, (1, 2), {}),
        (0, (1,), {'b': 2}),
        (0, (), {'a': 1, 'b': 2}),
        (0, (), {'a': 2, 'b': 1}),
        (0, (), {'a': 1, 'b': 2, 'c': 3})
    ]
)
def test_agg_args_kwargs(
    axis: int,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> None:

    def f(x: Series, a: int, b: int, c: int = 3) -> float:
        return x.sum() + (a + b) / c

    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        expected = Series([5.0, 7.0])
    else:
        expected = Series([4.0, 8.0])
    result = df.agg(f, axis, *args, **kwargs)
    tm.assert_series_equal(result, expected)
    pytest.skip("Duplicate test name")


def test_frequency_is_original(
    num_cols: int,
    engine: str,
    request: pytest.FixtureRequest
) -> None:
    if engine == 'numba':
        mark = pytest.mark.xfail(reason='numba engine only supports numeric indices')
        request.node.add_marker(mark)
    index = pd.DatetimeIndex(['1950-06-30', '1952-10-24', '1953-05-29'])
    original = index.copy()
    df = DataFrame(1, index=index, columns=range(num_cols))
    df.apply(lambda x: x, engine=engine)
    assert index.freq == original.freq
