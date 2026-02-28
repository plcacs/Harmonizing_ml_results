import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
from pandas import DataFrame, Series, option_context
import pandas._testing as tm
from pandas.util.version import Version

pytestmark: list[pytest.Mark] = [pytest.mark.single_cpu]
numba = pytest.importorskip('numba')
pytestmark.append(pytest.mark.skipif(Version(numba.__version__) == Version('0.61') and is_platform_arm(), reason=f'Segfaults on ARM platforms with numba {numba.__version__}'))

def test_correct_function_signature() -> None:
    """Test that a function with incorrect signature raises NumbaUtilError."""
    pytest.importorskip('numba')

    def incorrect_function(x: np.ndarray) -> np.ndarray:
        return x + 1
    data: DataFrame = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key').transform(incorrect_function, engine='numba')
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key')['data'].transform(incorrect_function, engine='numba')

def test_check_nopython_kwargs() -> None:
    """Test that numba engine raises TypeError if nopython keyword is used."""
    pytest.importorskip('numba')

    def incorrect_function(values: np.ndarray, index: np.ndarray, *, a: int) -> np.ndarray:
        return values + a

    def correct_function(values: np.ndarray, index: np.ndarray, a: int) -> np.ndarray:
        return values + a
    data: DataFrame = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    with pytest.raises(TypeError, match="missing a required (keyword-only argument|argument): 'a'"):
        data.groupby('key').transform(incorrect_function, engine='numba', b=1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        data.groupby('key').transform(correct_function, engine='numba', b=1)
    with pytest.raises(TypeError, match="missing a required (keyword-only argument|argument): 'a'"):
        data.groupby('key')['data'].transform(incorrect_function, engine='numba', b=1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        data.groupby('key')['data'].transform(correct_function, engine='numba', b=1)
    with pytest.raises(NumbaUtilError, match='numba does not support'):
        data.groupby('key').transform(incorrect_function, engine='numba', a=1)
    actual: DataFrame = data.groupby('key').transform(correct_function, engine='numba', a=1)
    tm.assert_frame_equal(data[['data']] + 1, actual)
    with pytest.raises(NumbaUtilError, match='numba does not support'):
        data.groupby('key')['data'].transform(incorrect_function, engine='numba', a=1)
    actual: Series = data.groupby('key')['data'].transform(correct_function, engine='numba', a=1)
    tm.assert_series_equal(data['data'] + 1, actual)

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False], ids=['jit', 'nojit'])
def test_numba_vs_cython(jit: bool, frame_or_series: type, nogil: bool, parallel: bool, nopython: bool, as_index: bool) -> None:
    """Test that numba engine produces the same results as cython engine."""
    pytest.importorskip('numba')

    def func(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return values + 1
    if jit:
        import numba
        func = numba.jit(func)
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    engine_kwargs: dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    grouped: DataFrameGroupBy = data.groupby(0, as_index=as_index)
    if frame_or_series is Series:
        grouped = grouped[1]
    result: DataFrame = grouped.transform(func, engine='numba', engine_kwargs=engine_kwargs)
    expected: DataFrame = grouped.transform(lambda x: x + 1, engine='cython')
    tm.assert_equal(result, expected)

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False], ids=['jit', 'nojit'])
def test_cache(jit: bool, frame_or_series: type, nogil: bool, parallel: bool, nopython: bool) -> None:
    """Test that numba engine caches the results."""
    pytest.importorskip('numba')

    def func_1(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return values + 1

    def func_2(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return values * 5
    if jit:
        import numba
        func_1 = numba.jit(func_1)
        func_2 = numba.jit(func_2)
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    engine_kwargs: dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    grouped: DataFrameGroupBy = data.groupby(0)
    if frame_or_series is Series:
        grouped = grouped[1]
    result: DataFrame = grouped.transform(func_1, engine='numba', engine_kwargs=engine_kwargs)
    expected: DataFrame = grouped.transform(lambda x: x + 1, engine='cython')
    tm.assert_equal(result, expected)
    result: DataFrame = grouped.transform(func_2, engine='numba', engine_kwargs=engine_kwargs)
    expected: DataFrame = grouped.transform(lambda x: x * 5, engine='cython')
    tm.assert_equal(result, expected)
    result: DataFrame = grouped.transform(func_1, engine='numba', engine_kwargs=engine_kwargs)
    expected: DataFrame = grouped.transform(lambda x: x + 1, engine='cython')
    tm.assert_equal(result, expected)

def test_use_global_config() -> None:
    """Test that global config is used when engine is None."""
    pytest.importorskip('numba')

    def func_1(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return values + 1
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    grouped: DataFrameGroupBy = data.groupby(0)
    expected: DataFrame = grouped.transform(func_1, engine='numba')
    with option_context('compute.use_numba', True):
        result: DataFrame = grouped.transform(func_1, engine=None)
    tm.assert_frame_equal(expected, result)

@pytest.mark.parametrize('agg_func', [['min', 'max'], 'min', {'B': ['min', 'max'], 'C': 'sum'}], ids=['list', 'single', 'dict'])
def test_string_cython_vs_numba(agg_func: str | list[str] | dict[str, str | list[str]], numba_supported_reductions: tuple[str, dict[str, str | list[str]]]) -> None:
    """Test that numba engine produces the same results as cython engine."""
    pytest.importorskip('numba')
    agg_func, kwargs = numba_supported_reductions
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    grouped: DataFrameGroupBy = data.groupby(0)
    result: DataFrame = grouped.transform(agg_func, engine='numba', **kwargs)
    expected: DataFrame = grouped.transform(agg_func, engine='cython', **kwargs)
    tm.assert_frame_equal(result, expected)
    result: Series = grouped[1].transform(agg_func, engine='numba', **kwargs)
    expected: Series = grouped[1].transform(agg_func, engine='cython', **kwargs)
    tm.assert_series_equal(result, expected)

def test_args_not_cached() -> None:
    """Test that numba engine does not cache the results."""
    pytest.importorskip('numba')

    def sum_last(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
        return values[-n:].sum()
    df: DataFrame = DataFrame({'id': [0, 0, 1, 1], 'x': [1, 1, 1, 1]})
    grouped_x: SeriesGroupBy = df.groupby('id')['x']
    result: Series = grouped_x.transform(sum_last, 1, engine='numba')
    expected: Series = Series([1.0] * 4, name='x')
    tm.assert_series_equal(result, expected)
    result: Series = grouped_x.transform(sum_last, 2, engine='numba')
    expected: Series = Series([2.0] * 4, name='x')
    tm.assert_series_equal(result, expected)

def test_index_data_correctly_passed() -> None:
    """Test that index data is correctly passed to the function."""
    pytest.importorskip('numba')

    def f(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return index - 1
    df: DataFrame = DataFrame({'group': ['A', 'A', 'B'], 'v': [4, 5, 6]}, index=[-1, -2, -3])
    result: DataFrame = df.groupby('group').transform(f, engine='numba')
    expected: DataFrame = DataFrame([-2.0, -3.0, -4.0], columns=['v'], index=[-1, -2, -3])
    tm.assert_frame_equal(result, expected)

def test_index_order_consistency_preserved() -> None:
    """Test that index order consistency is preserved."""
    pytest.importorskip('numba')

    def f(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return values
    df: DataFrame = DataFrame({'vals': [0.0, 1.0, 2.0, 3.0], 'group': [0, 1, 0, 1]}, index=range(3, -1, -1))
    result: Series = df.groupby('group')['vals'].transform(f, engine='numba')
    expected: Series = Series([0.0, 1.0, 2.0, 3.0], index=range(3, -1, -1), name='vals')
    tm.assert_series_equal(result, expected)

def test_engine_kwargs_not_cached() -> None:
    """Test that engine kwargs are not cached."""
    pytest.importorskip('numba')
    nogil: bool = True
    parallel: bool = False
    nopython: bool = True

    def func_kwargs(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return nogil + parallel + nopython
    engine_kwargs: dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    df: DataFrame = DataFrame({'value': [0, 0, 0]})
    result: DataFrame = df.groupby(level=0).transform(func_kwargs, engine='numba', engine_kwargs=engine_kwargs)
    expected: DataFrame = DataFrame({'value': [2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)
    nogil = False
    engine_kwargs: dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    result: DataFrame = df.groupby(level=0).transform(func_kwargs, engine='numba', engine_kwargs=engine_kwargs)
    expected: DataFrame = DataFrame({'value': [1.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore')
def test_multiindex_one_key(nogil: bool, parallel: bool, nopython: bool) -> None:
    """Test that numba engine works with multiindex."""
    pytest.importorskip('numba')

    def numba_func(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return 1
    df: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3}]).set_index(['A', 'B'])
    engine_kwargs: dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    result: DataFrame = df.groupby('A').transform(numba_func, engine='numba', engine_kwargs=engine_kwargs)
    expected: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 1.0}]).set_index(['A', 'B'])
    tm.assert_frame_equal(result, expected)

def test_multiindex_multi_key_not_supported(nogil: bool, parallel: bool, nopython: bool) -> None:
    """Test that numba engine does not support multiindex with multiple keys."""
    pytest.importorskip('numba')

    def numba_func(values: np.ndarray, index: np.ndarray) -> np.ndarray:
        return 1
    df: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3}]).set_index(['A', 'B'])
    engine_kwargs: dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    with pytest.raises(NotImplementedError, match='more than 1 grouping labels'):
        df.groupby(['A', 'B']).transform(numba_func, engine='numba', engine_kwargs=engine_kwargs)

def test_multilabel_numba_vs_cython(numba_supported_reductions: tuple[str, dict[str, str | list[str]]]) -> None:
    """Test that numba engine produces the same results as cython engine."""
    pytest.importorskip('numba')
    reduction, kwargs = numba_supported_reductions
    df: DataFrame = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.random.default_rng(2).standard_normal(8)})
    gb: DataFrameGroupBy = df.groupby(['A', 'B'])
    res_agg: DataFrame = gb.transform(reduction, engine='numba', **kwargs)
    expected_agg: DataFrame = gb.transform(reduction, engine='cython', **kwargs)
    tm.assert_frame_equal(res_agg, expected_agg)

def test_multilabel_udf_numba_vs_cython() -> None:
    """Test that numba engine produces the same results as cython engine."""
    pytest.importorskip('numba')
    df: DataFrame = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.random.default_rng(2).standard_normal(8)})
    gb: DataFrameGroupBy = df.groupby(['A', 'B'])
    result: DataFrame = gb.transform(lambda values, index: (values - values.min()) / (values.max() - values.min()), engine='numba')
    expected: DataFrame = gb.transform(lambda x: (x - x.min()) / (x.max() - x.min()), engine='cython')
    tm.assert_frame_equal(result, expected)
