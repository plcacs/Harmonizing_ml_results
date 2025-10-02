import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
from pandas import DataFrame, Index, NamedAgg, Series, option_context
import pandas._testing as tm
from pandas.util.version import Version
from typing import Any, Dict, List, Optional, Union, Tuple

pytestmark: List[Any] = [pytest.mark.single_cpu]
numba = pytest.importorskip('numba')
pytestmark.append(pytest.mark.skipif(Version(numba.__version__) == Version('0.61') and is_platform_arm(), reason=f'Segfaults on ARM platforms with numba {numba.__version__}'))

def test_correct_function_signature() -> None:
    pytest.importorskip('numba')

    def incorrect_function(x: Any) -> Any:
        return sum(x) * 2.7
    data: DataFrame = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key').agg(incorrect_function, engine='numba')
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key')['data'].agg(incorrect_function, engine='numba')

def test_check_nopython_kwargs() -> None:
    pytest.importorskip('numba')

    def incorrect_function(values: Any, index: Any, *, a: Any) -> Any:
        return sum(values) * 2.7 + a

    def correct_function(values: Any, index: Any, a: Any) -> Any:
        return sum(values) * 2.7 + a
    data: DataFrame = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    expected: DataFrame = data.groupby('key').sum() * 2.7
    with pytest.raises(TypeError, match="missing a required (keyword-only argument|argument): 'a'"):
        data.groupby('key').agg(incorrect_function, engine='numba', b=1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        data.groupby('key').agg(correct_function, engine='numba', b=1)
    with pytest.raises(TypeError, match="missing a required (keyword-only argument|argument): 'a'"):
        data.groupby('key')['data'].agg(incorrect_function, engine='numba', b=1)
    with pytest.raises(TypeError, match="missing a required argument: 'a'"):
        data.groupby('key')['data'].agg(correct_function, engine='numba', b=1)
    with pytest.raises(NumbaUtilError, match='numba does not support'):
        data.groupby('key').agg(incorrect_function, engine='numba', a=1)
    actual: DataFrame = data.groupby('key').agg(correct_function, engine='numba', a=1)
    tm.assert_frame_equal(expected + 1, actual)
    with pytest.raises(NumbaUtilError, match='numba does not support'):
        data.groupby('key')['data'].agg(incorrect_function, engine='numba', a=1)
    actual: Series = data.groupby('key')['data'].agg(correct_function, engine='numba', a=1)
    tm.assert_series_equal(expected['data'] + 1, actual)

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_numba_vs_cython(jit: bool, frame_or_series: Any, nogil: bool, parallel: bool, nopython: bool, as_index: bool) -> None:
    pytest.importorskip('numba')

    def func_numba(values: Any, index: Any) -> Any:
        return np.mean(values) * 2.7
    if jit:
        import numba
        func_numba = numba.jit(func_numba)
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    grouped = data.groupby(0, as_index=as_index)
    if frame_or_series is Series:
        grouped = grouped[1]
    result = grouped.agg(func_numba, engine='numba', engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine='cython')
    tm.assert_equal(result, expected)

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_cache(jit: bool, frame_or_series: Any, nogil: bool, parallel: bool, nopython: bool) -> None:
    pytest.importorskip('numba')

    def func_1(values: Any, index: Any) -> Any:
        return np.mean(values) - 3.4

    def func_2(values: Any, index: Any) -> Any:
        return np.mean(values) * 2.7
    if jit:
        import numba
        func_1 = numba.jit(func_1)
        func_2 = numba.jit(func_2)
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    grouped = data.groupby(0)
    if frame_or_series is Series:
        grouped = grouped[1]
    result = grouped.agg(func_1, engine='numba', engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine='cython')
    tm.assert_equal(result, expected)
    result = grouped.agg(func_2, engine='numba', engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine='cython')
    tm.assert_equal(result, expected)
    result = grouped.agg(func_1, engine='numba', engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine='cython')
    tm.assert_equal(result, expected)

def test_use_global_config() -> None:
    pytest.importorskip('numba')

    def func_1(values: Any, index: Any) -> Any:
        return np.mean(values) - 3.4
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    grouped = data.groupby(0)
    expected: DataFrame = grouped.agg(func_1, engine='numba')
    with option_context('compute.use_numba', True):
        result = grouped.agg(func_1, engine=None)
    tm.assert_frame_equal(expected, result)

@pytest.mark.parametrize('agg_kwargs', [{'func': ['min', 'max']}, {'func': 'min'}, {'func': {1: ['min', 'max'], 2: 'sum'}}, {'bmin': NamedAgg(column=1, aggfunc='min')}])
def test_multifunc_numba_vs_cython_frame(agg_kwargs: Dict[str, Any]) -> None:
    pytest.importorskip('numba')
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[0, 1, 2])
    grouped = data.groupby(0)
    result = grouped.agg(**agg_kwargs, engine='numba')
    expected = grouped.agg(**agg_kwargs, engine='cython')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func', ['sum', 'mean', 'var', 'std', 'min', 'max'])
def test_multifunc_numba_vs_cython_frame_noskipna(func: str) -> None:
    pytest.importorskip('numba')
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, np.nan, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[0, 1, 2])
    grouped = data.groupby(0)
    result = grouped.agg(func, skipna=False, engine='numba')
    expected = grouped.agg(func, skipna=False, engine='cython')
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('agg_kwargs,expected_func', [({'func': lambda values, index: values.sum()}, 'sum'), pytest.param({'func': [lambda values, index: values.sum(), lambda values, index: values.min()]}, ['sum', 'min'], marks=pytest.mark.xfail(reason="This doesn't work yet! Fails in nopython pipeline!"))])
def test_multifunc_numba_udf_frame(agg_kwargs: Dict[str, Any], expected_func: Any) -> None:
    pytest.importorskip('numba')
    data: DataFrame = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[0, 1, 2])
    grouped = data.groupby(0)
    result = grouped.agg(**agg_kwargs, engine='numba')
    expected = grouped.agg(expected_func, engine='cython')
    tm.assert_frame_equal(result, expected, check_dtype=False)

@pytest.mark.parametrize('agg_kwargs', [{'func': ['min', 'max']}, {'func': 'min'}, {'min_val': 'min', 'max_val': 'max'}])
def test_multifunc_numba_vs_cython_series(agg_kwargs: Dict[str, Any]) -> None:
    pytest.importorskip('numba')
    labels: List[str] = ['a', 'a', 'b', 'b', 'a']
    data: Series = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    grouped = data.groupby(labels)
    agg_kwargs['engine'] = 'numba'
    result = grouped.agg(**agg_kwargs)
    agg_kwargs['engine'] = 'cython'
    expected = grouped.agg(**agg_kwargs)
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)

@pytest.mark.single_cpu
@pytest.mark.parametrize('data,agg_kwargs', [(Series([1.0, 2.0, 3.0, 4.0, 5.0]), {'func': ['min', 'max']}), (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {'func': 'min'}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': ['min', 'max']}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': 'min'}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': {1: ['min', 'max'], 2: 'sum'}}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'min_col': NamedAgg(column=1, aggfunc='min')})])
def test_multifunc_numba_kwarg_propagation(data: Union[Series, DataFrame], agg_kwargs: Dict[str, Any]) -> None:
    pytest.importorskip('numba')
    labels: List[str] = ['a', 'a', 'b', 'b', 'a']
    grouped = data.groupby(labels)
    result = grouped.agg(**agg_kwargs, engine='numba', engine_kwargs={'parallel': True})
    expected = grouped.agg(**agg_kwargs, engine='numba')
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)

def test_args_not_cached() -> None:
    pytest.importorskip('numba')

    def sum_last(values: Any, index: Any, n: int) -> Any:
        return values[-n:].sum()
    df: DataFrame = DataFrame({'id': [0, 0, 1, 1], 'x': [1, 1, 1, 1]})
    grouped_x = df.groupby('id')['x']
    result = grouped_x.agg(sum_last, 1, engine='numba')
    expected = Series([1.0] * 2, name='x', index=Index([0, 1], name='id'))
    tm.assert_series_equal(result, expected)
    result = grouped_x.agg(sum_last, 2, engine='numba')
    expected = Series([2.0] * 2, name='x', index=Index([0, 1], name='id'))
    tm.assert_series_equal(result, expected)

def test_index_data_correctly_passed() -> None:
    pytest.importorskip('numba')

    def f(values: Any, index: Any) -> Any:
        return np.mean(index)
    df: DataFrame = DataFrame({'group': ['A', 'A', 'B'], 'v': [4, 5, 6]}, index=[-1, -2, -3])
    result = df.groupby('group').aggregate(f, engine='numba')
    expected = DataFrame([-1.5, -3.0], columns=['v'], index=Index(['A', 'B'], name='group'))
    tm.assert_frame_equal(result, expected)

def test_engine_kwargs_not_cached() -> None:
    pytest.importorskip('numba')
    nogil: bool = True
    parallel: bool = False
    nopython: bool = True

    def func_kwargs(values: Any, index: Any) -> Any:
        return nogil + parallel + nopython
    engine_kwargs: Dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    df: DataFrame = DataFrame({'value': [0, 0, 0]})
    result = df.groupby(level=0).aggregate(func_kwargs, engine='numba', engine_kwargs=engine_kwargs)
    expected = DataFrame({'value': [2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)
    nogil = False
    engine_kwargs = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    result = df.groupby(level=0).aggregate(func_kwargs, engine='numba', engine_kwargs=engine_kwargs)
    expected = DataFrame({'value': [1.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)

@pytest.mark.filterwarnings('ignore')
def test_multiindex_one_key(nogil: bool, parallel: bool, nopython: bool) -> None:
    pytest.importorskip('numba')

    def numba_func(values: Any, index: Any) -> Any:
        return 1
    df: DataFrame = DataFrame([{'A': 1, 'B': 2, 'C': 3}]).set_index(['A', 'B'])
    engine_kwargs: Dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    result = df.groupby('A').agg(numba_func, engine='numba', engine_kwargs=engine_kwargs)
    expected = DataFrame([1.0], index=Index([1], name='A'), columns=['C'])
    tm.assert_frame_equal(result, expected)

def test_multiindex_multi_key_not_supported(nogil: bool, parallel: bool, nopython: bool) -> None:
    pytest