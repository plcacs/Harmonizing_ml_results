import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
from pandas import DataFrame, Index, NamedAgg, Series, option_context
import pandas._testing as tm
from pandas.util.version import Version
pytestmark = [pytest.mark.single_cpu]
numba = pytest.importorskip('numba')
pytestmark.append(pytest.mark.skipif(Version(numba.__version__) == Version('0.61') and is_platform_arm(), reason=f'Segfaults on ARM platforms with numba {numba.__version__}'))

def test_correct_function_signature() -> None:
    pytest.importorskip('numba')

    def incorrect_function(x: np.ndarray) -> float:
        return sum(x) * 2.7
    data = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key').agg(incorrect_function, engine='numba')
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key')['data'].agg(incorrect_function, engine='numba')

def test_check_nopython_kwargs() -> None:
    pytest.importorskip('numba')

    def incorrect_function(values: np.ndarray, index: Index, *, a: float) -> float:
        return sum(values) * 2.7 + a

    def correct_function(values: np.ndarray, index: Index, a: float) -> float:
        return sum(values) * 2.7 + a
    data = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    expected = data.groupby('key').sum() * 2.7
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
    actual = data.groupby('key').agg(correct_function, engine='numba', a=1)
    tm.assert_frame_equal(expected + 1, actual)
    with pytest.raises(NumbaUtilError, match='numba does not support'):
        data.groupby('key')['data'].agg(incorrect_function, engine='numba', a=1)
    actual = data.groupby('key')['data'].agg(correct_function, engine='numba', a=1)
    tm.assert_series_equal(expected['data'] + 1, actual)

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_numba_vs_cython(jit: bool, frame_or_series: Series | DataFrame, nogil: bool, parallel: bool, nopython: bool, as_index: bool) -> None:
    pytest.importorskip('numba')

    def func_numba(values: np.ndarray, index: Index) -> float:
        return np.mean(values) * 2.7
    if jit:
        import numba
        func_numba = numba.jit(func_numba)
    data = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    grouped = data.groupby(0, as_index=as_index)
    if frame_or_series is Series:
        grouped = grouped[1]
    result = grouped.agg(func_numba, engine='numba', engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine='cython')
    tm.assert_equal(result, expected)

@pytest.mark.filterwarnings('ignore')
@pytest.mark.parametrize('jit', [True, False])
def test_cache(jit: bool, frame_or_series: Series | DataFrame, nogil: bool, parallel: bool, nopython: bool) -> None:
    pytest.importorskip('numba')

    def func_1(values: np.ndarray, index: Index) -> float:
        return np.mean(values) - 3.4

    def func_2(values: np.ndarray, index: Index) -> float:
        return np.mean(values) * 2.7
    if jit:
        import numba
        func_1 = numba.jit(func_1)
        func_2 = numba.jit(func_2)
    data = DataFrame({0: ['a', 'a', 'b', 'b', 'a'], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1])
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
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

# ... and so on for the rest of the functions
