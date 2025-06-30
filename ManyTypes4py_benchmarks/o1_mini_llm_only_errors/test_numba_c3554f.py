import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import DataFrame, Series, option_context, to_datetime
import pandas._testing as tm
from pandas.util.version import Version
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

pytestmark = [pytest.mark.single_cpu]
numba = pytest.importorskip('numba')
pytestmark.append(pytest.mark.skipif(
    Version(numba.__version__) == Version('0.61') and is_platform_arm(),
    reason=f'Segfaults on ARM platforms with numba {numba.__version__}'
))

@pytest.fixture(params=['single', 'table'])
def method(request: pytest.FixtureRequest) -> str:
    """method keyword in rolling/expanding/ewm constructor"""
    return request.param

@pytest.fixture(params=[
    ['sum', {}],
    ['mean', {}],
    ['median', {}],
    ['max', {}],
    ['min', {}],
    ['var', {}],
    ['var', {'ddof': 0}],
    ['std', {}],
    ['std', {'ddof': 0}]
])
def arithmetic_numba_supported_operators(request: pytest.FixtureRequest) -> Tuple[str, Dict[str, Any]]:
    return request.param

@pytest.fixture
def roll_frame() -> DataFrame:
    return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40)})

@td.skip_if_no('numba')
@pytest.mark.filterwarnings('ignore')
class TestEngine:

    @pytest.mark.parametrize('jit', [True, False])
    def test_numba_vs_cython_apply(
        self,
        jit: bool,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        center: bool,
        step: int
    ) -> None:

        def f(x: np.ndarray, *args: Any) -> float:
            arg_sum = 0
            for arg in args:
                arg_sum += arg
            return np.mean(x) + arg_sum
        
        if jit:
            import numba
            f = numba.jit(f)
        
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        args: Tuple[int, ...] = (2,)
        s: Series = Series(range(10))
        result: Series = s.rolling(2, center=center, step=step).apply(
            f,
            args=args,
            engine='numba',
            engine_kwargs=engine_kwargs,
            raw=True
        )
        expected: Series = s.rolling(2, center=center, step=step).apply(
            f,
            engine='cython',
            args=args,
            raw=True
        )
        tm.assert_series_equal(result, expected)

    def test_apply_numba_with_kwargs(self, roll_frame: DataFrame) -> None:

        def func(sr: np.ndarray, a: int = 0) -> float:
            return sr.sum() + a
        
        data: DataFrame = DataFrame(range(10))
        result: DataFrame = data.rolling(5).apply(
            func,
            engine='numba',
            raw=True,
            kwargs={'a': 1}
        )
        expected: DataFrame = data.rolling(5).sum() + 1
        tm.assert_frame_equal(result, expected)

        result = data.rolling(5).apply(
            func,
            engine='numba',
            raw=True,
            args=(1,)
        )
        tm.assert_frame_equal(result, expected)

        result = data.expanding().apply(
            func,
            engine='numba',
            raw=True,
            kwargs={'a': 1}
        )
        expected = data.expanding().sum() + 1
        tm.assert_frame_equal(result, expected)

        result = data.expanding().apply(
            func,
            engine='numba',
            raw=True,
            args=(1,)
        )
        tm.assert_frame_equal(result, expected)

        result = roll_frame.groupby('A').rolling(5).apply(
            func,
            engine='numba',
            raw=True,
            kwargs={'a': 1}
        )
        expected = roll_frame.groupby('A').rolling(5).sum() + 1
        tm.assert_frame_equal(result, expected)

        result = roll_frame.groupby('A').rolling(5).apply(
            func,
            engine='numba',
            raw=True,
            args=(1,)
        )
        tm.assert_frame_equal(result, expected)

        result = roll_frame.groupby('A').expanding().apply(
            func,
            engine='numba',
            raw=True,
            kwargs={'a': 1}
        )
        expected = roll_frame.groupby('A').expanding().sum() + 1
        tm.assert_frame_equal(result, expected)

        result = roll_frame.groupby('A').expanding().apply(
            func,
            engine='numba',
            raw=True,
            args=(1,)
        )
        tm.assert_frame_equal(result, expected)

    def test_numba_min_periods(self) -> None:

        def last_row(x: np.ndarray) -> float:
            assert len(x) == 3
            return x[-1]
        
        df: DataFrame = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
        result: DataFrame = df.rolling(3, method='table', min_periods=3).apply(
            last_row,
            raw=True,
            engine='numba'
        )
        expected: DataFrame = DataFrame([[np.nan, np.nan], [np.nan, np.nan], [5, 6], [7, 8]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('data', [
        DataFrame(np.eye(5)),
        DataFrame([
            [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
            [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
            [np.nan, np.nan, 5, 6, 7, 5, 5, 5, 5, 5]
        ]).T,
        Series(range(5), name='foo'),
        Series([20, 10, 10, np.inf, 1, 1, 2, 3]),
        Series([20, 10, 10, np.nan, 10, 1, 2, 3])
    ])
    def test_numba_vs_cython_rolling_methods(
        self,
        data: Union[DataFrame, Series],
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: Tuple[str, Dict[str, Any]],
        step: int
    ) -> None:
        method, kwargs = arithmetic_numba_supported_operators
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        roll = data.rolling(3, step=step)
        result: Union[Series, DataFrame] = getattr(roll, method)(
            engine='numba',
            engine_kwargs=engine_kwargs,
            **kwargs
        )
        expected: Union[Series, DataFrame] = getattr(roll, method)(
            engine='cython',
            **kwargs
        )
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('data', [
        DataFrame(np.eye(5)),
        Series(range(5), name='foo')
    ])
    def test_numba_vs_cython_expanding_methods(
        self,
        data: Union[DataFrame, Series],
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: Tuple[str, Dict[str, Any]]
    ) -> None:
        method, kwargs = arithmetic_numba_supported_operators
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        data = DataFrame(np.eye(5))
        expand = data.expanding()
        result: Union[Series, DataFrame] = getattr(expand, method)(
            engine='numba',
            engine_kwargs=engine_kwargs,
            **kwargs
        )
        expected: Union[Series, DataFrame] = getattr(expand, method)(
            engine='cython',
            **kwargs
        )
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('jit', [True, False])
    def test_cache_apply(
        self,
        jit: bool,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        step: int
    ) -> None:

        def func_1(x: np.ndarray) -> float:
            return np.mean(x) + 4

        def func_2(x: np.ndarray) -> float:
            return np.std(x) * 5

        if jit:
            import numba
            func_1 = numba.jit(func_1)
            func_2 = numba.jit(func_2)
        
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        roll = Series(range(10)).rolling(2, step=step)
        result: Series = roll.apply(
            func_1,
            engine='numba',
            engine_kwargs=engine_kwargs,
            raw=True
        )
        expected: Series = roll.apply(
            func_1,
            engine='cython',
            raw=True
        )
        tm.assert_series_equal(result, expected)

        result = roll.apply(
            func_2,
            engine='numba',
            engine_kwargs=engine_kwargs,
            raw=True
        )
        expected = roll.apply(
            func_2,
            engine='cython',
            raw=True
        )
        tm.assert_series_equal(result, expected)

        result = roll.apply(
            func_1,
            engine='numba',
            engine_kwargs=engine_kwargs,
            raw=True
        )
        expected = roll.apply(
            func_1,
            engine='cython',
            raw=True
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('window,window_kwargs', [
        ['rolling', {'window': 3, 'min_periods': 0}],
        ['expanding', {}]
    ])
    def test_dont_cache_args(
        self,
        window: str,
        window_kwargs: Dict[str, Any],
        nogil: bool,
        parallel: bool,
        nopython: bool,
        method: str
    ) -> None:

        def add(values: np.ndarray, x: int) -> float:
            return np.sum(values) + x
        
        engine_kwargs: Dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
        df: DataFrame = DataFrame({'value': [0, 0, 0]})
        result: DataFrame = getattr(df, window)(method=method, **window_kwargs).apply(
            add,
            raw=True,
            engine='numba',
            engine_kwargs=engine_kwargs,
            args=(1,)
        )
        expected: DataFrame = DataFrame({'value': [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)

        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add,
            raw=True,
            engine='numba',
            engine_kwargs=engine_kwargs,
            args=(2,)
        )
        expected = DataFrame({'value': [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

    def test_dont_cache_engine_kwargs(self) -> None:
        nogil: bool = False
        parallel: bool = True
        nopython: bool = True

        def func(x: np.ndarray) -> float:
            return nogil + parallel + nopython
        
        engine_kwargs: Dict[str, bool] = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
        df: DataFrame = DataFrame({'value': [0, 0, 0]})
        result: DataFrame = df.rolling(1).apply(
            func,
            raw=True,
            engine='numba',
            engine_kwargs=engine_kwargs
        )
        expected: DataFrame = DataFrame({'value': [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

        parallel = False
        engine_kwargs = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
        result = df.rolling(1).apply(
            func,
            raw=True,
            engine='numba',
            engine_kwargs=engine_kwargs
        )
        expected = DataFrame({'value': [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)

@td.skip_if_no('numba')
class TestEWM:

    @pytest.mark.parametrize('grouper', [lambda x: x, lambda x: x.groupby('A')], ids=['None', 'groupby'])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_invalid_engine(
        self,
        grouper: Callable[[DataFrame], Union[DataFrame, Any]],
        method: str
    ) -> None:
        df: DataFrame = DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': range(4)})
        with pytest.raises(ValueError, match='engine must be either'):
            getattr(grouper(df).ewm(com=1.0), method)(engine='foo')

    @pytest.mark.parametrize('grouper', [lambda x: x, lambda x: x.groupby('A')], ids=['None', 'groupby'])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_invalid_engine_kwargs(
        self,
        grouper: Callable[[DataFrame], Union[DataFrame, Any]],
        method: str
    ) -> None:
        df: DataFrame = DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': range(4)})
        with pytest.raises(ValueError, match='cython engine does not'):
            getattr(grouper(df).ewm(com=1.0), method)(engine='cython', engine_kwargs={'nopython': True})

    @pytest.mark.parametrize('grouper', ['None', 'groupby'])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_cython_vs_numba(
        self,
        grouper: str,
        method: str,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        ignore_na: bool,
        adjust: bool
    ) -> None:
        df: DataFrame = DataFrame({'B': range(4)})
        if grouper == 'None':
            group_func: Callable[[DataFrame], Union[DataFrame, Any]] = lambda x: x
        else:
            df['A'] = ['a', 'b', 'a', 'b']
            group_func = lambda x: x.groupby('A')
        if method == 'sum':
            adjust = True
        ewm = group_func(df).ewm(com=1.0, adjust=adjust, ignore_na=ignore_na)
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        result: DataFrame = getattr(ewm, method)(
            engine='numba',
            engine_kwargs=engine_kwargs
        )
        expected: DataFrame = getattr(ewm, method)(engine='cython')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('grouper', ['None', 'groupby'])
    def test_cython_vs_numba_times(
        self,
        grouper: str,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        ignore_na: bool
    ) -> None:
        df: DataFrame = DataFrame({'B': [0, 0, 1, 1, 2, 2]})
        if grouper == 'None':
            group_func: Callable[[DataFrame], Union[DataFrame, Any]] = lambda x: x
        else:
            df['A'] = ['a', 'b', 'a', 'b', 'b', 'a']
            group_func = lambda x: x.groupby('A')
        halflife: str = '23 days'
        times: Series = to_datetime([
            '2020-01-01',
            '2020-01-01',
            '2020-01-02',
            '2020-01-10',
            '2020-02-23',
            '2020-01-03'
        ])
        ewm = group_func(df).ewm(
            halflife=halflife,
            adjust=True,
            ignore_na=ignore_na,
            times=times
        )
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        result: DataFrame = ewm.mean(engine='numba', engine_kwargs=engine_kwargs)
        expected: DataFrame = ewm.mean(engine='cython')
        tm.assert_frame_equal(result, expected)

@td.skip_if_no('numba')
def test_use_global_config() -> None:

    def f(x: np.ndarray) -> float:
        return np.mean(x) + 2
    
    s: Series = Series(range(10))
    with option_context('compute.use_numba', True):
        result: Series = s.rolling(2).apply(f, engine=None, raw=True)
    expected: Series = s.rolling(2).apply(f, engine='numba', raw=True)
    tm.assert_series_equal(expected, result)

@td.skip_if_no('numba')
def test_invalid_kwargs_nopython() -> None:
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'a'"):
        Series(range(1)).rolling(1).apply(
            lambda x: x,
            kwargs={'a': 1},
            engine='numba',
            raw=True
        )
    with pytest.raises(NumbaUtilError, match='numba does not support keyword-only arguments'):
        Series(range(1)).rolling(1).apply(
            lambda x, *, a: x,
            kwargs={'a': 1},
            engine='numba',
            raw=True
        )
    tm.assert_series_equal(
        Series(range(1), dtype=float) + 1,
        Series(range(1)).rolling(1).apply(
            lambda x, a: (x + a).sum(),
            kwargs={'a': 1},
            engine='numba',
            raw=True
        )
    )

@td.skip_if_no('numba')
@pytest.mark.slow
@pytest.mark.filterwarnings('ignore')
class TestTableMethod:

    def test_table_series_valueerror(self) -> None:

        def f(x: np.ndarray) -> float:
            return np.sum(x, axis=0) + 1
        
        with pytest.raises(ValueError, match="method='table' not applicable for Series objects."):
            Series(range(1)).rolling(1, method='table').apply(
                f,
                engine='numba',
                raw=True
            )

    def test_table_method_rolling_methods(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: Tuple[str, Dict[str, Any]],
        step: int
    ) -> None:
        method, kwargs = arithmetic_numba_supported_operators
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        df: DataFrame = DataFrame(np.eye(3))
        roll_table = df.rolling(2, method='table', min_periods=0, step=step)
        if method in ('var', 'std'):
            with pytest.raises(NotImplementedError, match=f'{method} not supported'):
                getattr(roll_table, method)(
                    engine_kwargs=engine_kwargs,
                    engine='numba',
                    **kwargs
                )
        else:
            roll_single = df.rolling(2, method='single', min_periods=0, step=step)
            result: DataFrame = getattr(roll_table, method)(
                engine_kwargs=engine_kwargs,
                engine='numba',
                **kwargs
            )
            expected: DataFrame = getattr(roll_single, method)(
                engine_kwargs=engine_kwargs,
                engine='numba',
                **kwargs
            )
            tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_apply(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        step: int
    ) -> None:
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}

        def f(x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=0) + 1
        
        df: DataFrame = DataFrame(np.eye(3))
        result: DataFrame = df.rolling(2, method='table', min_periods=0, step=step).apply(
            f,
            raw=True,
            engine_kwargs=engine_kwargs,
            engine='numba'
        )
        expected: DataFrame = df.rolling(2, method='single', min_periods=0, step=step).apply(
            f,
            raw=True,
            engine_kwargs=engine_kwargs,
            engine='numba'
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_apply_col_order(self) -> None:

        def f(x: np.ndarray) -> float:
            return np.nanmean(x[:, 0] - x[:, 1])
        
        df: DataFrame = DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [6, 7, 8, 5, 6, 7]
        })
        result: DataFrame = df.rolling(3, method='table', min_periods=0)[['a', 'b']].apply(
            f,
            raw=True,
            engine='numba'
        )
        expected: DataFrame = DataFrame({
            'a': [-5, -5, -5, -3.66667, -2.33333, -1],
            'b': [-5, -5, -5, -3.66667, -2.33333, -1]
        })
        tm.assert_almost_equal(result, expected)

        result = df.rolling(3, method='table', min_periods=0)[['b', 'a']].apply(
            f,
            raw=True,
            engine='numba'
        )
        expected = DataFrame({
            'b': [5, 5, 5, 3.66667, 2.33333, 1],
            'a': [5, 5, 5, 3.66667, 2.33333, 1]
        })
        tm.assert_almost_equal(result, expected)

    def test_table_method_rolling_weighted_mean(
        self,
        step: int
    ) -> None:

        def weighted_mean(x: np.ndarray) -> np.ndarray:
            arr = np.ones((1, x.shape[1]))
            arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
            return arr
        
        df: DataFrame = DataFrame([
            [1, 2, 0.6],
            [2, 3, 0.4],
            [3, 4, 0.2],
            [4, 5, 0.7]
        ])
        result: DataFrame = df.rolling(2, method='table', min_periods=0, step=step).apply(
            weighted_mean,
            raw=True,
            engine='numba'
        )
        expected: DataFrame = DataFrame([
            [1.0, 2.0, 1.0],
            [1.8, 2.0, 1.0],
            [3.333333, 2.333333, 1.0],
            [1.555556, 7, 1.0]
        ])[::step]
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_apply(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool
    ) -> None:
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}

        def f(x: np.ndarray) -> np.ndarray:
            return np.sum(x, axis=0) + 1
        
        df: DataFrame = DataFrame(np.eye(3))
        result: DataFrame = df.expanding(method='table').apply(
            f,
            raw=True,
            engine_kwargs=engine_kwargs,
            engine='numba'
        )
        expected: DataFrame = df.expanding(method='single').apply(
            f,
            raw=True,
            engine_kwargs=engine_kwargs,
            engine='numba'
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_methods(
        self,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        arithmetic_numba_supported_operators: Tuple[str, Dict[str, Any]]
    ) -> None:
        method, kwargs = arithmetic_numba_supported_operators
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        df: DataFrame = DataFrame(np.eye(3))
        expand_table = df.expanding(method='table')
        if method in ('var', 'std'):
            with pytest.raises(NotImplementedError, match=f'{method} not supported'):
                getattr(expand_table, method)(
                    engine_kwargs=engine_kwargs,
                    engine='numba',
                    **kwargs
                )
        else:
            expand_single = df.expanding(method='single')
            result: DataFrame = getattr(expand_table, method)(
                engine_kwargs=engine_kwargs,
                engine='numba',
                **kwargs
            )
            expected: DataFrame = getattr(expand_single, method)(
                engine_kwargs=engine_kwargs,
                engine='numba',
                **kwargs
            )
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('data', [
        np.eye(3),
        np.ones((2, 3)),
        np.ones((3, 2))
    ])
    @pytest.mark.parametrize('method', ['mean', 'sum'])
    def test_table_method_ewm(
        self,
        data: np.ndarray,
        method: str,
        nogil: bool,
        parallel: bool,
        nopython: bool
    ) -> None:
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        df: DataFrame = DataFrame(data)
        result: DataFrame = getattr(df.ewm(com=1, method='table'), method)(
            engine_kwargs=engine_kwargs,
            engine='numba'
        )
        expected: DataFrame = getattr(df.ewm(com=1, method='single'), method)(
            engine_kwargs=engine_kwargs,
            engine='numba'
        )
        tm.assert_frame_equal(result, expected)

@td.skip_if_no('numba')
def test_npfunc_no_warnings() -> None:
    df: DataFrame = DataFrame({'col1': [1, 2, 3, 4, 5]})
    with tm.assert_produces_warning(False):
        df.col1.rolling(2).apply(
            np.prod,
            raw=True,
            engine='numba'
        )
