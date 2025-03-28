import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

class Methods:
    params: Tuple[List[str], List[Tuple[str, Dict[str, Any]]], List[str], List[str]] = (['DataFrame', 'Series'], [('rolling', {'window': 10}), ('rolling', {'window': 1000}), ('expanding', {})], ['int', 'float'], ['median', 'mean', 'max', 'min', 'std', 'count', 'skew', 'kurt', 'sum', 'sem'])
    param_names: List[str] = ['constructor', 'window_kwargs', 'dtype', 'method']
    window: Any

    def setup(self, constructor, window_kwargs, dtype, method):
        N: int = 10 ** 5
        window, kwargs = window_kwargs
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        obj: Union[pd.DataFrame, pd.Series] = getattr(pd, constructor)(arr)
        self.window = getattr(obj, window)(**kwargs)

    def time_method(self, constructor, window_kwargs, dtype, method):
        getattr(self.window, method)()

    def peakmem_method(self, constructor, window_kwargs, dtype, method):
        getattr(self.window, method)()

class Apply:
    params: Tuple[List[str], List[int], List[str], List[Union[Callable[..., Any], Callable[[np.ndarray], Any]]], List[bool]] = (['DataFrame', 'Series'], [3, 300], ['int', 'float'], [sum, np.sum, lambda x: np.sum(x) + 5], [True, False])
    param_names: List[str] = ['constructor', 'window', 'dtype', 'function', 'raw']
    roll: Any

    def setup(self, constructor, window, dtype, function, raw):
        N: int = 10 ** 3
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)

    def time_rolling(self, constructor, window, dtype, function, raw):
        self.roll.apply(function, raw=raw)

class NumbaEngineMethods:
    params: Tuple[List[str], List[str], List[Tuple[str, Dict[str, Any]]], List[str], List[bool], List[Optional[int]]] = (['DataFrame', 'Series'], ['int', 'float'], [('rolling', {'window': 10}), ('expanding', {})], ['sum', 'max', 'min', 'median', 'mean', 'var', 'std'], [True, False], [None, 100])
    param_names: List[str] = ['constructor', 'dtype', 'window_kwargs', 'method', 'parallel', 'cols']
    window: Any

    def setup(self, constructor, dtype, window_kwargs, method, parallel, cols):
        N: int = 10 ** 3
        window, kwargs = window_kwargs
        shape: Union[int, Tuple[int, int]] = (N, cols) if cols is not None and constructor != 'Series' else N
        arr: Union[np.ndarray, np.ndarray] = (100 * np.random.random(shape)).astype(dtype)
        data: Union[pd.DataFrame, pd.Series] = getattr(pd, constructor)(arr)
        with warnings.catch_warnings(record=True):
            self.window = getattr(data, window)(**kwargs)
            getattr(self.window, method)(engine='numba', engine_kwargs={'parallel': parallel})

    def test_method(self, constructor, dtype, window_kwargs, method, parallel, cols):
        with warnings.catch_warnings(record=True):
            getattr(self.window, method)(engine='numba', engine_kwargs={'parallel': parallel})

class NumbaEngineApply:
    params: Tuple[List[str], List[str], List[Tuple[str, Dict[str, Any]]], List[Union[Callable[..., Any], Callable[[np.ndarray], Any]]], List[bool], List[Optional[int]]] = (['DataFrame', 'Series'], ['int', 'float'], [('rolling', {'window': 10}), ('expanding', {})], [np.sum, lambda x: np.sum(x) + 5], [True, False], [None, 100])
    param_names: List[str] = ['constructor', 'dtype', 'window_kwargs', 'function', 'parallel', 'cols']
    window: Any

    def setup(self, constructor, dtype, window_kwargs, function, parallel, cols):
        N: int = 10 ** 3
        window, kwargs = window_kwargs
        shape: Union[int, Tuple[int, int]] = (N, cols) if cols is not None and constructor != 'Series' else N
        arr: Union[np.ndarray, np.ndarray] = (100 * np.random.random(shape)).astype(dtype)
        data: Union[pd.DataFrame, pd.Series] = getattr(pd, constructor)(arr)
        with warnings.catch_warnings(record=True):
            self.window = getattr(data, window)(**kwargs)
            self.window.apply(function, raw=True, engine='numba', engine_kwargs={'parallel': parallel})

    def test_method(self, constructor, dtype, window_kwargs, function, parallel, cols):
        with warnings.catch_warnings(record=True):
            self.window.apply(function, raw=True, engine='numba', engine_kwargs={'parallel': parallel})

class EWMMethods:
    params: Tuple[List[str], List[Tuple[Dict[str, Any], str]], List[str]] = (['DataFrame', 'Series'], [({'halflife': 10}, 'mean'), ({'halflife': 10}, 'std'), ({'halflife': 1000}, 'mean'), ({'halflife': 1000}, 'std'), ({'halflife': '1 Day', 'times': pd.date_range('1900', periods=10 ** 5, freq='23s')}, 'mean')], ['int', 'float'])
    param_names: List[str] = ['constructor', 'kwargs_method', 'dtype']
    ewm: Any
    method: str

    def setup(self, constructor, kwargs_method, dtype):
        N: int = 10 ** 5
        kwargs, method = kwargs_method
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        self.method = method
        self.ewm = getattr(pd, constructor)(arr).ewm(**kwargs)

    def time_ewm(self, constructor, kwargs_method, dtype):
        getattr(self.ewm, self.method)()

class VariableWindowMethods(Methods):
    params: Tuple[List[str], List[str], List[str], List[str]] = (['DataFrame', 'Series'], ['50s', '1h', '1d'], ['int', 'float'], ['median', 'mean', 'max', 'min', 'std', 'count', 'skew', 'kurt', 'sum', 'sem'])
    param_names: List[str] = ['constructor', 'window', 'dtype', 'method']

    def setup(self, constructor, window, dtype, method):
        N: int = 10 ** 5
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        index: pd.DatetimeIndex = pd.date_range('2017-01-01', periods=N, freq='5s')
        self.window = getattr(pd, constructor)(arr, index=index).rolling(window)

class Pairwise:
    params: Tuple[List[Tuple[Dict[str, Any], str]], List[str], List[bool]] = ([({'window': 10}, 'rolling'), ({'window': 1000}, 'rolling'), ({}, 'expanding')], ['corr', 'cov'], [True, False])
    param_names: List[str] = ['window_kwargs', 'method', 'pairwise']
    df: pd.DataFrame
    window: Any
    window_group: Any

    def setup(self, kwargs_window, method, pairwise):
        N: int = 10 ** 4
        kwargs, window = kwargs_window
        groups: List[int] = [i for _ in range(N // 20) for i in range(20)]
        arr: np.ndarray = np.random.random(N)
        self.df = pd.DataFrame(arr)
        self.window = getattr(self.df, window)(**kwargs)
        self.window_group = getattr(pd.DataFrame({'A': groups, 'B': arr}).groupby('A'), window)(**kwargs)

    def time_pairwise(self, kwargs_window, method, pairwise):
        getattr(self.window, method)(self.df, pairwise=pairwise)

    def time_groupby(self, kwargs_window, method, pairwise):
        getattr(self.window_group, method)(self.df, pairwise=pairwise)

class Quantile:
    params: Tuple[List[str], List[int], List[str], List[float], List[str]] = (['DataFrame', 'Series'], [10, 1000], ['int', 'float'], [0.0, 0.5, 1.0], ['linear', 'nearest', 'lower', 'higher', 'midpoint'])
    param_names: List[str] = ['constructor', 'window', 'dtype', 'percentile', 'interpolation']
    roll: Any

    def setup(self, constructor, window, dtype, percentile, interpolation):
        N: int = 10 ** 5
        arr: np.ndarray = np.random.random(N).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)

    def time_quantile(self, constructor, window, dtype, percentile, interpolation):
        self.roll.quantile(percentile, interpolation=interpolation)

class Rank:
    params: Tuple[List[str], List[int], List[str], List[bool], List[bool], List[str]] = (['DataFrame', 'Series'], [10, 1000], ['int', 'float'], [True, False], [True, False], ['min', 'max', 'average'])
    param_names: List[str] = ['constructor', 'window', 'dtype', 'percentile', 'ascending', 'method']
    roll: Any

    def setup(self, constructor, window, dtype, percentile, ascending, method):
        N: int = 10 ** 5
        arr: np.ndarray = np.random.random(N).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)

    def time_rank(self, constructor, window, dtype, percentile, ascending, method):
        self.roll.rank(pct=percentile, ascending=ascending, method=method)

class PeakMemFixedWindowMinMax:
    params: List[str] = ['min', 'max']
    param_names: List[str] = ['operation']
    roll: pd.core.window.rolling.Rolling

    def setup(self, operation):
        N: int = 10 ** 6
        arr: np.ndarray = np.random.random(N)
        self.roll = pd.Series(arr).rolling(2)

    def peakmem_fixed(self, operation):
        for _ in range(5):
            getattr(self.roll, operation)()

class ForwardWindowMethods:
    params: Tuple[List[str], List[int], List[str], List[str]] = (['DataFrame', 'Series'], [10, 1000], ['int', 'float'], ['median', 'mean', 'max', 'min', 'kurt', 'sum'])
    param_names: List[str] = ['constructor', 'window_size', 'dtype', 'method']
    roll: Any

    def setup(self, constructor, window_size, dtype, method):
        N: int = 10 ** 5
        arr: np.ndarray = np.random.random(N).astype(dtype)
        indexer: pd.api.indexers.FixedForwardWindowIndexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        self.roll = getattr(pd, constructor)(arr).rolling(window=indexer)

    def time_rolling(self, constructor, window_size, dtype, method):
        getattr(self.roll, method)()

    def peakmem_rolling(self, constructor, window_size, dtype, method):
        getattr(self.roll, method)()

class Groupby:
    params: Tuple[List[str], List[Tuple[str, Dict[str, Any]]]] = (['sum', 'median', 'mean', 'max', 'min', 'kurt', 'sum'], [('rolling', {'window': 2}), ('rolling', {'window': '30s'}), ('expanding', {})])
    param_names: List[str] = ['method', 'window_kwargs']
    groupby_window: Any
    df: pd.DataFrame

    def setup(self, method, window_kwargs):
        N: int = 1000
        window, kwargs = window_kwargs
        df: pd.DataFrame = pd.DataFrame({'A': [str(i) for i in range(N)] * 10, 'B': list(range(N)) * 10})
        if isinstance(kwargs.get('window', None), str):
            df.index = pd.date_range(start='1900-01-01', freq='1min', periods=N * 10)
        self.groupby_window = getattr(df.groupby('A'), window)(**kwargs)

    def time_method(self, method, window_kwargs):
        getattr(self.groupby_window, method)()

class GroupbyLargeGroups:
    df: pd.DataFrame

    def setup(self):
        N: int = 100000
        self.df = pd.DataFrame({'A': [1, 2] * (N // 2), 'B': np.random.randn(N)})

    def time_rolling_multiindex_creation(self):
        self.df.groupby('A').rolling(3).mean()

class GroupbyEWM:
    params: List[str] = ['var', 'std', 'cov', 'corr']
    param_names: List[str] = ['method']
    gb_ewm: Any

    def setup(self, method):
        df: pd.DataFrame = pd.DataFrame({'A': range(50), 'B': range(50)})
        self.gb_ewm = df.groupby('A').ewm(com=1.0)

    def time_groupby_method(self, method):
        getattr(self.gb_ewm, method)()

class GroupbyEWMEngine:
    params: List[str] = ['cython', 'numba']
    param_names: List[str] = ['engine']
    gb_ewm: Any

    def setup(self, engine):
        df: pd.DataFrame = pd.DataFrame({'A': range(50), 'B': range(50)})
        self.gb_ewm = df.groupby('A').ewm(com=1.0)

    def time_groupby_mean(self, engine):
        self.gb_ewm.mean(engine=engine)

def table_method_func(x):
    return np.sum(x, axis=0) + 1

class TableMethod:
    params: Tuple[List[str],] = (['single', 'table'],)
    param_names: List[str] = ['method']
    df: pd.DataFrame

    def setup(self, method):
        self.df = pd.DataFrame(np.random.randn(10, 1000))

    def time_apply(self, method):
        self.df.rolling(2, method=method).apply(table_method_func, raw=True, engine='numba')

    def time_ewm_mean(self, method):
        self.df.ewm(1, method=method).mean(engine='numba')
from .pandas_vb_common import setup