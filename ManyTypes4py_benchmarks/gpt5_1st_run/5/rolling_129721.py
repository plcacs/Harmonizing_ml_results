import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class Methods:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        [('rolling', {'window': 10}), ('rolling', {'window': 1000}), ('expanding', {})],
        ['int', 'float'],
        ['median', 'mean', 'max', 'min', 'std', 'count', 'skew', 'kurt', 'sum', 'sem'],
    )
    param_names: List[str] = ['constructor', 'window_kwargs', 'dtype', 'method']

    def setup(
        self,
        constructor: str,
        window_kwargs: Tuple[str, Dict[str, Any]],
        dtype: str,
        method: str,
    ) -> None:
        N: int = 10 ** 5
        window, kwargs = window_kwargs
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        obj: Union[pd.DataFrame, pd.Series] = getattr(pd, constructor)(arr)
        self.window: Any = getattr(obj, window)(**kwargs)

    def time_method(
        self,
        constructor: str,
        window_kwargs: Tuple[str, Dict[str, Any]],
        dtype: str,
        method: str,
    ) -> None:
        getattr(self.window, method)()

    def peakmem_method(
        self,
        constructor: str,
        window_kwargs: Tuple[str, Dict[str, Any]],
        dtype: str,
        method: str,
    ) -> None:
        getattr(self.window, method)()


class Apply:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        [3, 300],
        ['int', 'float'],
        [sum, np.sum, lambda x: np.sum(x) + 5],
        [True, False],
    )
    param_names: List[str] = ['constructor', 'window', 'dtype', 'function', 'raw']

    def setup(
        self,
        constructor: str,
        window: int,
        dtype: str,
        function: Callable[[Any], Any],
        raw: bool,
    ) -> None:
        N: int = 10 ** 3
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        self.roll: Any = getattr(pd, constructor)(arr).rolling(window)

    def time_rolling(
        self,
        constructor: str,
        window: int,
        dtype: str,
        function: Callable[[Any], Any],
        raw: bool,
    ) -> None:
        self.roll.apply(function, raw=raw)


class NumbaEngineMethods:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        ['int', 'float'],
        [('rolling', {'window': 10}), ('expanding', {})],
        ['sum', 'max', 'min', 'median', 'mean', 'var', 'std'],
        [True, False],
        [None, 100],
    )
    param_names: List[str] = ['constructor', 'dtype', 'window_kwargs', 'method', 'parallel', 'cols']

    def setup(
        self,
        constructor: str,
        dtype: str,
        window_kwargs: Tuple[str, Dict[str, Any]],
        method: str,
        parallel: bool,
        cols: Optional[int],
    ) -> None:
        N: int = 10 ** 3
        window, kwargs = window_kwargs
        shape: Union[int, Tuple[int, int]] = (N, cols) if cols is not None and constructor != 'Series' else N
        arr: np.ndarray = (100 * np.random.random(shape)).astype(dtype)
        data: Union[pd.DataFrame, pd.Series] = getattr(pd, constructor)(arr)
        with warnings.catch_warnings(record=True):
            self.window: Any = getattr(data, window)(**kwargs)
            getattr(self.window, method)(engine='numba', engine_kwargs={'parallel': parallel})

    def test_method(
        self,
        constructor: str,
        dtype: str,
        window_kwargs: Tuple[str, Dict[str, Any]],
        method: str,
        parallel: bool,
        cols: Optional[int],
    ) -> None:
        with warnings.catch_warnings(record=True):
            getattr(self.window, method)(engine='numba', engine_kwargs={'parallel': parallel})


class NumbaEngineApply:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        ['int', 'float'],
        [('rolling', {'window': 10}), ('expanding', {})],
        [np.sum, lambda x: np.sum(x) + 5],
        [True, False],
        [None, 100],
    )
    param_names: List[str] = ['constructor', 'dtype', 'window_kwargs', 'function', 'parallel', 'cols']

    def setup(
        self,
        constructor: str,
        dtype: str,
        window_kwargs: Tuple[str, Dict[str, Any]],
        function: Callable[[Any], Any],
        parallel: bool,
        cols: Optional[int],
    ) -> None:
        N: int = 10 ** 3
        window, kwargs = window_kwargs
        shape: Union[int, Tuple[int, int]] = (N, cols) if cols is not None and constructor != 'Series' else N
        arr: np.ndarray = (100 * np.random.random(shape)).astype(dtype)
        data: Union[pd.DataFrame, pd.Series] = getattr(pd, constructor)(arr)
        with warnings.catch_warnings(record=True):
            self.window: Any = getattr(data, window)(**kwargs)
            self.window.apply(function, raw=True, engine='numba', engine_kwargs={'parallel': parallel})

    def test_method(
        self,
        constructor: str,
        dtype: str,
        window_kwargs: Tuple[str, Dict[str, Any]],
        function: Callable[[Any], Any],
        parallel: bool,
        cols: Optional[int],
    ) -> None:
        with warnings.catch_warnings(record=True):
            self.window.apply(function, raw=True, engine='numba', engine_kwargs={'parallel': parallel})


class EWMMethods:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        [
            ({'halflife': 10}, 'mean'),
            ({'halflife': 10}, 'std'),
            ({'halflife': 1000}, 'mean'),
            ({'halflife': 1000}, 'std'),
            ({'halflife': '1 Day', 'times': pd.date_range('1900', periods=10 ** 5, freq='23s')}, 'mean'),
        ],
        ['int', 'float'],
    )
    param_names: List[str] = ['constructor', 'kwargs_method', 'dtype']

    def setup(
        self,
        constructor: str,
        kwargs_method: Tuple[Dict[str, Any], str],
        dtype: str,
    ) -> None:
        N: int = 10 ** 5
        kwargs, method = kwargs_method
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        self.method: str = method
        self.ewm: Any = getattr(pd, constructor)(arr).ewm(**kwargs)

    def time_ewm(
        self,
        constructor: str,
        kwargs_method: Tuple[Dict[str, Any], str],
        dtype: str,
    ) -> None:
        getattr(self.ewm, self.method)()


class VariableWindowMethods(Methods):
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        ['50s', '1h', '1d'],
        ['int', 'float'],
        ['median', 'mean', 'max', 'min', 'std', 'count', 'skew', 'kurt', 'sum', 'sem'],
    )
    param_names: List[str] = ['constructor', 'window', 'dtype', 'method']

    def setup(
        self,
        constructor: str,
        window: str,
        dtype: str,
        method: str,
    ) -> None:
        N: int = 10 ** 5
        arr: np.ndarray = (100 * np.random.random(N)).astype(dtype)
        index: pd.DatetimeIndex = pd.date_range('2017-01-01', periods=N, freq='5s')
        self.window = getattr(pd, constructor)(arr, index=index).rolling(window)


class Pairwise:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any]] = (
        [({'window': 10}, 'rolling'), ({'window': 1000}, 'rolling'), ({}, 'expanding')],
        ['corr', 'cov'],
        [True, False],
    )
    param_names: List[str] = ['window_kwargs', 'method', 'pairwise']

    def setup(
        self,
        kwargs_window: Tuple[Dict[str, Any], str],
        method: str,
        pairwise: bool,
    ) -> None:
        N: int = 10 ** 4
        n_groups: int = 20
        kwargs, window = kwargs_window
        groups: List[int] = [i for _ in range(N // n_groups) for i in range(n_groups)]
        arr: np.ndarray = np.random.random(N)
        self.df: pd.DataFrame = pd.DataFrame(arr)
        self.window: Any = getattr(self.df, window)(**kwargs)
        self.window_group: Any = getattr(pd.DataFrame({'A': groups, 'B': arr}).groupby('A'), window)(**kwargs)

    def time_pairwise(
        self,
        kwargs_window: Tuple[Dict[str, Any], str],
        method: str,
        pairwise: bool,
    ) -> None:
        getattr(self.window, method)(self.df, pairwise=pairwise)

    def time_groupby(
        self,
        kwargs_window: Tuple[Dict[str, Any], str],
        method: str,
        pairwise: bool,
    ) -> None:
        getattr(self.window_group, method)(self.df, pairwise=pairwise)


class Quantile:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        [10, 1000],
        ['int', 'float'],
        [0, 0.5, 1],
        ['linear', 'nearest', 'lower', 'higher', 'midpoint'],
    )
    param_names: List[str] = ['constructor', 'window', 'dtype', 'percentile']

    def setup(
        self,
        constructor: str,
        window: int,
        dtype: str,
        percentile: float,
        interpolation: str,
    ) -> None:
        N: int = 10 ** 5
        arr: np.ndarray = np.random.random(N).astype(dtype)
        self.roll: Any = getattr(pd, constructor)(arr).rolling(window)

    def time_quantile(
        self,
        constructor: str,
        window: int,
        dtype: str,
        percentile: float,
        interpolation: str,
    ) -> None:
        self.roll.quantile(percentile, interpolation=interpolation)


class Rank:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        [10, 1000],
        ['int', 'float'],
        [True, False],
        [True, False],
        ['min', 'max', 'average'],
    )
    param_names: List[str] = ['constructor', 'window', 'dtype', 'percentile', 'ascending', 'method']

    def setup(
        self,
        constructor: str,
        window: int,
        dtype: str,
        percentile: bool,
        ascending: bool,
        method: str,
    ) -> None:
        N: int = 10 ** 5
        arr: np.ndarray = np.random.random(N).astype(dtype)
        self.roll: Any = getattr(pd, constructor)(arr).rolling(window)

    def time_rank(
        self,
        constructor: str,
        window: int,
        dtype: str,
        percentile: bool,
        ascending: bool,
        method: str,
    ) -> None:
        self.roll.rank(pct=percentile, ascending=ascending, method=method)


class PeakMemFixedWindowMinMax:
    params: Sequence[str] = ['min', 'max']

    def setup(self, operation: str) -> None:
        N: int = 10 ** 6
        arr: np.ndarray = np.random.random(N)
        self.roll: Any = pd.Series(arr).rolling(2)

    def peakmem_fixed(self, operation: str) -> None:
        for _ in range(5):
            getattr(self.roll, operation)()


class ForwardWindowMethods:
    params: Tuple[Sequence[Any], Sequence[Any], Sequence[Any], Sequence[Any]] = (
        ['DataFrame', 'Series'],
        [10, 1000],
        ['int', 'float'],
        ['median', 'mean', 'max', 'min', 'kurt', 'sum'],
    )
    param_names: List[str] = ['constructor', 'window_size', 'dtype', 'method']

    def setup(
        self,
        constructor: str,
        window_size: int,
        dtype: str,
        method: str,
    ) -> None:
        N: int = 10 ** 5
        arr: np.ndarray = np.random.random(N).astype(dtype)
        indexer: Any = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        self.roll: Any = getattr(pd, constructor)(arr).rolling(window=indexer)

    def time_rolling(
        self,
        constructor: str,
        window_size: int,
        dtype: str,
        method: str,
    ) -> None:
        getattr(self.roll, method)()

    def peakmem_rolling(
        self,
        constructor: str,
        window_size: int,
        dtype: str,
        method: str,
    ) -> None:
        getattr(self.roll, method)()


class Groupby:
    params: Tuple[Sequence[Any], Sequence[Any]] = (
        ['sum', 'median', 'mean', 'max', 'min', 'kurt', 'sum'],
        [('rolling', {'window': 2}), ('rolling', {'window': '30s'}), ('expanding', {})],
    )

    def setup(self, method: str, window_kwargs: Tuple[str, Dict[str, Any]]) -> None:
        N: int = 1000
        window, kwargs = window_kwargs
        df: pd.DataFrame = pd.DataFrame({'A': [str(i) for i in range(N)] * 10, 'B': list(range(N)) * 10})
        if isinstance(kwargs.get('window', None), str):
            df.index = pd.date_range(start='1900-01-01', freq='1min', periods=N * 10)
        self.groupby_window: Any = getattr(df.groupby('A'), window)(**kwargs)

    def time_method(self, method: str, window_kwargs: Tuple[str, Dict[str, Any]]) -> None:
        getattr(self.groupby_window, method)()


class GroupbyLargeGroups:
    def setup(self) -> None:
        N: int = 100000
        self.df: pd.DataFrame = pd.DataFrame({'A': [1, 2] * (N // 2), 'B': np.random.randn(N)})

    def time_rolling_multiindex_creation(self) -> None:
        self.df.groupby('A').rolling(3).mean()


class GroupbyEWM:
    params: Sequence[str] = ['var', 'std', 'cov', 'corr']
    param_names: List[str] = ['method']

    def setup(self, method: str) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': range(50), 'B': range(50)})
        self.gb_ewm: Any = df.groupby('A').ewm(com=1.0)

    def time_groupby_method(self, method: str) -> None:
        getattr(self.gb_ewm, method)()


class GroupbyEWMEngine:
    params: Sequence[str] = ['cython', 'numba']
    param_names: List[str] = ['engine']

    def setup(self, engine: str) -> None:
        df: pd.DataFrame = pd.DataFrame({'A': range(50), 'B': range(50)})
        self.gb_ewm: Any = df.groupby('A').ewm(com=1.0)

    def time_groupby_mean(self, engine: str) -> None:
        self.gb_ewm.mean(engine=engine)


def table_method_func(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=0) + 1


class TableMethod:
    params: Sequence[str] = ['single', 'table']
    param_names: List[str] = ['method']

    def setup(self, method: str) -> None:
        self.df: pd.DataFrame = pd.DataFrame(np.random.randn(10, 1000))

    def time_apply(self, method: str) -> None:
        self.df.rolling(2, method=method).apply(table_method_func, raw=True, engine='numba')

    def time_ewm_mean(self, method: str) -> None:
        self.df.ewm(1, method=method).mean(engine='numba')


from .pandas_vb_common import setup