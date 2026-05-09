from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
import pytest

@pytest.mark.parametrize('compare_func, roll_func, kwargs', [[np.mean, 'mean', {}], [np.nansum, 'sum', {}], [lambda x: np.isfinite(x).astype(float).sum(), 'count', {}], [np.median, 'median', {}], [np.min, 'min', {}], [np.max, 'max', {}], [lambda x: np.std(x, ddof=1), 'std', {}], [lambda x: np.std(x, ddof=0), 'std', {'ddof': 0}], [lambda x: np.var(x, ddof=1), 'var', {}], [lambda x: np.var(x, ddof=0), 'var', {'ddof': 0}]])
def test_series(series: Series, compare_func: Callable, roll_func: str, kwargs: Dict, step: int) -> None:
    ...

@pytest.mark.parametrize('compare_func, roll_func, kwargs', [[np.mean, 'mean', {}], [np.nansum, 'sum', {}], [lambda x: np.isfinite(x).astype(float).sum(), 'count', {}], [np.median, 'median', {}], [np.min, 'min', {}], [np.max, 'max', {}], [lambda x: np.std(x, ddof=1), 'std', {}], [lambda x: np.std(x, ddof=0), 'std', {'ddof': 0}], [lambda x: np.var(x, ddof=1), 'var', {}], [lambda x: np.var(x, ddof=0), 'var', {'ddof': 0}]])
def test_frame(raw: bool, frame: DataFrame, compare_func: Callable, roll_func: str, kwargs: Dict, step: int) -> None:
    ...

@pytest.mark.parametrize('compare_func, roll_func, kwargs, minp', [[np.mean, 'mean', {}, 10], [np.nansum, 'sum', {}, 10], [lambda x: np.isfinite(x).astype(float).sum(), 'count', {}, 0], [np.median, 'median', {}, 10], [np.min, 'min', {}, 10], [np.max, 'max', {}, 10], [lambda x: np.std(x, ddof=1), 'std', {}, 10], [lambda x: np.std(x, ddof=0), 'std', {'ddof': 0}, 10], [lambda x: np.var(x, ddof=1), 'var', {}, 10], [lambda x: np.var(x, ddof=0), 'var', {'ddof': 0}, 10]])
def test_time_rule_series(series: Series, compare_func: Callable, roll_func: str, kwargs: Dict, minp: int) -> None:
    ...

@pytest.mark.parametrize('compare_func, roll_func, kwargs, minp', [[np.mean, 'mean', {}, 10], [np.nansum, 'sum', {}, 10], [lambda x: np.isfinite(x).astype(float).sum(), 'count', {}, 0], [np.median, 'median', {}, 10], [np.min, 'min', {}, 10], [np.max, 'max', {}, 10], [lambda x: np.std(x, ddof=1), 'std', {}, 10], [lambda x: np.std(x, ddof=0), 'std', {'ddof': 0}, 10], [lambda x: np.var(x, ddof=1), 'var', {}, 10], [lambda x: np.var(x, ddof=0), 'var', {'ddof': 0}, 10]])
def test_time_rule_frame(raw: bool, frame: DataFrame, compare_func: Callable, roll_func: str, kwargs: Dict, minp: int) -> None:
    ...

@pytest.mark.parametrize('compare_func, roll_func, kwargs', [[np.mean, 'mean', {}], [np.nansum, 'sum', {}], [np.median, 'median', {}], [np.min, 'min', {}], [np.max, 'max', {}], [lambda x: np.std(x, ddof=1), 'std', {}], [lambda x: np.std(x, ddof=0), 'std', {'ddof': 0}], [lambda x: np.var(x, ddof=1), 'var', {}], [lambda x: np.var(x, ddof=0), 'var', {'ddof': 0}]])
def test_nans(compare_func: Callable, roll_func: str, kwargs: Dict) -> None:
    ...

def test_nans_count() -> None:
    ...

@pytest.mark.parametrize('roll_func, kwargs', [['mean', {}], ['sum', {}], ['median', {}], ['min', {}], ['max', {}], ['std', {}], ['std', {'ddof': 0}], ['var', {}], ['var', {'ddof': 0}]])
@pytest.mark.parametrize('minp', [0, 99, 100])
def test_min_periods(series: Series, minp: int, roll_func: str, kwargs: Dict, step: int) -> None:
    ...

def test_min_periods_count(series: Series, step: int) -> None:
    ...

@pytest.mark.parametrize('roll_func, kwargs, minp', [['mean', {}, 15], ['sum', {}, 15], ['count', {}, 0], ['median', {}, 15], ['min', {}, 15], ['max', {}, 15], ['std', {}, 15], ['std', {'ddof': 0}, 15], ['var', {}, 15], ['var', {'ddof': 0}, 15]])
def test_center(roll_func: str, kwargs: Dict, minp: int) -> None:
    ...

@pytest.mark.parametrize('roll_func, kwargs, minp, fill_value', [['mean', {}, 10, None], ['sum', {}, 10, None], ['count', {}, 0, 0], ['median', {}, 10, None], ['min', {}, 10, None], ['max', {}, 10, None], ['std', {}, 10, None], ['std', {'ddof': 0}, 10, None], ['var', {}, 10, None], ['var', {'ddof': 0}, 10, None]])
def test_center_reindex_series(series: Series, roll_func: str, kwargs: Dict, minp: int, fill_value: Optional[Any]) -> None:
    ...

@pytest.mark.parametrize('roll_func, kwargs, minp, fill_value', [['mean', {}, 10, None], ['sum', {}, 10, None], ['count', {}, 0, 0], ['median', {}, 10, None], ['min', {}, 10, None], ['max', {}, 10, None], ['std', {}, 10, None], ['std', {'ddof': 0}, 10, None], ['var', {}, 10, None], ['var', {'ddof': 0}, 10, None]])
def test_center_reindex_frame(frame: DataFrame, roll_func: str, kwargs: Dict, minp: int, fill_value: Optional[Any]) -> None:
    ...

@pytest.mark.parametrize('f', [lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False), lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False), lambda x: x.rolling(window=10, min_periods=5).max(), lambda x: x.rolling(window=10, min_periods=5).min(), lambda x: x.rolling(window=10, min_periods=5).sum(), lambda x: x.rolling(window=10, min_periods=5).mean(), lambda x: x.rolling(window=10, min_periods=5).std(), lambda x: x.rolling(window=10, min_periods=5).var(), lambda x: x.rolling(window=10, min_periods=5).skew(), lambda x: x.rolling(window=10, min_periods=5).kurt(), lambda x: x.rolling(window=10, min_periods=5).first(), lambda x: x.rolling(window=10, min_periods=5).last(), lambda x: x.rolling(window=10, min_periods=5).quantile(q=0.5), lambda x: x.rolling(window=10, min_periods=5).median(), lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False), lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True), pytest.param(lambda x: x.rolling(win_type='boxcar', window=10, min_periods=5).mean(), marks=td.skip_if_no('scipy'))])
def test_rolling_functions_window_non_shrinkage(f: Callable) -> None:
    ...

def test_rolling_max_gh6297(step: int) -> None:
    ...

def test_rolling_max_resample(step: int) -> None:
    ...

def test_rolling_min_resample(step: int) -> None:
    ...

def test_rolling_median_resample() -> None:
    ...

def test_rolling_median_memory_error() -> None:
    ...

def test_rolling_min_max_numeric_types(any_real_numpy_dtype: np.dtype) -> None:
    ...

@pytest.mark.parametrize('f', [lambda x: x.rolling(window=10, min_periods=0).count(), lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False), lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False), lambda x: x.rolling(window=10, min_periods=5).max(), lambda x: x.rolling(window=10, min_periods=5).min(), lambda x: x.rolling(window=10, min_periods=5).sum(), lambda x: x.rolling(window=10, min_periods=5).mean(), lambda x: x.rolling(window=10, min_periods=5).std(), lambda x: x.rolling(window=10, min_periods=5).var(), lambda x: x.rolling(window=10, min_periods=5).skew(), lambda x: x.rolling(window=10, min_periods=5).kurt(), lambda x: x.rolling(window=10, min_periods=5).first(), lambda x: x.rolling(window=10, min_periods=5).last(), lambda x: x.rolling(window=10, min_periods=5).quantile(0.5), lambda x: x.rolling(window=10, min_periods=5).median(), lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False), lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True), pytest.param(lambda x: x.rolling(win_type='boxcar', window=10, min_periods=5).mean(), marks=td.skip_if_no('scipy'))])
def test_moment_functions_zero_length(f: Callable) -> None:
    ...