from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame, DatetimeIndex, Series, concat, isna, notna
import pandas._testing as tm
from pandas.tseries import offsets

@pytest.mark.parametrize('compare_func, roll_func, kwargs', [[np.mean, 'mean', {}], [np.nansum, 'sum', {}], [lambda x: np.isfinite(x).astype(float).sum(), 'count', {}], [np.median, 'median', {}], [np.min, 'min', {}], [np.max, 'max', {}], [lambda x: np.std(x, ddof=1), 'std', {}], [lambda x: np.std(x, ddof=0), 'std', {'ddof': 0}], [lambda x: np.var(x, ddof=1), 'var', {}], [lambda x: np.var(x, ddof=0), 'var', {'ddof': 0}]], 
                                                             type: List[Tuple[Callable[[np.ndarray], np.number], str, Dict[str, Any]]])
def test_series(series: Series, compare_func: Callable[[np.ndarray], np.number], roll_func: str, kwargs: Dict[str, Any], step: int) -> None:
    # ...

@pytest.mark.parametrize('compare_func, roll_func, kwargs', [[np.mean, 'mean', {}], [np.nansum, 'sum', {}], [lambda x: np.isfinite(x).astype(float).sum(), 'count', {}], [np.median, 'median', {}], [np.min, 'min', {}], [np.max, 'max', {}], [lambda x: np.std(x, ddof=1), 'std', {}], [lambda x: np.std(x, ddof=0), 'std', {'ddof': 0}], [lambda x: np.var(x, ddof=1), 'var', {}], [lambda x: np.var(x, ddof=0), 'var', {'ddof': 0}]], 
                                                             type: List[Tuple[Callable[[np.ndarray], np.number], str, Dict[str, Any]]])
def test_frame(raw: Any, frame: DataFrame, compare_func: Callable[[np.ndarray], np.number], roll_func: str, kwargs: Dict[str, Any], step: int) -> None:
    # ...

@pytest.mark.parametrize('roll_func, kwargs, minp', [[np.mean, {}, 10], [np.nansum, {}, 10], [lambda x: np.isfinite(x).astype(float).sum(), 'count', {}], [np.median, {}, 10], [np.min, {}, 10], [np.max, {}, 10], [lambda x: np.std(x, ddof=1), {}, 10], [lambda x: np.std(x, ddof=0), {'ddof': 0}, 10], [lambda x: np.var(x, ddof=1), {}, 10], [lambda x: np.var(x, ddof=0), {'ddof': 0}, 10]], 
                                                      type: List[Tuple[Callable[[np.ndarray], np.number], Dict[str, Any], int]])
def test_min_periods(series: Series, roll_func: Callable[[np.ndarray], np.number], kwargs: Dict[str, Any], minp: int) -> None:
    # ...

@pytest.mark.parametrize('f', [lambda x: x.rolling(window=10, min_periods=0).cov(x, pairwise=False), lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False), lambda x: x.rolling(window=10, min_periods=5).max(), lambda x: x.rolling(window=10, min_periods=5).min(), lambda x: x.rolling(window=10, min_periods=5).sum(), lambda x: x.rolling(window=10, min_periods=5).mean(), lambda x: x.rolling(window=10, min_periods=5).std(), lambda x: x.rolling(window=10, min_periods=5).var(), lambda x: x.rolling(window=10, min_periods=5).skew(), lambda x: x.rolling(window=10, min_periods=5).kurt(), lambda x: x.rolling(window=10, min_periods=5).first(), lambda x: x.rolling(window=10, min_periods=5).last(), lambda x: x.rolling(window=10, min_periods=5).quantile(0.5), lambda x: x.rolling(window=10, min_periods=5).median(), lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False), lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True), pytest.param(lambda x: x.rolling(win_type='boxcar', window=10, min_periods=5).mean(), marks=td.skip_if_no('scipy'))], 
                              type: List[Callable[[Series], Series]]
def test_moment_functions_zero_length(f: Callable[[Series], Series]) -> None:
    # ...
