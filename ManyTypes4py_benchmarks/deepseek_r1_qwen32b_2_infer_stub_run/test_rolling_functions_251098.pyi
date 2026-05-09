from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, DatetimeIndex, Series
from pandas.tseries import offsets

def test_series(series: Series, compare_func: Callable[[np.ndarray], float], roll_func: str, kwargs: Dict[str, Any], step: int) -> None:
    ...

def test_frame(raw: bool, frame: DataFrame, compare_func: Callable[[np.ndarray], float], roll_func: str, kwargs: Dict[str, Any], step: int) -> None:
    ...

def test_time_rule_series(series: Series, compare_func: Callable[[np.ndarray], float], roll_func: str, kwargs: Dict[str, Any], minp: int) -> None:
    ...

def test_time_rule_frame(raw: bool, frame: DataFrame, compare_func: Callable[[np.ndarray], float], roll_func: str, kwargs: Dict[str, Any], minp: int) -> None:
    ...

def test_nans(compare_func: Callable[[np.ndarray], float], roll_func: str, kwargs: Dict[str, Any]) -> None:
    ...

def test_nans_count() -> None:
    ...

def test_min_periods(series: Series, minp: int, roll_func: str, kwargs: Dict[str, Any], step: int) -> None:
    ...

def test_min_periods_count(series: Series, step: int) -> None:
    ...

def test_center(roll_func: str, kwargs: Dict[str, Any], minp: int) -> None:
    ...

def test_center_reindex_series(series: Series, roll_func: str, kwargs: Dict[str, Any], minp: int, fill_value: Optional[Any]) -> None:
    ...

def test_center_reindex_frame(frame: DataFrame, roll_func: str, kwargs: Dict[str, Any], minp: int, fill_value: Optional[Any]) -> None:
    ...

def test_rolling_functions_window_non_shrinkage(f: Callable[[Union[Series, DataFrame]], Union[Series, DataFrame]]) -> None:
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

def test_moment_functions_zero_length(f: Callable[[Union[Series, DataFrame]], Union[Series, DataFrame]]) -> None:
    ...