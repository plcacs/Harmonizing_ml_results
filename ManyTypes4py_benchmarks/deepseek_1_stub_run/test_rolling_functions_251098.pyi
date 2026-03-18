```python
from typing import Any, Callable, Literal, TypeVar
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
from pandas.tseries import offsets

_T = TypeVar("_T")

def test_series(
    series: Series,
    compare_func: Callable[[Any], Any],
    roll_func: str,
    kwargs: dict[str, Any],
    step: int,
) -> None: ...

def test_frame(
    raw: bool,
    frame: DataFrame,
    compare_func: Callable[[Any], Any],
    roll_func: str,
    kwargs: dict[str, Any],
    step: int,
) -> None: ...

def test_time_rule_series(
    series: Series,
    compare_func: Callable[[Any], Any],
    roll_func: str,
    kwargs: dict[str, Any],
    minp: int,
) -> None: ...

def test_time_rule_frame(
    raw: bool,
    frame: DataFrame,
    compare_func: Callable[[Any], Any],
    roll_func: str,
    kwargs: dict[str, Any],
    minp: int,
) -> None: ...

def test_nans(
    compare_func: Callable[[Any], Any],
    roll_func: str,
    kwargs: dict[str, Any],
) -> None: ...

def test_nans_count() -> None: ...

def test_min_periods(
    series: Series,
    minp: int,
    roll_func: str,
    kwargs: dict[str, Any],
    step: int,
) -> None: ...

def test_min_periods_count(
    series: Series,
    step: int,
) -> None: ...

def test_center(
    roll_func: str,
    kwargs: dict[str, Any],
    minp: int,
) -> None: ...

def test_center_reindex_series(
    series: Series,
    roll_func: str,
    kwargs: dict[str, Any],
    minp: int,
    fill_value: Any,
) -> None: ...

def test_center_reindex_frame(
    frame: DataFrame,
    roll_func: str,
    kwargs: dict[str, Any],
    minp: int,
    fill_value: Any,
) -> None: ...

def test_rolling_functions_window_non_shrinkage(
    f: Callable[[Any], Any],
) -> None: ...

def test_rolling_max_gh6297(
    step: int,
) -> None: ...

def test_rolling_max_resample(
    step: int,
) -> None: ...

def test_rolling_min_resample(
    step: int,
) -> None: ...

def test_rolling_median_resample() -> None: ...

def test_rolling_median_memory_error() -> None: ...

def test_rolling_min_max_numeric_types(
    any_real_numpy_dtype: Any,
) -> None: ...

def test_moment_functions_zero_length(
    f: Callable[[Any], Any],
) -> None: ...
```