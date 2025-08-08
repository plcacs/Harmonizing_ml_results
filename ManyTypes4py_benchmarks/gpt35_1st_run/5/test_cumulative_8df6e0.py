import re
from typing import List, Tuple
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm

methods: dict[str, callable] = {'cumsum': np.cumsum, 'cumprod': np.cumprod, 'cummin': np.minimum.accumulate, 'cummax': np.maximum.accumulate}

class TestSeriesCumulativeOps:

    def test_datetime_series(self, datetime_series: pd.Series, func: callable) -> None:
    
    def test_cummin_cummax(self, datetime_series: pd.Series, method: str) -> None:
    
    def test_cummin_cummax_datetimelike(self, ts: pd.Timestamp, method: str, skipna: bool, exp_tdi: List[str]) -> None:
    
    def test_cumsum_datetimelike(self) -> None:
    
    def test_cummin_cummax_period(self, func: str, exp: str) -> None:
    
    def test_cummethods_bool(self, arg: List[bool], func: callable, method: str) -> None:
    
    def test_cummethods_bool_in_object_dtype(self, method: str, expected: pd.Series) -> None:
    
    def test_cummax_cummin_on_ordered_categorical(self, method: str, order: str) -> None:
    
    def test_cummax_cummin_ordered_categorical_nan(self, skip: bool, exp: List[str], method: str, order: str) -> None:
    
    def test_cumprod_timedelta(self) -> None:
    
    def test_cum_methods_pyarrow_strings(self, pyarrow_string_dtype, data: List[str], op: str, skipna: bool, expected_data: List[str]) -> None:
    
    def test_cumprod_pyarrow_strings(self, pyarrow_string_dtype, skipna: bool) -> None:
