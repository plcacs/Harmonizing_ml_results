import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from typing import List, Tuple, Any

methods: dict[str, Any] = {'cumsum': np.cumsum, 'cumprod': np.cumprod, 'cummin': np.minimum.accumulate, 'cummax': np.maximum.accumulate}

class TestSeriesCumulativeOps:

    def test_datetime_series(self, datetime_series: pd.Series, func: Any) -> None:

    def test_cummin_cummax(self, datetime_series: pd.Series, method: str) -> None:

    def test_cummin_cummax_datetimelike(self, ts: pd.Timestamp, method: str, skipna: bool, exp_tdi: List[str]) -> None:

    def test_cumsum_datetimelike(self) -> None:

    def test_cummin_cummax_period(self, func: str, exp: str) -> None:

    def test_cummethods_bool(self, arg: List[bool], func: Any, method: str) -> None:

    def test_cummethods_bool_in_object_dtype(self, method: str, expected: pd.Series) -> None:

    def test_cummax_cummin_on_ordered_categorical(self, method: str, order: str) -> None:

    def test_cummax_cummin_ordered_categorical_nan(self, skip: bool, exp: List[str], method: str, order: str) -> None:

    def test_cumprod_timedelta(self) -> None:

    def test_cum_methods_pyarrow_strings(self, pyarrow_string_dtype: Any, data: List[Any], op: str, skipna: bool, expected_data: List[Any]) -> None:

    def test_cumprod_pyarrow_strings(self, pyarrow_string_dtype: Any, skipna: bool) -> None:
