import inspect
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Series, date_range
import pandas._testing as tm

class TestDatetimeLikeStatReductions:

    def test_dt64_mean(self, tz_naive_fixture: pd.tzinfo, index_or_series_or_array: callable) -> None:
    
    def test_period_mean(self, index_or_series_or_array: callable, freq: str) -> None:
    
    def test_td64_mean(self, index_or_series_or_array: callable) -> None:

class TestSeriesStatReductions:

    def _check_stat_op(self, name: str, alternate: callable, string_series_: Series, check_objects: bool = False, check_allna: bool = False) -> None:
    
    def test_sum(self) -> None:
    
    def test_mean(self) -> None:
    
    def test_median(self) -> None:
    
    def test_prod(self) -> None:
    
    def test_min(self) -> None:
    
    def test_max(self) -> None:
    
    def test_var_std(self) -> None:
    
    def test_sem(self) -> None:
    
    def test_skew(self) -> None:
    
    def test_kurt(self) -> None:
    
    def test_kurt_corner(self) -> None:
