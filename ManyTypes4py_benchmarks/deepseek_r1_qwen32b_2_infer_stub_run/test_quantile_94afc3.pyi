from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp
from pandas._testing import tm
import pytest

@pytest.fixture
def interp_method() -> Tuple[str, str]:
    ...

class TestDataFrameQuantile:

    def test_quantile_sparse(self, df: pd.DataFrame, expected: pd.Series) -> None:
        ...

    def test_quantile(self, datetime_frame: pd.DataFrame, interp_method: Tuple[str, str], request: pytest.FixtureRequest) -> None:
        ...

    def test_empty(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_non_numeric_exclusion(self, interp_method: Tuple[str, str], request: pytest.FixtureRequest) -> None:
        ...

    def test_axis(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_axis_numeric_only_true(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_date_range(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_axis_mixed(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_axis_parameter(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_interpolation(self) -> None:
        ...

    def test_quantile_interpolation_datetime(self, datetime_frame: pd.DataFrame) -> None:
        ...

    def test_quantile_interpolation_int(self, int_frame: pd.DataFrame) -> None:
        ...

    def test_quantile_multi(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_multi_axis_1(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_multi_empty(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_datetime(self, unit: str) -> None:
        ...

    def test_quantile_dt64_empty(self, dtype: str, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_invalid(self, invalid: Any, datetime_frame: pd.DataFrame, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_box(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_box_nat(self) -> None:
        ...

    def test_quantile_nan(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_nat(self, interp_method: Tuple[str, str], unit: str) -> None:
        ...

    def test_quantile_empty_no_rows_floats(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_empty_no_rows_ints(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_empty_no_rows_dt64(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_empty_no_columns(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_item_cache(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_invalid_method(self) -> None:
        ...

    def test_table_invalid_interpolation(self) -> None:
        ...

class TestQuantileExtensionDtype:

    @pytest.fixture
    def index(self) -> pd.Index:
        ...

    @pytest.fixture
    def obj(self, index: pd.Index) -> Union[pd.DataFrame, pd.Series]:
        ...

    def compute_quantile(self, obj: Union[pd.DataFrame, pd.Series], qs: Union[float, List[float]]) -> Union[pd.Series, pd.DataFrame]:
        ...

    def test_quantile_ea(self, request: pytest.FixtureRequest, obj: Union[pd.DataFrame, pd.Series], index: pd.Index) -> None:
        ...

    def test_quantile_ea_with_na(self, obj: Union[pd.DataFrame, pd.Series], index: pd.Index) -> None:
        ...

    def test_quantile_ea_all_na(self, request: pytest.FixtureRequest, obj: Union[pd.DataFrame, pd.Series], index: pd.Index) -> None:
        ...

    def test_quantile_ea_scalar(self, request: pytest.FixtureRequest, obj: Union[pd.DataFrame, pd.Series], index: pd.Index) -> None:
        ...

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis', [
        ('float64', [], [], 1),
        ('int64', [], [], 1),
        ('float64', [np.nan, np.nan], ['a', 'b'], 0),
        ('int64', [np.nan, np.nan], ['a', 'b'], 0)
    ])
    def test_empty_numeric(self, dtype: str, expected_data: List[float], expected_index: List[str], axis: int) -> None:
        ...

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis, expected_dtype', [
        ('datetime64[ns]', [], [], 1, 'datetime64[ns]'),
        ('datetime64[ns]', [pd.NaT, pd.NaT], ['a', 'b'], 0, 'datetime64[ns]')
    ])
    def test_empty_datelike(self, dtype: str, expected_data: List[Any], expected_index: List[str], axis: int, expected_dtype: str) -> None:
        ...

    @pytest.mark.parametrize('expected_data, expected_index, axis', [
        [[np.nan, np.nan], range(2), 1],
        [[], [], 0]
    ])
    def test_datelike_numeric_only(self, expected_data: List[Any], expected_index: List[Any], axis: int) -> None:
        ...

def test_multi_quantile_numeric_only_retains_columns() -> None:
    ...