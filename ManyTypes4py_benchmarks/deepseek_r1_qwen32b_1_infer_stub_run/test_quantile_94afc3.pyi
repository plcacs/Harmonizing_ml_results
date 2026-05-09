import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Series, Index, Timestamp
from typing import Tuple, List, Optional

@pytest.fixture(params=[['linear', 'single'], ['nearest', 'table']], ids=lambda x: '-'.join(x))
def interp_method(request) -> Tuple[str, str]:
    ...

class TestDataFrameQuantile:

    @pytest.mark.parametrize('df,expected', [[DataFrame({0: Series(pd.arrays.SparseArray([1, 2])), 1: Series(pd.arrays.SparseArray([3, 4]))}), Series([1.5, 3.5], name=0.5)], [DataFrame(Series([0.0, None, 1.0, 2.0], dtype='Sparse[float]')), Series([1.0], name=0.5)]])
    def test_quantile_sparse(self, df: DataFrame, expected: Series) -> None:
        ...

    def test_quantile(self, datetime_frame: DataFrame, interp_method: Tuple[str, str], request: pytest.FixtureRequest) -> None:
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

    def test_quantile_interpolation_datetime(self, datetime_frame: DataFrame) -> None:
        ...

    def test_quantile_interpolation_int(self, int_frame: DataFrame) -> None:
        ...

    def test_quantile_multi(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_multi_axis_1(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_multi_empty(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_datetime(self, unit: str) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['datetime64[ns]', 'datetime64[ns, US/Pacific]', 'timedelta64[ns]', 'Period[D]'])
    def test_quantile_dt64_empty(self, dtype: str, interp_method: Tuple[str, str]) -> None:
        ...

    @pytest.mark.parametrize('invalid', [-1, 2, [0.5, -1], [0.5, 2]])
    def test_quantile_invalid(self, invalid: float, datetime_frame: DataFrame, interp_method: Tuple[str, str]) -> None:
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
    def obj(self, index: pd.Index, frame_or_series: str) -> Union[Series, DataFrame]:
        ...

    def compute_quantile(self, obj: Union[Series, DataFrame], qs: List[float]) -> Union[Series, DataFrame]:
        ...

    def test_quantile_ea(self, request: pytest.FixtureRequest, obj: Union[Series, DataFrame], index: pd.Index) -> None:
        ...

    def test_quantile_ea_with_na(self, obj: Union[Series, DataFrame], index: pd.Index) -> None:
        ...

    def test_quantile_ea_all_na(self, request: pytest.FixtureRequest, obj: Union[Series, DataFrame], index: pd.Index) -> None:
        ...

    def test_quantile_ea_scalar(self, request: pytest.FixtureRequest, obj: Union[Series, DataFrame], index: pd.Index) -> None:
        ...

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis', [['float64', [], [], 1], ['int64', [], [], 1], ['float64', [np.nan, np.nan], ['a', 'b'], 0], ['int64', [np.nan, np.nan], ['a', 'b'], 0]])
    def test_empty_numeric(self, dtype: str, expected_data: List[float], expected_index: List[str], axis: int) -> None:
        ...

    @pytest.mark.parametrize('dtype, expected_data, expected_index, axis, expected_dtype', [['datetime64[ns]', [], [], 1, 'datetime64[ns]'], ['datetime64[ns]', [pd.NaT, pd.NaT], ['a', 'b'], 0, 'datetime64[ns]']])
    def test_empty_datelike(self, dtype: str, expected_data: List[pd.Timestamp], expected_index: List[str], axis: int, expected_dtype: str) -> None:
        ...

    @pytest.mark.parametrize('expected_data, expected_index, axis', [[[np.nan, np.nan], range(2), 1], [[], [], 0]])
    def test_datelike_numeric_only(self, expected_data: List[float], expected_index: List[Union[str, int]], axis: int) -> None:
        ...

def test_multi_quantile_numeric_only_retains_columns() -> None:
    ...