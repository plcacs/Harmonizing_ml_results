import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp
from pandas._testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
from typing import Any, Union, List, Tuple

@pytest.fixture
def interp_method(request: pytest.FixtureRequest) -> Tuple[str, str]:
    ...

class TestDataFrameQuantile:
    @pytest.mark.parametrize("df,expected", ...)
    def test_quantile_sparse(self, df: DataFrame, expected: Series) -> None:
        ...

    def test_quantile(
        self,
        datetime_frame: DataFrame,
        interp_method: Tuple[str, str],
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    def test_empty(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_non_numeric_exclusion(
        self,
        interp_method: Tuple[str, str],
        request: pytest.FixtureRequest,
    ) -> None:
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

    @pytest.mark.parametrize("dtype", ...)
    def test_quantile_dt64_empty(
        self,
        dtype: str,
        interp_method: Tuple[str, str],
    ) -> None:
        ...

    @pytest.mark.parametrize("invalid", ...)
    def test_quantile_invalid(
        self,
        invalid: Union[int, float, List[float]],
        datetime_frame: DataFrame,
        interp_method: Tuple[str, str],
    ) -> None:
        ...

    def test_quantile_box(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_box_nat(self) -> None:
        ...

    def test_quantile_nan(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_quantile_nat(
        self, interp_method: Tuple[str, str], unit: str
    ) -> None:
        ...

    def test_quantile_empty_no_rows_floats(
        self, interp_method: Tuple[str, str]
    ) -> None:
        ...

    def test_quantile_empty_no_rows_ints(
        self, interp_method: Tuple[str, str]
    ) -> None:
        ...

    def test_quantile_empty_no_rows_dt64(
        self, interp_method: Tuple[str, str]
    ) -> None:
        ...

    def test_quantile_empty_no_columns(
        self, interp_method: Tuple[str, str]
    ) -> None:
        ...

    def test_quantile_item_cache(self, interp_method: Tuple[str, str]) -> None:
        ...

    def test_invalid_method(self) -> None:
        ...

    def test_table_invalid_interpolation(self) -> None:
        ...

class TestQuantileExtensionDtype:
    @pytest.fixture
    def index(self, request: pytest.FixtureRequest) -> Index:
        ...

    @pytest.fixture
    def obj(
        self, index: Index, frame_or_series: Union[DataFrame, Series]
    ) -> Union[DataFrame, Series]:
        ...

    def compute_quantile(
        self, obj: Union[DataFrame, Series], qs: Union[float, List[float]]
    ) -> Union[DataFrame, Series]:
        ...

    def test_quantile_ea(
        self,
        request: pytest.FixtureRequest,
        obj: Union[DataFrame, Series],
        index: Index,
    ) -> None:
        ...

    def test_quantile_ea_with_na(
        self, obj: Union[DataFrame, Series], index: Index
    ) -> None:
        ...

    def test_quantile_ea_all_na(
        self,
        request: pytest.FixtureRequest,
        obj: Union[DataFrame, Series],
        index: Index,
    ) -> None:
        ...

    def test_quantile_ea_scalar(
        self,
        request: pytest.FixtureRequest,
        obj: Union[DataFrame, Series],
        index: Index,
    ) -> None:
        ...

    @pytest.mark.parametrize("dtype, expected_data, expected_index, axis", ...)
    def test_empty_numeric(
        self,
        dtype: str,
        expected_data: List[float],
        expected_index: List[str],
        axis: int,
    ) -> None:
        ...

    @pytest.mark.parametrize(
        "dtype, expected_data, expected_index, axis, expected_dtype", ...
    )
    def test_empty_datelike(
        self,
        dtype: str,
        expected_data: List[pd.Timestamp],
        expected_index: List[str],
        axis: int,
        expected_dtype: str,
    ) -> None:
        ...

    @pytest.mark.parametrize("expected_data, expected_index, axis", ...)
    def test_datelike_numeric_only(
        self,
        expected_data: List[float],
        expected_index: List[Union[str, int]],
        axis: int,
    ) -> None:
        ...

def test_multi_quantile_numeric_only_retains_columns() -> None:
    ...