from __future__ import annotations
from typing import cast, List, Tuple
import numpy as np
import pytest
import pandas as pd
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base

def maybe_split_array(arr: pd.Series, chunked: bool) -> pd.Series:
    ...

@pytest.fixture(params=[True, False])
def chunked(request: pytest.FixtureRequest) -> bool:
    ...

@pytest.fixture
def dtype(string_dtype_arguments: Tuple[str, str]) -> StringDtype:
    ...

@pytest.fixture
def data(dtype: StringDtype, chunked: bool) -> pd.Series:
    ...

@pytest.fixture
def data_missing(dtype: StringDtype, chunked: bool) -> pd.Series:
    ...

@pytest.fixture
def data_for_sorting(dtype: StringDtype, chunked: bool) -> pd.Series:
    ...

@pytest.fixture
def data_missing_for_sorting(dtype: StringDtype, chunked: bool) -> pd.Series:
    ...

@pytest.fixture
def data_for_grouping(dtype: StringDtype, chunked: bool) -> pd.Series:
    ...

class TestStringArray(base.ExtensionTests):

    def test_eq_with_str(self, dtype: StringDtype) -> None:
        ...

    def test_is_not_string_type(self, dtype: StringDtype) -> None:
        ...

    def test_is_dtype_from_name(self, dtype: StringDtype, using_infer_string: bool) -> None:
        ...

    def test_construct_from_string_own_name(self, dtype: StringDtype, using_infer_string: bool) -> None:
        ...

    def test_view(self, data: pd.Series) -> None:
        ...

    def test_from_dtype(self, data: pd.Series) -> None:
        ...

    def test_transpose(self, data: pd.Series) -> None:
        ...

    def test_setitem_preserves_views(self, data: pd.Series) -> None:
        ...

    def test_dropna_array(self, data_missing: pd.Series) -> None:
        ...

    def test_fillna_no_op_returns_copy(self, data: pd.Series) -> None:
        ...

    def _get_expected_exception(self, op_name: str, obj: pd.Series, other: pd.Series) -> None:
        ...

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> None:
        ...

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> None:
        ...

    def _cast_pointwise_result(self, op_name: str, obj: pd.Series, other: pd.Series, pointwise_result: pd.Series) -> None:
        ...

    def test_compare_scalar(self, data: pd.Series, comparison_op: str) -> None:
        ...

    def test_groupby_extension_apply(self, data_for_grouping: pd.Series, groupby_apply_op: str) -> None:
        ...

    def test_combine_add(self, data_repeated: List[pd.Series], using_infer_string: bool, request: pytest.FixtureRequest) -> None:
        ...

    def test_arith_series_with_array(self, data: pd.Series, all_arithmetic_operators: str, using_infer_string: bool, request: pytest.FixtureRequest) -> None:
        ...

class Test2DCompat(base.Dim2CompatTests):

    @pytest.fixture(autouse=True)
    def arrow_not_supported(self, data: pd.Series) -> None:
        ...

def test_searchsorted_with_na_raises(data_for_sorting: pd.Series, as_series: bool) -> None:
    ...
