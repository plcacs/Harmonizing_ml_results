"""
Stub file for test_string_c0dad8.py
"""

from __future__ import annotations
import pytest
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base

@pytest.fixture
def chunked(request) -> bool:
    ...

@pytest.fixture
def dtype(string_dtype_arguments) -> StringDtype:
    ...

@pytest.fixture
def data(dtype, chunked) -> Union[StringDtype, ArrowStringArray]:
    ...

@pytest.fixture
def data_missing(dtype, chunked) -> Union[StringDtype, ArrowStringArray]:
    ...

@pytest.fixture
def data_for_sorting(dtype, chunked) -> Union[StringDtype, ArrowStringArray]:
    ...

@pytest.fixture
def data_missing_for_sorting(dtype, chunked) -> Union[StringDtype, ArrowStringArray]:
    ...

@pytest.fixture
def data_for_grouping(dtype, chunked) -> Union[StringDtype, ArrowStringArray]:
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

    def test_view(self, data: Union[StringDtype, ArrowStringArray]) -> None:
        ...

    def test_from_dtype(self, data: Union[StringDtype, ArrowStringArray]) -> None:
        ...

    def test_transpose(self, data: Union[StringDtype, ArrowStringArray]) -> None:
        ...

    def test_setitem_preserves_views(self, data: Union[StringDtype, ArrowStringArray]) -> None:
        ...

    def test_dropna_array(self, data_missing: Union[StringDtype, ArrowStringArray]) -> None:
        ...

    def test_fillna_no_op_returns_copy(self, data: Union[StringDtype, ArrowStringArray]) -> None:
        ...

    def _get_expected_exception(self, op_name: str, obj: Any, other: Any) -> Optional[Type[Exception]]:
        ...

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        ...

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        ...

    def _cast_pointwise_result(self, op_name: str, obj: Any, other: Any, pointwise_result: Any) -> Any:
        ...

    def test_compare_scalar(self, data: Union[StringDtype, ArrowStringArray], comparison_op: Any) -> None:
        ...

    def test_groupby_extension_apply(self, data_for_grouping: Union[StringDtype, ArrowStringArray], groupby_apply_op: Any) -> None:
        ...

    def test_combine_add(self, data_repeated: Any, using_infer_string: bool, request: pytest.FixtureRequest) -> None:
        ...

    def test_arith_series_with_array(self, data: Union[StringDtype, ArrowStringArray], all_arithmetic_operators: str, using_infer_string: bool, request: pytest.FixtureRequest) -> None:
        ...

class Test2DCompat(base.Dim2CompatTests):
    @pytest.fixture(autouse=True)
    def arrow_not_supported(self, data: Union[StringDtype, ArrowStringArray]) -> None:
        ...

def test_searchsorted_with_na_raises(data_for_sorting: Union[StringDtype, ArrowStringArray], as_series: bool) -> None:
    ...