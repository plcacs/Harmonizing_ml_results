from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
    DataFrame,
    Series,
    Index,
    MultiIndex,
    array,
)
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)

@pytest.fixture
def dtype() -> DecimalDtype:
    ...

@pytest.fixture
def data() -> DecimalArray:
    ...

@pytest.fixture
def data_for_twos() -> DecimalArray:
    ...

@pytest.fixture
def data_missing() -> DecimalArray:
    ...

@pytest.fixture
def data_for_sorting() -> DecimalArray:
    ...

@pytest.fixture
def data_missing_for_sorting() -> DecimalArray:
    ...

@pytest.fixture
def na_cmp() -> str:
    ...

@pytest.fixture
def data_for_grouping() -> DecimalArray:
    ...

class TestDecimalArray(base.ExtensionTests):
    def _get_expected_exception(self, op_name: str, obj: object, other: object) -> type[Exception] | None:
        ...

    def _supports_reduction(self, ser: Series, op_name: str) -> bool:
        ...

    def check_reduce(self, ser: Series, op_name: str, skipna: bool) -> None:
        ...

    def test_reduce_series_numeric(self, data: DecimalArray, all_numeric_reductions: str, skipna: bool) -> None:
        ...

    def test_reduce_frame(self, data: DecimalArray, all_numeric_reductions: str, skipna: bool) -> None:
        ...

    def test_compare_scalar(self, data: DecimalArray, comparison_op: str) -> None:
        ...

    def test_compare_array(self, data: DecimalArray, comparison_op: str) -> None:
        ...

    def test_arith_series_with_array(self, data: DecimalArray, all_arithmetic_operators: str) -> None:
        ...

    def test_fillna_frame(self, data_missing: DecimalArray) -> None:
        ...

    def test_fillna_series(self, data_missing: DecimalArray) -> None:
        ...

    def test_fillna_with_none(self, data_missing: DecimalArray) -> None:
        ...

    def test_fillna_limit_frame(self, data_missing: DecimalArray) -> None:
        ...

    def test_fillna_limit_series(self, data_missing: DecimalArray) -> None:
        ...

    @pytest.mark.parametrize('dropna', [True, False])
    def test_value_counts(self, all_data: DecimalArray, dropna: bool) -> None:
        ...

    def test_series_repr(self, data: DecimalArray) -> None:
        ...

    @pytest.mark.xfail(reason='Inconsistent array-vs-scalar behavior')
    @pytest.mark.parametrize('ufunc', [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data: DecimalArray, ufunc: type[np.ufunc]) -> None:
        ...

def test_take_na_value_other_decimal() -> None:
    ...

def test_series_constructor_coerce_data_to_extension_dtype() -> None:
    ...

def test_series_constructor_with_dtype() -> None:
    ...

def test_dataframe_constructor_with_dtype() -> None:
    ...

def test_astype_dispatches(frame_or_series: type[pd.DataFrame | pd.Series]) -> None:
    ...

class DecimalArrayWithoutFromSequence(DecimalArray):
    ...

class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):
    ...

def test_combine_from_sequence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

@pytest.mark.parametrize('class_', [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion])
def test_scalar_ops_from_sequence_raises(class_: type[DecimalArray]) -> None:
    ...

@pytest.mark.parametrize('reverse, expected_div, expected_mod', [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])])
def test_divmod_array(reverse: bool, expected_div: list[int], expected_mod: list[int]) -> None:
    ...

def test_ufunc_fallback(data: DecimalArray) -> None:
    ...

def test_array_ufunc() -> None:
    ...

def test_array_ufunc_series() -> None:
    ...

def test_array_ufunc_series_scalar_other() -> None:
    ...

def test_array_ufunc_series_defer() -> None:
    ...

def test_groupby_agg() -> None:
    ...

def test_groupby_agg_ea_method(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

def test_indexing_no_materialize(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

def test_to_numpy_keyword() -> None:
    ...

def test_array_copy_on_write() -> None:
    ...