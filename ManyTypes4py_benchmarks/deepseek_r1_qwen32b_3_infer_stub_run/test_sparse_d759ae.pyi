"""
Stub file for test_sparse_d759ae module
"""

import numpy as np
import pytest
from pandas import SparseDtype
from pandas.arrays import SparseArray
from pandas.tests.extension import base

def make_data(fill_value: np.floating | int) -> np.ndarray:
    ...

@pytest.fixture
def dtype() -> SparseDtype:
    ...

@pytest.fixture
def data(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture
def data_for_twos() -> SparseArray:
    ...

@pytest.fixture
def data_missing(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture
def data_repeated(request: pytest.FixtureRequest) -> callable:
    ...

@pytest.fixture
def data_for_sorting(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture
def data_missing_for_sorting(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture
def na_cmp() -> callable:
    ...

@pytest.fixture
def data_for_grouping(request: pytest.FixtureRequest) -> SparseArray:
    ...

@pytest.fixture
def data_for_compare(request: pytest.FixtureRequest) -> SparseArray:
    ...

class TestSparseArray(base.ExtensionTests):
    def _supports_reduction(self, obj: object, op_name: str) -> bool:
        ...

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_series_numeric(self, data: SparseArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest) -> None:
        ...

    @pytest.mark.parametrize('skipna', [True, False])
    def test_reduce_frame(self, data: SparseArray, all_numeric_reductions: str, skipna: bool, request: pytest.FixtureRequest) -> None:
        ...

    def _check_unsupported(self, data: SparseArray) -> None:
        ...

    def test_concat_mixed_dtypes(self, data: SparseArray) -> None:
        ...

    @pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
    @pytest.mark.parametrize('columns', [['A', 'B'], pd.MultiIndex])
    @pytest.mark.parametrize('future_stack', [True, False])
    def test_stack(self, data: SparseArray, columns: list[str] | pd.MultiIndex, future_stack: bool) -> None:
        ...

    def test_concat_columns(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_concat_extension_arrays_copy_false(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_align(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_align_frame(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_align_series_frame(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_merge(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_get(self, data: SparseArray) -> None:
        ...

    def test_reindex(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_isna(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_no_op_returns_copy(self, data: SparseArray) -> None:
        ...

    @pytest.mark.xfail
    def test_fillna_series(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_frame(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_limit_frame(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_limit_series(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_copy_frame(self, data_missing: SparseArray) -> None:
        ...

    def test_fillna_copy_series(self, data_missing: SparseArray) -> None:
        ...

    @pytest.mark.xfail
    def test_fillna_length_mismatch(self, data_missing: SparseArray) -> None:
        ...

    def test_where_series(self, data: SparseArray, na_value: object) -> None:
        ...

    def test_searchsorted(self, performance_warning: object, data_for_sorting: SparseArray, as_series: bool) -> None:
        ...

    def test_shift_0_periods(self, data: SparseArray) -> None:
        ...

    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    def test_argmin_argmax_all_na(self, method: str, data: SparseArray, na_value: object) -> None:
        ...

    @pytest.mark.fails_arm_wheels
    @pytest.mark.parametrize('box', [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data: SparseArray, na_value: object, as_series: bool, box: type) -> None:
        ...

    @pytest.mark.fails_arm_wheels
    def test_equals_same_data_different_object(self, data: SparseArray) -> None:
        ...

    @pytest.mark.parametrize('func, na_action, expected', [(lambda x: x, None, SparseArray), (lambda x: x, 'ignore', SparseArray), (str, None, SparseArray), (str, 'ignore', SparseArray)])
    def test_map(self, func: callable, na_action: str | None, expected: SparseArray) -> None:
        ...

    @pytest.mark.parametrize('na_action', [None, 'ignore'])
    def test_map_raises(self, data: SparseArray, na_action: str | None) -> None:
        ...

    @pytest.mark.xfail
    def test_astype_string(self, data: SparseArray, nullable_string_dtype: object) -> None:
        ...

    def test_arith_series_with_scalar(self, data: SparseArray, all_arithmetic_operators: str) -> None:
        ...

    def test_arith_series_with_array(self, data: SparseArray, all_arithmetic_operators: str) -> None:
        ...

    def test_arith_frame_with_scalar(self, data: SparseArray, all_arithmetic_operators: str, request: pytest.FixtureRequest) -> None:
        ...

    def _compare_other(self, ser: pd.Series, data_for_compare: SparseArray, comparison_op: callable, other: object) -> None:
        ...

    def test_scalar(self, data_for_compare: SparseArray, comparison_op: callable) -> None:
        ...

    def test_array(self, data_for_compare: SparseArray, comparison_op: callable, request: pytest.FixtureRequest) -> None:
        ...

    def test_sparse_array(self, data_for_compare: SparseArray, comparison_op: callable, request: pytest.FixtureRequest) -> None:
        ...

    @pytest.mark.xfail
    def test_array_repr(self, data: SparseArray, size: int) -> None:
        ...

    @pytest.mark.xfail
    @pytest.mark.parametrize('as_index', [True, False])
    def test_groupby_extension_agg(self, as_index: bool, data_for_grouping: SparseArray) -> None:
        ...

def test_array_type_with_arg(dtype: SparseDtype) -> None:
    ...