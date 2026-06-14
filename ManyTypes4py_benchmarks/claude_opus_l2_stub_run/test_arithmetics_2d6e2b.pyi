import operator
from typing import Any, Callable

import numpy as np
import pytest

from pandas.core.arrays.sparse import SparseArray

@pytest.fixture(params=["integer", "block"])
def kind(request: pytest.FixtureRequest) -> str: ...

@pytest.fixture(params=[True, False])
def mix(request: pytest.FixtureRequest) -> bool: ...

class TestSparseArrayArithmetics:
    def _assert(self, a: np.ndarray, b: np.ndarray) -> None: ...
    def _check_numeric_ops(
        self,
        a: SparseArray,
        b: Any,
        a_dense: np.ndarray,
        b_dense: Any,
        mix: bool,
        op: Callable[..., Any],
    ) -> None: ...
    def _check_bool_result(self, res: SparseArray) -> None: ...
    def _check_comparison_ops(
        self,
        a: SparseArray,
        b: Any,
        a_dense: np.ndarray,
        b_dense: Any,
    ) -> None: ...
    def _check_logical_ops(
        self,
        a: SparseArray,
        b: SparseArray,
        a_dense: np.ndarray,
        b_dense: np.ndarray,
    ) -> None: ...
    def test_float_scalar(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[..., Any],
        fill_value: float | None,
        scalar: int,
        request: pytest.FixtureRequest,
    ) -> None: ...
    def test_float_scalar_comparison(self, kind: str) -> None: ...
    def test_float_same_index_without_nans(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[..., Any],
    ) -> None: ...
    def test_float_same_index_with_nans(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[..., Any],
        request: pytest.FixtureRequest,
    ) -> None: ...
    def test_float_same_index_comparison(self, kind: str) -> None: ...
    def test_float_array(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[..., Any],
    ) -> None: ...
    def test_float_array_different_kind(
        self,
        mix: bool,
        all_arithmetic_functions: Callable[..., Any],
    ) -> None: ...
    def test_float_array_comparison(self, kind: str) -> None: ...
    def test_int_array(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[..., Any],
    ) -> None: ...
    def test_int_array_comparison(self, kind: str) -> None: ...
    def test_bool_same_index(self, kind: str, fill_value: bool | float) -> None: ...
    def test_bool_array_logical(self, kind: str, fill_value: bool | float) -> None: ...
    def test_mixed_array_float_int(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[..., Any],
        request: pytest.FixtureRequest,
    ) -> None: ...
    def test_mixed_array_comparison(self, kind: str) -> None: ...
    def test_xor(self) -> None: ...

def test_with_list(op: Callable[..., Any]) -> None: ...
def test_with_dataframe() -> None: ...
def test_with_zerodim_ndarray() -> None: ...
def test_ufuncs(ufunc: Callable[..., Any], arr: SparseArray) -> None: ...
def test_binary_ufuncs(ufunc: Callable[..., Any], a: SparseArray, b: np.ndarray) -> None: ...
def test_ndarray_inplace() -> None: ...
def test_sparray_inplace() -> None: ...
def test_mismatched_length_cmp_op(cons: type) -> None: ...
def test_mismatched_length_arith_op(
    a: list[int], b: list[int], all_arithmetic_functions: Callable[..., Any]
) -> None: ...
def test_binary_operators(op: str, fill_value: float) -> None: ...