import numpy as np
import pandas as pd
from pandas import SparseDtype
from pandas.core.arrays.sparse import SparseArray
import operator
from typing import Any, Callable, Literal, Union, Optional

def __getattr__(name: str) -> Any: ...

@pytest: Any

@pytest.fixture(params: list[str] = ...)
def kind(request: Any) -> str: ...

@pytest.fixture(params: list[bool] = ...)
def mix(request: Any) -> bool: ...

class TestSparseArrayArithmetics:
    def _assert(self, a: np.ndarray, b: np.ndarray) -> None: ...
    def _check_numeric_ops(
        self,
        a: SparseArray,
        b: Union[SparseArray, np.ndarray, int, float],
        a_dense: Union[np.ndarray, pd.Series],
        b_dense: Union[np.ndarray, pd.Series, int, float],
        mix: bool,
        op: Callable[[Any, Any], Any]
    ) -> None: ...
    def _check_bool_result(self, res: SparseArray) -> None: ...
    def _check_comparison_ops(
        self,
        a: SparseArray,
        b: Union[SparseArray, np.ndarray, int, float],
        a_dense: np.ndarray,
        b_dense: np.ndarray
    ) -> None: ...
    def _check_logical_ops(
        self,
        a: SparseArray,
        b: Union[SparseArray, np.ndarray],
        a_dense: np.ndarray,
        b_dense: np.ndarray
    ) -> None: ...
    def test_float_scalar(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
        fill_value: Optional[Union[int, float]],
        scalar: int,
        request: Any
    ) -> None: ...
    def test_float_scalar_comparison(self, kind: str) -> None: ...
    def test_float_same_index_without_nans(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any]
    ) -> None: ...
    def test_float_same_index_with_nans(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
        request: Any
    ) -> None: ...
    def test_float_same_index_comparison(self, kind: str) -> None: ...
    def test_float_array(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any]
    ) -> None: ...
    def test_float_array_different_kind(
        self,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any]
    ) -> None: ...
    def test_float_array_comparison(self, kind: str) -> None: ...
    def test_int_array(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any]
    ) -> None: ...
    def test_int_array_comparison(self, kind: str) -> None: ...
    def test_bool_same_index(
        self,
        kind: str,
        fill_value: Union[bool, float]
    ) -> None: ...
    def test_bool_array_logical(
        self,
        kind: str,
        fill_value: Union[bool, float]
    ) -> None: ...
    def test_mixed_array_float_int(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
        request: Any
    ) -> None: ...
    def test_mixed_array_comparison(self, kind: str) -> None: ...
    def test_xor(self) -> None: ...

@pytest.mark.parametrize("op", ...)
def test_with_list(op: Callable[[Any, Any], Any]) -> None: ...

def test_with_dataframe() -> None: ...

def test_with_zerodim_ndarray() -> None: ...

@pytest.mark.parametrize("ufunc", ...)
@pytest.mark.parametrize("arr", ...)
def test_ufuncs(ufunc: Callable[[Any], Any], arr: SparseArray) -> None: ...

@pytest.mark.parametrize("a, b", ...)
@pytest.mark.parametrize("ufunc", ...)
def test_binary_ufuncs(
    ufunc: Callable[[Any, Any], Any],
    a: SparseArray,
    b: np.ndarray
) -> None: ...

def test_ndarray_inplace() -> None: ...

def test_sparray_inplace() -> None: ...

@pytest.mark.parametrize("cons", ...)
def test_mismatched_length_cmp_op(cons: Callable[[Any], Any]) -> None: ...

@pytest.mark.parametrize("a, b", ...)
def test_mismatched_length_arith_op(
    a: list[int],
    b: list[int],
    all_arithmetic_functions: Callable[[Any, Any], Any]
) -> None: ...

@pytest.mark.parametrize("op", ...)
@pytest.mark.parametrize("fill_value", ...)
def test_binary_operators(
    op: str,
    fill_value: Union[float, int]
) -> None: ...