import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
from pandas.core.arrays.sparse import SparseArray
from numpy.typing import NDArray
from typing import Any, Union, Optional, List, Callable, Tuple

@pytest.fixture
def kind(request) -> str:
    ...

@pytest.fixture
def mix(request) -> bool:
    ...

class TestSparseArrayArithmetics:
    def _assert(self, a: np.ndarray, b: np.ndarray) -> None:
        ...

    def _check_numeric_ops(
        self, a: SparseArray, b: Union[SparseArray, float], a_dense: np.ndarray,
        b_dense: Union[np.ndarray, float], mix: bool, op: Callable
    ) -> None:
        ...

    def _check_bool_result(self, res: SparseArray) -> None:
        ...

    def _check_comparison_ops(
        self, a: SparseArray, b: Union[SparseArray, float], a_dense: np.ndarray,
        b_dense: Union[np.ndarray, float]
    ) -> None:
        ...

    def _check_logical_ops(
        self, a: SparseArray, b: SparseArray, a_dense: np.ndarray,
        b_dense: np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('scalar', [0, 1, 3])
    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_scalar(
        self, kind: str, mix: bool, all_arithmetic_functions: Callable,
        fill_value: Optional[Union[int, float]], scalar: Union[int, float],
        request: pytest.FixtureRequest
    ) -> None:
        ...

    def test_float_scalar_comparison(self, kind: str) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_same_index_without_nans(
        self, kind: str, mix: bool, all_arithmetic_functions: Callable,
        fill_value: Optional[Union[int, float]]
    ) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_same_index_with_nans(
        self, kind: str, mix: bool, all_arithmetic_functions: Callable,
        request: pytest.FixtureRequest
    ) -> None:
        ...

    def test_float_same_index_comparison(self, kind: str) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_array(
        self, kind: str, mix: bool, all_arithmetic_functions: Callable,
        fill_value: Optional[Union[int, float]]
    ) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_array_different_kind(
        self, mix: bool, all_arithmetic_functions: Callable,
        fill_value: Optional[Union[int, float]]
    ) -> None:
        ...

    def test_float_array_comparison(self, kind: str) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_int_array(
        self, kind: str, mix: bool, all_arithmetic_functions: Callable,
        fill_value: Optional[Union[int, float]]
    ) -> None:
        ...

    def test_int_array_comparison(self, kind: str) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_same_index(
        self, kind: str, fill_value: Union[bool, float]
    ) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_array_logical(
        self, kind: str, fill_value: Union[bool, float]
    ) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_mixed_array_float_int(
        self, kind: str, mix: bool, all_arithmetic_functions: Callable,
        request: pytest.FixtureRequest, fill_value: Optional[Union[int, float]]
    ) -> None:
        ...

    def test_mixed_array_comparison(self, kind: str) -> None:
        ...

    def test_xor(self) -> None:
        ...

@pytest.mark.parametrize('op', [operator.eq, operator.add])
def test_with_list(op: Callable, arr: SparseArray, expected: SparseArray) -> None:
    ...

def test_with_dataframe(arr: SparseArray, df: pd.DataFrame) -> None:
    ...

def test_with_zerodim_ndarray(arr: SparseArray) -> None:
    ...

@pytest.mark.parametrize('ufunc', [np.abs, np.exp])
@pytest.mark.parametrize('arr', [SparseArray([0, 0, -1, 1]), SparseArray([None, None, -1, 1])])
def test_ufuncs(ufunc: Callable, arr: SparseArray) -> None:
    ...

@pytest.mark.parametrize('ufunc', [np.add, np.greater])
@pytest.mark.parametrize('a, b', [(SparseArray([0, 0, 0]), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2]))])
def test_binary_ufuncs(ufunc: Callable, a: SparseArray, b: np.ndarray) -> None:
    ...

def test_ndarray_inplace(sparray: SparseArray, ndarray: np.ndarray) -> None:
    ...

def test_sparray_inplace(sparray: SparseArray, ndarray: np.ndarray) -> None:
    ...

@pytest.mark.parametrize('cons', [list, np.array, SparseArray])
def test_mismatched_length_cmp_op(cons: Callable) -> None:
    ...

@pytest.mark.parametrize('a, b', [([0, 1, 2], [0, 1, 2, 3]), ([0, 1, 2, 3], [0, 1, 2])])
def test_mismatched_length_arith_op(
    a: List[int], b: List[int], all_arithmetic_functions: Callable
) -> None:
    ...

@pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'])
@pytest.mark.parametrize('fill_value', [np.nan, 3])
def test_binary_operators(op: str, fill_value: Union[float, int]) -> None:
    ...