import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
from pandas.core.arrays.sparse import SparseArray

@pytest.fixture(params=['integer', 'block'])
def kind(request) -> str:
    ...

@pytest.fixture(params=[True, False])
def mix(request) -> bool:
    ...

class TestSparseArrayArithmetics:
    def _assert(self, a: np.ndarray, b: np.ndarray) -> None:
        ...

    def _check_numeric_ops(
        self, a: SparseArray, b: SparseArray, a_dense: np.ndarray, b_dense: np.ndarray,
        mix: bool, op: operator
    ) -> None:
        ...

    def _check_bool_result(self, res: SparseArray) -> None:
        ...

    def _check_comparison_ops(
        self, a: SparseArray, b: SparseArray, a_dense: np.ndarray, b_dense: np.ndarray
    ) -> None:
        ...

    def _check_logical_ops(
        self, a: SparseArray, b: SparseArray, a_dense: np.ndarray, b_dense: np.ndarray
    ) -> None:
        ...

    @pytest.mark.parametrize('scalar', [0, 1, 3])
    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_scalar(
        self, kind: str, mix: bool, all_arithmetic_functions: operator, fill_value: int | None,
        scalar: int, request: pytest.FixtureRequest
    ) -> None:
        ...

    def test_float_scalar_comparison(self, kind: str) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_same_index(self, kind: str, fill_value: bool | float) -> None:
        ...

    @pytest.mark.parametrize('fill_value', [True, False, np.nan])
    def test_bool_array_logical(self, kind: str, fill_value: bool | float) -> None:
        ...

    def test_xor(self) -> None:
        ...

@pytest.mark.parametrize('op', [operator.eq, operator.add])
def test_with_list(op: operator) -> None:
    ...

def test_with_dataframe() -> None:
    ...

def test_with_zerodim_ndarray() -> None:
    ...

@pytest.mark.parametrize('ufunc', [np.abs, np.exp])
@pytest.mark.parametrize('arr', [SparseArray([0, 0, -1, 1]), SparseArray([None, None, -1, 1])])
def test_ufuncs(ufunc: np.ufunc, arr: SparseArray) -> None:
    ...

@pytest.mark.parametrize('ufunc', [np.add, np.greater])
def test_binary_ufuncs(ufunc: np.ufunc, a: SparseArray, b: np.ndarray) -> None:
    ...

def test_ndarray_inplace() -> None:
    ...

def test_sparray_inplace() -> None:
    ...

@pytest.mark.parametrize('cons', [list, np.array, SparseArray])
def test_mismatched_length_cmp_op(cons: type) -> None:
    ...

@pytest.mark.parametrize('a, b', [([0, 1, 2], [0, 1, 2, 3]), ([0, 1, 2, 3], [0, 1, 2])])
def test_mismatched_length_arith_op(
    a: list[int], b: list[int], all_arithmetic_functions: operator
) -> None:
    ...

@pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'])
@pytest.mark.parametrize('fill_value', [np.nan, 3])
def test_binary_operators(op: str, fill_value: float | int) -> None:
    ...