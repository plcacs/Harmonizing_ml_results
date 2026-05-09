import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray

@pytest.fixture(params=['integer', 'block'])
def kind(request):
    """kind kwarg to pass to SparseArray"""
    return request.param

@pytest.fixture(params=[True, False])
def mix(request):
    """
    Fixture returning True or False, determining whether to operate
    op(sparse, dense) instead of op(sparse, sparse)
    """
    return request.param

class TestSparseArrayArithmetics:
    def _assert(self, a: SparseArray, b: np.ndarray) -> None:
        tm.assert_numpy_array_equal(a, b)

    def _check_numeric_ops(self, a: SparseArray, b: SparseArray, a_dense: np.ndarray, b_dense: np.ndarray, mix: bool, op: operator) -> None:
        # ...

    def _check_bool_result(self, res: SparseArray) -> None:
        # ...

    # ...

    @pytest.mark.parametrize('scalar', [0, 1, 3])
    @pytest.mark.parametrize('fill_value', [None, 0, 2])
    def test_float_scalar(self, kind: str, mix: bool, all_arithmetic_functions: operator, fill_value: float, scalar: float, request: pytest.Request) -> None:
        # ...

    def test_float_array(self, kind: str, mix: bool, all_arithmetic_functions: operator) -> None:
        # ...

    def test_int_array(self, kind: str, mix: bool, all_arithmetic_functions: operator) -> None:
        # ...

    def test_bool_same_index(self, kind: str, fill_value: bool) -> None:
        # ...

    def test_mixed_array_float_int(self, kind: str, mix: bool, all_arithmetic_functions: operator, request: pytest.Request) -> None:
        # ...

    def test_with_list(self, op: operator) -> None:
        # ...

    def test_with_dataframe(self) -> None:
        # ...

    def test_with_zerodim_ndarray(self) -> None:
        # ...

    @pytest.mark.parametrize('ufunc', [np.abs, np.exp])
    @pytest.mark.parametrize('arr', [SparseArray([0, 0, -1, 1]), SparseArray([None, None, -1, 1])])
    def test_ufuncs(self, ufunc: np.ufunc, arr: SparseArray) -> None:
        # ...

    @pytest.mark.parametrize('a, b', [(SparseArray([0, 0, 0]), np.array([0, 1, 2])), (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2]))])
    @pytest.mark.parametrize('ufunc', [np.add, np.greater])
    def test_binary_ufuncs(self, ufunc: np.ufunc, a: SparseArray, b: np.ndarray) -> None:
        # ...

    def test_ndarray_inplace(self) -> None:
        # ...

    def test_sparray_inplace(self) -> None:
        # ...

    @pytest.mark.parametrize('cons', [list, np.array, SparseArray])
    def test_mismatched_length_cmp_op(self, cons: type) -> None:
        # ...

    @pytest.mark.parametrize('a, b', [([0, 1, 2], [0, 1, 2, 3]), ([0, 1, 2, 3], [0, 1, 2])])
    def test_mismatched_length_arith_op(self, a: list, b: list, all_arithmetic_functions: operator) -> None:
        # ...

    @pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'truediv', 'floordiv', 'pow'])
    @pytest.mark.parametrize('fill_value', [np.nan, 3])
    def test_binary_operators(self, op: str, fill_value: float) -> None:
        # ...
