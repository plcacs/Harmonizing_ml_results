import operator
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray


@pytest.fixture(params=["integer", "block"])
def kind(request: pytest.FixtureRequest) -> str:
    """kind kwarg to pass to SparseArray"""
    return request.param


@pytest.fixture(params=[True, False])
def mix(request: pytest.FixtureRequest) -> bool:
    """
    Fixture returning True or False, determining whether to operate
    op(sparse, dense) instead of op(sparse, sparse)
    """
    return request.param


class TestSparseArrayArithmetics:
    def _assert(self, a: np.ndarray, b: np.ndarray) -> None:
        # We have to use tm.assert_sp_array_equal. See GH #45126
        tm.assert_numpy_array_equal(a, b)

    def _check_numeric_ops(
        self,
        a: SparseArray,
        b: Union[SparseArray, np.ndarray, int, float],
        a_dense: np.ndarray,
        b_dense: Union[np.ndarray, int, float],
        mix: bool,
        op: Callable[[Any, Any], Any],
    ) -> None:
        # Check that arithmetic behavior matches non-Sparse Series arithmetic

        if isinstance(a_dense, np.ndarray):
            expected = op(pd.Series(a_dense), b_dense).values
        elif isinstance(b_dense, np.ndarray):
            expected = op(a_dense, pd.Series(b_dense)).values
        else:
            raise NotImplementedError

        with np.errstate(invalid="ignore", divide="ignore"):
            if mix:
                result = op(a, b_dense).to_dense()
            else:
                result = op(a, b).to_dense()

        self._assert(result, expected)

    def _check_bool_result(self, res: SparseArray) -> None:
        assert isinstance(res, SparseArray)
        assert isinstance(res.dtype, SparseDtype)
        assert res.dtype.subtype == np.bool_
        assert isinstance(res.fill_value, bool)

    def _check_comparison_ops(
        self,
        a: SparseArray,
        b: Union[SparseArray, np.ndarray, int, float],
        a_dense: np.ndarray,
        b_dense: Union[np.ndarray, int, float],
    ) -> None:
        with np.errstate(invalid="ignore"):
            # Unfortunately, trying to wrap the computation of each expected
            # value is with np.errstate() is too tedious.
            #
            # sparse & sparse
            self._check_bool_result(a == b)
            self._assert((a == b).to_dense(), a_dense == b_dense)

            self._check_bool_result(a != b)
            self._assert((a != b).to_dense(), a_dense != b_dense)

            self._check_bool_result(a >= b)
            self._assert((a >= b).to_dense(), a_dense >= b_dense)

            self._check_bool_result(a <= b)
            self._assert((a <= b).to_dense(), a_dense <= b_dense)

            self._check_bool_result(a > b)
            self._assert((a > b).to_dense(), a_dense > b_dense)

            self._check_bool_result(a < b)
            self._assert((a < b).to_dense(), a_dense < b_dense)

            # sparse & dense
            self._check_bool_result(a == b_dense)
            self._assert((a == b_dense).to_dense(), a_dense == b_dense)

            self._check_bool_result(a != b_dense)
            self._assert((a != b_dense).to_dense(), a_dense != b_dense)

            self._check_bool_result(a >= b_dense)
            self._assert((a >= b_dense).to_dense(), a_dense >= b_dense)

            self._check_bool_result(a <= b_dense)
            self._assert((a <= b_dense).to_dense(), a_dense <= b_dense)

            self._check_bool_result(a > b_dense)
            self._assert((a > b_dense).to_dense(), a_dense > b_dense)

            self._check_bool_result(a < b_dense)
            self._assert((a < b_dense).to_dense(), a_dense < b_dense)

    def _check_logical_ops(
        self,
        a: SparseArray,
        b: Union[SparseArray, np.ndarray],
        a_dense: np.ndarray,
        b_dense: np.ndarray,
    ) -> None:
        # sparse & sparse
        self._check_bool_result(a & b)
        self._assert((a & b).to_dense(), a_dense & b_dense)

        self._check_bool_result(a | b)
        self._assert((a | b).to_dense(), a_dense | b_dense)
        # sparse & dense
        self._check_bool_result(a & b_dense)
        self._assert((a & b_dense).to_dense(), a_dense & b_dense)

        self._check_bool_result(a | b_dense)
        self._assert((a | b_dense).to_dense(), a_dense | b_dense)

    @pytest.mark.parametrize("scalar", [0, 1, 3])
    @pytest.mark.parametrize("fill_value", [None, 0, 2])
    def test_float_scalar(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
        fill_value: Optional[Union[int, float]],
        scalar: Union[int, float],
        request: pytest.FixtureRequest,
    ) -> None:
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        a = SparseArray(values, kind=kind, fill_value=fill_value)
        self._check_numeric_ops(a, scalar, values, scalar, mix, op)

    def test_float_scalar_comparison(self, kind: str) -> None:
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])

        a = SparseArray(values, kind=kind)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

        a = SparseArray(values, kind=kind, fill_value=0)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

        a = SparseArray(values, kind=kind, fill_value=2)
        self._check_comparison_ops(a, 1, values, 1)
        self._check_comparison_ops(a, 0, values, 0)
        self._check_comparison_ops(a, 3, values, 3)

    def test_float_same_index_without_nans(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
    ) -> None:
        # when sp_index are the same
        op = all_arithmetic_functions

        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_with_nans(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
        request: pytest.FixtureRequest,
    ) -> None:
        # when sp_index are the same
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_same_index_comparison(self, kind: str) -> None:
        # when sp_index are the same
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_float_array(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
    ) -> None:
        op = all_arithmetic_functions

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_different_kind(
        self,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
    ) -> None:
        op = all_arithmetic_functions

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind="integer")
        b = SparseArray(rvalues, kind="block")
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block")
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block", fill_value=0)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, kind="integer", fill_value=1)
        b = SparseArray(rvalues, kind="block", fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_float_array_comparison(self, kind: str) -> None:
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    def test_int_array(
        self,
        kind: str,
        mix: bool,
        all_arithmetic_functions: Callable[[Any, Any], Any],
    ) -> None:
        op = all_arithmetic_functions

        # have to specify dtype explicitly until fixing GH 667
        dtype = np.int64

        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)

        a = SparseArray(values, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, fill_value=0, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        a = SparseArray(values, fill_value=1, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype, fill_value=1)
        b = SparseArray(rvalues, fill_value=2, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype, fill_value=2)
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_int_array_comparison(self, kind: str) -> None:
        dtype = "int64"
        # int32 NI ATM

        values = np.array([0, 1, 2, 0, 0, 0, 1,