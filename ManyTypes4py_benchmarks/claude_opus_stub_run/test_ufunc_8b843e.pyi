from collections import deque
from typing import Any

import numpy as np
import pytest

import pandas as pd
from pandas.arrays import SparseArray

@pytest.fixture
def ufunc(request: pytest.FixtureRequest) -> np.ufunc: ...

@pytest.fixture
def sparse(request: pytest.FixtureRequest) -> bool: ...

@pytest.fixture
def arrays_for_binary_ufunc() -> tuple[np.ndarray, np.ndarray]: ...

def test_unary_ufunc(ufunc: np.ufunc, sparse: bool) -> None: ...

def test_binary_ufunc_with_array(
    flip: bool,
    sparse: bool,
    ufunc: np.ufunc,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None: ...

def test_binary_ufunc_with_index(
    flip: bool,
    sparse: bool,
    ufunc: np.ufunc,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None: ...

def test_binary_ufunc_with_series(
    flip: bool,
    shuffle: bool,
    sparse: bool,
    ufunc: np.ufunc,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None: ...

def test_binary_ufunc_scalar(
    ufunc: np.ufunc,
    sparse: bool,
    flip: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None: ...

def test_multiple_output_binary_ufuncs(
    ufunc: np.ufunc,
    sparse: bool,
    shuffle: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None: ...

def test_multiple_output_ufunc(
    sparse: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None: ...

def test_binary_ufunc_drops_series_name(
    ufunc: np.ufunc,
    sparse: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None: ...

def test_object_series_ok() -> None: ...

@pytest.fixture
def values_for_np_reduce(request: pytest.FixtureRequest) -> Any: ...

class TestNumpyReductions:
    def test_multiply(
        self,
        values_for_np_reduce: Any,
        box_with_array: type,
        request: pytest.FixtureRequest,
    ) -> None: ...
    def test_add(
        self,
        values_for_np_reduce: Any,
        box_with_array: type,
    ) -> None: ...
    def test_max(
        self,
        values_for_np_reduce: Any,
        box_with_array: type,
    ) -> None: ...
    def test_min(
        self,
        values_for_np_reduce: Any,
        box_with_array: type,
    ) -> None: ...

def test_binary_ufunc_other_types(type_: type) -> None: ...

def test_object_dtype_ok() -> None: ...

def test_outer() -> None: ...

def test_np_matmul() -> None: ...

def test_np_matmul_1D(box: type) -> None: ...

def test_array_ufuncs_for_many_arguments() -> None: ...

def test_np_fix() -> None: ...