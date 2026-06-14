from collections import deque
from typing import Any

import numpy as np
import pytest

import pandas as pd

@pytest.fixture(params=[np.add, np.logaddexp])
def ufunc(request: pytest.FixtureRequest) -> np.ufunc: ...

@pytest.fixture(params=[pytest.param(True, marks=pytest.mark.fails_arm_wheels), False], ids=["sparse", "dense"])
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

@pytest.fixture(
    params=[
        pd.array([1, 3, 2], dtype=np.int64),
        pd.array([1, 3, 2], dtype="Int64"),
        pd.array([1, 3, 2], dtype="Float32"),
        pd.array([1, 10, 2], dtype="Sparse[int]"),
        pd.to_datetime(["2000", "2010", "2001"]),
        pd.to_datetime(["2000", "2010", "2001"]).tz_localize("CET"),
        pd.to_datetime(["2000", "2010", "2001"]).to_period(freq="D"),
        pd.to_timedelta(["1 Day", "3 Days", "2 Days"]),
        pd.IntervalIndex([pd.Interval(0, 1), pd.Interval(2, 3), pd.Interval(1, 2)]),
    ],
    ids=lambda x: str(x.dtype),
)
def values_for_np_reduce(request: pytest.FixtureRequest) -> Any: ...

class TestNumpyReductions:
    def test_multiply(self, values_for_np_reduce: Any, box_with_array: Any, request: pytest.FixtureRequest) -> None: ...
    def test_add(self, values_for_np_reduce: Any, box_with_array: Any) -> None: ...
    def test_max(self, values_for_np_reduce: Any, box_with_array: Any) -> None: ...
    def test_min(self, values_for_np_reduce: Any, box_with_array: Any) -> None: ...

def test_binary_ufunc_other_types(type_: type) -> None: ...

def test_object_dtype_ok() -> None: ...

def test_outer() -> None: ...

def test_np_matmul() -> None: ...

def test_np_matmul_1D(box: type[pd.Index] | type[pd.Series]) -> None: ...

def test_array_ufuncs_for_many_arguments() -> None: ...

def test_np_fix() -> None: ...