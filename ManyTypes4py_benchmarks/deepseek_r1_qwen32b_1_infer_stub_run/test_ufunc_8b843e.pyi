from collections import deque
import re
import string
import numpy as np
import pytest
import pandas as pd
from pandas.arrays import SparseArray
from pandas._testing import tm

@pytest.fixture
def ufunc(request) -> np.ufunc:
    ...

@pytest.fixture
def sparse(request) -> bool:
    ...

@pytest.fixture
def arrays_for_binary_ufunc() -> tuple[np.ndarray[np.int64], np.ndarray[np.int64]]:
    ...

@pytest.mark.parametrize('ufunc', [np.positive, np.floor, np.exp])
def test_unary_ufunc(ufunc: np.ufunc, sparse: bool) -> None:
    ...

@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_array(flip: bool, sparse: bool, ufunc: np.ufunc, arrays_for_binary_ufunc: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]) -> None:
    ...

@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_index(flip: bool, sparse: bool, ufunc: np.ufunc, arrays_for_binary_ufunc: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]) -> None:
    ...

@pytest.mark.parametrize('shuffle', [True, False], ids=['unaligned', 'aligned'])
@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_series(flip: bool, shuffle: bool, sparse: bool, ufunc: np.ufunc, arrays_for_binary_ufunc: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]) -> None:
    ...

@pytest.mark.parametrize('flip', [True, False])
def test_binary_ufunc_scalar(ufunc: np.ufunc, sparse: bool, flip: bool, arrays_for_binary_ufunc: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]) -> None:
    ...

@pytest.mark.parametrize('ufunc', [np.divmod])
@pytest.mark.parametrize('shuffle', [True, False])
def test_multiple_output_binary_ufuncs(ufunc: np.ufunc, sparse: bool, shuffle: bool, arrays_for_binary_ufunc: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]) -> None:
    ...

def test_multiple_output_ufunc(sparse: bool, arrays_for_binary_ufunc: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]) -> None:
    ...

def test_binary_ufunc_drops_series_name(ufunc: np.ufunc, sparse: bool, arrays_for_binary_ufunc: tuple[np.ndarray[np.int64], np.ndarray[np.int64]]) -> None:
    ...

def test_object_series_ok() -> None:
    ...

@pytest.fixture
def values_for_np_reduce(request) -> pd.arrays.NumpyExtensionArray:
    ...

class TestNumpyReductions:
    def test_multiply(self, values_for_np_reduce: pd.arrays.NumpyExtensionArray, box_with_array: type[pd.DataFrame | pd.Series | pd.Index], request: pytest.FixtureRequest) -> None:
        ...
    
    def test_add(self, values_for_np_reduce: pd.arrays.NumpyExtensionArray, box_with_array: type[pd.DataFrame | pd.Series | pd.Index]) -> None:
        ...
    
    def test_max(self, values_for_np_reduce: pd.arrays.NumpyExtensionArray, box_with_array: type[pd.DataFrame | pd.Series | pd.Index]) -> None:
        ...
    
    def test_min(self, values_for_np_reduce: pd.arrays.NumpyExtensionArray, box_with_array: type[pd.DataFrame | pd.Series | pd.Index]) -> None:
        ...

@pytest.mark.parametrize('type_', [list, deque, tuple])
def test_binary_ufunc_other_types(type_: type[list | deque | tuple]) -> None:
    ...

def test_object_dtype_ok() -> None:
    ...

def test_outer() -> None:
    ...

def test_np_matmul() -> None:
    ...

@pytest.mark.parametrize('box', [pd.Index, pd.Series])
def test_np_matmul_1D(box: type[pd.Index | pd.Series]) -> None:
    ...

def test_array_ufuncs_for_many_arguments() -> None:
    ...

@pytest.mark.xfail
def test_np_fix() -> None:
    ...