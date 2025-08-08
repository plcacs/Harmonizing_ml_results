from collections import deque
import re
import string
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray

@pytest.fixture(params=[np.add, np.logaddexp])
def ufunc(request: pytest.FixtureRequest) -> np.ufunc:
    return request.param

@pytest.fixture(params=[pytest.param(True, marks=pytest.mark.fails_arm_wheels), False], ids=['sparse', 'dense'])
def sparse(request: pytest.FixtureRequest) -> bool:
    return request.param

@pytest.fixture
def arrays_for_binary_ufunc() -> tuple:
    """
    A pair of random, length-100 integer-dtype arrays, that are mostly 0.
    """
    a1 = np.random.default_rng(2).integers(0, 10, 100, dtype='int64')
    a2 = np.random.default_rng(2).integers(0, 10, 100, dtype='int64')
    a1[::3] = 0
    a2[::4] = 0
    return (a1, a2)

@pytest.mark.parametrize('ufunc', [np.positive, np.floor, np.exp])
def test_unary_ufunc(ufunc: np.ufunc, sparse: bool) -> None:
    arr = np.random.default_rng(2).integers(0, 10, 10, dtype='int64')
    arr[::2] = 0
    if sparse:
        arr = SparseArray(arr, dtype=pd.SparseDtype('int64', 0))
    index = list(string.ascii_letters[:10])
    name = 'name'
    series = pd.Series(arr, index=index, name=name)
    result = ufunc(series)
    expected = pd.Series(ufunc(arr), index=index, name=name)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('flip', [True, False], ids=['flipped', 'straight'])
def test_binary_ufunc_with_array(flip: bool, sparse: bool, ufunc: np.ufunc, arrays_for_binary_ufunc: tuple) -> None:
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype('int64', 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype('int64', 0))
    name = 'name'
    series = pd.Series(a1, name=name)
    other = a2
    array_args = (a1, a2)
    series_args = (series, other)
    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)
    expected = pd.Series(ufunc(*array_args), name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)

# Remaining functions and tests are not included for brevity
