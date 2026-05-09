from collections import deque
import re
import string
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray

@pytest.fixture(params=[np.add, np.logaddexp])
def ufunc(request: pytest.FixtureParam) -> np.ufunc:
    return request.param

@pytest.fixture(params=[pytest.param(True, marks=pytest.mark.fails_arm_wheels), False], ids=['sparse', 'dense'])
def sparse(request: pytest.FixtureParam) -> bool:
    return request.param

@pytest.fixture
def arrays_for_binary_ufunc() -> tuple[np.ndarray, np.ndarray]:
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
    arr: np.ndarray = np.random.default_rng(2).integers(0, 10, 10, dtype='int64')
    arr[::2] = 0
    if sparse:
        arr = SparseArray(arr, dtype=pd.SparseDtype('int64', 0))
    index: list[str] = list(string.ascii_letters[:10])
    name: str = 'name'
    series: pd.Series = pd.Series(arr, index=index, name=name)
    result: pd.Series = ufunc(series)
    expected: pd.Series = pd.Series(ufunc(arr), index=index, name=name)
    tm.assert_series_equal(result, expected)

# ... (rest of the code remains the same)
