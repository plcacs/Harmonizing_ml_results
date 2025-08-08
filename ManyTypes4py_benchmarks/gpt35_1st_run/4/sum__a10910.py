from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple
import numba
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt

@numba.jit(nopython=True, nogil=True, parallel=False)
def add_sum(val: float, nobs: int, sum_x: float, compensation: float, num_consecutive_same_value: int, prev_value: float) -> Tuple[int, float, float, int, float]:
    ...

@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_sum(val: float, nobs: int, sum_x: float, compensation: float) -> Tuple[int, float, float]:
    ...

@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_sum(values: np.ndarray, result_dtype: np.dtype, start: np.ndarray, end: np.ndarray, min_periods: int) -> Tuple[np.ndarray, list]:
    ...

@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_kahan_sum(values: np.ndarray, result_dtype: np.dtype, labels: np.ndarray, ngroups: int, skipna: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...

@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_sum(values: np.ndarray, result_dtype: np.dtype, labels: np.ndarray, ngroups: int, min_periods: int, skipna: bool) -> Tuple[np.ndarray, list]:
    ...
