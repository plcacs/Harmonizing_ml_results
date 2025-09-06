from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple
import numba
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt

def func_ew08sc5f(val: float, nobs: int, sum_x: float, compensation: float,
                   num_consecutive_same_value: int, prev_value: float) -> Tuple[int, float, float, int, float]:
    ...

def func_dpkd9czy(val: float, nobs: int, sum_x: float, compensation: float) -> Tuple[int, float, float]:
    ...

def func_qe4g7jip(values: np.ndarray, result_dtype: Any, start: np.ndarray, end: np.ndarray, min_periods: int) -> Tuple[np.ndarray, list]:
    ...

def func_su3xsce8(values: np.ndarray, result_dtype: Any, labels: np.ndarray, ngroups: int, skipna: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...

def func_ln93qdd3(values: np.ndarray, result_dtype: Any, labels: np.ndarray, ngroups: int, min_periods: int, skipna: bool) -> Tuple[np.ndarray, list]:
    ...
