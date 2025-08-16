from __future__ import annotations
import functools
from typing import TYPE_CHECKING, Any, Callable, Scalar
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import jit_user_function
if TYPE_CHECKING:
    import numba
    from collections.abc import Callable
    from pandas._typing import Scalar

@functools.cache
def generate_numba_apply_func(func: Callable, nopython: bool, nogil: bool, parallel: bool) -> Callable:
    ...

@functools.cache
def generate_numba_ewm_func(nopython: bool, nogil: bool, parallel: bool, com: float, adjust: bool, ignore_na: bool, deltas: tuple, normalize: bool) -> Callable:
    ...

@functools.cache
def generate_numba_table_func(func: Callable, nopython: bool, nogil: bool, parallel: bool) -> Callable:
    ...

@functools.cache
def generate_manual_numpy_nan_agg_with_axis(nan_func: Callable) -> Callable:
    ...

@functools.cache
def generate_numba_ewm_table_func(nopython: bool, nogil: bool, parallel: bool, com: float, adjust: bool, ignore_na: bool, deltas: tuple, normalize: bool) -> Callable:
    ...
