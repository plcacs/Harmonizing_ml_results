from __future__ import annotations
import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Scalar
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import NumbaUtilError, jit_user_function
if TYPE_CHECKING:
    import numba

def validate_udf(func: Callable) -> None:
    if not callable(func):
        raise NotImplementedError('Numba engine can only be used with a single function.')
    udf_signature = list(inspect.signature(func).parameters.keys())
    expected_args = ['values', 'index']
    min_number_args = len(expected_args)
    if len(udf_signature) < min_number_args or udf_signature[:min_number_args] != expected_args:
        raise NumbaUtilError(f'The first {min_number_args} arguments to {func.__name__} must be {expected_args}')

@functools.cache
def generate_numba_agg_func(func: Callable, nopython: bool, nogil: bool, parallel: bool) -> Callable:
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_agg(values, index, begin, end, num_columns, *args) -> np.ndarray:
        assert len(begin) == len(end)
        num_groups = len(begin)
        result = np.empty((num_groups, num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i]:end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i]:end[i], j]
                result[i, j] = numba_func(group, group_index, *args)
        return result
    return group_agg

@functools.cache
def generate_numba_transform_func(func: Callable, nopython: bool, nogil: bool, parallel: bool) -> Callable:
    numba_func = jit_user_function(func)
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_transform(values, index, begin, end, num_columns, *args) -> np.ndarray:
        assert len(begin) == len(end)
        num_groups = len(begin)
        result = np.empty((len(values), num_columns))
        for i in numba.prange(num_groups):
            group_index = index[begin[i]:end[i]]
            for j in numba.prange(num_columns):
                group = values[begin[i]:end[i], j]
                result[begin[i]:end[i], j] = numba_func(group, group_index, *args)
        return result
    return group_transform
