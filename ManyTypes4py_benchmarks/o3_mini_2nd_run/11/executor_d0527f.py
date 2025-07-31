from __future__ import annotations
import functools
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import jit_user_function
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable as AbcCallable  # type: ignore
    from pandas._typing import Scalar

@functools.cache
def generate_apply_looper(
    func: Callable[..., Any],
    nopython: bool = True,
    nogil: bool = True,
    parallel: bool = False,
) -> Callable[..., np.ndarray]:
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")
    nb_compat_func = jit_user_function(func)

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def nb_looper(values: np.ndarray, axis: int, *args: Any) -> np.ndarray:
        if axis == 0:
            first_elem = values[:, 0]
            dim0 = values.shape[1]
        else:
            first_elem = values[0]
            dim0 = values.shape[0]
        res0 = nb_compat_func(first_elem, *args)
        buf_shape = (dim0,) + np.atleast_1d(np.asarray(res0)).shape
        if axis == 0:
            buf_shape = buf_shape[::-1]
        buff = np.empty(buf_shape)
        if axis == 1:
            buff[0] = res0
            for i in numba.prange(1, values.shape[0]):
                buff[i] = nb_compat_func(values[i], *args)
        else:
            buff[:, 0] = res0
            for j in numba.prange(1, values.shape[1]):
                buff[:, j] = nb_compat_func(values[:, j], *args)
        return buff

    return nb_looper

@functools.cache
def make_looper(
    func: Callable[..., Tuple[np.ndarray, Dict[Any, np.ndarray]]],
    result_dtype: np.dtype,
    is_grouped_kernel: bool,
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[..., Tuple[np.ndarray, Dict[Any, np.ndarray]]]:
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")
    if is_grouped_kernel:

        @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
        def column_looper(
            values: np.ndarray,
            labels: np.ndarray,
            ngroups: int,
            min_periods: int,
            *args: Any,
        ) -> Tuple[np.ndarray, Dict[Any, np.ndarray]]:
            result = np.empty((values.shape[0], ngroups), dtype=result_dtype)
            na_positions: Dict[Any, np.ndarray] = {}
            for i in numba.prange(values.shape[0]):
                output, na_pos = func(values[i], result_dtype, labels, ngroups, min_periods, *args)
                result[i] = output
                if len(na_pos) > 0:
                    na_positions[i] = np.array(na_pos)
            return (result, na_positions)
    else:

        @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
        def column_looper(
            values: np.ndarray,
            start: np.ndarray,
            end: np.ndarray,
            min_periods: int,
            *args: Any,
        ) -> Tuple[np.ndarray, Dict[Any, np.ndarray]]:
            result = np.empty((values.shape[0], len(start)), dtype=result_dtype)
            na_positions: Dict[Any, np.ndarray] = {}
            for i in numba.prange(values.shape[0]):
                output, na_pos = func(values[i], result_dtype, start, end, min_periods, *args)
                result[i] = output
                if len(na_pos) > 0:
                    na_positions[i] = np.array(na_pos)
            return (result, na_positions)
    return column_looper

default_dtype_mapping: Dict[np.dtype, Any] = {
    np.dtype("int8"): np.int64,
    np.dtype("int16"): np.int64,
    np.dtype("int32"): np.int64,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint64,
    np.dtype("uint16"): np.uint64,
    np.dtype("uint32"): np.uint64,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex128,
    np.dtype("complex128"): np.complex128,
}
float_dtype_mapping: Dict[np.dtype, Any] = {
    np.dtype("int8"): np.float64,
    np.dtype("int16"): np.float64,
    np.dtype("int32"): np.float64,
    np.dtype("int64"): np.float64,
    np.dtype("uint8"): np.float64,
    np.dtype("uint16"): np.float64,
    np.dtype("uint32"): np.float64,
    np.dtype("uint64"): np.float64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.float64,
    np.dtype("complex128"): np.float64,
}
identity_dtype_mapping: Dict[np.dtype, Any] = {
    np.dtype("int8"): np.int8,
    np.dtype("int16"): np.int16,
    np.dtype("int32"): np.int32,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint8,
    np.dtype("uint16"): np.uint16,
    np.dtype("uint32"): np.uint32,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float32,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex64,
    np.dtype("complex128"): np.complex128,
}

def generate_shared_aggregator(
    func: Callable[..., Tuple[np.ndarray, Dict[Any, np.ndarray]]],
    dtype_mapping: Dict[np.dtype, np.dtype],
    is_grouped_kernel: bool,
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[..., np.ndarray]:
    """
    Generate a Numba function that loops over the columns 2D object and applies
    a 1D numba kernel over each column.

    Parameters
    ----------
    func : function
        aggregation function to be applied to each column
    dtype_mapping: dict or None
        If not None, maps a dtype to a result dtype.
        Otherwise, will fall back to default mapping.
    is_grouped_kernel: bool, default False
        Whether func operates using the group labels (True)
        or using starts/ends arrays

        If true, you also need to pass the number of groups to this function
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
    def looper_wrapper(
        values: np.ndarray,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        ngroups: Optional[int] = None,
        min_periods: int = 0,
        **kwargs: Any,
    ) -> np.ndarray:
        result_dtype: np.dtype = dtype_mapping[values.dtype]
        column_looper: Callable[..., Tuple[np.ndarray, Dict[Any, np.ndarray]]] = make_looper(
            func, result_dtype, is_grouped_kernel, nopython, nogil, parallel
        )
        if is_grouped_kernel:
            result, na_positions = column_looper(values, labels, ngroups, min_periods, *kwargs.values())
        else:
            result, na_positions = column_looper(values, start, end, min_periods, *kwargs.values())
        if result.dtype.kind == "i":
            for na_pos in na_positions.values():
                if len(na_pos) > 0:
                    result = result.astype("float64")
                    break
        for i, na_pos in na_positions.items():
            if len(na_pos) > 0:
                result[i, na_pos] = np.nan
        return result

    return looper_wrapper