from __future__ import annotations
import functools
from typing import Any, Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import jit_user_function
if False:  # TYPE_CHECKING
    from collections.abc import Callable
    from pandas._typing import Scalar

@functools.cache
def generate_numba_apply_func(
    func: Callable[..., Any],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any], int, Any], NDArray[Any]]:
    """
    Generate a numba jitted apply function specified by values from engine_kwargs.
    """
    numba_func = jit_user_function(func)
    if False:  # TYPE_CHECKING
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_apply(
        values: NDArray[Any],
        begin: NDArray[Any],
        end: NDArray[Any],
        minimum_periods: int,
        *args: Any
    ) -> NDArray[Any]:
        result = np.empty(len(begin))
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window))
            if len(window) - count_nan >= minimum_periods:
                result[i] = numba_func(window, *args)
            else:
                result[i] = np.nan
        return result
    return roll_apply

@functools.cache
def generate_numba_ewm_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: Tuple[float, ...],
    normalize: bool
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any], int], NDArray[Any]]:
    """
    Generate a numba jitted ewm mean or sum function specified by values from engine_kwargs.
    """
    if False:  # TYPE_CHECKING
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewm(
        values: NDArray[Any],
        begin: NDArray[Any],
        end: NDArray[Any],
        minimum_periods: int
    ) -> NDArray[Any]:
        result = np.empty(len(values))
        alpha = 1.0 / (1.0 + com)
        old_wt_factor = 1.0 - alpha
        new_wt = 1.0 if adjust else alpha
        for i in numba.prange(len(begin)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            sub_result = np.empty(len(window))
            weighted = window[0]
            nobs = int(not np.isnan(weighted))
            sub_result[0] = weighted if nobs >= minimum_periods else np.nan
            old_wt = 1.0
            for j in range(1, len(window)):
                cur = window[j]
                is_observation = not np.isnan(cur)
                nobs += is_observation
                if not np.isnan(weighted):
                    if is_observation or not ignore_na:
                        if normalize:
                            old_wt *= old_wt_factor ** deltas[start + j - 1]
                            if not adjust and com == 1:
                                new_wt = 1.0 - old_wt
                        else:
                            weighted = old_wt_factor * weighted
                        if is_observation:
                            if normalize:
                                if weighted != cur:
                                    weighted = old_wt * weighted + new_wt * cur
                                    if normalize:
                                        weighted = weighted / (old_wt + new_wt)
                                if adjust:
                                    old_wt += new_wt
                                else:
                                    old_wt = 1.0
                            else:
                                weighted += cur
                elif is_observation:
                    weighted = cur
                sub_result[j] = weighted if nobs >= minimum_periods else np.nan
            result[start:stop] = sub_result
        return result
    return ewm

@functools.cache
def generate_numba_table_func(
    func: Callable[..., Any],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any], int, Any], NDArray[Any]]:
    """
    Generate a numba jitted function to apply window calculations table-wise.
    """
    numba_func = jit_user_function(func)
    if False:  # TYPE_CHECKING
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_table(
        values: NDArray[Any],
        begin: NDArray[Any],
        end: NDArray[Any],
        minimum_periods: int,
        *args: Any
    ) -> NDArray[Any]:
        result = np.empty((len(begin), values.shape[1]))
        min_periods_mask = np.empty(result.shape)
        for i in numba.prange(len(result)):
            start = begin[i]
            stop = end[i]
            window = values[start:stop]
            count_nan = np.sum(np.isnan(window), axis=0)
            nan_mask = len(window) - count_nan >= minimum_periods
            if nan_mask.any():
                result[i, :] = numba_func(window, *args)
            min_periods_mask[i, :] = nan_mask
        result = np.where(min_periods_mask, result, np.nan)
        return result
    return roll_table

@functools.cache
def generate_manual_numpy_nan_agg_with_axis(
    nan_func: Callable[[NDArray[Any]], Any]
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    if False:  # TYPE_CHECKING
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=True, nogil=True, parallel=True)
    def nan_agg_with_axis(table: NDArray[Any]) -> NDArray[Any]:
        result = np.empty(table.shape[1])
        for i in numba.prange(table.shape[1]):
            partition = table[:, i]
            result[i] = nan_func(partition)
        return result
    return nan_agg_with_axis

@functools.cache
def generate_numba_ewm_table_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: Tuple[float, ...],
    normalize: bool
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any], int], NDArray[Any]]:
    """
    Generate a numba jitted ewm mean or sum function applied table wise specified
    by values from engine_kwargs.
    """
    if False:  # TYPE_CHECKING
        import numba
    else:
        numba = import_optional_dependency('numba')

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewm_table(
        values: NDArray[Any],
        begin: NDArray[Any],
        end: NDArray[Any],
        minimum_periods: int
    ) -> NDArray[Any]:
        alpha = 1.0 / (1.0 + com)
        old_wt_factor = 1.0 - alpha
        new_wt = 1.0 if adjust else alpha
        old_wt = np.ones(values.shape[1])
        result = np.empty(values.shape)
        weighted = values[0].copy()
        nobs = (~np.isnan(weighted)).astype(np.int64)
        result[0] = np.where(nobs >= minimum_periods, weighted, np.nan)
        for i in range(1, len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted[j]):
                    if is_observations[j] or not ignore_na:
                        if normalize:
                            old_wt[j] *= old_wt_factor ** deltas[i - 1]
                            if not adjust and com == 1:
                                new_wt = 1.0 - old_wt[j]
                        else:
                            weighted[j] = old_wt_factor * weighted[j]
                        if is_observations[j]:
                            if normalize:
                                if weighted[j] != cur[j]:
                                    weighted[j] = old_wt[j] * weighted[j] + new_wt * cur[j]
                                    if normalize:
                                        weighted[j] = weighted[j] / (old_wt[j] + new_wt)
                                if adjust:
                                    old_wt[j] += new_wt
                                else:
                                    old_wt[j] = 1.0
                            else:
                                weighted[j] += cur[j]
                elif is_observations[j]:
                    weighted[j] = cur[j]
            result[i] = np.where(nobs >= minimum_periods, weighted, np.nan)
        return result
    return ewm_table