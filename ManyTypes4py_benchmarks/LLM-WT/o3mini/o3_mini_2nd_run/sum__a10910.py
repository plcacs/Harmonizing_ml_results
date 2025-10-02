#!/usr/bin/env python3
"""
Numba 1D sum kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""
from __future__ import annotations
from typing import Any, Tuple, List
import numba
from numba.extending import register_jitable
import numpy as np
from pandas._typing import npt

@numba.jit(nopython=True, nogil=True, parallel=False)
def add_sum(val: float,
            nobs: int,
            sum_x: float,
            compensation: float,
            num_consecutive_same_value: int,
            prev_value: float) -> Tuple[int, float, float, int, float]:
    if not np.isnan(val):
        nobs += 1
        y = val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val
    return (nobs, sum_x, compensation, num_consecutive_same_value, prev_value)

@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_sum(val: float,
               nobs: int,
               sum_x: float,
               compensation: float) -> Tuple[int, float, float]:
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
    return (nobs, sum_x, compensation)

@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_sum(values: npt.NDArray[Any],
                result_dtype: Any,
                start: npt.NDArray[Any],
                end: npt.NDArray[Any],
                min_periods: int) -> Tuple[npt.NDArray[Any], List[int]]:
    dtype = values.dtype
    na_val: float = np.nan
    if dtype.kind == 'i':
        na_val = 0
    N: int = len(start)
    nobs: int = 0
    sum_x: float = 0  # type: ignore
    compensation_add: float = 0  # type: ignore
    compensation_remove: float = 0  # type: ignore
    na_pos: List[int] = []
    is_monotonic_increasing_bounds: bool = is_monotonic_increasing(start) and is_monotonic_increasing(end)
    output: npt.NDArray[Any] = np.empty(N, dtype=result_dtype)
    for i in range(N):
        s: int = start[i]
        e: int = end[i]
        if i == 0 or not is_monotonic_increasing_bounds:
            prev_value: float = values[s]
            num_consecutive_same_value: int = 0
            for j in range(s, e):
                val: float = values[j]
                nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value = add_sum(
                    val, nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value)
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, sum_x, compensation_remove = remove_sum(val, nobs, sum_x, compensation_remove)
            for j in range(end[i - 1], e):
                val = values[j]
                nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value = add_sum(
                    val, nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value)
        if nobs == 0 == min_periods:
            result: float = 0  # type: ignore
        elif nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = na_val
            if dtype.kind == 'i':
                na_pos.append(i)
        output[i] = result
        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0
            compensation_remove = 0
    return (output, na_pos)

@register_jitable
def grouped_kahan_sum(values: npt.NDArray[Any],
                      result_dtype: Any,
                      labels: npt.NDArray[Any],
                      ngroups: int,
                      skipna: bool) -> Tuple[npt.NDArray[Any],
                                             npt.NDArray[np.int64],
                                             npt.NDArray[Any],
                                             npt.NDArray[np.int64],
                                             npt.NDArray[Any]]:
    N: int = len(labels)
    nobs_arr: npt.NDArray[np.int64] = np.zeros(ngroups, dtype=np.int64)
    comp_arr: npt.NDArray[Any] = np.zeros(ngroups, dtype=values.dtype)
    consecutive_counts: npt.NDArray[np.int64] = np.zeros(ngroups, dtype=np.int64)
    prev_vals: npt.NDArray[Any] = np.zeros(ngroups, dtype=values.dtype)
    output: npt.NDArray[Any] = np.zeros(ngroups, dtype=result_dtype)
    for i in range(N):
        lab: int = labels[i]
        val: float = values[i]
        if lab < 0 or np.isnan(output[lab]):
            continue
        if not skipna and np.isnan(val):
            output[lab] = np.nan  # type: ignore
            nobs_arr[lab] += 1
            comp_arr[lab] = np.nan  # type: ignore
            consecutive_counts[lab] = 1
            prev_vals[lab] = np.nan  # type: ignore
            continue
        sum_x: float = output[lab]
        nobs: int = nobs_arr[lab]
        compensation_add: float = comp_arr[lab]
        num_consecutive_same_value: int = consecutive_counts[lab]
        prev_value: float = prev_vals[lab]
        nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value = add_sum(
            val, nobs, sum_x, compensation_add, num_consecutive_same_value, prev_value)
        output[lab] = sum_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs
    return (output, nobs_arr, comp_arr, consecutive_counts, prev_vals)

@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_sum(values: npt.NDArray[Any],
                result_dtype: Any,
                labels: npt.NDArray[Any],
                ngroups: int,
                min_periods: int,
                skipna: bool) -> Tuple[npt.NDArray[Any], List[int]]:
    na_pos: List[int] = []
    output, nobs_arr, comp_arr, consecutive_counts, prev_vals = grouped_kahan_sum(
        values, result_dtype, labels, ngroups, skipna)
    for lab in range(ngroups):
        nobs: int = nobs_arr[lab]
        num_consecutive_same_value: int = consecutive_counts[lab]
        prev_value: float = prev_vals[lab]
        sum_x: float = output[lab]
        if nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = sum_x
            na_pos.append(lab)
        output[lab] = result
    return (output, na_pos)

def is_monotonic_increasing(a: npt.NDArray[Any]) -> bool:
    # Placeholder for the actual implementation.
    # Here we assume the array is monotonic increasing.
    for i in range(1, len(a)):
        if a[i] < a[i-1]:
            return False
    return True
