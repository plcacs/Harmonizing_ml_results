from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import missing as libmissing
from pandas.core.nanops import check_below_min_count
from collections.abc import Callable
from pandas._typing import AxisInt, npt

def _reductions(func: Callable[[npt], npt], values: npt, mask: np.ndarray[bool], *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None, **kwargs) -> npt:
    ...

def sum(values: npt, mask: np.ndarray[bool], *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None) -> npt:
    return _reductions(np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def prod(values: npt, mask: np.ndarray[bool], *, skipna: bool = True, min_count: int = 0, axis: AxisInt = None) -> npt:
    return _reductions(np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def _minmax(func: Callable[[npt], npt], values: npt, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    ...

def min(values: npt, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    return _minmax(np.min, values=values, mask=mask, skipna=skipna, axis=axis)

def max(values: npt, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    return _minmax(np.max, values=values, mask=mask, skipna=skipna, axis=axis)

def mean(values: npt, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None) -> npt:
    ...

def var(values: npt, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None, ddof: int = 1) -> npt:
    ...

def std(values: npt, mask: np.ndarray[bool], *, skipna: bool = True, axis: AxisInt = None, ddof: int = 1) -> npt:
    ...
