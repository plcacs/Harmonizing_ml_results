from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Union
import warnings
import numpy as np
from numpy.typing import NDArray
from pandas._libs import missing as libmissing
from pandas.core.nanops import check_below_min_count
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import AxisInt, npt

def _reductions(func, values, mask, *, skipna: bool=True, min_count: int=0, axis: Optional[AxisInt]=None, **kwargs):
    if not skipna:
        if mask.any() or check_below_min_count(values.shape, None, min_count):
            return libmissing.NA
        else:
            return func(values, axis=axis, **kwargs)
    else:
        if check_below_min_count(values.shape, mask, min_count) and (axis is None or values.ndim == 1):
            return libmissing.NA
        if values.dtype == np.dtype(object):
            values = values[~mask]
            return func(values, axis=axis, **kwargs)
        return func(values, where=~mask, axis=axis, **kwargs)

def sum(values, mask, *, skipna: bool=True, min_count: int=0, axis: Optional[AxisInt]=None):
    return _reductions(np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def prod(values, mask, *, skipna: bool=True, min_count: int=0, axis: Optional[AxisInt]=None):
    return _reductions(np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def _minmax(func, values, mask, *, skipna: bool=True, axis: Optional[AxisInt]=None):
    if not skipna:
        if mask.any() or not values.size:
            return libmissing.NA
        else:
            return func(values, axis=axis)
    else:
        subset = values[~mask]
        if subset.size:
            return func(subset, axis=axis)
        else:
            return libmissing.NA

def min(values, mask, *, skipna: bool=True, axis: Optional[AxisInt]=None):
    return _minmax(np.min, values=values, mask=mask, skipna=skipna, axis=axis)

def max(values, mask, *, skipna: bool=True, axis: Optional[AxisInt]=None):
    return _minmax(np.max, values=values, mask=mask, skipna=skipna, axis=axis)

def mean(values, mask, *, skipna: bool=True, axis: Optional[AxisInt]=None):
    if not values.size or mask.all():
        return libmissing.NA
    return _reductions(np.mean, values=values, mask=mask, skipna=skipna, axis=axis)

def var(values, mask, *, skipna: bool=True, axis: Optional[AxisInt]=None, ddof: int=1):
    if not values.size or mask.all():
        return libmissing.NA
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _reductions(np.var, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof)

def std(values, mask, *, skipna: bool=True, axis: Optional[AxisInt]=None, ddof: int=1):
    if not values.size or mask.all():
        return libmissing.NA
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _reductions(np.std, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof)