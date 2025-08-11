"""
masked_reductions.py is for reduction algorithms using a mask-based approach
for missing values.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import missing as libmissing
from pandas.core.nanops import check_below_min_count
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import AxisInt, npt

def _reductions(func: Union[int, None, numpy.ndarray], values: numpy.ndarray, mask: Union[int, numpy.ndarray], *, skipna: bool=True, min_count: int=0, axis: Union[None, int, numpy.ndarray]=None, **kwargs):
    """
    Sum, mean or product for 1D masked array.

    Parameters
    ----------
    func : np.sum or np.prod
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    min_count : int, default 0
        The required number of valid values to perform the operation. If fewer than
        ``min_count`` non-NA values are present the result will be NA.
    axis : int, optional, default None
    """
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

def sum(values: Union[int, numpy.ndarray], mask: Union[int, numpy.ndarray], *, skipna: bool=True, min_count: int=0, axis: Union[None, int, numpy.ndarray]=None):
    return _reductions(np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def prod(values: Union[int, numpy.ndarray], mask: Union[int, numpy.ndarray], *, skipna: bool=True, min_count: int=0, axis: Union[None, int, numpy.ndarray]=None):
    return _reductions(np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def _minmax(func: Union[int, float], values: numpy.ndarray, mask: int, *, skipna: bool=True, axis: Union[None, int, float]=None) -> Union[str, slice, simulation.core.common.Terrain]:
    """
    Reduction for 1D masked array.

    Parameters
    ----------
    func : np.min or np.max
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    axis : int, optional, default None
    """
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

def min(values: Union[int, float], mask: Union[int, float], *, skipna: bool=True, axis: Union[None, int, float]=None):
    return _minmax(np.min, values=values, mask=mask, skipna=skipna, axis=axis)

def max(values: Union[int, numpy.ndarray], mask: Union[int, numpy.ndarray], *, skipna: bool=True, axis: Union[None, int, numpy.ndarray]=None):
    return _minmax(np.max, values=values, mask=mask, skipna=skipna, axis=axis)

def mean(values: Union[int, numpy.ndarray], mask: Union[int, numpy.ndarray], *, skipna: bool=True, axis: Union[None, bool, numpy.ndarray]=None):
    if not values.size or mask.all():
        return libmissing.NA
    return _reductions(np.mean, values=values, mask=mask, skipna=skipna, axis=axis)

def var(values: Union[tuple[int], int, str], mask: Union[tuple[int], int, str], *, skipna: bool=True, axis: Union[None, bool, str]=None, ddof: int=1):
    if not values.size or mask.all():
        return libmissing.NA
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _reductions(np.var, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof)

def std(values: Union[int, numpy.ndarray], mask: Union[int, numpy.ndarray], *, skipna: bool=True, axis: Union[None, bool, astropy.units.quantity.Quantity, numpy.ndarray]=None, ddof: int=1):
    if not values.size or mask.all():
        return libmissing.NA
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _reductions(np.std, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof)