from __future__ import annotations
from typing import Any, Callable, Optional
import warnings
import numpy as np
from pandas._libs import missing as libmissing
from pandas.core.nanops import check_below_min_count
from pandas._typing import npt

def _reductions(
    func: Callable[..., Any],
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: Optional[int] = None,
    **kwargs: Any
) -> Any:
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

def sum(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: Optional[int] = None
) -> Any:
    return _reductions(np.sum, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def prod(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: Optional[int] = None
) -> Any:
    return _reductions(np.prod, values=values, mask=mask, skipna=skipna, min_count=min_count, axis=axis)

def _minmax(
    func: Callable[..., Any],
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: Optional[int] = None
) -> Any:
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

def min(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: Optional[int] = None
) -> Any:
    return _minmax(np.min, values=values, mask=mask, skipna=skipna, axis=axis)

def max(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: Optional[int] = None
) -> Any:
    return _minmax(np.max, values=values, mask=mask, skipna=skipna, axis=axis)

def mean(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: Optional[int] = None
) -> Any:
    if not values.size or mask.all():
        return libmissing.NA
    return _reductions(np.mean, values=values, mask=mask, skipna=skipna, axis=axis)

def var(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: Optional[int] = None,
    ddof: int = 1
) -> Any:
    if not values.size or mask.all():
        return libmissing.NA
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _reductions(np.var, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof)

def std(
    values: npt.NDArray[Any],
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
    axis: Optional[int] = None,
    ddof: int = 1
) -> Any:
    if not values.size or mask.all():
        return libmissing.NA
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return _reductions(np.std, values=values, mask=mask, skipna=skipna, axis=axis, ddof=ddof)