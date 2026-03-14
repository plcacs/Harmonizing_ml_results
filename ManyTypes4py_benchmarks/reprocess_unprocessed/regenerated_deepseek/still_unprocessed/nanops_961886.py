from __future__ import annotations
import functools
import itertools
from typing import TYPE_CHECKING, Any, cast, overload
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import NaT, NaTType, iNaT, lib
from pandas._typing import ArrayLike, AxisInt, CorrelationMethod, Dtype, DtypeObj, F, Scalar, Shape, npt
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import is_complex, is_float, is_float_dtype, is_integer, is_numeric_dtype, is_object_dtype, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna
if TYPE_CHECKING:
    from collections.abc import Callable
    import bottleneck
bn = import_optional_dependency('bottleneck', errors='warn')
_BOTTLENECK_INSTALLED = bn is not None
_USE_BOTTLENECK = False

def set_use_bottleneck(v: bool = True) -> None:
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v
set_use_bottleneck(get_option('compute.use_bottleneck'))

class disallow:

    def __init__(self, *dtypes: Dtype | str) -> None:
        super().__init__()
        self.dtypes = tuple((pandas_dtype(dtype).type for dtype in dtypes))

    def check(self, obj: Any) -> bool:
        return hasattr(obj, 'dtype') and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:

        @functools.wraps(f)
        def _f(*args: Any, **kwargs: Any) -> Any:
            obj_iter = itertools.chain(args, kwargs.values())
            if any((self.check(obj) for obj in obj_iter)):
                f_name = f.__name__.replace('nan', '')
                raise TypeError(f"reduction operation '{f_name}' not allowed for this dtype")
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise
        return cast(F, _f)

class bottleneck_switch:

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt: F) -> F:
        bn_name = self.name or alt.__name__
        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):
            bn_func = None

        @functools.wraps(alt)
        def f(values: np.ndarray, *, axis: AxisInt | None = None, skipna: bool = True, **kwds: Any) -> Any:
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v
            if values.size == 0 and kwds.get('min_count') is None:
                return _na_for_min_count(values, axis)
            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get('mask', None) is None:
                    kwds.pop('mask', None)
                    result = bn_func(values, axis=axis, **kwds)
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)
            return result
        return cast(F, f)

def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    if dtype != object and (not needs_i8_conversion(dtype)):
        return name not in ['nansum', 'nanprod', 'nanmean']
    return False

def _has_infs(result: Any) -> bool:
    if isinstance(result, np.ndarray):
        if result.dtype in ('f8', 'f4'):
            return lib.has_infs(result.ravel('K'))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        return False

def _get_fill_value(dtype: DtypeObj, fill_value: Any = None, fill_value_typ: str | None = None) -> Any:
    """return the correct fill value for the dtype of the values"""
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        elif fill_value_typ == '+inf':
            return np.inf
        else:
            return -np.inf
    elif fill_value_typ == '+inf':
        return lib.i8max
    else:
        return iNaT

def _maybe_get_mask(values: np.ndarray, skipna: bool, mask: np.ndarray | None) -> np.ndarray | None:
    """
    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray[bool]]
    """
    if mask is None:
        if values.dtype.kind in 'biu':
            return None
        if skipna or values.dtype.kind in 'mM':
            mask = isna(values)
    return mask

def _get_values(values: np.ndarray, skipna: bool, fill_value: Any = None, fill_value_typ: str | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Utility to get the values view, mask, dtype, dtype_max, and fill_value.

    If both mask and fill_value/fill_value_typ are not None and skipna is True,
    the values array will be copied.

    For input arrays of boolean or integer dtypes, copies will only occur if a
    precomputed mask, a fill_value/fill_value_typ, and skipna=True are
    provided.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    fill_value : Any
        value to fill NaNs with
    fill_value_typ : str
        Set to '+inf' or '-inf' to handle dtype-specific infinities
    mask : Optional[np.ndarray[bool]]
        nan-mask if known

    Returns
    -------
    values : ndarray
        Potential copy of input value array
    mask : Optional[ndarray[bool]]
        Mask for values, if deemed necessary to compute
    """
    mask = _maybe_get_mask(values, skipna, mask)
    dtype = values.dtype
    datetimelike = False
    if values.dtype.kind in 'mM':
        values = np.asarray(values.view('i8'))
        datetimelike = True
    if skipna and mask is not None:
        fill_value = _get_fill_value(dtype, fill_value=fill_value, fill_value_typ=fill_value_typ)
        if fill_value is not None:
            if mask.any():
                if datetimelike or _na_ok_dtype(dtype):
                    values = values.copy()
                    np.putmask(values, mask, fill_value)
                else:
                    values = np.where(~mask, values, fill_value)
    return (values, mask)

def _get_dtype_max(dtype: DtypeObj) -> DtypeObj:
    dtype_max = dtype
    if dtype.kind in 'bi':
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == 'u':
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == 'f':
        dtype_max = np.dtype(np.float64)
    return dtype_max

def _na_ok_dtype(dtype: DtypeObj) -> bool:
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)

def _wrap_results(result: Any, dtype: DtypeObj, fill_value: Any = None) -> Any:
    """wrap our results if needed"""
    if result is NaT:
        pass
    elif dtype.kind == 'M':
        if fill_value is None:
            fill_value = iNaT
        if not isinstance(result, np.ndarray):
            assert not isna(fill_value), 'Expected non-null fill_value'
            if result == fill_value:
                result = np.nan
            if isna(result):
                result = np.datetime64('NaT', 'ns').astype(dtype)
            else:
                result = np.int64(result).view(dtype)
            result = result.astype(dtype, copy=False)
        else:
            result = result.astype(dtype)
    elif dtype.kind == 'm':
        if not isinstance(result, np.ndarray):
            if result == fill_value or np.isnan(result):
                result = np.timedelta64('NaT').astype(dtype)
            elif np.fabs(result) > lib.i8max:
                raise ValueError('overflow in timedelta operation')
            else:
                result = np.int64(result).astype(dtype, copy=False)
        else:
            result = result.astype('m8[ns]').view(dtype)
    return result

def _datetimelike_compat(func: F) -> F:
    """
    If we have datetime64 or timedelta64 values, ensure we have a correct
    mask before calling the wrapped function, then cast back afterwards.
    """

    @functools.wraps(func)
    def new_func(values: np.ndarray, *, axis: AxisInt | None = None, skipna: bool = True, mask: np.ndarray | None = None, **kwargs: Any) -> Any:
        orig_values = values
        datetimelike = values.dtype.kind in 'mM'
        if datetimelike and mask is None:
            mask = isna(values)
        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
            if not skipna:
                assert mask is not None
                result = _mask_datetimelike_result(result, axis, mask, orig_values)
        return result
    return cast(F, new_func)

def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Any:
    """
    Return the missing value for `values`.

    Parameters
    ----------
    values : ndarray
    axis : int or None
        axis for the reduction, required if values.ndim > 1.

    Returns
    -------
    result : scalar or ndarray
        For 1-D values, returns a scalar of the correct missing type.
        For 2-D values, returns a 1-D array where each element is missing.
    """
    if values.dtype.kind in 'iufcb':
        values = values.astype('float64')
    fill_value = na_value_for_dtype(values.dtype)
    if values.ndim == 1:
        return fill_value
    elif axis is None:
        return fill_value
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1:]
        return np.full(result_shape, fill_value, dtype=values.dtype)

def maybe_operate_rowwise(func: F) -> F:
    """
    NumPy operations on C-contiguous ndarrays with axis=1 can be
    very slow if axis 1 >> axis 0.
    Operate row-by-row and concatenate the results.
    """

    @functools.wraps(func)
    def newfunc(values: np.ndarray, *, axis: AxisInt | None = None, **kwargs: Any) -> Any:
        if axis == 1 and values.ndim == 2 and values.flags['C_CONTIGUOUS'] and (values.shape[1] / 1000 > values.shape[0]) and (values.dtype != object) and (values.dtype != bool):
            arrs = list(values)
            if kwargs.get('mask') is not None:
                mask = kwargs.pop('mask')
                results = [func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))]
            else:
                results = [func(x, **kwargs) for x in arrs]
            return np.array(results)
        return func(values, axis=axis, **kwargs)
    return cast(F, newfunc)

def nanany(values: np.ndarray, *, axis: AxisInt | None = None, skipna: bool = True, mask: np.ndarray | None = None) -> bool | np.ndarray:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    False
    """
    if values.dtype.kind in 'iub' and mask is None:
        return values.any(axis)
    if values.dtype.kind == 'M':
        raise TypeError("datetime64 type does not support operation 'any'")
    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)
    if values.dtype == object:
        values = values.astype(bool)
    return values.any(axis)

def nanall(values: np.ndarray, *, axis: AxisInt | None = None, skipna: bool = True, mask: np.ndarray | None = None) -> bool | np.ndarray:
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s.values)
    False
    """
    if values.dtype.kind in 'iub' and mask is None:
        return values.all(axis)
    if values.dtype.kind == 'M':
        raise TypeError("datetime64 type does not support operation 'all'")
    values, _ = _get_values(values, skipna, fill_value=True, mask=mask)
    if values.dtype == object:
        values = values.astype(bool)
    return values.all(axis)

@disallow('M8')
@_datetimelike_compat
@maybe_operate_rowwise
def nansum(values: np.ndarray, *, axis: AxisInt | None = None, skipna: bool = True, min_count: int = 0, mask: np.ndarray | None = None) -> Any:
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : dtype

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s.values)
    3.0
    """
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    if dtype.kind == 'f':
        dtype_sum = dtype
    elif dtype.kind == 'm':
        dtype_sum = np.dtype(np.float64)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)
    return the_sum

def _mask_datetimelike_result(result: Any, axis: AxisInt | None, mask: np.ndarray, orig_values: np.ndarray) -> Any:
    if isinstance(result, np.ndarray):
        result = result.astype('i8').view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        result[axis_mask] = iNaT
    elif mask.any():
        return np.int64(iNaT).view(orig_values.dtype)
    return result

@bottleneck_switch()
@_datetimelike_compat
def nanmean(values: np.ndarray, *, axis: AxisInt | None = None, skipna: bool = True, mask: np