"""
Routines for filling missing data.
"""
from __future__ import annotations
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, cast, overload, Optional, Union, List, Tuple, Dict, Callable, Set
import numpy as np
from pandas._libs import NaT, algos, lib
from pandas._typing import ArrayLike, AxisInt, F, ReindexMethod, npt
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_numeric_dtype, is_numeric_v_string_like, is_object_dtype, needs_i8_conversion
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna, na_value_for_dtype
if TYPE_CHECKING:
    from pandas import Index

def check_value_size(value: ArrayLike, mask: np.ndarray, length: int) -> ArrayLike:
    """
    Validate the size of the values passed to ExtensionArray.fillna.
    """
    if is_array_like(value):
        if len(value) != length:
            raise ValueError(f"Length of 'value' does not match. Got ({len(value)})  expected {length}")
        value = value[mask]
    return value

def mask_missing(arr: ArrayLike, values_to_mask: Union[List[Any], Tuple[Any, ...], Any]) -> np.ndarray:
    """
    Return a masking array of same size/shape as arr
    with entries equaling any member of values_to_mask set to True

    Parameters
    ----------
    arr : ArrayLike
    values_to_mask: list, tuple, or scalar

    Returns
    -------
    np.ndarray[bool]
    """
    dtype, values_to_mask = infer_dtype_from(values_to_mask)
    if isinstance(dtype, np.dtype):
        values_to_mask = np.array(values_to_mask, dtype=dtype)
    else:
        cls = dtype.construct_array_type()
        if not lib.is_list_like(values_to_mask):
            values_to_mask = [values_to_mask]
        values_to_mask = cls._from_sequence(values_to_mask, dtype=dtype, copy=False)
    potential_na = False
    if is_object_dtype(arr.dtype):
        potential_na = True
        arr_mask = ~isna(arr)
    na_mask = isna(values_to_mask)
    nonna = values_to_mask[~na_mask]
    mask = np.zeros(arr.shape, dtype=bool)
    if is_numeric_dtype(arr.dtype) and (not is_bool_dtype(arr.dtype)) and is_bool_dtype(nonna.dtype):
        pass
    elif is_bool_dtype(arr.dtype) and is_numeric_dtype(nonna.dtype) and (not is_bool_dtype(nonna.dtype)):
        pass
    else:
        for x in nonna:
            if is_numeric_v_string_like(arr, x):
                pass
            else:
                if potential_na:
                    new_mask = np.zeros(arr.shape, dtype=np.bool_)
                    new_mask[arr_mask] = arr[arr_mask] == x
                else:
                    new_mask = arr == x
                    if not isinstance(new_mask, np.ndarray):
                        new_mask = new_mask.to_numpy(dtype=bool, na_value=False)
                mask |= new_mask
    if na_mask.any():
        mask |= isna(arr)
    return mask

@overload
def clean_fill_method(method: str, *, allow_nearest: bool = ...) -> str: ...

@overload
def clean_fill_method(method: str, *, allow_nearest: bool) -> str: ...

def clean_fill_method(method: str, *, allow_nearest: bool = False) -> str:
    if isinstance(method, str):
        method = method.lower()
        if method == 'ffill':
            method = 'pad'
        elif method == 'bfill':
            method = 'backfill'
    valid_methods = ['pad', 'backfill']
    expecting = 'pad (ffill) or backfill (bfill)'
    if allow_nearest:
        valid_methods.append('nearest')
        expecting = 'pad (ffill), backfill (bfill) or nearest'
    if method not in valid_methods:
        raise ValueError(f'Invalid fill method. Expecting {expecting}. Got {method}')
    return method

NP_METHODS: List[str] = ['linear', 'time', 'index', 'values']
SP_METHODS: List[str] = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline']

def clean_interp_method(method: str, index: Index, **kwargs: Any) -> str:
    order = kwargs.get('order')
    if method in ('spline', 'polynomial') and order is None:
        raise ValueError('You must specify the order of the spline or polynomial.')
    valid = NP_METHODS + SP_METHODS
    if method not in valid:
        raise ValueError(f"method must be one of {valid}. Got '{method}' instead.")
    if method in ('krogh', 'piecewise_polynomial', 'pchip'):
        if not index.is_monotonic_increasing:
            raise ValueError(f'{method} interpolation requires that the index be monotonic.')
    return method

def find_valid_index(how: Literal['first', 'last'], is_valid: np.ndarray) -> Optional[int]:
    """
    Retrieves the positional index of the first valid value.

    Parameters
    ----------
    how : {'first', 'last'}
        Use this parameter to change between the first or last valid index.
    is_valid: np.ndarray
        Mask to find na_values.

    Returns
    -------
    int or None
    """
    assert how in ['first', 'last']
    if len(is_valid) == 0:
        return None
    if is_valid.ndim == 2:
        is_valid = is_valid.any(axis=1)
    if how == 'first':
        idxpos = is_valid[:].argmax()
    elif how == 'last':
        idxpos = len(is_valid) - 1 - is_valid[::-1].argmax()
    chk_notna = is_valid[idxpos]
    if not chk_notna:
        return None
    return idxpos

def validate_limit_direction(limit_direction: str) -> str:
    valid_limit_directions = ['forward', 'backward', 'both']
    limit_direction = limit_direction.lower()
    if limit_direction not in valid_limit_directions:
        raise ValueError(f"Invalid limit_direction: expecting one of {valid_limit_directions}, got '{limit_direction}'.")
    return limit_direction

def validate_limit_area(limit_area: Optional[str]) -> Optional[str]:
    if limit_area is not None:
        valid_limit_areas = ['inside', 'outside']
        limit_area = limit_area.lower()
        if limit_area not in valid_limit_areas:
            raise ValueError(f'Invalid limit_area: expecting one of {valid_limit_areas}, got {limit_area}.')
    return limit_area

def infer_limit_direction(limit_direction: Optional[str], method: str) -> str:
    if limit_direction is None:
        if method in ('backfill', 'bfill'):
            limit_direction = 'backward'
        else:
            limit_direction = 'forward'
    else:
        if method in ('pad', 'ffill') and limit_direction != 'forward':
            raise ValueError(f"`limit_direction` must be 'forward' for method `{method}`")
        if method in ('backfill', 'bfill') and limit_direction != 'backward':
            raise ValueError(f"`limit_direction` must be 'backward' for method `{method}`")
    return limit_direction

def get_interp_index(method: str, index: Index) -> Index:
    if method == 'linear':
        from pandas import Index
        if isinstance(index.dtype, DatetimeTZDtype) or lib.is_np_dtype(index.dtype, 'mM'):
            index = Index(index.view('i8'))
        elif not is_numeric_dtype(index.dtype):
            index = Index(range(len(index)))
    else:
        methods = {'index', 'values', 'nearest', 'time'}
        is_numeric_or_datetime = is_numeric_dtype(index.dtype) or isinstance(index.dtype, DatetimeTZDtype) or lib.is_np_dtype(index.dtype, 'mM')
        valid = NP_METHODS + SP_METHODS
        if method in valid:
            if method not in methods and (not is_numeric_or_datetime):
                raise ValueError(f'Index column must be numeric or datetime type when using {method} method other than linear. Try setting a numeric or datetime index column before interpolating.')
        else:
            raise ValueError(f'Can not interpolate with method={method}.')
    if isna(index).any():
        raise NotImplementedError('Interpolation with NaNs in the index has not been implemented. Try filling those NaNs before interpolating.')
    return index

def interpolate_2d_inplace(
    data: np.ndarray,
    index: Index,
    axis: AxisInt,
    method: str = 'linear',
    limit: Optional[int] = None,
    limit_direction: str = 'forward',
    limit_area: Optional[str] = None,
    fill_value: Optional[Any] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs: Any
) -> None:
    """
    Column-wise application of _interpolate_1d.

    Notes
    -----
    Alters 'data' in-place.

    The signature does differ from _interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """
    clean_interp_method(method, index, **kwargs)
    if is_valid_na_for_dtype(fill_value, data.dtype):
        fill_value = na_value_for_dtype(data.dtype, compat=False)
    if method == 'time':
        if not needs_i8_conversion(index.dtype):
            raise ValueError('time-weighted interpolation only works on Series or DataFrames with a DatetimeIndex')
        method = 'values'
    limit_direction = validate_limit_direction(limit_direction)
    limit_area_validated = validate_limit_area(limit_area)
    limit = algos.validate_limit(nobs=None, limit=limit)
    indices = _index_to_interp_indices(index, method)

    def func(yvalues: np.ndarray) -> None:
        _interpolate_1d(indices=indices, yvalues=yvalues, method=method, limit=limit, limit_direction=limit_direction, limit_area=limit_area_validated, fill_value=fill_value, bounds_error=False, mask=mask, **kwargs)
    np.apply_along_axis(func, axis, data)

def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
    xarr = index._values
    if needs_i8_conversion(xarr.dtype):
        xarr = xarr.view('i8')
    if method == 'linear':
        inds = xarr
        inds = cast(np.ndarray, inds)
    else:
        inds = np.asarray(xarr)
        if method in ('values', 'index'):
            if inds.dtype == np.object_:
                inds = lib.maybe_convert_objects(inds)
    return inds

def _interpolate_1d(
    indices: np.ndarray,
    yvalues: np.ndarray,
    method: str = 'linear',
    limit: Optional[int] = None,
    limit_direction: str = 'forward',
    limit_area: Optional[str] = None,
    fill_value: Optional[Any] = None,
    bounds_error: bool = False,
    order: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs: Any
) -> None:
    """
    Logic for the 1-d interpolation.  The input
    indices and yvalues will each be 1-d arrays of the same length.

    Bounds_error is currently hardcoded to False since non-scipy ones don't
    take it as an argument.

    Notes
    -----
    Fills 'yvalues' in-place.
    """
    if mask is not None:
        invalid = mask
    else:
        invalid = isna(yvalues)
    valid = ~invalid
    if not valid.any():
        return
    if valid.all():
        return
    all_nans = np.flatnonzero(invalid)
    first_valid_index = find_valid_index(how='first', is_valid=valid)
    if first_valid_index is None:
        first_valid_index = 0
    start_nans = np.arange(first_valid_index)
    last_valid_index = find_valid_index(how='last', is_valid=valid)
    if last_valid_index is None:
        last_valid_index = len(yvalues)
    end_nans = np.arange(1 + last_valid_index, len(valid))
    if limit_direction == 'forward':
        preserve_nans = np.union1d(start_nans, _interp_limit(invalid, limit, 0))
    elif limit_direction == 'backward':
        preserve_nans = np.union1d(end_nans, _interp_limit(invalid, 0, limit))
    else:
        preserve_nans = np.unique(_interp_limit(invalid, limit, limit))
    if limit_area == 'inside':
        preserve_nans = np.union1d(preserve_nans, start_nans)
        preserve_nans = np.union1d(preserve_nans, end_nans)
    elif limit_area == 'outside':
        mid_nans = np.setdiff1d(all_nans, start_nans, assume_unique=True)
        mid_nans = np.setdiff1d(mid_nans, end_nans, assume_unique=True)
        preserve_nans = np.union1d(preserve_nans, mid_nans)
    is_datetimelike = yvalues.dtype.kind in 'mM'
    if is_datetimelike:
        yvalues = yvalues.view('i8')
    if method in NP_METHODS:
        indexer = np.argsort(indices[valid])
        yvalues[invalid] = np.interp(indices[invalid], indices[valid][indexer], yvalues[valid][indexer])
    else:
        yvalues[invalid] = _interpolate_scipy_wrapper(indices[valid], yvalues[valid], indices[invalid], method=method, fill_value=fill_value, bounds_error=bounds_error, order=order, **kwargs)
    if mask is not None:
        mask[:] = False
        mask[preserve_nans] = True
    elif is_datetimelike:
        yvalues[preserve_nans] = NaT.value
    else:
        yvalues[preserve_nans] = np.nan
    return

def _interpolate_scipy_wrapper(
    x: np.ndarray,
    y: np.ndarray,
    new_x: np.ndarray,
    method: str,
    fill_value: Optional[Any] = None,
    bounds_error: bool = False,
    order: Optional[int] = None,
    **kwargs: Any
) -> np.ndarray:
    """
    Passed off to scipy.interpolate.interp1d. method is scipy's kind.
    Returns an array interpolated at new_x.  Add any new methods to
    the list in _clean_interp_method.
    """
    extra = f'{method} interpolation requires SciPy.'
    import_optional_dependency('scipy', extra=extra)
    from scipy import interpolate
    new_x = np.asarray(new_x)
    alt_methods = {'barycentric': interpolate.barycentric_interpolate, 'krogh': interpolate.krogh_interpolate, 'from_derivatives': _from_derivatives, 'piecewise_polynomial': _from_derivatives, 'cubicspline': _cubicspline_interpolate, 'akima': _akima_interpolate, 'pchip': interpolate.pchip_interpolate}
    interp1d_methods = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial']
    if method in interp1d_methods:
        if method == 'polynomial':
            kind = order
        else:
            kind = method
        terp = interpolate.interp1d(x, y, kind=kind, fill_value=fill_value, bounds_error=bounds_error)
        new_y = terp(new_x)
    elif method == 'spline':
        if isna(order) or order <= 0:
            raise ValueError(f'order needs to be specified and greater than 0; got order: {order}')
        terp = interpolate.UnivariateSpline(x, y, k=order, **kwargs)
        new_y = terp(new_x)
    else:
        if not x.flags.writeable:
            x = x.copy()
        if not y.flags.writeable:
            y = y.copy()
        if not new_x.flags.writeable:
            new_x = new_x.copy()
        terp = alt_methods.get(method, None)
        if terp is None:
            raise ValueError(f'Can not interpolate with method={method}.')
        kwargs.pop('downcast', None)
        new_y = terp(x, y, new_x, **kwargs)
    return new_y

def _from_derivatives(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    order: Optional[int] = None,
    der: int = 0,
    extrapolate: bool = False
) -> np.ndarray:
    """
    Convenience function for interpolate.BPoly.from_derivatives.

    Construct a piecewise polynomial in the Bernstein basis, compatible
    with the specified values and derivatives at breakpoints.

    Parameters
    ----------
    xi : array-like
        sorted 1D array of x-coordinates
    yi : array-like or list of array-likes
        yi[i][j] is the j-th derivative known at xi[i]
    order: None or int or array-like of ints. Default: