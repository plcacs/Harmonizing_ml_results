#!/usr/bin/env python3
"""
Routines for filling missing data.
"""

from __future__ import annotations

from functools import wraps
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    overload,
    Tuple,
    Union,
    cast,
)
import numpy as np

from pandas._libs import (
    NaT,
    algos,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    F,
    ReindexMethod,
    npt,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_numeric_dtype,
    is_numeric_v_string_like,
    is_object_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)

if False:  # TYPE_CHECKING
    from pandas import Index

def check_value_size(value: ArrayLike, mask: npt.NDArray[np.bool_], length: int) -> Any:
    """
    Validate the size of the values passed to ExtensionArray.fillna.
    """
    if is_array_like(value):
        if len(value) != length:
            raise ValueError(
                f"Length of 'value' does not match. Got ({len(value)}) "
                f" expected {length}"
            )
        value = value[mask]

    return value


def mask_missing(arr: ArrayLike, values_to_mask: Any) -> npt.NDArray[np.bool_]:
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
    # When called from Block.replace/replace_list, values_to_mask is a scalar
    #  known to be holdable by arr.
    # When called from Series._single_replace, values_to_mask is tuple or list
    dtype, values_to_mask = infer_dtype_from(values_to_mask)

    if isinstance(dtype, np.dtype):
        values_to_mask = np.array(values_to_mask, dtype=dtype)
    else:
        cls = dtype.construct_array_type()
        if not lib.is_list_like(values_to_mask):
            values_to_mask = [values_to_mask]
        values_to_mask = cls._from_sequence(values_to_mask, dtype=dtype, copy=False)

    potential_na: bool = False
    if is_object_dtype(arr.dtype):
        # pre-compute mask to avoid comparison to NA
        potential_na = True
        arr_mask = ~isna(arr)

    na_mask = isna(values_to_mask)
    nonna = values_to_mask[~na_mask]

    # GH 21977
    mask = np.zeros(arr.shape, dtype=bool)
    if (
        is_numeric_dtype(arr.dtype)
        and not is_bool_dtype(arr.dtype)
        and is_bool_dtype(nonna.dtype)
    ):
        pass
    elif (
        is_bool_dtype(arr.dtype)
        and is_numeric_dtype(nonna.dtype)
        and not is_bool_dtype(nonna.dtype)
    ):
        pass
    else:
        for x in nonna:
            if is_numeric_v_string_like(arr, x):
                # GH#29553 prevent numpy deprecation warnings
                pass
            else:
                if potential_na:
                    new_mask = np.zeros(arr.shape, dtype=bool)
                    new_mask[arr_mask] = arr[arr_mask] == x
                else:
                    new_mask = arr == x

                    if not isinstance(new_mask, np.ndarray):
                        # usually BooleanArray
                        new_mask = new_mask.to_numpy(dtype=bool, na_value=False)
                mask |= new_mask

    if na_mask.any():
        mask |= isna(arr)

    return mask


@overload
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill"],
    *,
    allow_nearest: Literal[False] = ...,
) -> Literal["pad", "backfill"]: ...
    

@overload
def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: Literal[True],
) -> Literal["pad", "backfill", "nearest"]: ...


def clean_fill_method(
    method: Literal["ffill", "pad", "bfill", "backfill", "nearest"],
    *,
    allow_nearest: bool = False,
) -> Literal["pad", "backfill", "nearest"]:
    if isinstance(method, str):
        method = method.lower()  # type: ignore[assignment]
        if method == "ffill":
            method = "pad"
        elif method == "bfill":
            method = "backfill"

    valid_methods: List[str] = ["pad", "backfill"]
    expecting = "pad (ffill) or backfill (bfill)"
    if allow_nearest:
        valid_methods.append("nearest")
        expecting = "pad (ffill), backfill (bfill) or nearest"
    if method not in valid_methods:
        raise ValueError(f"Invalid fill method. Expecting {expecting}. Got {method}")
    return method


NP_METHODS: List[str] = ["linear", "time", "index", "values"]

SP_METHODS: List[str] = [
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "krogh",
    "spline",
    "polynomial",
    "from_derivatives",
    "piecewise_polynomial",
    "pchip",
    "akima",
    "cubicspline",
]


def clean_interp_method(method: str, index: Any, **kwargs: Any) -> str:
    order = kwargs.get("order")

    if method in ("spline", "polynomial") and order is None:
        raise ValueError("You must specify the order of the spline or polynomial.")

    valid = NP_METHODS + SP_METHODS
    if method not in valid:
        raise ValueError(f"method must be one of {valid}. Got '{method}' instead.")

    if method in ("krogh", "piecewise_polynomial", "pchip"):
        if not index.is_monotonic_increasing:
            raise ValueError(
                f"{method} interpolation requires that the index be monotonic."
            )

    return method


def find_valid_index(how: str, is_valid: npt.NDArray[np.bool_]) -> Optional[int]:
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
    assert how in ["first", "last"]

    if len(is_valid) == 0:  # early stop
        return None

    if is_valid.ndim == 2:
        is_valid = is_valid.any(axis=1)  # reduce axis 1

    if how == "first":
        idxpos = is_valid[::].argmax()
    elif how == "last":
        idxpos = len(is_valid) - 1 - is_valid[::-1].argmax()

    chk_notna = is_valid[idxpos]

    if not chk_notna:
        return None
    return idxpos  # type: ignore[return-value]


def validate_limit_direction(
    limit_direction: str,
) -> Literal["forward", "backward", "both"]:
    valid_limit_directions: List[str] = ["forward", "backward", "both"]
    limit_direction = limit_direction.lower()
    if limit_direction not in valid_limit_directions:
        raise ValueError(
            "Invalid limit_direction: expecting one of "
            f"{valid_limit_directions}, got '{limit_direction}'."
        )
    return limit_direction  # type: ignore[return-value]


def validate_limit_area(limit_area: Optional[str]) -> Optional[Literal["inside", "outside"]]:
    if limit_area is not None:
        valid_limit_areas: List[str] = ["inside", "outside"]
        limit_area = limit_area.lower()
        if limit_area not in valid_limit_areas:
            raise ValueError(
                f"Invalid limit_area: expecting one of {valid_limit_areas}, got "
                f"{limit_area}."
            )
    return limit_area  # type: ignore[return-value]


def infer_limit_direction(
    limit_direction: Optional[Literal["backward", "forward", "both"]], method: str
) -> Literal["backward", "forward", "both"]:
    # Set `limit_direction` depending on `method`
    if limit_direction is None:
        if method in ("backfill", "bfill"):
            limit_direction = "backward"
        else:
            limit_direction = "forward"
    else:
        if method in ("pad", "ffill") and limit_direction != "forward":
            raise ValueError(
                f"`limit_direction` must be 'forward' for method `{method}`"
            )
        if method in ("backfill", "bfill") and limit_direction != "backward":
            raise ValueError(
                f"`limit_direction` must be 'backward' for method `{method}`"
            )
    return limit_direction


def get_interp_index(method: str, index: Any) -> Any:
    # create/use the index
    if method == "linear":
        from pandas import Index
        if isinstance(index.dtype, DatetimeTZDtype) or lib.is_np_dtype(
            index.dtype, "mM"
        ):
            index = Index(index.view("i8"))
        elif not is_numeric_dtype(index.dtype):
            index = Index(range(len(index)))
    else:
        methods = {"index", "values", "nearest", "time"}
        is_numeric_or_datetime = (
            is_numeric_dtype(index.dtype)
            or isinstance(index.dtype, DatetimeTZDtype)
            or lib.is_np_dtype(index.dtype, "mM")
        )
        valid = NP_METHODS + SP_METHODS
        if method in valid:
            if method not in methods and not is_numeric_or_datetime:
                raise ValueError(
                    "Index column must be numeric or datetime type when "
                    f"using {method} method other than linear. "
                    "Try setting a numeric or datetime index column before "
                    "interpolating."
                )
        else:
            raise ValueError(f"Can not interpolate with method={method}.")

    if isna(index).any():
        raise NotImplementedError(
            "Interpolation with NaNs in the index "
            "has not been implemented. Try filling "
            "those NaNs before interpolating."
        )
    return index


def interpolate_2d_inplace(
    data: np.ndarray,  # floating dtype
    index: Any,
    axis: AxisInt,
    method: str = "linear",
    limit: Optional[int] = None,
    limit_direction: str = "forward",
    limit_area: Optional[str] = None,
    fill_value: Optional[Any] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
    **kwargs: Any,
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

    if method == "time":
        if not needs_i8_conversion(index.dtype):
            raise ValueError(
                "time-weighted interpolation only works "
                "on Series or DataFrames with a "
                "DatetimeIndex"
            )
        method = "values"

    limit_direction = validate_limit_direction(limit_direction)
    limit_area_validated: Optional[Literal["inside", "outside"]] = validate_limit_area(limit_area)

    limit = algos.validate_limit(nobs=None, limit=limit)

    indices: np.ndarray = _index_to_interp_indices(index, method)

    def func(yvalues: np.ndarray) -> None:
        _interpolate_1d(
            indices=indices,
            yvalues=yvalues,
            method=method,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area_validated,
            fill_value=fill_value,
            bounds_error=False,
            mask=mask,
            **kwargs,
        )

    np.apply_along_axis(func, axis, data)  # type: ignore[call-overload]


def _index_to_interp_indices(index: Any, method: str) -> np.ndarray:
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
    xarr = index._values
    if needs_i8_conversion(xarr.dtype):
        xarr = xarr.view("i8")

    if method == "linear":
        inds = xarr
        inds = cast(np.ndarray, inds)
    else:
        inds = np.asarray(xarr)
        if method in ("values", "index"):
            if inds.dtype == np.object_:
                inds = lib.maybe_convert_objects(inds)
    return inds


def _interpolate_1d(
    indices: np.ndarray,
    yvalues: np.ndarray,
    method: str = "linear",
    limit: Optional[int] = None,
    limit_direction: str = "forward",
    limit_area: Optional[Literal["inside", "outside"]] = None,
    fill_value: Optional[Any] = None,
    bounds_error: bool = False,
    order: Optional[int] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
    **kwargs: Any,
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

    first_valid_index = find_valid_index(how="first", is_valid=valid)
    if first_valid_index is None:
        first_valid_index = 0
    start_nans = np.arange(first_valid_index)

    last_valid_index = find_valid_index(how="last", is_valid=valid)
    if last_valid_index is None:
        last_valid_index = len(yvalues)
    end_nans = np.arange(1 + last_valid_index, len(valid))

    if limit_direction == "forward":
        preserve_nans = np.union1d(start_nans, _interp_limit(invalid, limit, 0))
    elif limit_direction == "backward":
        preserve_nans = np.union1d(end_nans, _interp_limit(invalid, 0, limit))
    else:
        preserve_nans = np.unique(_interp_limit(invalid, limit, limit))

    if limit_area == "inside":
        preserve_nans = np.union1d(preserve_nans, start_nans)
        preserve_nans = np.union1d(preserve_nans, end_nans)
    elif limit_area == "outside":
        mid_nans = np.setdiff1d(all_nans, start_nans, assume_unique=True)
        mid_nans = np.setdiff1d(mid_nans, end_nans, assume_unique=True)
        preserve_nans = np.union1d(preserve_nans, mid_nans)

    is_datetimelike = yvalues.dtype.kind in "mM"

    if is_datetimelike:
        yvalues = yvalues.view("i8")

    if method in NP_METHODS:
        indexer = np.argsort(indices[valid])
        yvalues[invalid] = np.interp(
            indices[invalid], indices[valid][indexer], yvalues[valid][indexer]
        )
    else:
        yvalues[invalid] = _interpolate_scipy_wrapper(
            indices[valid],
            yvalues[valid],
            indices[invalid],
            method=method,
            fill_value=fill_value,
            bounds_error=bounds_error,
            order=order,
            **kwargs,
        )

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
    fill_value: Any = None,
    bounds_error: bool = False,
    order: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Passed off to scipy.interpolate.interp1d. method is scipy's kind.
    Returns an array interpolated at new_x.
    """
    extra = f"{method} interpolation requires SciPy."
    import_optional_dependency("scipy", extra=extra)
    from scipy import interpolate

    new_x = np.asarray(new_x)

    alt_methods: dict[str, Callable[..., Any]] = {
        "barycentric": interpolate.barycentric_interpolate,
        "krogh": interpolate.krogh_interpolate,
        "from_derivatives": _from_derivatives,
        "piecewise_polynomial": _from_derivatives,
        "cubicspline": _cubicspline_interpolate,
        "akima": _akima_interpolate,
        "pchip": interpolate.pchip_interpolate,
    }

    interp1d_methods: List[str] = [
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "polynomial",
    ]
    if method in interp1d_methods:
        if method == "polynomial":
            kind: Any = order
        else:
            kind = method
        terp = interpolate.interp1d(
            x, y, kind=kind, fill_value=fill_value, bounds_error=bounds_error
        )
        new_y: np.ndarray = terp(new_x)
    elif method == "spline":
        if isna(order) or (order <= 0):
            raise ValueError(
                f"order needs to be specified and greater than 0; got order: {order}"
            )
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
            raise ValueError(f"Can not interpolate with method={method}.")
        kwargs.pop("downcast", None)
        new_y = terp(x, y, new_x, **kwargs)
    return new_y


def _from_derivatives(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    order: Optional[Union[int, ArrayLike]] = None,
    der: Union[int, List[int]] | None = 0,
    extrapolate: bool = False,
) -> Any:
    """
    Convenience function for interpolate.BPoly.from_derivatives.
    """
    from scipy import interpolate
    method = interpolate.BPoly.from_derivatives
    m = method(xi, yi.reshape(-1, 1), orders=order, extrapolate=extrapolate)
    return m(x)


def _akima_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    der: Union[int, List[int]] | None = 0,
    axis: AxisInt = 0,
) -> np.ndarray:
    """
    Convenience function for akima interpolation.
    """
    from scipy import interpolate
    P = interpolate.Akima1DInterpolator(xi, yi, axis=axis)
    return P(x, nu=der)


def _cubicspline_interpolate(
    xi: np.ndarray,
    yi: np.ndarray,
    x: np.ndarray,
    axis: AxisInt = 0,
    bc_type: Union[str, Tuple[Any, Any]] = "not-a-knot",
    extrapolate: Optional[Union[bool, str]] = None,
) -> np.ndarray:
    """
    Convenience function for cubic spline data interpolator.
    """
    from scipy import interpolate
    P = interpolate.CubicSpline(
        xi, yi, axis=axis, bc_type=bc_type, extrapolate=extrapolate
    )
    return P(x)


def pad_or_backfill_inplace(
    values: np.ndarray,
    method: Literal["pad", "backfill"] = "pad",
    axis: AxisInt = 0,
    limit: Optional[int] = None,
    limit_area: Optional[Literal["inside", "outside"]] = None,
) -> None:
    """
    Perform an actual interpolation of values; values will be made 2-d if
    needed and filled in-place.
    """
    transf: Callable[[np.ndarray], np.ndarray] = (lambda x: x) if axis == 0 else (lambda x: x.T)

    if values.ndim == 1:
        if axis != 0:
            raise AssertionError("cannot interpolate on a ndim == 1 with axis != 0")
        values = values.reshape((1,) + values.shape)

    method = clean_fill_method(method)
    tvalues: np.ndarray = transf(values)

    func: Callable[..., Tuple[np.ndarray, npt.NDArray[np.bool_]]] = get_fill_func(method, ndim=2)
    func(tvalues, limit=limit, limit_area=limit_area)


def _fillna_prep(
    values: ArrayLike, mask: Optional[npt.NDArray[np.bool_]] = None
) -> npt.NDArray[np.bool_]:
    if mask is None:
        mask = isna(values)
    return mask


def _datetimelike_compat(func: F) -> F:
    """
    Wrapper to handle datetime64 and timedelta64 dtypes.
    """
    @wraps(func)
    def new_func(
        values: np.ndarray,
        limit: Optional[int] = None,
        limit_area: Optional[Literal["inside", "outside"]] = None,
        mask: Optional[npt.NDArray[np.bool_]] = None,
    ) -> Tuple[np.ndarray, npt.NDArray[np.bool_]]:
        if needs_i8_conversion(values.dtype):
            if mask is None:
                mask = isna(values)
            result, mask = func(
                values.view("i8"), limit=limit, limit_area=limit_area, mask=mask
            )
            return result.view(values.dtype), mask
        return func(values, limit=limit, limit_area=limit_area, mask=mask)
    return cast(F, new_func)


@_datetimelike_compat
def _pad_1d(
    values: np.ndarray,
    limit: Optional[int] = None,
    limit_area: Optional[Literal["inside", "outside"]] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[np.ndarray, npt.NDArray[np.bool_]]:
    mask = _fillna_prep(values, mask)
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    algos.pad_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _backfill_1d(
    values: np.ndarray,
    limit: Optional[int] = None,
    limit_area: Optional[Literal["inside", "outside"]] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[np.ndarray, npt.NDArray[np.bool_]]:
    mask = _fillna_prep(values, mask)
    if limit_area is not None and not mask.all():
        _fill_limit_area_1d(mask, limit_area)
    algos.backfill_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _pad_2d(
    values: np.ndarray,
    limit: Optional[int] = None,
    limit_area: Optional[Literal["inside", "outside"]] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[np.ndarray, npt.NDArray[np.bool_]]:
    mask = _fillna_prep(values, mask)
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)
    if values.size:
        algos.pad_2d_inplace(values, mask, limit=limit)
    return values, mask


@_datetimelike_compat
def _backfill_2d(
    values: np.ndarray,
    limit: Optional[int] = None,
    limit_area: Optional[Literal["inside", "outside"]] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[np.ndarray, npt.NDArray[np.bool_]]:
    mask = _fillna_prep(values, mask)
    if limit_area is not None:
        _fill_limit_area_2d(mask, limit_area)
    if values.size:
        algos.backfill_2d_inplace(values, mask, limit=limit)
    return values, mask


def _fill_limit_area_1d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    neg_mask = ~mask
    first = neg_mask.argmax()
    last = len(neg_mask) - neg_mask[::-1].argmax() - 1
    if limit_area == "inside":
        mask[:first] = False
        mask[last + 1 :] = False
    elif limit_area == "outside":
        mask[first + 1 : last] = False


def _fill_limit_area_2d(
    mask: npt.NDArray[np.bool_], limit_area: Literal["outside", "inside"]
) -> None:
    neg_mask = ~mask.T
    if limit_area == "outside":
        la_mask = (
            np.maximum.accumulate(neg_mask, axis=0)
            & np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    else:
        la_mask = (
            ~np.maximum.accumulate(neg_mask, axis=0)
            | ~np.maximum.accumulate(neg_mask[::-1], axis=0)[::-1]
        )
    mask[la_mask.T] = False


_fill_methods: dict[str, Callable[..., Any]] = {"pad": _pad_1d, "backfill": _backfill_1d}


def get_fill_func(method: str, ndim: int = 1) -> Callable[..., Tuple[np.ndarray, npt.NDArray[np.bool_]]]:
    method = clean_fill_method(method)
    if ndim == 1:
        return _fill_methods[method]
    return {"pad": _pad_2d, "backfill": _backfill_2d}[method]


def clean_reindex_fill_method(method: Any) -> Optional[ReindexMethod]:
    if method is None:
        return None
    return clean_fill_method(method, allow_nearest=True)


def _interp_limit(
    invalid: npt.NDArray[np.bool_], fw_limit: Optional[int], bw_limit: Optional[int]
) -> np.ndarray:
    N = len(invalid)
    f_idx: np.ndarray = np.array([], dtype=np.int64)
    b_idx: np.ndarray = np.array([], dtype=np.int64)
    assume_unique: bool = True

    def inner(invalid_arr: npt.NDArray[np.bool_], limit: int) -> np.ndarray:
        limit = min(limit, N)
        windowed = np.lib.stride_tricks.sliding_window_view(invalid_arr, limit + 1).all(1)
        idx = np.union1d(
            np.where(windowed)[0] + limit,
            np.where((~invalid_arr[: limit + 1]).cumsum() == 0)[0],
        )
        return idx

    if fw_limit is not None:
        if fw_limit == 0:
            f_idx = np.where(invalid)[0]
            assume_unique = False
        else:
            f_idx = inner(invalid, fw_limit)

    if bw_limit is not None:
        if bw_limit == 0:
            return f_idx
        else:
            b_idx = N - 1 - inner(invalid[::-1], bw_limit)
            if fw_limit == 0:
                return b_idx

    return np.intersect1d(f_idx, b_idx, assume_unique=assume_unique)