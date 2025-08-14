"""
Routines for filling missing data.
"""

from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray

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

if TYPE_CHECKING:
    from pandas import Index

T = TypeVar('T')

def check_value_size(value: Union[ArrayLike, Any], mask: npt.NDArray[np.bool_], length: int) -> ArrayLike:
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


def mask_missing(arr: ArrayLike, values_to_mask: Union[Sequence[Any], Any]) -> npt.NDArray[np.bool_]:
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

    potential_na = False
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
                    new_mask = np.zeros(arr.shape, dtype=np.bool_)
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
        method = method.lower()
        if method == "ffill":
            method = "pad"
        elif method == "bfill":
            method = "backfill"

    valid_methods = ["pad", "backfill"]
    expecting = "pad (ffill) or backfill (bfill)"
    if allow_nearest:
        valid_methods.append("nearest")
        expecting = "pad (ffill), backfill (bfill) or nearest"
    if method not in valid_methods:
        raise ValueError(f"Invalid fill method. Expecting {expecting}. Got {method}")
    return method


# interpolation methods that dispatch to np.interp

NP_METHODS: list[str] = ["linear", "time", "index", "values"]

# interpolation methods that dispatch to _interpolate_scipy_wrapper

SP_METHODS: list[str] = [
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


def clean_interp_method(method: str, index: Index, **kwargs) -> str:
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


def find_valid_index(how: Literal["first", "last"], is_valid: npt.NDArray[np.bool_]) -> Optional[int]:
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
    return idxpos


def validate_limit_direction(
    limit_direction: str,
) -> Literal["forward", "backward", "both"]:
    valid_limit_directions = ["forward", "backward", "both"]
    limit_direction = limit_direction.lower()
    if limit_direction not in valid_limit_directions:
        raise ValueError(
            "Invalid limit_direction: expecting one of "
            f"{valid_limit_directions}, got '{limit_direction}'."
        )
    return limit_direction


def validate_limit_area(limit_area: Optional[str]) -> Optional[Literal["inside", "outside"]]:
    if limit_area is not None:
        valid_limit_areas = ["inside", "outside"]
        limit_area = limit_area.lower()
        if limit_area not in valid_limit_areas:
            raise ValueError(
                f"Invalid limit_area: expecting one of {valid_limit_areas}, got "
                f"{limit_area}."
            )
    return limit_area


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


def get_interp_index(method: str, index: Index) -> Index:
    # create/use the index
    if method == "linear":
        # prior default
        from pandas import Index

        if isinstance(index.dtype, DatetimeTZDtype) or lib.is_np_dtype(
            index.dtype, "mM"
        ):
            # Convert datetime-like indexes to int64
            index = Index(index.view("i8"))

        elif not is_numeric_dtype(index.dtype):
            # We keep behavior consistent with prior versions of pandas for
            # non-numeric, non-datetime indexes
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
    index: Index,
    axis: AxisInt,
    method: str = "linear",
    limit: Optional[int] = None,
    limit_direction: str = "forward",
    limit_area: Optional[str] = None,
    fill_value: Optional[Any] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
    **kwargs,
) -> None:
    """
    Column-wise application of _interpolate_1d.

    Notes
    -----
    Alters 'data' in-place.

    The signature does differ from _interpolate_1d because it only
    includes what is needed for Block.interpolate.
    """
    # validate the interp method
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
    limit_area_validated = validate_limit_area(limit_area)

    # default limit is unlimited GH #16282
    limit = algos.validate_limit(nobs=None, limit=limit)

    indices = _index_to_interp_indices(index, method)

    def func(yvalues: np.ndarray) -> None:
        # process 1-d slices in the axis direction

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

    np.apply_along_axis(func, axis, data)


def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
    xarr = index._values
    if needs_i8_conversion(xarr.dtype):
        # GH#1646 for dt64tz
        xarr = xarr.view("i8")

    if method == "linear":
        inds = xarr
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
    **kwargs,
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

    # These index pointers to invalid values... i.e. {0, 1, etc...
    all_nans = np.flatnonzero(invalid)

    first_valid_index = find_valid_index(how="first", is_valid=valid)
    if first_valid_index is None:  # no nan found in start
        first_valid_index = 0
    start_nans = np.arange(first_valid_index)

    last_valid_index = find_valid_index(how="last", is_valid=valid)
    if last_valid_index is None:  # no nan found in end
        last_valid_index = len(yvalues)
    end_nans = np.arange(1 + last_valid_index, len(valid))

    # preserve_nans contains indices of invalid values,
    # but in this case, it is the final set of indices that need to be
    # preserved as NaN after the interpolation.

    # For example if limit_direction='forward' then preserve_nans will
    # contain indices of NaNs at the beginning of the series, and NaNs that
    # are more than 'limit' away from the prior non-NaN.

    # set preserve_nans based on direction using _interp_limit
    if limit_direction == "forward":
        preserve_nans = np.union1d(start_nans, _interp_limit(invalid, limit, 0))
    elif limit_direction == "backward":
        preserve_nans = np.union1d(end_nans, _interp_limit(invalid, 0, limit))
    else:
        # both directions... just use _interp_limit
        preserve_nans = np.unique(_interp_limit(invalid, limit, limit))

    # if limit_area is set, add either mid or outside indices
    # to preserve_nans GH #16284
    if limit_area == "inside":
        # preserve NaNs on the outside
        preserve_nans = np.union1d(preserve_nans, start_nans)
        preserve_nans = np.union1d(preserve_nans, end_nans)
    elif limit_area == "outside":
        # preserve NaNs on the inside
        mid_nans = np.setdiff1d(all_nans, start_nans, assume_unique=True)
        mid_nans = np.setdiff1d(mid_nans, end_nans, assume_unique=True)
        preserve_nans = np.union1d(preserve_nans, mid_nans)

    is_datetimelike = yvalues.dtype.kind in "mM"

    if is_datetimelike:
        yvalues = yvalues.view("i8")

    if method in NP_METHODS:
        # np.interp requires sorted X values, #21037

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
            bounds_error=bounds_error