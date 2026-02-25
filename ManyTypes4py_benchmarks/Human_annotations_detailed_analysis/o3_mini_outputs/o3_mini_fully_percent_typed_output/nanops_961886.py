from __future__ import annotations

import functools
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Tuple,
    Union,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    NaTType,
    iNaT,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    CorrelationMethod,
    Dtype,
    DtypeObj,
    F,
    Scalar,
    Shape,
    npt,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.common import (
    is_complex,
    is_float,
    is_float_dtype,
    is_integer,
    is_numeric_dtype,
    is_object_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

if TYPE_CHECKING:
    from collections.abc import Callable

bn = import_optional_dependency("bottleneck", errors="warn")
_BOTTLENECK_INSTALLED: bool = bn is not None
_USE_BOTTLENECK: bool = False


def set_use_bottleneck(v: bool = True) -> None:
    # set/unset to use bottleneck
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


set_use_bottleneck(get_option("compute.use_bottleneck"))


class disallow:
    def __init__(self, *dtypes: Dtype) -> None:
        super().__init__()
        self.dtypes: Tuple[type, ...] = tuple(pandas_dtype(dtype).type for dtype in dtypes)

    def check(self, obj: Any) -> bool:
        return hasattr(obj, "dtype") and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:
        @functools.wraps(f)
        def _f(*args: Any, **kwargs: Any) -> Any:
            obj_iter = itertools.chain(args, kwargs.values())
            if any(self.check(obj) for obj in obj_iter):
                f_name = f.__name__.replace("nan", "")
                raise TypeError(
                    f"reduction operation '{f_name}' not allowed for this dtype"
                )
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                # we want to transform an object array
                # ValueError message to the more typical TypeError
                # e.g. this is normally a disallowed function on
                # object arrays that contain strings
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise

        return cast(F, _f)


class bottleneck_switch:
    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        self.name: Optional[str] = name
        self.kwargs: dict[str, Any] = kwargs

    def __call__(self, alt: F) -> F:
        bn_name: str = self.name or alt.__name__

        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(
            values: np.ndarray,
            *,
            axis: Optional[AxisInt] = None,
            skipna: bool = True,
            **kwds: Any,
        ) -> Union[np.ndarray, float]:
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v

            if values.size == 0 and kwds.get("min_count") is None:
                return _na_for_min_count(values, axis)

            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get("mask", None) is None:
                    kwds.pop("mask", None)
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
    if dtype != object and not needs_i8_conversion(dtype):
        return name not in ["nansum", "nanprod", "nanmean"]
    return False


def _has_infs(result: Any) -> bool:
    if isinstance(result, np.ndarray):
        if result.dtype in ("f8", "f4"):
            return lib.has_infs(result.ravel("K"))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        return False


def _get_fill_value(
    dtype: DtypeObj, fill_value: Optional[Scalar] = None, fill_value_typ: Optional[Any] = None
) -> Any:
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        else:
            if fill_value_typ == "+inf":
                return np.inf
            else:
                return -np.inf
    else:
        if fill_value_typ == "+inf":
            return lib.i8max
        else:
            return iNaT


def _maybe_get_mask(
    values: np.ndarray,
    skipna: bool,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Optional[npt.NDArray[np.bool_]]:
    if mask is None:
        if values.dtype.kind in "biu":
            return None

        if skipna or values.dtype.kind in "mM":
            mask = isna(values)
    return mask


def _get_values(
    values: np.ndarray,
    skipna: bool,
    fill_value: Any = None,
    fill_value_typ: Optional[str] = None,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[np.ndarray, Optional[npt.NDArray[np.bool_]]]:
    mask = _maybe_get_mask(values, skipna, mask)
    dtype = values.dtype
    datetimelike: bool = False
    if values.dtype.kind in "mM":
        values = np.asarray(values.view("i8"))
        datetimelike = True

    if skipna and (mask is not None):
        fill_value = _get_fill_value(
            dtype, fill_value=fill_value, fill_value_typ=fill_value_typ
        )
        if fill_value is not None:
            if mask.any():
                if datetimelike or _na_ok_dtype(dtype):
                    values = values.copy()
                    np.putmask(values, mask, fill_value)
                else:
                    values = np.where(~mask, values, fill_value)
    return values, mask


def _get_dtype_max(dtype: np.dtype[Any]) -> np.dtype[Any]:
    dtype_max: np.dtype[Any] = dtype
    if dtype.kind in "bi":
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == "u":
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == "f":
        dtype_max = np.dtype(np.float64)
    return dtype_max


def _na_ok_dtype(dtype: DtypeObj) -> bool:
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)


def _wrap_results(
    result: Union[np.ndarray, float, np.datetime64, np.timedelta64, NaTType],
    dtype: np.dtype[Any],
    fill_value: Optional[Any] = None,
) -> Union[np.ndarray, float, np.datetime64, np.timedelta64, NaTType]:
    if result is NaT:
        pass
    elif dtype.kind == "M":
        if fill_value is None:
            fill_value = iNaT
        if not isinstance(result, np.ndarray):
            assert not isna(fill_value), "Expected non-null fill_value"
            if result == fill_value:
                result = np.nan
            if isna(result):
                result = np.datetime64("NaT", "ns").astype(dtype)
            else:
                result = np.int64(result).view(dtype)
            result = result.astype(dtype, copy=False)
        else:
            result = result.astype(dtype)
    elif dtype.kind == "m":
        if not isinstance(result, np.ndarray):
            if result == fill_value or np.isnan(result):
                result = np.timedelta64("NaT").astype(dtype)
            elif np.fabs(result) > lib.i8max:
                raise ValueError("overflow in timedelta operation")
            else:
                result = np.int64(result).astype(dtype, copy=False)
        else:
            result = result.astype("m8[ns]").view(dtype)
    return result


def _datetimelike_compat(func: F) -> F:
    @functools.wraps(func)
    def new_func(
        values: np.ndarray,
        *,
        axis: Optional[AxisInt] = None,
        skipna: bool = True,
        mask: Optional[npt.NDArray[np.bool_]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, np.datetime64, np.timedelta64, NaTType, float]:
        orig_values = values
        datetimelike: bool = values.dtype.kind in "mM"
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


def _na_for_min_count(values: np.ndarray, axis: Optional[AxisInt]) -> Union[Scalar, np.ndarray]:
    if values.dtype.kind in "iufcb":
        values = values.astype("float64")
    fill_value = na_value_for_dtype(values.dtype)
    if values.ndim == 1:
        return fill_value
    elif axis is None:
        return fill_value
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1 :]
        return np.full(result_shape, fill_value, dtype=values.dtype)


def maybe_operate_rowwise(func: F) -> F:
    @functools.wraps(func)
    def newfunc(values: np.ndarray, *, axis: Optional[AxisInt] = None, **kwargs: Any) -> Any:
        if (
            axis == 1
            and values.ndim == 2
            and values.flags["C_CONTIGUOUS"]
            and (values.shape[1] / 1000) > values.shape[0]
            and values.dtype != object
            and values.dtype != bool
        ):
            arrs = list(values)
            if kwargs.get("mask") is not None:
                mask = kwargs.pop("mask")
                results = [
                    func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))
                ]
            else:
                results = [func(x, **kwargs) for x in arrs]
            return np.array(results)
        return func(values, axis=axis, **kwargs)
    return cast(F, newfunc)


def nanany(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> bool:
    if values.dtype.kind in "iub" and mask is None:
        return bool(values.any(axis))
    if values.dtype.kind == "M":
        raise TypeError("datetime64 type does not support operation 'any'")
    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)
    if values.dtype == object:
        values = values.astype(bool)
    return bool(values.any(axis))


def nanall(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> bool:
    if values.dtype.kind in "iub" and mask is None:
        return bool(values.all(axis))
    if values.dtype.kind == "M":
        raise TypeError("datetime64 type does not support operation 'all'")
    values, _ = _get_values(values, skipna, fill_value=True, mask=mask)
    if values.dtype == object:
        values = values.astype(bool)
    return bool(values.all(axis))


@disallow("M8")
@_datetimelike_compat
@maybe_operate_rowwise
def nansum(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> float | np.ndarray:
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum: np.dtype[Any] = _get_dtype_max(dtype)
    if dtype.kind == "f":
        dtype_sum = dtype
    elif dtype.kind == "m":
        dtype_sum = np.dtype(np.float64)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)
    return the_sum


def _mask_datetimelike_result(
    result: Union[np.ndarray, np.datetime64, np.timedelta64],
    axis: Optional[AxisInt],
    mask: npt.NDArray[np.bool_],
    orig_values: np.ndarray,
) -> Union[np.ndarray, np.datetime64, np.timedelta64, NaTType]:
    if isinstance(result, np.ndarray):
        result = result.astype("i8").view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        result[axis_mask] = iNaT  # type: ignore[index]
    else:
        if mask.any():
            return np.int64(iNaT).view(orig_values.dtype)
    return result


@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> float | np.ndarray:
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum: np.dtype[Any] = _get_dtype_max(dtype)
    dtype_count = np.dtype(np.float64)
    if dtype.kind in "mM":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in "iu":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == "f":
        dtype_sum = dtype
        dtype_count = dtype
    count: float | np.ndarray = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _ensure_numeric(the_sum)
    if axis is not None and getattr(the_sum, "ndim", False):
        count = cast(np.ndarray, count)
        with np.errstate(all="ignore"):
            the_mean = the_sum / count
        ct_mask = count == 0
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = the_sum / count if count > 0 else np.nan
    return the_mean


@bottleneck_switch()
def nanmedian(
    values: np.ndarray, *, axis: Optional[AxisInt] = None, skipna: bool = True, mask: Any = None
) -> float | np.ndarray:
    using_nan_sentinel: bool = values.dtype.kind == "f" and mask is None

    def get_median(x: np.ndarray, _mask: Optional[np.ndarray] = None) -> float:
        if _mask is None:
            _mask_local = notna(x)
        else:
            _mask_local = ~_mask
        if not skipna and not _mask_local.all():
            return np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            res = np.nanmedian(x[_mask_local])
        return res

    dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask, fill_value=None)
    if values.dtype.kind != "f":
        if values.dtype == object:
            inferred = lib.infer_dtype(values)
            if inferred in ["string", "mixed"]:
                raise TypeError(f"Cannot convert {values} to numeric")
        try:
            values = values.astype("f8")
        except ValueError as err:
            raise TypeError(str(err)) from err
    if not using_nan_sentinel and mask is not None:
        if not values.flags.writeable:
            values = values.copy()
        values[mask] = np.nan
    notempty: int = values.size
    res: float | np.ndarray
    if values.ndim > 1 and axis is not None:
        if notempty:
            if not skipna:
                res = np.apply_along_axis(get_median, axis, values)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "All-NaN slice encountered", RuntimeWarning
                    )
                    if (values.shape[1] == 1 and axis == 0) or (
                        values.shape[0] == 1 and axis == 1
                    ):
                        res = np.nanmedian(np.squeeze(values), keepdims=True)
                    else:
                        res = np.nanmedian(values, axis=axis)
        else:
            res = _get_empty_reduction_result(values.shape, axis)
    else:
        res = get_median(values, mask) if notempty else np.nan
    return _wrap_results(res, dtype)


def _get_empty_reduction_result(
    shape: Shape,
    axis: AxisInt,
) -> np.ndarray:
    shp = np.array(shape)
    dims = np.arange(len(shape))
    ret = np.empty(shp[dims != axis], dtype=np.float64)
    ret.fill(np.nan)
    return ret


def _get_counts_nanvar(
    values_shape: Shape,
    mask: Optional[npt.NDArray[np.bool_]],
    axis: Optional[AxisInt],
    ddof: int,
    dtype: np.dtype[Any] = np.dtype(np.float64),
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    count: float | np.ndarray = _get_counts(values_shape, mask, axis, dtype=dtype)
    d: float | np.ndarray = count - dtype.type(ddof)
    if is_float(count):
        if count <= ddof:
            count = np.nan  # type: ignore[assignment]
            d = np.nan
    else:
        count_arr = cast(np.ndarray, count)
        mask_arr = count_arr <= ddof
        if mask_arr.any():
            np.putmask(d, mask_arr, np.nan)
            np.putmask(count_arr, mask_arr, np.nan)
            count = count_arr
    return count, d


@bottleneck_switch(ddof=1)
def nanstd(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: Any = None,
) -> float | np.ndarray:
    if values.dtype == "M8[ns]":
        values = values.view("m8[ns]")
    orig_dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask)
    result: float | np.ndarray = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    return _wrap_results(result, orig_dtype)


@disallow("M8", "m8")
@bottleneck_switch(ddof=1)
def nanvar(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: Any = None,
) -> float | np.ndarray:
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if dtype.kind in "iu":
        values = values.astype("f8")
        if mask is not None:
            values[mask] = np.nan
    if values.dtype.kind == "f":
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof)
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)
    sqr = _ensure_numeric((avg - values) ** 2)
    if mask is not None:
        np.putmask(sqr, mask, 0)
    result = sqr.sum(axis=axis, dtype=np.float64) / d
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    return result


@disallow("M8", "m8")
def nansem(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> float:
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
    if not skipna and mask is not None and mask.any():
        return np.nan
    count: float | np.ndarray = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)[0]
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)
    return np.sqrt(var) / np.sqrt(count)


def _nanminmax(meth: str, fill_value_typ: str) -> Callable[
    [np.ndarray, *, Optional[AxisInt], bool, Optional[npt.NDArray[np.bool_]]], float | np.ndarray
]:
    @bottleneck_switch(name=f"nan{meth}")
    @_datetimelike_compat
    def reduction(
        values: np.ndarray,
        *,
        axis: Optional[AxisInt] = None,
        skipna: bool = True,
        mask: Optional[npt.NDArray[np.bool_]] = None,
    ) -> float | np.ndarray:
        if values.size == 0:
            return _na_for_min_count(values, axis)
        dtype = values.dtype
        values, mask = _get_values(values, skipna, fill_value_typ=fill_value_typ, mask=mask)
        result = getattr(values, meth)(axis)
        result = _maybe_null_out(
            result, axis, mask, values.shape, datetimelike=dtype.kind in "mM"
        )
        return result
    return reduction


nanmin: Callable[[np.ndarray, Any], float | np.ndarray] = _nanminmax("min", fill_value_typ="+inf")
nanmax: Callable[[np.ndarray, Any], float | np.ndarray] = _nanminmax("max", fill_value_typ="-inf")


def nanargmax(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> int | np.ndarray:
    values, mask = _get_values(values, True, fill_value_typ="-inf", mask=mask)
    result: np.ndarray = values.argmax(axis)
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result


def nanargmin(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> int | np.ndarray:
    values, mask = _get_values(values, True, fill_value_typ="+inf", mask=mask)
    result: np.ndarray = values.argmin(axis)
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanskew(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> float:
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count: float | np.ndarray = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)
    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted3 = adjusted2 * adjusted
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m3 = adjusted3.sum(axis, dtype=np.float64)
    m2 = _zero_out_fperr(m2)
    m3 = _zero_out_fperr(m3)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)
    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    if isinstance(result, np.ndarray):
        result = np.where(m2 == 0, 0, result)
        result[count < 3] = np.nan
    else:
        result = dtype.type(0) if m2 == 0 else result
        if count < 3:
            return np.nan
    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nankurt(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> float:
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count: float | np.ndarray = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)
    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)
    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted4 = adjusted2**2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2**2
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)
    if not isinstance(denominator, np.ndarray):
        if count < 4:
            return np.nan
        if denominator == 0:
            return values.dtype.type(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator - adj
    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan
    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanprod(
    values: np.ndarray,
    *,
    axis: Optional[AxisInt] = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> float | np.ndarray:
    mask = _maybe_get_mask(values, skipna, mask)
    if skipna and mask is not None:
        values = values.copy()
        values[mask] = 1
    result = values.prod(axis)
    return _maybe_null_out(result, axis, mask, values.shape, min_count=min_count)


def _maybe_arg_null_out(
    result: np.ndarray,
    axis: Optional[AxisInt],
    mask: Optional[npt.NDArray[np.bool_]],
    skipna: bool,
) -> int | np.ndarray:
    if mask is None:
        return result
    if axis is None or not getattr(result, "ndim", False):
        if skipna and mask.all():
            raise ValueError("Encountered all NA values")
        elif not skipna and mask.any():
            raise ValueError("Encountered an NA value with skipna=False")
    else:
        if skipna and mask.all(axis).any():
            raise ValueError("Encountered all NA values")
        elif not skipna and mask.any(axis).any():
            raise ValueError("Encountered an NA value with skipna=False")
    return result


def _get_counts(
    values_shape: Shape,
    mask: Optional[npt.NDArray[np.bool_]],
    axis: Optional[AxisInt],
    dtype: np.dtype[Any] = np.dtype(np.float64),
) -> float | np.ndarray:
    if axis is None:
        if mask is not None:
            n = mask.size - mask.sum()
        else:
            n = np.prod(values_shape)
        return dtype.type(n)
    if mask is not None:
        count = mask.shape[axis] - mask.sum(axis)
    else:
        count = values_shape[axis]
    if is_integer(count):
        return dtype.type(count)
    return count.astype(dtype, copy=False)


def _maybe_null_out(
    result: Union[np.ndarray, float, NaTType],
    axis: Optional[AxisInt],
    mask: Optional[npt.NDArray[np.bool_]],
    shape: Tuple[int, ...],
    min_count: int = 1,
    datetimelike: bool = False,
) -> Union[np.ndarray, float, NaTType]:
    if mask is None and min_count == 0:
        return result
    if axis is not None and isinstance(result, np.ndarray):
        if mask is not None:
            null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        else:
            below_count = shape[axis] - min_count < 0
            new_shape = shape[:axis] + shape[axis + 1 :]
            null_mask = np.broadcast_to(below_count, new_shape)
        if np.any(null_mask):
            if datetimelike:
                result[null_mask] = iNaT
            elif is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    result = result.astype("c16")
                elif not is_float_dtype(result):
                    result = result.astype("f8", copy=False)
                result[null_mask] = np.nan
            else:
                result[null_mask] = None
    elif result is not NaT:
        if check_below_min_count(shape, mask, min_count):
            result_dtype = getattr(result, "dtype", None)
            if is_float_dtype(result_dtype):
                result = result_dtype.type("nan")  # type: ignore[union-attr]
            else:
                result = np.nan
    return result


def check_below_min_count(
    shape: Tuple[int, ...], mask: Optional[npt.NDArray[np.bool_]], min_count: int
) -> bool:
    if min_count > 0:
        if mask is None:
            non_nulls = np.prod(shape)
        else:
            non_nulls = mask.size - mask.sum()
        if non_nulls < min_count:
            return True
    return False


def _zero_out_fperr(arg: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(arg, np.ndarray):
        return np.where(np.abs(arg) < 1e-14, 0, arg)
    else:
        return arg if np.abs(arg) >= 1e-14 else type(arg)(0)


@disallow("M8", "m8")
def nancorr(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: Optional[int] = None,
) -> float:
    if len(a) != len(b):
        raise AssertionError("Operands to nancorr must have same size")
    if min_periods is None:
        min_periods = 1
    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]
    if len(a) < min_periods:
        return np.nan
    a = _ensure_numeric(a)
    b = _ensure_numeric(b)
    f: Callable[[np.ndarray, np.ndarray], float] = get_corr_func(method)
    return f(a, b)


def get_corr_func(
    method: CorrelationMethod,
) -> Callable[[np.ndarray, np.ndarray], float]:
    if method == "kendall":
        from scipy.stats import kendalltau

        def func(a: np.ndarray, b: np.ndarray) -> float:
            return kendalltau(a, b)[0]
        return func
    elif method == "spearman":
        from scipy.stats import spearmanr

        def func(a: np.ndarray, b: np.ndarray) -> float:
            return spearmanr(a, b)[0]
        return func
    elif method == "pearson":
        def func(a: np.ndarray, b: np.ndarray) -> float:
            return np.corrcoef(a, b)[0, 1]
        return func
    elif callable(method):
        return method
    raise ValueError(
        f"Unknown method '{method}', expected one of "
        "'kendall', 'spearman', 'pearson', or callable"
    )


@disallow("M8", "m8")
def nancov(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_periods: Optional[int] = None,
    ddof: Optional[int] = 1,
) -> float:
    if len(a) != len(b):
        raise AssertionError("Operands to nancov must have same size")
    if min_periods is None:
        min_periods = 1
    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]
    if len(a) < min_periods:
        return np.nan
    a = _ensure_numeric(a)
    b = _ensure_numeric(b)
    return np.cov(a, b, ddof=ddof)[0, 1]


def _ensure_numeric(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "biu":
            x = x.astype(np.float64)
        elif x.dtype == object:
            inferred = lib.infer_dtype(x)
            if inferred in ["string", "mixed"]:
                raise TypeError(f"Could not convert {x} to numeric")
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    raise TypeError(f"Could not convert {x} to numeric") from err
            else:
                if not np.any(np.imag(x)):
                    x = x.real
    elif not (is_float(x) or is_integer(x) or is_complex(x)):
        if isinstance(x, str):
            raise TypeError(f"Could not convert string '{x}' to numeric")
        try:
            x = float(x)
        except (TypeError, ValueError):
            try:
                x = complex(x)
            except ValueError as err:
                raise TypeError(f"Could not convert {x} to numeric") from err
    return x


def na_accum_func(values: ArrayLike, accum_func: Callable, *, skipna: bool) -> ArrayLike:
    mask_a, mask_b = {
        np.cumprod: (1.0, np.nan),
        np.maximum.accumulate: (-np.inf, np.nan),
        np.cumsum: (0.0, np.nan),
        np.minimum.accumulate: (np.inf, np.nan),
    }[accum_func]
    assert values.dtype.kind not in "mM"
    if skipna and not issubclass(values.dtype.type, (np.integer, np.bool_)):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)
    return result