from __future__ import annotations

import functools
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np
from numpy.typing import NDArray

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
    from collections.abc import Callable as CallableT
    from typing import Protocol

    class ReductionFunc(Protocol):
        def __call__(
            self,
            values: np.ndarray,
            *,
            axis: AxisInt | None = None,
            skipna: bool = True,
            mask: NDArray[np.bool_] | None = None,
            **kwargs,
        ) -> Any: ...

bn = import_optional_dependency("bottleneck", errors="warn")
_BOTTLENECK_INSTALLED = bn is not None
_USE_BOTTLENECK = False


def set_use_bottleneck(v: bool = True) -> None:
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


set_use_bottleneck(get_option("compute.use_bottleneck"))


class disallow:
    def __init__(self, *dtypes: Dtype) -> None:
        super().__init__()
        self.dtypes = tuple(pandas_dtype(dtype).type for dtype in dtypes)

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
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(
            values: np.ndarray,
            *,
            axis: AxisInt | None = None,
            skipna: bool = True,
            **kwds: Any,
        ) -> Any:
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
    dtype: DtypeObj,
    fill_value: Scalar | None = None,
    fill_value_typ: str | None = None,
) -> Scalar:
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
    mask: NDArray[np.bool_] | None,
) -> NDArray[np.bool_] | None:
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
    fill_value_typ: str | None = None,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, NDArray[np.bool_] | None]:
    mask = _maybe_get_mask(values, skipna, mask)

    dtype = values.dtype

    datetimelike = False
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


def _get_dtype_max(dtype: np.dtype) -> np.dtype:
    dtype_max = dtype
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
    result: Any, dtype: np.dtype, fill_value: Any = None
) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
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
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: NDArray[np.bool_] | None = None,
        **kwargs: Any,
    ) -> Any:
        orig_values = values

        datetimelike = values.dtype.kind in "mM"
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


def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Scalar | np.ndarray:
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
    def newfunc(
        values: np.ndarray, *, axis: AxisInt | None = None, **kwargs: Any
    ) -> Any:
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
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: NDArray[np.bool_] | None = None,
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
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: NDArray[np.bool_] | None = None,
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
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    if dtype.kind == "f":
        dtype_sum = dtype
    elif dtype.kind == "m":
        dtype_sum = np.dtype(np.float64)

    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)

    return the_sum


def _mask_datetimelike_result(
    result: np.ndarray | np.datetime64 | np.timedelta64,
    axis: AxisInt | None,
    mask: NDArray[np.bool_],
    orig_values: np.ndarray,
) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    if isinstance(result, np.ndarray):
        result = result.astype("i8").view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        result[axis_mask] = iNaT
    else:
        if mask.any():
            return np.int64(iNaT).view(orig_values.dtype)
    return result


@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    dtype_count = np.dtype(np.float64)

    if dtype.kind in "mM":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in "iu":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == "f":
        dtype_sum = dtype
        dtype_count = dtype

    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
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
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: NDArray[np.bool_] | None = None,
) -> float | np.ndarray:
    using_nan_sentinel = values.dtype.kind == "f" and mask is None

    def get_median(x: np.ndarray, _mask: NDArray[np.bool_] | None = None) -> float:
        if _mask is None:
            _mask = notna(x)
        else:
            _mask = ~_mask
        if not skipna and not _mask.all():
            return np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            res = np.nanmedian(x[_mask])
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

    notempty = values.size

    res: float | np.ndarray

    if values.ndim > 1 and axis is not None:
        if notempty:
            if not skipna:
                res = np.apply_along_axis(get_median, axis, values)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "All-NaN slice encountered",