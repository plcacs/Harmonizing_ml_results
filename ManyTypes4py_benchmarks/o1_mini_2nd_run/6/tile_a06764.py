"""
Quantilization functions and related stuff
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Union, Sequence, Optional, Tuple
import numpy as np
from pandas._libs import Timedelta, Timestamp, lib
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas import Categorical, Index, IntervalIndex
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import DtypeObj, IntervalLeftRight


def cut(
    x: Union[np.ndarray, ABCSeries],
    bins: Union[int, Sequence[Union[int, float, np.number]], IntervalIndex],
    right: bool = True,
    labels: Optional[Union[Sequence[Any], bool]] = None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = "raise",
    ordered: bool = True,
) -> Union[Categorical, ABCSeries, np.ndarray, Tuple[Union[Categorical, ABCSeries, np.ndarray], Union[np.ndarray, IntervalIndex]]]:
    """
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins, or a
    pre-specified array of bins.

    ... [docstring truncated for brevity] ...
    """
    original = x
    x_idx = _preprocess_for_cut(x)
    x_idx, _ = _coerce_to_type(x_idx)
    if not np.iterable(bins):
        bins = _nbins_to_bins(x_idx, bins, right)
    elif isinstance(bins, IntervalIndex):
        if bins.is_overlapping:
            raise ValueError("Overlapping IntervalIndex is not accepted.")
    else:
        bins = Index(bins)
        if not bins.is_monotonic_increasing:
            raise ValueError("bins must increase monotonically.")
    fac, bins = _bins_to_cuts(
        x_idx,
        bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        duplicates=duplicates,
        ordered=ordered,
    )
    return _postprocess_for_cut(fac, bins, retbins, original)


def qcut(
    x: Union[np.ndarray, ABCSeries],
    q: Union[int, Sequence[float]],
    labels: Optional[Union[Sequence[Any], bool]] = None,
    retbins: bool = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
) -> Union[Categorical, ABCSeries, np.ndarray, Tuple[Union[Categorical, ABCSeries, np.ndarray], np.ndarray]]:
    """
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or based
    on sample quantiles. For example 1000 values for 10 quantiles would
    produce a Categorical object indicating quantile membership for each data point.

    ... [docstring truncated for brevity] ...
    """
    original = x
    x_idx = _preprocess_for_cut(x)
    x_idx, _ = _coerce_to_type(x_idx)
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
        np.putmask(quantiles, q * quantiles != np.arange(q + 1), np.nextafter(quantiles, 1))
    else:
        quantiles = q
    bins = x_idx.to_series().dropna().quantile(quantiles)
    fac, bins = _bins_to_cuts(
        x_idx,
        Index(bins),
        labels=labels,
        precision=precision,
        include_lowest=True,
        duplicates=duplicates,
    )
    return _postprocess_for_cut(fac, bins, retbins, original)


def _nbins_to_bins(
    x_idx: Index,
    nbins: int,
    right: bool,
) -> IntervalIndex:
    """
    If a user passed an integer N for bins, convert this to a sequence of N
    equal(ish)-sized bins.
    """
    if is_scalar(nbins) and nbins < 1:
        raise ValueError("`bins` should be a positive integer.")
    if x_idx.size == 0:
        raise ValueError("Cannot cut empty array")
    rng = (x_idx.min(), x_idx.max())
    mn, mx = rng
    if is_numeric_dtype(x_idx.dtype) and (np.isinf(mn) or np.isinf(mx)):
        raise ValueError("cannot specify integer `bins` when input data contains infinity")
    if mn == mx:
        if _is_dt_or_td(x_idx.dtype):
            unit = dtype_to_unit(x_idx.dtype)
            td = Timedelta(seconds=1).as_unit(unit)
            bins = x_idx._values._generate_range(
                start=mn - td, end=mx + td, periods=nbins + 1, freq=None, unit=unit
            )
        else:
            mn -= 0.001 * abs(mn) if mn != 0 else 0.001
            mx += 0.001 * abs(mx) if mx != 0 else 0.001
            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
    else:
        if _is_dt_or_td(x_idx.dtype):
            unit = dtype_to_unit(x_idx.dtype)
            bins = x_idx._values._generate_range(
                start=mn, end=mx, periods=nbins + 1, freq=None, unit=unit
            )
        else:
            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
        adj = (mx - mn) * 0.001
        if right:
            bins[0] -= adj
        else:
            bins[-1] += adj
    return Index(bins)


def _bins_to_cuts(
    x_idx: Index,
    bins: Union[Index, IntervalIndex],
    right: bool = True,
    labels: Optional[Union[Sequence[Any], bool]] = None,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = "raise",
    ordered: bool = True,
) -> Tuple[Union[Categorical, np.ndarray], Union[Index, IntervalIndex]]:
    if not ordered and labels is None:
        raise ValueError("'labels' must be provided if 'ordered = False'")
    if duplicates not in ["raise", "drop"]:
        raise ValueError(
            "invalid value for 'duplicates' parameter, valid options are: raise, drop"
        )
    if isinstance(bins, IntervalIndex):
        ids = bins.get_indexer(x_idx)
        cat_dtype = CategoricalDtype(bins, ordered=True)
        result = Categorical.from_codes(ids, dtype=cat_dtype, validate=False)
        return result, bins
    unique_bins = algos.unique(bins)
    if len(unique_bins) < len(bins) and len(bins) != 2:
        if duplicates == "raise":
            raise ValueError(
                f"Bin edges must be unique: {bins!r}.\nYou can drop duplicate edges by setting the 'duplicates' kwarg"
            )
        bins = unique_bins
    side = "left" if right else "right"
    try:
        ids = bins.searchsorted(x_idx, side=side)
    except TypeError as err:
        if x_idx.dtype.kind == "m":
            raise ValueError("bins must be of timedelta64 dtype") from err
        elif x_idx.dtype.kind == "M" and bins.dtype.kind == "M":
            raise ValueError(
                "Cannot use timezone-naive bins with timezone-aware values, or vice-versa"
            ) from err
        elif x_idx.dtype.kind == "M":
            raise ValueError("bins must be of datetime64 dtype") from err
        else:
            raise
    ids = ensure_platform_int(ids)
    if include_lowest:
        ids[x_idx == bins[0]] = 1
    na_mask = isna(x_idx) | (ids == len(bins)) | (ids == 0)
    has_nas = na_mask.any()
    if labels is not False:
        if not (labels is None or is_list_like(labels)):
            raise ValueError(
                "Bin labels must either be False, None or passed in as a list-like argument"
            )
        if labels is None:
            labels = _format_labels(bins, precision, right=right, include_lowest=include_lowest)
        elif ordered and len(set(labels)) != len(labels):
            raise ValueError(
                "labels must be unique if ordered=True; pass ordered=False for duplicate labels"
            )
        elif len(labels) != len(bins) - 1:
            raise ValueError("Bin labels must be one fewer than the number of bin edges")
        if not isinstance(getattr(labels, "dtype", None), CategoricalDtype):
            labels = Categorical(
                labels,
                categories=labels if len(set(labels)) == len(labels) else None,
                ordered=ordered,
            )
        np.putmask(ids, na_mask, 0)
        result = algos.take_nd(labels, ids - 1)
    else:
        result = ids - 1
        if has_nas:
            result = result.astype(np.float64)
            np.putmask(result, na_mask, np.nan)
    return result, bins


def _coerce_to_type(x: Index) -> Tuple[Index, Optional[ExtensionDtype]]:
    """
    if the passed data is of datetime/timedelta, bool or nullable int type,
    this method converts it to numeric so that cut or qcut method can
    handle it
    """
    dtype: Optional[ExtensionDtype] = None
    if _is_dt_or_td(x.dtype):
        dtype = x.dtype
    elif is_bool_dtype(x.dtype):
        x = x.astype(np.int64)
    elif isinstance(x.dtype, ExtensionDtype) and is_numeric_dtype(x.dtype):
        x_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
        x = Index(x_arr)
    return x, dtype


def _is_dt_or_td(dtype: Any) -> bool:
    return isinstance(dtype, DatetimeTZDtype) or lib.is_np_dtype(dtype, "mM")


def _format_labels(
    bins: Index,
    precision: int,
    right: bool = True,
    include_lowest: bool = False,
) -> IntervalIndex:
    """based on the dtype, return our labels"""
    closed = "right" if right else "left"
    if _is_dt_or_td(bins.dtype):
        unit = dtype_to_unit(bins.dtype)
        formatter: Callable[[Any], Any] = lambda x: x
        adjust: Callable[[Any], Any] = lambda x: x - Timedelta(1, unit=unit).as_unit(unit)
    else:
        precision = _infer_precision(precision, bins)
        formatter = lambda x: _round_frac(x, precision)
        adjust = lambda x: x - 10 ** (-precision)
    breaks = [formatter(b) for b in bins]
    if right and include_lowest:
        breaks[0] = adjust(breaks[0])
    if _is_dt_or_td(bins.dtype):
        breaks = type(bins)(breaks).as_unit(unit)
    return IntervalIndex.from_breaks(breaks, closed=closed)


def _preprocess_for_cut(x: Union[np.ndarray, ABCSeries]) -> Index:
    """
    handles preprocessing for cut where we convert passed
    input to array, strip the index information and store it
    separately
    """
    ndim = getattr(x, "ndim", None)
    if ndim is None:
        x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input array must be 1 dimensional")
    return Index(x)


def _postprocess_for_cut(
    fac: Union[Categorical, np.ndarray],
    bins: Union[Index, IntervalIndex],
    retbins: bool,
    original: Union[np.ndarray, ABCSeries],
) -> Union[Categorical, ABCSeries, np.ndarray, Tuple[Union[Categorical, ABCSeries, np.ndarray], Union[np.ndarray, IntervalIndex]]]:
    """
    handles post processing for the cut method where
    we combine the index information if the originally passed
    datatype was a series
    """
    if isinstance(original, ABCSeries):
        fac = original._constructor(fac, index=original.index, name=original.name)
    if not retbins:
        return fac
    if isinstance(bins, Index) and is_numeric_dtype(bins.dtype):
        bins = bins._values
    return (fac, bins)


def _round_frac(x: float, precision: int) -> float:
    """
    Round the fractional part of the given number
    """
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return np.around(x, digits)


def _infer_precision(base_precision: int, bins: Index) -> int:
    """
    Infer an appropriate precision for _round_frac
    """
    for precision in range(base_precision, 20):
        levels = np.asarray([_round_frac(b, precision) for b in bins])
        if algos.unique(levels).size == bins.size:
            return precision
    return base_precision
