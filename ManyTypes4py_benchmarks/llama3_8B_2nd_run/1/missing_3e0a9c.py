from __future__ import annotations
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, cast, overload
import numpy as np
from pandas._libs import NaT, algos, lib
from pandas._typing import ArrayLike, AxisInt, F, ReindexMethod, npt
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_numeric_dtype, is_numeric_v_string_like, is_object_dtype, needs_i8_conversion
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna, na_value_for_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna, na_value_for_dtype

def check_value_size(value: Any, mask: np.ndarray, length: int) -> Any:
    ...

def mask_missing(arr: ArrayLike, values_to_mask: Any) -> np.ndarray:
    ...

@overload
def clean_fill_method(method: str, *, allow_nearest: ...) -> str:
    ...

@overload
def clean_fill_method(method: str, *, allow_nearest: Literal[True]) -> str:
    ...

def clean_fill_method(method: str, *, allow_nearest: bool = False) -> str:
    ...

def clean_interp_method(method: str, index: Index, **kwargs: Any) -> str:
    ...

def find_valid_index(how: Literal['first', 'last'], is_valid: np.ndarray) -> int | None:
    ...

def validate_limit_direction(limit_direction: str) -> str:
    ...

def validate_limit_area(limit_area: str | None) -> str | None:
    ...

def interpolate_2d_inplace(data: ArrayLike, index: Index, axis: AxisInt, method: str, limit: Any, limit_direction: str, limit_area: str | None, fill_value: Any, mask: np.ndarray | None, **kwargs: Any) -> None:
    ...

def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    ...

def _interpolate_1d(indices: np.ndarray, yvalues: ArrayLike, method: str, limit: Any, limit_direction: str, limit_area: str | None, fill_value: Any, bounds_error: bool, order: int | None, mask: np.ndarray | None, **kwargs: Any) -> ArrayLike:
    ...

def _interpolate_scipy_wrapper(x: np.ndarray, y: ArrayLike, new_x: np.ndarray, method: str, fill_value: Any, bounds_error: bool, order: int | None, **kwargs: Any) -> ArrayLike:
    ...

def _from_derivatives(xi: np.ndarray, yi: ArrayLike, x: np.ndarray, order: int | None, der: int | None, extrapolate: bool) -> ArrayLike:
    ...

def _akima_interpolate(xi: np.ndarray, yi: ArrayLike, x: np.ndarray, der: int | None, axis: int) -> ArrayLike:
    ...

def _cubicspline_interpolate(xi: np.ndarray, yi: ArrayLike, x: np.ndarray, axis: int, bc_type: str | tuple, extrapolate: bool | None) -> ArrayLike:
    ...

def pad_or_backfill_inplace(values: ArrayLike, method: str, axis: int, limit: Any, limit_area: str | None) -> tuple:
    ...

def _fillna_prep(values: ArrayLike, mask: np.ndarray | None) -> np.ndarray:
    ...

def _datetimelike_compat(func: F) -> F:
    ...

@_datetimelike_compat
def _pad_1d(values: ArrayLike, limit: Any, limit_area: str | None, mask: np.ndarray | None) -> tuple:
    ...

@_datetimelike_compat
def _backfill_1d(values: ArrayLike, limit: Any, limit_area: str | None, mask: np.ndarray | None) -> tuple:
    ...

@_datetimelike_compat
def _pad_2d(values: ArrayLike, limit: Any, limit_area: str | None, mask: np.ndarray | None) -> tuple:
    ...

@_datetimelike_compat
def _backfill_2d(values: ArrayLike, limit: Any, limit_area: str | None, mask: np.ndarray | None) -> tuple:
    ...

def _fill_limit_area_1d(mask: np.ndarray, limit_area: str) -> None:
    ...

def _fill_limit_area_2d(mask: np.ndarray, limit_area: str) -> None:
    ...

def get_fill_func(method: str, ndim: int) -> F:
    ...

def clean_reindex_fill_method(method: str) -> str | None:
    ...

def _interp_limit(invalid: np.ndarray, fw_limit: int | None, bw_limit: int | None) -> np.ndarray:
    ...
