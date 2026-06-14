from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pandas._libs import lib
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import (
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        Self,
        npt,
        ArrayLike,
        DtypeObj,
    )
    from pandas import Index


class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    _typ: str
    __array_priority__: int
    _internal_fill_value: float

    def __init__(self, values: np.ndarray | NumpyExtensionArray, copy: bool = ...) -> None: ...

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any] | np.ndarray,
        *,
        dtype: Dtype | None = ...,
        copy: bool = ...,
    ) -> NumpyExtensionArray: ...

    @property
    def dtype(self) -> NumpyEADtype: ...

    def __array__(
        self, dtype: NpDtype | None = ..., copy: bool | None = ...
    ) -> np.ndarray: ...

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any: ...

    def astype(self, dtype: Dtype, copy: bool = ...) -> ArrayLike: ...

    def isna(self) -> np.ndarray: ...

    def _validate_scalar(self, fill_value: Scalar) -> Scalar: ...

    def _values_for_factorize(self) -> tuple[np.ndarray, float | None]: ...

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        copy: bool = ...,
    ) -> Self: ...

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: AxisInt,
        index: Index,
        limit: int | None,
        limit_direction: str,
        limit_area: str | None,
        copy: bool,
        **kwargs: Any,
    ) -> Self: ...

    def any(
        self,
        *,
        axis: AxisInt | None = ...,
        out: None = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def all(
        self,
        *,
        axis: AxisInt | None = ...,
        out: None = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def min(
        self, *, axis: AxisInt | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> Any: ...

    def max(
        self, *, axis: AxisInt | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> Any: ...

    def sum(
        self,
        *,
        axis: AxisInt | None = ...,
        skipna: bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> Any: ...

    def prod(
        self,
        *,
        axis: AxisInt | None = ...,
        skipna: bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> Any: ...

    def mean(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: NpDtype | None = ...,
        out: None = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def median(
        self,
        *,
        axis: AxisInt | None = ...,
        out: None = ...,
        overwrite_input: bool = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def std(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: NpDtype | None = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def var(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: NpDtype | None = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def sem(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: NpDtype | None = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def kurt(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: NpDtype | None = ...,
        out: None = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def skew(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: NpDtype | None = ...,
        out: None = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...

    def to_numpy(
        self,
        dtype: NpDtype | None = ...,
        copy: bool = ...,
        na_value: Scalar = ...,
    ) -> np.ndarray: ...

    def __invert__(self) -> NumpyExtensionArray: ...
    def __neg__(self) -> NumpyExtensionArray: ...
    def __pos__(self) -> NumpyExtensionArray: ...
    def __abs__(self) -> NumpyExtensionArray: ...

    def _cmp_method(self, other: Any, op: Any) -> Any: ...
    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray) -> Any: ...