from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pandas import Index
from pandas._typing import AxisInt, Dtype, FillnaOptions, InterpolateOptions, Scalar
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.strings.object_array import ObjectStringArrayMixin


class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    _typ: ClassVar[str]
    __array_priority__: ClassVar[int]
    _internal_fill_value: ClassVar[float]

    def __init__(self, values: NDArray[Any] | NumpyExtensionArray, copy: bool = False) -> None: ...
    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Dtype | None = None, copy: bool = False) -> Self: ...
    @property
    def dtype(self) -> NumpyEADtype: ...
    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> NDArray[Any]: ...
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, /, *inputs: Any, **kwargs: Any) -> Any: ...
    def astype(self, dtype: Dtype, copy: bool = True) -> NumpyExtensionArray | NDArray[Any]: ...
    def isna(self) -> NDArray[np.bool_]: ...
    def _validate_scalar(self, fill_value: Scalar | None) -> Scalar: ...
    def _values_for_factorize(self) -> tuple[NDArray[Any], float | None]: ...
    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self: ...
    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: AxisInt,
        index: Index | None,
        limit: int | None,
        limit_direction: Literal["forward", "backward", "both"],
        limit_area: Literal["inside", "outside"] | None,
        copy: bool,
        **kwargs: Any,
    ) -> Self: ...
    def any(self, *, axis: AxisInt | None = None, out: Any = None, keepdims: bool = False, skipna: bool = True) -> Any: ...
    def all(self, *, axis: AxisInt | None = None, out: Any = None, keepdims: bool = False, skipna: bool = True) -> Any: ...
    def min(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any) -> Any: ...
    def max(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any) -> Any: ...
    def sum(self, *, axis: AxisInt | None = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any: ...
    def prod(self, *, axis: AxisInt | None = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any: ...
    def mean(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...
    def median(
        self,
        *,
        axis: AxisInt | None = None,
        out: Any = None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...
    def std(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...
    def var(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...
    def sem(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...
    def kurt(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...
    def skew(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: DTypeLike | None = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...
    def to_numpy(self, dtype: DTypeLike | None = None, copy: bool = False, na_value: object = ...) -> NDArray[Any]: ...
    def __invert__(self) -> NumpyExtensionArray: ...
    def __neg__(self) -> NumpyExtensionArray: ...
    def __pos__(self) -> NumpyExtensionArray: ...
    def __abs__(self) -> NumpyExtensionArray: ...
    def _cmp_method(self, other: Any, op: Any) -> Any: ...
    def _arith_method(self, other: Any, op: Any) -> Any: ...
    def _wrap_ndarray_result(self, result: NDArray[Any]) -> NDArrayBackedExtensionArray: ...