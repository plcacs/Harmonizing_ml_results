from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from pandas._libs import lib
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.core.dtypes.dtypes import NumpyEADtype

if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        Self,
        npt,
    )
    from pandas import Index
    from collections.abc import Sequence
    import operator

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    _typ: str
    __array_priority__: int
    _internal_fill_value: float

    def __init__(self, values: np.ndarray | NumpyExtensionArray, copy: bool = False) -> None: ...

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[Any] | np.ndarray, *, dtype: Dtype | None = None, copy: bool = False
    ) -> NumpyExtensionArray: ...

    @property
    def dtype(self) -> NumpyEADtype: ...

    def __array__(self, dtype: NpDtype | None = None, copy: bool | None = None) -> np.ndarray: ...

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> NumpyExtensionArray | tuple[NumpyExtensionArray, ...] | Any: ...

    def astype(self, dtype: Dtype, copy: bool = True) -> NumpyExtensionArray | np.ndarray | Any: ...

    def isna(self) -> np.ndarray: ...

    def _validate_scalar(self, fill_value: Scalar) -> Scalar: ...

    def _values_for_factorize(self) -> tuple[np.ndarray, float | None]: ...

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
        axis: int,
        index: Index,
        limit: int | None,
        limit_direction: str,
        limit_area: Literal["inside", "outside"] | None,
        copy: bool,
        **kwargs: Any,
    ) -> Self: ...

    def any(
        self,
        *,
        axis: AxisInt | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def all(
        self,
        *,
        axis: AxisInt | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any) -> Any: ...

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs: Any) -> Any: ...

    def sum(
        self, *, axis: AxisInt | None = None, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> Any: ...

    def prod(
        self, *, axis: AxisInt | None = None, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> Any: ...

    def mean(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def median(
        self,
        *,
        axis: AxisInt | None = None,
        out: np.ndarray | None = None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def std(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out: np.ndarray | None = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def var(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out: np.ndarray | None = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def sem(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out: np.ndarray | None = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def kurt(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def skew(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Any: ...

    def to_numpy(
        self,
        dtype: NpDtype | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
    ) -> np.ndarray: ...

    def __invert__(self) -> NumpyExtensionArray: ...
    def __neg__(self) -> NumpyExtensionArray: ...
    def __pos__(self) -> NumpyExtensionArray: ...
    def __abs__(self) -> NumpyExtensionArray: ...

    def _cmp_method(self, other: Any, op: Any) -> Any: ...
    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray) -> Any: ...