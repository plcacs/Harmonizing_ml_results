```python
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
    Union,
    Sequence,
    Callable,
    TypeVar,
    Generic,
)
import numpy as np

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
    from pandas.core.dtypes.dtypes import NumpyEADtype

_T = TypeVar("_T")
_Scalar = TypeVar("_Scalar", bound=np.generic)

class NumpyExtensionArray(
    Generic[_T],
    OpsMixin,
    NDArrayBackedExtensionArray[_T],
    ObjectStringArrayMixin,
):
    _typ: str = ...
    __array_priority__: int = ...
    _internal_fill_value: Any = ...

    def __init__(self, values: np.ndarray[_T] | NumpyExtensionArray[_T], copy: bool = ...) -> None: ...
    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[_Scalar] | Any,
        *,
        dtype: Dtype | NumpyEADtype | None = ...,
        copy: bool = ...,
    ) -> Self: ...
    @property
    def dtype(self) -> NumpyEADtype: ...
    def __array__(self, dtype: npt.DTypeLike | None = ..., copy: bool | None = ...) -> np.ndarray[_T]: ...
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...
    def astype(self, dtype: Dtype, copy: bool = ...) -> Any: ...
    def isna(self) -> np.ndarray[bool]: ...
    def _validate_scalar(self, fill_value: Any) -> Any: ...
    def _values_for_factorize(self) -> tuple[np.ndarray[_T], Any]: ...
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
        index: Index | None,
        limit: int | None,
        limit_direction: Literal["forward", "backward", "both"] | None,
        limit_area: Literal["inside", "outside"] | None,
        copy: bool,
        **kwargs: Any,
    ) -> Self: ...
    def any(
        self,
        *,
        axis: AxisInt | None = ...,
        out: Any = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def all(
        self,
        *,
        axis: AxisInt | None = ...,
        out: Any = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def min(
        self,
        *,
        axis: AxisInt | None = ...,
        skipna: bool = ...,
        **kwargs: Any,
    ) -> Any: ...
    def max(
        self,
        *,
        axis: AxisInt | None = ...,
        skipna: bool = ...,
        **kwargs: Any,
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
        dtype: npt.DTypeLike | None = ...,
        out: Any = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def median(
        self,
        *,
        axis: AxisInt | None = ...,
        out: Any = ...,
        overwrite_input: bool = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def std(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: npt.DTypeLike | None = ...,
        out: Any = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def var(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: npt.DTypeLike | None = ...,
        out: Any = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def sem(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: npt.DTypeLike | None = ...,
        out: Any = ...,
        ddof: int = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def kurt(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: npt.DTypeLike | None = ...,
        out: Any = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def skew(
        self,
        *,
        axis: AxisInt | None = ...,
        dtype: npt.DTypeLike | None = ...,
        out: Any = ...,
        keepdims: bool = ...,
        skipna: bool = ...,
    ) -> Any: ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = ...,
        na_value: Any = ...,
    ) -> np.ndarray[_T]: ...
    def __invert__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def _cmp_method(self, other: Any, op: Callable[..., Any]) -> Any: ...
    def _arith_method(self, other: Any, op: Callable[..., Any]) -> Any: ...
    def _wrap_ndarray_result(self, result: np.ndarray[Any]) -> Any: ...
```