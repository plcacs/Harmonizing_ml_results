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
from pandas._libs import lib
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from typing_extensions import Self
    from pandas._typing import (
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        npt,
    )
    from pandas import Index
    from numpy.typing import NDArray
    import sys
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

_T = TypeVar("_T")
_ScalarType = TypeVar("_ScalarType", bound=np.generic)

class NumpyExtensionArray(
    OpsMixin,
    NDArrayBackedExtensionArray,
    ObjectStringArrayMixin,
    Generic[_ScalarType]
):
    _typ: str
    __array_priority__: int
    _internal_fill_value: float
    _ndarray: NDArray[_ScalarType]
    _dtype: NumpyEADtype

    def __init__(
        self,
        values: Union[NDArray[_ScalarType], NumpyExtensionArray[_ScalarType]],
        copy: bool = False
    ) -> None: ...

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Union[Dtype, NumpyEADtype, None] = None,
        copy: bool = False
    ) -> Self: ...

    @property
    def dtype(self) -> NumpyEADtype: ...

    def __array__(
        self,
        dtype: Union[NpDtype, None] = None,
        copy: Union[bool, None] = None
    ) -> NDArray[Any]: ...

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "accumulate", "outer", "inner", "at"],
        *inputs: Any,
        **kwargs: Any
    ) -> Union[
        Self,
        tuple[Self, ...],
        np.ndarray,
        tuple[np.ndarray, ...],
        np.generic,
        None
    ]: ...

    def astype(
        self,
        dtype: Dtype,
        copy: bool = True
    ) -> Union[Self, NDArrayBackedExtensionArray]: ...

    def isna(self) -> NDArray[np.bool_]: ...

    def _validate_scalar(
        self,
        fill_value: Any
    ) -> Any: ...

    def _values_for_factorize(self) -> tuple[NDArray[_ScalarType], Union[None, float]]: ...

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: Union[int, None] = None,
        limit_area: Union[Literal["inside", "outside"], None] = None,
        copy: bool = True
    ) -> Self: ...

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: AxisInt,
        index: Index,
        limit: Union[int, None],
        limit_direction: Literal["forward", "backward", "both"],
        limit_area: Union[Literal["inside", "outside"], None],
        copy: bool,
        **kwargs: Any
    ) -> Self: ...

    def any(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, np.bool_]: ...

    def all(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, np.bool_]: ...

    def min(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        skipna: bool = True,
        **kwargs: Any
    ) -> Union[Self, _ScalarType]: ...

    def max(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        skipna: bool = True,
        **kwargs: Any
    ) -> Union[Self, _ScalarType]: ...

    def sum(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any
    ) -> Union[Self, _ScalarType]: ...

    def prod(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any
    ) -> Union[Self, _ScalarType]: ...

    def mean(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        dtype: Union[NpDtype, None] = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, _ScalarType]: ...

    def median(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        out: Any = None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, _ScalarType]: ...

    def std(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        dtype: Union[NpDtype, None] = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, _ScalarType]: ...

    def var(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        dtype: Union[NpDtype, None] = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, _ScalarType]: ...

    def sem(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        dtype: Union[NpDtype, None] = None,
        out: Any = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, _ScalarType]: ...

    def kurt(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        dtype: Union[NpDtype, None] = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, _ScalarType]: ...

    def skew(
        self,
        *,
        axis: Union[AxisInt, None] = None,
        dtype: Union[NpDtype, None] = None,
        out: Any = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Union[Self, _ScalarType]: ...

    def to_numpy(
        self,
        dtype: Union[NpDtype, None] = None,
        copy: bool = False,
        na_value: Any = lib.no_default
    ) -> NDArray[Any]: ...

    def __invert__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...

    def _cmp_method(
        self,
        other: Any,
        op: Callable[..., Any]
    ) -> Union[Self, np.ndarray, tuple[Self, Self], tuple[Any, Any]]: ...

    _arith_method: Any = _cmp_method

    def _wrap_ndarray_result(
        self,
        result: NDArray[Any]
    ) -> Union[Self, Any]: ...