from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)
import numpy as np
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
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.dtypes.common import NumpyEADtype

if TYPE_CHECKING:
    from pandas import Index

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    _typ: ClassVar[str] = 'npy_extension'
    __array_priority__: ClassVar[int] = 1000
    _internal_fill_value: ClassVar[np.ndarray] = np.nan

    def __init__(self, values: np.ndarray, copy: bool = False) -> None:
        ...

    @classmethod
    def _from_sequence(
        cls, scalars: Iterable[Any], dtype: Optional[NumpyEADtype] = None, copy: bool = False
    ) -> Self:
        ...

    @property
    def dtype(self) -> NumpyEADtype:
        ...

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        ...

    def __array_ufunc__(
        self,
        ufunc: Callable,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Union[Self, Tuple[Self, ...], None]:
        ...

    def astype(self, dtype: Dtype, copy: bool = True) -> ExtensionArray:
        ...

    def isna(self) -> np.ndarray:
        ...

    def _validate_scalar(self, fill_value: Any) -> Any:
        ...

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        ...

    def _pad_or_backfill(
        self,
        method: Literal['ffill', 'bfill'],
        limit: Optional[int] = None,
        limit_area: Optional[str] = None,
        copy: bool = True,
    ) -> Optional[Self]:
        ...

    def interpolate(
        self,
        method: InterpolateOptions = 'linear',
        axis: AxisInt = 0,
        index: Optional[Index] = None,
        limit: Optional[int] = None,
        limit_direction: Optional[str] = None,
        limit_area: Optional[str] = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> Self:
        ...

    def any(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[bool, np.ndarray]:
        ...

    def all(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[bool, np.ndarray]:
        ...

    def min(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def max(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def sum(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def prod(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def mean(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def median(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def std(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def var(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def sem(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def kurt(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def skew(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[Scalar, np.ndarray]:
        ...

    def to_numpy(
        self,
        dtype: Optional[np.dtype] = None,
        copy: bool = False,
        na_value: Any = lib.no_default,
    ) -> np.ndarray:
        ...

    def __invert__(self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __pos__(self) -> Self:
        ...

    def __abs__(self) -> Self:
        ...

    def _cmp_method(
        self,
        other: Any,
        op: Callable,
    ) -> Union[Self, Tuple[Self, Self], np.ndarray, Any]:
        ...

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray) -> Union[Self, Any]:
        ...