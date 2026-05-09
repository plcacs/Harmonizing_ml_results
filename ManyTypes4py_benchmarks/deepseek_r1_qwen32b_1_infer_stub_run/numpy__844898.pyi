from __future__ import annotations
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)
from numpy import (
    ndarray,
    dtype,
    ufunc,
    generic,
    int64,
    float64,
    bool_,
    datetime64,
    timedelta64,
)
from pandas._libs import lib
from pandas._libs.tslibs import is_supported_dtype
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.arrays import ExtensionArray
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.core.construction import ensure_wrapped_if_datetimelike

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    _typ: ClassVar[str] = 'npy_extension'
    __array_priority__: ClassVar[int] = 1000
    _internal_fill_value: ClassVar[float] = np.nan

    def __init__(self, values: np.ndarray, copy: bool = False) -> None:
        ...

    @classmethod
    def _from_sequence(
        cls, scalars: Iterable[Any], dtype: Optional[NumpyEADtype] = None, copy: bool = False
    ) -> NumpyExtensionArray:
        ...

    @property
    def dtype(self) -> NumpyEADtype:
        ...

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        ...

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        out: Tuple[Union[NumpyExtensionArray, np.ndarray], ...] = (),
        **kwargs: Any,
    ) -> Union[NumpyExtensionArray, np.ndarray, None, Tuple[NumpyExtensionArray, ...]]:
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
        method: str,
        limit: Optional[int] = None,
        limit_area: Optional[str] = None,
        copy: bool = True,
    ) -> Optional[NumpyExtensionArray]:
        ...

    def interpolate(
        self,
        method: str,
        axis: int,
        index: Any,
        limit: Optional[int] = None,
        limit_direction: Optional[str] = None,
        limit_area: Optional[str] = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> NumpyExtensionArray:
        ...

    def any(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[bool, np.ndarray, NumpyExtensionArray]:
        ...

    def all(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[bool, np.ndarray, NumpyExtensionArray]:
        ...

    def min(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> Union[int, float, np.generic, NumpyExtensionArray]:
        ...

    def max(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> Union[int, float, np.generic, NumpyExtensionArray]:
        ...

    def sum(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Union[int, float, np.generic, NumpyExtensionArray]:
        ...

    def prod(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Union[int, float, np.generic, NumpyExtensionArray]:
        ...

    def mean(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[float, np.generic, NumpyExtensionArray]:
        ...

    def median(
        self,
        axis: Optional[int] = None,
        out: Optional[np.ndarray] = None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[float, np.generic, NumpyExtensionArray]:
        ...

    def std(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[float, np.generic, NumpyExtensionArray]:
        ...

    def var(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[float, np.generic, NumpyExtensionArray]:
        ...

    def sem(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[float, np.generic, NumpyExtensionArray]:
        ...

    def kurt(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[float, np.generic, NumpyExtensionArray]:
        ...

    def skew(
        self,
        axis: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        out: Optional[np.ndarray] = None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Union[float, np.generic, NumpyExtensionArray]:
        ...

    def to_numpy(
        self,
        dtype: Optional[np.dtype] = None,
        copy: bool = False,
        na_value: Any = lib.no_default,
    ) -> np.ndarray:
        ...

    def __invert__(self) -> NumpyExtensionArray:
        ...

    def __neg__(self) -> NumpyExtensionArray:
        ...

    def __pos__(self) -> NumpyExtensionArray:
        ...

    def __abs__(self) -> NumpyExtensionArray:
        ...

    def _cmp_method(
        self, other: Any, op: Callable
    ) -> Union[NumpyExtensionArray, np.ndarray, Any]:
        ...

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray) -> Union[NumpyExtensionArray, Any]:
        ...