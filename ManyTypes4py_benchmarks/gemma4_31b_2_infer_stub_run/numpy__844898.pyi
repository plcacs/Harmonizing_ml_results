from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Union, Optional, Any, overload, overload
import numpy as np
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arraylike import OpsMixin
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from pandas._typing import Dtype, Scalar, npt

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    _typ: Literal['npy_extension']
    __array_priority__: int
    _internal_fill_value: float

    def __init__(self, values: Union[np.ndarray, NumpyExtensionArray], copy: bool = False) -> None: ...

    @classmethod
    def _from_sequence(
        cls, 
        scalars: Any, 
        *, 
        dtype: Optional[Union[Dtype, np.dtype]] = None, 
        copy: bool = False
    ) -> NumpyExtensionArray: ...

    @property
    def dtype(self) -> NumpyEADtype: ...

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray: ...

    def __array_ufunc__(
        self, 
        ufunc: Any, 
        method: str, 
        *inputs: Any, 
        **kwargs: Any
    ) -> Any: ...

    def astype(self, dtype: Dtype, copy: bool = True) -> Any: ...

    def isna(self) -> np.ndarray: ...

    def _validate_scalar(self, fill_value: Any) -> Any: ...

    def _values_for_factorize(self) -> tuple[np.ndarray, Optional[float]]: ...

    def _pad_or_backfill(
        self, 
        *, 
        method: str, 
        limit: Optional[int] = None, 
        limit_area: Optional[str] = None, 
        copy: bool = True
    ) -> NumpyExtensionArray: ...

    def interpolate(
        self, 
        *, 
        method: str, 
        axis: int, 
        index: Any, 
        limit: Optional[int], 
        limit_direction: str, 
        limit_area: str, 
        copy: bool, 
        **kwargs: Any
    ) -> NumpyExtensionArray: ...

    def any(self, *, axis: Optional[int] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def all(self, *, axis: Optional[int] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def min(self, *, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any: ...

    def max(self, *, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any: ...

    def sum(self, *, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any: ...

    def prod(self, *, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any: ...

    def mean(self, *, axis: Optional[int] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def median(self, *, axis: Optional[int] = None, out: Optional[np.ndarray] = None, overwrite_input: bool = False, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def std(self, *, axis: Optional[int] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def var(self, *, axis: Optional[int] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def sem(self, *, axis: Optional[int] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def kurt(self, *, axis: Optional[int] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def skew(self, *, axis: Optional[int] = None, dtype: Optional[np.dtype] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> Any: ...

    def to_numpy(self, dtype: Optional[np.dtype] = None, copy: bool = False, na_value: Any = ...) -> np.ndarray: ...

    def __invert__(self) -> NumpyExtensionArray: ...

    def __neg__(self) -> NumpyExtensionArray: ...

    def __pos__(self) -> NumpyExtensionArray: ...

    def __abs__(self) -> NumpyExtensionArray: ...

    def _cmp_method(self, other: Any, op: Any) -> Any: ...
    _arith_method: Any # Alias for _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray) -> Any: ...