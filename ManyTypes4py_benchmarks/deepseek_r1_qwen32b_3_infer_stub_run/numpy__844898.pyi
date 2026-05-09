from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING, overload
import numpy as np
from pandas.core.dtypes import pandas_dtype
from pandas.core.dtypes.missing import IsNaOutput
from pandas.core.dtypes.generic import ABCExtensionArray
from pandas.core.dtypes.common import Dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.base import Index

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        Self,
        npt,
    )

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    _typ: str
    __array_priority__: int
    _internal_fill_value: float

    def __init__(self, values: np.ndarray, copy: bool = False) -> None:
        ...

    @classmethod
    def _from_sequence(cls, scalars: Any, dtype: Optional[Dtype] = None, copy: bool = False) -> NumpyExtensionArray:
        ...

    @property
    def dtype(self) -> Dtype:
        ...

    def __array__(self, dtype: Optional[NpDtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        ...

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any) -> Any:
        ...

    def astype(self, dtype: Union[pandas_dtype, str], copy: bool = True) -> ExtensionArray:
        ...

    def isna(self) -> IsNaOutput:
        ...

    def _validate_scalar(self, fill_value: Scalar) -> Scalar:
        ...

    def _values_for_factorize(self) -> tuple[np.ndarray, float]:
        ...

    def _pad_or_backfill(self, method: str, limit: Optional[int] = None, limit_area: Optional[str] = None, copy: bool = True) -> NumpyExtensionArray:
        ...

    def interpolate(self, method: InterpolateOptions, axis: int, index: Index, limit: Optional[int] = None, limit_direction: Optional[str] = None, limit_area: Optional[str] = None, copy: bool = True, **kwargs: Any) -> NumpyExtensionArray:
        ...

    def any(self, axis: Optional[int] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def all(self, axis: Optional[int] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def min(self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> np.generic:
        ...

    def max(self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> np.generic:
        ...

    def sum(self, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> np.generic:
        ...

    def prod(self, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> np.generic:
        ...

    def mean(self, axis: Optional[int] = None, dtype: Optional[NpDtype] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def median(self, axis: Optional[int] = None, out: Optional[np.ndarray] = None, overwrite_input: bool = False, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def std(self, axis: Optional[int] = None, dtype: Optional[NpDtype] = None, out: Optional[np.ndarray] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def var(self, axis: Optional[int] = None, dtype: Optional[NpDtype] = None, out: Optional[np.ndarray] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def sem(self, axis: Optional[int] = None, dtype: Optional[NpDtype] = None, out: Optional[np.ndarray] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def kurt(self, axis: Optional[int] = None, dtype: Optional[NpDtype] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def skew(self, axis: Optional[int] = None, dtype: Optional[NpDtype] = None, out: Optional[np.ndarray] = None, keepdims: bool = False, skipna: bool = True) -> np.generic:
        ...

    def to_numpy(self, dtype: Optional[NpDtype] = None, copy: bool = False, na_value: Scalar = lib.no_default) -> np.ndarray:
        ...

    def __invert__(self) -> NumpyExtensionArray:
        ...

    def __neg__(self) -> NumpyExtensionArray:
        ...

    def __pos__(self) -> NumpyExtensionArray:
        ...

    def __abs__(self) -> NumpyExtensionArray:
        ...

    def _cmp_method(self, other: Any, op: Callable) -> Any:
        ...

    def _arith_method(self, other: Any, op: Callable) -> Any:
        ...

    def _wrap_ndarray_result(self, result: np.ndarray) -> Union[NumpyExtensionArray, Any]:
        ...

    def _wrap_reduction_result(self, axis: int, result: np.generic) -> np.generic:
        ...