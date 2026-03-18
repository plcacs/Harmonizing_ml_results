```pyi
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas._libs.arrays import NDArrayBacked
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray

if TYPE_CHECKING:
    import pyarrow
    from pandas._typing import ArrayLike, AxisInt, Dtype, DtypeObj, NumpySorter, NumpyValueArrayLike, Scalar, Self, npt, type_t
    from pandas import Series

class StringDtype(StorageExtensionDtype):
    storage: str
    _na_value: Any
    _metadata: tuple[str, str]
    
    @property
    def name(self) -> str: ...
    @property
    def na_value(self) -> Any: ...
    
    def __init__(self, storage: str | None = ..., na_value: Any = ...) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __reduce__(self) -> tuple[type[StringDtype], tuple[str, Any]]: ...
    
    @property
    def type(self) -> type[str]: ...
    
    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype: ...
    def construct_array_type(self) -> type[StringArray] | type[Any]: ...
    def _get_common_dtype(self, dtypes: list[Any]) -> StringDtype | None: ...
    def __from_arrow__(self, array: Any) -> Any: ...

class BaseStringArray(ExtensionArray):
    def tolist(self) -> list[Any]: ...
    
    @classmethod
    def _from_scalars(cls, scalars: Any, dtype: Any) -> Any: ...
    
    def _formatter(self, boxed: bool = ...) -> Any: ...
    def _str_map(
        self,
        f: Any,
        na_value: Any = ...,
        dtype: Any | None = ...,
        convert: bool = ...,
    ) -> Any: ...
    def _str_map_str_or_object(
        self,
        dtype: Any,
        na_value: Any,
        arr: np.ndarray[Any, Any],
        f: Any,
        mask: np.ndarray[Any, Any],
    ) -> Any: ...
    def _str_map_nan_semantics(
        self,
        f: Any,
        na_value: Any = ...,
        dtype: Any | None = ...,
    ) -> Any: ...
    def view(self, dtype: Any | None = ...) -> Any: ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str
    _storage: str
    _na_value: Any
    
    def __init__(self, values: Any, copy: bool = ...) -> None: ...
    
    def _validate(self) -> None: ...
    def _validate_scalar(self, value: Any) -> Any: ...
    
    @classmethod
    def _from_sequence(
        cls,
        scalars: Any,
        *,
        dtype: Any | None = ...,
        copy: bool = ...,
    ) -> StringArray: ...
    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Any,
        *,
        dtype: Any,
        copy: bool = ...,
    ) -> StringArray: ...
    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: Any) -> StringArray: ...
    
    def __arrow_array__(self, type: Any | None = ...) -> Any: ...
    
    def _values_for_factorize(self) -> tuple[np.ndarray[Any, Any], Any]: ...
    def _maybe_convert_setitem_value(self, value: Any) -> Any: ...
    
    def __setitem__(self, key: Any, value: Any) -> None: ...
    
    def _putmask(self, mask: np.ndarray[Any, Any], value: Any) -> None: ...
    def _where(self, mask: np.ndarray[Any, Any], value: Any) -> Any: ...
    
    def isin(self, values: Any) -> np.ndarray[Any, np.dtype[np.bool_]]: ...
    
    def astype(self, dtype: Any, copy: bool = ...) -> Any: ...
    
    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = ...,
        keepdims: bool = ...,
        axis: int = ...,
        **kwargs: Any,
    ) -> Any: ...
    def _wrap_reduction_result(self, axis: Any, result: Any) -> Any: ...
    
    def min(self, axis: int | None = ..., skipna: bool = ..., **kwargs: Any) -> Any: ...
    def max(self, axis: int | None = ..., skipna: bool = ..., **kwargs: Any) -> Any: ...
    def sum(
        self,
        *,
        axis: int | None = ...,
        skipna: bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> Any: ...
    
    def value_counts(self, dropna: bool = ...) -> Any: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    
    def searchsorted(
        self,
        value: Any,
        side: str = ...,
        sorter: Any | None = ...,
    ) -> np.ndarray[Any, Any]: ...
    
    def _cmp_method(self, other: Any, op: Any) -> Any: ...
    def _arith_method(self, other: Any, op: Any) -> Any: ...

class StringArrayNumpySemantics(StringArray):
    _storage: str
    _na_value: Any
    
    def _validate(self) -> None: ...
    
    @classmethod
    def _from_sequence(
        cls,
        scalars: Any,
        *,
        dtype: Any | None = ...,
        copy: bool = ...,
    ) -> StringArrayNumpySemantics: ...
```