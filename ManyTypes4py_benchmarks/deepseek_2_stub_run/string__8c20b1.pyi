```python
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
    Sequence,
    Iterator,
    Callable,
    TypeVar,
    Union,
)
import numpy as np
from pandas._libs import missing as libmissing
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas._libs.arrays import NDArrayBacked
import pyarrow

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        DtypeObj,
        NumpySorter,
        NumpyValueArrayLike,
        Scalar,
        Self,
        npt,
        type_t,
    )
    from pandas import Series
    from pandas.arrays import BooleanArray
    from pandas.core.arrays.integer import IntegerArray, IntegerDtype
    from pandas.core.arrays.floating import FloatingArray, FloatingDtype
    from pandas.core.arrays.string_arrow import (
        ArrowStringArray,
        ArrowStringArrayNumpySemantics,
    )

_T = TypeVar("_T")

@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    storage: str
    _na_value: Any
    _metadata: tuple[str, str]
    
    def __init__(
        self,
        storage: str | None = None,
        na_value: Any = libmissing.NA
    ) -> None: ...
    
    @property
    def name(self) -> str: ...
    
    @property
    def na_value(self) -> Any: ...
    
    def __repr__(self) -> str: ...
    
    def __eq__(self, other: Any) -> bool: ...
    
    def __hash__(self) -> int: ...
    
    def __reduce__(self) -> tuple[type[StringDtype], tuple[str, Any]]: ...
    
    @property
    def type(self) -> type[str]: ...
    
    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype: ...
    
    def construct_array_type(self) -> type[ExtensionArray]: ...
    
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    
    def __from_arrow__(self, array: Any) -> ExtensionArray: ...

class BaseStringArray(ExtensionArray):
    def tolist(self) -> list[Any]: ...
    
    @classmethod
    def _from_scalars(
        cls,
        scalars: Sequence[Any],
        dtype: Dtype | None
    ) -> ExtensionArray: ...
    
    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]: ...
    
    def _str_map(
        self,
        f: Callable[..., Any],
        na_value: Any = ...,
        dtype: Dtype | None = None,
        convert: bool = True
    ) -> ExtensionArray: ...
    
    def _str_map_nan_semantics(
        self,
        f: Callable[..., Any],
        na_value: Any = ...,
        dtype: Dtype | None = None
    ) -> ExtensionArray: ...
    
    def _str_map_str_or_object(
        self,
        dtype: Dtype,
        na_value: Any,
        arr: np.ndarray,
        f: Callable[..., Any],
        mask: np.ndarray
    ) -> ExtensionArray: ...
    
    def view(self, dtype: Dtype | None = None) -> Any: ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str
    _storage: str
    _na_value: Any
    
    def __init__(self, values: ArrayLike, copy: bool = False) -> None: ...
    
    def _validate(self) -> None: ...
    
    def _validate_scalar(self, value: Any) -> Any: ...
    
    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Dtype | None = None,
        copy: bool = False
    ) -> StringArray: ...
    
    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Sequence[str],
        *,
        dtype: Dtype,
        copy: bool = False
    ) -> StringArray: ...
    
    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: Dtype) -> StringArray: ...
    
    def __arrow_array__(self, type: Any = None) -> pyarrow.Array: ...
    
    def _values_for_factorize(self) -> tuple[np.ndarray, Any]: ...
    
    def _maybe_convert_setitem_value(self, value: Any) -> Any: ...
    
    def __setitem__(self, key: Any, value: Any) -> None: ...
    
    def _putmask(self, mask: np.ndarray, value: Any) -> None: ...
    
    def _where(self, mask: np.ndarray, value: Any) -> ExtensionArray: ...
    
    def isin(self, values: ArrayLike) -> np.ndarray: ...
    
    def astype(self, dtype: Dtype, copy: bool = True) -> ExtensionArray: ...
    
    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        axis: int = 0,
        **kwargs: Any
    ) -> Any: ...
    
    def _wrap_reduction_result(self, axis: int | None, result: Any) -> Any: ...
    
    def min(
        self,
        axis: int | None = None,
        skipna: bool = True,
        **kwargs: Any
    ) -> Any: ...
    
    def max(
        self,
        axis: int | None = None,
        skipna: bool = True,
        **kwargs: Any
    ) -> Any: ...
    
    def sum(
        self,
        *,
        axis: int | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any
    ) -> Any: ...
    
    def value_counts(self, dropna: bool = True) -> Series: ...
    
    def memory_usage(self, deep: bool = False) -> int: ...
    
    def searchsorted(
        self,
        value: ArrayLike,
        side: str = "left",
        sorter: NumpySorter | None = None
    ) -> np.ndarray: ...
    
    def _cmp_method(self, other: Any, op: Callable[..., Any]) -> ExtensionArray: ...
    
    _arith_method = _cmp_method

class StringArrayNumpySemantics(StringArray):
    _storage: str
    _na_value: Any
    
    def _validate(self) -> None: ...
    
    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Dtype | None = None,
        copy: bool = False
    ) -> StringArrayNumpySemantics: ...
```