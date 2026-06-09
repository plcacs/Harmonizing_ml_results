from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: array, bool_, dtype, empty, floating, int32, int64, issubdtype, nan, vstack, zeros

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Series

# === Internal dependency: pandas._libs.lib ===
def ensure_string_array(arr, na_value: object = ..., convert_na_value: bool = ..., copy: bool = ..., skipna: bool = ...) -> npt.NDArray[np.object_]: ...

# === Internal dependency: pandas._libs.missing ===
class NAType: ...
NA: NAType

# === Internal dependency: pandas._typing ===
NpDtype: Any

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat.pyarrow import pa_version_under10p1
# re-export: from pandas.compat.pyarrow import pa_version_under13p0

# === Internal dependency: pandas.core.arrays._arrow_string_mixins ===
class ArrowStringArrayMixin: ...

# === Internal dependency: pandas.core.arrays.arrow ===
# re-export: from pandas.core.arrays.arrow.array import ArrowExtensionArray

# === Internal dependency: pandas.core.arrays.boolean ===
class BooleanDtype(BaseMaskedDtype): ...

# === Internal dependency: pandas.core.arrays.floating ===
class Float64Dtype(FloatingDtype): ...

# === Internal dependency: pandas.core.arrays.integer ===
class Int64Dtype(IntegerDtype): ...

# === Internal dependency: pandas.core.arrays.masked ===
class BaseMaskedArray(OpsMixin, ExtensionArray): ...

# === Internal dependency: pandas.core.arrays.numeric ===
class NumericDtype(BaseMaskedDtype): ...

# === Internal dependency: pandas.core.arrays.string_ ===
class StringDtype(StorageExtensionDtype):
    def __init__(self, storage: str | None = ..., na_value: libmissing.NAType | float = ...) -> None: ...
class BaseStringArray(ExtensionArray):
    ...

# === Internal dependency: pandas.core.dtypes.common ===
def pandas_dtype(dtype) -> DtypeObj: ...
# re-export: from pandas.core.dtypes.inference import is_scalar

# === Internal dependency: pandas.core.dtypes.missing ===
def isna(obj: Scalar | Pattern | NAType | NaTType) -> bool: ...
def isna(obj: ArrayLike | Index | list) -> npt.NDArray[np.bool_]: ...
def isna(obj: NDFrameT) -> NDFrameT: ...
def isna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]: ...
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...

# === Internal dependency: pandas.core.strings.object_array ===
class ObjectStringArrayMixin(BaseStringArrayMethods):
    ...

# === Internal dependency: pandas.util._exceptions ===
def find_stack_level() -> int: ...

# === Third-party dependency: pyarrow ===
# Used symbols: Array, ChunkedArray, Scalar, array, from_numpy_dtype, large_string, null, scalar, string, types

# === Third-party dependency: pyarrow.compute ===
def cast(arr, target_type = ..., safe = ..., options = ..., memory_pool = ...) -> Any: ...