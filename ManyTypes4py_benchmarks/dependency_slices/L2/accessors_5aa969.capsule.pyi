from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: asarray, ndarray

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._libs.lib ===
def is_np_dtype(dtype: object, kinds: str | None = ...) -> TypeGuard[np.dtype]: ...

# === Internal dependency: pandas.core.accessor ===
class PandasDelegate:
    ...
def delegate_names(delegate, accessors: list[str], typ: str, overwrite: bool = ..., accessor_mapping: Callable[[str], str] = ..., raise_on_missing: bool = ...) -> Any: ...

# === Internal dependency: pandas.core.arrays ===
# re-export: from pandas.core.arrays.datetimes import DatetimeArray
# re-export: from pandas.core.arrays.period import PeriodArray
# re-export: from pandas.core.arrays.timedeltas import TimedeltaArray

# === Internal dependency: pandas.core.arrays.arrow.array ===
class ArrowExtensionArray(OpsMixin, ExtensionArraySupportsAnyAll, ArrowStringArrayMixin, BaseStringArrayMethods): ...

# === Internal dependency: pandas.core.base ===
class PandasObject(DirNamesMixin):
    def _constructor(self) -> type[Self]: ...
class NoNewAttributesMixin:
    ...

# === Internal dependency: pandas.core.dtypes.common ===
def is_integer_dtype(arr_or_dtype) -> bool: ...
# re-export: from pandas.core.dtypes.inference import is_list_like

# === Internal dependency: pandas.core.dtypes.dtypes ===
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype): ...
class DatetimeTZDtype(PandasExtensionDtype): ...
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...
class ArrowDtype(StorageExtensionDtype): ...

# === Internal dependency: pandas.core.dtypes.generic ===
ABCSeries: cast

# === Internal dependency: pandas.core.indexes.datetimes ===
class DatetimeIndex(DatetimeTimedeltaMixin): ...

# === Internal dependency: pandas.core.indexes.timedeltas ===
class TimedeltaIndex(DatetimeTimedeltaMixin): ...

# === Internal dependency: pandas.util._exceptions ===
def find_stack_level() -> int: ...