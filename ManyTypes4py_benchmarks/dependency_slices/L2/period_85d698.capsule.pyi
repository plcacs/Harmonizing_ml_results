from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: asarray, int64, ndarray, object_, timedelta64

# === Internal dependency: pandas._libs.index ===
class PeriodEngine(Int64Engine): ...

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.dtypes import Resolution
# re-export: from pandas._libs.tslibs.nattype import NaT
# re-export: from pandas._libs.tslibs.offsets import BaseOffset
# re-export: from pandas._libs.tslibs.offsets import Tick

# === Internal dependency: pandas._libs.tslibs.dtypes ===
OFFSET_TO_PERIOD_FREQSTR: dict[str, str]

# === Internal dependency: pandas._typing ===
Self: Any
npt: Any

# === Internal dependency: pandas.core.arrays.period ===
class PeriodArray(dtl.DatelikeOps, libperiod.PeriodMixin):
    def __init__(self, values, dtype: Dtype | None = ..., copy: bool = ...) -> None: ...
def raise_on_incompatible(left, right) -> IncompatibleFrequency: ...
def period_array(data: Sequence[Period | str | None] | AnyArrayLike, freq: str | Tick | BaseOffset | None = ..., copy: bool = ...) -> PeriodArray: ...
def validate_dtype_freq(dtype, freq: BaseOffsetT) -> BaseOffsetT: ...
def validate_dtype_freq(dtype, freq: timedelta | str | None) -> BaseOffset: ...
def validate_dtype_freq(dtype, freq: BaseOffsetT | BaseOffset | timedelta | str | None) -> BaseOffsetT: ...

# === Internal dependency: pandas.core.common ===
def count_not_none(*args) -> int: ...

# === Internal dependency: pandas.core.dtypes.common ===
# re-export: from pandas.core.dtypes.inference import is_integer

# === Internal dependency: pandas.core.dtypes.dtypes ===
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    ...

# === Internal dependency: pandas.core.dtypes.generic ===
ABCSeries: cast

# === Internal dependency: pandas.core.dtypes.missing ===
def is_valid_na_for_dtype(obj, dtype: DtypeObj) -> bool: ...

# === Internal dependency: pandas.core.indexes.base ===
_index_doc_kwargs: dict[str, str]
def maybe_extract_name(name, obj, cls) -> Hashable: ...

# === Internal dependency: pandas.core.indexes.datetimelike ===
class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    def freq(self) -> BaseOffset | None: ...
    def freq(self, value) -> None: ...
    def asi8(self) -> npt.NDArray[np.int64]: ...
    def freqstr(self) -> str: ...
    def _resolution_obj(self) -> Resolution: ...
    def _formatter_func(self) -> Any: ...

# === Internal dependency: pandas.core.indexes.datetimes ===
class DatetimeIndex(DatetimeTimedeltaMixin): ...
# re-export: from pandas.core.indexes.base import Index

# === Internal dependency: pandas.core.indexes.extension ===
def inherit_names(names: list[str], delegate: type, cache: bool = ..., wrap: bool = ...) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]: ...

# === Internal dependency: pandas.util._decorators ===
def doc(*docstrings: None | str | Callable, **params: object) -> Callable[[F], F]: ...
# re-export: from pandas._libs.properties import cache_readonly