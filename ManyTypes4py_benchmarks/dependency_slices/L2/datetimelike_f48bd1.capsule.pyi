from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: array_equal, asarray, int64, intp, ndarray, str_

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import PeriodIndex

# === Internal dependency: pandas._libs ===
# re-export: from pandas._libs.tslibs import NaT
# re-export: from pandas._libs.tslibs import Timedelta

# === Internal dependency: pandas._libs.lib ===
def maybe_indices_to_slice(indices: npt.NDArray[np.intp], max_len: int) -> slice | npt.NDArray[np.intp]: ...

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.dtypes import Resolution
# re-export: from pandas._libs.tslibs.offsets import BaseOffset
# re-export: from pandas._libs.tslibs.offsets import Tick
# re-export: from pandas._libs.tslibs.offsets import to_offset

# === Internal dependency: pandas._libs.tslibs.parsing ===
def parse_datetime_string_with_reso(date_string: str, freq: str | None = ..., dayfirst: bool | None = ..., yearfirst: bool | None = ...) -> tuple[datetime, str]: ...

# === Internal dependency: pandas._typing ===
Self: Any
npt: Any

# === Internal dependency: pandas.compat.numpy.function ===
validate_take: CompatValidator

# === Internal dependency: pandas.core.arrays ===
# re-export: from pandas.core.arrays.base import ExtensionArray
# re-export: from pandas.core.arrays.datetimes import DatetimeArray
# re-export: from pandas.core.arrays.period import PeriodArray
# re-export: from pandas.core.arrays.timedeltas import TimedeltaArray

# === Internal dependency: pandas.core.arrays.datetimelike ===
class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray): ...

# === Internal dependency: pandas.core.common ===
def asarray_tuplesafe(values: ArrayLike | list | tuple | zip, dtype: NpDtype | None = ...) -> np.ndarray: ...
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = ...) -> ArrayLike: ...

# === Internal dependency: pandas.core.dtypes.common ===
# re-export: from pandas.core.dtypes.inference import is_integer
# re-export: from pandas.core.dtypes.inference import is_list_like

# === Internal dependency: pandas.core.dtypes.concat ===
def concat_compat(to_concat: Sequence[ArrayLike], axis: AxisInt = ..., ea_compat_axis: bool = ...) -> ArrayLike: ...

# === Internal dependency: pandas.core.dtypes.dtypes ===
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype): ...
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...

# === Internal dependency: pandas.core.indexes.base ===
_index_doc_kwargs: dict[str, str]
_index_shared_docs: dict[str, str]
class Index(IndexOpsMixin, PandasObject):
    ...

# === Internal dependency: pandas.core.indexes.extension ===
class NDArrayBackedExtensionIndex(ExtensionIndex):
    ...

# === Internal dependency: pandas.core.indexes.range ===
class RangeIndex(Index): ...

# === Internal dependency: pandas.core.tools.timedeltas ===
def to_timedelta(arg: str | float | timedelta, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...) -> Timedelta: ...
def to_timedelta(arg: Series, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...) -> Series: ...
def to_timedelta(arg: list | tuple | range | ArrayLike | Index, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...) -> TimedeltaIndex: ...
def to_timedelta(arg: str | int | float | timedelta | list | tuple | range | ArrayLike | Index | Series, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...) -> Timedelta | TimedeltaIndex | Series | NaTType | Any: ...

# === Internal dependency: pandas.errors ===
class NullFrequencyError(ValueError): ...
class InvalidIndexError(Exception): ...

# === Internal dependency: pandas.util._decorators ===
def doc(*docstrings: None | str | Callable, **params: object) -> Callable[[F], F]: ...
class Appender: ...
# re-export: from pandas._libs.properties import cache_readonly