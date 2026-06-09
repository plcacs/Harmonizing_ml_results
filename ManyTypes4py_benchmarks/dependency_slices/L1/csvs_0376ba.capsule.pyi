from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: empty, ndarray, object_

# === Internal dependency: pandas._libs.writers ===
def write_csv_rows(data, data_index, nlevels, cols, writer): ...

# === Internal dependency: pandas._typing ===
class SequenceNotStr(Protocol[_T_co]): ...
IndexLabel = Union[Hashable, Sequence[Hashable]]
CompressionDict = dict[str, Any]
CompressionOptions = Optional[Union[Literal['infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd', 'tar'], CompressionDict]]
FloatFormatType = Union[str, Callable, 'EngFormatter']

# === Internal dependency: pandas.core.dtypes.generic ===
def create_pandas_abc_type(name, attr, comp): ...
ABCMultiIndex = cast(...)
ABCDatetimeIndex = cast(...)
ABCPeriodIndex = cast(...)
ABCIndex = cast(...)

# === Internal dependency: pandas.core.dtypes.missing ===
def notna(obj): ...

# === Internal dependency: pandas.core.indexes.api ===
from pandas.core.indexes.base import Index

# === Internal dependency: pandas.io.common ===
def get_handle(path_or_buf, mode, *, encoding=..., compression=..., memory_map=..., is_text, errors=..., storage_options=...): ...
def get_handle(path_or_buf, mode, *, encoding=..., compression=..., memory_map=..., is_text=..., errors=..., storage_options=...): ...

# === Internal dependency: pandas.util._decorators ===
from pandas._libs.properties import cache_readonly