from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: empty, ndarray, object_

# === Internal dependency: pandas._libs.writers ===
def write_csv_rows(data: list[ArrayLike], data_index: np.ndarray, nlevels: int, cols: np.ndarray, writer: object) -> None: ...

# === Internal dependency: pandas._typing ===
class SequenceNotStr(Protocol[_T_co]): ...
IndexLabel: Any
CompressionOptions: Any
FloatFormatType: Any

# === Internal dependency: pandas.core.dtypes.generic ===
ABCMultiIndex: cast
ABCDatetimeIndex: cast
ABCPeriodIndex: cast
ABCIndex: cast

# === Internal dependency: pandas.core.dtypes.missing ===
def notna(obj: Scalar | Pattern | NAType | NaTType) -> bool: ...
def notna(obj: ArrayLike | Index | list) -> npt.NDArray[np.bool_]: ...
def notna(obj: NDFrameT) -> NDFrameT: ...
def notna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]: ...
def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...

# === Internal dependency: pandas.core.indexes.api ===
# re-export: from pandas.core.indexes.base import Index

# === Internal dependency: pandas.io.common ===
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: Literal[False], errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[bytes]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: Literal[True] = ..., errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[str]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: bool = ..., errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[str] | IOHandles[bytes]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions | None = ..., memory_map: bool = ..., is_text: bool = ..., errors: str | None = ..., storage_options: StorageOptions | None = ...) -> IOHandles[str] | IOHandles[bytes]: ...

# === Internal dependency: pandas.util._decorators ===
# re-export: from pandas._libs.properties import cache_readonly