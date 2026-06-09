from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any
UnsupportedOperation: Any

# === Third-party dependency: numpy ===
# Used symbols: arange

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat
# re-export: from pandas.io.api import read_excel
# re-export: from pandas.io.api import read_csv
# re-export: from pandas.io.api import read_fwf
# re-export: from pandas.io.api import read_table
# re-export: from pandas.io.api import read_pickle
# re-export: from pandas.io.api import read_hdf
# re-export: from pandas.io.api import read_feather
# re-export: from pandas.io.api import read_json
# re-export: from pandas.io.api import read_stata
# re-export: from pandas.io.api import read_sas

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_categorical_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
def is_platform_windows() -> bool: ...
# re-export: from pandas.compat._constants import WASM

# === Internal dependency: pandas.io.common ===
def _expand_user(filepath_or_buffer: str) -> str: ...
def _expand_user(filepath_or_buffer: BaseBufferT) -> BaseBufferT: ...
def _expand_user(filepath_or_buffer: str | BaseBufferT) -> str | BaseBufferT: ...
def stringify_path(filepath_or_buffer: FilePath, convert_file_like: bool = ...) -> str: ...
def stringify_path(filepath_or_buffer: BaseBufferT, convert_file_like: bool = ...) -> BaseBufferT: ...
def stringify_path(filepath_or_buffer: FilePath | BaseBufferT, convert_file_like: bool = ...) -> str | BaseBufferT: ...
def is_fsspec_url(url: FilePath | BaseBuffer) -> bool: ...
def infer_compression(filepath_or_buffer: FilePath | BaseBuffer, compression: str | None) -> str | None: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: Literal[False], errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[bytes]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: Literal[True] = ..., errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[str]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions = ..., memory_map: bool = ..., is_text: bool = ..., errors: str | None = ..., storage_options: StorageOptions = ...) -> IOHandles[str] | IOHandles[bytes]: ...
def get_handle(path_or_buf: FilePath | BaseBuffer, mode: str, *, encoding: str | None = ..., compression: CompressionOptions | None = ..., memory_map: bool = ..., is_text: bool = ..., errors: str | None = ..., storage_options: StorageOptions | None = ...) -> IOHandles[str] | IOHandles[bytes]: ...
def _maybe_memory_map(handle: str | BaseBuffer, memory_map: bool) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises