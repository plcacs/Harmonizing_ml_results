from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any
UnsupportedOperation: Any

# === Third-party dependency: numpy ===
# Used symbols: arange

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import to_datetime
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas.io.api import read_excel
from pandas.io.api import read_csv
from pandas.io.api import read_fwf
from pandas.io.api import read_table
from pandas.io.api import read_pickle
from pandas.io.api import read_hdf
from pandas.io.api import read_feather
from pandas.io.api import read_json
from pandas.io.api import read_stata
from pandas.io.api import read_sas

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_categorical_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
def is_platform_windows(): ...
from pandas.compat._constants import WASM

# === Internal dependency: pandas.io.common ===
def _expand_user(filepath_or_buffer): ...
def stringify_path(filepath_or_buffer, convert_file_like=...): ...
def is_fsspec_url(url): ...
def infer_compression(filepath_or_buffer, compression): ...
def get_handle(path_or_buf, mode, *, encoding=..., compression=..., memory_map=..., is_text, errors=..., storage_options=...): ...
def get_handle(path_or_buf, mode, *, encoding=..., compression=..., memory_map=..., is_text=..., errors=..., storage_options=...): ...
def _maybe_memory_map(handle, memory_map): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises