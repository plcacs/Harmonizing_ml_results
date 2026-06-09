from typing import Any

# === Third-party dependency: botocore ===
# Used symbols: exceptions

# === Internal dependency: io ===
BytesIO: Any

# === Third-party dependency: numpy ===
# Used symbols: zeros

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_frame_equal

# === Internal dependency: pandas.io.feather_format ===
def read_feather(path: FilePath | ReadBuffer[bytes], columns: Sequence[Hashable] | None = ..., use_threads: bool = ..., storage_options: StorageOptions | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame: ...

# === Internal dependency: pandas.io.parsers ===
# re-export: from pandas.io.parsers.readers import read_csv

# === Internal dependency: pandas.util._test_decorators ===
skip_if_not_us_locale: mark

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises, skip