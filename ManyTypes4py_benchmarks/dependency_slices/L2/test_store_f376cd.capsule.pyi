from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, datetime64, float64, int64, nan, random

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import timedelta_range
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.tseries import offsets
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._io import round_trip_pathlib
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_series_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat._constants import PY312

# === Internal dependency: pandas.io.pytables ===
def read_hdf(path_or_buf: FilePath | HDFStore, key = ..., mode: str = ..., errors: str = ..., where: str | list | None = ..., start: int | None = ..., stop: int | None = ..., columns: list[str] | None = ..., iterator: bool = ..., chunksize: int | None = ..., **kwargs) -> Any: ...
class HDFStore: ...

# === Internal dependency: pandas.tests.io.pytables.common ===
def ensure_clean_store(path, mode = ..., complevel = ..., complib = ..., fletcher32 = ...) -> Generator[HDFStore, None, None]: ...
def _maybe_remove(store, key) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises