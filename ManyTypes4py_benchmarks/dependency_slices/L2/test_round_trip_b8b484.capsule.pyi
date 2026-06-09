from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, float64, int64, nan, random

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import bdate_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.io.api import read_hdf

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.timestamps import Timestamp

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_series_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
def is_platform_windows() -> bool: ...

# === Internal dependency: pandas.tests.io.pytables.common ===
def ensure_clean_store(path, mode = ..., complevel = ..., complib = ..., fletcher32 = ...) -> Generator[HDFStore, None, None]: ...
def _maybe_remove(store, key) -> Any: ...

# === Internal dependency: pandas.util._test_decorators ===
skip_if_windows: mark

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, param, raises