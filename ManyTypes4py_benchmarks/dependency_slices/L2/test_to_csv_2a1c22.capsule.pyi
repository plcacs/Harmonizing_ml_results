from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: arange, nan

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Grouper
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.io.api import read_csv

# === Internal dependency: pandas._testing ===
def convert_rows_list_to_csv_str(rows_list: list[str]) -> str: ...
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
def is_platform_windows() -> bool: ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises