from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: allclose, arange, dtype, nan, random

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import ArrowDtype
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.io.api import read_csv
# re-export: from pandas.io.api import read_html

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_frame_equal

# === Internal dependency: pandas.arrays ===
# re-export: from pandas.core.arrays import ArrowExtensionArray

# === Internal dependency: pandas.compat ===
def is_platform_windows() -> bool: ...

# === Internal dependency: pandas.io.common ===
def file_path_to_url(path: str) -> str: ...

# === Internal dependency: pandas.io.html ===
def _remove_whitespace(s: str, regex: Pattern = ...) -> str: ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Third-party dependency: pyarrow ===
# Used symbols: array

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises, skip