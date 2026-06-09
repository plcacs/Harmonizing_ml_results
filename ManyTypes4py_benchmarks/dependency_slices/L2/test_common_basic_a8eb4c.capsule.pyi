from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: float64, int64, nan

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
def is_platform_windows() -> bool: ...

# === Internal dependency: pandas.errors ===
class ParserError(ValueError): ...
class EmptyDataError(ValueError): ...
class ParserWarning(Warning): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, skip