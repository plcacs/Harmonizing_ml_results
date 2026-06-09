from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: array, nan

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._libs.parsers ===
STR_NA_VALUES: set[str]

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_dict_equal
# re-export: from pandas._testing.asserters import assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises