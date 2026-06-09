from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: array, nan

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import DataFrame

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._libs.parsers ===
STR_NA_VALUES = ...

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_dict_equal
from pandas._testing.asserters import assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises