from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: float64, int64, nan

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import DataFrame

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
def is_platform_windows(): ...

# === Internal dependency: pandas.errors ===
class ParserError(ValueError): ...
class EmptyDataError(ValueError): ...
class ParserWarning(Warning): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, skip