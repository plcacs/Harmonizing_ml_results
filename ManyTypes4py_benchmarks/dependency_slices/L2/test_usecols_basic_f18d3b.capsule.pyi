from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: array, nan

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_frame_equal

# === Internal dependency: pandas.errors ===
class ParserError(ValueError): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, skip