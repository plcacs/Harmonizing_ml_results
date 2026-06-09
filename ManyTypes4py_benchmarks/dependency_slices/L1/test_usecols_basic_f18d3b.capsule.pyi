from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: array, nan

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import array
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_frame_equal

# === Internal dependency: pandas.errors ===
class ParserError(ValueError): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, skip