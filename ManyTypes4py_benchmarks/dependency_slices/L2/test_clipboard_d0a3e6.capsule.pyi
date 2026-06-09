from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, int64, random

# === Internal dependency: pandas ===
# re-export: from pandas._config import get_option
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import ArrowDtype
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas import io
# re-export: from pandas.io.api import read_clipboard

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal

# === Internal dependency: pandas.arrays ===
# re-export: from pandas.core.arrays import ArrowExtensionArray

# === Internal dependency: pandas.errors ===
class PyperclipException(RuntimeError): ...
class PyperclipWindowsException(PyperclipException): ...

# === Internal dependency: pandas.io.clipboard ===
def _stringifyText(text) -> str: ...
def init_qt_clipboard() -> Any: ...
class CheckedCall:
    def __init__(self, f) -> None: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises