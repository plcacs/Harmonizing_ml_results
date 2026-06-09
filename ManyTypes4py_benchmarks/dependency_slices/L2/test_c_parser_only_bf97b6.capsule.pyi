from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any
TextIOWrapper: Any

# === Third-party dependency: numpy ===
# Used symbols: int32, linspace, nan, random

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.contexts import ensure_clean
ENDIAN: Any

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat._constants import WASM

# === Internal dependency: pandas.compat.numpy ===
np_version_gte1p24: Any

# === Internal dependency: pandas.errors ===
class ParserError(ValueError): ...
class ParserWarning(Warning): ...

# === Internal dependency: pandas.util._test_decorators ===
skip_if_32bit: mark

# === Third-party dependency: pytest ===
# Used symbols: mark, raises