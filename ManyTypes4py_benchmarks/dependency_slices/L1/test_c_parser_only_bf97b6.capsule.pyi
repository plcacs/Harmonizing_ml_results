from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any
TextIOWrapper: Any

# === Third-party dependency: numpy ===
# Used symbols: int32, linspace, nan, random

# === Internal dependency: pandas ===
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.contexts import ensure_clean
ENDIAN = {'little': '<', 'big': '>'}[byteorder]

# === Internal dependency: pandas.compat ===
from pandas.compat._constants import WASM

# === Internal dependency: pandas.compat.numpy ===
_np_version = np.__version__
_nlv = Version(...)
np_version_gte1p24 = _nlv >= Version('1.24')

# === Internal dependency: pandas.errors ===
class ParserError(ValueError): ...
class ParserWarning(Warning): ...

# === Internal dependency: pandas.util._test_decorators ===
skip_if_32bit = pytest.mark.skipif(...)

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises