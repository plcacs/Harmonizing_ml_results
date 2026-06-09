from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: allclose, arange, dtype, nan, random

# === Internal dependency: pandas ===
from pandas._config import option_context
from pandas.core.api import ArrowDtype
from pandas.core.api import StringDtype
from pandas.core.api import NA
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import to_datetime
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.io.api import read_csv
from pandas.io.api import read_html

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_frame_equal

# === Internal dependency: pandas.arrays ===
from pandas.core.arrays import ArrowExtensionArray

# === Internal dependency: pandas.compat ===
def is_platform_windows(): ...

# === Internal dependency: pandas.io.common ===
def file_path_to_url(path): ...

# === Internal dependency: pandas.io.html ===
def _remove_whitespace(s, regex=...): ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package, min_version=...): ...

# === Third-party dependency: pyarrow ===
# Used symbols: array

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises, skip