from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: nan

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import ArrowDtype
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._io import write_to_compressed
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.arrays ===
# re-export: from pandas.core.arrays import ArrowExtensionArray

# === Internal dependency: pandas.errors ===
class EmptyDataError(ValueError): ...

# === Internal dependency: pandas.io.common ===
def urlopen(*args: Any, **kwargs: Any) -> Any: ...

# === Internal dependency: pandas.io.parsers ===
# re-export: from pandas.io.parsers.readers import read_csv
# re-export: from pandas.io.parsers.readers import read_fwf

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises