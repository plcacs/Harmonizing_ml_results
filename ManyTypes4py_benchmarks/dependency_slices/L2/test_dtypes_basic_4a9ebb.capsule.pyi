from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: array, float32, float64, int32, int64, nan, object_, random, str_, uint32

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import ArrowDtype
# re-export: from pandas.core.api import Int64Dtype
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal
# re-export: from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.core.arrays ===
# re-export: from pandas.core.arrays.integer import IntegerArray

# === Internal dependency: pandas.errors ===
class ParserWarning(Warning): ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises