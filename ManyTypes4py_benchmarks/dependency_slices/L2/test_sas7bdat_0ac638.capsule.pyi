from typing import Any

# === Internal dependency: io ===
BytesIO: Any

# === Third-party dependency: numpy ===
# Used symbols: array, datetime64, float64, int64, timedelta64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import to_timedelta
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.io.api import read_csv
# re-export: from pandas.io.api import read_sas

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.compat._constants ===
IS64: Any
WASM: Any

# === Internal dependency: pandas.errors ===
class EmptyDataError(ValueError): ...

# === Internal dependency: pandas.io.sas.sas7bdat ===
class SAS7BDATReader(SASReader): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises