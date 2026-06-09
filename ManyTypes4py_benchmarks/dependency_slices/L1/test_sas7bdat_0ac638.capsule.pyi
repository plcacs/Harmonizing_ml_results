from typing import Any

# === Internal dependency: io ===
BytesIO: Any

# === Third-party dependency: numpy ===
# Used symbols: array, datetime64, float64, int64, timedelta64

# === Internal dependency: pandas ===
from pandas.core.api import to_timedelta
from pandas.core.api import DataFrame
from pandas.io.api import read_csv
from pandas.io.api import read_sas

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.compat._constants ===
IS64 = sys.maxsize > 2 ** 32
WASM = sys.platform == 'emscripten' or platform.machine() in ['wasm32', 'wasm64']

# === Internal dependency: pandas.errors ===
class EmptyDataError(ValueError): ...

# === Internal dependency: pandas.io.sas.sas7bdat ===
class SAS7BDATReader(SASReader): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises