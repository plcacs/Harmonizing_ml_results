# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, float64, int64, nan, random

# === Internal dependency: pandas ===
from pandas._config import option_context
from pandas.core.api import Index
from pandas.core.api import DatetimeIndex
from pandas.core.api import date_range
from pandas.core.api import bdate_range
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.io.api import read_hdf

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.timestamps import Timestamp

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_series_equal
from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
def is_platform_windows(): ...

# === Internal dependency: pandas.tests.io.pytables.common ===
def ensure_clean_store(path, mode=..., complevel=..., complib=..., fletcher32=...): ...
def _maybe_remove(store, key): ...

# === Internal dependency: pandas.util._test_decorators ===
skip_if_windows = pytest.mark.skipif(...)

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, param, raises