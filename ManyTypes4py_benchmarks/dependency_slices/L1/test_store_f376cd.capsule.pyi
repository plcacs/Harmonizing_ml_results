# === Third-party dependency: numpy ===
# Used symbols: arange, array, datetime64, float64, int64, nan, random

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import period_range
from pandas.core.api import timedelta_range
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.tseries import offsets
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._io import round_trip_pathlib
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_series_equal
from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.compat ===
from pandas.compat._constants import PY312

# === Internal dependency: pandas.io.pytables ===
def read_hdf(path_or_buf, key=..., mode=..., errors=..., where=..., start=..., stop=..., columns=..., iterator=..., chunksize=..., **kwargs): ...
class HDFStore: ...

# === Internal dependency: pandas.tests.io.pytables.common ===
def ensure_clean_store(path, mode=..., complevel=..., complib=..., fletcher32=...): ...
def _maybe_remove(store, key): ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises