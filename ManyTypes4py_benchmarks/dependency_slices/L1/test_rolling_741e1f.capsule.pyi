# === Third-party dependency: numpy ===
# Used symbols: abs, allclose, append, arange, array, empty, float64, inf, int64, isfinite, isnan, nan, ones, percentile, random, sqrt

# === Internal dependency: pandas ===
from pandas.core.api import MultiIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import period_range
from pandas.core.api import Timedelta
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.indexers ===
from pandas.core.indexers.objects import BaseIndexer

# === Internal dependency: pandas.compat ===
def is_platform_arm(): ...
def is_platform_power(): ...
def is_platform_riscv64(): ...
from pandas.compat._constants import IS64

# === Internal dependency: pandas.core.indexers.objects ===
class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(self, index_array=..., window_size=..., index=..., offset=..., **kwargs): ...

# === Internal dependency: pandas.tseries.offsets ===
from pandas._libs.tslibs.offsets import BusinessDay

# === Third-party dependency: pytest ===
# Used symbols: mark, raises