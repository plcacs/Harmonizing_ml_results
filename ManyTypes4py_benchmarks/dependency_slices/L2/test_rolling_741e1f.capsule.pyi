# === Third-party dependency: numpy ===
# Used symbols: abs, allclose, append, arange, array, empty, float64, inf, int64, isfinite, isnan, nan, ones, percentile, random, sqrt

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.indexers ===
# re-export: from pandas.core.indexers.objects import BaseIndexer

# === Internal dependency: pandas.compat ===
def is_platform_arm() -> bool: ...
def is_platform_power() -> bool: ...
def is_platform_riscv64() -> bool: ...
# re-export: from pandas.compat._constants import IS64

# === Internal dependency: pandas.core.indexers.objects ===
class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(self, index_array: np.ndarray | None = ..., window_size: int = ..., index: DatetimeIndex | None = ..., offset: BaseOffset | None = ..., **kwargs) -> None: ...

# === Internal dependency: pandas.tseries.offsets ===
# re-export: from pandas._libs.tslibs.offsets import BusinessDay

# === Third-party dependency: pytest ===
# Used symbols: mark, raises