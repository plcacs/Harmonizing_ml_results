# === Third-party dependency: numpy ===
# Used symbols: arange, array, bincount, float64, int32, int64, lexsort, mean, nan, ones, ones_like, prod, r_, random, repeat, searchsorted, std

# === Internal dependency: pandas ===
from pandas.core.api import isna
from pandas.core.api import notna
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import NaT
from pandas.core.api import Timedelta
from pandas.core.api import Timestamp
from pandas.core.api import to_datetime
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas import tseries

# === Internal dependency: pandas._libs.lib ===
class _NoDefault(Enum):
    no_default = Ellipsis
no_default = _NoDefault.no_default

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas._typing ===
DatetimeNaTType = Union[datetime, 'NaTType']

# === Internal dependency: pandas.compat ===
def is_platform_windows(): ...

# === Internal dependency: pandas.core.groupby.grouper ===
class Grouper:
    def __init__(self, key=..., level=..., freq=..., sort=..., dropna=...): ...

# === Internal dependency: pandas.core.indexes.datetimes ===
def date_range(start=..., end=..., periods=..., freq=..., tz=..., normalize=..., name=..., inclusive=..., *, unit=..., **kwargs): ...

# === Internal dependency: pandas.core.indexes.period ===
def period_range(start=..., end=..., periods=..., freq=..., name=...): ...
from pandas._libs.tslibs import Period

# === Internal dependency: pandas.core.resample ===
def _get_timestamp_range_edges(first, last, freq, unit, closed=..., origin=..., offset=...): ...
from pandas.core.indexes.datetimes import DatetimeIndex

# === Internal dependency: pandas.tseries.offsets ===
from pandas._libs.tslibs.offsets import DateOffset
from pandas._libs.tslibs.offsets import Hour
from pandas._libs.tslibs.offsets import Minute

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package, min_version=...): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises, skip