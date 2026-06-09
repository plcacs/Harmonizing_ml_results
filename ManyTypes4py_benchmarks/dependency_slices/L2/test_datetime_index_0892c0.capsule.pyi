from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, bincount, float64, int32, int64, lexsort, mean, nan, ones, ones_like, prod, r_, random, repeat, searchsorted, std

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import isna
# re-export: from pandas.core.api import notna
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat
# re-export: from pandas import tseries

# === Internal dependency: pandas._libs.lib ===
no_default: Final

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas._typing ===
DatetimeNaTType: Any

# === Internal dependency: pandas.compat ===
def is_platform_windows() -> bool: ...

# === Internal dependency: pandas.core.groupby.grouper ===
class Grouper:
    def __init__(self, key = ..., level = ..., freq = ..., sort: bool = ..., dropna: bool = ...) -> None: ...

# === Internal dependency: pandas.core.indexes.datetimes ===
def date_range(start = ..., end = ..., periods = ..., freq = ..., tz = ..., normalize: bool = ..., name: Hashable | None = ..., inclusive: IntervalClosedType = ..., *, unit: str | None = ..., **kwargs) -> DatetimeIndex: ...

# === Internal dependency: pandas.core.indexes.period ===
def period_range(start = ..., end = ..., periods: int | None = ..., freq = ..., name: Hashable | None = ...) -> PeriodIndex: ...
# re-export: from pandas._libs.tslibs import Period

# === Internal dependency: pandas.core.resample ===
def _get_timestamp_range_edges(first: Timestamp, last: Timestamp, freq: BaseOffset, unit: str, closed: Literal['right', 'left'] = ..., origin: TimeGrouperOrigin = ..., offset: Timedelta | None = ...) -> tuple[Timestamp, Timestamp]: ...
# re-export: from pandas.core.indexes.datetimes import DatetimeIndex

# === Internal dependency: pandas.tseries.offsets ===
# re-export: from pandas._libs.tslibs.offsets import DateOffset
# re-export: from pandas._libs.tslibs.offsets import Hour
# re-export: from pandas._libs.tslibs.offsets import Minute

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises, skip