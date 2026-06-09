# === Third-party dependency: numpy ===
# Used symbols: arange, int64, mean, random, std, sum, zeros_like

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Grouper
# re-export: from pandas.core.api import NamedAgg
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat
# re-export: from pandas import errors

# === Internal dependency: pandas._libs.lib ===
no_default: Final

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.indexes.datetimes ===
def date_range(start = ..., end = ..., periods = ..., freq = ..., tz = ..., normalize: bool = ..., name: Hashable | None = ..., inclusive: IntervalClosedType = ..., *, unit: str | None = ..., **kwargs) -> DatetimeIndex: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises