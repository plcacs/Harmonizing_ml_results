# === Third-party dependency: numpy ===
# Used symbols: arange, array, concatenate, dtype, int64, nan, ndarray, repeat

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.indexers ===
# re-export: from pandas.core.indexers.objects import BaseIndexer

# === Internal dependency: pandas.core.groupby.groupby ===
def get_groupby(obj: NDFrame, by: _KeysArgType | None = ..., grouper: ops.BaseGrouper | None = ..., group_keys: bool = ...) -> GroupBy: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises