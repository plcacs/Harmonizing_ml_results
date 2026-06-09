# === Third-party dependency: numpy ===
# Used symbols: arange, array, concatenate, empty, float64, int64, max, median, min, nan, std, var

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.indexers ===
# re-export: from pandas.core.indexers.objects import BaseIndexer
# re-export: from pandas.core.indexers.objects import FixedForwardWindowIndexer

# === Internal dependency: pandas.core.indexers.objects ===
class FixedWindowIndexer(BaseIndexer): ...
class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(self, index_array: np.ndarray | None = ..., window_size: int = ..., index: DatetimeIndex | None = ..., offset: BaseOffset | None = ..., **kwargs) -> None: ...
class ExpandingIndexer(BaseIndexer):
    ...

# === Internal dependency: pandas.tseries.offsets ===
# re-export: from pandas._libs.tslibs.offsets import BusinessDay

# === Third-party dependency: pytest ===
# Used symbols: mark, raises