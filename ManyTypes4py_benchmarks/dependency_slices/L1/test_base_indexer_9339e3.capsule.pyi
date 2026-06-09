# === Third-party dependency: numpy ===
# Used symbols: arange, array, concatenate, empty, float64, int64, max, median, min, nan, std, var

# === Internal dependency: pandas ===
from pandas.core.api import MultiIndex
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.indexers ===
from pandas.core.indexers.objects import BaseIndexer
from pandas.core.indexers.objects import FixedForwardWindowIndexer

# === Internal dependency: pandas.core.indexers.objects ===
class FixedWindowIndexer(BaseIndexer): ...
class VariableOffsetWindowIndexer(BaseIndexer):
    def __init__(self, index_array=..., window_size=..., index=..., offset=..., **kwargs): ...
class ExpandingIndexer(BaseIndexer):
    ...

# === Internal dependency: pandas.tseries.offsets ===
from pandas._libs.tslibs.offsets import BusinessDay

# === Third-party dependency: pytest ===
# Used symbols: mark, raises