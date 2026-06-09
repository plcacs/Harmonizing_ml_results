# === Third-party dependency: numpy ===
# Used symbols: arange, array, concatenate, dtype, int64, nan, ndarray, repeat

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import to_datetime
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.indexers ===
from pandas.core.indexers.objects import BaseIndexer

# === Internal dependency: pandas.core.groupby.groupby ===
def get_groupby(obj, by=..., grouper=..., group_keys=...): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises