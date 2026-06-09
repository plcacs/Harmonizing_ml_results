# === Third-party dependency: numpy ===
# Used symbols: __version__, arange, array, lexsort, nan, random, repeat, tile, uint64, uint8

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import DatetimeIndex
from pandas.core.api import NaT
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import get_dummies

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises