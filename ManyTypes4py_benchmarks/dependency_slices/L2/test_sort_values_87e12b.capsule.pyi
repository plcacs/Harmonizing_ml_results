# === Third-party dependency: numpy ===
# Used symbols: __version__, arange, array, lexsort, nan, random, repeat, tile, uint64, uint8

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import get_dummies

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises