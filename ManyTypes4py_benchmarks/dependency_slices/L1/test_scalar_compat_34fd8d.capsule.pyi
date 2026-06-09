# === Third-party dependency: hypothesis ===
# Used symbols: given

# === Third-party dependency: hypothesis.strategies ===
# Used symbols: datetimes, integers, sampled_from

# === Third-party dependency: numpy ===
# Used symbols: arange, array, int32, isnan, ndarray

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import DatetimeIndex
from pandas.core.api import NaT
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs.timezones ===
def maybe_get_tz(tz): ...

# === Internal dependency: pandas._testing ===
from pandas._config.localization import get_locales
from pandas._config.localization import set_locale
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.core.arrays ===
from pandas.core.arrays.datetimes import DatetimeArray

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Extension dependency: unicodedata ===
# Used symbols: normalize