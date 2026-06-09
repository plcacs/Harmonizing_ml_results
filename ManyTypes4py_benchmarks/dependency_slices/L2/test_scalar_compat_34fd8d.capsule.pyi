# === Third-party dependency: hypothesis ===
# Used symbols: given

# === Third-party dependency: hypothesis.strategies ===
# Used symbols: datetimes, integers, sampled_from

# === Third-party dependency: numpy ===
# Used symbols: arange, array, int32, isnan, ndarray

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs.timezones ===
def maybe_get_tz(tz: str | int | np.int64 | tzinfo | None) -> tzinfo | None: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._config.localization import get_locales
# re-export: from pandas._config.localization import set_locale
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.core.arrays ===
# re-export: from pandas.core.arrays.datetimes import DatetimeArray

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Extension dependency: unicodedata ===
# Used symbols: normalize