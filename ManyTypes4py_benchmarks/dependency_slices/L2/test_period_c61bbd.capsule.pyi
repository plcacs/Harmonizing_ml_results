# === Third-party dependency: numpy ===
# Used symbols: datetime64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Period
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.nattype import iNaT

# === Internal dependency: pandas._libs.tslibs.ccalendar ===
DAYS: list[str]
MONTHS: list[str]

# === Internal dependency: pandas._libs.tslibs.np_datetime ===
class OutOfBoundsDatetime(ValueError): ...

# === Internal dependency: pandas._libs.tslibs.parsing ===
class DateParseError(ValueError): ...

# === Internal dependency: pandas._libs.tslibs.period ===
INVALID_FREQ_ERR_MSG: str

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._io import round_trip_pickle
# re-export: from pandas._testing._warnings import assert_produces_warning

# === Third-party dependency: pytest ===
# Used symbols: mark, raises