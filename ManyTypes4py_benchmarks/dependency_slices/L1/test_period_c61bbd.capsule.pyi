# === Third-party dependency: numpy ===
# Used symbols: datetime64

# === Internal dependency: pandas ===
from pandas.core.api import NaT
from pandas.core.api import Period
from pandas.core.api import Timedelta
from pandas.core.api import Timestamp
from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.nattype import iNaT

# === Internal dependency: pandas._libs.tslibs.ccalendar ===
DAYS = ...
MONTHS = ...

# === Internal dependency: pandas._libs.tslibs.np_datetime ===
class OutOfBoundsDatetime(ValueError): ...

# === Internal dependency: pandas._libs.tslibs.parsing ===
class DateParseError(ValueError): ...

# === Internal dependency: pandas._libs.tslibs.period ===
INVALID_FREQ_ERR_MSG = ...

# === Internal dependency: pandas._testing ===
from pandas._testing._io import round_trip_pickle
from pandas._testing._warnings import assert_produces_warning

# === Third-party dependency: pytest ===
# Used symbols: mark, raises