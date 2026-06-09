# === Third-party dependency: numpy ===
# Used symbols: array, datetime64, int64, timedelta64

# === Internal dependency: pandas ===
from pandas.core.api import DatetimeIndex
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.nattype import NaT
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp

# === Internal dependency: pandas._libs.tslibs.conversion ===
def localize_pydatetime(dt, tz): ...

# === Internal dependency: pandas._libs.tslibs.offsets ===
def _get_offset(name): ...
def to_offset(freq, is_period=...): ...
class MonthOffset(SingleConstructorOffset): ...
_relativedelta_kwds = ...
_offset_map = ...

# === Internal dependency: pandas._libs.tslibs.period ===
INVALID_FREQ_ERR_MSG = ...

# === Internal dependency: pandas._libs.tslibs.timezones ===
def maybe_get_tz(tz): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._io import round_trip_pickle
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.tests.tseries.offsets.common ===
class WeekDay: ...

# === Internal dependency: pandas.tseries.offsets ===
from pandas._libs.tslibs.offsets import FY5253
from pandas._libs.tslibs.offsets import BDay
from pandas._libs.tslibs.offsets import BMonthEnd
from pandas._libs.tslibs.offsets import BusinessHour
from pandas._libs.tslibs.offsets import CustomBusinessDay
from pandas._libs.tslibs.offsets import CustomBusinessHour
from pandas._libs.tslibs.offsets import CustomBusinessMonthBegin
from pandas._libs.tslibs.offsets import CustomBusinessMonthEnd
from pandas._libs.tslibs.offsets import DateOffset
from pandas._libs.tslibs.offsets import Easter
from pandas._libs.tslibs.offsets import FY5253Quarter
from pandas._libs.tslibs.offsets import LastWeekOfMonth
from pandas._libs.tslibs.offsets import MonthBegin
from pandas._libs.tslibs.offsets import Nano
from pandas._libs.tslibs.offsets import Tick
from pandas._libs.tslibs.offsets import Week
from pandas._libs.tslibs.offsets import WeekOfMonth
__all__ = ['Day', 'BaseOffset', 'BusinessDay', 'BusinessMonthBegin', 'BusinessMonthEnd', 'BDay', 'CustomBusinessDay', 'CustomBusinessMonthBegin', ...]

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises