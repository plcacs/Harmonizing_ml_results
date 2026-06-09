from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: array, datetime64, int64, timedelta64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.nattype import NaT
# re-export: from pandas._libs.tslibs.timedeltas import Timedelta
# re-export: from pandas._libs.tslibs.timestamps import Timestamp

# === Internal dependency: pandas._libs.tslibs.conversion ===
def localize_pydatetime(dt: datetime, tz: tzinfo | None) -> datetime: ...

# === Internal dependency: pandas._libs.tslibs.offsets ===
_relativedelta_kwds: set[str]
def _get_offset(name: str) -> BaseOffset: ...
def to_offset(freq: None, is_period: bool = ...) -> None: ...
def to_offset(freq: _BaseOffsetT, is_period: bool = ...) -> _BaseOffsetT: ...
def to_offset(freq: timedelta | str, is_period: bool = ...) -> BaseOffset: ...
class MonthOffset(SingleConstructorOffset): ...
_offset_map: dict[str, BaseOffset]

# === Internal dependency: pandas._libs.tslibs.period ===
INVALID_FREQ_ERR_MSG: str

# === Internal dependency: pandas._libs.tslibs.timezones ===
def maybe_get_tz(tz: str | int | np.int64 | tzinfo | None) -> tzinfo | None: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._io import round_trip_pickle
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.tests.tseries.offsets.common ===
class WeekDay: ...

# === Internal dependency: pandas.tseries.offsets ===
# re-export: from pandas._libs.tslibs.offsets import FY5253
# re-export: from pandas._libs.tslibs.offsets import BDay
# re-export: from pandas._libs.tslibs.offsets import BMonthEnd
# re-export: from pandas._libs.tslibs.offsets import BusinessHour
# re-export: from pandas._libs.tslibs.offsets import CustomBusinessDay
# re-export: from pandas._libs.tslibs.offsets import CustomBusinessHour
# re-export: from pandas._libs.tslibs.offsets import CustomBusinessMonthBegin
# re-export: from pandas._libs.tslibs.offsets import CustomBusinessMonthEnd
# re-export: from pandas._libs.tslibs.offsets import DateOffset
# re-export: from pandas._libs.tslibs.offsets import Easter
# re-export: from pandas._libs.tslibs.offsets import FY5253Quarter
# re-export: from pandas._libs.tslibs.offsets import LastWeekOfMonth
# re-export: from pandas._libs.tslibs.offsets import MonthBegin
# re-export: from pandas._libs.tslibs.offsets import Nano
# re-export: from pandas._libs.tslibs.offsets import Tick
# re-export: from pandas._libs.tslibs.offsets import Week
# re-export: from pandas._libs.tslibs.offsets import WeekOfMonth
__all__: Any

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises