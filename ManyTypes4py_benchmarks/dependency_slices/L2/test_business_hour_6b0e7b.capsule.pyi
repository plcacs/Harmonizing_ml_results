from typing import Any

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import date_range

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.timedeltas import Timedelta
# re-export: from pandas._libs.tslibs.timestamps import Timestamp

# === Internal dependency: pandas._libs.tslibs.offsets ===
class Nano(Tick): ...
class BusinessDay(BusinessMixin): ...
class BusinessHour(BusinessMixin):
    def __init__(self, n: int = ..., normalize: bool = ..., start: str | time | Collection[str | time] = ..., end: str | time | Collection[str | time] = ..., offset: timedelta = ...) -> None: ...
BDay = BusinessDay

# === Internal dependency: pandas._testing ===
def get_finest_unit(left: str, right: str) -> str: ...
# re-export: from pandas._testing.asserters import assert_index_equal

# === Internal dependency: pandas.tests.tseries.offsets.common ===
def assert_offset_equal(offset, base, expected) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises