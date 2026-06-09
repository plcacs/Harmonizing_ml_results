# === Internal dependency: pandas ===
from pandas.core.api import DatetimeIndex
from pandas.core.api import date_range

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp

# === Internal dependency: pandas._libs.tslibs.offsets ===
class Nano(Tick): ...
class BusinessDay(BusinessMixin): ...
class BusinessHour(BusinessMixin):
    def __init__(self, n=..., normalize=..., start=..., end=..., offset=...): ...
BDay = BusinessDay

# === Internal dependency: pandas._testing ===
def get_finest_unit(left, right): ...
from pandas._testing.asserters import assert_index_equal

# === Internal dependency: pandas.tests.tseries.offsets.common ===
def assert_offset_equal(offset, base, expected): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises