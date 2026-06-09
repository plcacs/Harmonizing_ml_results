# === Third-party dependency: numpy ===
# Used symbols: add, all, arange, array, bool_, datetime64, float32, int64, maximum, nan, ndarray, subtract, sum, timedelta64

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import TimedeltaIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import NaT
from pandas.core.api import Period
from pandas.core.api import period_range
from pandas.core.api import Timedelta
from pandas.core.api import timedelta_range
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import DateOffset
from pandas.core.api import to_datetime
from pandas.core.api import array
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs.conversion ===
def localize_pydatetime(dt, tz): ...

# === Internal dependency: pandas._libs.tslibs.offsets ===
def shift_months(dtindex, months, day_opt=..., reso=...): ...

# === Internal dependency: pandas._testing ===
def box_expected(expected, box_cls, transpose=...): ...
def get_finest_unit(left, right): ...
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_datetime_array_equal
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.roperator ===
def radd(left, right): ...

# === Internal dependency: pandas.tests.arithmetic.common ===
def assert_cannot_add(left, right, msg=...): ...
def assert_invalid_addsub_type(left, right, msg=...): ...
def get_upcast_box(left, right, is_cmp=...): ...
def assert_invalid_comparison(left, right, box): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, skip