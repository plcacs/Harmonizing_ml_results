from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: add, all, arange, array, bool_, datetime64, float32, int64, maximum, nan, ndarray, subtract, sum, timedelta64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import TimedeltaIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Period
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import timedelta_range
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import DateOffset
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs.conversion ===
def localize_pydatetime(dt: datetime, tz: tzinfo | None) -> datetime: ...

# === Internal dependency: pandas._libs.tslibs.offsets ===
def shift_months(dtindex: npt.NDArray[np.int64], months: int, day_opt: str | None = ..., reso: int = ...) -> npt.NDArray[np.int64]: ...

# === Internal dependency: pandas._testing ===
def box_expected(expected, box_cls, transpose: bool = ...) -> Any: ...
def get_finest_unit(left: str, right: str) -> str: ...
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_datetime_array_equal
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.roperator ===
def radd(left, right) -> Any: ...

# === Internal dependency: pandas.tests.arithmetic.common ===
def assert_cannot_add(left, right, msg = ...) -> Any: ...
def assert_invalid_addsub_type(left, right, msg = ...) -> Any: ...
def get_upcast_box(left, right, is_cmp: bool = ...) -> Any: ...
def assert_invalid_comparison(left, right, box) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, skip