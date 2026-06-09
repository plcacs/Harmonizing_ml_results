# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, dtype, iinfo, int32, int64, ones, roll, timedelta64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.timezones import tz_compare

# === Internal dependency: pandas._testing ===
def get_finest_unit(left: str, right: str) -> str: ...
def shares_memory(left, right) -> bool: ...
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_datetime_array_equal
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_extension_array_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.arrays ===
# re-export: from pandas.core.arrays.datetimes import DatetimeArray
# re-export: from pandas.core.arrays.timedeltas import TimedeltaArray

# === Internal dependency: pandas.core.dtypes.dtypes ===
class DatetimeTZDtype(PandasExtensionDtype):
    def __init__(self, unit: str_type | DatetimeTZDtype = ..., tz = ...) -> None: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises