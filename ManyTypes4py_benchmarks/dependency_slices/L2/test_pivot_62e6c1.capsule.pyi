from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, dtype, empty, fill_diagonal, float64, full, int32, int64, intp, max, mean, min, nan, random, size, sum

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import DatetimeTZDtype
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import IntervalIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import PeriodIndex
# re-export: from pandas.core.api import Period
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Interval
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Grouper
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat
# re-export: from pandas.core.reshape.api import pivot

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.types ===
# re-export: from pandas.core.dtypes.dtypes import CategoricalDtype

# === Internal dependency: pandas.compat.numpy ===
np_version_gte1p25: Any

# === Internal dependency: pandas.core.reshape.pivot ===
def pivot_table(data: DataFrame, values = ..., index = ..., columns = ..., aggfunc: AggFuncType = ..., fill_value = ..., margins: bool = ..., dropna: bool = ..., margins_name: Hashable = ..., observed: bool = ..., sort: bool = ..., **kwargs) -> DataFrame: ...

# === Internal dependency: pandas.core.reshape.reshape ===
class _Unstacker:
    ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises