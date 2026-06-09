from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, finfo, float64, hstack, iinfo, int64, max, mean, min, nan, nanmean, ones, percentile, quantile, random, std, sum, uint64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import RangeIndex
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import interval_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import NamedAgg
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
FLOAT_NUMPY_DTYPES: list[NpDtype]
FLOAT_EA_DTYPES: list[Dtype]
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal
ALL_INT_NUMPY_DTYPES: Any
ALL_INT_EA_DTYPES: Any

# === Internal dependency: pandas.core.dtypes.common ===
def is_integer_dtype(arr_or_dtype) -> bool: ...

# === Internal dependency: pandas.core.groupby.grouper ===
class Grouping:
    def __init__(self, index: Index, grouper = ..., obj: NDFrame | None = ..., level = ..., sort: bool = ..., observed: bool = ..., in_axis: bool = ..., dropna: bool = ..., uniques: ArrayLike | None = ...) -> None: ...

# === Internal dependency: pandas.errors ===
class SpecificationError(Exception): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, param, raises