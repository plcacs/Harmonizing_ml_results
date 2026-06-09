from typing import Any

# === Third-party dependency: IPython.core.completer ===
def provisionalcompleter(action = ...) -> Any: ...

# === Third-party dependency: numpy ===
# Used symbols: append, arange, argsort, array, bool_, datetime64, dtype, float64, iinfo, int64, nan, ndarray, object_, ones, random, uint64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import ArrowDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import RangeIndex
# re-export: from pandas.core.api import IntervalIndex
# re-export: from pandas.core.api import TimedeltaIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import PeriodIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import timedelta_range
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Interval
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import to_timedelta
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.tseries.api import infer_freq

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_contains_all
# re-export: from pandas._testing.asserters import assert_dict_equal
# re-export: from pandas._testing.asserters import assert_extension_array_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal
ALL_REAL_NUMPY_DTYPES: Any

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat._constants import IS64
# re-export: from pandas.compat.pyarrow import HAS_PYARROW

# === Internal dependency: pandas.core.dtypes.common ===
def is_object_dtype(arr_or_dtype) -> bool: ...
def is_numeric_dtype(arr_or_dtype) -> bool: ...
def is_any_real_numeric_dtype(arr_or_dtype) -> bool: ...

# === Internal dependency: pandas.core.indexes.api ===
def _get_combined_index(indexes: list[Index], intersect: bool = ..., sort: bool = ...) -> Index: ...
# re-export: from pandas.core.indexes.base import Index
# re-export: from pandas.core.indexes.base import ensure_index
# re-export: from pandas.core.indexes.base import ensure_index_from_sequences
# re-export: from pandas.core.indexes.multi import MultiIndex

# === Internal dependency: pandas.errors ===
class InvalidIndexError(Exception): ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Third-party dependency: pyarrow ===
# Used symbols: int64, list_

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises, skip