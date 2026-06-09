from typing import Any

# === Third-party dependency: IPython.core.completer ===
def provisionalcompleter(action = ...) -> Any: ...

# === Third-party dependency: numpy ===
# Used symbols: append, arange, argsort, array, bool_, datetime64, dtype, float64, iinfo, int64, nan, ndarray, object_, ones, random, uint64

# === Internal dependency: pandas ===
from pandas.core.api import ArrowDtype
from pandas.core.api import NA
from pandas.core.api import CategoricalIndex
from pandas.core.api import RangeIndex
from pandas.core.api import IntervalIndex
from pandas.core.api import TimedeltaIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import PeriodIndex
from pandas.core.api import NaT
from pandas.core.api import period_range
from pandas.core.api import Timedelta
from pandas.core.api import timedelta_range
from pandas.core.api import date_range
from pandas.core.api import Interval
from pandas.core.api import to_datetime
from pandas.core.api import to_timedelta
from pandas.core.api import array
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.tseries.api import infer_freq

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_contains_all
from pandas._testing.asserters import assert_dict_equal
from pandas._testing.asserters import assert_extension_array_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal
UNSIGNED_INT_NUMPY_DTYPES = ['uint8', 'uint16', 'uint32', 'uint64']
SIGNED_INT_NUMPY_DTYPES = [int, 'int8', 'int16', 'int32', 'int64']
ALL_INT_NUMPY_DTYPES = UNSIGNED_INT_NUMPY_DTYPES + SIGNED_INT_NUMPY_DTYPES
FLOAT_NUMPY_DTYPES = [float, 'float32', 'float64']
ALL_REAL_NUMPY_DTYPES = FLOAT_NUMPY_DTYPES + ALL_INT_NUMPY_DTYPES

# === Internal dependency: pandas.compat ===
from pandas.compat._constants import IS64
from pandas.compat.pyarrow import HAS_PYARROW

# === Internal dependency: pandas.core.dtypes.common ===
def is_object_dtype(arr_or_dtype): ...
def is_numeric_dtype(arr_or_dtype): ...
def is_any_real_numeric_dtype(arr_or_dtype): ...

# === Internal dependency: pandas.core.indexes.api ===
def _get_combined_index(indexes, intersect=..., sort=...): ...
from pandas.core.indexes.base import Index
from pandas.core.indexes.base import ensure_index
from pandas.core.indexes.base import ensure_index_from_sequences
from pandas.core.indexes.multi import MultiIndex

# === Internal dependency: pandas.errors ===
class InvalidIndexError(Exception): ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package, min_version=...): ...

# === Third-party dependency: pyarrow ===
# Used symbols: int64, list_

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, param, raises, skip