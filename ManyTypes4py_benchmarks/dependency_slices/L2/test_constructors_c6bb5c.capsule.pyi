from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: all, arange, array, asarray, bool_, broadcast_to, complex128, complex64, concatenate, datetime64, dtype, empty, float64, iinfo, int64, isnan, issubdtype, ma, nan, ndarray, object_, ones, r_, random, shares_memory, str_, timedelta64, uint16, uint32, uint64, zeros

# === Third-party dependency: numpy.dtypes ===
# Used symbols: StringDType

# === Third-party dependency: numpy.ma ===
# Used symbols: mrecords

# === Internal dependency: pandas ===
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import Int64Dtype
# re-export: from pandas.core.api import Float64Dtype
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import BooleanDtype
# re-export: from pandas.core.api import isna
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import RangeIndex
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import IntervalIndex
# re-export: from pandas.core.api import TimedeltaIndex
# re-export: from pandas.core.api import DatetimeIndex
# re-export: from pandas.core.api import PeriodIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Period
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Interval
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import cut
# re-export: from pandas import arrays

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._libs.lib ===
def is_np_dtype(dtype: object, kinds: str | None = ...) -> TypeGuard[np.dtype]: ...

# === Internal dependency: pandas._testing ===
DATETIME64_DTYPES: list[Dtype]
TIMEDELTA64_DTYPES: list[Dtype]
BOOL_DTYPES: list[Dtype]
BYTES_DTYPES: list[Dtype]
OBJECT_DTYPES: list[Dtype]
ALL_NUMERIC_DTYPES: list[Dtype]
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal
# re-export: from pandas._testing.compat import get_dtype

# === Internal dependency: pandas.arrays ===
# re-export: from pandas.core.arrays import DatetimeArray
# re-export: from pandas.core.arrays import IntervalArray
# re-export: from pandas.core.arrays import PeriodArray
# re-export: from pandas.core.arrays import SparseArray
# re-export: from pandas.core.arrays import TimedeltaArray

# === Internal dependency: pandas.compat.numpy ===
np_version_gt2: Any

# === Internal dependency: pandas.core.dtypes.common ===
def is_integer_dtype(arr_or_dtype) -> bool: ...

# === Internal dependency: pandas.core.dtypes.dtypes ===
class DatetimeTZDtype(PandasExtensionDtype):
    def __init__(self, unit: str_type | DatetimeTZDtype = ..., tz = ...) -> None: ...
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...
class IntervalDtype(PandasExtensionDtype): ...
class NumpyEADtype(ExtensionDtype): ...

# === Internal dependency: pandas.errors ===
class IntCastingNaNError(ValueError): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises, skip