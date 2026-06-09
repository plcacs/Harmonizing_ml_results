# === Third-party dependency: numpy ===
# Used symbols: arange, array, float64, int64, nan, uint64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import CategoricalDtype
# re-export: from pandas.core.api import notna
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import IntervalIndex
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import timedelta_range
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Interval
# re-export: from pandas.core.api import Categorical

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.core.arrays ===
# re-export: from pandas.core.arrays.interval import IntervalArray

# === Internal dependency: pandas.core.common ===
def asarray_tuplesafe(values: ArrayLike | list | tuple | zip, dtype: NpDtype | None = ...) -> np.ndarray: ...
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = ...) -> ArrayLike: ...

# === Internal dependency: pandas.core.dtypes.common ===
def is_unsigned_integer_dtype(arr_or_dtype) -> bool: ...

# === Internal dependency: pandas.core.dtypes.dtypes ===
class IntervalDtype(PandasExtensionDtype):
    def __init__(self, subtype = ..., closed: IntervalClosedType | None = ...) -> None: ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises, skip