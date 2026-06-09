# === Third-party dependency: numpy ===
# Used symbols: arange, array, float64, int64, nan, uint64

# === Internal dependency: pandas ===
from pandas.core.api import CategoricalDtype
from pandas.core.api import notna
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import IntervalIndex
from pandas.core.api import period_range
from pandas.core.api import timedelta_range
from pandas.core.api import date_range
from pandas.core.api import Interval
from pandas.core.api import Categorical

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.core.arrays ===
from pandas.core.arrays.interval import IntervalArray

# === Internal dependency: pandas.core.common ===
def asarray_tuplesafe(values, dtype=...): ...

# === Internal dependency: pandas.core.dtypes.common ===
def is_unsigned_integer_dtype(arr_or_dtype): ...

# === Internal dependency: pandas.core.dtypes.dtypes ===
class IntervalDtype(PandasExtensionDtype):
    def __init__(self, subtype=..., closed=...): ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package, min_version=...): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises, skip