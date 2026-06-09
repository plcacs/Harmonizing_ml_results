# === Third-party dependency: numpy ===
# Used symbols: arange, array, finfo, float64, hstack, iinfo, int64, max, mean, min, nan, nanmean, ones, percentile, quantile, random, std, sum, uint64

# === Internal dependency: pandas ===
from pandas.core.api import NA
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import RangeIndex
from pandas.core.api import MultiIndex
from pandas.core.api import NaT
from pandas.core.api import period_range
from pandas.core.api import Timedelta
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import interval_range
from pandas.core.api import to_datetime
from pandas.core.api import NamedAgg
from pandas.core.api import array
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal
UNSIGNED_INT_NUMPY_DTYPES = ['uint8', 'uint16', 'uint32', 'uint64']
UNSIGNED_INT_EA_DTYPES = ['UInt8', 'UInt16', 'UInt32', 'UInt64']
SIGNED_INT_NUMPY_DTYPES = [int, 'int8', 'int16', 'int32', 'int64']
SIGNED_INT_EA_DTYPES = ['Int8', 'Int16', 'Int32', 'Int64']
ALL_INT_NUMPY_DTYPES = UNSIGNED_INT_NUMPY_DTYPES + SIGNED_INT_NUMPY_DTYPES
ALL_INT_EA_DTYPES = UNSIGNED_INT_EA_DTYPES + SIGNED_INT_EA_DTYPES
FLOAT_NUMPY_DTYPES = [float, 'float32', 'float64']
FLOAT_EA_DTYPES = ['Float32', 'Float64']

# === Internal dependency: pandas.core.dtypes.common ===
def is_integer_dtype(arr_or_dtype): ...

# === Internal dependency: pandas.core.groupby.grouper ===
class Grouping:
    def __init__(self, index, grouper=..., obj=..., level=..., sort=..., observed=..., in_axis=..., dropna=..., uniques=...): ...

# === Internal dependency: pandas.errors ===
class SpecificationError(Exception): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, param, raises