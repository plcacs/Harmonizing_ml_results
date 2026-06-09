# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, concatenate, dtype, float64, int16, int32, int64, nan, object_, prod, random, repeat, tile, vstack, zeros

# === Internal dependency: pandas ===
from pandas.core.api import Int64Dtype
from pandas.core.api import Float64Dtype
from pandas.core.api import StringDtype
from pandas.core.api import NA
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import RangeIndex
from pandas.core.api import MultiIndex
from pandas.core.api import PeriodIndex
from pandas.core.api import NaT
from pandas.core.api import Period
from pandas.core.api import Timedelta
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import to_datetime
from pandas.core.api import array
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._libs.lib ===
class _NoDefault(Enum):
    no_default = Ellipsis
no_default = _NoDefault.no_default

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.reshape.reshape ===
class _Unstacker:
    ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises