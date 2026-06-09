# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, dtype, empty, fill_diagonal, float64, full, int32, int64, intp, max, mean, min, nan, random, size, sum

# === Internal dependency: pandas ===
from pandas.core.api import DatetimeTZDtype
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import MultiIndex
from pandas.core.api import IntervalIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import PeriodIndex
from pandas.core.api import Period
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import Interval
from pandas.core.api import to_datetime
from pandas.core.api import Grouper
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas.core.reshape.api import pivot

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.types ===
from pandas.core.dtypes.dtypes import CategoricalDtype

# === Internal dependency: pandas.compat.numpy ===
_np_version = np.__version__
_nlv = Version(...)
np_version_gte1p25 = _nlv >= Version('1.25')

# === Internal dependency: pandas.core.reshape.pivot ===
def pivot_table(data, values=..., index=..., columns=..., aggfunc=..., fill_value=..., margins=..., dropna=..., margins_name=..., observed=..., sort=..., **kwargs): ...

# === Internal dependency: pandas.core.reshape.reshape ===
class _Unstacker:
    ...

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises