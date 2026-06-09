# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, concatenate, datetime64, dtype, int64, intp, isnan, may_share_memory, nan, ndarray, newaxis, ones, random, timedelta64

# === Internal dependency: pandas ===
from pandas._config import option_context
from pandas.core.api import StringDtype
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import TimedeltaIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import PeriodIndex
from pandas.core.api import Period
from pandas.core.api import period_range
from pandas.core.api import Timedelta
from pandas.core.api import timedelta_range
from pandas.core.api import date_range
from pandas.core.api import to_datetime
from pandas.core.api import to_timedelta
from pandas.core.api import array
from pandas.core.api import Categorical
from pandas.core.api import Series

# === Internal dependency: pandas._libs ===
from pandas._libs.tslibs import NaT
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas._libs.tslibs import Timestamp

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.offsets import to_offset

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_datetime_array_equal
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_extension_array_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_period_array_equal

# === Internal dependency: pandas.compat.numpy ===
_np_version = np.__version__
_nlv = Version(...)
np_version_gt2 = _nlv >= Version('2.0.0')

# === Internal dependency: pandas.core.arrays ===
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.period import PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray

# === Internal dependency: pandas.core.dtypes.dtypes ===
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...

# === Internal dependency: pandas.util.version ===
class Version(_BaseVersion): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises