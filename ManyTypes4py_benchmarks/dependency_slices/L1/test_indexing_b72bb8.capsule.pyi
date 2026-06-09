# === Third-party dependency: numpy ===
# Used symbols: arange, array, bool_, int64, intp, nan, ones, random

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import IndexSlice
from pandas.core.api import timedelta_range
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.errors ===
class InvalidIndexError(Exception): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises