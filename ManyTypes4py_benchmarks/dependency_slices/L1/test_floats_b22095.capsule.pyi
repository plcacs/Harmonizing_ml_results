# === Third-party dependency: numpy ===
# Used symbols: arange, array, float64, int64, random

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import RangeIndex
from pandas.core.api import period_range
from pandas.core.api import timedelta_range
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
def setitem(x): ...
def loc(x): ...
def iloc(x): ...
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_series_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises