from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, eye, float64, int64, isnan, nan, ones, random, shares_memory

# === Internal dependency: pandas ===
from pandas.core.api import CategoricalDtype
from pandas.core.api import isna
from pandas.core.api import Index
from pandas.core.api import NaT
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import Interval
from pandas.core.api import interval_range
from pandas.core.api import to_datetime
from pandas.core.api import array
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._testing ===
def shares_memory(left, right): ...
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_categorical_equal
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal
from pandas._testing.compat import get_obj

# === Internal dependency: pandas.api.types ===
is_scalar: Any

# === Internal dependency: pandas.errors ===
class IndexingError(Exception): ...

# === Internal dependency: pandas.tests.indexing.common ===
def check_indexing_smoketest_or_raises(obj, method, key, axes=..., fails=...): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises