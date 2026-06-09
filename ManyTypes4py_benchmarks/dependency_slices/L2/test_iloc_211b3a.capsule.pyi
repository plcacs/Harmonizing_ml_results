from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, eye, float64, int64, isnan, nan, ones, random, shares_memory

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import CategoricalDtype
# re-export: from pandas.core.api import isna
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Interval
# re-export: from pandas.core.api import interval_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._testing ===
def shares_memory(left, right) -> bool: ...
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_categorical_equal
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal
# re-export: from pandas._testing.compat import get_obj

# === Internal dependency: pandas.api.types ===
is_scalar: Any

# === Internal dependency: pandas.errors ===
class IndexingError(Exception): ...

# === Internal dependency: pandas.tests.indexing.common ===
def check_indexing_smoketest_or_raises(obj, method: Literal['iloc', 'loc'], key: Any, axes: Literal[0, 1] | None = ..., fails = ...) -> None: ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises