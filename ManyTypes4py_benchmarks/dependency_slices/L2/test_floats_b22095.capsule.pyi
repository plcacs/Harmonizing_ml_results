from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, float64, int64, random

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import RangeIndex
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import timedelta_range
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
def setitem(x) -> Any: ...
def loc(x) -> Any: ...
def iloc(x) -> Any: ...
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Third-party dependency: pytest ===
# Used symbols: mark, raises