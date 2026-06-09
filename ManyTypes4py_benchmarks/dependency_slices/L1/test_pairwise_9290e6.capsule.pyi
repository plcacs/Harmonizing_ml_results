# === Third-party dependency: numpy ===
# Used symbols: abs, arange, array, corrcoef, cov, float64, full, int64, nan, nan_to_num, ones, random, repeat, vstack, zeros

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.compat ===
from pandas.compat._constants import IS64

# === Internal dependency: pandas.core.algorithms ===
def safe_sort(values, codes=..., use_na_sentinel=..., assume_unique=..., verify=...): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises