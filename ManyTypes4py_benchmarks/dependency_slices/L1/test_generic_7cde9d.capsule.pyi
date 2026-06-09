# === Third-party dependency: numpy ===
# Used symbols: array, array_split, empty, eye, float64, nan, prod, random, repeat, squeeze, transpose

# === Internal dependency: pandas ===
from pandas.core.api import Index
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_metadata_equivalent
from pandas._testing.asserters import assert_series_equal
from pandas._testing.compat import get_obj

# === Internal dependency: pandas.core.dtypes.common ===
from pandas.core.dtypes.inference import is_scalar

# === Third-party dependency: pytest ===
# Used symbols: mark, raises