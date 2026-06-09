# === Third-party dependency: numpy ===
# Used symbols: array, int64, nan, random

# === Internal dependency: pandas ===
from pandas.core.api import StringDtype
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas.core.reshape.api import lreshape
from pandas.core.reshape.api import melt
from pandas.core.reshape.api import wide_to_long

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_frame_equal

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises