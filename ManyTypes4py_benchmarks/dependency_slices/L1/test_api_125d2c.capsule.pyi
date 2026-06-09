# === Third-party dependency: numpy ===
# Used symbols: arange, array, asanyarray, asarray, int16, int64, int8, nan, object_, random

# === Internal dependency: pandas ===
from pandas.core.api import StringDtype
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_categorical_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.compat ===
from pandas.compat._constants import PY311

# === Internal dependency: pandas.core.arrays.categorical ===
def recode_for_categories(codes, old_categories, new_categories, copy=...): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises