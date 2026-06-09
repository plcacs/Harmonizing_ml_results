# === Third-party dependency: numpy ===
# Used symbols: arange, array, asanyarray, asarray, int16, int64, int8, nan, object_, random

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_categorical_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat._constants import PY311

# === Internal dependency: pandas.core.arrays.categorical ===
def recode_for_categories(codes: np.ndarray, old_categories, new_categories, copy: bool = ...) -> np.ndarray: ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises