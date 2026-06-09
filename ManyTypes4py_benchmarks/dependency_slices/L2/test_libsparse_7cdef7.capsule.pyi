# === Third-party dependency: numpy ===
# Used symbols: arange, array, int32, ones

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Series

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.core.arrays.sparse ===
# re-export: from pandas.core.arrays.sparse.array import BlockIndex
# re-export: from pandas.core.arrays.sparse.array import IntIndex
# re-export: from pandas.core.arrays.sparse.array import make_sparse_index

# === Internal dependency: pandas.util._test_decorators ===
skip_if_windows: mark

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises