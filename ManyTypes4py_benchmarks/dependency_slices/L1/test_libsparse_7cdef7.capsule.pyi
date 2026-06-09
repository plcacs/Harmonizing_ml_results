# === Third-party dependency: numpy ===
# Used symbols: arange, array, int32, ones

# === Internal dependency: pandas ===
from pandas.core.api import Series

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_numpy_array_equal

# === Internal dependency: pandas.core.arrays.sparse ===
from pandas.core.arrays.sparse.array import BlockIndex
from pandas.core.arrays.sparse.array import IntIndex
from pandas.core.arrays.sparse.array import make_sparse_index

# === Internal dependency: pandas.util._test_decorators ===
skip_if_windows = pytest.mark.skipif(...)

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises