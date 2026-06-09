# === Third-party dependency: numpy ===
# Used symbols: arange, array, bool_, float64, full, int32, int64, nan, random, unique, zeros

# === Internal dependency: pandas ===
from pandas.core.api import isna
from pandas.core.api import MultiIndex
from pandas.core.api import date_range
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.dtypes.dtypes import SparseDtype

# === Internal dependency: pandas.arrays ===
from pandas.core.arrays import SparseArray

# === Third-party dependency: scipy.sparse ===
# Used symbols: coo_matrix, rand