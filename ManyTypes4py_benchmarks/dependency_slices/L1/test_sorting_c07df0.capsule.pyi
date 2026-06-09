# === Third-party dependency: numpy ===
# Used symbols: arange, array, array_split, concatenate, int64, intp, nan, ones, ones_like, random, tile, unique, vstack, zeros

# === Internal dependency: pandas ===
from pandas.core.api import NA
from pandas.core.api import MultiIndex
from pandas.core.api import array
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas.core.reshape.api import merge

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_extension_array_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.algorithms ===
def safe_sort(values, codes=..., use_na_sentinel=..., assume_unique=..., verify=...): ...

# === Internal dependency: pandas.core.common ===
def asarray_tuplesafe(values, dtype=...): ...

# === Internal dependency: pandas.core.sorting ===
def get_group_index(labels, shape, sort, xnull): ...
def is_int64_overflow_possible(shape): ...
def _decons_group_index(comp_labels, shape): ...
def lexsort_indexer(keys, orders=..., na_position=..., key=..., codes_given=...): ...
def nargsort(items, kind=..., ascending=..., na_position=..., key=..., mask=...): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises