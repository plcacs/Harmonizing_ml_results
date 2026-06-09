# === Third-party dependency: numpy ===
# Used symbols: all, any, arange, argmax, argmin, array, asarray, concatenate, empty, fromiter, int64, intp, lexsort, log, ndarray, nonzero, prod, putmask, where

# === Internal dependency: pandas ===
from pandas.core.api import MultiIndex
from pandas.core.api import Series

# === Internal dependency: pandas._libs.algos ===
def groupsort_indexer(index, ngroups): ...

# === Internal dependency: pandas._libs.hashtable ===
def unique_label_indices(labels): ...
class Int64HashTable(HashTable):
    def get_labels_groupby(self, values): ...

# === Internal dependency: pandas._libs.lib ===
def indices_fast(index, labels, keys, sorted_labels): ...
i8max = ...

# === Internal dependency: pandas._typing ===
ArrayLike = Union['ExtensionArray', np.ndarray]

# === Internal dependency: pandas.core.construction ===
def extract_array(obj, extract_numpy=..., extract_range=...): ...

# === Internal dependency: pandas.core.dtypes.common ===
ensure_int64 = algos.ensure_int64
ensure_platform_int = algos.ensure_platform_int

# === Internal dependency: pandas.core.dtypes.generic ===
def create_pandas_abc_type(name, attr, comp): ...
ABCRangeIndex = cast(...)
ABCMultiIndex = cast(...)

# === Internal dependency: pandas.core.dtypes.missing ===
def isna(obj): ...

# === Internal dependency: pandas.core.indexes.api ===
from pandas.core.indexes.base import Index

# === Internal dependency: pandas.core.indexes.base ===
class Index(IndexOpsMixin, PandasObject): ...