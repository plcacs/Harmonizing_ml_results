from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: all, any, arange, argmax, argmin, array, asarray, concatenate, empty, fromiter, int64, intp, lexsort, log, ndarray, nonzero, prod, putmask, where

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import Series

# === Internal dependency: pandas._libs.algos ===
def groupsort_indexer(index: np.ndarray, ngroups: int) -> tuple[np.ndarray, np.ndarray]: ...

# === Internal dependency: pandas._libs.hashtable ===
def unique_label_indices(labels: np.ndarray) -> np.ndarray: ...
class Int64HashTable(HashTable):
    def get_labels_groupby(self, values: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]: ...

# === Internal dependency: pandas._libs.lib ===
i8max: int
def indices_fast(index: npt.NDArray[np.intp], labels: np.ndarray, keys: list, sorted_labels: list[npt.NDArray[np.int64]]) -> dict[Hashable, npt.NDArray[np.intp]]: ...

# === Internal dependency: pandas._typing ===
ArrayLike: Any

# === Internal dependency: pandas.core.construction ===
def extract_array(obj: Series | Index, extract_numpy: bool = ..., extract_range: bool = ...) -> ArrayLike: ...
def extract_array(obj: T, extract_numpy: bool = ..., extract_range: bool = ...) -> T | ArrayLike: ...

# === Internal dependency: pandas.core.dtypes.common ===
ensure_int64: Any
ensure_platform_int: Any

# === Internal dependency: pandas.core.dtypes.generic ===
ABCRangeIndex: cast
ABCMultiIndex: cast

# === Internal dependency: pandas.core.dtypes.missing ===
def isna(obj: Scalar | Pattern | NAType | NaTType) -> bool: ...
def isna(obj: ArrayLike | Index | list) -> npt.NDArray[np.bool_]: ...
def isna(obj: NDFrameT) -> NDFrameT: ...
def isna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]: ...
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...

# === Internal dependency: pandas.core.indexes.api ===
# re-export: from pandas.core.indexes.base import Index

# === Internal dependency: pandas.core.indexes.base ===
class Index(IndexOpsMixin, PandasObject): ...