# === Third-party dependency: numpy ===
# Used symbols: arange, array, array_split, concatenate, int64, intp, nan, ones, ones_like, random, tile, unique, vstack, zeros

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat
# re-export: from pandas.core.reshape.api import merge

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_extension_array_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.algorithms ===
def safe_sort(values: Index | ArrayLike, codes: npt.NDArray[np.intp] | None = ..., use_na_sentinel: bool = ..., assume_unique: bool = ..., verify: bool = ...) -> AnyArrayLike | tuple[AnyArrayLike, np.ndarray]: ...

# === Internal dependency: pandas.core.common ===
def asarray_tuplesafe(values: ArrayLike | list | tuple | zip, dtype: NpDtype | None = ...) -> np.ndarray: ...
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = ...) -> ArrayLike: ...

# === Internal dependency: pandas.core.sorting ===
def get_group_index(labels, shape: Shape, sort: bool, xnull: bool) -> npt.NDArray[np.int64]: ...
def is_int64_overflow_possible(shape: Shape) -> bool: ...
def _decons_group_index(comp_labels: npt.NDArray[np.intp], shape: Shape) -> list[npt.NDArray[np.intp]]: ...
def lexsort_indexer(keys: Sequence[ArrayLike | Index | Series], orders = ..., na_position: str = ..., key: Callable | None = ..., codes_given: bool = ...) -> npt.NDArray[np.intp]: ...
def nargsort(items: ArrayLike | Index | Series, kind: SortKind = ..., ascending: bool = ..., na_position: str = ..., key: Callable | None = ..., mask: npt.NDArray[np.bool_] | None = ...) -> npt.NDArray[np.intp]: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises