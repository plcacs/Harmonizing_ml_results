# === Third-party dependency: numpy ===
# Used symbols: abs, arange, array, corrcoef, cov, float64, full, int64, nan, nan_to_num, ones, random, repeat, vstack, zeros

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat._constants import IS64

# === Internal dependency: pandas.core.algorithms ===
def safe_sort(values: Index | ArrayLike, codes: npt.NDArray[np.intp] | None = ..., use_na_sentinel: bool = ..., assume_unique: bool = ..., verify: bool = ...) -> AnyArrayLike | tuple[AnyArrayLike, np.ndarray]: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises