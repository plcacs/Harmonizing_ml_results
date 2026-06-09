from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, dtype, errstate, intp, nan, repeat, uint64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import ArrowDtype
# re-export: from pandas.core.api import StringDtype
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import factorize
# re-export: from pandas.core.api import unique
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._testing ===
def box_expected(expected, box_cls, transpose: bool = ...) -> Any: ...
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_extension_array_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas._typing ===
Dtype: Any

# === Internal dependency: pandas.core.dtypes.common ===
def is_bool_dtype(arr_or_dtype) -> bool: ...

# === Internal dependency: pandas.core.dtypes.dtypes ===
class NumpyEADtype(ExtensionDtype):
    def __init__(self, dtype: npt.DTypeLike | NumpyEADtype | None) -> None: ...

# === Internal dependency: pandas.core.dtypes.missing ===
def na_value_for_dtype(dtype: DtypeObj, compat: bool = ...) -> Any: ...

# === Internal dependency: pandas.core.sorting ===
def nargsort(items: ArrayLike | Index | Series, kind: SortKind = ..., ascending: bool = ..., na_position: str = ..., key: Callable | None = ..., mask: npt.NDArray[np.bool_] | None = ...) -> npt.NDArray[np.intp]: ...

# === Internal dependency: pandas.core.util.hashing ===
_default_hash_key: str

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, skip