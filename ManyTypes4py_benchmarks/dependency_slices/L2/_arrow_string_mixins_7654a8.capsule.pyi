from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: iinfo, int64

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import array

# === Internal dependency: pandas._typing ===
Self: Any

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat.pyarrow import pa_version_under10p1
# re-export: from pandas.compat.pyarrow import pa_version_under11p0
# re-export: from pandas.compat.pyarrow import pa_version_under13p0
# re-export: from pandas.compat.pyarrow import pa_version_under17p0

# === Internal dependency: pandas.core.dtypes.missing ===
def isna(obj: Scalar | Pattern | NAType | NaTType) -> bool: ...
def isna(obj: ArrayLike | Index | list) -> npt.NDArray[np.bool_]: ...
def isna(obj: NDFrameT) -> NDFrameT: ...
def isna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]: ...
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...

# === Third-party dependency: pyarrow ===
# Used symbols: ChunkedArray, chunked_array, scalar

# === Third-party dependency: pyarrow.compute ===
# Used symbols: add, ends_with, find_substring, greater, greater_equal, if_else, invert, is_null, less, match_substring, match_substring_regex, not_equal, or_, replace_substring, replace_substring_regex, starts_with, utf8_capitalize, utf8_center, utf8_is_alnum, utf8_is_alpha, utf8_is_decimal, utf8_is_digit, utf8_is_lower, utf8_is_numeric, utf8_is_space, utf8_is_title, utf8_is_upper, utf8_length, utf8_lower, utf8_lpad, utf8_ltrim, utf8_ltrim_whitespace, utf8_replace_slice, utf8_rpad, utf8_rtrim, utf8_rtrim_whitespace, utf8_slice_codeunits, utf8_swapcase, utf8_title, utf8_trim, utf8_trim_whitespace, utf8_upper