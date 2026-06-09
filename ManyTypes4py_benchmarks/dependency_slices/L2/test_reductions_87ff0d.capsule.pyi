from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, finfo, iinfo, int64, max, mean, median, min, nan, nanmedian, prod, random, repeat, size, std, sum, uint64, var

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import isna
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import CategoricalIndex
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import Timedelta
# re-export: from pandas.core.api import timedelta_range
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import to_timedelta
# re-export: from pandas.core.api import Grouper
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import cut
# re-export: from pandas import errors

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.nattype import iNaT

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.dtypes.common ===
def pandas_dtype(dtype) -> DtypeObj: ...

# === Internal dependency: pandas.core.dtypes.missing ===
def na_value_for_dtype(dtype: DtypeObj, compat: bool = ...) -> Any: ...

# === Internal dependency: pandas.util._test_decorators ===
def skip_if_no(package: str, min_version: str | None = ...) -> pytest.MarkDecorator: ...

# === Third-party dependency: pytest ===
# Used symbols: mark, param, raises

# === Third-party dependency: scipy.stats ===
# Used symbols: sem