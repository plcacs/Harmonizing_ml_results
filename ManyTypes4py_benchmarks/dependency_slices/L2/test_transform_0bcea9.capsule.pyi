from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: abs, all, any, arange, errstate, float64, int32, max, mean, min, nan, nansum, random, repeat, take, tile

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import RangeIndex
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import timedelta_range
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Grouper
# re-export: from pandas.core.api import Categorical
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas._libs.lib ===
def is_np_dtype(dtype: object, kinds: str | None = ...) -> TypeGuard[np.dtype]: ...

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.compat ===
# re-export: from pandas.compat.pyarrow import HAS_PYARROW

# === Internal dependency: pandas.core.dtypes.common ===
ensure_platform_int: Any

# === Internal dependency: pandas.tests.groupby ===
def get_groupby_method_args(name, obj) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises