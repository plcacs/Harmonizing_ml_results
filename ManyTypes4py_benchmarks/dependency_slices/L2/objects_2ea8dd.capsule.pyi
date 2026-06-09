from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: append, arange, array, clip, concatenate, empty, int64, ndarray, zeros

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.offsets import BaseOffset

# === Internal dependency: pandas._libs.window.indexers ===
def calculate_variable_window_bounds(num_values: int, window_size: int, min_periods, center: bool, closed: str | None, index: np.ndarray) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...

# === Internal dependency: pandas.core.dtypes.common ===
ensure_platform_int: Any

# === Internal dependency: pandas.core.indexes.datetimes ===
class DatetimeIndex(DatetimeTimedeltaMixin): ...

# === Internal dependency: pandas.tseries.offsets ===
# re-export: from pandas._libs.tslibs.offsets import Nano

# === Internal dependency: pandas.util._decorators ===
class Appender: ...