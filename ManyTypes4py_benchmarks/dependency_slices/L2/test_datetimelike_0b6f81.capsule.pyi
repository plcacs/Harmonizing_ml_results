from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, fabs, float64, ma, nan, random, zeros

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import isna
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat

# === Internal dependency: pandas._libs.tslibs ===
# re-export: from pandas._libs.tslibs.offsets import BaseOffset
# re-export: from pandas._libs.tslibs.offsets import to_offset

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_index_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.dtypes.dtypes ===
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...

# === Internal dependency: pandas.core.indexes.datetimes ===
class DatetimeIndex(DatetimeTimedeltaMixin):
    ...
def date_range(start = ..., end = ..., periods = ..., freq = ..., tz = ..., normalize: bool = ..., name: Hashable | None = ..., inclusive: IntervalClosedType = ..., *, unit: str | None = ..., **kwargs) -> DatetimeIndex: ...
def bdate_range(start = ..., end = ..., periods: int | None = ..., freq: Frequency | dt.timedelta = ..., tz = ..., normalize: bool = ..., name: Hashable | None = ..., weekmask = ..., holidays = ..., inclusive: IntervalClosedType = ..., **kwargs) -> DatetimeIndex: ...

# === Internal dependency: pandas.core.indexes.period ===
class PeriodIndex(DatetimeIndexOpsMixin):
    ...
def period_range(start = ..., end = ..., periods: int | None = ..., freq = ..., name: Hashable | None = ...) -> PeriodIndex: ...
# re-export: from pandas._libs.tslibs import Period

# === Internal dependency: pandas.core.indexes.timedeltas ===
def timedelta_range(start = ..., end = ..., periods: int | None = ..., freq = ..., name = ..., closed = ..., *, unit: str | None = ...) -> TimedeltaIndex: ...

# === Internal dependency: pandas.plotting._matplotlib.converter ===
def get_datevalue(date, freq) -> Any: ...
class DatetimeConverter(mdates.DateConverter):
    ...
def _daily_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def _monthly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def _quarterly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def _annual_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def get_finder(freq: BaseOffset) -> Any: ...

# === Internal dependency: pandas.tests.plotting.common ===
def _check_ticks_props(axes, xlabelsize = ..., xrot = ..., ylabelsize = ..., yrot = ...) -> Any: ...

# === Internal dependency: pandas.tseries.offsets ===
# re-export: from pandas._libs.tslibs.offsets import WeekOfMonth

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises