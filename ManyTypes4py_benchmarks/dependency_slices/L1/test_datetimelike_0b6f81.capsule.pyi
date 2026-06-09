# === Third-party dependency: numpy ===
# Used symbols: arange, array, asarray, fabs, float64, ma, nan, random, zeros

# === Internal dependency: pandas ===
from pandas.core.api import isna
from pandas.core.api import Index
from pandas.core.api import NaT
from pandas.core.api import to_datetime
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat

# === Internal dependency: pandas._libs.tslibs ===
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.offsets import to_offset

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.dtypes.dtypes ===
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype): ...

# === Internal dependency: pandas.core.indexes.datetimes ===
class DatetimeIndex(DatetimeTimedeltaMixin):
    ...
def date_range(start=..., end=..., periods=..., freq=..., tz=..., normalize=..., name=..., inclusive=..., *, unit=..., **kwargs): ...
def bdate_range(start=..., end=..., periods=..., freq=..., tz=..., normalize=..., name=..., weekmask=..., holidays=..., inclusive=..., **kwargs): ...

# === Internal dependency: pandas.core.indexes.period ===
class PeriodIndex(DatetimeIndexOpsMixin):
    ...
def period_range(start=..., end=..., periods=..., freq=..., name=...): ...
from pandas._libs.tslibs import Period

# === Internal dependency: pandas.core.indexes.timedeltas ===
def timedelta_range(start=..., end=..., periods=..., freq=..., name=..., closed=..., *, unit=...): ...

# === Internal dependency: pandas.plotting._matplotlib.converter ===
def get_datevalue(date, freq): ...
class DatetimeConverter(mdates.DateConverter):
    ...
def _daily_finder(vmin, vmax, freq): ...
def _monthly_finder(vmin, vmax, freq): ...
def _quarterly_finder(vmin, vmax, freq): ...
def _annual_finder(vmin, vmax, freq): ...
def get_finder(freq): ...

# === Internal dependency: pandas.tests.plotting.common ===
def _check_ticks_props(axes, xlabelsize=..., xrot=..., ylabelsize=..., yrot=...): ...

# === Internal dependency: pandas.tseries.offsets ===
from pandas._libs.tslibs.offsets import WeekOfMonth

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises