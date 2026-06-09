# === Third-party dependency: numpy ===
# Used symbols: arange, int64, mean, random, std, sum, zeros_like

# === Internal dependency: pandas ===
from pandas.core.api import MultiIndex
from pandas.core.api import DatetimeIndex
from pandas.core.api import Timestamp
from pandas.core.api import to_datetime
from pandas.core.api import Grouper
from pandas.core.api import NamedAgg
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas import errors

# === Internal dependency: pandas._libs.lib ===
class _NoDefault(Enum):
    no_default = Ellipsis
no_default = _NoDefault.no_default

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.core.indexes.datetimes ===
def date_range(start=..., end=..., periods=..., freq=..., tz=..., normalize=..., name=..., inclusive=..., *, unit=..., **kwargs): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises