# === Third-party dependency: numpy ===
# Used symbols: array, timedelta64

# === Internal dependency: pandas ===
from pandas.core.api import NaT
from pandas.core.api import Period
from pandas.core.api import Timedelta
from pandas.core.api import Timestamp
from pandas.tseries import offsets

# === Internal dependency: pandas._libs.tslibs.period ===
class IncompatibleFrequency(ValueError): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises