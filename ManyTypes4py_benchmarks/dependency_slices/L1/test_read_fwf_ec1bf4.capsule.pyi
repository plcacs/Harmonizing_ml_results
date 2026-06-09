from typing import Any

# === Internal dependency: io ===
BytesIO: Any
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: nan

# === Internal dependency: pandas ===
from pandas._config import option_context
from pandas.core.api import ArrowDtype
from pandas.core.api import StringDtype
from pandas.core.api import NA
from pandas.core.api import Index
from pandas.core.api import DatetimeIndex
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing._io import write_to_compressed
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.contexts import ensure_clean

# === Internal dependency: pandas.arrays ===
from pandas.core.arrays import ArrowExtensionArray

# === Internal dependency: pandas.errors ===
class EmptyDataError(ValueError): ...

# === Internal dependency: pandas.io.common ===
def urlopen(*args, **kwargs): ...

# === Internal dependency: pandas.io.parsers ===
from pandas.io.parsers.readers import read_csv
from pandas.io.parsers.readers import read_fwf

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises