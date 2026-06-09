from typing import Any

# === Third-party dependency: botocore ===
# Used symbols: exceptions

# === Internal dependency: io ===
BytesIO: Any

# === Third-party dependency: numpy ===
# Used symbols: zeros

# === Internal dependency: pandas ===
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_frame_equal

# === Internal dependency: pandas.io.feather_format ===
def read_feather(path, columns=..., use_threads=..., storage_options=..., dtype_backend=...): ...

# === Internal dependency: pandas.io.parsers ===
from pandas.io.parsers.readers import read_csv

# === Internal dependency: pandas.util._test_decorators ===
skip_if_not_us_locale = pytest.mark.skipif(...)

# === Third-party dependency: pytest ===
# Used symbols: fixture, importorskip, mark, raises, skip