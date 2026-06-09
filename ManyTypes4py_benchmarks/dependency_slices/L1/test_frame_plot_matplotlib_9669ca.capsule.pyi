from typing import Any

# === Internal dependency: databricks.koalas ===
from_pandas: Any

# === Internal dependency: databricks.koalas.config ===
def set_option(key, value): ...
def reset_option(key): ...

# === Internal dependency: databricks.koalas.testing.utils ===
class ReusedSQLTestCase(unittest.TestCase, SQLTestUtils):
    def tearDownClass(cls): ...
class TestUtils(object):
    ...

# === Third-party dependency: distutils.version ===
class LooseVersion(Version): ...

# === Third-party dependency: matplotlib ===
def use(backend, *, force = ...) -> Any: ...

# === Third-party dependency: numpy ===
# Used symbols: allclose, cumsum, random

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, MultiIndex, __version__, date_range, reset_option, set_option