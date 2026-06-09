from typing import Any

# === Internal dependency: databricks.koalas ===
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.series import Series

# === Internal dependency: databricks.koalas.config ===
def set_option(key, value): ...
def reset_option(key): ...

# === Internal dependency: databricks.koalas.frame ===
class DataFrame(Frame, Generic[T]): ...

# === Internal dependency: databricks.koalas.testing.utils ===
class SQLTestUtils(object):
    ...
class ReusedSQLTestCase(unittest.TestCase, SQLTestUtils):
    def tearDownClass(cls): ...

# === Internal dependency: databricks.koalas.typedef.typehints ===
extension_dtypes: Any
extension_dtypes_available: Any
extension_float_dtypes_available: Any
extension_object_dtypes_available: Any

# === Third-party dependency: distutils.version ===
class LooseVersion(Version): ...

# === Third-party dependency: numpy ===
# Used symbols: inf, nan, random

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, Index, MultiIndex, Series, __version__, concat

# === Third-party dependency: pyspark ===
# Used symbols: __version__