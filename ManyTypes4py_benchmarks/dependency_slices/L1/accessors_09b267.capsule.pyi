# === Internal dependency: databricks.koalas ===
from databricks.koalas.indexes.multi import MultiIndex

# === Internal dependency: databricks.koalas.frame ===
class DataFrame(Frame, Generic[T]): ...
class CachedDataFrame(DataFrame): ...

# === Internal dependency: databricks.koalas.internal ===
NATURAL_ORDER_COLUMN_NAME = '__natural_order__'
HIDDEN_COLUMNS = {NATURAL_ORDER_COLUMN_NAME}

# === Internal dependency: databricks.koalas.series ===
class Series(Frame, IndexOpsMixin, Generic[T]): ...
def first_series(df): ...

# === Internal dependency: databricks.koalas.utils ===
def name_like_string(name): ...

# === Third-party dependency: distutils.version ===
class LooseVersion(Version): ...

# === Third-party dependency: pyspark ===
# Used symbols: StorageLevel, __version__

# === Third-party dependency: pyspark.sql ===
# Used symbols: Column, DataFrame

# === Third-party dependency: pyspark.sql.types ===
class DataType: ...
class StructType(DataType): ...