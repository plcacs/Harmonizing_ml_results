from typing import Any

# === Internal dependency: databricks.koalas ===
# re-export: from databricks.koalas.indexes.multi import MultiIndex

# === Internal dependency: databricks.koalas.frame ===
class DataFrame(Frame, Generic[T]): ...
class CachedDataFrame(DataFrame): ...

# === Internal dependency: databricks.koalas.internal ===
HIDDEN_COLUMNS: Any

# === Internal dependency: databricks.koalas.series ===
class Series(Frame, IndexOpsMixin, Generic[T]): ...
def first_series(df) -> Union['Series', pd.Series]: ...

# === Internal dependency: databricks.koalas.utils ===
def name_like_string(name: Optional[Union[str, Tuple]]) -> str: ...

# === Third-party dependency: distutils.version ===
class LooseVersion(Version): ...

# === Third-party dependency: pyspark ===
# Used symbols: StorageLevel, __version__

# === Third-party dependency: pyspark.sql ===
# Used symbols: Column, DataFrame

# === Third-party dependency: pyspark.sql.types ===
class DataType: ...
class StructType(DataType): ...