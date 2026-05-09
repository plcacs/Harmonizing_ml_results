"""
Spark related features. Usually, the features here are missing in pandas
but Spark has it.
"""

from abc import ABCMeta
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Union,
    overload,
)
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame as SparkDataFrame
from pyspark.sql.types import DataType, StructType
from databricks.koalas import DataFrame, Index, MultiIndex, Series
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import CachedDataFrame

class SparkIndexOpsMethods(metaclass=ABCMeta):
    """Spark related features. Usually, the features here are missing in pandas
    but Spark has it."""

    def __init__(self, data: IndexOpsMixin):
        ...

    @property
    def data_type(self) -> DataType:
        ...

    @property
    def nullable(self) -> bool:
        ...

    @property
    def column(self) -> Column:
        ...

    def transform(self, func: Callable[[Column], Column]) -> Union[Series, Index]:
        ...

    @property
    @abstractmethod
    def analyzed(self) -> Union[Series, Index]:
        ...

class SparkSeriesMethods(SparkIndexOpsMethods):
    def transform(self, func: Callable[[Column], Column]) -> Series:
        ...

    def apply(self, func: Callable[[Column], Column]) -> Series:
        ...

    @property
    def analyzed(self) -> Series:
        ...

class SparkIndexMethods(SparkIndexOpsMethods):
    def transform(self, func: Callable[[Column], Column]) -> Index:
        ...

    @property
    def analyzed(self) -> Index:
        ...

class SparkFrameMethods:
    """Spark related features. Usually, the features here are missing in pandas
    but Spark has it."""

    def __init__(self, frame: DataFrame):
        ...

    def schema(self, index_col: Optional[Union[str, List[str]]] = None) -> StructType:
        ...

    def print_schema(self, index_col: Optional[Union[str, List[str]]] = None) -> None:
        ...

    def frame(self, index_col: Optional[Union[str, List[str]]] = None) -> SparkDataFrame:
        ...

    def cache(self) -> CachedDataFrame:
        ...

    def persist(
        self, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK
    ) -> CachedDataFrame:
        ...

    def hint(self, name: str, *parameters: Any) -> DataFrame:
        ...

    def to_table(
        self,
        name: str,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        ...

    def to_spark_io(
        self,
        path: Optional[str] = None,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any
    ) -> None:
        ...

    def explain(
        self, extended: Union[bool, str] = False, mode: Optional[str] = None
    ) -> None:
        ...

    def apply(
        self,
        func: Callable[[SparkDataFrame], SparkDataFrame],
        index_col: Optional[Union[str, List[str]]] = None,
    ) -> DataFrame:
        ...

    def repartition(self, num_partitions: int) -> DataFrame:
        ...

    def coalesce(self, num_partitions: int) -> DataFrame:
        ...

    def checkpoint(self, eager: bool = True) -> DataFrame:
        ...

    def local_checkpoint(self, eager: bool = True) -> DataFrame:
        ...

    @property
    def analyzed(self) -> DataFrame:
        ...

class CachedSparkFrameMethods(SparkFrameMethods):
    """Spark related features for cached DataFrame. This is usually created via
    `df.spark.cache()`."""

    def __init__(self, frame: DataFrame):
        ...

    @property
    def storage_level(self) -> StorageLevel:
        ...

    def unpersist(self) -> None:
        ...