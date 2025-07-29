from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union, Iterable
import pyspark
from pyspark import StorageLevel
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Other necessary imports assumed to be here

def _reduce_spark_multi(sdf: SparkDataFrame, aggs: List[Any]) -> List[Any]:
    """
    Performs a reduction on a spark DataFrame, the functions being known sql aggregate functions.
    """
    assert isinstance(sdf, SparkDataFrame)
    sdf0 = sdf.agg(*aggs)
    l = sdf0.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row = l.iloc[0]
    l2 = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2


class CachedDataFrame(DataFrame):
    """
    Cached Koalas DataFrame, which corresponds to pandas DataFrame logically, but internally
    it caches the corresponding Spark DataFrame.
    """
    def __init__(self, internal: Any, storage_level: Optional[StorageLevel] = None) -> None:
        if storage_level is None:
            object.__setattr__(self, '_cached', internal.spark_frame.cache())
        elif isinstance(storage_level, StorageLevel):
            object.__setattr__(self, '_cached', internal.spark_frame.persist(storage_level))
        else:
            raise TypeError('Only a valid pyspark.StorageLevel type is acceptable for the `storage_level`')
        super().__init__(internal)

    def __enter__(self) -> "CachedDataFrame":
        return self

    def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
        self.spark.unpersist()

    spark: Any  # CachedAccessor is attached externally via CachedAccessor('spark', CachedSparkFrameMethods)

    @property
    def storage_level(self) -> StorageLevel:
        import warnings
        warnings.warn('DataFrame.storage_level is deprecated as of DataFrame.spark.storage_level. Please use the API instead.', FutureWarning)
        return self.spark.storage_level

    def unpersist(self) -> Any:
        import warnings
        warnings.warn('DataFrame.unpersist is deprecated as of DataFrame.spark.unpersist. Please use the API instead.', FutureWarning)
        return self.spark.unpersist()