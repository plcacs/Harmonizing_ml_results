from typing import Any, List, Optional, Dict, Iterable, Tuple, Type
from pyspark import StorageLevel
from pyspark.sql import DataFrame as SparkDataFrame, functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType, StringType, ArrayType
import warnings

# Assume that the following names are defined elsewhere:
# DataFrame, InternalFrame, CachedAccessor, CachedSparkFrameMethods, _MissingPandasLikeDataFrame, is_testing

def _reduce_spark_multi(sdf: SparkDataFrame, aggs: List[Any]) -> List[Any]:
    """
    Performs a reduction on a spark DataFrame, the functions being known SQL aggregate functions.
    """
    assert isinstance(sdf, SparkDataFrame)
    sdf0 = sdf.agg(*aggs)
    l = sdf0.limit(2).toPandas()
    assert len(l) == 1, (sdf, l)
    row = l.iloc[0]
    l2: List[Any] = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2


class CachedDataFrame(DataFrame):
    """
    Cached Koalas DataFrame, which corresponds to a pandas DataFrame logically, but
    internally it caches the corresponding Spark DataFrame.
    """
    def __init__(self, internal: InternalFrame, storage_level: Optional[StorageLevel] = None) -> None:
        if storage_level is None:
            object.__setattr__(self, '_cached', internal.spark_frame.cache())
        elif isinstance(storage_level, StorageLevel):
            object.__setattr__(self, '_cached', internal.spark_frame.persist(storage_level))
        else:
            raise TypeError('Only a valid pyspark.StorageLevel type is acceptable for the `storage_level`')
        super().__init__(internal)

    def __enter__(self) -> "CachedDataFrame":
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException], traceback: Optional[Any]) -> None:
        self.spark.unpersist()

    spark = CachedAccessor('spark', CachedSparkFrameMethods)

    @property
    def storage_level(self) -> StorageLevel:
        warnings.warn('DataFrame.storage_level is deprecated as of DataFrame.spark.storage_level. Please use the API instead.', FutureWarning)
        return self.spark.storage_level

    def unpersist(self) -> Any:
        warnings.warn('DataFrame.unpersist is deprecated as of DataFrame.spark.unpersist. Please use the API instead.', FutureWarning)
        return self.spark.unpersist()

    def hint(self, name: str, *parameters: Any) -> Any:
        warnings.warn('DataFrame.hint is deprecated as of DataFrame.spark.hint. Please use the API instead.', FutureWarning)
        return self.spark.hint(name, *parameters)

    def to_table(self, name: str, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[List[str]] = None, index_col: Optional[str] = None, **options: Any) -> Any:
        return self.spark.to_table(name, format, mode, partition_cols, index_col, **options)

    def to_delta(self, path: str, mode: str = 'overwrite', partition_cols: Optional[List[str]] = None, index_col: Optional[str] = None, **options: Any) -> None:
        self.spark.to_spark_io(path=path, mode=mode, format='delta', partition_cols=partition_cols, index_col=index_col, **options)

    def to_parquet(self, path: str, mode: str = 'overwrite', partition_cols: Optional[List[str]] = None, compression: Optional[str] = None, index_col: Optional[str] = None, **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        builder = self.to_spark(index_col=index_col).write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format('parquet').save(path)

    def to_orc(self, path: str, mode: str = 'overwrite', partition_cols: Optional[List[str]] = None, index_col: Optional[str] = None, **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self.spark.to_spark_io(path=path, mode=mode, format='orc', partition_cols=partition_cols, index_col=index_col, **options)

    def to_spark_io(self, path: Optional[str] = None, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[List[str]] = None, index_col: Optional[str] = None, **options: Any) -> Any:
        return self.spark.to_spark_io(path, format, mode, partition_cols, index_col, **options)

    def to_spark(self, index_col: Optional[str] = None) -> Any:
        return self.spark.frame(index_col)

    def to_pandas(self) -> Any:
        # Assuming pandas DataFrame
        return self._internal.to_pandas_frame.copy()

    def toPandas(self) -> Any:
        warnings.warn('DataFrame.toPandas is deprecated as of DataFrame.to_pandas. Please use the API instead.', FutureWarning)
        return self.to_pandas() 