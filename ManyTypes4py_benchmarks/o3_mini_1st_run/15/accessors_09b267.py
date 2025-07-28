from abc import ABCMeta, abstractmethod
from distutils.version import LooseVersion
from typing import TYPE_CHECKING, Optional, Union, List, cast, Any, Callable
import pyspark
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame as SparkDataFrame
from pyspark.sql.types import DataType, StructType

if TYPE_CHECKING:
    import databricks.koalas as ks
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import CachedDataFrame, DataFrame

class SparkIndexOpsMethods(metaclass=ABCMeta):
    def __init__(self, data: "IndexOpsMixin") -> None:
        self._data = data

    @property
    def data_type(self) -> DataType:
        return self._data._internal.spark_type_for(self._data._column_label)

    @property
    def nullable(self) -> bool:
        return self._data._internal.spark_column_nullable_for(self._data._column_label)

    @property
    def column(self) -> Column:
        return self._data._internal.spark_column_for(self._data._column_label)

    def transform(self, func: Callable[[Column], Column]) -> "IndexOpsMixin":
        from databricks.koalas import MultiIndex
        if isinstance(self._data, MultiIndex):
            raise NotImplementedError('MultiIndex does not support spark.transform yet.')
        output: Any = func(self._data.spark.column)
        if not isinstance(output, Column):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.Column; however, got [%s].' % (func, type(output)))
        new_ser = self._data._with_new_scol(scol=output)
        new_ser._internal.to_internal_spark_frame
        return new_ser

    @property
    @abstractmethod
    def analyzed(self) -> "IndexOpsMixin":
        pass

class SparkSeriesMethods(SparkIndexOpsMethods):
    def transform(self, func: Callable[[Column], Column]) -> "ks.Series":
        return cast("ks.Series", super().transform(func))
    transform.__doc__ = SparkIndexOpsMethods.transform.__doc__

    def apply(self, func: Callable[[Column], Column]) -> "ks.Series":
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, first_series
        from databricks.koalas.internal import HIDDEN_COLUMNS
        output: Any = func(self._data.spark.column)
        if not isinstance(output, Column):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.Column; however, got [%s].' % (func, type(output)))
        assert isinstance(self._data, Series)
        sdf: SparkDataFrame = self._data._internal.spark_frame.drop(*HIDDEN_COLUMNS).select(output)
        return first_series(DataFrame(sdf)).rename(self._data.name)

    @property
    def analyzed(self) -> "ks.Series":
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import first_series
        return first_series(DataFrame(self._data._internal.resolved_copy))

class SparkIndexMethods(SparkIndexOpsMethods):
    def transform(self, func: Callable[[Column], Column]) -> "ks.Index":
        return cast("ks.Index", super().transform(func))
    transform.__doc__ = SparkIndexOpsMethods.transform.__doc__

    @property
    def analyzed(self) -> "ks.Index":
        from databricks.koalas.frame import DataFrame
        return DataFrame(self._data._internal.resolved_copy).index

class SparkFrameMethods:
    def __init__(self, frame: "ks.DataFrame") -> None:
        self._kdf = frame

    def schema(self, index_col: Optional[Union[str, List[str]]] = None) -> StructType:
        return self.frame(index_col).schema

    def print_schema(self, index_col: Optional[Union[str, List[str]]] = None) -> None:
        self.frame(index_col).printSchema()

    def frame(self, index_col: Optional[Union[str, List[str]]] = None) -> SparkDataFrame:
        from databricks.koalas.utils import name_like_string
        kdf = self._kdf
        data_column_names: List[str] = []
        data_columns: List[Column] = []
        for i, (label, spark_column, column_name) in enumerate(zip(kdf._internal.column_labels,
                                                                    kdf._internal.data_spark_columns,
                                                                    kdf._internal.data_spark_column_names)):
            name: str = str(i) if label is None else name_like_string(label)
            data_column_names.append(name)
            if column_name != name:
                spark_column = spark_column.alias(name)
            data_columns.append(spark_column)
        if index_col is None:
            return kdf._internal.spark_frame.select(data_columns)
        else:
            if isinstance(index_col, str):
                index_col = [index_col]
            old_index_scols: List[Column] = kdf._internal.index_spark_columns
            if len(index_col) != len(old_index_scols):
                raise ValueError("length of index columns is %s; however, the length of the given 'index_col' is %s." % (len(old_index_scols), len(index_col)))
            if any((col in data_column_names for col in index_col)):
                raise ValueError("'index_col' cannot be overlapped with other columns.")
            new_index_scols: List[Column] = [index_scol.alias(col) for index_scol, col in zip(old_index_scols, index_col)]
            return kdf._internal.spark_frame.select(new_index_scols + data_columns)

    def cache(self) -> "CachedDataFrame":
        from databricks.koalas.frame import CachedDataFrame
        self._kdf._update_internal_frame(self._kdf._internal.resolved_copy, requires_same_anchor=False)
        return CachedDataFrame(self._kdf._internal)

    def persist(self, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> "CachedDataFrame":
        from databricks.koalas.frame import CachedDataFrame
        self._kdf._update_internal_frame(self._kdf._internal.resolved_copy, requires_same_anchor=False)
        return CachedDataFrame(self._kdf._internal, storage_level=storage_level)

    def hint(self, name: str, *parameters: Any) -> "DataFrame":
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        return DataFrame(internal.with_new_sdf(internal.spark_frame.hint(name, *parameters)))

    def to_table(self,
                 name: str,
                 format: Optional[str] = None,
                 mode: str = 'overwrite',
                 partition_cols: Optional[Union[str, List[str]]] = None,
                 index_col: Optional[Union[str, List[str]]] = None,
                 **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self._kdf.spark.frame(index_col=index_col).write.saveAsTable(name=name,
                                                                      format=format,
                                                                      mode=mode,
                                                                      partitionBy=partition_cols,
                                                                      **options)

    def to_spark_io(self,
                    path: Optional[str] = None,
                    format: Optional[str] = None,
                    mode: str = 'overwrite',
                    partition_cols: Optional[Union[str, List[str]]] = None,
                    index_col: Optional[Union[str, List[str]]] = None,
                    **options: Any) -> None:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self._kdf.spark.frame(index_col=index_col).write.save(path=path,
                                                               format=format,
                                                               mode=mode,
                                                               partitionBy=partition_cols,
                                                               **options)

    def explain(self, extended: Optional[Union[bool, str]] = None, mode: Optional[str] = None) -> None:
        if LooseVersion(pyspark.__version__) < LooseVersion('3.0'):
            if mode is not None and extended is not None:
                raise Exception('extended and mode should not be set together.')
            if extended is not None and isinstance(extended, str):
                mode = extended
            if mode is not None:
                if mode == 'simple':
                    extended = False
                elif mode == 'extended':
                    extended = True
                else:
                    raise ValueError("Unknown spark.explain mode: {}. Accepted spark.explain modes are 'simple', 'extended'.".format(mode))
            if extended is None:
                extended = False
            self._kdf._internal.to_internal_spark_frame.explain(extended)
        else:
            self._kdf._internal.to_internal_spark_frame.explain(extended, mode)

    def apply(self, func: Callable[[SparkDataFrame], SparkDataFrame],
              index_col: Optional[Union[str, List[str]]] = None) -> "ks.DataFrame":
        output: Any = func(self.frame(index_col))
        if not isinstance(output, SparkDataFrame):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.DataFrame; however, got [%s].' % (func, type(output)))
        return output.to_koalas(index_col)

    def repartition(self, num_partitions: int) -> "DataFrame":
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        repartitioned_sdf: SparkDataFrame = internal.spark_frame.repartition(num_partitions)
        return DataFrame(internal.with_new_sdf(repartitioned_sdf))

    def coalesce(self, num_partitions: int) -> "DataFrame":
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        coalesced_sdf: SparkDataFrame = internal.spark_frame.coalesce(num_partitions)
        return DataFrame(internal.with_new_sdf(coalesced_sdf))

    def checkpoint(self, eager: bool = True) -> "DataFrame":
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        checkpointed_sdf: SparkDataFrame = internal.spark_frame.checkpoint(eager)
        return DataFrame(internal.with_new_sdf(checkpointed_sdf))

    def local_checkpoint(self, eager: bool = True) -> "DataFrame":
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        checkpointed_sdf: SparkDataFrame = internal.spark_frame.localCheckpoint(eager)
        return DataFrame(internal.with_new_sdf(checkpointed_sdf))

    @property
    def analyzed(self) -> "DataFrame":
        from databricks.koalas.frame import DataFrame
        return DataFrame(self._kdf._internal.resolved_copy)

class CachedSparkFrameMethods(SparkFrameMethods):
    def __init__(self, frame: "ks.DataFrame") -> None:
        super().__init__(frame)

    @property
    def storage_level(self) -> StorageLevel:
        return self._kdf._cached.storageLevel

    def unpersist(self) -> None:
        if self._kdf._cached.is_cached:
            self._kdf._cached.unpersist()