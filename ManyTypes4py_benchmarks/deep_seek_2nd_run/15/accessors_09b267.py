"""
Spark related features. Usually, the features here are missing in pandas
but Spark has it.
"""
from abc import ABCMeta, abstractmethod
from distutils.version import LooseVersion
from typing import TYPE_CHECKING, Optional, Union, List, cast, Any, Callable, Dict, TypeVar, Generic
import pyspark
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame as SparkDataFrame
from pyspark.sql.types import DataType, StructType

if TYPE_CHECKING:
    import databricks.koalas as ks
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import CachedDataFrame

T = TypeVar('T')

class SparkIndexOpsMethods(Generic[T], metaclass=ABCMeta):
    """Spark related features. Usually, the features here are missing in pandas
    but Spark has it."""

    def __init__(self, data: T) -> None:
        self._data = data

    @property
    def data_type(self) -> DataType:
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        return self._data._internal.spark_type_for(self._data._column_label)

    @property
    def nullable(self) -> bool:
        """ Returns the nullability as defined by Spark. """
        return self._data._internal.spark_column_nullable_for(self._data._column_label)

    @property
    def column(self) -> Column:
        """
        Spark Column object representing the Series/Index.

        .. note:: This Spark Column object is strictly stick to its base DataFrame the Series/Index
            was derived from.
        """
        return self._data._internal.spark_column_for(self._data._column_label)

    def transform(self, func: Callable[[Column], Column]) -> T:
        """
        Applies a function that takes and returns a Spark column. It allows to natively
        apply a Spark function and column APIs with the Spark column internally used
        in Series or Index. The output length of the Spark column should be same as input's.

        .. note:: It requires to have the same input and output length; therefore,
            the aggregate Spark functions such as count does not work.

        Parameters
        ----------
        func : function
            Function to use for transforming the data by using Spark columns.

        Returns
        -------
        Series or Index

        Raises
        ------
        ValueError : If the output from the function is not a Spark column.
        """
        from databricks.koalas import MultiIndex
        if isinstance(self._data, MultiIndex):
            raise NotImplementedError('MultiIndex does not support spark.transform yet.')
        output = func(self._data.spark.column)
        if not isinstance(output, Column):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.Column; however, got [%s].' % (func, type(output)))
        new_ser = self._data._with_new_scol(scol=output)
        new_ser._internal.to_internal_spark_frame
        return new_ser

    @property
    @abstractmethod
    def analyzed(self) -> T:
        pass

class SparkSeriesMethods(SparkIndexOpsMethods['ks.Series']):

    def transform(self, func: Callable[[Column], Column]) -> 'ks.Series':
        return cast('ks.Series', super().transform(func))
    transform.__doc__ = SparkIndexOpsMethods.transform.__doc__

    def apply(self, func: Callable[[Column], Column]) -> 'ks.Series':
        """
        Applies a function that takes and returns a Spark column. It allows to natively
        apply a Spark function and column APIs with the Spark column internally used
        in Series or Index.

        .. note:: It forces to lose the index and end up with using default index. It is
            preferred to use :meth:`Series.spark.transform` or `:meth:`DataFrame.spark.apply`
            with specifying the `inedx_col`.

        .. note:: It does not require to have the same length of the input and output.
            However, it requires to create a new DataFrame internally which will require
            to set `compute.ops_on_diff_frames` to compute even with the same origin
            DataFrame that is expensive, whereas :meth:`Series.spark.transform` does not
            require it.

        Parameters
        ----------
        func : function
            Function to apply the function against the data by using Spark columns.

        Returns
        -------
        Series

        Raises
        ------
        ValueError : If the output from the function is not a Spark column.
        """
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, first_series
        from databricks.koalas.internal import HIDDEN_COLUMNS
        output = func(self._data.spark.column)
        if not isinstance(output, Column):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.Column; however, got [%s].' % (func, type(output)))
        assert isinstance(self._data, Series)
        sdf = self._data._internal.spark_frame.drop(*HIDDEN_COLUMNS).select(output)
        return first_series(DataFrame(sdf)).rename(self._data.name)

    @property
    def analyzed(self) -> 'ks.Series':
        """
        Returns a new Series with the analyzed Spark DataFrame.

        After multiple operations, the underlying Spark plan could grow huge
        and make the Spark planner take a long time to finish the planning.

        This function is for the workaround to avoid it.

        .. note:: After analyzed, operations between the analyzed Series and the original one
            will **NOT** work without setting a config `compute.ops_on_diff_frames` to `True`.

        Returns
        -------
        Series
        """
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import first_series
        return first_series(DataFrame(self._data._internal.resolved_copy))

class SparkIndexMethods(SparkIndexOpsMethods['ks.Index']):

    def transform(self, func: Callable[[Column], Column]) -> 'ks.Index':
        return cast('ks.Index', super().transform(func))
    transform.__doc__ = SparkIndexOpsMethods.transform.__doc__

    @property
    def analyzed(self) -> 'ks.Index':
        """
        Returns a new Index with the analyzed Spark DataFrame.

        After multiple operations, the underlying Spark plan could grow huge
        and make the Spark planner take a long time to finish the planning.

        This function is for the workaround to avoid it.

        .. note:: After analyzed, operations between the analyzed Series and the original one
            will **NOT** work without setting a config `compute.ops_on_diff_frames` to `True`.

        Returns
        -------
        Index
        """
        from databricks.koalas.frame import DataFrame
        return DataFrame(self._data._internal.resolved_copy).index

class SparkFrameMethods(object):
    """Spark related features. Usually, the features here are missing in pandas
    but Spark has it."""

    def __init__(self, frame: 'ks.DataFrame') -> None:
        self._kdf = frame

    def schema(self, index_col: Optional[Union[str, List[str]]] = None) -> StructType:
        """
        Returns the underlying Spark schema.

        Returns
        -------
        pyspark.sql.types.StructType
            The underlying Spark schema.

        Parameters
        ----------
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        """
        return self.frame(index_col).schema

    def print_schema(self, index_col: Optional[Union[str, List[str]]] = None) -> None:
        """
        Prints out the underlying Spark schema in the tree format.

        Parameters
        ----------
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.

        Returns
        -------
        None
        """
        self.frame(index_col).printSchema()

    def frame(self, index_col: Optional[Union[str, List[str]]] = None) -> SparkDataFrame:
        """
        Return the current DataFrame as a Spark DataFrame.  :meth:`DataFrame.spark.frame` is an
        alias of  :meth:`DataFrame.to_spark`.

        Parameters
        ----------
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        """
        from databricks.koalas.utils import name_like_string
        kdf = self._kdf
        data_column_names = []
        data_columns = []
        for i, (label, spark_column, column_name) in enumerate(zip(kdf._internal.column_labels, kdf._internal.data_spark_columns, kdf._internal.data_spark_column_names)):
            name = str(i) if label is None else name_like_string(label)
            data_column_names.append(name)
            if column_name != name:
                spark_column = spark_column.alias(name)
            data_columns.append(spark_column)
        if index_col is None:
            return kdf._internal.spark_frame.select(data_columns)
        else:
            if isinstance(index_col, str):
                index_col = [index_col]
            old_index_scols = kdf._internal.index_spark_columns
            if len(index_col) != len(old_index_scols):
                raise ValueError("length of index columns is %s; however, the length of the given 'index_col' is %s." % (len(old_index_scols), len(index_col)))
            if any((col in data_column_names for col in index_col)):
                raise ValueError("'index_col' cannot be overlapped with other columns.")
            new_index_scols = [index_scol.alias(col) for index_scol, col in zip(old_index_scols, index_col)]
            return kdf._internal.spark_frame.select(new_index_scols + data_columns)

    def cache(self) -> 'CachedDataFrame':
        """
        Yields and caches the current DataFrame.

        The Koalas DataFrame is yielded as a protected resource and its corresponding
        data is cached which gets uncached after execution goes of the context.

        If you want to specify the StorageLevel manually, use :meth:`DataFrame.spark.persist`

        See Also
        --------
        DataFrame.spark.persist
        """
        from databricks.koalas.frame import CachedDataFrame
        self._kdf._update_internal_frame(self._kdf._internal.resolved_copy, requires_same_anchor=False)
        return CachedDataFrame(self._kdf._internal)

    def persist(self, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> 'CachedDataFrame':
        """
        Yields and caches the current DataFrame with a specific StorageLevel.
        If a StogeLevel is not given, the `MEMORY_AND_DISK` level is used by default like PySpark.

        The Koalas DataFrame is yielded as a protected resource and its corresponding
        data is cached which gets uncached after execution goes of the context.

        See Also
        --------
        DataFrame.spark.cache
        """
        from databricks.koalas.frame import CachedDataFrame
        self._kdf._update_internal_frame(self._kdf._internal.resolved_copy, requires_same_anchor=False)
        return CachedDataFrame(self._kdf._internal, storage_level=storage_level)

    def hint(self, name: str, *parameters: Any) -> 'ks.DataFrame':
        """
        Specifies some hint on the current DataFrame.

        Parameters
        ----------
        name : A name of the hint.
        parameters : Optional parameters.

        Returns
        -------
        ret : DataFrame with the hint.
        """
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        return DataFrame(internal.with_new_sdf(internal.spark_frame.hint(name, *parameters)))

    def to_table(self, name: str, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        """
        Write the DataFrame into a Spark table. :meth:`DataFrame.spark.to_table`
        is an alias of :meth:`DataFrame.to_table`.

        Parameters
        ----------
        name : str, required
            Table name in Spark.
        format : string, optional
            Specifies the output data source format.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when the table exists.
        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index.
        options
            Additional options passed directly to Spark.
        """
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self._kdf.spark.frame(index_col=index_col).write.saveAsTable(name=name, format=format, mode=mode, partitionBy=partition_cols, **options)

    def to_spark_io(self, path: Optional[str] = None, format: Optional[str] = None, mode: str = 'overwrite', partition_cols: Optional[Union[str, List[str]]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> None:
        """Write the DataFrame out to a Spark data source. :meth:`DataFrame.spark.to_spark_io`
        is an alias of :meth:`DataFrame.to_spark_io`.

        Parameters
        ----------
        path : string, optional
            Path to the data source.
        format : string, optional
            Specifies the output data source format.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when data already.
        partition_cols : str or list of str, optional
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index.
        options : dict
            All other options passed directly into Spark's data source.
        """
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        self._kdf.spark.frame(index_col=index_col).write.save(path=path, format=format, mode=mode, partitionBy=partition_cols, **options)

    def explain(self, extended: Optional[Union[bool, str]] = None, mode: Optional[str] = None) -> None:
        """
        Prints the underlying (logical and physical) Spark plans to the console for debugging
        purpose.

        Parameters
        ----------
        extended : boolean, default ``False``.
            If ``False``, prints only the physical plan.
        mode : string, default ``None``.
            The expected output format of plans.
        """
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

    def apply(self, func: Callable[[SparkDataFrame], SparkDataFrame], index_col: Optional[Union[str, List[str]]] = None) -> 'ks.DataFrame':
        """
        Applies a function that takes and returns a Spark DataFrame. It allows natively
        apply a Spark function and column APIs with the Spark column internally used
        in Series or Index.

        Parameters
        ----------
        func : function
            Function to apply the function against the data by using Spark DataFrame.

        Returns
        -------
        DataFrame

        Raises
        ------
        ValueError : If the output from the function is not a Spark DataFrame.
        """
        output = func(self.frame(index_col))
        if not isinstance(output, SparkDataFrame):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.DataFrame; however, got [%s].' % (func, type(output)))
        return output.to_koalas(index_col)

    def repartition(self, num_partitions: int) -> 'ks.DataFrame':
        """
        Returns a new DataFrame partitioned by the given partitioning expressions. The
        resulting DataFrame is hash partitioned.

        Parameters
        ----------
        num_partitions : int
            The target number of partitions.

        Returns
        -------
        DataFrame
        """
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        repartitioned_sdf = internal.spark_frame.repartition(num_partitions)
        return DataFrame(internal.with_new_sdf(repartitioned_sdf))

    def coalesce(self, num_partitions: int) -> 'ks.DataFrame':
        """
        Returns a new DataFrame that has exactly `num_partitions` partitions.

        Parameters
        ----------
        num_partitions : int
            The target number of partitions.

        Returns
        -------
        DataFrame
        """
        from databricks.koalas.frame import DataFrame
        internal = self._kdf._internal.resolved_copy
        coalesced_sdf = internal.spark_frame.coales