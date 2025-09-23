"""
Spark related features. Usually, the features here are missing in pandas
but Spark has it.
"""
from abc import ABCMeta, abstractmethod
from distutils.version import LooseVersion
from typing import TYPE_CHECKING, Optional, Union, List, cast, Any, Callable, Iterator
import pyspark
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame as SparkDataFrame
from pyspark.sql.types import DataType, StructType
if TYPE_CHECKING:
    import databricks.koalas as ks
    from databricks.koalas.base import IndexOpsMixin
    from databricks.koalas.frame import CachedDataFrame

class SparkIndexOpsMethods(object, metaclass=ABCMeta):
    """Spark related features. Usually, the features here are missing in pandas
    but Spark has it."""

    def __init__(self, data: "IndexOpsMixin") -> None:
        self._data: "IndexOpsMixin" = data

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

    def transform(self, func: Callable[[Column], Column]) -> "IndexOpsMixin":
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

        Examples
        --------
        >>> from pyspark.sql.functions import log
        >>> df = ks.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, columns=["a", "b"])
        >>> df
           a  b
        0  1  4
        1  2  5
        2  3  6

        >>> df.a.spark.transform(lambda c: log(c))
        0    0.000000
        1    0.693147
        2    1.098612
        Name: a, dtype: float64

        >>> df.index.spark.transform(lambda c: c + 10)
        Int64Index([10, 11, 12], dtype='int64')

        >>> df.a.spark.transform(lambda c: c + df.b.spark.column)
        0    5
        1    7
        2    9
        Name: a, dtype: int64
        """
        from databricks.koalas import MultiIndex
        if isinstance(self._data, MultiIndex):
            raise NotImplementedError('MultiIndex does not support spark.transform yet.')
        output: Column = func(self._data.spark.column)
        if not isinstance(output, Column):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.Column; however, got [%s].' % (func, type(output)))
        new_ser: "IndexOpsMixin" = self._data._with_new_scol(scol=output)
        new_ser._internal.to_internal_spark_frame
        return new_ser

    @property
    @abstractmethod
    def analyzed(self) -> "IndexOpsMixin":
        pass

class SparkSeriesMethods(SparkIndexOpsMethods):

    def transform(self, func: Callable[[Column], Column]) -> "ks.Series":
        return cast('ks.Series', super().transform(func))
    transform.__doc__ = SparkIndexOpsMethods.transform.__doc__

    def apply(self, func: Callable[[Column], Column]) -> "ks.Series":
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

        Examples
        --------
        >>> from databricks import koalas as ks
        >>> from pyspark.sql.functions import count, lit
        >>> df = ks.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, columns=["a", "b"])
        >>> df
           a  b
        0  1  4
        1  2  5
        2  3  6

        >>> df.a.spark.apply(lambda c: count(c))
        0    3
        Name: a, dtype: int64

        >>> df.a.spark.apply(lambda c: c + df.b.spark.column)
        0    5
        1    7
        2    9
        Name: a, dtype: int64
        """
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, first_series
        from databricks.koalas.internal import HIDDEN_COLUMNS
        output: Column = func(self._data.spark.column)
        if not isinstance(output, Column):
            raise ValueError('The output of the function [%s] should be of a pyspark.sql.Column; however, got [%s].' % (func, type(output)))
        assert isinstance(self._data, Series)
        sdf: SparkDataFrame = self._data._internal.spark_frame.drop(*HIDDEN_COLUMNS).select(output)
        return first_series(DataFrame(sdf)).rename(self._data.name)

    @property
    def analyzed(self) -> "ks.Series":
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

        Examples
        --------
        >>> ser = ks.Series([1, 2, 3])
        >>> ser
        0    1
        1    2
        2    3
        dtype: int64

        The analyzed one should return the same value.

        >>> ser.spark.analyzed
        0    1
        1    2
        2    3
        dtype: int64

        However, it won't work with the same anchor Series.

        >>> ser + ser.spark.analyzed
        Traceback (most recent call last):
        ...
        ValueError: ... enable 'compute.ops_on_diff_frames' option.

        >>> with ks.option_context('compute.ops_on_diff_frames', True):
        ...     (ser + ser.spark.analyzed).sort_index()
        0    2
        1    4
        2    6
        dtype: int64
        """
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import first_series
        return first_series(DataFrame(self._data._internal.resolved_copy))

class SparkIndexMethods(SparkIndexOpsMethods):

    def transform(self, func: Callable[[Column], Column]) -> "ks.Index":
        return cast('ks.Index', super().transform(func))
    transform.__doc__ = SparkIndexOpsMethods.transform.__doc__

    @property
    def analyzed(self) -> "ks.Index":
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

        Examples
        --------
        >>> idx = ks.Index([1, 2, 3])
        >>> idx
        Int64Index([1, 2, 3], dtype='int64')

        The analyzed one should return the same value.

        >>> idx.spark.analyzed
        Int64Index([1, 2, 3], dtype='int64')

        However, it won't work with the same anchor Index.

        >>> idx + idx.spark.analyzed
        Traceback (most recent call last):
        ...
        ValueError: ... enable 'compute.ops_on_diff_frames' option.

        >>> with ks.option_context('compute.ops_on_diff_frames', True):
        ...     (idx + idx.spark.analyzed).sort_values()
        Int64Index([2, 4, 6], dtype='int64')
        """
        from databricks.koalas.frame import DataFrame
        return DataFrame(self._data._internal.resolved_copy).index

class SparkFrameMethods(object):
    """Spark related features. Usually, the features here are missing in pandas
    but Spark has it."""

    def __init__(self, frame: "ks.DataFrame") -> None:
        self._kdf: "ks.DataFrame" = frame

    def schema(self, index_col: Optional[Union[str, List[str]]]=None) -> StructType:
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

        Examples
        --------
        >>> df = ks.DataFrame({'a': list('abc'),
        ...                    'b': list(range(1, 4)),
        ...                    'c': np.arange(3, 6).astype('i1'),
        ...                    'd': np.arange(4.0, 7.0, dtype='float64'),
        ...                    'e': [True, False, True],
        ...                    'f': pd.date_range('20130101', periods=3)},
        ...                   columns=['a', 'b', 'c', 'd', 'e', 'f'])
        >>> df.spark.schema().simpleString()
        'struct<a:string,b:bigint,c:tinyint,d:double,e:boolean,f:timestamp>'
        >>> df.spark.schema(index_col='index').simpleString()
        'struct<index:bigint,a:string,b:bigint,c:tinyint,d:double,e:boolean,f:timestamp>'
        """
        return self.frame(index_col).schema

    def print_schema(self, index_col: Optional[Union[str, List[str]]]=None) -> None:
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

        Examples
        --------
        >>> df = ks.DataFrame({'a': list('abc'),
        ...                    'b': list(range(1, 4)),
        ...                    'c': np.arange(3, 6).astype('i1'),
        ...                    'd': np.arange(4.0, 7.0, dtype='float64'),
        ...                    'e': [True, False, True],
        ...                    'f': pd.date_range('20130101', periods=3)},
        ...                   columns=['a', 'b', 'c', 'd', 'e', 'f'])
        >>> df.spark.print_schema()  # doctest: +NORMALIZE_WHITESPACE
        root
         |-- a: string (nullable = false)
         |-- b: long (nullable = false)
         |-- c: byte (nullable = false)
         |-- d: double (nullable = false)
         |-- e: boolean (nullable = false)
         |-- f: timestamp (nullable = false)
        >>> df.spark.print_schema(index_col='index')  # doctest: +NORMALIZE_WHITESPACE
        root
         |-- index: long (nullable = false)
         |-- a: string (nullable = false)
         |-- b: long (nullable = false)
         |-- c: byte (nullable = false)
         |-- d: double (nullable = false)
         |-- e: boolean (nullable = false)
         |-- f: timestamp (nullable = false)
        """
        self.frame(index_col).printSchema()

    def frame(self, index_col: Optional[Union[str, List[str]]]=None) -> SparkDataFrame:
        """
        Return the current DataFrame as a Spark DataFrame.  :meth:`DataFrame.spark.frame` is an
        alias of  :meth:`DataFrame.to_spark`.

        Parameters
        ----------
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.

        See Also
        --------
        DataFrame.to_spark
        DataFrame.to_koalas
        DataFrame.spark.frame

        Examples
        --------
        By default, this method loses the index as below.

        >>> df = ks.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        >>> df.to_spark().show()  # doctest: +NORMALIZE_WHITESPACE
        +---+---+---+
        |  a|  b|  c|
        +---+---+---+
        |  1|  4|  7|
        |  2|  5|  8|
        |  3|  6|  9|
        +---+---+---+

        >>> df = ks.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        >>> df.spark.frame().show()  # doctest: +NORMALIZE_WHITESPACE
        +---+---+---+
        |  a|  b|  c|
        +---+---+---+
        |  1|  4|  7|
        |  2|  5|  8|
        |  3|  6|  9|
        +---+---+---+

        If `index_col` is set, it keeps the index column as specified.

        >>> df.to_spark(index_col="index").show()  # doctest: +NORMALIZE_WHITESPACE
        +-----+---+---+---+
        |index|  a|  b|  c|
        +-----+---+---+---+
        |    0|  1|  4|  7|
        |    1|  2|  5|  8|
        |    2|  3|  6|  9|
        +-----+---+---+---+

        Keeping index column is useful when you want to call some Spark APIs and
        convert it back to Koalas DataFrame without creating a default index, which
        can affect performance.

        >>> spark_df = df.to_spark(index_col="index")
        >>> spark_df = spark_df.filter("a == 2")
        >>> spark_df.to_koalas(index_col="index")  # doctest: +NORMALIZE_WHITESPACE
               a  b  c
        index
        1      2  5  8

        In case of multi-index, specify a list to `index_col`.

        >>> new_df = df.set_index("a", append=True)
        >>> new_spark_df = new_df.to_spark(index_col=["index_1", "index_2"])
        >>> new_spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
        +-------+-------+---+---+
        |index_1|index_2|  b|  c|
        +-------+-------+---+---+
        |      0|      1|  4|  7|
        |      1|      2|  5|  8|
        |      2|      3|  6|  9|
        +-------+-------+---+---+

        Likewise, can be converted to back to Koalas DataFrame.

        >>> new_spark_df.to_koalas(
        ...     index_col=["index_1", "index_2"])  # doctest: +NORMALIZE_WHITESPACE
                         b  c
        index_1 index_2
        0       1        4  7
        1       2        5  8
        2       3        6  9
        """
        from databricks.koalas.utils import name_like_string
        kdf: "ks.DataFrame" = self._kdf
        data_column_names: List[str] = []
       