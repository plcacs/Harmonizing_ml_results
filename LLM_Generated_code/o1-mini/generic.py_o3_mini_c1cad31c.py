#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A base class of DataFrame/Column to behave similar to pandas DataFrame/Series.
"""
from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING, cast
import warnings

import numpy as np  # noqa: F401
import pandas as pd
from pandas.api.types import is_list_like

import pyspark
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegralType,
    LongType,
    NumericType,
)

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer
from databricks.koalas.internal import InternalFrame
from databricks.koalas.spark import functions as SF
from databricks.koalas.typedef import Scalar, spark_type_to_pandas_dtype
from databricks.koalas.utils import (
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    scol_for,
    sql_conf,
    validate_arguments_and_invoke_function,
    validate_axis,
    SPARK_CONF_ARROW_ENABLED,
)
from databricks.koalas.window import Rolling, Expanding

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
    from databricks.koalas.series import Series


class Frame(object, metaclass=ABCMeta):
    """
    The base class for both DataFrame and Series.
    """

    @abstractmethod
    def __getitem__(self, key: Any) -> Union["DataFrame", "Series"]:
        pass

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @abstractmethod
    def _apply_series_op(
        self, op: Any, should_resolve: bool = False
    ) -> Union["Series", "DataFrame"]:
        pass

    @abstractmethod
    def _reduce_for_stat_function(
        self,
        sfun: Any,
        name: str,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = True,
        **kwargs: Any,
    ) -> Union[Scalar, "Series"]:
        pass

    @property
    @abstractmethod
    def dtypes(self) -> Union[pd.Series, Any]:
        pass

    @abstractmethod
    def to_pandas(self) -> Union[pd.DataFrame, pd.Series]:
        pass

    @property
    @abstractmethod
    def index(self) -> Any:
        pass

    @abstractmethod
    def copy(self) -> Union["DataFrame", "Series"]:
        pass

    @abstractmethod
    def _to_internal_pandas(self) -> Union[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def head(self, n: int = 5) -> Union["DataFrame", "Series"]:
        pass

    # TODO: add 'axis' parameter
    def cummin(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        """
        Return cumulative minimum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative minimum.

        .. note:: the current implementation of cummin uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.min : Return the minimum over DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        Series.min : Return the minimum over Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [1.0, 0.0]], columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the minimum in each column.

        >>> df.cummin()
             A    B
        0  2.0  1.0
        1  2.0  NaN
        2  1.0  0.0

        It works identically in Series.

        >>> df.A.cummin()
        0    2.0
        1    2.0
        2    1.0
        Name: A, dtype: float64
        """
        return self._apply_series_op(lambda kser: kser._cum(F.min, skipna), should_resolve=True)

    # TODO: add 'axis' parameter
    def cummax(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        """
        Return cumulative maximum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative maximum.

        .. note:: the current implementation of cummax uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.max : Return the maximum over DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.
        Series.max : Return the maximum over Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [1.0, 0.0]], columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the maximum in each column.

        >>> df.cummax()
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  3.0  1.0

        It works identically in Series.

        >>> df.B.cummax()
        0    1.0
        1    NaN
        2    1.0
        Name: B, dtype: float64
        """
        return self._apply_series_op(lambda kser: kser._cum(F.max, skipna), should_resolve=True)

    # TODO: add 'axis' parameter
    def cumsum(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        """
        Return cumulative sum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative sum.

        .. note:: the current implementation of cumsum uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.sum : Return the sum over DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.
        Series.sum : Return the sum over Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [1.0, 0.0]], columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the sum in each column.

        >>> df.cumsum()
             A    B
        0  2.0  1.0
        1  5.0  NaN
        2  6.0  1.0

        It works identically in Series.

        >>> df.A.cumsum()
        0    2.0
        1    5.0
        2    6.0
        Name: A, dtype: float64
        """
        return self._apply_series_op(lambda kser: kser._cumsum(skipna), should_resolve=True)

    # TODO: add 'axis' parameter
    # TODO: use pandas_udf to support negative values and other options later
    #  other window except unbounded ones is supported as of Spark 3.0.
    def cumprod(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        """
        Return cumulative product over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative product.

        .. note:: the current implementation of cumprod uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        .. note:: unlike pandas', Koalas' emulates cumulative product by ``exp(sum(log(...)))``
            trick. Therefore, it only works for positive numbers.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Raises
        ------
        Exception : If the values is equal to or lower than 0.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [4.0, 10.0]], columns=list('AB'))
        >>> df
             A     B
        0  2.0   1.0
        1  3.0   NaN
        2  4.0  10.0

        By default, iterates over rows and finds the sum in each column.

        >>> df.cumprod()
              A     B
        0   2.0   1.0
        1   6.0   NaN
        2  24.0  10.0

        It works identically in Series.

        >>> df.A.cumprod()
        0     2.0
        1     6.0
        2    24.0
        Name: A, dtype: float64
        """
        return self._apply_series_op(lambda kser: kser._cumprod(skipna), should_resolve=True)

    # TODO: Although this has removed pandas >= 1.0.0, but we're keeping this as deprecated
    # since we're using this for `DataFrame.info` internally.
    # We can drop it once our minimal pandas version becomes 1.0.0.
    def get_dtype_counts(self) -> pd.Series:
        """
        Return counts of unique dtypes in this object.

        .. deprecated:: 0.14.0

        Returns
        -------
        dtype : pd.Series
            Series with the count of columns with each dtype.

        See Also
        --------
        dtypes : Return the dtypes in this object.

        Examples
        --------
        >>> a = [['a', 1, 1], ['b', 2, 2], ['c', 3, 3]]
        >>> df = ks.DataFrame(a, columns=['str', 'int1', 'int2'])
        >>> df
          str  int1  int2
        0   a     1     1
        1   b     2     2
        2   c     3     3

        >>> df.get_dtype_counts().sort_values()
        object    1
        int64     2
        dtype: int64

        >>> df.str.get_dtype_counts().sort_values()
        object    1
        dtype: int64
        """
        warnings.warn(
            "`get_dtype_counts` has been deprecated and will be "
            "removed in a future version. For DataFrames use "
            "`.dtypes.value_counts()",
            FutureWarning,
        )
        if not isinstance(self.dtypes, Iterable):
            dtypes = [self.dtypes]
        else:
            dtypes = list(self.dtypes)
        return pd.Series(dict(Counter([d.name for d in dtypes])))

    def pipe(
        self,
        func: Union[Any, Tuple[Any, str]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        r"""
        Apply func(self, \*args, \*\*kwargs).

        Parameters
        ----------
        func : function
            function to apply to the DataFrame.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the data.
        args : Iterable[Any], optional
            Positional arguments passed into ``func``.
        kwargs : Mapping[str, Any], optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects. For example, given

        >>> df = ks.DataFrame({'category': ['A', 'A', 'B'],
        ...                    'col1': [1, 2, 3],
        ...                    'col2': [4, 5, 6]},
        ...                   columns=['category', 'col1', 'col2'])
        >>> def keep_category_a(df: "DataFrame") -> "DataFrame":
        ...     return df[df['category'] == 'A']
        >>> def add_one(df: "DataFrame", column: str) -> "DataFrame":
        ...     return df.assign(col3=df[column] + 1)
        >>> def multiply(df: "DataFrame", column1: str, column2: str) -> "DataFrame":
        ...     return df.assign(col4=df[column1] * df[column2])

        instead of writing

        >>> multiply(add_one(keep_category_a(df), column="col1"), column1="col2", column2="col3")
          category  col1  col2  col3  col4
        0        A     1     4     2     8
        1        A     2     5     3    15


        You can write

        >>> (df.pipe(keep_category_a)
        ...    .pipe(add_one, column="col1")
        ...    .pipe(multiply, column1="col2", column2="col3")
        ... )
          category  col1  col2  col3  col4
        0        A     1     4     2     8
        1        A     2     5     3    15


        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``df``:

        >>> def multiply_2(column1: str, df: "DataFrame", column2: str) -> "DataFrame":
        ...     return df.assign(col4=df[column1] * df[column2])

        Then you can write

        >>> (df.pipe(keep_category_a)
        ...    .pipe(add_one, column="col1")
        ...    .pipe((multiply_2, 'df'), column1="col2", column2="col3")
        ... )
          category  col1  col2  col3  col4
        0        A     1     4     2     8
        1        A     2     5     3    15

        You can use lambda as well

        >>> ks.Series([1, 2, 3]).pipe(lambda x: (x + 1).rename("value"))
        0    2
        1    3
        2    4
        Name: value, dtype: int64
        """

        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError("%s is both the pipe target and a keyword " "argument" % target)
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    def to_numpy(self) -> np.ndarray:
        """
        A NumPy ndarray representing the values in this DataFrame or Series.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> ks.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
        array([[1, 3],
               [2, 4]])

        With heterogeneous data, the lowest common type will have to be used.

        >>> ks.DataFrame({"A": [1, 2], "B": [3.0, 4.5]}).to_numpy()
        array([[1. , 3. ],
               [2. , 4.5]])

        For a mix of numeric and non-numeric types, the output array will have object dtype.

        >>> df = ks.DataFrame({"A": [1, 2], "B": [3.0, 4.5], "C": pd.date_range('2000', periods=2)})
        >>> df.to_numpy()
        array([[1, 3.0, Timestamp('2000-01-01 00:00:00')],
               [2, 4.5, Timestamp('2000-01-02 00:00:00')]], dtype=object)

        For Series,

        >>> ks.Series(['a', 'b', 'a']).to_numpy()
        array(['a', 'b', 'a'], dtype=object)
        """
        return self.to_pandas().values

    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of the DataFrame or the Series.

        .. warning:: We recommend using `DataFrame.to_numpy()` or `Series.to_numpy()` instead.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        A DataFrame where all columns are the same type (e.g., int64) results in an array of
        the same type.

        >>> df = ks.DataFrame({'age':    [ 3,  29],
        ...                    'height': [94, 170],
        ...                    'weight': [31, 115]})
        >>> df
           age  height  weight
        0    3      94      31
        1   29     170     115
        >>> df.dtypes
        age       int64
        height    int64
        weight    int64
        dtype: object
        >>> df.values
        array([[  3,  94,  31],
               [ 29, 170, 115]])

        A DataFrame with mixed type columns(e.g., str/object, int64, float32) results in an ndarray
        of the broadest type that accommodates these mixed types (e.g., object).

        >>> df2 = ks.DataFrame([('parrot',   24.0, 'second'),
        ...                     ('lion',     80.5, 'first'),
        ...                     ('monkey', np.nan, None)],
        ...                   columns=('name', 'max_speed', 'rank'))
        >>> df2.dtypes
        name          object
        max_speed    float64
        rank          object
        dtype: object
        >>> df2.values
        array([['parrot', 24.0, 'second'],
               ['lion', 80.5, 'first'],
               ['monkey', nan, None]], dtype=object)

        For Series,

        >>> ks.Series([1, 2, 3]).values
        array([1, 2, 3])

        >>> ks.Series(list('aabc')).values
        array(['a', 'a', 'b', 'c'], dtype=object)
        """
        warnings.warn("We recommend using `{}.to_numpy()` instead.".format(type(self).__name__))
        return self.to_numpy()

    def to_csv(
        self,
        path: Optional[str] = None,
        sep: str = ",",
        na_rep: str = "",
        columns: Optional[Union[List[str], List[Tuple[str, ...]]]] = None,
        header: Union[bool, List[str]] = True,
        quotechar: str = '"',
        date_format: Optional[str] = None,
        escapechar: Optional[str] = None,
        num_files: Optional[int] = None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any,
    ) -> Optional[str]:
        r"""
        Write object to a comma-separated values (csv) file.

        .. note:: Koalas `to_csv` writes files to a path or URI. Unlike pandas', Koalas
            respects HDFS's property such as 'fs.default.name'.

        .. note:: Koalas writes CSV files into the directory, `path`, and writes
            multiple `part-...` files in the directory when `path` is specified.
            This behaviour was inherited from Apache Spark. The number of files can
            be controlled by `num_files`.

        Parameters
        ----------
        path : Optional[str], default None
            File path. If None is provided the result is returned as a string.
        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        columns : Optional[Union[List[str], List[Tuple[str, ...]]]], optional
            Columns to write.
        header : Union[bool, List[str]], default True
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.
        quotechar : str, default '\"'
            String of length 1. Character used to quote fields.
        date_format : Optional[str], default None
            Format string for datetime objects.
        escapechar : Optional[str], default None
            String of length 1. Character used to escape `sep` and `quotechar`
            when appropriate.
        num_files : Optional[int], default None
            The number of files to be written in `path` directory when
            this is a path.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'},
            default 'overwrite'. Specifies the behavior of the save operation when the
            destination exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : Optional[Union[str, List[str]]], optional, default None
            Names of partitioning columns
        index_col: Optional[Union[str, List[str]]], optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options: Any
            Keyword arguments for additional options specific to PySpark.
            These kwargs are specific to PySpark's CSV options to pass. Check
            the options in PySpark's API documentation for spark.write.csv(...).
            It has higher priority and overwrites all other options.
            This parameter only works when `path` is specified.

        Returns
        -------
        Optional[str]

        See Also
        --------
        read_csv
        DataFrame.to_delta
        DataFrame.to_table
        DataFrame.to_parquet
        DataFrame.to_spark_io

        Examples
        --------
        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df.sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                           date country  code
        ... 2012-01-31 12:00:00      KR     1
        ... 2012-02-29 12:00:00      US     2
        ... 2012-03-31 12:00:00      JP     3

        >>> print(df.to_csv())  # doctest: +NORMALIZE_WHITESPACE
        date,country,code
        2012-01-31 12:00:00,KR,1
        2012-02-29 12:00:00,US,2
        2012-03-31 12:00:00,JP,3

        >>> df.cummax().to_csv(path=r'%s/to_csv/foo.csv' % path, num_files=1)
        >>> ks.read_csv(
        ...    path=r'%s/to_csv/foo.csv' % path
        ... ).sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                           date country  code
        ... 2012-01-31 12:00:00      KR     1
        ... 2012-02-29 12:00:00      US     2
        ... 2012-03-31 12:00:00      US     3

        In case of Series,

        >>> print(df.date.to_csv())  # doctest: +NORMALIZE_WHITESPACE
        date
        2012-01-31 12:00:00
        2012-02-29 12:00:00
        2012-03-31 12:00:00

        >>> df.date.to_csv(path=r'%s/to_csv/foo.csv' % path, num_files=1)
        >>> ks.read_csv(
        ...     path=r'%s/to_csv/foo.csv' % path
        ... ).sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                           date
        ... 2012-01-31 12:00:00
        ... 2012-02-29 12:00:00
        ... 2012-03-31 12:00:00

        You can preserve the index in the roundtrip as below.

        >>> df.set_index("country", append=True, inplace=True)
        >>> df.date.to_csv(
        ...     path=r'%s/to_csv/bar.csv' % path,
        ...     num_files=1,
        ...     index_col=["index1", "index2"])
        >>> ks.read_csv(
        ...     path=r'%s/to_csv/bar.csv' % path, index_col=["index1", "index2"]
        ... ).sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                                     date
        index1 index2
        ...    ...    2012-01-31 12:00:00
        ...    ...    2012-02-29 12:00:00
        ...    ...    2012-03-31 12:00:00
        """
        if "options" in kwargs and isinstance(kwargs.get("options"), dict) and len(kwargs) == 1:
            options = kwargs.get("options")  # type: ignore

        if path is None:
            # If path is none, just collect and use pandas's to_csv.
            kdf_or_ser = self
            if (LooseVersion("0.24") > LooseVersion(pd.__version__)) and isinstance(
                self, ks.Series
            ):
                # 0.23 seems not having 'columns' parameter in Series' to_csv.
                return cast(Optional[str], kdf_or_ser.to_pandas().to_csv(  # type: ignore
                    None,
                    sep=sep,
                    na_rep=na_rep,
                    header=header,
                    date_format=date_format,
                    index=False,
                ))
            else:
                return cast(Optional[str], kdf_or_ser.to_pandas().to_csv(  # type: ignore
                    None,
                    sep=sep,
                    na_rep=na_rep,
                    columns=columns,
                    header=header,
                    quotechar=quotechar,
                    date_format=date_format,
                    escapechar=escapechar,
                    index=False,
                ))

        kdf: "DataFrame"
        if isinstance(self, ks.Series):
            kdf = self.to_frame()

        if columns is None:
            column_labels = kdf._internal.column_labels
        else:
            column_labels: List[Tuple[str, ...]] = []
            for label in columns:
                if not is_name_like_tuple(label):
                    label = (label,)
                if label not in kdf._internal.column_labels:
                    raise KeyError(name_like_string(label))
                column_labels.append(label)

        if isinstance(index_col, str):
            index_cols: List[str] = [index_col]
        elif index_col is None:
            index_cols: List[str] = []
        else:
            index_cols = cast(List[str], index_col)

        if header is True and kdf._internal.column_labels_level > 1:
            raise ValueError("to_csv only support one-level index column now")
        elif isinstance(header, list):
            sdf: SparkDataFrame = kdf.to_spark(index_col)  # type: ignore
            sdf = sdf.select(
                [scol_for(sdf, name_like_string(label)) for label in index_cols]
                + [
                    scol_for(sdf, str(i) if label is None else name_like_string(label)).alias(
                        new_name
                    )
                    for i, (label, new_name) in enumerate(zip(column_labels, header))
                ]
            )
            header = True
        else:
            sdf = kdf.to_spark(index_col)  # type: ignore
            sdf = sdf.select(
                [scol_for(sdf, name_like_string(label)) for label in index_cols]
                + [
                    scol_for(sdf, str(i) if label is None else name_like_string(label))
                    for i, label in enumerate(column_labels)
                ]
            )

        if num_files is not None:
            sdf = sdf.repartition(num_files)

        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder = builder.partitionBy(partition_cols)
        builder._set_opts(
            sep=sep,
            nullValue=na_rep,
            header=header,
            quote=quotechar,
            dateFormat=date_format,
            charToEscapeQuoteEscaping=escapechar,
        )
        builder.options(**options).format("csv").save(path)
        return None

    def to_json(
        self,
        path: Optional[str] = None,
        compression: str = "uncompressed",
        num_files: Optional[int] = None,
        mode: str = "overwrite",
        orient: str = "records",
        lines: bool = True,
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options: Any,
    ) -> Optional[str]:
        """
        Convert the object to a JSON string.

        .. note:: Koalas `to_json` writes files to a path or URI. Unlike pandas', Koalas
            respects HDFS's property such as 'fs.default.name'.

        .. note:: Koalas writes JSON files into the directory, `path`, and writes
            multiple `part-...` files in the directory when `path` is specified.
            This behaviour was inherited from Apache Spark. The number of files can
            be controlled by `num_files`.

        .. note:: output JSON format is different from pandas'. It always use `orient='records'`
            for its output. This behaviour might have to change in the near future.

        Note NaN's and None will be converted to null and datetime objects
        will be converted to UNIX timestamps.

        Parameters
        ----------
        path : Optional[str], optional
            File path. If not specified, the result is returned as
            a string.
        lines : bool, default True
            If ‘orient’ is ‘records’ write out line delimited json format.
            Will throw ValueError if incorrect ‘orient’ since others are not
            list like. It should be always True for now.
        orient : str, default 'records'
             It should be always 'records' for now.
        compression : str, default 'uncompressed'
            A string representing the compression to use in the output file,
            only used when the first argument is a filename. By default, the
            compression is inferred from the filename.
        num_files : Optional[int], default None
            The number of files to be written in `path` directory when
            this is a path.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'},
            default 'overwrite'. Specifies the behavior of the save operation when the
            destination exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : Optional[Union[str, List[str]]], optional, default None
            Names of partitioning columns
        index_col: Optional[Union[str, List[str]]], optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options: Any
            Keyword arguments for additional options specific to PySpark.
            It is specific to PySpark's JSON options to pass. Check
            the options in PySpark's API documentation for `spark.write.json(...)`.
            It has a higher priority and overwrites all other options.
            This parameter only works when `path` is specified.

        Returns
        --------
        Optional[str]

        Examples
        --------
        >>> df = ks.DataFrame([['a', 'b'], ['c', 'd']],
        ...                   columns=['col 1', 'col 2'])
        >>> df.to_json()
        '[{"col 1":"a","col 2":"b"},{"col 1":"c","col 2":"d"}]'

        >>> df['col 1'].to_json()
        '[{"col 1":"a"},{"col 1":"c"}]'

        >>> df.to_json(path=r'%s/to_json/foo.json' % path, num_files=1)
        >>> ks.read_json(
        ...     path=r'%s/to_json/foo.json' % path
        ... ).sort_values(by="col 1")
          col 1 col 2
        0     a     b
        1     c     d

        >>> df['col 1'].to_json(path=r'%s/to_json/foo.json' % path, num_files=1, index_col="index")
        >>> ks.read_json(
        ...     path=r'%s/to_json/foo.json' % path, index_col="index"
        ... ).sort_values(by="col 1")  # doctest: +NORMALIZE_WHITESPACE
              col 1
        index
        0         a
        1         c

        """
        if "options" in kwargs and isinstance(kwargs.get("options"), dict) and len(kwargs) == 1:
            options = kwargs.get("options")  # type: ignore

        if not lines:
            raise NotImplementedError("lines=False is not implemented yet.")

        if orient != "records":
            raise NotImplementedError("orient='records' is supported only for now.")

        if path is None:
            # If path is none, just collect and use pandas's to_json.
            kdf_or_ser = self
            pdf = cast(Union[pd.DataFrame, pd.Series], kdf_or_ser.to_pandas())  # type: ignore
            if isinstance(self, ks.Series):
                pdf = pdf.to_frame()
            # To make the format consistent and readable by `read_json`, convert it to pandas' and
            # use 'records' orient for now.
            return cast(Optional[str], pdf.to_json(orient="records"))

        kdf: "DataFrame"
        if isinstance(self, ks.Series):
            kdf = self.to_frame()
        else:
            kdf = cast("DataFrame", self)

        sdf: SparkDataFrame = kdf.to_spark(index_col=index_col)  # type: ignore

        if num_files is not None:
            sdf = sdf.repartition(num_files)

        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder = builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format("json").save(path)
        return None

    def to_excel(
        self,
        excel_writer: Union[str, Any],
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: Optional[str] = None,
        columns: Optional[Union[List[str], List[Tuple[str, ...]]]] = None,
        header: Union[bool, List[str]] = True,
        index: bool = True,
        index_label: Optional[Union[str, List[str]]] = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Optional[str] = None,
        merge_cells: bool = True,
        encoding: Optional[str] = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Write object to an Excel sheet.

        .. note:: This method should only be used if the resulting DataFrame is expected
                  to be small, as all the data is loaded into the driver's memory.

        To write a single object to an Excel .xlsx file it is only necessary to
        specify a target file name. To write to multiple sheets it is necessary to
        create an `ExcelWriter` object with a target file name, and specify a sheet
        in the file to write to.

        Multiple sheets may be written to by specifying unique `sheet_name`.
        With all data written to the file it is necessary to save the changes.
        Note that creating an `ExcelWriter` object with a file name that already
        exists will result in the contents of the existing file being erased.

        Parameters
        ----------
        excel_writer : Union[str, Any]
            File path or existing ExcelWriter.
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame.
        na_rep : str, default ''
            Missing data representation.
        float_format : Optional[str], optional
            Format string for floating point numbers. For example
            ``float_format="%%.2f"`` will format 0.1234 to 0.12.
        columns : Optional[Union[List[str], List[Tuple[str, ...]]]], optional
            Columns to write.
        header : Union[bool, List[str]], default True
            Write out the column names. If a list of string is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : Optional[Union[str, List[str]]], optional
            Column label for index column(s) if desired. If not specified, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the DataFrame uses MultiIndex.
        startrow : int, default 0
            Upper left cell row to dump data frame.
        startcol : int, default 0
            Upper left cell column to dump data frame.
        engine : Optional[str], optional
            Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this
            via the options ``io.excel.xlsx.writer``, ``io.excel.xls.writer``, and
            ``io.excel.xlsm.writer``.
        merge_cells : bool, default True
            Write MultiIndex and Hierarchical Rows as merged cells.
        encoding : Optional[str], optional
            Encoding of the resulting excel file. Only necessary for xlwt,
            other writers support unicode natively.
        inf_rep : str, default 'inf'
            Representation for infinity (there is no native representation for
            infinity in Excel).
        verbose : bool, default True
            Display more information in the error logs.
        freeze_panes : Optional[Tuple[int, int]], optional
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen.

        Notes
        -----
        Once a workbook has been saved it is not possible write further data
        without rewriting the whole workbook.

        See Also
        --------
        read_excel : Read Excel file.

        Examples
        --------
        Create, write to and save a workbook:

        >>> df1 = ks.DataFrame([['a', 'b'], ['c', 'd']],
        ...                    index=['row 1', 'row 2'],
        ...                    columns=['col 1', 'col 2'])
        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP

        To specify the sheet name:

        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP
        >>> df1.to_excel("output.xlsx",
        ...              sheet_name='Sheet_name_1')  # doctest: +SKIP

        If you wish to write to more than one sheet in the workbook, it is
        necessary to specify an ExcelWriter object:

        >>> with pd.ExcelWriter('output.xlsx') as writer:  # doctest: +SKIP
        ...      df1.to_excel(writer, sheet_name='Sheet_name_1')
        ...      df2.to_excel(writer, sheet_name='Sheet_name_2')

        To set the library that is used to write the Excel file,
        you can pass the `engine` keyword (the default engine is
        automatically chosen depending on the file extension):

        >>> df1.to_excel('output1.xlsx', engine='xlsxwriter')  # doctest: +SKIP
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kdf = self

        if isinstance(self, ks.DataFrame):
            f = pd.DataFrame.to_excel
        elif isinstance(self, ks.Series):
            f = pd.Series.to_excel
        else:
            raise TypeError(
                "Constructor expects DataFrame or Series; however, " "got [%s]" % (self,)
            )
        return validate_arguments_and_invoke_function(
            kdf._to_internal_pandas(), self.to_excel, f, args
        )

    def mean(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return the mean of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        mean : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.mean()
        a    2.0
        b    0.2
        dtype: float64

        >>> df.mean(axis=1)
        0    0.55
        1    1.10
        2    1.65
        3     NaN
        dtype: float64

        On a Series:

        >>> df['a'].mean()
        2.0
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def mean(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.mean(spark_column)

        return self._reduce_for_stat_function(
            mean, name="mean", axis=axis, numeric_only=numeric_only
        )

    def sum(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None,
        min_count: int = 0
    ) -> Union[Scalar, "Series"]:
        """
        Return the sum of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
             ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        sum : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, np.nan, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.sum()
        a    6.0
        b    0.4
        dtype: float64

        >>> df.sum(axis=1)
        0    1.1
        1    2.0
        2    3.3
        3    0.0
        dtype: float64

        >>> df.sum(min_count=3)
        a    6.0
        b    NaN
        dtype: float64

        >>> df.sum(axis=1, min_count=1)
        0    1.1
        1    2.0
        2    3.3
        3    NaN
        dtype: float64

        On a Series:

        >>> df['a'].sum()
        6.0

        >>> df['a'].sum(min_count=3)
        6.0
        >>> df['b'].sum(min_count=3)
        nan
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def sum_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.coalesce(F.sum(spark_column), F.lit(0))

        return self._reduce_for_stat_function(
            sum_func, name="sum", axis=axis, numeric_only=numeric_only, min_count=min_count
        )

    def product(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None,
        min_count: int = 0
    ) -> Union[Scalar, "Series"]:
        """
        Return the product of the values.

        .. note:: unlike pandas', Koalas' emulates product by ``exp(sum(log(...)))``
            trick. Therefore, it only works for positive numbers.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        product : Union[Scalar, "Series"]

        Examples
        --------
        On a DataFrame:

        Non-numeric type column is not included to the result.

        >>> kdf = ks.DataFrame({'A': [1, 2, 3, 4, 5],
        ...                     'B': [10, 20, 30, 40, 50],
        ...                     'C': ['a', 'b', 'c', 'd', 'e']})
        >>> kdf
           A   B  C
        0  1  10  a
        1  2  20  b
        2  3  30  c
        3  4  40  d
        4  5  50  e

        >>> kdf.prod()
        A         120
        B    12000000
        dtype: int64

        If there is no numeric type columns, returns empty Series.

        >>> ks.DataFrame({"key": ['a', 'b', 'c'], "val": ['x', 'y', 'z']}).prod()
        Series([], dtype: float64)

        On a Series:

        >>> ks.Series([1, 2, 3, 4, 5]).prod()
        120

        By default, the product of an empty or all-NA Series is ``1``

        >>> ks.Series([]).prod()
        1.0

        This can be controlled with the ``min_count`` parameter

        >>> ks.Series([]).prod(min_count=1)
        nan
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def prod(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                scol = F.min(F.coalesce(spark_column, F.lit(True))).cast(LongType())
            elif isinstance(spark_type, NumericType):
                num_zeros = F.sum(F.when(spark_column == 0, 1).otherwise(0))
                sign = F.when(
                    F.sum(F.when(spark_column < 0, 1).otherwise(0)) % 2 == 0, 1
                ).otherwise(-1)

                scol = F.when(num_zeros > 0, 0).otherwise(
                    sign * F.exp(F.sum(F.log(F.abs(spark_column))))
                )

                if isinstance(spark_type, IntegralType):
                    scol = F.round(scol).cast(LongType())
            else:
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )

            return F.coalesce(scol, F.lit(1))

        return self._reduce_for_stat_function(
            prod, name="prod", axis=axis, numeric_only=numeric_only, min_count=min_count
        )

    prod = product

    def skew(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return unbiased skew normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        skew : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.skew()  # doctest: +SKIP
        a    0.000000e+00
        b   -3.319678e-16
        dtype: float64

        On a Series:

        >>> df['a'].skew()
        0.0
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def skew_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.skewness(spark_column)

        return self._reduce_for_stat_function(
            skew_func, name="skew", axis=axis, numeric_only=numeric_only
        )

    def kurtosis(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return unbiased kurtosis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).
        Normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        kurt : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.kurtosis()
        a   -1.5
        b   -1.5
        dtype: float64

        On a Series:

        >>> df['a'].kurtosis()
        -1.5
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def kurtosis_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.kurtosis(spark_column)

        return self._reduce_for_stat_function(
            kurtosis_func, name="kurtosis", axis=axis, numeric_only=numeric_only
        )

    kurt = kurtosis

    def min(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return the minimum of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            If True, include only float, int, boolean columns. This parameter is mainly for
            pandas compatibility. False is supported; however, the columns should
            be all numeric or all non-numeric.

        Returns
        -------
        min : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.min()
        a    1.0
        b    0.1
        dtype: float64

        >>> df.min(axis=1)
        0    0.1
        1    0.2
        2    0.3
        3    NaN
        dtype: float64

        On a Series:

        >>> df['a'].min()
        1.0
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        return self._reduce_for_stat_function(
            F.min, name="min", axis=axis, numeric_only=numeric_only
        )

    def max(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return the maximum of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            If True, include only float, int, boolean columns. This parameter is mainly for
            pandas compatibility. False is supported; however, the columns should
            be all numeric or all non-numeric.

        Returns
        -------
        max : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.max()
        a    3.0
        b    0.3
        dtype: float64

        >>> df.max(axis=1)
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        dtype: float64

        On a Series:

        >>> df['a'].max()
        3.0
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        return self._reduce_for_stat_function(
            F.max, name="max", axis=axis, numeric_only=numeric_only
        )

    def count(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: bool = False
    ) -> Union[Scalar, "Series"]:
        """
        Count non-NA cells for each column.

        The values `None`, `NaN` are considered NA.

        Parameters
        ----------
        axis : {0 or ‘index’, 1 or ‘columns’}, default 0
            If 0 or ‘index’ counts are generated for each column. If 1 or ‘columns’ counts are
            generated for each row.
        numeric_only : bool, default False
            If True, include only float, int, boolean columns. This parameter is mainly for
            pandas compatibility.

        Returns
        -------
        max : Union[Scalar, "Series"]

        See Also
        --------
        DataFrame.shape: Number of DataFrame rows and columns (including NA
            elements).
        DataFrame.isna: Boolean same-sized DataFrame showing places of NA
            elements.

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = ks.DataFrame({"Person":
        ...                    ["John", "Myla", "Lewis", "John", "Myla"],
        ...                    "Age": [24., np.nan, 21., 33, 26],
        ...                    "Single": [False, True, True, True, False]},
        ...                   columns=["Person", "Age", "Single"])
        >>> df
          Person   Age  Single
        0   John  24.0   False
        1   Myla   NaN    True
        2  Lewis  21.0    True
        3   John  33.0    True
        4   Myla  26.0   False

        Notice the uncounted NA values:

        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64

        >>> df.count(axis=1)
        0    3
        1    2
        2    3
        3    3
        4    3
        dtype: int64

        On a Series:

        >>> df['Person'].count()
        5

        >>> df['Age'].count()
        4
        """
        return self._reduce_for_stat_function(
            Frame._count_expr, name="count", axis=axis, numeric_only=numeric_only
        )

    def std(
        self,
        axis: Optional[Union[int, str]] = None,
        ddof: int = 1,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return sample standard deviation.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        std : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.std()
        a    1.0
        b    0.1
        dtype: float64

        >>> df.std(axis=1)
        0    0.636396
        1    1.272792
        2    1.909188
        3         NaN
        dtype: float64

        >>> df.std(ddof=0)
        a    0.816497
        b    0.081650
        dtype: float64

        On a Series:

        >>> df['a'].std()
        1.0

        >>> df['a'].std(ddof=0)
        0.816496580927726
        """
        assert ddof in (0, 1)

        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def std_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            if ddof == 0:
                return F.stddev_pop(spark_column)
            else:
                return F.stddev_samp(spark_column)

        return self._reduce_for_stat_function(
            std_func, name="std", axis=axis, numeric_only=numeric_only, ddof=ddof
        )

    def var(
        self,
        axis: Optional[Union[int, str]] = None,
        ddof: int = 1,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return unbiased variance.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        var : Union[Scalar, "Series"]

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.var()
        a    1.00
        b    0.01
        dtype: float64

        >>> df.var(axis=1)
        0    0.405
        1    1.620
        2    3.645
        3      NaN
        dtype: float64

        >>> df.var(ddof=0)
        a    0.666667
        b    0.006667
        dtype: float64

        On a Series:

        >>> df['a'].var()
        1.0

        >>> df['a'].var(ddof=0)
        0.6666666666666666
        """
        assert ddof in (0, 1)

        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def var_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            if ddof == 0:
                return F.var_pop(spark_column)
            else:
                return F.var_samp(spark_column)

        return self._reduce_for_stat_function(
            var_func, name="var", axis=axis, numeric_only=numeric_only, ddof=ddof
        )

    def median(
        self,
        axis: Optional[Union[int, str]] = None,
        numeric_only: Optional[bool] = None,
        accuracy: int = 10000
    ) -> Union[Scalar, "Series"]:
        """
        Return the median of the values for the requested axis.

        .. note:: Unlike pandas', the median in Koalas is an approximated median based upon
            approximate percentile computation because computing median across a large dataset
            is extremely expensive.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.
        accuracy : int, optional
            Default accuracy of approximation. Larger value means better accuracy.
            The relative error can be deduced by 1.0 / accuracy.

        Returns
        -------
        median : Union[Scalar, "Series"]

        Examples
        --------
        >>> df = ks.DataFrame({
        ...     'a': [24., 21., 25., 33., 26.], 'b': [1, 2, 3, 4, 5]}, columns=['a', 'b'])
        >>> df
              a  b
        0  24.0  1
        1  21.0  2
        2  25.0  3
        3  33.0  4
        4  26.0  5

        On a DataFrame:

        >>> df.median()
        a    25.0
        b     3.0
        dtype: float64

        On a Series:

        >>> df['a'].median()
        25.0
        >>> (df['b'] + 100).median()
        103.0

        For multi-index columns,

        >>> df.columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        >>> df
              x  y
              a  b
        0  24.0  1
        1  21.0  2
        2  25.0  3
        3  33.0  4
        4  26.0  5

        On a DataFrame:

        >>> df.median()
        x  a    25.0
        y  b     3.0
        dtype: float64

        >>> df.median(axis=1)
        0    12.5
        1    11.5
        2    14.0
        3    18.5
        4    15.5
        dtype: float64

        On a Series:

        >>> df[('x', 'a')].median()
        25.0
        >>> (df[('y', 'b')] + 100).median()
        103.0
        """
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        if not isinstance(accuracy, int):
            raise ValueError(
                "accuracy must be an integer; however, got [%s]" % type(accuracy).__name__
            )

        def median_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, (BooleanType, NumericType)):
                return SF.percentile_approx(spark_column.cast(DoubleType()), 0.5, accuracy)
            else:
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )

        return self._reduce_for_stat_function(
            median_func, name="median", numeric_only=numeric_only, axis=axis
        )

    def sem(
        self,
        axis: Optional[Union[int, str]] = None,
        ddof: int = 1,
        numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        """
        Return unbiased standard error of the mean over requested axis.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : Optional[bool], default None
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        scalar(for Series) or Series(for DataFrame)

        Examples
        --------
        >>> kdf = ks.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> kdf
           a  b
        0  1  4
        1  2  5
        2  3  6

        >>> kdf.sem()
        a    0.57735
        b    0.57735
        dtype: float64

        >>> kdf.sem(ddof=0)
        a    0.471405
        b    0.471405
        dtype: float64

        >>> kdf.sem(axis=1)
        0    1.5
        1    1.5
        2    1.5
        dtype: float64

        Support for Series

        >>> kser = kdf.a
        >>> kser
        0    1
        1    2
        2    3
        Name: a, dtype: int64

        >>> kser.sem()
        0.5773502691896258

        >>> kser.sem(ddof=0)
        0.47140452079103173
        """
        assert ddof in (0, 1)

        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def std_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            if ddof == 0:
                return F.stddev_pop(spark_column)
            else:
                return F.stddev_samp(spark_column)

        def sem_func(spark_column: Any, spark_type: Any) -> Any:
            return std_func(spark_column, spark_type) / pow(
                Frame._count_expr(spark_column, spark_type), 0.5
            )

        return self._reduce_for_stat_function(
            sem_func, name="sem", numeric_only=numeric_only, axis=axis, ddof=ddof
        )

    @property
    def size(self) -> int:
        """
        Return an int representing the number of elements in this object.

        Return the number of rows if Series. Otherwise return the number of
        rows times number of columns if DataFrame.

        Examples
        --------
        >>> s = ks.Series({'a': 1, 'b': 2, 'c': None})
        >>> s.size
        3

        >>> df = ks.DataFrame({'col1': [1, 2, None], 'col2': [3, 4, None]})
        >>> df.size
        6

        >>> df = ks.DataFrame(index=[1, 2, None])
        >>> df.size
        0
        """
        num_columns = len(self._internal.data_spark_columns)
        if num_columns == 0:
            return 0
        else:
            return len(self) * num_columns  # type: ignore

    def abs(self) -> Union["DataFrame", "Series"]:
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        Returns
        -------
        abs : Union["DataFrame", "Series"] containing the absolute value of each element.

        Examples
        --------

        Absolute numeric values in a Series.

        >>> s = ks.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        dtype: float64

        Absolute numeric values in a DataFrame.

        >>> df = ks.DataFrame({
        ...     'a': [4, 5, 6, 7],
        ...     'b': [10, 20, 30, 40],
        ...     'c': [100, 50, -30, -50]
        ...   },
        ...   columns=['a', 'b', 'c'])
        >>> df.abs()
           a   b    c
        0  4  10  100
        1  5  20   50
        2  6  30   30
        3  7  40   50

        """
        def abs_op(kser: "Series") -> "Series":
            if isinstance(kser.spark.data_type, BooleanType):
                return kser
            elif isinstance(kser.spark.data_type, NumericType):
                return kser.spark.transform(F.abs)
            else:
                raise TypeError(
                    "bad operand type for abs(): {} ({})".format(
                        spark_type_to_pandas_dtype(kser.spark.data_type),
                        kser.spark.data_type.simpleString(),
                    )
                )

        return self._apply_series_op(abs_op)

    # TODO: by argument only support the grouping name and as_index only for now. Documentation
    # should be updated when it's supported.
    def groupby(
        self,
        by: Union["Series", Any, List[Any]],
        axis: int = 0,
        as_index: bool = True,
        dropna: bool = True
    ) -> Union["DataFrameGroupBy", "SeriesGroupBy"]:
        """
        Group DataFrame or Series using a Series of columns.

        A groupby operation involves some combination of splitting the
        object, applying a function, and combining the results. This can be
        used to group large amounts of data and compute operations on these
        groups.

        Parameters
        ----------
        by : Union["Series", Any, List[Any]]
            Used to determine the groups for the groupby.
            If Series is passed, the Series or dict VALUES
            will be used to determine the groups. A label or list of
            labels may be passed to group by the columns in ``self``.
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.
        as_index : bool, default True
            For aggregated output, return object with group labels as the
            index. Only relevant for DataFrame input. as_index=False is
            effectively "SQL-style" grouped output.
        dropna : bool, default True
            If True, and if group keys contain NA values,
            NA values together with row/column will be dropped.
            If False, NA values will also be treated as the key in groups.

        Returns
        -------
        DataFrameGroupBy or SeriesGroupBy
            Depends on the calling object and returns groupby object that
            contains information about the groups.

        See Also
        --------
        koalas.groupby.GroupBy

        Examples
        --------
        >>> df = ks.DataFrame({'Animal': ['Falcon', 'Falcon',
        ...                               'Parrot', 'Parrot'],
        ...                    'Max Speed': [380., 370., 24., 26.]},
        ...                   columns=['Animal', 'Max Speed'])
        >>> df
           Animal  Max Speed
        0  Falcon      380.0
        1  Falcon      370.0
        2  Parrot       24.0
        3  Parrot       26.0

        >>> df.groupby(['Animal']).mean().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                Max Speed
        Animal
        Falcon      375.0
        Parrot       25.0

        >>> df.groupby(['Animal'], as_index=False).mean().sort_values('Animal')
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
           Animal  Max Speed
        ...Falcon      375.0
        ...Parrot       25.0

        We can also choose to include NA in group keys or not by setting dropna parameter,
        the default setting is True:

        >>> l = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]
        >>> df = ks.DataFrame(l, columns=["a", "b", "c"])
        >>> df.groupby(by=["b"]).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             a  c
        b
        1.0  2  3
        2.0  2  5

        >>> df.groupby(by=["b"], dropna=False).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             a  c
        b
        1.0  2  3
        2.0  2  5
        NaN  1  4
        """
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy

        if isinstance(by, ks.DataFrame):
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        elif isinstance(by, ks.Series):
            by = [by]
        elif is_name_like_tuple(by):
            if isinstance(self, ks.Series):
                raise KeyError(by)
            by = [by]
        elif is_name_like_value(by):
            if isinstance(self, ks.Series):
                raise KeyError(by)
            by = [(cast(str, by),)]
        elif is_list_like(by):
            new_by: List[Union[Tuple[str, ...], "Series"]] = []
            for key in by:
                if isinstance(key, ks.DataFrame):
                    raise ValueError(
                        "Grouper for '{}' not 1-dimensional".format(type(key).__name__)
                    )
                elif isinstance(key, ks.Series):
                    new_by.append(key)
                elif is_name_like_tuple(key):
                    if isinstance(self, ks.Series):
                        raise KeyError(key)
                    new_by.append(key)
                elif is_name_like_value(key):
                    if isinstance(self, ks.Series):
                        raise KeyError(key)
                    new_by.append((cast(str, key),))
                else:
                    raise ValueError(
                        "Grouper for '{}' not 1-dimensional".format(type(key).__name__)
                    )
            by = new_by
        else:
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        if not len(by):
            raise ValueError("No group keys passed!")
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        if isinstance(self, ks.DataFrame):
            return DataFrameGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        elif isinstance(self, ks.Series):
            return SeriesGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        else:
            raise TypeError(
                "Constructor expects DataFrame or Series; however, " "got [%s]" % (self,)
            )

    def bool(self) -> bool:
        """
        Return the bool of a single element in the current object.

        This must be a boolean scalar value, either True or False. Raise a ValueError if
        the object does not have exactly 1 element, or that element is not boolean

        Returns
        --------
        bool

        Examples
        --------
        >>> ks.DataFrame({'a': [True]}).bool()
        True

        >>> ks.Series([False]).bool()
        False

        If there are non-boolean or multiple values exist, it raises an exception in all
        cases as below.

        >>> ks.DataFrame({'a': ['a']}).bool()
        Traceback (most recent call last):
          ...
        ValueError: bool cannot act on a non-boolean single element DataFrame

        >>> ks.DataFrame({'a': [True], 'b': [False]}).bool()  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
          ...
        ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(),
        a.item(), a.any() or a.all().

        >>> ks.Series([1]).bool()
        Traceback (most recent call last):
          ...
        ValueError: bool cannot act on a non-boolean single element DataFrame
        """
        if isinstance(self, ks.DataFrame):
            df = self
        elif isinstance(self, ks.Series):
            df = self.to_dataframe()
        else:
            raise TypeError("bool() expects DataFrame or Series; however, " "got [%s]" % (self,))
        return df.head(2)._to_internal_pandas().bool()

    def first_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        """
        Retrieves the index of the first valid value.

        Returns
        -------
        Optional[Union[Scalar, Tuple[Scalar, ...]]]

        Examples
        --------

        Support for DataFrame

        >>> kdf = ks.DataFrame({'a': [None, 2, 3, 2],
        ...                     'b': [None, 2.0, 3.0, 1.0],
        ...                     'c': [None, 200, 400, 200]},
        ...                     index=['Q', 'W', 'E', 'R'])
        >>> kdf
             a    b      c
        Q  NaN  NaN    NaN
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  2.0  1.0  200.0

        >>> kdf.first_valid_index()
        'W'

        Support for MultiIndex columns

        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
             a    b      c
             x    y      z
        Q  NaN  NaN    NaN
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  2.0  1.0  200.0

        >>> kdf.first_valid_index()
        'W'

        Support for Series.

        >>> s = ks.Series([None, None, 3, 4, 5], index=[100, 200, 300, 400, 500])
        >>> s
        100    NaN
        200    NaN
        300    3.0
        400    4.0
        500    5.0
        dtype: float64

        >>> s.first_valid_index()
        300

        Support for MultiIndex

        >>> midx = pd.MultiIndex.from_tuples([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      names=[None, None])
        >>> s = ks.Series([None, None, None, 1, 250, 1.5, 320, 1, 0.3], index=midx)
        >>> s
        lama    speed       NaN
                weight      NaN
                length      NaN
        cow     speed       1.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        dtype: float64

        >>> s.first_valid_index()
        ('cow', 'speed')
        """
        data_spark_columns = self._internal.data_spark_columns

        if len(data_spark_columns) == 0:
            return None

        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))

        with sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            # Disable Arrow to keep row ordering.
            first_valid_row = (
                self._internal.spark_frame.filter(cond)
                .select(self._internal.index_spark_columns)
                .limit(1)
                .toPandas()
            )

        # For Empty Series or DataFrame, returns None.
        if len(first_valid_row) == 0:
            return None

        first_valid_row = first_valid_row.iloc[0]
        if len(first_valid_row) == 1:
            return first_valid_row.iloc[0]
        else:
            return tuple(first_valid_row)

    def last_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        """
        Return index for last non-NA/null value.

        Returns
        -------
        Optional[Union[Scalar, Tuple[Scalar, ...]]]

        Notes
        -----
        This API only works with PySpark >= 3.0.

        Examples
        --------

        Support for DataFrame

        >>> kdf = ks.DataFrame({'a': [1, 2, 3, None],
        ...                     'b': [1.0, 2.0, 3.0, None],
        ...                     'c': [100, 200, 400, None]},
        ...                     index=['Q', 'W', 'E', 'R'])
        >>> kdf
             a    b      c
        Q  1.0  1.0  100.0
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  NaN  NaN    NaN

        >>> kdf.last_valid_index()  # doctest: +SKIP
        'E'

        Support for MultiIndex columns

        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
             a    b      c
             x    y      z
        Q  1.0  1.0  100.0
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  NaN  NaN    NaN

        >>> kdf.last_valid_index()  # doctest: +SKIP
        'E'

        Support for Series.

        >>> s = ks.Series([1, 2, 3, None, None], index=[100, 200, 300, 400, 500])
        >>> s
        100    1.0
        200    2.0
        300    3.0
        400    NaN
        500    NaN
        dtype: float64

        >>> s.last_valid_index()  # doctest: +SKIP
        300

        Support for MultiIndex

        >>> midx = pd.MultiIndex.from_tuples([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      names=[None, None])
        >>> s = ks.Series([250, 1.5, 320, 1, 0.3, None, None, None, None], index=midx)
        >>> s
        lama    speed     250.0
                weight      1.5
                length    320.0
        cow     speed       1.0
                weight      0.3
                length      NaN
        falcon  speed       NaN
                weight      NaN
                length      NaN
        dtype: float64

        >>> s.last_valid_index()  # doctest: +SKIP
        ('cow', 'weight')
        """
        if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
            raise RuntimeError("last_valid_index can be used in PySpark >= 3.0")

        data_spark_columns = self._internal.data_spark_columns

        if len(data_spark_columns) == 0:
            return None

        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))

        last_valid_rows = (
            self._internal.spark_frame.filter(cond)
            .select(self._internal.index_spark_columns)
            .tail(1)
        )

        # For Empty Series or DataFrame, returns None.
        if len(last_valid_rows) == 0:
            return None

        last_valid_row = last_valid_rows[0]

        if len(last_valid_row) == 1:
            return last_valid_row[0]
        else:
            return tuple(last_valid_row)

    # TODO: add 'center', 'win_type', 'on', 'axis' parameter should be implemented.
    def rolling(self, window: Union[int, str], min_periods: Optional[int] = None) -> Rolling:
        """
        Provide rolling transformations.

        .. note:: 'min_periods' in Koalas works as a fixed window size unlike pandas.
            Unlike pandas, NA is also counted as the period. This might be changed
            in the near future.

        Parameters
        ----------
        window : Union[int, str]
            Size of the moving window.
            This is the number of observations used for calculating the statistic.
            Each window will be a fixed size.

        min_periods : Optional[int], default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA).

        Returns
        -------
        Rolling
            A Rolling object.
        """
        return Rolling(self, window=window, min_periods=min_periods)

    # TODO: add 'center' and 'axis' parameter should be implemented.
    #   'axis' implementation, refer https://github.com/databricks/koalas/pull/607
    def expanding(self, min_periods: int = 1) -> Expanding:
        """
        Provide expanding transformations.

        .. note:: 'min_periods' in Koalas works as a fixed window size unlike pandas.
            Unlike pandas, NA is also counted as the period. This might be changed
            in the near future.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum number of observations in window required to have a value
            (otherwise result is NA).

        Returns
        -------
        Expanding
            An Expanding object.
        """
        return Expanding(self, min_periods=min_periods)

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get item from object for given key (DataFrame column, Panel slice,
        etc.). Returns default value if not found.

        Parameters
        ----------
        key : Any

        Returns
        -------
        Any
            The value corresponding to the key or the default.

        Examples
        --------
        >>> df = ks.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']},
        ...                   columns=['x', 'y', 'z'], index=[10, 20, 20])
        >>> df
            x  y  z
        10  0  a  a
        20  1  b  b
        20  2  b  b

        >>> df.get('x')
        10    0
        20    1
        20    2
        Name: x, dtype: int64

        >>> df.get(['x', 'y'])
            x  y
        10  0  a
        20  1  b
        20  2  b

        >>> df.x.get(10)
        0

        >>> df.x.get(20)
        20    1
        20    2
        Name: x, dtype: int64

        >>> df.x.get(15, -1)
        -1
        """
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def squeeze(self, axis: Optional[Union[int, str]] = None) -> Union[Scalar, "DataFrame", "Series"]:
        """
        Squeeze 1 dimensional axis objects into scalars.

        Series or DataFrames with a single element are squeezed to a scalar.
        DataFrames with a single column or a single row are squeezed to a
        Series. Otherwise the object is unchanged.

        This method is most useful when you don't know if your
        object is a Series or DataFrame, but you do know it has just a single
        column. In that case you can safely call `squeeze` to ensure you have a
        Series.

        Parameters
        ----------
        axis : Optional[Union[int, str]], default None
            A specific axis to squeeze. By default, all length-1 axes are
            squeezed.

        Returns
        -------
        Union["DataFrame", "Series", Scalar]
            The projection after squeezing `axis` or all the axes.

        See Also
        --------
        Series.iloc : Integer-location based indexing for selecting scalars.
        DataFrame.iloc : Integer-location based indexing for selecting Series.
        Series.to_frame : Inverse of DataFrame.squeeze for a
            single-column DataFrame.

        Examples
        --------
        >>> primes = ks.Series([2, 3, 5, 7])

        Slicing might produce a Series with a single value:

        >>> even_primes = primes[primes % 2 == 0]
        >>> even_primes
        0    2
        dtype: int64

        >>> even_primes.squeeze()
        2

        Squeezing objects with more than one value in every axis does nothing:

        >>> odd_primes = primes[primes % 2 == 1]
        >>> odd_primes
        1    3
        2    5
        3    7
        dtype: int64

        >>> odd_primes.squeeze()
        1    3
        2    5
        3    7
        dtype: int64

        Squeezing is even more effective when used with DataFrames.

        >>> df = ks.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
        >>> df
           a  b
        0  1  2
        1  3  4

        Slicing a single column will produce a DataFrame with the columns
        having only one value:

        >>> df_a = df[['a']]
        >>> df_a
           a
        0  1
        1  3

        So the columns can be squeezed down, resulting in a Series:

        >>> df_a.squeeze('columns')
        0    1
        1    3
        Name: a, dtype: int64

        Slicing a single row from a single column will produce a single
        scalar DataFrame:

        >>> df_1a = df.loc[[1], ['a']]
        >>> df_1a
           a
        1  3

        Squeezing the rows produces a single scalar Series:

        >>> df_1a.squeeze('rows')
        a    3
        Name: 1, dtype: int64

        Squeezing all axes will project directly into a scalar:

        >>> df_1a.squeeze()
        3
        """
        if axis is not None:
            axis = "index" if axis == "rows" else axis
            axis = validate_axis(axis)

        if isinstance(self, ks.DataFrame):
            from databricks.koalas.series import first_series

            is_squeezable = len(self.columns[:2]) == 1
            # If DataFrame has multiple columns, there is no change.
            if not is_squeezable:
                return self
            series_from_column = first_series(self)
            has_single_value = len(series_from_column.head(2)) == 1
            # If DataFrame has only a single value, use pandas API directly.
            if has_single_value:
                result: Union[pd.DataFrame, pd.Series, Scalar] = self._to_internal_pandas().squeeze(axis)
                return ks.Series(result) if isinstance(result, pd.Series) else cast(Scalar, result)
            elif axis == 0:
                return self
            else:
                return series_from_column
        else:
            # The case of Series is simple.
            # If Series has only a single value, just return it as a scalar.
            # Otherwise, there is no change.
            self_top_two = self.head(2)
            has_single_value = len(self_top_two) == 1
            return cast(Union[Scalar, ks.Series], self_top_two[0] if has_single_value else self)

    def truncate(
        self,
        before: Optional[Union[str, int, float, Any]] = None,
        after: Optional[Union[str, int, float, Any]] = None,
        axis: Optional[Union[int, str]] = None,
        copy: bool = True
    ) -> Union["DataFrame", "Series"]:
        """
        Truncate a Series or DataFrame before and after some index value.

        This is a useful shorthand for boolean indexing based on index
        values above or below certain thresholds.

        .. note:: This API is dependent on :meth:`Index.is_monotonic_increasing`
            which can be expensive.

        Parameters
        ----------
        before : Union[str, int, float, Any], optional
            Truncate all rows before this index value.
        after : Union[str, int, float, Any], optional
            Truncate all rows after this index value.
        axis : Optional[Union[int, str]], default None
            Axis to truncate. Truncates the index (rows) by default.
        copy : bool, default is True,
            Fill in place (do not create a new object)

        Returns
        -------
        Union["DataFrame", "Series"]
            The truncated Series or DataFrame.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by label.
        DataFrame.iloc : Select a subset of a DataFrame by position.

        Examples
        --------
        >>> df = ks.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
        ...                    'B': ['f', 'g', 'h', 'i', 'j'],
        ...                    'C': ['k', 'l', 'm', 'n', 'o']},
        ...                   index=[1, 2, 3, 4, 5])
        >>> df
           A  B  C
        1  a  f  k
        2  b  g  l
        3  c  h  m
        4  d  i  n
        5  e  j  o

        >>> df.truncate(before=2, after=4)
           A  B  C
        2  b  g  l
        3  c  h  m
        4  d  i  n

        The columns of a DataFrame can be truncated.

        >>> df.truncate(before="A", after="B", axis="columns")
           A  B
        1  a  f
        2  b  g
        3  c  h
        4  d  i
        5  e  j

        For Series, only rows can be truncated.

        >>> df['A'].truncate(before=2, after=4)
        2    b
        3    c
        4    d
        Name: A, dtype: object

        A Series has index that sorted integers.

        >>> s = ks.Series([10, 20, 30, 40, 50, 60, 70],
        ...               index=[1, 2, 3, 4, 5, 6, 7])
        >>> s
        1    10
        2    20
        3    30
        4    40
        5    50
        6    60
        7    70
        dtype: int64

        >>> s.truncate(2, 5)
        2    20
        3    30
        4    40
        5    50
        dtype: int64

        A Series has index that sorted strings.

        >>> s = ks.Series([10, 20, 30, 40, 50, 60, 70],
        ...               index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        >>> s
        a    10
        b    20
        c    30
        d    40
        e    50
        f    60
        g    70
        dtype: int64

        >>> s.truncate('b', 'e')
        b    20
        c    30
        d    40
        e    50
        dtype: int64
        """
        from databricks.koalas.series import first_series

        axis = validate_axis(axis)
        indexes = self.index
        indexes_increasing = indexes.is_monotonic_increasing
        if not indexes_increasing and not indexes.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")
        if (before is None) and (after is None):
            return cast(Union[ks.DataFrame, ks.Series], self.copy() if copy else self)
        if (before is not None and after is not None) and before > after:
            raise ValueError("Truncate: %s must be after %s" % (after, before))

        if isinstance(self, ks.Series):
            if indexes_increasing:
                result: Union["Series", "DataFrame"] = first_series(self.to_frame().loc[before:after]).rename(self.name)
            else:
                result = first_series(self.to_frame().loc[after:before]).rename(self.name)
        elif isinstance(self, ks.DataFrame):
            if axis == 0:
                if indexes_increasing:
                    result = self.loc[before:after]
                else:
                    result = self.loc[after:before]
            elif axis == 1:
                result = self.loc[:, before:after]

        return cast(Union[ks.DataFrame, ks.Series], result.copy() if copy else result)

    def to_markdown(self, buf: Optional[Any] = None, mode: Optional[str] = None, **kwargs: Any) -> str:
        """
        Print Series or DataFrame in Markdown-friendly format.

        .. note:: This method should only be used if the resulting pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        buf : Optional[Any], default None
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        mode : Optional[str], optional
            Mode in which file is opened.
        **kwargs : Any
            These parameters will be passed to `tabulate`.

        Returns
        -------
        str
            Series or DataFrame in Markdown-friendly format.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        Examples
        --------
        >>> kser = ks.Series(["elk", "pig", "dog", "quetzal"], name="animal")
        >>> print(kser.to_markdown())  # doctest: +SKIP
        |    | animal   |
        |---:|:---------|
        |  0 | elk      |
        |  1 | pig      |
        |  2 | dog      |
        |  3 | quetzal  |

        >>> kdf = ks.DataFrame(
        ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
        ... )
        >>> print(kdf.to_markdown())  # doctest: +SKIP
        |    | animal_1   | animal_2   |
        |---:|:-----------|:-----------|
        |  0 | elk        | dog        |
        |  1 | pig        | quetzal    |
        """
        # `to_markdown` is supported in pandas >= 1.0.0 since it's newly added in pandas 1.0.0.
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            raise NotImplementedError(
                "`to_markdown()` only supported in Koalas with pandas >= 1.0.0"
            )
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kser_or_kdf = self
        internal_pandas = kser_or_kdf._to_internal_pandas()
        return cast(str, validate_arguments_and_invoke_function(
            internal_pandas, self.to_markdown, type(internal_pandas).to_markdown, args
        ))

    @abstractmethod
    def fillna(
        self,
        value: Any = None,
        method: Optional[str] = None,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None
    ) -> Optional[Union["DataFrame", "Series"]]:
        pass

    # TODO: add 'downcast' when value parameter exists
    def bfill(
        self,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None
    ) -> Union["DataFrame", "Series"]:
        """
        Synonym for `DataFrame.fillna()` or `Series.fillna()` with ``method=`bfill