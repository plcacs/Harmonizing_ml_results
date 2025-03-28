from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from itertools import accumulate
import py4j

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_datetime64_dtype, is_datetime64tz_dtype
import pyspark
from pyspark import sql as spark
from pyspark._globals import _NoValue, _NoValueType
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import BooleanType, DataType, IntegralType, StructField, StructType, LongType

try:
    from pyspark.sql.types import to_arrow_type
except ImportError:
    from pyspark.sql.pandas.types import to_arrow_type  # noqa: F401

# For running doctests and reference resolution in PyCharm.
from databricks import koalas as ks  # noqa: F401

if TYPE_CHECKING:
    # This is required in old Python 3.5 to prevent circular reference.
    from databricks.koalas.series import Series
from databricks.koalas.config import get_option
from databricks.koalas.typedef import (
    Dtype,
    DtypeDataTypes,
    as_spark_type,
    extension_dtypes,
    infer_pd_series_spark_type,
    spark_type_to_pandas_dtype,
)
from databricks.koalas.utils import (
    column_labels_level,
    default_session,
    is_name_like_tuple,
    is_testing,
    lazy_property,
    name_like_string,
    scol_for,
    verify_temp_column_name,
)


# A function to turn given numbers to Spark columns that represent Koalas index.
SPARK_INDEX_NAME_FORMAT = "__index_level_{}__".format
SPARK_DEFAULT_INDEX_NAME = SPARK_INDEX_NAME_FORMAT(0)
# A pattern to check if the name of a Spark column is a Koalas index name or not.
SPARK_INDEX_NAME_PATTERN = re.compile(r"__index_level_[0-9]+__")

NATURAL_ORDER_COLUMN_NAME = "__natural_order__"

HIDDEN_COLUMNS = {NATURAL_ORDER_COLUMN_NAME}

DEFAULT_SERIES_NAME = 0
SPARK_DEFAULT_SERIES_NAME = str(DEFAULT_SERIES_NAME)


class InternalFrame(object):
    """
    The internal immutable DataFrame which manages Spark DataFrame and column names and index
    information.

    .. note:: this is an internal class. It is not supposed to be exposed to users and users
        should not directly access to it.

    The internal immutable DataFrame represents the index information for a DataFrame it belongs to.
    For instance, if we have a Koalas DataFrame as below, pandas DataFrame does not store the index
    as columns.

    >>> kdf = ks.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [5, 6, 7, 8],
    ...     'C': [9, 10, 11, 12],
    ...     'D': [13, 14, 15, 16],
    ...     'E': [17, 18, 19, 20]}, columns = ['A', 'B', 'C', 'D', 'E'])
    >>> kdf  # doctest: +NORMALIZE_WHITESPACE
       A  B   C   D   E
    0  1  5   9  13  17
    1  2  6  10  14  18
    2  3  7  11  15  19
    3  4  8  12  16  20

    However, all columns including index column are also stored in Spark DataFrame internally
    as below.

    >>> kdf._internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+

    In order to fill this gap, the current metadata is used by mapping Spark's internal column
    to Koalas' index. See the method below:

    * `spark_frame` represents the internal Spark DataFrame

    * `data_spark_column_names` represents non-indexing Spark column names

    * `data_spark_columns` represents non-indexing Spark columns

    * `data_dtypes` represents external non-indexing dtypes

    * `index_spark_column_names` represents internal index Spark column names

    * `index_spark_columns` represents internal index Spark columns

    * `index_dtypes` represents external index dtypes

    * `spark_column_names` represents all columns

    * `index_names` represents the external index name as a label

    * `to_internal_spark_frame` represents Spark DataFrame derived by the metadata. Includes index.

    * `to_pandas_frame` represents pandas DataFrame derived by the metadata

    >>> internal = kdf._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['__index_level_0__']
    >>> internal.spark_column_names
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None]
    >>> internal.data_dtypes
    [dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64')]
    >>> internal.index_dtypes
    [dtype('int64')]
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.to_pandas_frame
       A  B   C   D   E
    0  1  5   9  13  17
    1  2  6  10  14  18
    2  3  7  11  15  19
    3  4  8  12  16  20

    In case that index is set to one of the existing column as below:

    >>> kdf1 = kdf.set_index("A")
    >>> kdf1  # doctest: +NORMALIZE_WHITESPACE
       B   C   D   E
    A
    1  5   9  13  17
    2  6  10  14  18
    3  7  11  15  19
    4  8  12  16  20

    >>> kdf1._internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+

    >>> internal = kdf1._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['A']
    >>> internal.spark_column_names
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [('A',)]
    >>> internal.data_dtypes
    [dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64')]
    >>> internal.index_dtypes
    [dtype('int64')]
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+
    >>> internal.to_pandas_frame  # doctest: +NORMALIZE_WHITESPACE
       B   C   D   E
    A
    1  5   9  13  17
    2  6  10  14  18
    3  7  11  15  19
    4  8  12  16  20

    In case that index becomes a multi index as below:

    >>> kdf2 = kdf.set_index("A", append=True)
    >>> kdf2  # doctest: +NORMALIZE_WHITESPACE
         B   C   D   E
      A
    0 1  5   9  13  17
    1 2  6  10  14  18
    2 3  7  11  15  19
    3 4  8  12  16  20

    >>> kdf2._internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+

    >>> internal = kdf2._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['__index_level_0__', 'A']
    >>> internal.spark_column_names
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None, ('A',)]
    >>> internal.data_dtypes
    [dtype('int64'), dtype('int64'), dtype('int64'), dtype('int64')]
    >>> internal.index_dtypes
    [dtype('int64'), dtype('int64')]
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.to_pandas_frame  # doctest: +NORMALIZE_WHITESPACE
         B   C   D   E
      A
    0 1  5   9  13  17
    1 2  6  10  14  18
    2 3  7  11  15  19
    3 4  8  12  16  20

    For multi-level columns, it also holds column_labels

    >>> columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'),
    ...                                      ('Y', 'C'), ('Y', 'D')])
    >>> kdf3 = ks.DataFrame([
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ...     [9, 10, 11, 12],
    ...     [13, 14, 15, 16],
    ...     [17, 18, 19, 20]], columns = columns)
    >>> kdf3  # doctest: +NORMALIZE_WHITESPACE
        X       Y
        A   B   C   D
    0   1   2   3   4
    1   5   6   7   8
    2   9  10  11  12
    3  13  14  15  16
    4  17  18  19  20

    >>> internal = kdf3._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+------+------+------+------+-----------------+
    |__index_level_0__|(X, A)|(X, B)|(Y, C)|(Y, D)|__natural_order__|
    +-----------------+------+------+------+------+-----------------+
    |                0|     1|     2|     3|     4|              ...|
    |                1|     5|     6|     7|     8|              ...|
    |                2|     9|    10|    11|    12|              ...|
    |                3|    13|    14|    15|    16|              ...|
    |                4|    17|    18|    19|    20|              ...|
    +-----------------+------+------+------+------+-----------------+
    >>> internal.data_spark_column_names
    ['(X, A)', '(X, B)', '(Y, C)', '(Y, D)']
    >>> internal.column_labels
    [('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Y', 'D')]

    For Series, it also holds scol to represent the column.

    >>> kseries = kdf1.B
    >>> kseries
    A
    1    5
    2    6
    3    7
    4    8
    Name: B, dtype: int64

    >>> internal = kseries._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B']
    >>> internal.index_spark_column_names
    ['A']
    >>> internal.spark_column_names
    ['A', 'B']
    >>> internal.index_names
    [('A',)]
    >>> internal.data_dtypes
    [dtype('int64')]
    >>> internal.index_dtypes
    [dtype('int64')]
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+
    |  A|  B|
    +---+---+
    |  1|  5|
    |  2|  6|
    |  3|  7|
    |  4|  8|
    +---+---+
    >>> internal.to_pandas_frame  # doctest: +NORMALIZE_WHITESPACE
       B
    A
    1  5
    2  6
    3  7
    4  8
    """

    def __init__(
        self,
        spark_frame: spark.DataFrame,
        index_spark_columns: Optional[List[spark.Column]],
        index_names: Optional[List[Optional[Tuple]]] = None,
        index_dtypes: Optional[List[Dtype]] = None,
        column_labels: Optional[List[Tuple]] = None,
        data_spark_columns: Optional[List[spark.Column]] = None,
        data_dtypes: Optional[List[Dtype]] = None,
        column_label_names: Optional[List[Optional[Tuple]]] = None,
    ) -> None:
        """
        Create a new internal immutable DataFrame to manage Spark DataFrame, column fields and
        index fields and names.

        :param spark_frame: Spark DataFrame to be managed.
        :param index_spark_columns: list of Spark Column
                                    Spark Columns for the index.
        :param index_names: list of tuples
                            the index names.
        :param index_dtypes: list of dtypes
                             the index dtypes.
        :param column_labels: list of tuples with the same length
                              The multi-level values in the tuples.
        :param data_spark_columns: list of Spark Column
                                   Spark Columns to appear as columns. If this is None, calculated
                                   from spark_frame.
        :param data_dtypes: list of dtypes.
                            the data dtypes.
        :param column_label_names: Names for each of the column index levels.

        See the examples below to refer what each parameter means.

        >>> column_labels = pd.MultiIndex.from_tuples(
        ...     [('a', 'x'), ('a', 'y'), ('b', 'z')], names=["column_labels_a", "column_labels_b"])
        >>> row_index = pd.MultiIndex.from_tuples(
        ...     [('foo', 'bar'), ('foo', 'bar'), ('zoo', 'bar')],
        ...     names=["row_index_a", "row_index_b"])
        >>> kdf = ks.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=row_index, columns=column_labels)
        >>> kdf.set_index(('a', 'x'), append=True, inplace=True)
        >>> kdf  # doctest: +NORMALIZE_WHITESPACE
        column_labels_a                  a  b
        column_labels_b                  y  z
        row_index_a row_index_b (a, x)
        foo         bar         1       2  3
                                4       5  6
        zoo         bar         7       8  9

        >>> internal = kdf._internal

        >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        +-----------------+-----------------+------+------+------+...
        |__index_level_0__|__index_level_1__|(a, x)|(a, y)|(b, z)|...
        +-----------------+-----------------+------+------+------+...
        |              foo|              bar|     1|     2|     3|...
        |              foo|              bar|     4|     5|     6|...
        |              zoo|              bar|     7|     8|     9|...
        +-----------------+-----------------+------+------+------+...

        >>> internal.index_spark_columns  # doctest: +SKIP
        [Column<'__index_level_0__'>, Column<'__index_level_1__'>, Column<'(a, x)'>]

        >>> internal.index_names
        [('row_index_a',), ('row_index_b',), ('a', 'x')]

        >>> internal.index_dtypes
        [dtype('O'), dtype('O'), dtype('int64')]

        >>> internal.column_labels
        [('a', 'y'), ('b', 'z')]

        >>> internal.data_spark_columns  # doctest: +SKIP
        [Column<'(a, y)'>, Column<'(b, z)'>]

        >>> internal.data_dtypes
        [dtype('int64'), dtype('int64')]

        >>> internal.column_label_names
        [('column_labels_a',), ('column_labels_b',)]
        """

        assert isinstance(spark_frame, spark.DataFrame)
        assert not spark_frame.isStreaming, "Koalas does not support Structured Streaming."

        if not index_spark_columns:
            if data_spark_columns is not None:
                if column_labels is not None:
                    data_spark_columns = [
                        scol.alias(name_like_string(label))
                        for scol, label in zip(data_spark_columns, column_labels)
                    ]
                spark_frame = spark_frame.select(data_spark_columns)

            assert not any(SPARK_INDEX_NAME_PATTERN.match(name) for name in spark_frame.columns), (
                "Index columns should not appear in columns of the Spark DataFrame. Avoid "
                "index column names [%s]." % SPARK_INDEX_NAME_PATTERN
            )

            # Create default index.
            spark_frame = InternalFrame.attach_default_index(spark_frame)
            index_spark_columns = [scol_for(spark_frame, SPARK_DEFAULT_INDEX_NAME)]

            if data_spark_columns is not None:
                data_spark_columns = [
                    scol_for(spark_frame, col)
                    for col in spark_frame.columns
                    if col != SPARK_DEFAULT_INDEX_NAME
                ]

        if NATURAL_ORDER_COLUMN_NAME not in spark_frame.columns:
            spark_frame = spark_frame.withColumn(
                NATURAL_ORDER_COLUMN_NAME, F.monotonically_increasing_id()
            )

        self._sdf = spark_frame  # type: spark.DataFrame

        # index_spark_columns
        assert all(
            isinstance(index_scol, spark.Column) for index_scol in index_spark_columns
        ), index_spark_columns

        self._index_spark_columns = index_spark_columns  # type: List[spark.Column]

        # index_names
        if not index_names:
            index_names = [None] * len(index_spark_columns)

        assert len(index_spark_columns) == len(index_names), (
            len(index_spark_columns),
            len(index_names),
        )
        assert all(
            is_name_like_tuple(index_name, check_type=True) for index_name in index_names
        ), index_names

        self._index_names = index_names  # type: List[Optional[Tuple]]

        # index_dtypes
        if not index_dtypes:
            index_dtypes = [None] * len(index_spark_columns)

        assert len(index_spark_columns) == len(index_dtypes), (
            len(index_spark_columns),
            len(index_dtypes),
        )

        index_dtypes = [
            spark_type_to_pandas_dtype(spark_frame.select(scol).schema[0].dataType)
            if dtype is None or dtype == np.dtype("object")
            else dtype
            for dtype, scol in zip(index_dtypes, index_spark_columns)
        ]

        assert all(
            isinstance(dtype, DtypeDataTypes)
            and (dtype == np.dtype("object") or as_spark_type(dtype, raise_error=False) is not None)
            for dtype in index_dtypes
        ), index_dtypes

        self._index_dtypes = index_dtypes

        # data_spark-columns
        if data_spark_columns is None:
            data_spark_columns = [
                scol_for(spark_frame, col)
                for col in spark_frame.columns
                if all(
                    not scol_for(spark_frame, col)._jc.equals(index_scol._jc)
                    for index_scol in index_spark_columns
                )
                and col not in HIDDEN_COLUMNS
            ]
            self._data_spark_columns = data_spark_columns
        else:
            assert all(isinstance(scol, spark.Column) for scol in data_spark_columns)
            self._data_spark_columns = data_spark_columns

        # column_labels
        if column_labels is None:
            self._column_labels = [
                (col,) for col in spark_frame.select(self._data_spark_columns).columns
            ]  # type: List[Tuple]
        else:
            assert len(column_labels) == len(self._data_spark_columns), (
                len(column_labels),
                len(self._data_spark_columns),
            )
            if len(column_labels) == 1:
                column_label = column_labels[0]
                assert is_name_like_tuple(column_label, check_type=True), column_label
            else:
                assert all(
                    is_name_like_tuple(column_label, check_type=True)
                    for column_label in column_labels
                ), column_labels
                assert len(set(len(label) for label in column_labels)) <= 1, column_labels
            self._column_labels = column_labels

        # data_dtypes
        if not data_dtypes:
            data_dtypes = [None] * len(data_spark_columns)

        assert len(data_spark_columns) == len(data_dtypes), (
            len(data_spark_columns),
            len(data_dtypes),
        )

        data_dtypes = [
            spark_type_to_pandas_dtype(spark_frame.select(scol).schema[0].dataType)
            if dtype is None or dtype == np.dtype("object")
            else dtype
            for dtype, scol in zip(data_dtypes, data_spark_columns)
        ]

        assert all(
            isinstance(dtype, DtypeDataTypes)
            and (dtype == np.dtype("object") or as_spark_type(dtype, raise_error=False) is not None)
            for dtype in data_dtypes
        ), data_dtypes

        self._data_dtypes = data_dtypes

        # column_label_names
        if column_label_names is None:
            self._column_label_names = [None] * column_labels_level(
                self._column_labels
            )  # type: List[Optional[Tuple]]
        else:
            if len(self._column_labels) > 0:
                assert len(column_label_names) == column_labels_level(self._column_labels), (
                    len(column_label_names),
                    column_labels_level(self._column_labels),
                )
            else:
                assert len(column_label_names) > 0, len(column_label_names)
            assert all(
                is_name_like_tuple(column_label_name, check_type=True)
                for column_label_name in column_label_names
            ), column_label_names
            self._column_label_names = column_label_names

    @staticmethod
    def attach_default_index(sdf, default_index_type=None):
        """
        This method attaches a default index to Spark DataFrame. Spark does not have the index
        notion so corresponding column should be generated.
        There are several types of default index can be configured by `compute.default_index_type`.

        >>> spark_frame = ks.range(10).to_spark()
        >>> spark_frame
        DataFrame[id: bigint]

        It adds the default index column '__index_level_0__'.

        >>> spark_frame = InternalFrame.attach_default_index(spark_frame)
        >>> spark_frame
        DataFrame[__index_level_0__: bigint, id: bigint]

        It throws an exception if the given column name already exists.

        >>> InternalFrame.attach_default_index(spark_frame)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        AssertionError: '__index_level_0__' already exists...
        """
        index_column = SPARK_DEFAULT_INDEX_NAME
        assert (
            index_column not in sdf.columns
        ), "'%s' already exists in the Spark column names '%s'" % (index_column, sdf.columns)

        if default_index_type is None:
            default_index_type = get_option("compute.default_index_type")

        if default_index_type == "sequence":
            return InternalFrame.attach_sequence_column(sdf, column_name=index_column)
        elif default_index_type == "distributed-sequence":
            return InternalFrame.attach_distributed_sequence_column(sdf, column_name=index_column)
        elif default_index_type == "distributed":
            return InternalFrame.attach_distributed_column(sdf, column_name=index_column)
        else:
            raise ValueError(
                "'compute.default_index_type' should be one of 'sequence',"
                " 'distributed-sequence' and 'distributed'"
            )

    @staticmethod
    def attach_sequence_column(sdf, column_name):
        scols = [scol_for(sdf, column) for column in sdf.columns]
        sequential_index = (
            F.row_number().over(Window.orderBy(F.monotonically_increasing_id())).cast("long") - 1
        )
        return sdf.select(sequential_index.alias(column_name), *scols)

    @staticmethod
    def attach_distributed_column(sdf, column_name):
        scols = [scol_for(sdf, column) for column in sdf.columns]
        return sdf.select(F.monotonically_increasing_id().alias(column_name), *scols)

    @staticmethod
    def attach_distributed_sequence_column(sdf, column_name):
        """
        This method attaches a Spark column that has a sequence in a distributed manner.
        This is equivalent to the column assigned when default index type 'distributed-sequence'.

        >>> sdf = ks.DataFrame(['a', 'b', 'c']).to_spark()
        >>> sdf = InternalFrame.attach_distributed_sequence_column(sdf, column_name="sequence")
        >>> sdf.show()  # doctest: +NORMALIZE_WHITESPACE
        +--------+---+
        |sequence|  0|
        +--------+---+
        |       0|  a|
        |       1|  b|
        |       2|  c|
        +--------+---+
        """
        if len(sdf.columns) > 0:
            try:
                jdf = sdf._jdf.toDF()

                sql_ctx = sdf.sql_ctx
                encoders = sql_ctx._jvm.org.apache.spark.sql.Encoders
                encoder = encoders.tuple(jdf.exprEnc(), encoders.scalaLong())

                jrdd = jdf.localCheckpoint(False).rdd().zipWithIndex()

                df = spark.DataFrame(
                    sql_ctx.sparkSession._jsparkSession.createDataset(jrdd, encoder).toDF(), sql_ctx
                )
                columns = df.columns
                return df.selectExpr(
                    "`{}` as `{}`".format(columns[1], column_name), "`{}`.*".format(columns[0])
                )
            except py4j.protocol.Py4JError:
                if is_testing():
                    raise
                return InternalFrame._attach_distributed_sequence_column(sdf, column_name)
        else:
            cnt = sdf.count()
            if cnt > 0:
                return default_session().range(cnt).toDF(column_name)
            else:
                return default_session().createDataFrame(
                    [], schema=StructType().add(column_name, data_type=LongType(), nullable=False)
                )

    @staticmethod
    def _attach_distributed_sequence_column(sdf, column_name):
        """
        >>> sdf = ks.DataFrame(['a', 'b', 'c']).to_spark()
        >>> sdf = InternalFrame._attach_distributed_sequence_column(sdf, column_name="sequence")
        >>> sdf.sort("sequence").show()  # doctest: +NORMALIZE_WHITESPACE
        +--------+---+
        |sequence|  0|
        +--------+---+
        |       0|  a|
        |       1|  b|
        |       2|  c|
        +--------+---+
        """
        scols = [scol_for(sdf, column) for column in sdf.columns]

        spark_partition_column = verify_temp_column_name(sdf, "__spark_partition_id__")
        offset_column = verify_temp_column_name(sdf, "__offset__")
        row_number_column = verify_temp_column_name(sdf, "__row_number__")

        # 1. Calculates counts per each partition ID. `counts` here is, for instance,
        #     {
        #         1: 83,
        #         6: 83,
        #         3: 83,
        #         ...
        #     }
        sdf = sdf.withColumn(spark_partition_column, F.spark_part