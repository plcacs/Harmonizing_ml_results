"""
An internal immutable DataFrame with some metadata to manage indexes.
"""
from distutils.version import LooseVersion
import re
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Any, Sequence, cast
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
    from pyspark.sql.pandas.types import to_arrow_type
from databricks import koalas as ks
if TYPE_CHECKING:
    from databricks.koalas.series import Series
from databricks.koalas.config import get_option
from databricks.koalas.typedef import Dtype, DtypeDataTypes, as_spark_type, extension_dtypes, infer_pd_series_spark_type, spark_type_to_pandas_dtype
from databricks.koalas.utils import column_labels_level, default_session, is_name_like_tuple, is_testing, lazy_property, name_like_string, scol_for, verify_temp_column_name

SPARK_INDEX_NAME_FORMAT = '__index_level_{}__'.format
SPARK_DEFAULT_INDEX_NAME = SPARK_INDEX_NAME_FORMAT(0)
SPARK_INDEX_NAME_PATTERN = re.compile('__index_level_[0-9]+__')
NATURAL_ORDER_COLUMN_NAME = '__natural_order__'
HIDDEN_COLUMNS = {NATURAL_ORDER_COLUMN_NAME}
DEFAULT_SERIES_NAME = 0
SPARK_DEFAULT_SERIES_NAME = str(DEFAULT_SERIES_NAME)

class InternalFrame(object):
    """
    The internal immutable DataFrame which manages Spark DataFrame and column names and index
    information.
    """

    def __init__(
        self,
        spark_frame: spark.DataFrame,
        index_spark_columns: List[spark.Column],
        index_names: Optional[List[Optional[Tuple[Any, ...]]]] = None,
        index_dtypes: Optional[List[Dtype]] = None,
        column_labels: Optional[List[Tuple[Any, ...]]] = None,
        data_spark_columns: Optional[List[spark.Column]] = None,
        data_dtypes: Optional[List[Dtype]] = None,
        column_label_names: Optional[List[Optional[Tuple[Any, ...]]]] = None
    ) -> None:
        """
        Create a new internal immutable DataFrame to manage Spark DataFrame, column fields and
        index fields and names.
        """
        assert isinstance(spark_frame, spark.DataFrame)
        assert not spark_frame.isStreaming, 'Koalas does not support Structured Streaming.'
        if not index_spark_columns:
            if data_spark_columns is not None:
                if column_labels is not None:
                    data_spark_columns = [scol.alias(name_like_string(label)) for scol, label in zip(data_spark_columns, column_labels)]
                spark_frame = spark_frame.select(data_spark_columns)
            assert not any((SPARK_INDEX_NAME_PATTERN.match(name) for name in spark_frame.columns)), 'Index columns should not appear in columns of the Spark DataFrame. Avoid index column names [%s].' % SPARK_INDEX_NAME_PATTERN
            spark_frame = InternalFrame.attach_default_index(spark_frame)
            index_spark_columns = [scol_for(spark_frame, SPARK_DEFAULT_INDEX_NAME)]
            if data_spark_columns is not None:
                data_spark_columns = [scol_for(spark_frame, col) for col in spark_frame.columns if col != SPARK_DEFAULT_INDEX_NAME]
        if NATURAL_ORDER_COLUMN_NAME not in spark_frame.columns:
            spark_frame = spark_frame.withColumn(NATURAL_ORDER_COLUMN_NAME, F.monotonically_increasing_id())
        self._sdf = spark_frame
        assert all((isinstance(index_scol, spark.Column) for index_scol in index_spark_columns)), index_spark_columns
        self._index_spark_columns = index_spark_columns
        if not index_names:
            index_names = [None] * len(index_spark_columns)
        assert len(index_spark_columns) == len(index_names), (len(index_spark_columns), len(index_names))
        assert all((is_name_like_tuple(index_name, check_type=True) for index_name in index_names)), index_names
        self._index_names = index_names
        if not index_dtypes:
            index_dtypes = [None] * len(index_spark_columns)
        assert len(index_spark_columns) == len(index_dtypes), (len(index_spark_columns), len(index_dtypes))
        index_dtypes = [spark_type_to_pandas_dtype(spark_frame.select(scol).schema[0].dataType) if dtype is None or dtype == np.dtype('object') else dtype for dtype, scol in zip(index_dtypes, index_spark_columns)]
        assert all((isinstance(dtype, DtypeDataTypes) and (dtype == np.dtype('object') or as_spark_type(dtype, raise_error=False) is not None) for dtype in index_dtypes)), index_dtypes
        self._index_dtypes = index_dtypes
        if data_spark_columns is None:
            data_spark_columns = [scol_for(spark_frame, col) for col in spark_frame.columns if all((not scol_for(spark_frame, col)._jc.equals(index_scol._jc) for index_scol in index_spark_columns)) and col not in HIDDEN_COLUMNS]
            self._data_spark_columns = data_spark_columns
        else:
            assert all((isinstance(scol, spark.Column) for scol in data_spark_columns)
            self._data_spark_columns = data_spark_columns
        if column_labels is None:
            self._column_labels = [(col,) for col in spark_frame.select(self._data_spark_columns).columns]
        else:
            assert len(column_labels) == len(self._data_spark_columns), (len(column_labels), len(self._data_spark_columns))
            if len(column_labels) == 1:
                column_label = column_labels[0]
                assert is_name_like_tuple(column_label, check_type=True), column_label
            else:
                assert all((is_name_like_tuple(column_label, check_type=True) for column_label in column_labels)), column_labels
                assert len(set((len(label) for label in column_labels))) <= 1, column_labels
            self._column_labels = column_labels
        if not data_dtypes:
            data_dtypes = [None] * len(data_spark_columns)
        assert len(data_spark_columns) == len(data_dtypes), (len(data_spark_columns), len(data_dtypes))
        data_dtypes = [spark_type_to_pandas_dtype(spark_frame.select(scol).schema[0].dataType) if dtype is None or dtype == np.dtype('object') else dtype for dtype, scol in zip(data_dtypes, data_spark_columns)]
        assert all((isinstance(dtype, DtypeDataTypes) and (dtype == np.dtype('object') or as_spark_type(dtype, raise_error=False) is not None) for dtype in data_dtypes)), data_dtypes
        self._data_dtypes = data_dtypes
        if column_label_names is None:
            self._column_label_names = [None] * column_labels_level(self._column_labels)
        else:
            if len(self._column_labels) > 0:
                assert len(column_label_names) == column_labels_level(self._column_labels), (len(column_label_names), column_labels_level(self._column_labels))
            else:
                assert len(column_label_names) > 0, len(column_label_names)
            assert all((is_name_like_tuple(column_label_name, check_type=True) for column_label_name in column_label_names)), column_label_names
            self._column_label_names = column_label_names

    @staticmethod
    def attach_default_index(sdf: spark.DataFrame, default_index_type: Optional[str] = None) -> spark.DataFrame:
        """
        This method attaches a default index to Spark DataFrame.
        """
        index_column = SPARK_DEFAULT_INDEX_NAME
        assert index_column not in sdf.columns, "'%s' already exists in the Spark column names '%s'" % (index_column, sdf.columns)
        if default_index_type is None:
            default_index_type = get_option('compute.default_index_type')
        if default_index_type == 'sequence':
            return InternalFrame.attach_sequence_column(sdf, column_name=index_column)
        elif default_index_type == 'distributed-sequence':
            return InternalFrame.attach_distributed_sequence_column(sdf, column_name=index_column)
        elif default_index_type == 'distributed':
            return InternalFrame.attach_distributed_column(sdf, column_name=index_column)
        else:
            raise ValueError("'compute.default_index_type' should be one of 'sequence', 'distributed-sequence' and 'distributed'")

    @staticmethod
    def attach_sequence_column(sdf: spark.DataFrame, column_name: str) -> spark.DataFrame:
        scols = [scol_for(sdf, column) for column in sdf.columns]
        sequential_index = F.row_number().over(Window.orderBy(F.monotonically_increasing_id())).cast('long') - 1
        return sdf.select(sequential_index.alias(column_name), *scols)

    @staticmethod
    def attach_distributed_column(sdf: spark.DataFrame, column_name: str) -> spark.DataFrame:
        scols = [scol_for(sdf, column) for column in sdf.columns]
        return sdf.select(F.monotonically_increasing_id().alias(column_name), *scols)

    @staticmethod
    def attach_distributed_sequence_column(sdf: spark.DataFrame, column_name: str) -> spark.DataFrame:
        """
        This method attaches a Spark column that has a sequence in a distributed manner.
        """
        if len(sdf.columns) > 0:
            try:
                jdf = sdf._jdf.toDF()
                sql_ctx = sdf.sql_ctx
                encoders = sql_ctx._jvm.org.apache.spark.sql.Encoders
                encoder = encoders.tuple(jdf.exprEnc(), encoders.scalaLong())
                jrdd = jdf.localCheckpoint(False).rdd().zipWithIndex()
                df = spark.DataFrame(sql_ctx.sparkSession._jsparkSession.createDataset(jrdd, encoder).toDF(), sql_ctx)
                columns = df.columns
                return df.selectExpr('`{}` as `{}`'.format(columns[1], column_name), '`{}`.*'.format(columns[0]))
            except py4j.protocol.Py4JError:
                if is_testing():
                    raise
                return InternalFrame._attach_distributed_sequence_column(sdf, column_name)
        else:
            cnt = sdf.count()
            if cnt > 0:
                return default_session().range(cnt).toDF(column_name)
            else:
                return default_session().createDataFrame([], schema=StructType().add(column_name, data_type=LongType(), nullable=False))

    @staticmethod
    def _attach_distributed_sequence_column(sdf: spark.DataFrame, column_name: str) -> spark.DataFrame:
        """
        Fallback method for attach_distributed_sequence_column
        """
        scols = [scol_for(sdf, column) for column in sdf.columns]
        spark_partition_column = verify_temp_column_name(sdf, '__spark_partition_id__')
        offset_column = verify_temp_column_name(sdf, '__offset__')
        row_number_column = verify_temp_column_name(sdf, '__row_number__')
        sdf = sdf.withColumn(spark_partition_column, F.spark_partition_id())
        sdf = sdf.localCheckpoint(eager=False)
        counts = map(lambda x: (x['key'], x['count']), sdf.groupby(sdf[spark_partition_column].alias('key')).count().collect())
        sorted_counts = sorted(counts, key=lambda x: x[0])
        cumulative_counts = [0] + list(accumulate(map(lambda count: count[1], sorted_counts)))
        sums = dict(zip(map(lambda count: count[0], sorted_counts), cumulative_counts))

        @pandas_udf(LongType(), PandasUDFType.SCALAR)
        def offset(id: pd.Series) -> pd.Series:
            current_partition_offset = sums[id.iloc[0]]
            return pd.Series(current_partition_offset).repeat(len(id))
        sdf = sdf.withColumn(offset_column, offset(spark_partition_column))
        w = Window.partitionBy(spark_partition_column).orderBy(F.monotonically_increasing_id())
        row_number = F.row_number().over(w)
        sdf = sdf.withColumn(row_number_column, row_number)
        return sdf.select((sdf[offset_column] + sdf[row_number_column] - 1).alias(column_name), *scols)

    def spark_column_for(self, label: Tuple[Any, ...]) -> spark.Column:
        """ Return Spark Column for the given column label. """
        column_labels_to_scol = dict(zip(self.column_labels, self.data_spark_columns))
        if label in column_labels_to_scol:
            return column_labels_to_scol[label]
        else:
            raise KeyError(name_like_string(label))

    def spark_column_name_for(self, label_or_scol: Union[Tuple[Any, ...], spark.Column]) -> str:
        """ Return the actual Spark column name for the given column label. """
        if isinstance(label_or_scol, spark.Column):
            scol = label_or_scol
        else:
            scol = self.spark_column_for(label_or_scol)
        return self.spark_frame.select(scol).columns[0]

    def spark_type_for(self, label_or_scol: Union[Tuple[Any, ...], spark.Column]) -> DataType:
        """ Return DataType for the given column label. """
        if isinstance(label_or_scol, spark.Column):
            scol = label_or_scol
        else:
            scol = self.spark_column_for(label_or_scol)
        return self.spark_frame.select(scol).schema[0].dataType

    def spark_column_nullable_for(self, label_or_scol: Union[Tuple[Any, ...], spark.Column]) -> bool:
        """ Return nullability for the given column label. """
        if isinstance(label_or_scol, spark.Column):
            scol = label_or_scol
        else:
            scol = self.spark_column_for(label_or_scol)
        return self.spark_frame.select(scol).schema[0].nullable

    def dtype_for(self, label: Tuple[Any, ...]) -> Dtype:
        """ Return dtype for the given column label. """
        column_labels_to_dtype = dict(zip(self.column_labels, self.data_dtypes))
        if label in column_labels_to_dtype:
            return column_labels_to_dtype[label]
        else:
            raise KeyError(name_like_string(label))

    @property
    def spark_frame(self) -> spark.DataFrame:
        """ Return the managed Spark DataFrame. """
        return self._sdf

    @lazy_property
    def data_spark_column_names(self) -> List[str]:
        """ Return the managed column field names. """
        return self.spark_frame.select(self.data_spark_columns).columns

    @property
    def data_spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed data columns. """
        return self._data_spark_columns

    @property
    def index_spark_column_names(self) -> List[str]:
        """ Return the managed index field names. """
        return self.spark_frame.select(self.index_spark_columns).columns

    @property
    def index_spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed index columns. """
        return self._index_spark_columns

    @lazy_property
    def spark_column_names(self) -> List[str]:
        """ Return all the field names including index field names. """
        return self.spark_frame.select(self.spark_columns).columns

    @lazy_property
    def spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed columns including index columns. """
        index_spark_columns = self.index_spark_columns
        return index_spark_columns + [spark_column for spark_column in self.data_spark_columns if all((not spark_column._jc.equals(scol._jc) for scol in index_spark_columns))]

    @property
    def index_names(self) -> List[Optional[Tuple[Any, ...]]]:
        """ Return the managed index names. """
        return self._index_names

    @lazy_property
    def index_level(self) -> int:
        """ Return the level of the index. """
        return len(self._index_names)

    @property
    def column_labels(self) -> List[Tuple[Any, ...]]:
        """ Return the managed column index. """
        return self._column_labels

    @lazy_property
    def column_labels_level(self) -> int:
        """ Return the level of the column index. """
        return len(self._column_label_names)

    @property
    def column_label_names(self) -> List[Optional[Tuple[Any, ...]]]:
        """ Return names of the index levels. """
        return self._column_label_names

    @property
    def index_dtypes(self) -> List[Dtype]:
        """ Return dtypes for the managed index columns. """
        return self._index_dtypes

    @property
    def data_dtypes(self) -> List[Dtype]:
        """ Return dtypes for the managed columns. """
        return self._data_dtypes

    @lazy_property
    def to_internal_spark_frame(self) -> spark.DataFrame:
        """
        Return as Spark DataFrame. This contains index columns as well
        and should be only used for internal purposes.
        """
        index_spark_columns = self.index_spark_columns
        data_columns = []
        for spark_column in self.data_spark_columns:
           