"""
An internal immutable DataFrame with some metadata to manage indexes.
"""
from distutils.version import LooseVersion
import re
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Any
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

SPARK_INDEX_NAME_FORMAT = "__index_level_{}__".format
SPARK_DEFAULT_INDEX_NAME = SPARK_INDEX_NAME_FORMAT(0)
SPARK_INDEX_NAME_PATTERN = re.compile("__index_level_[0-9]+__")
NATURAL_ORDER_COLUMN_NAME = "__natural_order__"
HIDDEN_COLUMNS = {NATURAL_ORDER_COLUMN_NAME}
DEFAULT_SERIES_NAME = 0
SPARK_DEFAULT_SERIES_NAME = str(DEFAULT_SERIES_NAME)


class InternalFrame:
    """
    The internal immutable DataFrame which manages Spark DataFrame and column names and index
    information.

    .. note:: this is an internal class. It is not supposed to be exposed to users and users
        should not directly access to it.
    
    [Docstring omitted for brevity]
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
        column_label_names: Optional[List[Optional[Tuple[Any, ...]]]] = None,
    ) -> None:
        """
        Create a new internal immutable DataFrame to manage Spark DataFrame, column fields and
        index fields and names.

        [Docstring omitted for brevity]
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
            assert not any(
                (SPARK_INDEX_NAME_PATTERN.match(name) for name in spark_frame.columns)
            ), (
                "Index columns should not appear in columns of the Spark DataFrame. "
                "Avoid index column names [%s]." % SPARK_INDEX_NAME_PATTERN
            )
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
        self._sdf: spark.DataFrame = spark_frame
        assert all(
            isinstance(index_scol, spark.Column) for index_scol in index_spark_columns
        ), index_spark_columns
        self._index_spark_columns: List[spark.Column] = index_spark_columns
        if not index_names:
            index_names = [None] * len(index_spark_columns)
        assert len(index_spark_columns) == len(index_names), (
            len(index_spark_columns),
            len(index_names),
        )
        assert all(
            is_name_like_tuple(index_name, check_type=True) for index_name in index_names
        ), index_names
        self._index_names: List[Optional[Tuple[Any, ...]]] = index_names
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
        self._index_dtypes: List[Dtype] = index_dtypes
        if data_spark_columns is None:
            self._data_spark_columns: List[spark.Column] = [
                scol_for(spark_frame, col)
                for col in spark_frame.columns
                if all(
                    not scol_for(spark_frame, col)._jc.equals(index_scol._jc)
                    for index_scol in index_spark_columns
                )
                and col not in HIDDEN_COLUMNS
            ]
        else:
            assert all(isinstance(scol, spark.Column) for scol in data_spark_columns)
            self._data_spark_columns: List[spark.Column] = data_spark_columns
        if column_labels is None:
            self._column_labels: List[Tuple[Any, ...]] = [
                (col,) for col in spark_frame.select(self._data_spark_columns).columns
            ]
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
                    is_name_like_tuple(column_label, check_type=True) for column_label in column_labels
                ), column_labels
                assert len(set(len(label) for label in column_labels)) <= 1, column_labels
            self._column_labels: List[Tuple[Any, ...]] = column_labels
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
        self._data_dtypes: List[Dtype] = data_dtypes
        if column_label_names is None:
            self._column_label_names: List[Optional[Tuple[Any, ...]]] = [
                None
            ] * column_labels_level(self._column_labels)
        else:
            if len(self._column_labels) > 0:
                assert len(column_label_names) == column_labels_level(
                    self._column_labels
                ), (
                    len(column_label_names),
                    column_labels_level(self._column_labels),
                )
            else:
                assert len(column_label_names) > 0, len(column_label_names)
            assert all(
                is_name_like_tuple(column_label_name, check_type=True)
                for column_label_name in column_label_names
            ), column_label_names
            self._column_label_names: List[Optional[Tuple[Any, ...]]] = column_label_names

    @staticmethod
    def attach_default_index(
        sdf: spark.DataFrame, default_index_type: Optional[str] = None
    ) -> spark.DataFrame:
        """
        This method attaches a default index to Spark DataFrame. Spark does not have the index
        notion so corresponding column should be generated.
        There are several types of default index can be configured by `compute.default_index_type`.

        [Docstring omitted for brevity]
        """
        index_column: str = SPARK_DEFAULT_INDEX_NAME
        assert index_column not in sdf.columns, (
            "'%s' already exists in the Spark column names '%s'" % (index_column, sdf.columns)
        )
        if default_index_type is None:
            default_index_type = get_option("compute.default_index_type")
        if default_index_type == "sequence":
            return InternalFrame.attach_sequence_column(sdf, column_name=index_column)
        elif default_index_type == "distributed-sequence":
            return InternalFrame.attach_distributed_sequence_column(
                sdf, column_name=index_column
            )
        elif default_index_type == "distributed":
            return InternalFrame.attach_distributed_column(sdf, column_name=index_column)
        else:
            raise ValueError(
                "'compute.default_index_type' should be one of 'sequence', 'distributed-sequence' and 'distributed'"
            )

    @staticmethod
    def attach_sequence_column(sdf: spark.DataFrame, column_name: str) -> spark.DataFrame:
        scols: List[spark.Column] = [scol_for(sdf, column) for column in sdf.columns]
        sequential_index: spark.Column = (
            F.row_number()
            .over(Window.orderBy(F.monotonically_increasing_id()))
            .cast("long")
            - 1
        )
        return sdf.select(sequential_index.alias(column_name), *scols)

    @staticmethod
    def attach_distributed_column(sdf: spark.DataFrame, column_name: str) -> spark.DataFrame:
        scols: List[spark.Column] = [scol_for(sdf, column) for column in sdf.columns]
        return sdf.select(F.monotonically_increasing_id().alias(column_name), *scols)

    @staticmethod
    def attach_distributed_sequence_column(
        sdf: spark.DataFrame, column_name: str
    ) -> spark.DataFrame:
        """
        This method attaches a Spark column that has a sequence in a distributed manner.
        This is equivalent to the column assigned when default index type 'distributed-sequence'.

        [Docstring omitted for brevity]
        """
        if len(sdf.columns) > 0:
            try:
                jdf = sdf._jdf.toDF()
                sql_ctx = sdf.sql_ctx
                encoders = sql_ctx._jvm.org.apache.spark.sql.Encoders
                encoder = encoders.tuple(jdf.exprEnc(), encoders.scalaLong())
                jrdd = jdf.localCheckpoint(False).rdd().zipWithIndex()
                df = spark.DataFrame(
                    sql_ctx.sparkSession._jsparkSession.createDataset(jrdd, encoder).toDF(),
                    sql_ctx,
                )
                columns: List[str] = df.columns
                return df.selectExpr(
                    "`{}` as `{}`".format(columns[1], column_name), "`{}`.*".format(columns[0])
                )
            except py4j.protocol.Py4JError:
                if is_testing():
                    raise
                return InternalFrame._attach_distributed_sequence_column(sdf, column_name)
        else:
            cnt: int = sdf.count()
            if cnt > 0:
                return default_session().range(cnt).toDF(column_name)
            else:
                return default_session().createDataFrame(
                    [],
                    schema=StructType().add(column_name, data_type=LongType(), nullable=False),
                )

    @staticmethod
    def _attach_distributed_sequence_column(
        sdf: spark.DataFrame, column_name: str
    ) -> spark.DataFrame:
        """
        [Docstring omitted for brevity]
        """
        scols: List[spark.Column] = [scol_for(sdf, column) for column in sdf.columns]
        spark_partition_column: str = verify_temp_column_name(
            sdf, "__spark_partition_id__"
        )
        offset_column: str = verify_temp_column_name(sdf, "__offset__")
        row_number_column: str = verify_temp_column_name(sdf, "__row_number__")
        sdf = sdf.withColumn(spark_partition_column, F.spark_partition_id())
        sdf = sdf.localCheckpoint(eager=False)
        counts = map(
            lambda x: (x["key"], x["count"]),
            sdf.groupby(sdf[spark_partition_column].alias("key")).count().collect(),
        )
        sorted_counts: List[Tuple[Any, Any]] = sorted(counts, key=lambda x: x[0])
        cumulative_counts: List[int] = [0] + list(
            accumulate(map(lambda count: count[1], sorted_counts))
        )
        sums: Dict[Any, int] = dict(
            zip(map(lambda count: count[0], sorted_counts), cumulative_counts)
        )

        @pandas_udf(LongType(), PandasUDFType.SCALAR)
        def offset(id: pd.Series) -> pd.Series:
            current_partition_offset: int = sums[id.iloc[0]]
            return pd.Series(current_partition_offset).repeat(len(id))

        sdf = sdf.withColumn(offset_column, offset(spark_partition_column))
        w: WindowSpec = Window.partitionBy(spark_partition_column).orderBy(
            F.monotonically_increasing_id()
        )
        row_number: spark.Column = F.row_number().over(w)
        sdf = sdf.withColumn(row_number_column, row_number)
        return sdf.select(
            (sdf[offset_column] + sdf[row_number_column] - 1).alias(column_name),
            *scols,
        )

    def spark_column_for(self, label: Tuple[Any, ...]) -> spark.Column:
        """ Return Spark Column for the given column label. """
        column_labels_to_scol: Dict[Tuple[Any, ...], spark.Column] = dict(
            zip(self.column_labels, self.data_spark_columns)
        )
        if label in column_labels_to_scol:
            return column_labels_to_scol[label]
        else:
            raise KeyError(name_like_string(label))

    def spark_column_name_for(self, label_or_scol: Union[Tuple[Any, ...], spark.Column]) -> str:
        """ Return the actual Spark column name for the given column label. """
        if isinstance(label_or_scol, spark.Column):
            scol: spark.Column = label_or_scol
        else:
            scol = self.spark_column_for(label_or_scol)
        return self.spark_frame.select(scol).columns[0]

    def spark_type_for(self, label_or_scol: Union[Tuple[Any, ...], spark.Column]) -> DataType:
        """ Return DataType for the given column label. """
        if isinstance(label_or_scol, spark.Column):
            scol: spark.Column = label_or_scol
        else:
            scol = self.spark_column_for(label_or_scol)
        return self.spark_frame.select(scol).schema[0].dataType

    def spark_column_nullable_for(self, label_or_scol: Union[Tuple[Any, ...], spark.Column]) -> bool:
        """ Return nullability for the given column label. """
        if isinstance(label_or_scol, spark.Column):
            scol: spark.Column = label_or_scol
        else:
            scol = self.spark_column_for(label_or_scol)
        return self.spark_frame.select(scol).schema[0].nullable

    def dtype_for(self, label: Tuple[Any, ...]) -> Dtype:
        """ Return dtype for the given column label. """
        column_labels_to_dtype: Dict[Tuple[Any, ...], Dtype] = dict(
            zip(self.column_labels, self.data_dtypes)
        )
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
        index_spark_columns: List[spark.Column] = self.index_spark_columns
        return index_spark_columns + [
            spark_column
            for spark_column in self.data_spark_columns
            if all(
                not spark_column._jc.equals(scol._jc) for scol in index_spark_columns
            )
        ]

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
        index_spark_columns: List[spark.Column] = self.index_spark_columns
        data_columns: List[spark.Column] = []
        for spark_column in self.data_spark_columns:
            if all(
                not spark_column._jc.equals(scol._jc) for scol in index_spark_columns
            ):
                data_columns.append(spark_column)
        return self.spark_frame.select(index_spark_columns + data_columns)

    @lazy_property
    def to_pandas_frame(self) -> pd.DataFrame:
        """ Return as pandas DataFrame. """
        sdf: spark.DataFrame = self.to_internal_spark_frame
        pdf: pd.DataFrame = sdf.toPandas()
        if len(pdf) == 0 and len(sdf.schema) > 0:
            pdf = pdf.astype(
                {
                    field.name: spark_type_to_pandas_dtype(field.dataType)
                    for field in sdf.schema
                }
            )
        elif LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
            for field in sdf.schema:
                if field.nullable and pdf[field.name].isnull().all():
                    if isinstance(field.dataType, BooleanType):
                        pdf[field.name] = pdf[field.name].astype(np.object)
                    elif isinstance(field.dataType, IntegralType):
                        pdf[field.name] = pdf[field.name].astype(np.float64)
                    else:
                        pdf[field.name] = pdf[field.name].astype(
                            spark_type_to_pandas_dtype(field.dataType)
                        )
        return InternalFrame.restore_index(pdf, **self.arguments_for_restore_index)

    @lazy_property
    def arguments_for_restore_index(self) -> Dict[str, Any]:
        """ Create arguments for `restore_index`. """
        column_names: List[Any] = []
        ext_dtypes: Dict[str, Any] = {
            col: dtype
            for col, dtype in zip(self.index_spark_column_names, self.index_dtypes)
            if isinstance(dtype, extension_dtypes)
        }
        categorical_dtypes: Dict[str, CategoricalDtype] = {
            col: dtype
            for col, dtype in zip(self.index_spark_column_names, self.index_dtypes)
            if isinstance(dtype, CategoricalDtype)
        }
        for spark_column, column_name, dtype in zip(
            self.data_spark_columns, self.data_spark_column_names, self.data_dtypes
        ):
            for index_spark_column_name, index_spark_column in zip(
                self.index_spark_column_names, self.index_spark_columns
            ):
                if spark_column._jc.equals(index_spark_column._jc):
                    column_names.append(index_spark_column_name)
                    break
            else:
                column_names.append(column_name)
                if isinstance(dtype, extension_dtypes):
                    ext_dtypes[column_name] = dtype
                elif isinstance(dtype, CategoricalDtype):
                    categorical_dtypes[column_name] = dtype
        return {
            "index_columns": self.index_spark_column_names,
            "index_names": self.index_names,
            "data_columns": column_names,
            "column_labels": self.column_labels,
            "column_label_names": self.column_label_names,
            "ext_dtypes": ext_dtypes,
            "categorical_dtypes": categorical_dtypes,
        }

    @staticmethod
    def restore_index(
        pdf: pd.DataFrame,
        *,
        index_columns: List[str],
        index_names: List[Optional[Tuple[Any, ...]]],
        data_columns: List[str],
        column_labels: List[Tuple[Any, ...]],
        column_label_names: List[Optional[Tuple[Any, ...]]],
        ext_dtypes: Optional[Dict[str, Any]] = None,
        categorical_dtypes: Optional[Dict[str, CategoricalDtype]] = None,
    ) -> pd.DataFrame:
        """
        Restore pandas DataFrame indices using the metadata.

        [Docstring omitted for brevity]
        """
        if ext_dtypes is not None and len(ext_dtypes) > 0:
            pdf = pdf.astype(ext_dtypes, copy=True)
        if categorical_dtypes is not None:
            for col, dtype in categorical_dtypes.items():
                pdf[col] = pd.Categorical.from_codes(
                    pdf[col], categories=dtype.categories, ordered=dtype.ordered
                )
        append: bool = False
        for index_field in index_columns:
            drop: bool = index_field not in data_columns
            pdf = pdf.set_index(index_field, drop=drop, append=append)
            append = True
        pdf = pdf[data_columns]
        pdf.index.names = [
            name if name is None or len(name) > 1 else name[0]
            for name in index_names
        ]
        names: List[Optional[Tuple[Any, ...]]] = [
            name if name is None or len(name) > 1 else name[0]
            for name in column_label_names
        ]
        if len(column_label_names) > 1:
            pdf.columns = pd.MultiIndex.from_tuples(column_labels, names=names)
        else:
            pdf.columns = pd.Index(
                [None if label is None else label[0] for label in column_labels],
                name=names[0],
            )
        return pdf

    @lazy_property
    def resolved_copy(self) -> "InternalFrame":
        """ Copy the immutable InternalFrame with the updates resolved. """
        sdf: spark.DataFrame = self.spark_frame.select(
            self.spark_columns + list(HIDDEN_COLUMNS)
        )
        return self.copy(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self.index_spark_column_names],
            data_spark_columns=[scol_for(sdf, col) for col in self.data_spark_column_names],
        )

    def with_new_sdf(
        self,
        spark_frame: spark.DataFrame,
        *,
        index_dtypes: Optional[List[Dtype]] = None,
        data_columns: Optional[List[str]] = None,
        data_dtypes: Optional[List[Dtype]] = None,
    ) -> "InternalFrame":
        """ Copy the immutable InternalFrame with the updates by the specified Spark DataFrame.

        :param spark_frame: the new Spark DataFrame
        :param index_dtypes: the index dtypes. If None, the original dtyeps are used.
        :param data_columns: the new column names. If None, the original one is used.
        :param data_dtypes: the data dtypes. If None, the original dtyeps are used.
        :return: the copied InternalFrame.
        """
        if index_dtypes is None:
            index_dtypes = self.index_dtypes
        else:
            assert len(index_dtypes) == len(self.index_dtypes), (
                len(index_dtypes),
                len(self.index_dtypes),
            )
        if data_columns is None:
            data_columns = self.data_spark_column_names
        else:
            assert len(data_columns) == len(self.column_labels), (
                len(data_columns),
                len(self.column_labels),
            )
        if data_dtypes is None:
            data_dtypes = self.data_dtypes
        else:
            assert len(data_dtypes) == len(self.column_labels), (
                len(data_dtypes),
                len(self.column_labels),
            )
        sdf: spark.DataFrame = spark_frame.drop(NATURAL_ORDER_COLUMN_NAME)
        return self.copy(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self.index_spark_column_names],
            index_dtypes=index_dtypes,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            data_dtypes=data_dtypes,
        )

    def with_new_columns(
        self,
        scols_or_ksers: List[Union[spark.Column, "Series"]],
        *,
        column_labels: Optional[List[Tuple[Any, ...]]] = None,
        data_dtypes: Optional[List[Dtype]] = None,
        column_label_names: Union[List[Optional[Tuple[Any, ...]]], _NoValueType] = _NoValue,
        keep_order: bool = True,
    ) -> "InternalFrame":
        """
        Copy the immutable InternalFrame with the updates by the specified Spark Columns or Series.

        :param scols_or_ksers: the new Spark Columns or Series.
        :param column_labels: the new column index.
            If None, the column_labels of the corresponding `scols_or_ksers` is used if it is
            Series; otherwise the original one is used.
        :param data_dtypes: the new dtypes.
            If None, the dtypes of the corresponding `scols_or_ksers` is used if it is Series;
            otherwise the dtypes will be inferred from the corresponding `scols_or_ksers`.
        :param column_label_names: the new names of the column index levels.
        :return: the copied InternalFrame.
        """
        from databricks.koalas.series import Series

        if column_labels is None:
            if all(isinstance(scol_or_kser, Series) for scol_or_kser in scols_or_ksers):
                column_labels = [kser._column_label for kser in scols_or_ksers]
            else:
                assert len(scols_or_ksers) == len(self.column_labels), (
                    len(scols_or_kser),
                    len(self.column_labels),
                )
                column_labels: List[Tuple[Any, ...]] = []
                for scol_or_kser, label in zip(scols_or_ksers, self.column_labels):
                    if isinstance(scol_or_kser, Series):
                        column_labels.append(scol_or_kser._column_label)
                    else:
                        column_labels.append(label)
        else:
            assert len(scols_or_ksers) == len(column_labels), (
                len(scols_or_kser),
                len(column_labels),
            )
        data_spark_columns: List[spark.Column] = []
        for scol_or_kser in scols_or_ksers:
            if isinstance(scol_or_kser, Series):
                scol: spark.Column = scol_or_kser.spark.column
            else:
                scol = scol_or_kser
            data_spark_columns.append(scol)
        if data_dtypes is None:
            data_dtypes = []
            for scol_or_kser in scols_or_ksers:
                if isinstance(scol_or_kser, Series):
                    data_dtypes.append(scol_or_kser.dtype)
                else:
                    data_dtypes.append(None)
        else:
            assert len(scols_or_ksers) == len(data_dtypes), (
                len(scols_or_kser),
                len(data_dtypes),
            )
        sdf: spark.DataFrame = self.spark_frame
        if not keep_order:
            sdf = self.spark_frame.select(self.index_spark_columns + data_spark_columns)
            index_spark_columns: List[spark.Column] = [
                scol_for(sdf, col) for col in self.index_spark_column_names
            ]
            data_spark_columns = [
                scol_for(sdf, col) for col in self.spark_frame.select(data_spark_columns).columns
            ]
        else:
            index_spark_columns = self.index_spark_columns
        if column_label_names is _NoValue:
            column_label_names_final: List[Optional[Tuple[Any, ...]]] = self._column_label_names
        else:
            column_label_names_final = column_label_names
        return self.copy(
            spark_frame=sdf,
            index_spark_columns=index_spark_columns,
            column_labels=column_labels,
            data_spark_columns=data_spark_columns,
            data_dtypes=data_dtypes,
            column_label_names=column_label_names_final,
        )

    def with_filter(self, pred: Union[spark.Column, "Series"]) -> "InternalFrame":
        """ Copy the immutable InternalFrame with the updates by the predicate.

        :param pred: the predicate to filter.
        :return: the copied InternalFrame.
        """
        from databricks.koalas.series import Series

        if isinstance(pred, Series):
            assert isinstance(pred.spark.data_type, BooleanType), pred.spark.data_type
            pred_column: spark.Column = pred.spark.column
        else:
            spark_type: DataType = self.spark_frame.select(pred).schema[0].dataType
            assert isinstance(spark_type, BooleanType), spark_type
            pred_column = pred
        return self.with_new_sdf(
            self.spark_frame.filter(pred_column).select(self.spark_columns)
        )

    def with_new_spark_column(
        self,
        column_label: Tuple[Any, ...],
        scol: spark.Column,
        *,
        dtype: Optional[Dtype] = None,
        keep_order: bool = True,
    ) -> "InternalFrame":
        """
        Copy the immutable InternalFrame with the updates by the specified Spark Column.

        :param column_label: the column label to be updated.
        :param scol: the new Spark Column
        :param dtype: the new dtype.
            If not specified, the dtypes will be inferred from the spark Column.
        :return: the copied InternalFrame.
        """
        assert column_label in self.column_labels, column_label
        idx: int = self.column_labels.index(column_label)
        data_spark_columns: List[spark.Column] = self.data_spark_columns.copy()
        data_spark_columns[idx] = scol
        data_dtypes: List[Dtype] = self.data_dtypes.copy()
        data_dtypes[idx] = dtype
        return self.with_new_columns(
            data_spark_columns, data_dtypes=data_dtypes, keep_order=keep_order
        )

    def select_column(self, column_label: Tuple[Any, ...]) -> "InternalFrame":
        """
        Copy the immutable InternalFrame with the specified column.

        :param column_label: the column label to use.
        :return: the copied InternalFrame.
        """
        assert column_label in self.column_labels, column_label
        return self.copy(
            column_labels=[column_label],
            data_spark_columns=[self.spark_column_for(column_label)],
            data_dtypes=[self.dtype_for(column_label)],
            column_label_names=None,
        )

    def copy(
        self,
        *,
        spark_frame: Union[spark.DataFrame, _NoValueType] = _NoValue,
        index_spark_columns: Union[List[spark.Column], _NoValueType] = _NoValue,
        index_names: Union[List[Optional[Tuple[Any, ...]]], _NoValueType] = _NoValue,
        index_dtypes: Union[List[Dtype], _NoValueType] = _NoValue,
        column_labels: Union[List[Tuple[Any, ...]], _NoValueType] = _NoValue,
        data_spark_columns: Union[List[spark.Column], _NoValueType] = _NoValue,
        data_dtypes: Union[List[Dtype], _NoValueType] = _NoValue,
        column_label_names: Union[List[Optional[Tuple[Any, ...]]], _NoValueType] = _NoValue,
    ) -> "InternalFrame":
        """ Copy the immutable InternalFrame.

        :param spark_frame: the new Spark DataFrame. If not specified, the original one is used.
        :param index_spark_columns: the list of Spark Column.
                                    If not specified, the original ones are used.
        :param index_names: the index names. If not specified, the original ones are used.
        :param index_dtypes: the index dtypes. If not specified, the original dtyeps are used.
        :param column_labels: the new column labels. If not specified, the original ones are used.
        :param data_spark_columns: the new Spark Columns.
                                   If not specified, the original ones are used.
        :param data_dtypes: the data dtypes. If not specified, the original dtyeps are used.
        :param column_label_names: the new names of the column index levels.
                                   If not specified, the original ones are used.
        :return: the copied immutable InternalFrame.
        """
        if spark_frame is _NoValue:
            spark_frame_final: spark.DataFrame = self.spark_frame
        else:
            spark_frame_final = spark_frame
        if index_spark_columns is _NoValue:
            index_spark_columns_final: List[spark.Column] = self.index_spark_columns
        else:
            index_spark_columns_final = index_spark_columns
        if index_names is _NoValue:
            index_names_final: List[Optional[Tuple[Any, ...]]] = self.index_names
        else:
            index_names_final = index_names
        if index_dtypes is _NoValue:
            index_dtypes_final: List[Dtype] = self.index_dtypes
        else:
            index_dtypes_final = index_dtypes
        if column_labels is _NoValue:
            column_labels_final: List[Tuple[Any, ...]] = self.column_labels
        else:
            column_labels_final = column_labels
        if data_spark_columns is _NoValue:
            data_spark_columns_final: List[spark.Column] = self.data_spark_columns
        else:
            data_spark_columns_final = data_spark_columns
        if data_dtypes is _NoValue:
            data_dtypes_final: List[Dtype] = self.data_dtypes
        else:
            data_dtypes_final = data_dtypes
        if column_label_names is _NoValue:
            column_label_names_final: List[Optional[Tuple[Any, ...]]] = self.column_label_names
        else:
            column_label_names_final = column_label_names
        return InternalFrame(
            spark_frame=spark_frame_final,
            index_spark_columns=index_spark_columns_final,
            index_names=index_names_final,
            index_dtypes=index_dtypes_final,
            column_labels=column_labels_final,
            data_spark_columns=data_spark_columns_final,
            data_dtypes=data_dtypes_final,
            column_label_names=column_label_names_final,
        )

    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> "InternalFrame":
        """ Create an immutable DataFrame from pandas DataFrame.

        :param pdf: :class:`pd.DataFrame`
        :return: the created immutable DataFrame
        """
        index_names: List[Optional[Tuple[Any, ...]]] = [
            name if name is None or isinstance(name, tuple) else (name,)
            for name in pdf.index.names
        ]
        columns = pdf.columns
        if isinstance(columns, pd.MultiIndex):
            column_labels: List[Tuple[Any, ...]] = columns.tolist()
        else:
            column_labels = [(col,) for col in columns]
        column_label_names: List[Optional[Tuple[Any, ...]]] = [
            name if name is None or isinstance(name, tuple) else (name,)
            for name in columns.names
        ]
        prepared_tuple = InternalFrame.prepare_pandas_frame(pdf)
        prepared: pd.DataFrame = prepared_tuple[0]
        index_columns: List[str] = prepared_tuple[1]
        index_dtypes: List[Dtype] = prepared_tuple[2]
        data_columns: List[str] = prepared_tuple[3]
        data_dtypes: List[Dtype] = prepared_tuple[4]
        schema: StructType = StructType(
            [
                StructField(
                    name, infer_pd_series_spark_type(col, dtype), nullable=bool(col.isnull().any())
                )
                for (name, col), dtype in zip(prepared.iteritems(), index_dtypes + data_dtypes)
            ]
        )
        sdf: spark.DataFrame = default_session().createDataFrame(pdf, schema=schema)
        return InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in index_columns],
            index_names=index_names,
            index_dtypes=index_dtypes,
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            data_dtypes=data_dtypes,
            column_label_names=column_label_names,
        )

    @staticmethod
    def prepare_pandas_frame(
        pdf: pd.DataFrame, *, retain_index: bool = True
    ) -> Tuple[pd.DataFrame, List[str], List[Dtype], List[str], List[Dtype]]:
        """
        Prepare pandas DataFrame for creating Spark DataFrame.

        :param pdf: the pandas DataFrame to be prepared.
        :param retain_index: whether the indices should be retained.
        :return: the tuple of
            - the prepared pandas dataFrame
            - index column names for Spark DataFrame
            - index dtypes of the given pandas DataFrame
            - data column names for Spark DataFrame
            - data dtypes of the given pandas DataFrame

        [Docstring omitted for brevity]
        """
        pdf = pdf.copy()
        data_columns: List[str] = [name_like_string(col) for col in pdf.columns]
        pdf.columns = data_columns
        if retain_index:
            index_nlevels: int = pdf.index.nlevels
            index_columns: List[str] = [SPARK_INDEX_NAME_FORMAT(i) for i in range(index_nlevels)]
            pdf.index.names = index_columns
            reset_index: pd.DataFrame = pdf.reset_index()
        else:
            index_nlevels = 0
            index_columns = []
            reset_index = pdf
        index_dtypes: List[Dtype] = list(reset_index.dtypes)[:index_nlevels]
        data_dtypes: List[Dtype] = list(reset_index.dtypes)[index_nlevels:]
        for name, col in reset_index.iteritems():
            dt: Any = col.dtype
            if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                continue
            elif isinstance(dt, CategoricalDtype):
                col = col.cat.codes
            reset_index[name] = col.replace({np.nan: None})
        return reset_index, index_columns, index_dtypes, data_columns, data_dtypes
