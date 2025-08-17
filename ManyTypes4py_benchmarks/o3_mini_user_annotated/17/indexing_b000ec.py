from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from functools import reduce
from typing import Any, Optional, List, Tuple, TYPE_CHECKING, Union, cast, Sized

import pandas as pd
from pandas.api.types import is_list_like
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, LongType
from pyspark.sql.utils import AnalysisException
import numpy as np

from databricks import koalas as ks  # noqa: F401
from databricks.koalas.internal import (
    InternalFrame,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_DEFAULT_SERIES_NAME,
)
from databricks.koalas.exceptions import SparkPandasIndexingError, SparkPandasNotImplementedError
from databricks.koalas.typedef.typehints import (
    Dtype,
    Scalar,
    extension_dtypes,
    spark_type_to_pandas_dtype,
)
from databricks.koalas.utils import (
    is_name_like_tuple,
    is_name_like_value,
    lazy_property,
    name_like_string,
    same_anchor,
    scol_for,
    verify_temp_column_name,
)

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.series import Series

class IndexerLike(object):
    def __init__(self, kdf_or_kser: Union["DataFrame", "Series"]) -> None:
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series

        assert isinstance(kdf_or_kser, (DataFrame, Series)), "unexpected argument type: {}".format(
            type(kdf_or_kser)
        )
        self._kdf_or_kser = kdf_or_kser

    @property
    def _is_df(self) -> bool:
        from databricks.koalas.frame import DataFrame
        return isinstance(self._kdf_or_kser, DataFrame)

    @property
    def _is_series(self) -> bool:
        from databricks.koalas.series import Series
        return isinstance(self._kdf_or_kser, Series)

    @property
    def _kdf(self) -> "DataFrame":
        if self._is_df:
            return self._kdf_or_kser  # type: ignore
        else:
            assert self._is_series
            return self._kdf_or_kser._kdf  # type: ignore

    @property
    def _internal(self) -> InternalFrame:
        return self._kdf._internal  # type: ignore

class AtIndexer(IndexerLike):
    def __getitem__(self, key: Any) -> Union["Series", "DataFrame", Scalar]:
        if self._is_df:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError("Use DataFrame.at like .at[row_index, column_name]")
            row_sel, col_sel = key
        else:
            assert self._is_series, type(self._kdf_or_kser)
            if isinstance(key, tuple) and len(key) != 1:
                raise TypeError("Use Series.at like .at[row_index]")
            row_sel = key
            col_sel = self._kdf_or_kser._column_label  # type: ignore

        if self._internal.index_level == 1:
            if not is_name_like_value(row_sel, allow_none=False, allow_tuple=False):
                raise ValueError("At based indexing on a single index can only have a single value")
            row_sel = (row_sel,)
        else:
            if not is_name_like_tuple(row_sel, allow_none=False):
                raise ValueError("At based indexing on multi-index can only have tuple values")

        if col_sel is not None:
            if not is_name_like_value(col_sel, allow_none=False):
                raise ValueError("At based indexing on multi-index can only have tuple values")
            if not is_name_like_tuple(col_sel):
                col_sel = (col_sel,)
        cond = reduce(
            lambda x, y: x & y,
            [scol == row for scol, row in zip(self._internal.index_spark_columns, row_sel)],
        )
        pdf: pd.DataFrame = (
            self._internal.spark_frame.drop(NATURAL_ORDER_COLUMN_NAME)
            .filter(cond)
            .select(self._internal.spark_column_for(col_sel))
            .toPandas()
        )

        if len(pdf) < 1:
            raise KeyError(name_like_string(row_sel))
        values = pdf.iloc[:, 0].values
        return (
            values if (len(row_sel) < self._internal.index_level or len(values) > 1) else values[0]
        )

class iAtIndexer(IndexerLike):
    def __getitem__(self, key: Any) -> Union["Series", "DataFrame", Scalar]:
        if self._is_df:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(
                    "Use DataFrame.iat like .iat[row_integer_position, column_integer_position]"
                )
            row_sel, col_sel = key
            if not isinstance(row_sel, int) or not isinstance(col_sel, int):
                raise ValueError("iAt based indexing can only have integer indexers")
            return self._kdf_or_kser.iloc[row_sel, col_sel]  # type: ignore
        else:
            assert self._is_series, type(self._kdf_or_kser)
            if not isinstance(key, int) and len(key) != 1:
                raise TypeError("Use Series.iat like .iat[row_integer_position]")
            if not isinstance(key, int):
                raise ValueError("iAt based indexing can only have integer indexers")
            return self._kdf_or_kser.iloc[key]  # type: ignore

class LocIndexerLike(IndexerLike, metaclass=ABCMeta):
    def _select_rows(self, rows_sel: Any) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        from databricks.koalas.series import Series
        if rows_sel is None:
            return None, None, None
        elif isinstance(rows_sel, Series):
            return self._select_rows_by_series(rows_sel)
        elif isinstance(rows_sel, spark.Column):
            return self._select_rows_by_spark_column(rows_sel)
        elif isinstance(rows_sel, slice):
            if rows_sel == slice(None):
                return None, None, None
            return self._select_rows_by_slice(rows_sel)
        elif isinstance(rows_sel, tuple):
            return self._select_rows_else(rows_sel)
        elif is_list_like(rows_sel):
            return self._select_rows_by_iterable(rows_sel)
        else:
            return self._select_rows_else(rows_sel)

    def _select_cols(
        self, cols_sel: Any, missing_keys: Optional[List[Tuple]] = None
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        from databricks.koalas.series import Series
        if cols_sel is None:
            column_labels: List[Tuple] = self._internal.column_labels
            data_spark_columns: List[spark.Column] = self._internal.data_spark_columns
            data_dtypes: List[Dtype] = self._internal.data_dtypes
            return column_labels, data_spark_columns, data_dtypes, False, None
        elif isinstance(cols_sel, Series):
            return self._select_cols_by_series(cols_sel, missing_keys)
        elif isinstance(cols_sel, spark.Column):
            return self._select_cols_by_spark_column(cols_sel, missing_keys)
        elif isinstance(cols_sel, slice):
            if cols_sel == slice(None):
                column_labels = self._internal.column_labels
                data_spark_columns = self._internal.data_spark_columns
                data_dtypes = self._internal.data_dtypes
                return column_labels, data_spark_columns, data_dtypes, False, None
            return self._select_cols_by_slice(cols_sel, missing_keys)
        elif isinstance(cols_sel, tuple):
            return self._select_cols_else(cols_sel, missing_keys)
        elif is_list_like(cols_sel):
            return self._select_cols_by_iterable(cols_sel, missing_keys)
        else:
            return self._select_cols_else(cols_sel, missing_keys)

    @abstractmethod
    def _select_rows_by_series(
        self, rows_sel: "Series"
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        pass

    @abstractmethod
    def _select_rows_by_spark_column(
        self, rows_sel: spark.Column
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        pass

    @abstractmethod
    def _select_rows_by_slice(
        self, rows_sel: slice
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        pass

    @abstractmethod
    def _select_rows_by_iterable(
        self, rows_sel: Iterable
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        pass

    @abstractmethod
    def _select_rows_else(
        self, rows_sel: Any
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        pass

    @abstractmethod
    def _select_cols_by_series(
        self, cols_sel: "Series", missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        pass

    @abstractmethod
    def _select_cols_by_spark_column(
        self, cols_sel: spark.Column, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        pass

    @abstractmethod
    def _select_cols_by_slice(
        self, cols_sel: slice, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        pass

    @abstractmethod
    def _select_cols_by_iterable(
        self, cols_sel: Iterable, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        pass

    @abstractmethod
    def _select_cols_else(
        self, cols_sel: Any, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        pass

    def __getitem__(self, key: Any) -> Union["Series", "DataFrame"]:
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, first_series

        if self._is_series:
            if isinstance(key, Series) and not same_anchor(key, self._kdf_or_kser):
                kdf = self._kdf_or_kser.to_frame()  # type: ignore
                temp_col = verify_temp_column_name(kdf, "__temp_col__")
                kdf[temp_col] = key
                return type(self)(kdf[self._kdf_or_kser.name])[kdf[temp_col]]
            cond, limit, remaining_index = self._select_rows(key)
            if cond is None and limit is None:
                return self._kdf_or_kser
            column_label = self._kdf_or_kser._column_label  # type: ignore
            column_labels: List[Tuple] = [column_label]
            data_spark_columns: List[spark.Column] = [self._internal.spark_column_for(column_label)]
            data_dtypes: List[Dtype] = [self._internal.dtype_for(column_label)]
            returns_series: bool = True
            series_name: Optional[Tuple] = self._kdf_or_kser.name  # type: ignore
        else:
            assert self._is_df
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise SparkPandasIndexingError("Only accepts pairs of candidates")
                rows_sel, cols_sel = key
            else:
                rows_sel = key
                cols_sel = None
            if isinstance(rows_sel, Series) and not same_anchor(rows_sel, self._kdf_or_kser):
                kdf = self._kdf_or_kser.copy()  # type: ignore
                temp_col = verify_temp_column_name(kdf, "__temp_col__")
                kdf[temp_col] = rows_sel
                return type(self)(kdf)[kdf[temp_col], cols_sel][list(self._kdf_or_kser.columns)]  # type: ignore
            cond, limit, remaining_index = self._select_rows(rows_sel)
            (
                column_labels,
                data_spark_columns,
                data_dtypes,
                returns_series,
                series_name,
            ) = self._select_cols(cols_sel)
            if cond is None and limit is None and returns_series:
                kser = self._kdf_or_kser._kser_for(column_labels[0])  # type: ignore
                if series_name is not None and series_name != kser.name:
                    kser = kser.rename(series_name)
                return kser

        if remaining_index is not None:
            index_spark_columns: List[spark.Column] = self._internal.index_spark_columns[-remaining_index:]
            index_names = self._internal.index_names[-remaining_index:]
            index_dtypes = self._internal.index_dtypes[-remaining_index:]
        else:
            index_spark_columns = self._internal.index_spark_columns
            index_names = self._internal.index_names
            index_dtypes = self._internal.index_dtypes

        if len(column_labels) > 0:
            column_labels = column_labels.copy()
            column_labels_level: int = max(len(label) if label is not None else 1 for label in column_labels)
            none_column: int = 0
            for i, label in enumerate(column_labels):
                if label is None:
                    label = (none_column,)
                    none_column += 1
                if len(label) < column_labels_level:
                    label = tuple(list(label) + ([""] * (column_labels_level - len(label))))
                column_labels[i] = label
            if i == 0 and none_column == 1:
                column_labels = [None]
            column_label_names = self._internal.column_label_names[-column_labels_level:]
        else:
            column_label_names = self._internal.column_label_names

        try:
            sdf = self._internal.spark_frame
            if cond is not None:
                index_columns = sdf.select(index_spark_columns).columns
                data_columns = sdf.select(data_spark_columns).columns
                sdf = sdf.filter(cond).select(index_spark_columns + data_spark_columns)
                index_spark_columns = [scol_for(sdf, col) for col in index_columns]
                data_spark_columns = [scol_for(sdf, col) for col in data_columns]
            if limit is not None:
                if limit >= 0:
                    sdf = sdf.limit(limit)
                else:
                    sdf = sdf.limit(sdf.count() + limit)
                sdf = sdf.drop(NATURAL_ORDER_COLUMN_NAME)
        except AnalysisException:
            raise KeyError("[{}] don't exist in columns".format(
                [col._jc.toString() for col in data_spark_columns]
            ))
        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=index_spark_columns,
            index_names=index_names,
            index_dtypes=index_dtypes,
            column_labels=column_labels,
            data_spark_columns=data_spark_columns,
            data_dtypes=data_dtypes,
            column_label_names=column_label_names,
        )
        kdf = DataFrame(internal)
        if returns_series:
            kdf_or_kser = first_series(kdf)
            if series_name is not None and series_name != kdf_or_kser.name:
                kdf_or_kser = kdf_or_kser.rename(series_name)
        else:
            kdf_or_kser = kdf
        if remaining_index is not None and remaining_index == 0:
            pdf_or_pser = kdf_or_kser.head(2).to_pandas()
            length = len(pdf_or_pser)
            if length == 0:
                raise KeyError(name_like_string(key))
            elif length == 1:
                return pdf_or_pser.iloc[0]
            else:
                return kdf_or_kser
        else:
            return kdf_or_kser

    def __setitem__(self, key: Any, value: Any) -> None:
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, first_series

        if self._is_series:
            if (
                isinstance(key, Series)
                and (isinstance(self, iLocIndexer) or not same_anchor(key, self._kdf_or_kser))
            ) or (
                isinstance(value, Series)
                and (isinstance(self, iLocIndexer) or not same_anchor(value, self._kdf_or_kser))
            ):
                if self._kdf_or_kser.name is None:
                    kdf = self._kdf_or_kser.to_frame()  # type: ignore
                    column_label = kdf._internal.column_labels[0]
                else:
                    kdf = self._kdf_or_kser._kdf.copy()  # type: ignore
                    column_label = self._kdf_or_kser._column_label  # type: ignore
                temp_natural_order = verify_temp_column_name(kdf, "__temp_natural_order__")
                temp_key_col = verify_temp_column_name(kdf, "__temp_key_col__")
                temp_value_col = verify_temp_column_name(kdf, "__temp_value_col__")
                kdf[temp_natural_order] = F.monotonically_increasing_id()
                if isinstance(key, Series):
                    kdf[temp_key_col] = key
                if isinstance(value, Series):
                    kdf[temp_value_col] = value
                kdf = kdf.sort_values(temp_natural_order).drop(temp_natural_order)
                kser = kdf._kser_for(column_label)
                if isinstance(key, Series):
                    key = F.col("`{}`".format(kdf[temp_key_col]._internal.data_spark_column_names[0]))
                if isinstance(value, Series):
                    value = F.col("`{}`".format(kdf[temp_value_col]._internal.data_spark_column_names[0]))
                type(self)(kser)[key] = value
                if self._kdf_or_kser.name is None:
                    kser = kser.rename()
                self._kdf_or_kser._kdf._update_internal_frame(
                    kser._kdf[self._kdf_or_kser._kdf._internal.column_labels]._internal.resolved_copy,  # type: ignore
                    requires_same_anchor=False,
                )
                return
            if isinstance(value, DataFrame):
                raise ValueError("Incompatible indexer with DataFrame")
            cond, limit, remaining_index = self._select_rows(key)
            if cond is None:
                cond = F.lit(True)
            if limit is not None:
                cond = cond & (self._internal.spark_frame[self._sequence_col] < F.lit(limit))
            if isinstance(value, (Series, spark.Column)):
                if remaining_index is not None and remaining_index == 0:
                    raise ValueError(
                        "No axis named {} for object type {}".format(key, type(value).__name__)
                    )
                if isinstance(value, Series):
                    value = value.spark.column
            else:
                value = F.lit(value)
            scol = (
                F.when(cond, value)
                .otherwise(self._internal.spark_column_for(self._kdf_or_kser._column_label))
                .alias(name_like_string(self._kdf_or_kser.name or SPARK_DEFAULT_SERIES_NAME))
            )
            internal = self._internal.with_new_spark_column(
                self._kdf_or_kser._column_label, scol  # type: ignore
            )
            self._kdf_or_kser._kdf._update_internal_frame(internal, requires_same_anchor=False)
        else:
            assert self._is_df
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise SparkPandasIndexingError("Only accepts pairs of candidates")
                rows_sel, cols_sel = key
            else:
                rows_sel = key
                cols_sel = None
            if isinstance(value, DataFrame):
                if len(value.columns) == 1:
                    value = first_series(value)
                else:
                    raise ValueError("Only a dataframe with one column can be assigned")
            if (
                isinstance(rows_sel, Series)
                and (isinstance(self, iLocIndexer) or not same_anchor(rows_sel, self._kdf_or_kser))
            ) or (
                isinstance(value, Series)
                and (isinstance(self, iLocIndexer) or not same_anchor(value, self._kdf_or_kser))
            ):
                kdf = self._kdf_or_kser.copy()  # type: ignore
                temp_natural_order = verify_temp_column_name(kdf, "__temp_natural_order__")
                temp_key_col = verify_temp_column_name(kdf, "__temp_key_col__")
                temp_value_col = verify_temp_column_name(kdf, "__temp_value_col__")
                kdf[temp_natural_order] = F.monotonically_increasing_id()
                if isinstance(rows_sel, Series):
                    kdf[temp_key_col] = rows_sel
                if isinstance(value, Series):
                    kdf[temp_value_col] = value
                kdf = kdf.sort_values(temp_natural_order).drop(temp_natural_order)
                if isinstance(rows_sel, Series):
                    rows_sel = F.col("`{}`".format(kdf[temp_key_col]._internal.data_spark_column_names[0]))
                if isinstance(value, Series):
                    value = F.col("`{}`".format(kdf[temp_value_col]._internal.data_spark_column_names[0]))
                type(self)(kdf)[rows_sel, cols_sel] = value
                self._kdf_or_kser._update_internal_frame(
                    kdf[list(self._kdf_or_kser.columns)]._internal.resolved_copy,  # type: ignore
                    requires_same_anchor=False,
                )
                return
            cond, limit, remaining_index = self._select_rows(rows_sel)
            missing_keys: List[Tuple] = []
            _, data_spark_columns, _, _, _ = self._select_cols(cols_sel, missing_keys=missing_keys)
            if cond is None:
                cond = F.lit(True)
            if limit is not None:
                cond = cond & (self._internal.spark_frame[self._sequence_col] < F.lit(limit))
            if isinstance(value, (Series, spark.Column)):
                if remaining_index is not None and remaining_index == 0:
                    raise ValueError("Incompatible indexer with Series")
                if len(data_spark_columns) > 1:
                    raise ValueError("shape mismatch")
                if isinstance(value, Series):
                    value = value.spark.column
            else:
                value = F.lit(value)
            new_data_spark_columns: List[spark.Column] = []
            new_dtypes: List[Optional[Dtype]] = []
            for new_scol, spark_column_name, new_dtype in zip(
                self._internal.data_spark_columns,
                self._internal.data_spark_column_names,
                self._internal.data_dtypes,
            ):
                for scol in data_spark_columns:
                    if new_scol._jc.equals(scol._jc):
                        new_scol = F.when(cond, value).otherwise(scol).alias(spark_column_name)
                        new_dtype = spark_type_to_pandas_dtype(
                            self._internal.spark_frame.select(new_scol).schema[0].dataType,
                            use_extension_dtypes=isinstance(new_dtype, extension_dtypes),
                        )
                        break
                new_data_spark_columns.append(new_scol)
                new_dtypes.append(new_dtype)
            column_labels = self._internal.column_labels.copy()
            for label in missing_keys:
                if not is_name_like_tuple(label):
                    label = (label,)
                if len(label) < self._internal.column_labels_level:
                    label = tuple(list(label) + ([""] * (self._internal.column_labels_level - len(label))))
                elif len(label) > self._internal.column_labels_level:
                    raise KeyError(
                        "Key length ({}) exceeds index depth ({})".format(
                            len(label), self._internal.column_labels_level
                        )
                    )
                column_labels.append(label)
                new_data_spark_columns.append(F.when(cond, value).alias(name_like_string(label)))
                new_dtypes.append(None)
            internal = self._internal.with_new_columns(
                new_data_spark_columns, column_labels=column_labels, data_dtypes=new_dtypes
            )
            self._kdf_or_kser._update_internal_frame(internal, requires_same_anchor=False)

class LocIndexer(LocIndexerLike):
    @staticmethod
    def _NotImplemented(description: str) -> SparkPandasNotImplementedError:
        return SparkPandasNotImplementedError(
            description=description,
            pandas_function=".loc[..., ...]",
            spark_target_function="select, where",
        )

    def _select_rows_by_series(
        self, rows_sel: "Series"
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        assert isinstance(rows_sel.spark.data_type, BooleanType), rows_sel.spark.data_type
        return rows_sel.spark.column, None, None

    def _select_rows_by_spark_column(
        self, rows_sel: spark.Column
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        spark_type = self._internal.spark_frame.select(rows_sel).schema[0].dataType
        assert isinstance(spark_type, BooleanType), spark_type
        return rows_sel, None, None

    def _select_rows_by_slice(
        self, rows_sel: slice
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        from databricks.koalas.indexes import MultiIndex
        if rows_sel.step is not None:
            raise LocIndexer._NotImplemented("Cannot use step with Spark.")
        elif self._internal.index_level == 1:
            sdf = self._internal.spark_frame
            index = self._kdf_or_kser.index  # type: ignore
            index_column = index.to_series()
            index_data_type = index_column.spark.data_type
            start = rows_sel.start
            stop = rows_sel.stop
            start_and_stop = (
                sdf.select(index_column.spark.column, NATURAL_ORDER_COLUMN_NAME)
                .where(
                    (index_column.spark.column == F.lit(start).cast(index_data_type))
                    | (index_column.spark.column == F.lit(stop).cast(index_data_type))
                )
                .collect()
            )
            start_vals = [row[1] for row in start_and_stop if row[0] == start]
            start = start_vals[0] if len(start_vals) > 0 else None
            stop_vals = [row[1] for row in start_and_stop if row[0] == stop]
            stop = stop_vals[-1] if len(stop_vals) > 0 else None
            cond = []
            if start is not None:
                cond.append(F.col(NATURAL_ORDER_COLUMN_NAME) >= F.lit(start).cast(LongType()))
            if stop is not None:
                cond.append(F.col(NATURAL_ORDER_COLUMN_NAME) <= F.lit(stop).cast(LongType()))
            if (start is None and rows_sel.start is not None) or (stop is None and rows_sel.stop is not None):
                inc = index_column.is_monotonic_increasing
                if inc is False:
                    dec = index_column.is_monotonic_decreasing
                if start is None and rows_sel.start is not None:
                    start = rows_sel.start
                    if inc is not False:
                        cond.append(index_column.spark.column >= F.lit(start).cast(index_data_type))
                    elif dec is not False:
                        cond.append(index_column.spark.column <= F.lit(start).cast(index_data_type))
                    else:
                        raise KeyError(rows_sel.start)
                if stop is None and rows_sel.stop is not None:
                    stop = rows_sel.stop
                    if inc is not False:
                        cond.append(index_column.spark.column <= F.lit(stop).cast(index_data_type))
                    elif dec is not False:
                        cond.append(index_column.spark.column >= F.lit(stop).cast(index_data_type))
                    else:
                        raise KeyError(rows_sel.stop)
            return reduce(lambda x, y: x & y, cond), None, None
        else:
            index = self._kdf_or_kser.index  # type: ignore
            index_data_type = [f.dataType for f in index.to_series().spark.data_type]
            start = rows_sel.start
            if start is not None:
                if not isinstance(start, tuple):
                    start = (start,)
                if len(start) == 0:
                    start = None
            stop = rows_sel.stop
            if stop is not None:
                if not isinstance(stop, tuple):
                    stop = (stop,)
                if len(stop) == 0:
                    stop = None
            depth = max(len(start) if start is not None else 0, len(stop) if stop is not None else 0)
            if depth == 0:
                return None, None, None
            elif (
                depth > self._internal.index_level
                or not index.droplevel(list(range(self._internal.index_level)[depth:])).is_monotonic
            ):
                raise KeyError("Key length ({}) was greater than MultiIndex sort depth".format(depth))
            conds: List[spark.Column] = []
            if start is not None:
                cond = F.lit(True)
                for scol, value, dt in list(zip(self._internal.index_spark_columns, start, index_data_type))[::-1]:
                    compare = MultiIndex._comparator_for_monotonic_increasing(dt)
                    cond = F.when(scol.eqNullSafe(F.lit(value).cast(dt)), cond).otherwise(
                        compare(scol, F.lit(value).cast(dt), spark.Column.__gt__)
                    )
                conds.append(cond)
            if stop is not None:
                cond = F.lit(True)
                for scol, value, dt in list(zip(self._internal.index_spark_columns, stop, index_data_type))[::-1]:
                    compare = MultiIndex._comparator_for_monotonic_increasing(dt)
                    cond = F.when(scol.eqNullSafe(F.lit(value).cast(dt)), cond).otherwise(
                        compare(scol, F.lit(value).cast(dt), spark.Column.__lt__)
                    )
                conds.append(cond)
            return reduce(lambda x, y: x & y, conds), None, None

    def _select_rows_by_iterable(
        self, rows_sel: Iterable
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        rows_sel = list(rows_sel)
        if len(rows_sel) == 0:
            return F.lit(False), None, None
        elif self._internal.index_level == 1:
            index_column = self._kdf_or_kser.index.to_series()  # type: ignore
            index_data_type = index_column.spark.data_type
            if len(rows_sel) == 1:
                return (index_column.spark.column == F.lit(rows_sel[0]).cast(index_data_type),
                        None,
                        None)
            else:
                return (index_column.spark.column.isin([F.lit(r).cast(index_data_type) for r in rows_sel]),
                        None,
                        None)
        else:
            raise LocIndexer._NotImplemented("Cannot select with MultiIndex with Spark.")

    def _select_rows_else(
        self, rows_sel: Any
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        if not isinstance(rows_sel, tuple):
            rows_sel = (rows_sel,)
        if len(rows_sel) > self._internal.index_level:
            raise SparkPandasIndexingError("Too many indexers")
        rows = [scol == value for scol, value in zip(self._internal.index_spark_columns, rows_sel)]
        return reduce(lambda x, y: x & y, rows), None, self._internal.index_level - len(rows_sel)

    def _get_from_multiindex_column(
        self, key: Any, missing_keys: Optional[List[Tuple]], labels: Optional[List[Tuple]] = None, recursed: int = 0
    ) -> Tuple[List[Tuple], Optional[List[spark.Column]], Any, bool, Optional[Tuple]]:
        assert isinstance(key, tuple)
        if labels is None:
            labels = [(label, label) for label in self._internal.column_labels]
        for k in key:
            labels = [
                (label, None if lbl is None else lbl[1:])
                for label, lbl in labels
                if (lbl is None and k is None) or (lbl is not None and lbl[0] == k)
            ]
            if len(labels) == 0:
                if missing_keys is None:
                    raise KeyError(k)
                else:
                    missing_keys.append(key)
                    return [], [], [], False, None
        if all(lbl is not None and len(lbl) > 0 and lbl[0] == "" for _, lbl in labels):
            labels = [(label, tuple([str(key), *lbl[1:]])) for i, (label, lbl) in enumerate(labels)]
            return self._get_from_multiindex_column((str(key),), missing_keys, labels, recursed + 1)
        else:
            returns_series: bool = all(lbl is None or len(lbl) == 0 for _, lbl in labels)
            if returns_series:
                labels_set = set(label for label, _ in labels)
                assert len(labels_set) == 1
                label = list(labels_set)[0]
                column_labels = [label]
                data_spark_columns = [self._internal.spark_column_for(label)]
                data_dtypes = [self._internal.dtype_for(label)]
                if label is None:
                    series_name = None
                else:
                    if recursed > 0:
                        label = label[:-recursed]
                    series_name = label if len(label) > 1 else label[0]
            else:
                column_labels = [None if lbl is None or lbl == (None,) else lbl for _, lbl in labels]
                data_spark_columns = [self._internal.spark_column_for(label) for label, _ in labels]
                data_dtypes = [self._internal.dtype_for(label) for label, _ in labels]
                series_name = None
            return column_labels, data_spark_columns, data_dtypes, returns_series, series_name

    def _select_cols_by_series(
        self, cols_sel: "Series", missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        column_labels: List[Tuple] = [cols_sel._column_label]
        data_spark_columns: List[spark.Column] = [cols_sel.spark.column]
        data_dtypes: List[Dtype] = [cols_sel.dtype]
        return column_labels, data_spark_columns, data_dtypes, True, None

    def _select_cols_by_spark_column(
        self, cols_sel: spark.Column, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        column_labels = [(self._internal.spark_frame.select(cols_sel).columns[0],)]
        data_spark_columns = [cols_sel]
        return column_labels, data_spark_columns, None, True, None

    def _select_cols_by_slice(
        self, cols_sel: slice, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        start, stop = self._kdf_or_kser.columns.slice_locs(start=cols_sel.start, end=cols_sel.stop)  # type: ignore
        column_labels = self._internal.column_labels[start:stop]
        data_spark_columns = self._internal.data_spark_columns[start:stop]
        data_dtypes = self._internal.data_dtypes[start:stop]
        return column_labels, data_spark_columns, data_dtypes, False, None

    def _select_cols_by_iterable(
        self, cols_sel: Iterable, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        from databricks.koalas.series import Series
        if all(isinstance(key, Series) for key in cols_sel):
            column_labels = [key._column_label for key in cols_sel]
            data_spark_columns = [key.spark.column for key in cols_sel]
            data_dtypes = [key.dtype for key in cols_sel]
        elif all(isinstance(key, spark.Column) for key in cols_sel):
            column_labels = [(self._internal.spark_frame.select(col).columns[0],) for col in cols_sel]
            data_spark_columns = list(cols_sel)
            data_dtypes = None
        elif all(isinstance(key, bool) for key in cols_sel) or all(isinstance(key, np.bool_) for key in cols_sel):
            if len(cast(Sized, cols_sel)) != len(self._internal.column_labels):
                raise IndexError("Boolean index has wrong length: %s instead of %s" %
                                 (len(cast(Sized, cols_sel)), len(self._internal.column_labels)))
            if isinstance(cols_sel, pd.Series):
                if not cols_sel.index.sort_values().equals(self._kdf_or_kser.columns.sort_values()):  # type: ignore
                    raise SparkPandasIndexingError(
                        "Unalignable boolean Series provided as indexer "
                        "(index of the boolean Series and of the indexed object do not match)"
                    )
                else:
                    column_labels = [
                        column_label for column_label in self._internal.column_labels
                        if cols_sel[column_label if len(column_label) > 1 else column_label[0]]
                    ]
                    data_spark_columns = [
                        self._internal.spark_column_for(column_label) for column_label in column_labels
                    ]
                    data_dtypes = [
                        self._internal.dtype_for(column_label) for column_label in column_labels
                    ]
            else:
                column_labels = [
                    self._internal.column_labels[i] for i, col in enumerate(cols_sel) if col
                ]
                data_spark_columns = [
                    self._internal.data_spark_columns[i] for i, col in enumerate(cols_sel) if col
                ]
                data_dtypes = [
                    self._internal.data_dtypes[i] for i, col in enumerate(cols_sel) if col
                ]
        elif any(isinstance(key, tuple) for key in cols_sel) and any(not is_name_like_tuple(key) for key in cols_sel):
            raise TypeError("Expected tuple, got {}".format(type(set(key for key in cols_sel if not is_name_like_tuple(key)).pop())))
        else:
            if missing_keys is None and all(isinstance(key, tuple) for key in cols_sel):
                level = self._internal.column_labels_level
                if any(len(key) != level for key in cols_sel):
                    raise ValueError("All the key level should be the same as column index level.")
            column_labels = []
            data_spark_columns = []
            data_dtypes = []
            for key in cols_sel:
                found = False
                for label in self._internal.column_labels:
                    if label == key or label[0] == key:
                        column_labels.append(label)
                        data_spark_columns.append(self._internal.spark_column_for(label))
                        data_dtypes.append(self._internal.dtype_for(label))
                        found = True
                if not found:
                    if missing_keys is None:
                        raise KeyError("['{}'] not in index".format(name_like_string(key)))
                    else:
                        missing_keys.append(key)
        return column_labels, data_spark_columns, data_dtypes, False, None

    def _select_cols_else(
        self, cols_sel: Any, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple],
        Optional[List[spark.Column]],
        Optional[List[Dtype]],
        bool,
        Optional[Tuple],
    ]:
        if not is_name_like_tuple(cols_sel):
            cols_sel = (cols_sel,)
        return self._get_from_multiindex_column(cols_sel, missing_keys)

class iLocIndexer(LocIndexerLike):
    @staticmethod
    def _NotImplemented(description: str) -> SparkPandasNotImplementedError:
        return SparkPandasNotImplementedError(
            description=description,
            pandas_function=".iloc[..., ...]",
            spark_target_function="select, where",
        )

    @lazy_property
    def _internal(self) -> InternalFrame:
        internal = super()._internal.resolved_copy
        sdf = InternalFrame.attach_distributed_sequence_column(
            internal.spark_frame, column_name=self._sequence_col
        )
        return internal.with_new_sdf(spark_frame=sdf.orderBy(NATURAL_ORDER_COLUMN_NAME))

    @lazy_property
    def _sequence_col(self) -> str:
        internal = super()._internal.resolved_copy
        return verify_temp_column_name(internal.spark_frame, "__distributed_sequence_column__")

    def _select_rows_by_series(
        self, rows_sel: "Series"
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        raise iLocIndexer._NotImplemented(
            ".iloc requires numeric slice, conditional boolean Index or a sequence of positions as int, got {}".format(type(rows_sel))
        )

    def _select_rows_by_spark_column(
        self, rows_sel: spark.Column
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        raise iLocIndexer._NotImplemented(
            ".iloc requires numeric slice, conditional boolean Index or a sequence of positions as int, got {}".format(type(rows_sel))
        )

    def _select_rows_by_slice(
        self, rows_sel: slice
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        def verify_type(i: Any) -> None:
            if not isinstance(i, int):
                raise TypeError("cannot do slice indexing with these indexers [{}] of {}".format(i, type(i)))
        has_negative = False
        start = rows_sel.start
        if start is not None:
            verify_type(start)
            if start == 0:
                start = None
            elif start < 0:
                has_negative = True
        stop = rows_sel.stop
        if stop is not None:
            verify_type(stop)
            if stop < 0:
                has_negative = True
        step = rows_sel.step
        if step is not None:
            verify_type(step)
            if step == 0:
                raise ValueError("slice step cannot be zero")
        else:
            step = 1
        if start is None and step == 1:
            return None, stop, None
        sdf = self._internal.spark_frame
        sequence_scol = sdf[self._sequence_col]
        if has_negative or (step < 0 and start is None):
            cnt = sdf.count()
        conds: List[spark.Column] = []
        if start is not None:
            if start < 0:
                start = start + cnt
            if step >= 0:
                conds.append(sequence_scol >= F.lit(start).cast(LongType()))
            else:
                conds.append(sequence_scol <= F.lit(start).cast(LongType()))
        if stop is not None:
            if stop < 0:
                stop = stop + cnt
            if step >= 0:
                conds.append(sequence_scol < F.lit(stop).cast(LongType()))
            else:
                conds.append(sequence_scol > F.lit(stop).cast(LongType()))
        if step != 1:
            if step > 0:
                start_val = start or 0
            else:
                start_val = start or (cnt - 1)
            conds.append(((sequence_scol - start_val) % F.lit(step).cast(LongType())) == F.lit(0))
        return reduce(lambda x, y: x & y, conds), None, None

    def _select_rows_by_iterable(
        self, rows_sel: Iterable
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        sdf = self._internal.spark_frame
        if any(isinstance(key, (int, np.int, np.int64, np.int32)) and key < 0 for key in rows_sel):
            offset = sdf.count()
        else:
            offset = 0
        new_rows_sel: List[int] = []
        for key in list(rows_sel):
            if not isinstance(key, (int, np.int, np.int64, np.int32)):
                raise TypeError("cannot do positional indexing with these indexers [{}] of {}".format(key, type(key)))
            if key < 0:
                key = key + offset
            new_rows_sel.append(key)
        if len(new_rows_sel) != len(set(new_rows_sel)):
            raise NotImplementedError("Duplicated row selection is not currently supported; however, normalised index was [%s]" % new_rows_sel)
        sequence_scol = sdf[self._sequence_col]
        conds: List[spark.Column] = []
        for key in new_rows_sel:
            conds.append(sequence_scol == F.lit(int(key)).cast(LongType()))
        if len(conds) == 0:
            conds = [F.lit(False)]
        return reduce(lambda x, y: x | y, conds), None, None

    def _select_rows_else(
        self, rows_sel: Any
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
        if isinstance(rows_sel, int):
            sdf = self._internal.spark_frame
            return (sdf[self._sequence_col] == rows_sel), None, 0
        elif isinstance(rows_sel, tuple):
            raise SparkPandasIndexingError("Too many indexers")
        else:
            raise iLocIndexer._NotImplemented(
                ".iloc requires numeric slice, conditional boolean Index or a sequence of positions as int, got {}".format(type(rows_sel))
            )

    def _select_cols_by_series(
        self, cols_sel: "Series", missing_keys: Optional[List[Tuple]]
    ) -> Tuple[List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]]:
        raise ValueError("Location based indexing can only have [integer, integer slice, listlike of integers, boolean array] types, got {}".format(cols_sel))

    def _select_cols_by_spark_column(
        self, cols_sel: spark.Column, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]]:
        raise ValueError("Location based indexing can only have [integer, integer slice, listlike of integers, boolean array] types, got {}".format(cols_sel))

    def _select_cols_by_slice(
        self, cols_sel: slice, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]]:
        if all(s is None or isinstance(s, int) for s in (cols_sel.start, cols_sel.stop, cols_sel.step)):
            column_labels = self._internal.column_labels[cols_sel]
            data_spark_columns = self._internal.data_spark_columns[cols_sel]
            data_dtypes = self._internal.data_dtypes[cols_sel]
            return column_labels, data_spark_columns, data_dtypes, False, None
        else:
            not_none = cols_sel.start if cols_sel.start is not None else cols_sel.stop if cols_sel.stop is not None else cols_sel.step
            raise TypeError("cannot do slice indexing with these indexers {} of {}".format(not_none, type(not_none)))

    def _select_cols_by_iterable(
        self, cols_sel: Iterable, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]]:
        if all(isinstance(s, bool) for s in cols_sel):
            cols_sel = [i for i, s in enumerate(cols_sel) if s]
        if all(isinstance(s, int) for s in cols_sel):
            column_labels = [self._internal.column_labels[s] for s in cols_sel]
            data_spark_columns = [self._internal.data_spark_columns[s] for s in cols_sel]
            data_dtypes = [self._internal.data_dtypes[s] for s in cols_sel]
            return column_labels, data_spark_columns, data_dtypes, False, None
        else:
            raise TypeError("cannot perform reduce with flexible type")

    def _select_cols_else(
        self, cols_sel: Any, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]]:
        if isinstance(cols_sel, int):
            if cols_sel > len(self._internal.column_labels):
                raise KeyError(cols_sel)
            column_labels = [self._internal.column_labels[cols_sel]]
            data_spark_columns = [self._internal.data_spark_columns[cols_sel]]
            data_dtypes = [self._internal.data_dtypes[cols_sel]]
            return column_labels, data_spark_columns, data_dtypes, True, None
        else:
            raise ValueError("Location based indexing can only have [integer, integer slice, listlike of integers, boolean array] types, got {}".format(cols_sel))

    def __setitem__(self, key: Any, value: Any) -> None:
        if is_list_like(value) and not isinstance(value, spark.Column):
            iloc_item = self[key]
            if not is_list_like(key) or not is_list_like(iloc_item):
                raise ValueError("setting an array element with a sequence.")
            else:
                shape_iloc_item = iloc_item.shape
                len_iloc_item = shape_iloc_item[0]
                len_value = len(value)
                if len_iloc_item != len_value:
                    if self._is_series:
                        raise ValueError("cannot set using a list-like indexer with a different length than the value")
                    else:
                        raise ValueError("shape mismatch: value array of shape ({},) could not be broadcast to indexing result of shape {}".format(len_value, shape_iloc_item))
        super().__setitem__(key, value)
        self._kdf._update_internal_frame(self._kdf._internal.resolved_copy, requires_same_anchor=False)  # type: ignore
        del self._internal
        del self._sequence_col