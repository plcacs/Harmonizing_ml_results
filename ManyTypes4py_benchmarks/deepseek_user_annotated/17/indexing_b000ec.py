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
A loc indexer for Koalas DataFrame/Series.
"""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from functools import reduce
from typing import Any, Optional, List, Tuple, TYPE_CHECKING, Union, cast, Sized, Dict, Set, Sequence

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
        self._kdf_or_kser: Union["DataFrame", "Series"] = kdf_or_kser

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
            return cast("DataFrame", self._kdf_or_kser)
        else:
            assert self._is_series
            return cast("Series", self._kdf_or_kser)._kdf

    @property
    def _internal(self) -> InternalFrame:
        return self._kdf._internal


class AtIndexer(IndexerLike):
    def __getitem__(self, key: Any) -> Union["Series", "DataFrame", Scalar]:
        if self._is_df:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError("Use DataFrame.at like .at[row_index, column_name]")
            row_sel: Any
            col_sel: Any
            row_sel, col_sel = key
        else:
            assert self._is_series, type(self._kdf_or_kser)
            if isinstance(key, tuple) and len(key) != 1:
                raise TypeError("Use Series.at like .at[row_index]")
            row_sel = key
            col_sel = cast("Series", self._kdf_or_kser)._column_label

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

        cond: spark.Column = reduce(
            lambda x, y: x & y,
            [scol == row for scol, row in zip(self._internal.index_spark_columns, row_sel)],
        )
        pdf = (
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
            row_sel: Any
            col_sel: Any
            row_sel, col_sel = key
            if not isinstance(row_sel, int) or not isinstance(col_sel, int):
                raise ValueError("iAt based indexing can only have integer indexers")
            return cast("DataFrame", self._kdf_or_kser).iloc[row_sel, col_sel]
        else:
            assert self._is_series, type(self._kdf_or_kser)
            if not isinstance(key, int) and len(key) != 1:
                raise TypeError("Use Series.iat like .iat[row_integer_position]")
            if not isinstance(key, int):
                raise ValueError("iAt based indexing can only have integer indexers")
            return cast("Series", self._kdf_or_kser).iloc[key]


class LocIndexerLike(IndexerLike, metaclass=ABCMeta):
    def _select_rows(
        self, rows_sel: Any
    ) -> Tuple[Optional[spark.Column], Optional[int], Optional[int]]:
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
        List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]
    ]:
        from databricks.koalas.series import Series

        if cols_sel is None:
            column_labels = self._internal.column_labels
            data_spark_columns = self._internal.data_spark_columns
            data_dtypes = self._internal.data_dtypes
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
        List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]
    ]:
        pass

    @abstractmethod
    def _select_cols_by_spark_column(
        self, cols_sel: spark.Column, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]
    ]:
        pass

    @abstractmethod
    def _select_cols_by_slice(
        self, cols_sel: slice, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]
    ]:
        pass

    @abstractmethod
    def _select_cols_by_iterable(
        self, cols_sel: Iterable, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]
    ]:
        pass

    @abstractmethod
    def _select_cols_else(
        self, cols_sel: Any, missing_keys: Optional[List[Tuple]]
    ) -> Tuple[
        List[Tuple], Optional[List[spark.Column]], Optional[List[Dtype]], bool, Optional[Tuple]
    ]:
        pass

    def __getitem__(self, key: Any) -> Union["Series", "DataFrame"]:
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, first_series

        if self._is_series:
            if isinstance(key, Series) and not same_anchor(key, self._kdf_or_kser):
                kdf = cast("Series", self._kdf_or_kser).to_frame()
                temp_col = verify_temp_column_name(kdf, "__temp_col__")

                kdf[temp_col] = key
                return type(self)(kdf[cast("Series", self._kdf_or_kser).name])[kdf[temp_col]]

            cond, limit, remaining_index = self._select_rows(key)
            if cond is None and limit is None:
                return cast("Series", self._kdf_or_kser)

            column_label = cast("Series", self._kdf_or_kser)._column_label
            column_labels = [column_label]
            data_spark_columns = [self._internal.spark_column_for(column_label)]
            data_dtypes = [self._internal.dtype_for(column_label)]
            returns_series = True
            series_name = cast("Series", self._kdf_or_kser).name
        else:
            assert self._is_df
            rows_sel: Any
            cols_sel: Any
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise SparkPandasIndexingError("Only accepts pairs of candidates")
                rows_sel, cols_sel = key
            else:
                rows_sel = key
                cols_sel = None

            if isinstance(rows_sel, Series) and not same_anchor(rows_sel, self._kdf_or_kser):
                kdf = cast("DataFrame", self._kdf_or_kser).copy()
                temp_col = verify_temp_column_name(kdf, "__temp_col__")

                kdf[temp_col] = rows_sel
                return type(self)(kdf)[kdf[temp_col], cols_sel][list(cast("DataFrame", self._kdf_or_kser).columns)]

            cond, limit, remaining_index = self._select_rows(rows_sel)
            (
                column_labels,
                data_spark_columns,
                data_dtypes,
                returns_series,
                series_name,
            ) = self._select_cols(cols_sel)

            if cond is None and limit is None and returns_series:
                kser = cast("DataFrame", self._kdf_or_kser)._kser_for(column_labels[0])
                if series_name is not None and series_name != kser.name:
                    kser = kser.rename(series_name)
                return kser

        if remaining_index is not None:
            index_spark_columns = self._internal.index_spark_columns[-remaining_index:]
            index_names = self._internal.index_names[-remaining_index:]
            index_dtypes = self._internal.index_dtypes[-remaining_index:]
        else:
            index_spark_columns = self._internal.index_spark_columns
            index_names = self._internal.index_names
            index_dtypes = self._internal.index_dtypes

        if len(column_labels) > 0:
            column_labels = column_labels.copy()
            column_labels_level = max(
                len(label) if label is not None else 1 for label in column_labels
            )
            none_column = 0
            for i, label in enumerate(column_labels):
                if label is None:
                    label = (none_column,)
                    none_column += 1
                if len(label) < column_labels_level:
                    label = tuple(list(label) + ([""]) * (column_labels_level - len(label)))
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
            raise KeyError(
                "[{}] don't exist in columns".format(
                    [col._jc.toString() for col in data_spark_columns]
                )
            )

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
        from databricks