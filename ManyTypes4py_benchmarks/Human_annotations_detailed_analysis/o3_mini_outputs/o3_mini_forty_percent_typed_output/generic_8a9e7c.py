#!/usr/bin/env python
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

from functools import reduce
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.types import BooleanType, DoubleType, FloatType, IntegralType, LongType, NumericType
import pandas as pd
import numpy as np

# Assume other internal imports like InternalFrame, Scalar, Rolling, Expanding,
# AtIndexer, iAtIndexer, iLocIndexer, LocIndexer are available in the environment.

class Frame:
    # Abstract methods that should be implemented in concrete subclasses.
    
    def __getitem__(self, key: Any) -> Any:
        raise NotImplementedError

    @property
    def _internal(self) -> Any:
        raise NotImplementedError

    def _apply_series_op(self, func: Callable[[Any], Any]) -> Any:
        raise NotImplementedError

    def _reduce_for_stat_function(
        self, func: Callable[[Any, Any], Any], name: str, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, **kwargs: Any
    ) -> Union[Any, "Series"]:
        raise NotImplementedError

    def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def dtypes(self) -> Any:
        raise NotImplementedError

    @property
    def index(self) -> Any:
        raise NotImplementedError

    def copy(self) -> Any:
        raise NotImplementedError

    def _to_internal_pandas(self) -> pd.DataFrame:
        raise NotImplementedError

    def head(self, n: int = 5) -> Any:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    def cummin(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        return self._apply_series_op(lambda kser: kser._cummin(skipna))  # type: ignore

    def cummax(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        return self._apply_series_op(lambda kser: kser._cummax(skipna))  # type: ignore

    def cumsum(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        return self._apply_series_op(lambda kser: kser._cumsum(skipna))  # type: ignore

    def cumprod(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
        return self._apply_series_op(lambda kser: kser._cumprod(skipna))  # type: ignore

    # -------------------------------------------------------------------------
    def get_dtype_counts(self) -> pd.Series:
        dtypes = self.dtypes if hasattr(self.dtypes, '__iter__') else [self.dtypes]
        return pd.Series({d.name: list(self.dtypes).count(d) for d in set(self.dtypes)})

    def pipe(self, func: Union[Callable[..., Any], Tuple[Callable[..., Any], str]], *args: Any, **kwargs: Any) -> Any:
        if isinstance(func, tuple):
            func_callable, target = func
            if target in kwargs:
                raise ValueError("%s is both the pipe target and a keyword argument" % target)
            kwargs[target] = self
            return func_callable(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    def to_numpy(self) -> np.ndarray:
        return self.to_pandas().values

    @property
    def values(self) -> np.ndarray:
        return self.to_numpy()

    def to_csv(
        self,
        path: Optional[str] = None,
        sep: str = ",",
        na_rep: str = "",
        columns: Optional[Iterable[Any]] = None,
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
        if path is None:
            kdf_or_ser = self
            if pd.__version__ < "0.24" and isinstance(self, Series):
                return kdf_or_ser.to_pandas().to_csv(
                    None, sep=sep, na_rep=na_rep, header=header, date_format=date_format, index=False
                )
            else:
                return kdf_or_ser.to_pandas().to_csv(
                    None,
                    sep=sep,
                    na_rep=na_rep,
                    columns=columns,
                    header=header,
                    quotechar=quotechar,
                    date_format=date_format,
                    escapechar=escapechar,
                    index=False,
                )
        kdf = self if not isinstance(self, Series) else self.to_frame()
        if columns is None:
            column_labels = kdf._internal.column_labels
        else:
            column_labels = []
            for label in columns:
                if not isinstance(label, tuple):
                    label = (label,)
                if label not in kdf._internal.column_labels:
                    raise KeyError(label)
                column_labels.append(label)
        if isinstance(index_col, str):
            index_cols: List[str] = [index_col]
        elif index_col is None:
            index_cols = []
        else:
            index_cols = index_col
        if header is True and kdf._internal.column_labels_level > 1:
            raise ValueError("to_csv only support one-level index column now")
        elif isinstance(header, list):
            sdf = kdf.to_spark(index_col)  # type: ignore
            sdf = sdf.select(
                [scol for scol in sdf.columns if scol in index_cols]
                + [scol for i, (label, new_name) in enumerate(zip(column_labels, header))]
            )
            header = True
        else:
            sdf = kdf.to_spark(index_col)  # type: ignore
            sdf = sdf.select(
                [scol for scol in sdf.columns if scol in index_cols]
                + [scol for i, label in enumerate(column_labels)]
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
        if not lines:
            raise NotImplementedError("lines=False is not implemented yet.")
        if orient != "records":
            raise NotImplementedError("orient='records' is supported only for now.")
        if path is None:
            kdf_or_ser = self
            pdf = kdf_or_ser.to_pandas()  # type: ignore
            if isinstance(self, Series):
                pdf = pdf.to_frame()
            return pdf.to_json(orient="records")
        kdf = self if not isinstance(self, Series) else self.to_frame()
        sdf = kdf.to_spark(index_col=index_col)  # type: ignore
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
        columns: Optional[Iterable[Any]] = None,
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
        args = locals()
        kdf = self
        if isinstance(self, DataFrame):
            f = pd.DataFrame.to_excel
        elif isinstance(self, Series):
            f = pd.Series.to_excel
        else:
            raise TypeError("Constructor expects DataFrame or Series; however, got [%s]" % (self,))
        return validate_arguments_and_invoke_function(
            kdf._to_internal_pandas(), self.to_excel, f, args
        )

    def mean(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        def mean(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.mean(spark_column)
        return self._reduce_for_stat_function(mean, name="mean", axis=axis, numeric_only=numeric_only)

    def sum(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        def sum_(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.coalesce(F.sum(spark_column), F.lit(0))
        return self._reduce_for_stat_function(sum_, name="sum", axis=axis, numeric_only=numeric_only, min_count=min_count)

    def product(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        def prod(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                scol = F.min(F.coalesce(spark_column, F.lit(True))).cast(LongType())
            elif isinstance(spark_type, NumericType):
                num_zeros = F.sum(F.when(spark_column == 0, 1).otherwise(0))
                sign = F.when(F.sum(F.when(spark_column < 0, 1).otherwise(0)) % 2 == 0, 1).otherwise(-1)
                scol = F.when(num_zeros > 0, 0).otherwise(sign * F.exp(F.sum(F.log(F.abs(spark_column)))))
                if isinstance(spark_type, IntegralType):
                    scol = F.round(scol).cast(LongType())
            else:
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.coalesce(scol, F.lit(1))
        return self._reduce_for_stat_function(prod, name="prod", axis=axis, numeric_only=numeric_only, min_count=min_count)
    prod = product

    def skew(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        def skew(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.skewness(spark_column)
        return self._reduce_for_stat_function(skew, name="skew", axis=axis, numeric_only=numeric_only)

    def kurtosis(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        def kurtosis(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.kurtosis(spark_column)
        return self._reduce_for_stat_function(kurtosis, name="kurtosis", axis=axis, numeric_only=numeric_only)
    kurt = kurtosis

    def min(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        return self._reduce_for_stat_function(F.min, name="min", axis=axis, numeric_only=numeric_only)

    def max(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        return self._reduce_for_stat_function(F.max, name="max", axis=axis, numeric_only=numeric_only)

    def count(self, axis: Optional[Union[int, str]] = None, numeric_only: bool = False) -> Union[Scalar, "Series"]:
        return self._reduce_for_stat_function(Frame._count_expr, name="count", axis=axis, numeric_only=numeric_only)

    def std(self, axis: Optional[Union[int, str]] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        def std(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.stddev_pop(spark_column) if ddof == 0 else F.stddev_samp(spark_column)
        return self._reduce_for_stat_function(std, name="std", axis=axis, numeric_only=numeric_only, ddof=ddof)

    def var(self, axis: Optional[Union[int, str]] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        def var(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.var_pop(spark_column) if ddof == 0 else F.var_samp(spark_column)
        return self._reduce_for_stat_function(var, name="var", axis=axis, numeric_only=numeric_only, ddof=ddof)

    def median(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, accuracy: int = 10000) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        if not isinstance(accuracy, int):
            raise ValueError("accuracy must be an integer; however, got [%s]" % type(accuracy).__name__)
        def median(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, (BooleanType, NumericType)):
                return F.expr("percentile_approx({}, 0.5, {})".format(spark_column.__str__(), accuracy))
            else:
                raise TypeError("Could not convert {} to numeric".format(spark_type))
        return self._reduce_for_stat_function(median, name="median", axis=axis, numeric_only=numeric_only)

    def sem(self, axis: Optional[Union[int, str]] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        def std(spark_column: Column, spark_type: Any) -> Column:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} to numeric".format(spark_type))
            return F.stddev_pop(spark_column) if ddof == 0 else F.stddev_samp(spark_column)
        def sem(spark_column: Column, spark_type: Any) -> Column:
            return std(spark_column, spark_type) / F.sqrt(Frame._count_expr(spark_column, spark_type))
        return self._reduce_for_stat_function(sem, name="sem", axis=axis, numeric_only=numeric_only, ddof=ddof)

    @property
    def size(self) -> int:
        num_columns = len(self._internal.data_spark_columns)
        if num_columns == 0:
            return 0
        else:
            return len(self) * num_columns  # type: ignore

    def abs(self) -> Union["DataFrame", "Series"]:
        def abs_op(kser: Any) -> Any:
            if isinstance(kser.spark.data_type, BooleanType):
                return kser
            elif isinstance(kser.spark.data_type, NumericType):
                return kser.spark.transform(F.abs)
            else:
                raise TypeError("bad operand type for abs(): {}".format(kser.spark.data_type))
        return self._apply_series_op(abs_op)

    def groupby(
        self,
        by: Any,
        axis: Union[int, str] = 0,
        as_index: bool = True,
        dropna: bool = True,
    ) -> Union["DataFrameGroupBy", "SeriesGroupBy"]:
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
        if isinstance(by, DataFrame):
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        elif isinstance(by, Series):
            by = [by]
        elif isinstance(by, tuple):
            if isinstance(self, Series):
                raise KeyError(by)
            by = [by]
        elif isinstance(by, (str, int)):
            if isinstance(self, Series):
                raise KeyError(by)
            by = [(by,)]
        elif isinstance(by, list):
            new_by: List[Any] = []
            for key in by:
                if isinstance(key, DataFrame):
                    raise ValueError("Grouper for '{}' not 1-dimensional".format(type(key).__name__))
                elif isinstance(key, Series):
                    new_by.append(key)
                elif isinstance(key, tuple):
                    if isinstance(self, Series):
                        raise KeyError(key)
                    new_by.append(key)
                elif isinstance(key, (str, int)):
                    if isinstance(self, Series):
                        raise KeyError(key)
                    new_by.append((key,))
                else:
                    raise ValueError("Grouper for '{}' not 1-dimensional".format(type(key).__name__))
            by = new_by
        else:
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        if not len(by):
            raise ValueError("No group keys passed!")
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if isinstance(self, DataFrame):
            return DataFrameGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        elif isinstance(self, Series):
            return SeriesGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        else:
            raise TypeError("Constructor expects DataFrame or Series; however, got [%s]" % (self,))

    def bool(self) -> bool:
        if isinstance(self, DataFrame):
            df = self
        elif isinstance(self, Series):
            df = self.to_dataframe()
        else:
            raise TypeError("bool() expects DataFrame or Series; however, got [%s]" % (self,))
        return df.head(2)._to_internal_pandas().bool()

    def first_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        data_spark_columns = self._internal.data_spark_columns
        if len(data_spark_columns) == 0:
            return None
        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))
        with sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            first_valid_row = self._internal.spark_frame.filter(cond).select(self._internal.index_spark_columns).limit(1).toPandas()
        if len(first_valid_row) == 0:
            return None
        first_valid_row = first_valid_row.iloc[0]
        if len(first_valid_row) == 1:
            return first_valid_row.iloc[0]
        else:
            return tuple(first_valid_row)

    def last_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        from distutils.version import LooseVersion
        import pyspark
        if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
            raise RuntimeError("last_valid_index can be used in PySpark >= 3.0")
        data_spark_columns = self._internal.data_spark_columns
        if len(data_spark_columns) == 0:
            return None
        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))
        last_valid_rows = self._internal.spark_frame.filter(cond).select(self._internal.index_spark_columns).tail(1)
        if len(last_valid_rows) == 0:
            return None
        last_valid_row = last_valid_rows[0]
        if len(last_valid_row) == 1:
            return last_valid_row[0]
        else:
            return tuple(last_valid_row)

    def rolling(self, window: Union[int, Any], min_periods: Optional[int] = None) -> "Rolling":
        return Rolling(self, window=window, min_periods=min_periods)

    def expanding(self, min_periods: int = 1) -> "Expanding":
        return Expanding(self, min_periods=min_periods)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def squeeze(self, axis: Optional[Union[int, str]] = None) -> Union[Scalar, "DataFrame", "Series"]:
        if axis is not None:
            axis = "index" if axis == "rows" else axis
            axis = validate_axis(axis)
        if isinstance(self, DataFrame):
            from databricks.koalas.series import first_series
            is_squeezable = len(self.columns[:2]) == 1
            if not is_squeezable:
                return self
            series_from_column = first_series(self)
            has_single_value = len(series_from_column.head(2)) == 1
            if has_single_value:
                result = self._to_internal_pandas().squeeze(axis)
                return Series(result) if isinstance(result, pd.Series) else result
            elif axis == 0:
                return self
            else:
                return series_from_column
        else:
            self_top_two = self.head(2)
            has_single_value = len(self_top_two) == 1
            return self_top_two[0] if has_single_value else self

    def truncate(
        self, before: Optional[Any] = None, after: Optional[Any] = None, axis: Optional[Union[int, str]] = None, copy: bool = True
    ) -> Union["DataFrame", "Series"]:
        from databricks.koalas.series import first_series
        axis = validate_axis(axis)
        indexes = self.index
        indexes_increasing = indexes.is_monotonic_increasing
        if not indexes_increasing and not indexes.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")
        if (before is None) and (after is None):
            return self.copy() if copy else self
        if (before is not None and after is not None) and before > after:
            raise ValueError("Truncate: %s must be after %s" % (after, before))
        if isinstance(self, Series):
            if indexes_increasing:
                result = first_series(self.to_frame().loc[before:after]).rename(self.name)
            else:
                result = first_series(self.to_frame().loc[after:before]).rename(self.name)
        elif isinstance(self, DataFrame):
            if axis == 0:
                result = self.loc[before:after] if indexes_increasing else self.loc[after:before]
            elif axis == 1:
                result = self.loc[:, before:after]
        return result.copy() if copy else result

    def to_markdown(self, buf: Optional[Any] = None, mode: Optional[str] = None) -> str:
        if pd.__version__ < "1.0.0":
            raise NotImplementedError("`to_markdown()` only supported in Koalas with pandas >= 1.0.0")
        args = locals()
        kser_or_kdf = self
        internal_pandas = kser_or_kdf._to_internal_pandas()
        return validate_arguments_and_invoke_function(internal_pandas, self.to_markdown, type(internal_pandas).to_markdown, args)

    def fillna(self, value: Any = None, method: Optional[str] = None, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series"]:
        raise NotImplementedError

    def bfill(self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series"]:
        return self.fillna(method="bfill", axis=axis, inplace=inplace, limit=limit)
    backfill = bfill

    def ffill(self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series"]:
        return self.fillna(method="ffill", axis=axis, inplace=inplace, limit=limit)
    pad = ffill

    @property
    def at(self) -> "AtIndexer":
        return AtIndexer(self)
    at.__doc__ = AtIndexer.__doc__

    @property
    def iat(self) -> "iAtIndexer":
        return iAtIndexer(self)
    iat.__doc__ = iAtIndexer.__doc__

    @property
    def iloc(self) -> "iLocIndexer":
        return iLocIndexer(self)
    iloc.__doc__ = iLocIndexer.__doc__

    @property
    def loc(self) -> "LocIndexer":
        return LocIndexer(self)
    loc.__doc__ = LocIndexer.__doc__

    def __bool__(self) -> bool:
        raise ValueError("The truth value of a {} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(self.__class__.__name__))

    @staticmethod
    def _count_expr(spark_column: Column, spark_type: Any) -> Column:
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(spark_column, F.lit(None)))
        else:
            return F.count(spark_column)