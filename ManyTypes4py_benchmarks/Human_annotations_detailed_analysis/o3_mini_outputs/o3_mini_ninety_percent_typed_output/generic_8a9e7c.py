from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Any, IO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.sql import functions as F  # type: ignore
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegralType,
    LongType,
    NumericType,
)
from pyspark import __version__ as pyspark_version
from distutils.version import LooseVersion

from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
from databricks.koalas.internal import InternalFrame
from databricks.koalas.window import Expanding, Rolling
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer
from databricks.koalas.typedef import Scalar
from databricks.koalas.utils import sql_conf, SPARK_CONF_ARROW_ENABLED
from databricks.koalas.spark import functions as SF
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer


class Frame(metaclass=ABCMeta):
    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        ...

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        ...

    @abstractmethod
    def _to_internal_pandas(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        ...

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def squeeze(self, axis: Optional[Union[int, str]] = None) -> Union[Scalar, "DataFrame", "Series"]:
        if axis is not None:
            axis = "index" if axis == "rows" else axis
            axis = validate_axis(axis)
        from databricks.koalas.series import DataFrame, Series  # type: ignore
        if isinstance(self, DataFrame):
            from databricks.koalas.series import first_series
            is_squeezable: bool = len(self.columns[:2]) == 1
            if not is_squeezable:
                return self
            series_from_column: Series = first_series(self)
            has_single_value: bool = len(series_from_column.head(2)) == 1
            if has_single_value:
                result = self._to_internal_pandas().squeeze(axis)
                return Series(result) if isinstance(result, pd.Series) else result
            elif axis == 0:
                return self
            else:
                return series_from_column
        else:
            # The case of Series
            self_top_two = self.head(2)
            has_single_value = len(self_top_two) == 1
            return self_top_two[0] if has_single_value else self

    def truncate(
        self,
        before: Optional[Any] = None,
        after: Optional[Any] = None,
        axis: Optional[Union[int, str]] = None,
        copy: bool = True,
    ) -> Union["DataFrame", "Series"]:
        from databricks.koalas.indexing import validate_axis
        axis = validate_axis(axis)
        indexes = self.index
        indexes_increasing = indexes.is_monotonic_increasing
        if not indexes_increasing and not indexes.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")
        if (before is None) and (after is None):
            return self.copy() if copy else self
        if (before is not None and after is not None) and before > after:
            raise ValueError("Truncate: %s must be after %s" % (after, before))
        from databricks.koalas.series import first_series
        if hasattr(self, "to_frame") and not isinstance(self, type(pd.DataFrame())):
            # Series
            if indexes_increasing:
                result = first_series(self.to_frame().loc[before:after]).rename(self.name)
            else:
                result = first_series(self.to_frame().loc[after:before]).rename(self.name)
        else:
            # DataFrame
            if axis == 0:
                if indexes_increasing:
                    result = self.loc[before:after]
                else:
                    result = self.loc[after:before]
            elif axis == 1:
                result = self.loc[:, before:after]
            else:
                raise ValueError("axis should be either 0 or 1")
        return result.copy() if copy else result

    def to_markdown(self, buf: Optional[IO[Any]] = None, mode: Optional[str] = None) -> str:
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            raise NotImplementedError("`to_markdown()` only supported in Koalas with pandas >= 1.0.0")
        args = locals()
        internal_pandas = self._to_internal_pandas()
        return validate_arguments_and_invoke_function(
            internal_pandas, self.to_markdown, type(internal_pandas).to_markdown, args
        )

    @abstractmethod
    def fillna(
        self,
        value: Optional[Any] = None,
        method: Optional[str] = None,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Union["DataFrame", "Series", None]:
        pass

    def bfill(
        self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None
    ) -> Union["DataFrame", "Series"]:
        return self.fillna(method="bfill", axis=axis, inplace=inplace, limit=limit)

    backfill = bfill

    def ffill(
        self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None
    ) -> Union["DataFrame", "Series"]:
        return self.fillna(method="ffill", axis=axis, inplace=inplace, limit=limit)

    pad = ffill

    @property
    def at(self) -> AtIndexer:
        return AtIndexer(self)

    @property
    def iat(self) -> iAtIndexer:
        return iAtIndexer(self)

    @property
    def iloc(self) -> iLocIndexer:
        return iLocIndexer(self)

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)

    def __bool__(self) -> bool:
        raise ValueError(
            "The truth value of a {0} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(
                self.__class__.__name__
            )
        )

    @staticmethod
    def _count_expr(spark_column: Any, spark_type: Any) -> Any:
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(spark_column, F.lit(None)))
        else:
            return F.count(spark_column)

    def rolling(self, window: Union[int, str], min_periods: Optional[int] = None) -> Rolling:
        return Rolling(self, window=window, min_periods=min_periods)

    def expanding(self, min_periods: int = 1) -> Expanding:
        return Expanding(self, min_periods=min_periods)

    def mean(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def mean_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            return F.mean(spark_column)

        return self._reduce_for_stat_function(mean_func, name="mean", axis=axis, numeric_only=numeric_only)

    def sum(self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def sum_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            return F.coalesce(F.sum(spark_column), F.lit(0))

        return self._reduce_for_stat_function(sum_func, name="sum", axis=axis, numeric_only=numeric_only, min_count=min_count)

    def product(self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def prod_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                scol = F.min(F.coalesce(spark_column, F.lit(True))).cast(LongType())
            elif isinstance(spark_type, NumericType):
                num_zeros = F.sum(F.when(spark_column == 0, 1).otherwise(0))
                sign = F.when(F.sum(F.when(spark_column < 0, 1).otherwise(0)) % 2 == 0, 1).otherwise(-1)
                scol = F.when(num_zeros > 0, 0).otherwise(sign * F.exp(F.sum(F.log(F.abs(spark_column)))))
                if isinstance(spark_type, IntegralType):
                    scol = F.round(scol).cast(LongType())
            else:
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            return F.coalesce(scol, F.lit(1))

        return self._reduce_for_stat_function(prod_func, name="prod", axis=axis, numeric_only=numeric_only, min_count=min_count)

    prod = product

    def skew(self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def skew_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            return F.skewness(spark_column)

        return self._reduce_for_stat_function(skew_func, name="skew", axis=axis, numeric_only=numeric_only)

    def kurtosis(self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def kurtosis_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            return F.kurtosis(spark_column)

        return self._reduce_for_stat_function(kurtosis_func, name="kurtosis", axis=axis, numeric_only=numeric_only)

    kurt = kurtosis

    def min(self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        return self._reduce_for_stat_function(F.min, name="min", axis=axis, numeric_only=numeric_only)

    def max(self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        return self._reduce_for_stat_function(F.max, name="max", axis=axis, numeric_only=numeric_only)

    def count(self, axis: Union[int, str] = None, numeric_only: bool = False) -> Union[Scalar, "Series"]:
        return self._reduce_for_stat_function(Frame._count_expr, name="count", axis=axis, numeric_only=numeric_only)

    def std(self, axis: Union[int, str] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def std_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            if ddof == 0:
                return F.stddev_pop(spark_column)
            else:
                return F.stddev_samp(spark_column)

        return self._reduce_for_stat_function(std_func, name="std", axis=axis, numeric_only=numeric_only, ddof=ddof)

    def var(self, axis: Union[int, str] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def var_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            if ddof == 0:
                return F.var_pop(spark_column)
            else:
                return F.var_samp(spark_column)

        return self._reduce_for_stat_function(var_func, name="var", axis=axis, numeric_only=numeric_only, ddof=ddof)

    def median(self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None, accuracy: int = 10000) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        if not isinstance(accuracy, int):
            raise ValueError("accuracy must be an integer; however, got [%s]" % type(accuracy).__name__)

        def median_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, (BooleanType, NumericType)):
                return SF.percentile_approx(spark_column.cast(DoubleType()), 0.5, accuracy)
            else:
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
        return self._reduce_for_stat_function(median_func, name="median", numeric_only=numeric_only, axis=axis)

    def sem(self, axis: Union[int, str] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def std_func(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError("Could not convert {} ({}) to numeric".format(
                    spark_type, spark_type.simpleString()
                ))
            if ddof == 0:
                return F.stddev_pop(spark_column)
            else:
                return F.stddev_samp(spark_column)

        def sem_func(spark_column: Any, spark_type: Any) -> Any:
            return std_func(spark_column, spark_type) / pow(Frame._count_expr(spark_column, spark_type), 0.5)

        return self._reduce_for_stat_function(sem_func, name="sem", numeric_only=numeric_only, axis=axis, ddof=ddof)

    @property
    def size(self) -> int:
        num_columns = len(self._internal.data_spark_columns)
        if num_columns == 0:
            return 0
        else:
            return len(self) * num_columns

    def abs(self) -> Union["DataFrame", "Series"]:
        def abs_func(kser: Any) -> Any:
            if isinstance(kser.spark.data_type, BooleanType):
                return kser
            elif isinstance(kser.spark.data_type, NumericType):
                return kser.spark.transform(F.abs)
            else:
                raise TypeError("bad operand type for abs(): {} ({})".format(
                    kser.spark.data_type, kser.spark.data_type.simpleString()
                ))
        return self._apply_series_op(abs_func)

    def groupby(
        self, by: Any, axis: Union[int, str] = 0, as_index: bool = True, dropna: bool = True
    ) -> Union[DataFrameGroupBy, SeriesGroupBy]:
        if isinstance(by, type(self)):
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        elif isinstance(by, self.__class__):
            by = [by]
        elif is_name_like_tuple(by):
            if isinstance(self, self.__class__):
                raise KeyError(by)
            by = [by]
        elif is_name_like_value(by):
            if isinstance(self, self.__class__):
                raise KeyError(by)
            by = [(by,)]
        elif hasattr(by, "__iter__"):
            new_by: List[Union[Tuple[Any, ...], Any]] = []
            for key in by:
                if isinstance(key, self.__class__):
                    raise ValueError("Grouper for '{}' not 1-dimensional".format(type(key).__name__))
                elif is_name_like_tuple(key):
                    if isinstance(self, self.__class__):
                        raise KeyError(key)
                    new_by.append((key,))
                elif is_name_like_value(key):
                    if isinstance(self, self.__class__):
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
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
        if hasattr(self, "columns"):
            return DataFrameGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        else:
            return SeriesGroupBy._build(self, by, as_index=as_index, dropna=dropna)

    def bool(self) -> bool:
        if hasattr(self, "to_dataframe"):
            df = self.to_dataframe()
        else:
            df = self
        return df.head(2)._to_internal_pandas().bool()

    def first_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        data_spark_columns = self._internal.data_spark_columns
        if len(data_spark_columns) == 0:
            return None
        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))
        with sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            first_valid_row = (
                self._internal.spark_frame.filter(cond)
                .select(self._internal.index_spark_columns)
                .limit(1)
                .toPandas()
            )
        if len(first_valid_row) == 0:
            return None
        first_valid_row = first_valid_row.iloc[0]
        if len(first_valid_row) == 1:
            return first_valid_row.iloc[0]
        else:
            return tuple(first_valid_row)

    def last_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        if LooseVersion(pyspark_version) < LooseVersion("3.0"):
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
        if len(last_valid_rows) == 0:
            return None
        last_valid_row = last_valid_rows[0]
        if len(last_valid_row) == 1:
            return last_valid_row[0]
        else:
            return tuple(last_valid_row)

    def to_excel(
        self,
        excel_writer: Any,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: Optional[str] = None,
        columns: Optional[List[str]] = None,
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
        if hasattr(self, "to_excel"):
            f = pd.DataFrame.to_excel  # type: ignore
        elif hasattr(self, "to_excel"):
            f = pd.Series.to_excel  # type: ignore
        else:
            raise TypeError("Constructor expects DataFrame or Series; however, got [%s]" % (self,))
        return validate_arguments_and_invoke_function(kdf._to_internal_pandas(), self.to_excel, f, args)  

    # Additional methods omitted for brevity.
    
    @abstractmethod
    def _apply_series_op(self, func: Any) -> Any:
        ...
    
    @abstractmethod
    def _reduce_for_stat_function(self, func: Any, name: str, axis: Union[int, str] = 0, numeric_only: Optional[bool] = None, **kwargs: Any) -> Any:
        ...