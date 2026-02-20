from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING, cast
from typing import IO

import warnings

import numpy as np  # noqa: F401
import pandas as pd
from pandas.api.types import is_list_like

import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegralType,
    LongType,
    NumericType,
    DataType,
)
from pyspark.sql.column import Column

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
    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        pass

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @abstractmethod
    def _apply_series_op(self, func: Any) -> Any:
        pass

    @abstractmethod
    def _reduce_for_stat_function(self, f, name: str, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def index(self) -> Any:
        pass

    @abstractmethod
    def copy(self) -> Any:
        pass

    @abstractmethod
    def _to_internal_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def to_spark(self, index_col: Optional[Union[str, List[str]]] = None) -> Any:
        pass

    @abstractmethod
    def head(self, n: int = 5) -> Any:
        pass

    def cummin(self) -> Any:
        # Implementation omitted.
        pass

    def cummax(self) -> Any:
        # Implementation omitted.
        pass

    def cumsum(self) -> Any:
        # Implementation omitted.
        pass

    def cumprod(self) -> Any:
        # Implementation omitted.
        pass

    def mean(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def mean_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.mean(spark_column)

        return self._reduce_for_stat_function(mean_func, name="mean", axis=axis, numeric_only=numeric_only)

    def sum(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def sum_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.coalesce(F.sum(spark_column), F.lit(0))

        return self._reduce_for_stat_function(sum_func, name="sum", axis=axis, numeric_only=numeric_only, min_count=min_count)

    def product(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def prod(spark_column: Column, spark_type: DataType) -> Any:
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

        return self._reduce_for_stat_function(prod, name="prod", axis=axis, numeric_only=numeric_only, min_count=min_count)

    prod = product

    def skew(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def skew_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.skewness(spark_column)

        return self._reduce_for_stat_function(skew_func, name="skew", axis=axis, numeric_only=numeric_only)

    def kurtosis(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def kurtosis_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )
            return F.kurtosis(spark_column)

        return self._reduce_for_stat_function(kurtosis_func, name="kurtosis", axis=axis, numeric_only=numeric_only)

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

        def std_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type),
                        spark_type.simpleString(),
                    )
                )
            if ddof == 0:
                return F.stddev_pop(spark_column)
            else:
                return F.stddev_samp(spark_column)

        return self._reduce_for_stat_function(std_func, name="std", axis=axis, numeric_only=numeric_only, ddof=ddof)

    def var(self, axis: Optional[Union[int, str]] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)

        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def var_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type),
                        spark_type.simpleString(),
                    )
                )
            if ddof == 0:
                return F.var_pop(spark_column)
            else:
                return F.var_samp(spark_column)

        return self._reduce_for_stat_function(var_func, name="var", axis=axis, numeric_only=numeric_only, ddof=ddof)

    def median(self, axis: Optional[Union[int, str]] = None, numeric_only: Optional[bool] = None, accuracy: int = 10000) -> Union[Scalar, "Series"]:
        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        if not isinstance(accuracy, int):
            raise ValueError("accuracy must be an integer; however, got [%s]" % type(accuracy).__name__)

        def median_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, (BooleanType, NumericType)):
                return SF.percentile_approx(spark_column.cast(DoubleType()), 0.5, accuracy)
            else:
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type),
                        spark_type.simpleString(),
                    )
                )

        return self._reduce_for_stat_function(median_func, name="median", numeric_only=numeric_only, axis=axis)

    def sem(self, axis: Optional[Union[int, str]] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)

        axis = validate_axis(axis)

        if numeric_only is None and axis == 0:
            numeric_only = True

        def std_func(spark_column: Column, spark_type: DataType) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type),
                        spark_type.simpleString(),
                    )
                )
            if ddof == 0:
                return F.stddev_pop(spark_column)
            else:
                return F.stddev_samp(spark_column)

        def sem_func(spark_column: Column, spark_type: DataType) -> Any:
            return std_func(spark_column, spark_type) / pow(Frame._count_expr(spark_column, spark_type), 0.5)

        return self._reduce_for_stat_function(sem_func, name="sem", numeric_only=numeric_only, axis=axis, ddof=ddof)

    @property
    def size(self) -> int:
        num_columns: int = len(self._internal.data_spark_columns)
        if num_columns == 0:
            return 0
        else:
            return len(self) * num_columns  # type: ignore

    def abs(self) -> Union["DataFrame", "Series"]:
        def abs_func(kser: Any) -> Any:
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

        return self._apply_series_op(abs_func)

    def groupby(self, by: Any, axis: Union[int, str] = 0, as_index: bool = True, dropna: bool = True) -> Union["DataFrameGroupBy", "SeriesGroupBy"]:
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
            by = [(by,)]
        elif is_list_like(by):
            new_by: List[Union[Tuple, ks.Series]] = []
            for key in by:
                if isinstance(key, ks.DataFrame):
                    raise ValueError("Grouper for '{}' not 1-dimensional".format(type(key).__name__))
                elif isinstance(key, ks.Series):
                    new_by.append(key)
                elif is_name_like_tuple(key):
                    if isinstance(self, ks.Series):
                        raise KeyError(key)
                    new_by.append(key)
                elif is_name_like_value(key):
                    if isinstance(self, ks.Series):
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

        if isinstance(self, ks.DataFrame):
            return DataFrameGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        elif isinstance(self, ks.Series):
            return SeriesGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        else:
            raise TypeError("Constructor expects DataFrame or Series; however, got [%s]" % (self,))

    def bool(self) -> bool:
        if isinstance(self, ks.DataFrame):
            df = self
        elif isinstance(self, ks.Series):
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

        if len(last_valid_rows) == 0:
            return None

        last_valid_row = last_valid_rows[0]

        if len(last_valid_row) == 1:
            return last_valid_row[0]
        else:
            return tuple(last_valid_row)

    def rolling(self, window: Any, min_periods: Optional[int] = None) -> Rolling:
        return Rolling(self, window=window, min_periods=min_periods)

    def expanding(self, min_periods: int = 1) -> Expanding:
        return Expanding(self, min_periods=min_periods)

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def squeeze(self, axis: Optional[Union[int, str]] = None) -> Union[Scalar, "DataFrame", "Series"]:
        if axis is not None:
            axis = "index" if axis == "rows" else axis
            axis = validate_axis(axis)

        if isinstance(self, ks.DataFrame):
            from databricks.koalas.series import first_series

            is_squeezable = len(self.columns[:2]) == 1
            if not is_squeezable:
                return self
            series_from_column = first_series(self)
            has_single_value = len(series_from_column.head(2)) == 1
            if has_single_value:
                result = self._to_internal_pandas().squeeze(axis)
                return ks.Series(result) if isinstance(result, pd.Series) else result
            elif axis == 0:
                return self
            else:
                return series_from_column
        else:
            self_top_two = self.head(2)
            has_single_value = len(self_top_two) == 1
            return cast(Union[Scalar, ks.Series], self_top_two[0] if has_single_value else self)

    def truncate(self, before: Optional[Any] = None, after: Optional[Any] = None, axis: Optional[Union[int, str]] = None, copy: bool = True) -> Union["DataFrame", "Series"]:
        from databricks.koalas.series import first_series

        axis = validate_axis(axis)
        indexes = self.index
        indexes_increasing = indexes.is_monotonic_increasing
        if not indexes_increasing and not indexes.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")
        if (before is None) and (after is None):
            return self.copy() if copy else self  # type: ignore
        if (before is not None and after is not None) and before > after:
            raise ValueError("Truncate: %s must be after %s" % (after, before))

        if isinstance(self, ks.Series):
            if indexes_increasing:
                result = first_series(self.to_frame().loc[before:after]).rename(self.name)
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

        return result.copy() if copy else result  # type: ignore

    def to_markdown(self, buf: Optional[IO[Any]] = None, mode: Optional[str] = None) -> str:
        args = locals()
        kser_or_kdf = self
        internal_pandas = kser_or_kdf._to_internal_pandas()
        return validate_arguments_and_invoke_function(
            internal_pandas, self.to_markdown, type(internal_pandas).to_markdown, args
        )

    @abstractmethod
    def fillna(self, value: Optional[Any] = None, method: Optional[str] = None, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series", None]:
        pass

    def bfill(self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series"]:
        return self.fillna(method="bfill", axis=axis, inplace=inplace, limit=limit)  # type: ignore

    backfill = bfill

    def ffill(self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series"]:
        return self.fillna(method="ffill", axis=axis, inplace=inplace, limit=limit)  # type: ignore

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
            "The truth value of a {0} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(self.__class__.__name__)
        )

    @staticmethod
    def _count_expr(spark_column: Column, spark_type: DataType) -> Column:
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(spark_column, F.lit(None)))
        else:
            return F.count(spark_column)