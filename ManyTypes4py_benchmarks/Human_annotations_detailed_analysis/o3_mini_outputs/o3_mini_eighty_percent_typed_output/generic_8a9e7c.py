#!/usr/bin/env python
#
# Copyright (C) Databricks Inc. All rights reserved.
#

from abc import abstractmethod
from functools import reduce
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from databricks.koalas.internal import InternalFrame
from databricks.koalas.spark import functions as SF
from databricks.koalas.typedef import Scalar, spark_type_to_pandas_dtype
from databricks.koalas.window import Expanding, Rolling

# Assuming these indexers are defined somewhere in the koalas package.
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer


class Frame:
    @abstractmethod
    def _to_internal_pandas(self) -> Any:
        pass

    @abstractmethod
    def _apply_series_op(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def _reduce_for_stat_function(self, func: Any, name: str, axis: Union[int, str] = 0, numeric_only: Optional[bool] = None, **kwargs: Any) -> Union[Scalar, "Series"]:
        pass

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def copy(self) -> Union["DataFrame", "Series"]:
        pass

    @abstractmethod
    def to_spark(self, index_col: Optional[Union[str, List[str]]] = None) -> Any:
        pass

    @abstractmethod
    def to_frame(self) -> "DataFrame":
        pass

    @abstractmethod
    def head(self, n: int = 5) -> Union["DataFrame", "Series"]:
        pass

    def pipe(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError(f"{target} is both the pipe target and a keyword argument")
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    def to_numpy(self) -> np.ndarray:
        return self.to_pandas().values

    @property
    def values(self) -> np.ndarray:
        import warnings
        warnings.warn("We recommend using `{}.to_numpy()` instead.".format(type(self).__name__))
        return self.to_numpy()

    def to_csv(
        self,
        path: Optional[str] = None,
        sep: str = ",",
        na_rep: str = "",
        columns: Optional[List[Any]] = None,
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
        from distutils.version import LooseVersion
        import pyspark

        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        if path is None:
            kdf_or_ser = self
            if (LooseVersion("0.24") > LooseVersion(pd.__version__)) and hasattr(self, "to_pandas"):
                return kdf_or_ser.to_pandas().to_csv(
                    None,
                    sep=sep,
                    na_rep=na_rep,
                    header=header,
                    date_format=date_format,
                    index=False,
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

        kdf = self
        if hasattr(self, "to_frame") and isinstance(self, Series):  # type: ignore
            kdf = self.to_frame()

        if columns is None:
            column_labels = kdf._internal.column_labels
        else:
            column_labels = []
            from databricks.koalas.utils import is_name_like_tuple, name_like_string
            for label in columns:
                if not is_name_like_tuple(label):
                    label = (label,)
                if label not in kdf._internal.column_labels:
                    raise KeyError(name_like_string(label))
                column_labels.append(label)

        if isinstance(index_col, str):
            index_cols = [index_col]
        elif index_col is None:
            index_cols = []
        else:
            index_cols = index_col

        if header is True and kdf._internal.column_labels_level > 1:
            raise ValueError("to_csv only support one-level index column now")
        elif isinstance(header, list):
            sdf = kdf.to_spark(index_col)  # type: ignore
            from databricks.koalas.utils import scol_for
            sdf = sdf.select(
                [scol_for(sdf, str(col)) for col in index_cols]
                + [
                    scol_for(sdf, str(i) if label is None else str(label)).alias(
                        new_name
                    )
                    for i, (label, new_name) in enumerate(zip(column_labels, header))
                ]
            )
            header = True
        else:
            sdf = kdf.to_spark(index_col)  # type: ignore
            from databricks.koalas.utils import scol_for
            sdf = sdf.select(
                [scol_for(sdf, str(col)) for col in index_cols]
                + [
                    scol_for(sdf, str(i) if label is None else str(label))
                    for i, label in enumerate(column_labels)
                ]
            )

        if num_files is not None:
            sdf = sdf.repartition(num_files)

        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
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
        from distutils.version import LooseVersion
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        if not lines:
            raise NotImplementedError("lines=False is not implemented yet.")

        if orient != "records":
            raise NotImplementedError("orient='records' is supported only for now.")

        if path is None:
            kdf_or_ser = self
            pdf = kdf_or_ser.to_pandas()  # type: ignore
            if hasattr(self, "to_frame") and isinstance(self, Series):  # type: ignore
                pdf = pdf.to_frame()
            return pdf.to_json(orient="records")

        kdf = self
        if hasattr(self, "to_frame") and isinstance(self, Series):  # type: ignore
            kdf = self.to_frame()
        sdf = kdf.to_spark(index_col=index_col)  # type: ignore

        if num_files is not None:
            sdf = sdf.repartition(num_files)

        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format("json").save(path)
        return None

    def to_excel(
        self,
        excel_writer: Union[str, pd.ExcelWriter],
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: Optional[str] = None,
        columns: Optional[Union[List[Any], Tuple[Any, ...]]] = None,
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
        if hasattr(self, "dtypes") and isinstance(self, DataFrame):
            f = pd.DataFrame.to_excel
        elif hasattr(self, "dtypes") and isinstance(self, Series):
            f = pd.Series.to_excel
        else:
            raise TypeError(
                "Constructor expects DataFrame or Series; however, got [%s]" % (self,)
            )
        from databricks.koalas.utils import validate_arguments_and_invoke_function
        return validate_arguments_and_invoke_function(
            kdf._to_internal_pandas(), self.to_excel, f, args
        )

    def mean(
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True

        def mean(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, LongType
            import pyspark.sql.functions as F
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
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None, min_count: int = 0
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def sum_func(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, LongType
            import pyspark.sql.functions as F
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
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None, min_count: int = 0
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def prod(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, IntegralType, LongType
            import pyspark.sql.functions as F
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
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True

        def skew(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, LongType
            import pyspark.sql.functions as F
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
            skew, name="skew", axis=axis, numeric_only=numeric_only
        )

    def kurtosis(
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True

        def kurtosis(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, LongType
            import pyspark.sql.functions as F
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
            kurtosis, name="kurtosis", axis=axis, numeric_only=numeric_only
        )

    kurt = kurtosis

    def min(
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        from pyspark.sql import functions as F
        return self._reduce_for_stat_function(
            F.min, name="min", axis=axis, numeric_only=numeric_only
        )

    def max(
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        from pyspark.sql import functions as F
        return self._reduce_for_stat_function(
            F.max, name="max", axis=axis, numeric_only=numeric_only
        )

    def count(
        self, axis: Union[int, str] = None, numeric_only: bool = False
    ) -> Union[Scalar, "Series"]:
        return self._reduce_for_stat_function(
            Frame._count_expr, name="count", axis=axis, numeric_only=numeric_only
        )

    def std(
        self, axis: Union[int, str] = None, ddof: int = 1, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True

        def std(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, LongType
            import pyspark.sql.functions as F
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
            std, name="std", axis=axis, numeric_only=numeric_only, ddof=ddof
        )

    def var(
        self, axis: Union[int, str] = None, ddof: int = 1, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True

        def var(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, LongType
            import pyspark.sql.functions as F
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
            var, name="var", axis=axis, numeric_only=numeric_only, ddof=ddof
        )

    def median(
        self, axis: Union[int, str] = None, numeric_only: Optional[bool] = None, accuracy: int = 10000
    ) -> Union[Scalar, "Series"]:
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True
        if not isinstance(accuracy, int):
            raise ValueError("accuracy must be an integer; however, got [%s]" % type(accuracy).__name__)

        def median(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, DoubleType
            import pyspark.sql.functions as F
            if isinstance(spark_type, (BooleanType, NumericType)):
                return SF.percentile_approx(spark_column.cast(DoubleType()), 0.5, accuracy)
            else:
                raise TypeError(
                    "Could not convert {} ({}) to numeric".format(
                        spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()
                    )
                )

        return self._reduce_for_stat_function(
            median, name="median", numeric_only=numeric_only, axis=axis
        )

    def sem(
        self, axis: Union[int, str] = None, ddof: int = 1, numeric_only: Optional[bool] = None
    ) -> Union[Scalar, "Series"]:
        assert ddof in (0, 1)
        axis = axis if axis is not None else 0
        if numeric_only is None and axis == 0:
            numeric_only = True

        def std(spark_column: Any, spark_type: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType, LongType
            import pyspark.sql.functions as F
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

        def sem(spark_column: Any, spark_type: Any) -> Any:
            import math
            return std(spark_column, spark_type) / math.sqrt(Frame._count_expr(spark_column, spark_type))

        return self._reduce_for_stat_function(
            sem, name="sem", numeric_only=numeric_only, axis=axis, ddof=ddof
        )

    @property
    def size(self) -> int:
        num_columns = len(self._internal.data_spark_columns)
        if num_columns == 0:
            return 0
        else:
            return len(self) * num_columns  # type: ignore

    def abs(self) -> Union["DataFrame", "Series"]:
        def abs_func(kser: Any) -> Any:
            from pyspark.sql.types import BooleanType, NumericType
            if isinstance(kser.spark.data_type, BooleanType):
                return kser
            elif isinstance(kser.spark.data_type, NumericType):
                import pyspark.sql.functions as F
                return kser.spark.transform(F.abs)
            else:
                raise TypeError(
                    "bad operand type for abs(): {} ({})".format(
                        spark_type_to_pandas_dtype(kser.spark.data_type),
                        kser.spark.data_type.simpleString(),
                    )
                )

        return self._apply_series_op(abs_func)

    def groupby(
        self, by: Any, axis: Union[int, str] = 0, as_index: bool = True, dropna: bool = True
    ) -> Union["DataFrameGroupBy", "SeriesGroupBy"]:
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
        from databricks.koalas.utils import is_name_like_tuple, is_name_like_value, is_list_like
        if isinstance(by, DataFrame):
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        elif isinstance(by, Series):
            by = [by]
        elif is_name_like_tuple(by):
            if isinstance(self, Series):
                raise KeyError(by)
            by = [by]
        elif is_name_like_value(by):
            if isinstance(self, Series):
                raise KeyError(by)
            by = [(by,)]
        elif is_list_like(by):
            new_by: List[Union[Tuple, Any]] = []
            for key in by:
                if isinstance(key, DataFrame):
                    raise ValueError("Grouper for '{}' not 1-dimensional".format(type(key).__name__))
                elif isinstance(key, Series):
                    new_by.append(key)
                elif is_name_like_tuple(key):
                    if isinstance(self, Series):
                        raise KeyError(key)
                    new_by.append(key)
                elif is_name_like_value(key):
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
        axis = axis if axis is not None else 0
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
        from databricks.koalas.utils import sql_conf, SPARK_CONF_ARROW_ENABLED
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
        from distutils.version import LooseVersion
        import pyspark
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

    def rolling(self, window: Union[int, str], min_periods: Optional[int] = None) -> Rolling:
        return Rolling(self, window=window, min_periods=min_periods)

    def expanding(self, min_periods: int = 1) -> Expanding:
        return Expanding(self, min_periods=min_periods)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def squeeze(self, axis: Optional[Union[int, str]] = None) -> Union[Scalar, "DataFrame", "Series"]:
        if axis is not None:
            axis = "index" if axis == "rows" else axis
            # Assuming validate_axis is defined somewhere
            from databricks.koalas.utils import validate_axis
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
        self, before: Any = None, after: Any = None, axis: Optional[Union[int, str]] = None, copy: bool = True
    ) -> Union["DataFrame", "Series"]:
        from databricks.koalas.series import first_series
        from databricks.koalas.utils import validate_axis
        axis = validate_axis(axis) if axis is not None else 0
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
                if indexes_increasing:
                    result = self.loc[before:after]
                else:
                    result = self.loc[after:before]
            elif axis == 1:
                result = self.loc[:, before:after]
        return result.copy() if copy else result

    def to_markdown(self, buf: Optional[Any] = None, mode: Optional[str] = None) -> str:
        from distutils.version import LooseVersion
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            raise NotImplementedError("`to_markdown()` only supported in Koalas with pandas >= 1.0.0")
        args = locals()
        kser_or_kdf = self
        internal_pandas = kser_or_kdf._to_internal_pandas()
        from databricks.koalas.utils import validate_arguments_and_invoke_function
        return validate_arguments_and_invoke_function(
            internal_pandas, self.to_markdown, type(internal_pandas).to_markdown, args
        )

    @abstractmethod
    def fillna(
        self,
        value: Any = None,
        method: Optional[str] = None,
        axis: Optional[Union[int, str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Union["DataFrame", "Series"]:
        pass

    def bfill(self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series"]:
        return self.fillna(method="bfill", axis=axis, inplace=inplace, limit=limit)

    backfill = bfill

    def ffill(self, axis: Optional[Union[int, str]] = None, inplace: bool = False, limit: Optional[int] = None) -> Union["DataFrame", "Series"]:
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
            "The truth value of a {0} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(self.__class__.__name__)
        )

    @staticmethod
    def _count_expr(spark_column: Any, spark_type: Any) -> Any:
        from pyspark.sql.types import FloatType, DoubleType
        import pyspark.sql.functions as F
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(spark_column, F.lit(None)))
        else:
            return F.count(spark_column)