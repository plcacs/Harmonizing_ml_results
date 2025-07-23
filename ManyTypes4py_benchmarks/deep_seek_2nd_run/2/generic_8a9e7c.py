from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING, cast, Dict, Callable, overload
import warnings
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, IntegralType, LongType, NumericType
from databricks import koalas as ks
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer
from databricks.koalas.internal import InternalFrame
from databricks.koalas.spark import functions as SF
from databricks.koalas.typedef import Scalar, spark_type_to_pandas_dtype
from databricks.koalas.utils import is_name_like_tuple, is_name_like_value, name_like_string, scol_for, sql_conf, validate_arguments_and_invoke_function, validate_axis, SPARK_CONF_ARROW_ENABLED
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
    def _apply_series_op(self, op: Callable, should_resolve: bool = False) -> Any:
        pass

    @abstractmethod
    def _reduce_for_stat_function(self, sfun: Callable, name: str, axis: Optional[int] = None, numeric_only: bool = True, **kwargs: Any) -> Any:
        pass

    @property
    @abstractmethod
    def dtypes(self) -> Any:
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def index(self) -> Any:
        pass

    @abstractmethod
    def copy(self) -> 'Frame':
        pass

    @abstractmethod
    def _to_internal_pandas(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def head(self, n: int = 5) -> 'Frame':
        pass

    def cummin(self, skipna: bool = True) -> 'Frame':
        return self._apply_series_op(lambda kser: kser._cum(F.min, skipna), should_resolve=True)

    def cummax(self, skipna: bool = True) -> 'Frame':
        return self._apply_series_op(lambda kser: kser._cum(F.max, skipna), should_resolve=True)

    def cumsum(self, skipna: bool = True) -> 'Frame':
        return self._apply_series_op(lambda kser: kser._cumsum(skipna), should_resolve=True)

    def cumprod(self, skipna: bool = True) -> 'Frame':
        return self._apply_series_op(lambda kser: kser._cumprod(skipna), should_resolve=True)

    def get_dtype_counts(self) -> pd.Series:
        warnings.warn('`get_dtype_counts` has been deprecated and will be removed in a future version. For DataFrames use `.dtypes.value_counts()', FutureWarning)
        if not isinstance(self.dtypes, Iterable):
            dtypes = [self.dtypes]
        else:
            dtypes = list(self.dtypes)
        return pd.Series(dict(Counter([d.name for d in dtypes])))

    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError('%s is both the pipe target and a keyword argument' % target)
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    def to_numpy(self) -> np.ndarray:
        return self.to_pandas().values

    @property
    def values(self) -> np.ndarray:
        warnings.warn('We recommend using `{}.to_numpy()` instead.'.format(type(self).__name__))
        return self.to_numpy()

    def to_csv(self, path: Optional[str] = None, sep: str = ',', na_rep: str = '', columns: Optional[List[str]] = None, header: Union[bool, List[str]] = True, quotechar: str = '"', date_format: Optional[str] = None, escapechar: Optional[str] = None, num_files: Optional[int] = None, mode: str = 'overwrite', partition_cols: Optional[List[str]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> Optional[str]:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        if path is None:
            kdf_or_ser = self
            if LooseVersion('0.24') > LooseVersion(pd.__version__) and isinstance(self, ks.Series):
                return kdf_or_ser.to_pandas().to_csv(None, sep=sep, na_rep=na_rep, header=header, date_format=date_format, index=False)
            else:
                return kdf_or_ser.to_pandas().to_csv(None, sep=sep, na_rep=na_rep, columns=columns, header=header, quotechar=quotechar, date_format=date_format, escapechar=escapechar, index=False)
        kdf = self
        if isinstance(self, ks.Series):
            kdf = self.to_frame()
        if columns is None:
            column_labels = kdf._internal.column_labels
        else:
            column_labels = []
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
            raise ValueError('to_csv only support one-level index column now')
        elif isinstance(header, list):
            sdf = kdf.to_spark(index_col)
            sdf = sdf.select([scol_for(sdf, name_like_string(label)) for label in index_cols] + [scol_for(sdf, str(i) if label is None else name_like_string(label)).alias(new_name) for i, (label, new_name) in enumerate(zip(column_labels, header))])
            header = True
        else:
            sdf = kdf.to_spark(index_col)
            sdf = sdf.select([scol_for(sdf, name_like_string(label)) for label in index_cols] + [scol_for(sdf, str(i) if label is None else name_like_string(label)) for i, label in enumerate(column_labels)])
        if num_files is not None:
            sdf = sdf.repartition(num_files)
        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(sep=sep, nullValue=na_rep, header=header, quote=quotechar, dateFormat=date_format, charToEscapeQuoteEscaping=escapechar)
        builder.options(**options).format('csv').save(path)
        return None

    def to_json(self, path: Optional[str] = None, compression: str = 'uncompressed', num_files: Optional[int] = None, mode: str = 'overwrite', orient: str = 'records', lines: bool = True, partition_cols: Optional[List[str]] = None, index_col: Optional[Union[str, List[str]]] = None, **options: Any) -> Optional[str]:
        if 'options' in options and isinstance(options.get('options'), dict) and (len(options) == 1):
            options = options.get('options')
        if not lines:
            raise NotImplementedError('lines=False is not implemented yet.')
        if orient != 'records':
            raise NotImplementedError("orient='records' is supported only for now.")
        if path is None:
            kdf_or_ser = self
            pdf = kdf_or_ser.to_pandas()
            if isinstance(self, ks.Series):
                pdf = pdf.to_frame()
            return pdf.to_json(orient='records')
        kdf = self
        if isinstance(self, ks.Series):
            kdf = self.to_frame()
        sdf = kdf.to_spark(index_col=index_col)
        if num_files is not None:
            sdf = sdf.repartition(num_files)
        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format('json').save(path)
        return None

    def to_excel(self, excel_writer: Any, sheet_name: str = 'Sheet1', na_rep: str = '', float_format: Optional[str] = None, columns: Optional[List[str]] = None, header: Union[bool, List[str]] = True, index: bool = True, index_label: Optional[Union[str, List[str]]] = None, startrow: int = 0, startcol: int = 0, engine: Optional[str] = None, merge_cells: bool = True, encoding: Optional[str] = None, inf_rep: str = 'inf', verbose: bool = True, freeze_panes: Optional[Tuple[int, int]] = None) -> None:
        args = locals()
        kdf_or_ser = self
        internal_pandas = kdf_or_ser._to_internal_pandas()
        return validate_arguments_and_invoke_function(internal_pandas, self.to_excel, type(internal_pandas).to_excel, args)

    def mean(self, axis: Optional[int] = None, numeric_only: Optional[bool] = None) -> Any:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def mean(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            return F.mean(spark_column)
        return self._reduce_for_stat_function(mean, name='mean', axis=axis, numeric_only=numeric_only)

    def sum(self, axis: Optional[int] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Any:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None

        def sum(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            return F.coalesce(F.sum(spark_column), F.lit(0))
        return self._reduce_for_stat_function(sum, name='sum', axis=axis, numeric_only=numeric_only, min_count=min_count)

    def product(self, axis: Optional[int] = None, numeric_only: Optional[bool] = None, min_count: int = 0) -> Any:
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
                sign = F.when(F.sum(F.when(spark_column < 0, 1).otherwise(0)) % 2 == 0, 1).otherwise(-1)
                scol = F.when(num_zeros > 0, 0).otherwise(sign * F.exp(F.sum(F.log(F.abs(spark_column)))))
                if isinstance(spark_type, IntegralType):
                    scol = F.round(scol).cast(LongType())
            else:
                raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            return F.coalesce(scol, F.lit(1))
        return self._reduce_for_stat_function(prod, name='prod', axis=axis, numeric_only=numeric_only, min_count=min_count)
    prod = product

    def skew(self, axis: Optional[int] = None, numeric_only: Optional[bool] = None) -> Any:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def skew(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            return F.skewness(spark_column)
        return self._reduce_for_stat_function(skew, name='skew', axis=axis, numeric_only=numeric_only)

    def kurtosis(self, axis: Optional[int] = None, numeric_only: Optional[bool] = None) -> Any:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def kurtosis(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            return F.kurtosis(spark_column)
        return self._reduce_for_stat_function(kurtosis, name='kurtosis', axis=axis, numeric_only=numeric_only)
    kurt = kurtosis

    def min(self, axis: Optional[int] = None, numeric_only: Optional[bool] = None) -> Any:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        return self._reduce_for_stat_function(F.min, name='min', axis=axis, numeric_only=numeric_only)

    def max(self, axis: Optional[int] = None, numeric_only: Optional[bool] = None) -> Any:
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True
        elif numeric_only is True and axis == 1:
            numeric_only = None
        return self._reduce_for_stat_function(F.max, name='max', axis=axis, numeric_only=numeric_only)

    def count(self, axis: Optional[int] = None, numeric_only: bool = False) -> Any:
        return self._reduce_for_stat_function(Frame._count_expr, name='count', axis=axis, numeric_only=numeric_only)

    def std(self, axis: Optional[int] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Any:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def std(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            if ddof == 0:
                return F.stddev_pop(spark_column)
            else:
                return F.stddev_samp(spark_column)
        return self._reduce_for_stat_function(std, name='std', axis=axis, numeric_only=numeric_only, ddof=ddof)

    def var(self, axis: Optional[int] = None, ddof: int = 1, numeric_only: Optional[bool] = None) -> Any:
        assert ddof in (0, 1)
        axis = validate_axis(axis)
        if numeric_only is None and axis == 0:
            numeric_only = True

        def var(spark_column: Any, spark_type: Any) -> Any:
            if isinstance(spark_type, BooleanType):
                spark_column = spark_column.cast(LongType())
            elif not isinstance(spark_type, NumericType):
                raise TypeError('Could not convert {} ({}) to numeric'.format(spark_type_to_pandas_dtype(spark_type), spark_type.simpleString()))
            if ddof == 0:
                return F.var_pop(spark_column)
            else:
                return F.var_samp(spark_column)
        return self._reduce_for_stat_function(var, name='var', axis=axis, numeric_only=numeric