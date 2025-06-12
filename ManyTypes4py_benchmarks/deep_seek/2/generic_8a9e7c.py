from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING, cast, Dict, Callable
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

    # ... (rest of the methods with similar type annotations)

    @abstractmethod
    def fillna(self, value: Any = None, method: Optional[str] = None, axis: Optional[int] = None, inplace: bool = False, limit: Optional[int] = None) -> Optional['Frame']:
        pass

    def bfill(self, axis: Optional[int] = None, inplace: bool = False, limit: Optional[int] = None) -> Optional['Frame']:
        return self.fillna(method='bfill', axis=axis, inplace=inplace, limit=limit)
    backfill = bfill

    def ffill(self, axis: Optional[int] = None, inplace: bool = False, limit: Optional[int] = None) -> Optional['Frame']:
        return self.fillna(method='ffill', axis=axis, inplace=inplace, limit=limit)
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
        raise ValueError('The truth value of a {0} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().'.format(self.__class__.__name__))

    @staticmethod
    def _count_expr(spark_column: Any, spark_type: Any) -> Any:
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(spark_column, F.lit(None)))
        else:
            return F.count(spark_column)
