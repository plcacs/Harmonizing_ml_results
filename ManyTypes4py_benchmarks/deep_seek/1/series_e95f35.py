from typing import (
    Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, Union, cast, Mapping,
    Sequence, Set, overload
)
import datetime
import re
import inspect
import sys
import warnings
from collections.abc import Mapping as AbcMapping
from distutils.version import LooseVersion
from functools import partial, wraps, reduce
import numpy as np
import pandas as pd
from pandas.core.accessor import CachedAccessor
from pandas.io.formats.printing import pprint_thing
from pandas.api.types import is_list_like, is_hashable
from pandas.api.extensions import ExtensionDtype
from pandas.tseries.frequencies import DateOffset
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F, Column
from pyspark.sql.types import (
    BooleanType, DoubleType, FloatType, IntegerType, LongType, NumericType, 
    StructType, IntegralType, ArrayType, StringType
)
from pyspark.sql.window import Window
from databricks import koalas as ks
from databricks.koalas.accessors import KoalasSeriesMethods
from databricks.koalas.categorical import CategoricalAccessor
from databricks.koalas.config import get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.exceptions import SparkPandasIndexingError
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import Frame
from databricks.koalas.internal import (
    InternalFrame, DEFAULT_SERIES_NAME, NATURAL_ORDER_COLUMN_NAME, 
    SPARK_DEFAULT_INDEX_NAME, SPARK_DEFAULT_SERIES_NAME
)
from databricks.koalas.missing.series import MissingPandasLikeSeries
from databricks.koalas.plot import KoalasPlotAccessor
from databricks.koalas.ml import corr
from databricks.koalas.utils import (
    combine_frames, is_name_like_tuple, is_name_like_value, name_like_string, 
    same_anchor, scol_for, sql_conf, validate_arguments_and_invoke_function, 
    validate_axis, validate_bool_kwarg, verify_temp_column_name, SPARK_CONF_ARROW_ENABLED
)
from databricks.koalas.datetimes import DatetimeMethods
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkSeriesMethods
from databricks.koalas.strings import StringMethods
from databricks.koalas.typedef import (
    infer_return_type, spark_type_to_pandas_dtype, ScalarType, Scalar, SeriesType
)

REPR_PATTERN = re.compile('Length: (?P<length>[0-9]+)')
_flex_doc_SERIES = '\nReturn {desc} of series and other, element-wise (binary operator `{op_name}`).\n\nEquivalent to ``{equiv}``\n\nParameters\n----------\nother : Series or scalar value\n\nReturns\n-------\nSeries\n    The result of the operation.\n\nSee Also\n--------\nSeries.{reverse}\n\n{series_examples}\n'

T = TypeVar('T')
str_type = str

def _create_type_for_series_type(param: Any) -> Any:
    from databricks.koalas.typedef import NameTypeHolder
    if isinstance(param, ExtensionDtype):
        new_class = type('NameType', (NameTypeHolder,), {})
        new_class.tpe = param
    else:
        new_class = param.type if isinstance(param, np.dtype) else param
    return SeriesType[new_class]

if (3, 5) <= sys.version_info < (3, 7):
    from typing import GenericMeta
    old_getitem = GenericMeta.__getitem__

    def new_getitem(self: GenericMeta, params: Any) -> Any:
        if hasattr(self, 'is_series'):
            return old_getitem(self, _create_type_for_series_type(params))
        else:
            return old_getitem(self, params)
    GenericMeta.__getitem__ = new_getitem

class Series(Frame, IndexOpsMixin, Generic[T]):
    def __init__(
        self, 
        data: Optional[Any] = None, 
        index: Optional[Any] = None, 
        dtype: Optional[Any] = None, 
        name: Optional[Any] = None, 
        copy: bool = False, 
        fastpath: bool = False
    ) -> None:
        assert data is not None
        if isinstance(data, DataFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath
            self._anchor = data
            self._col_label = index
        else:
            if isinstance(data, pd.Series):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert not fastpath
                s = data
            else:
                s = pd.Series(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
            internal = InternalFrame.from_pandas(pd.DataFrame(s))
            if s.name is None:
                internal = internal.copy(column_labels=[None])
            anchor = DataFrame(internal)
            self._anchor = anchor
            self._col_label = anchor._internal.column_labels[0]
            object.__setattr__(anchor, '_kseries', {self._column_label: self})

    @property
    def _kdf(self) -> DataFrame:
        return self._anchor

    @property
    def _internal(self) -> InternalFrame:
        return self._kdf._internal.select_column(self._column_label)

    @property
    def _column_label(self) -> Tuple[Any, ...]:
        return self._col_label

    def _update_anchor(self, kdf: DataFrame) -> None:
        assert kdf._internal.column_labels == [self._column_label], (kdf._internal.column_labels, [self._column_label])
        self._anchor = kdf
        object.__setattr__(kdf, '_kseries', {self._column_label: self})

    def _with_new_scol(self, scol: Column, *, dtype: Optional[Any] = None) -> 'Series':
        internal = self._internal.copy(
            data_spark_columns=[scol.alias(name_like_string(self._column_label))], 
            data_dtypes=[dtype]
        )
        return first_series(DataFrame(internal))

    spark = CachedAccessor('spark', SparkSeriesMethods)

    @property
    def dtypes(self) -> Any:
        return self.dtype

    @property
    def axes(self) -> List[Any]:
        return [self.index]

    @property
    def spark_type(self) -> Any:
        warnings.warn(
            'Series.spark_type is deprecated as of Series.spark.data_type. Please use the API instead.', 
            FutureWarning
        )
        return self.spark.data_type

    def add(self, other: Any) -> 'Series':
        return self + other

    def radd(self, other: Any) -> 'Series':
        return other + self

    def div(self, other: Any) -> 'Series':
        return self / other

    divide = div

    def rdiv(self, other: Any) -> 'Series':
        return other / self

    def truediv(self, other: Any) -> 'Series':
        return self / other

    def rtruediv(self, other: Any) -> 'Series':
        return other / self

    def mul(self, other: Any) -> 'Series':
        return self * other

    multiply = mul

    def rmul(self, other: Any) -> 'Series':
        return other * self

    def sub(self, other: Any) -> 'Series':
        return self - other

    subtract = sub

    def rsub(self, other: Any) -> 'Series':
        return other - self

    def mod(self, other: Any) -> 'Series':
        return self % other

    def rmod(self, other: Any) -> 'Series':
        return other % self

    def pow(self, other: Any) -> 'Series':
        return self ** other

    def rpow(self, other: Any) -> 'Series':
        return other ** self

    def floordiv(self, other: Any) -> 'Series':
        return self // other

    def rfloordiv(self, other: Any) -> 'Series':
        return other // self

    koalas = CachedAccessor('koalas', KoalasSeriesMethods)

    def eq(self, other: Any) -> 'Series':
        return self == other

    equals = eq

    def gt(self, other: Any) -> 'Series':
        return self > other

    def ge(self, other: Any) -> 'Series':
        return self >= other

    def lt(self, other: Any) -> 'Series':
        return self < other

    def le(self, other: Any) -> 'Series':
        return self <= other

    def ne(self, other: Any) -> 'Series':
        return self != other

    def divmod(self, other: Any) -> Tuple['Series', 'Series']:
        return (self.floordiv(other), self.mod(other))

    def rdivmod(self, other: Any) -> Tuple['Series', 'Series']:
        return (self.rfloordiv(other), self.rmod(other))

    def between(self, left: Any, right: Any, inclusive: bool = True) -> 'Series':
        if inclusive:
            lmask = self >= left
            rmask = self <= right
        else:
            lmask = self > left
            rmask = self < right
        return lmask & rmask

    def map(self, arg: Any) -> 'Series':
        if isinstance(arg, dict):
            is_start = True
            current = F.when(F.lit(False), F.lit(None).cast(self.spark.data_type))
            for to_replace, value in arg.items():
                if is_start:
                    current = F.when(self.spark.column == F.lit(to_replace), value)
                    is_start = False
                else:
                    current = current.when(self.spark.column == F.lit(to_replace), value)
            if hasattr(arg, '__missing__'):
                tmp_val = arg[np._NoValue]
                del arg[np._NoValue]
                current = current.otherwise(F.lit(tmp_val))
            else:
                current = current.otherwise(F.lit(None).cast(self.spark.data_type))
            return self._with_new_scol(current)
        else:
            return self.apply(arg)

    def alias(self, name: str) -> 'Series':
        warnings.warn(
            'Series.alias is deprecated as of Series.rename. Please use the API instead.', 
            FutureWarning
        )
        return self.rename(name)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self),)

    @property
    def name(self) -> Any:
        name = self._column_label
        if name is not None and len(name) == 1:
            return name[0]
        else:
            return name

    @name.setter
    def name(self, name: str) -> None:
        self.rename(name, inplace=True)

    def rename(self, index: Optional[Any] = None, **kwargs: Any) -> 'Series':
        if index is None:
            pass
        elif not is_hashable(index):
            raise TypeError('Series.name must be a hashable type')
        elif not isinstance(index, tuple):
            index = (index,)
        scol = self.spark.column.alias(name_like_string(index))
        internal = self._internal.copy(
            column_labels=[index], 
            data_spark_columns=[scol], 
            column_label_names=None
        )
        kdf = DataFrame(internal)
        if kwargs.get('inplace', False):
            self._col_label = index
            self._update_anchor(kdf)
            return self
        else:
            return first_series(kdf)

    def rename_axis(
        self, 
        mapper: Optional[Any] = None, 
        index: Optional[Any] = None, 
        inplace: bool = False
    ) -> Optional['Series']:
        kdf = self.to_frame().rename_axis(mapper=mapper, index=index, inplace=False)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    @property
    def index(self) -> Any:
        return self._kdf.index

    @property
    def is_unique(self) -> bool:
        scol = self.spark.column
        return self._internal.spark_frame.select(
            (F.count(scol) == F.countDistinct(scol)) & (F.count(F.when(scol.isNull(), 1).otherwise(None)) <= 1
        ).collect()[0][0]

    def reset_index(
        self, 
        level: Optional[Any] = None, 
        drop: bool = False, 
        name: Optional[Any] = None, 
        inplace: bool = False
    ) -> Optional['Series']:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if inplace and (not drop):
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')
        if drop:
            kdf = self._kdf[[self.name]]
        else:
            kser = self
            if name is not None:
                kser = kser.rename(name)
            kdf = kser.to_frame()
        kdf = kdf.reset_index(level=level, drop=drop)
        if drop:
            if inplace:
                self._update_anchor(kdf)
                return None
            else:
                return first_series(kdf)
        else:
            return kdf

    def to_frame(self, name: Optional[Any] = None) -> DataFrame:
        if name is not None:
            renamed = self.rename(name)
        elif self._column_label is None:
            renamed = self.rename(DEFAULT_SERIES_NAME)
        else:
            renamed = self
        return DataFrame(renamed._internal)

    to_dataframe = to_frame

    def to_string(
        self, 
        buf: Optional[Any] = None, 
        na_rep: str = 'NaN', 
        float_format: Optional[Any] = None, 
        header: bool = True, 
        index: bool = True, 
        length: bool = False, 
        dtype: bool = False, 
        name: bool = False, 
        max_rows: Optional[int] = None
    ) -> str:
        args = locals()
        if max_rows is not None:
            kseries = self.head(max_rows)
        else:
            kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), 
            self.to_string, 
            pd.Series.to_string, 
            args
        )

    def to_clipboard(self, excel: bool = True, sep: Optional[str] = None, **kwargs: Any) -> None:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), 
            self.to_clipboard, 
            pd.Series.to_clipboard, 
            args
        )

    def to_dict(self, into: type = dict) -> Dict[Any, Any]:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), 
            self.to_dict, 
            pd.Series.to_dict, 
            args
        )

    def to_latex(
        self, 
        buf: Optional[Any] = None, 
        columns: Optional[Any] = None, 
        col_space: Optional[Any] = None, 
        header: bool = True, 
        index: bool = True, 
        na_rep: str = 'NaN', 
        formatters: Optional[Any] = None, 
        float_format: Optional[Any] = None, 
        sparsify: Optional[Any] = None, 
        index_names: bool = True, 
        bold_rows: bool = False, 
        column_format: Optional[Any] = None, 
        longtable: Optional[Any] = None, 
        escape: Optional[Any] = None, 
        encoding: Optional[Any] = None, 
        decimal: str = '.', 
        multicolumn: Optional[Any] = None, 
        multicolumn_format: Optional[Any] = None, 
        multirow: Optional[Any] = None
    ) -> str:
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), 
            self.to_latex, 
            pd.Series.to_latex, 
            args
        )

    def to_pandas(self) -> pd.Series:
        return self._to_internal_pandas().copy()

    def toPandas(self) -> pd.Series:
        warnings.warn(
            'Series.toPandas is deprecated as of Series.to_pandas. Please use the API instead.', 
            FutureWarning
        )
        return self.to_pandas()

    def to_list(self) -> List[Any]:
        return self._to_internal_pandas().tolist()

    tolist = to_list

    def drop_duplicates(self, keep: str = 'first', inplace: bool = False) -> Optional['Series']:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kdf = self._kdf[[self.name]].drop_duplicates(keep=keep)
        if inplace:
            self._update_anchor(kdf)
            return None
        else:
            return first_series(kdf)

    def reindex(self, index: Optional[Any] = None, fill_value: Optional[Any] = None) -> 'Series':
        return first_series(
            self.to_frame().reindex(index=index, fill_value=fill_value)
        ).rename(self.name)

    def reindex_like(self, other: 'Series') -> 'Series':
        if isinstance(other, (Series, DataFrame)):
            return self.reindex(index=other.index)
        else:
