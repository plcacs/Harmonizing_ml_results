from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
import datetime
import re
import inspect
import sys
import warnings
from collections.abc import Mapping as AbcMapping
from distutils.version import LooseVersion
from functools import partial, wraps, reduce
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar, Union, cast

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
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    StructType,
    IntegralType,
    ArrayType,
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
    InternalFrame,
    DEFAULT_SERIES_NAME,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_DEFAULT_INDEX_NAME,
    SPARK_DEFAULT_SERIES_NAME,
)
from databricks.koalas.missing.series import MissingPandasLikeSeries
from databricks.koalas.plot import KoalasPlotAccessor
from databricks.koalas.ml import corr
from databricks.koalas.utils import (
    combine_frames,
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    same_anchor,
    scol_for,
    sql_conf,
    validate_arguments_and_invoke_function,
    validate_axis,
    validate_bool_kwarg,
    verify_temp_column_name,
    SPARK_CONF_ARROW_ENABLED,
)
from databricks.koalas.datetimes import DatetimeMethods
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkSeriesMethods
from databricks.koalas.strings import StringMethods
from databricks.koalas.typedef import (
    infer_return_type,
    spark_type_to_pandas_dtype,
    ScalarType,
    Scalar,
    SeriesType,
)

T = TypeVar("T")
str_type = str

class Series(Frame, IndexOpsMixin, Generic[T]):
    def __init__(
        self,
        data: Optional[Any] = None,
        index: Optional[Any] = None,
        dtype: Optional[Any] = None,
        name: Optional[Any] = None,
        copy: bool = False,
        fastpath: bool = False,
    ) -> None:
        ...

    @property
    def _kdf(self) -> DataFrame:
        ...

    @property
    def _internal(self) -> InternalFrame:
        ...

    @property
    def _column_label(self) -> Tuple:
        ...

    def _update_anchor(self, kdf: DataFrame) -> None:
        ...

    def _with_new_scol(self, scol: spark.Column, *, dtype: Optional[Any] = None) -> "Series":
        ...

    spark = CachedAccessor("spark", SparkSeriesMethods)

    @property
    def dtypes(self) -> np.dtype:
        ...

    @property
    def axes(self) -> List:
        ...

    @property
    def spark_type(self):
        ...

    def add(self, other: Any) -> "Series":
        ...

    def radd(self, other: Any) -> "Series":
        ...

    def div(self, other: Any) -> "Series":
        ...

    def divide(self, other: Any) -> "Series":
        ...

    def rdiv(self, other: Any) -> "Series":
        ...

    def truediv(self, other: Any) -> "Series":
        ...

    def rtruediv(self, other: Any) -> "Series":
        ...

    def mul(self, other: Any) -> "Series":
        ...

    def multiply(self, other: Any) -> "Series":
        ...

    def rmul(self, other: Any) -> "Series":
        ...

    def sub(self, other: Any) -> "Series":
        ...

    def subtract(self, other: Any) -> "Series":
        ...

    def rsub(self, other: Any) -> "Series":
        ...

    def mod(self, other: Any) -> "Series":
        ...

    def rmod(self, other: Any) -> "Series":
        ...

    def pow(self, other: Any) -> "Series":
        ...

    def rpow(self, other: Any) -> "Series":
        ...

    def floordiv(self, other: Any) -> "Series":
        ...

    def rfloordiv(self, other: Any) -> "Series":
        ...

    koalas = CachedAccessor("koalas", KoalasSeriesMethods)

    def eq(self, other: Any) -> bool:
        ...

    equals = eq

    def gt(self, other: Any) -> "Series":
        ...

    def ge(self, other: Any) -> "Series":
        ...

    def lt(self, other: Any) -> "Series":
        ...

    def le(self, other: Any) -> "Series":
        ...

    def ne(self, other: Any) -> "Series":
        ...

    def divmod(self, other: Any) -> Tuple["Series", "Series"]:
        ...

    def rdivmod(self, other: Any) -> Tuple["Series", "Series"]:
        ...

    def between(self, left: Any, right: Any, inclusive: bool = True) -> "Series":
        ...

    def map(self, arg: Any) -> "Series":
        ...

    def alias(self, name: Any) -> "Series":
        ...

    @property
    def shape(self) -> Tuple[int]:
        ...

    @property
    def name(self) -> Union[Any, Tuple]:
        ...

    @name.setter
    def name(self, name: Union[Any, Tuple]) -> None:
        ...

    def rename(self, index: Optional[Any] = None, **kwargs: Any) -> "Series":
        ...

    def rename_axis(
        self, mapper: Optional[Any] = None, index: Optional[Any] = None, inplace: bool = False
    ) -> Optional["Series"]:
        ...

    @property
    def index(self) -> "ks.Index":
        ...

    @property
    def is_unique(self) -> bool:
        ...

    def reset_index(
        self, level: Optional[Any] = None, drop: bool = False, name: Optional[Any] = None, inplace: bool = False
    ) -> Optional[Union["Series", DataFrame]]:
        ...

    def to_frame(self, name: Union[Any, Tuple] = None) -> DataFrame:
        ...

    to_dataframe = to_frame

    def to_string(
        self,
        buf: Optional[Any] = None,
        na_rep: str = "NaN",
        float_format: Optional[Any] = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: Optional[int] = None,
    ) -> str:
        ...

    def to_clipboard(self, excel: bool = True, sep: Optional[str] = None, **kwargs: Any) -> None:
        ...

    def to_dict(self, into: type = dict) -> Mapping:
        ...

    def to_latex(
        self,
        buf: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        col_space: Optional[int] = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[List[Any]] = None,
        float_format: Optional[Any] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: Optional[str] = None,
        longtable: Optional[bool] = None,
        escape: Optional[bool] = None,
        encoding: Optional[str] = None,
        decimal: str = ".",
        multicolumn: Optional[bool] = None,
        multicolumn_format: Optional[str] = None,
        multirow: Optional[bool] = None,
    ) -> Optional[str]:
        ...

    def to_pandas(self) -> pd.Series:
        ...

    def toPandas(self) -> pd.Series:
        ...

    def to_list(self) -> List:
        ...

    tolist = to_list

    def drop_duplicates(self, keep: str = "first", inplace: bool = False) -> Optional["Series"]:
        ...

    def reindex(self, index: Optional[Any] = None, fill_value: Optional[Any] = None) -> "Series":
        ...

    def reindex_like(self, other: Union["Series", DataFrame]) -> "Series":
        ...

    def fillna(
        self,
        value: Optional[Any] = None,
        method: Optional[str] = None,
        axis: Optional[Any] = None,
        inplace: bool = False,
        limit: Optional[int] = None,
    ) -> Optional["Series"]:
        ...

    def _fillna(
        self,
        value: Optional[Any] = None,
        method: Optional[str] = None,
        axis: Optional[Any] = None,
        limit: Optional[int] = None,
        part_cols: Tuple = (),
    ) -> "Series":
        ...

    def dropna(self, axis: int = 0, inplace: bool = False, **kwargs: Any) -> Optional["Series"]:
        ...

    def clip(self, lower: Union[float, int] = None, upper: Union[float, int] = None) -> "Series":
        ...

    def drop(
        self, labels: Optional[Any] = None, index: Union[Any, Tuple, List[Any], List[Tuple]] = None, level: Optional[Any] = None
    ) -> "Series":
        ...

    def _drop(
        self,
        labels: Optional[Any] = None,
        index: Union[Any, Tuple, List[Any], List[Tuple]] = None,
        level: Optional[Any] = None,
    ) -> "Series":
        ...

    def head(self, n: int = 5) -> "Series":
        ...

    def last(self, offset: Union[str, DateOffset]) -> "Series":
        ...

    def first(self, offset: Union[str, DateOffset]) -> "Series":
        ...

    def unique(self) -> "Series":
        ...

    def sort_values(
        self, ascending: bool = True, inplace: bool = False, na_position: str = "last"
    ) -> Optional["Series"]:
        ...

    def sort_index(
        self,
        axis: int = 0,
        level: Optional[Union[int, List[int]]] = None,
        ascending: bool = True,
        inplace: bool = False,
        kind: Optional[str] = None,
        na_position: str = "last",
    ) -> Optional["Series"]:
        ...

    def swaplevel(self, i: int = -2, j: int = -1, copy: bool = True) -> "Series":
        ...

    def swapaxes(self, i: Union[str, int], j: Union[str, int], copy: bool = True) -> "Series":
        ...

    def add_prefix(self, prefix: str) -> "Series":
        ...

    def add_suffix(self, suffix: str) -> "Series":
        ...

    def corr(self, other: "Series", method: str = "pearson") -> float:
        ...

    def nsmallest(self, n: int = 5) -> "Series":
        ...

    def nlargest(self, n: int = 5) -> "Series":
        ...

    def append(
        self, to_append: "Series", ignore_index: bool = False, verify_integrity: bool = False
    ) -> "Series":
        ...

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> "Series":
        ...

    def hist(self, bins: int = 10, **kwds: Any) -> Any:
        ...

    def apply(self, func: Any, args: Tuple = (), **kwds: Any) -> "Series":
        ...

    def aggregate(self, func: Union[str, List[str]]) -> Union[Scalar, "Series"]:
        ...

    agg = aggregate

    def transpose(self, *args: Any, **kwargs: Any) -> "Series":
        ...

    T = property(transpose)

    def transform(self, func: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Union["Series", DataFrame]:
        ...

    def transform_batch(self, func: Any, *args: Any, **kwargs: Any) -> "ks.Series":
        ...

    def round(self, decimals: int = 0) -> "Series":
        ...

    def quantile(
        self, q: Union[float, Iterable[float]] = 0.5, accuracy: int = 10000
    ) -> Union[Scalar, "Series"]:
        ...

    def rank(self, method: str = "average", ascending: bool = True) -> "Series":
        ...

    def _rank(self, method: str = "average", ascending: bool = True, *, part_cols: Tuple = ()) -> "Series":
        ...

    def filter(self, items: Optional[Any] = None, like: Optional[str] = None, regex: Optional[str] = None, axis: Optional[Any] = None) -> "Series":
        ...

    def describe(self, percentiles: Optional[List[float]] = None) -> "Series":
        ...

    def diff(self, periods: int = 1) -> "Series":
        ...

    def _diff(self, periods: int, *, part_cols: Tuple = ()) -> "Series":
        ...

    def idxmax(self, skipna: bool = True) -> Union[Tuple, Any]:
        ...

    def idxmin(self, skipna: bool = True) -> Union[Tuple, Any]:
        ...

    def pop(self, item: Any) -> Union["Series", Scalar]:
        ...

    def copy(self, deep: Optional[bool] = None) -> "Series":
        ...

    def mode(self, dropna: bool = True) -> "Series":
        ...

    def keys(self) -> "ks.Index":
        ...

    def iteritems(self) -> Iterable[Tuple[Any, Any]]:
        ...

    def items(self) -> Iterable[Tuple[Any, Any]]:
        ...

    def droplevel(self, level: Any) -> "Series":
        ...

    def tail(self, n: int = 5) -> "Series":
        ...

    def explode(self) -> "Series":
        ...

    def argsort(self) -> "Series":
        ...

    def argmax(self) -> int:
        ...

    def argmin(self) -> int:
        ...

    def compare(
        self, other: "Series", keep_shape: bool = False, keep_equal: bool = False
    ) -> DataFrame:
        ...

    def update(self, other: "Series") -> None:
        ...

    def where(self, cond: Any, other: Any = np.nan) -> "Series":
        ...

    def mask(self, cond: Any, other: Any = np.nan) -> "Series":
        ...

    def xs(self, key: Any, level: Optional[Any] = None) -> "Series":
        ...

    def pct_change(self, periods: int = 1) -> "Series":
        ...

    def combine_first(self, other: "Series") -> "Series":
        ...

    def dot(self, other: Union["Series", DataFrame]) -> Union[Scalar, "Series"]:
        ...

    def __matmul__(self, other: Any) -> Any:
        ...

    def repeat(self, repeats: Union[int, "Series"]) -> "Series":
        ...

    def asof(self, where: Any) -> Union[Scalar, "Series"]:
        ...

    def mad(self) -> float:
        ...

    def unstack(self, level: int = -1) -> DataFrame:
        ...

    def item(self) -> Scalar:
        ...

    def align(
        self,
        other: Union[DataFrame, "Series"],
        join: str = "outer",
        axis: Optional[Union[int, str]] = None,
        copy: bool = True,
    ) -> Tuple["Series", Union[DataFrame, "Series"]]:
        ...

    def between_time(
        self,
        start_time: Union[datetime.time, str],
        end_time: Union[datetime.time, str],
        include_start: bool = True,
        include_end: bool = True,
        axis: Union[int, str] = 0,
    ) -> "Series":
        ...

    def at_time(
        self, time: Union[datetime.time, str], asof: bool = False, axis: Union[int, str] = 0
    ) -> "Series":
        ...

    def _cum(self, func: Any, skipna: bool, part_cols: Tuple = (), ascending: bool = True) -> "Series":
        ...

    def _cumsum(self, skipna: bool, part_cols: Tuple = ()) -> "Series":
        ...

    def _cumprod(self, skipna: bool, part_cols: Tuple = ()) -> "Series":
        ...

    dt = CachedAccessor("dt", DatetimeMethods)
    str = CachedAccessor("str", StringMethods)
    cat = CachedAccessor("cat", CategoricalAccessor)
    plot = C