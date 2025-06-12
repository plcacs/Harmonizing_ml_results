from abc import ABCMeta, abstractmethod
import sys
import inspect
from collections import OrderedDict, namedtuple
from collections.abc import Callable
from distutils.version import LooseVersion
from functools import partial
from itertools import product
from typing import Any, List, Set, Tuple, Union, cast, Dict, Optional, Sequence
import pandas as pd
from pandas.api.types import is_hashable, is_list_like
from pyspark.sql import Window, functions as F
from pyspark.sql.types import FloatType, DoubleType, NumericType, StructField, StructType, StringType
from pyspark.sql.functions import PandasUDFType, pandas_udf, Column
from databricks import koalas as ks
from databricks.koalas.typedef import infer_return_type, DataFrameType, ScalarType, SeriesType
from databricks.koalas.frame import DataFrame
from databricks.koalas.internal import InternalFrame, HIDDEN_COLUMNS, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT, SPARK_DEFAULT_SERIES_NAME
from databricks.koalas.missing.groupby import MissingPandasLikeDataFrameGroupBy, MissingPandasLikeSeriesGroupBy
from databricks.koalas.series import Series, first_series
from databricks.koalas.config import get_option
from databricks.koalas.utils import align_diff_frames, is_name_like_tuple, is_name_like_value, name_like_string, same_anchor, scol_for, verify_temp_column_name
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.window import RollingGroupby, ExpandingGroupby
from databricks.koalas.exceptions import DataError
from databricks.koalas.spark import functions as SF
NamedAgg = namedtuple('NamedAgg', ['column', 'aggfunc'])

class GroupBy(object, metaclass=ABCMeta):
    def __init__(self, kdf: DataFrame, groupkeys: List[Series], as_index: bool, dropna: bool, column_labels_to_exlcude: Set[Tuple[str, ...]], agg_columns_selected: bool, agg_columns: List[Series]) -> None:
        self._kdf = kdf
        self._groupkeys = groupkeys
        self._as_index = as_index
        self._dropna = dropna
        self._column_labels_to_exlcude = column_labels_to_exlcude
        self._agg_columns_selected = agg_columns_selected
        self._agg_columns = agg_columns

    @property
    def _groupkeys_scols(self) -> List[Column]:
        return [s.spark.column for s in self._groupkeys]

    @property
    def _agg_columns_scols(self) -> List[Column]:
        return [s.spark.column for s in self._agg_columns]

    @abstractmethod
    def _apply_series_op(self, op: Callable, should_resolve: bool = False, numeric_only: bool = False) -> Any:
        pass

    def aggregate(self, func_or_funcs: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None, *args: Any, **kwargs: Any) -> DataFrame:
        pass

    agg = aggregate

    @staticmethod
    def _spark_groupby(kdf: DataFrame, func: Dict[str, Union[str, List[str]]], groupkeys: Sequence[Series] = ()) -> InternalFrame:
        pass

    def count(self) -> DataFrame:
        pass

    def first(self) -> DataFrame:
        pass

    def last(self) -> DataFrame:
        pass

    def max(self) -> DataFrame:
        pass

    def mean(self) -> DataFrame:
        pass

    def min(self) -> DataFrame:
        pass

    def std(self, ddof: int = 1) -> DataFrame:
        pass

    def sum(self) -> DataFrame:
        pass

    def var(self, ddof: int = 1) -> DataFrame:
        pass

    def all(self) -> DataFrame:
        pass

    def any(self) -> DataFrame:
        pass

    def size(self) -> Series:
        pass

    def diff(self, periods: int = 1) -> Union[DataFrame, Series]:
        pass

    def cumcount(self, ascending: bool = True) -> Series:
        pass

    def cummax(self) -> Union[DataFrame, Series]:
        pass

    def cummin(self) -> Union[DataFrame, Series]:
        pass

    def cumprod(self) -> Union[DataFrame, Series]:
        pass

    def cumsum(self) -> Union[DataFrame, Series]:
        pass

    def apply(self, func: Callable, *args: Any, **kwargs: Any) -> Union[DataFrame, Series]:
        pass

    def filter(self, func: Callable) -> Union[DataFrame, Series]:
        pass

    @staticmethod
    def _prepare_group_map_apply(kdf: DataFrame, groupkeys: List[Series], agg_columns: List[Series]) -> Tuple[DataFrame, List[Tuple[str, ...]], List[str]]:
        pass

    @staticmethod
    def _spark_group_map_apply(kdf: DataFrame, func: Callable, groupkeys_scols: List[Column], return_schema: StructType, retain_index: bool) -> Any:
        pass

    @staticmethod
    def _make_pandas_df_builder_func(kdf: DataFrame, func: Callable, return_schema: StructType, retain_index: bool) -> Callable:
        pass

    def rank(self, method: str = 'average', ascending: bool = True) -> Union[DataFrame, Series]:
        pass

    def idxmax(self, skipna: bool = True) -> Union[DataFrame, Series]:
        pass

    def idxmin(self, skipna: bool = True) -> Union[DataFrame, Series]:
        pass

    def fillna(self, value: Optional[Any] = None, method: Optional[str] = None, axis: Optional[int] = None, inplace: bool = False, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        pass

    def bfill(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        pass

    backfill = bfill

    def ffill(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        pass

    pad = ffill

    def _limit(self, n: int, asc: bool) -> Union[DataFrame, Series]:
        pass

    def head(self, n: int = 5) -> Union[DataFrame, Series]:
        pass

    def tail(self, n: int = 5) -> Union[DataFrame, Series]:
        pass

    def shift(self, periods: int = 1, fill_value: Optional[Any] = None) -> Union[DataFrame, Series]:
        pass

    def transform(self, func: Callable, *args: Any, **kwargs: Any) -> Union[DataFrame, Series]:
        pass

    def nunique(self, dropna: bool = True) -> Union[DataFrame, Series]:
        pass

    def rolling(self, window: int, min_periods: Optional[int] = None) -> RollingGroupby:
        pass

    def expanding(self, min_periods: int = 1) -> ExpandingGroupby:
        pass

    def get_group(self, name: Any) -> Union[DataFrame, Series]:
        pass

    def median(self, numeric_only: bool = True, accuracy: int = 10000) -> Union[DataFrame, Series]:
        pass

    def _reduce_for_stat_function(self, sfun: Callable, only_numeric: bool) -> Union[DataFrame, Series]:
        pass

    @staticmethod
    def _resolve_grouping_from_diff_dataframes(kdf: DataFrame, by: List[Any]) -> Tuple[DataFrame, List[Series], Set[Tuple[str, ...]]]:
        pass

    @staticmethod
    def _resolve_grouping(kdf: DataFrame, by: List[Any]) -> List[Series]:
        pass

class DataFrameGroupBy(GroupBy):
    @staticmethod
    def _build(kdf: DataFrame, by: List[Any], as_index: bool, dropna: bool) -> 'DataFrameGroupBy':
        pass

    def __init__(self, kdf: DataFrame, by: List[Series], as_index: bool, dropna: bool, column_labels_to_exlcude: Set[Tuple[str, ...]], agg_columns: Optional[List[Tuple[str, ...]]] = None) -> None:
        pass

    def __getattr__(self, item: str) -> Any:
        pass

    def __getitem__(self, item: Any) -> Union['SeriesGroupBy', 'DataFrameGroupBy']:
        pass

    def _apply_series_op(self, op: Callable, should_resolve: bool = False, numeric_only: bool = False) -> DataFrame:
        pass

    def describe(self) -> DataFrame:
        pass

class SeriesGroupBy(GroupBy):
    @staticmethod
    def _build(kser: Series, by: List[Any], as_index: bool, dropna: bool) -> 'SeriesGroupBy':
        pass

    def __init__(self, kser: Series, by: List[Series], as_index: bool = True, dropna: bool = True) -> None:
        pass

    def __getattr__(self, item: str) -> Any:
        pass

    def _apply_series_op(self, op: Callable, should_resolve: bool = False, numeric_only: bool = False) -> Series:
        pass

    def _reduce_for_stat_function(self, sfun: Callable, only_numeric: bool) -> Series:
        pass

    def agg(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def aggregate(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def transform(self, func: Callable, *args: Any, **kwargs: Any) -> Series:
        pass

    def idxmin(self, skipna: bool = True) -> Series:
        pass

    def idxmax(self, skipna: bool = True) -> Series:
        pass

    def head(self, n: int = 5) -> Series:
        pass

    def tail(self, n: int = 5) -> Series:
        pass

    def size(self) -> Series:
        pass

    def get_group(self, name: Any) -> Series:
        pass

    def nsmallest(self, n: int = 5) -> Series:
        pass

    def nlargest(self, n: int = 5) -> Series:
        pass

    def value_counts(self, sort: Optional[bool] = None, ascending: Optional[bool] = None, dropna: bool = True) -> Series:
        pass

    def unique(self) -> Series:
        pass

def is_multi_agg_with_relabel(**kwargs: Any) -> bool:
    pass

def normalize_keyword_aggregation(kwargs: Dict[str, Any]) -> Tuple[OrderedDict, Tuple[str, ...], List[Tuple[str, str]]]:
    pass
