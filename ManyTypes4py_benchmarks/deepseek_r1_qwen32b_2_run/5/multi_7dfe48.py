from distutils.version import LooseVersion
from functools import partial
from typing import Any, Optional, Tuple, Union, List, Sequence, Callable, Iterable, Iterator, Tuple, TypeVar, cast
import warnings
import pandas as pd
from pandas.api.types import is_list_like
from pandas.api.types import is_hashable
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F, Window
from databricks import koalas as ks
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeMultiIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.utils import compare_disallow_null, default_session, is_name_like_tuple, name_like_string, scol_for, verify_temp_column_name
from databricks.koalas.internal import InternalFrame, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.typedef import Scalar, ArrayLike

T = TypeVar('T')

class MultiIndex(Index):
    def __new__(cls, levels: Optional[Sequence[ArrayLike]] = None, codes: Optional[Sequence[ArrayLike]] = None, sortorder: Optional[int] = None, names: Optional[Sequence[Hashable]] = None, dtype: Optional[Any] = None, copy: bool = False, name: Optional[Hashable] = None, verify_integrity: bool = True) -> 'MultiIndex':
        ...

    @property
    def _internal(self) -> InternalFrame:
        ...

    @property
    def _column_label(self) -> None:
        ...

    def __abs__(self) -> None:
        ...

    def _with_new_scol(self, scol: spark.Column, dtype: Optional[Any] = None) -> None:
        ...

    def _align_and_column_op(self, f: Callable, *args: Any) -> None:
        ...

    def any(self, *args: Any, **kwargs: Any) -> None:
        ...

    def all(self, *args: Any, **kwargs: Any) -> None:
        ...

    @staticmethod
    def from_tuples(tuples: Sequence[TupleLike], sortorder: Optional[int] = None, names: Optional[Sequence[str]] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_arrays(arrays: Sequence[ArrayLike], sortorder: Optional[int] = None, names: Optional[Sequence[str]] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_product(iterables: Sequence[Iterable], sortorder: Optional[int] = None, names: Optional[Sequence[str]] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_frame(df: DataFrame, names: Optional[Sequence[str]] = None) -> 'MultiIndex':
        ...

    @property
    def name(self) -> None:
        ...

    @name.setter
    def name(self, name: Any) -> None:
        ...

    def _verify_for_rename(self, name: Any) -> Optional[List[Tuple[str]]]:
        ...

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1) -> 'MultiIndex':
        ...

    @property
    def levshape(self) -> Tuple[int, ...]:
        ...

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type: Any) -> Callable[[spark.Column, spark.Column, Any], spark.Column]:
        ...

    def _is_monotonic(self, order: str) -> bool:
        ...

    def _is_monotonic_increasing(self) -> Series:
        ...

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type: Any) -> Callable[[spark.Column, spark.Column, Any], spark.Column]:
        ...

    def _is_monotonic_decreasing(self) -> Series:
        ...

    def to_frame(self, index: bool = True, name: Optional[List[str]] = None) -> DataFrame:
        ...

    def to_pandas(self) -> pd.MultiIndex:
        ...

    def toPandas(self) -> pd.MultiIndex:
        ...

    def nunique(self, dropna: bool = True) -> None:
        ...

    def copy(self, deep: Optional[bool] = None) -> 'MultiIndex':
        ...

    def symmetric_difference(self, other: Union[Index, Iterable], result_name: Optional[List[str]] = None, sort: Optional[bool] = None) -> 'MultiIndex':
        ...

    def drop(self, codes: ArrayLike, level: Optional[Union[int, str]] = None) -> 'MultiIndex':
        ...

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, bins: Optional[int] = None, dropna: bool = True) -> Series:
        ...

    def argmax(self) -> None:
        ...

    def argmin(self) -> None:
        ...

    def asof(self, label: Any) -> None:
        ...

    @property
    def is_all_dates(self) -> bool:
        ...

    def __getattr__(self, item: str) -> Any:
        ...

    def _get_level_number(self, level: Union[int, str]) -> int:
        ...

    def get_level_values(self, level: Union[int, str]) -> Index:
        ...

    def insert(self, loc: int, item: TupleLike) -> 'MultiIndex':
        ...

    def item(self) -> Tuple[Any, ...]:
        ...

    def intersection(self, other: Union[Index, Iterable]) -> 'MultiIndex':
        ...

    @property
    def hasnans(self) -> None:
        ...

    @property
    def inferred_type(self) -> str:
        ...

    @property
    def asi8(self) -> None:
        ...

    def factorize(self, sort: bool = True, na_sentinel: int = -1) -> Tuple[Series, List[Any]]:
        ...

    def __iter__(self) -> Iterator[Tuple]:
        ...