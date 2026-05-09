from typing import Any, Optional, Tuple, Union, List, Dict, AnyStr, overload
import warnings
import pandas as pd
from pandas.api.types import is_list_like
import pyspark
from pyspark.sql import functions as F, Window
from databricks import koalas as ks
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeMultiIndex
from databricks.koalas.series import Series
from databricks.koalas.internal import InternalFrame
from databricks.koalas.typedef import Scalar

class MultiIndex(Index):
    """
    Koalas MultiIndex that corresponds to pandas MultiIndex logically.
    """
    def __new__(cls, levels: Optional[Any] = None, codes: Optional[Any] = None, sortorder: Optional[int] = None, names: Optional[Any] = None, dtype: Optional[Any] = None, copy: bool = False, name: Optional[Any] = None, verify_integrity: bool = True) -> 'MultiIndex':
        ...

    @property
    def _internal(self) -> InternalFrame:
        ...

    @property
    def _column_label(self) -> None:
        ...

    def __abs__(self) -> None:
        ...

    def _with_new_scol(self, scol: Any, dtype: Optional[Any] = None) -> None:
        ...

    def _align_and_column_op(self, f: Any, *args: Any) -> None:
        ...

    def any(self, *args: Any, **kwargs: Any) -> None:
        ...

    def all(self, *args: Any, **kwargs: Any) -> None:
        ...

    @staticmethod
    def from_tuples(tuples: Any, sortorder: Optional[int] = None, names: Optional[Any] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_arrays(arrays: Any, sortorder: Optional[int] = None, names: Optional[Any] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_product(iterables: Any, sortorder: Optional[int] = None, names: Optional[Any] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_frame(df: DataFrame, names: Optional[Any] = None) -> 'MultiIndex':
        ...

    @property
    def name(self) -> None:
        ...

    @name.setter
    def name(self, name: Any) -> None:
        ...

    def _verify_for_rename(self, name: Any) -> Any:
        ...

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1) -> 'MultiIndex':
        ...

    @property
    def levshape(self) -> Tuple[int, ...]:
        ...

    def _is_monotonic(self, order: str) -> None:
        ...

    def _is_monotonic_increasing(self) -> Series:
        ...

    def _is_monotonic_decreasing(self) -> Series:
        ...

    def to_frame(self, index: bool = True, name: Optional[Union[str, List[str]]] = None) -> DataFrame:
        ...

    def to_pandas(self) -> pd.MultiIndex:
        ...

    def toPandas(self) -> pd.MultiIndex:
        ...

    def nunique(self, dropna: bool = True) -> None:
        ...

    def copy(self, deep: Optional[Any] = None) -> 'MultiIndex':
        ...

    def symmetric_difference(self, other: Any, result_name: Optional[List[str]] = None, sort: Optional[bool] = None) -> 'MultiIndex':
        ...

    def drop(self, codes: Any, level: Optional[Union[int, str]] = None) -> 'MultiIndex':
        ...

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, bins: Optional[Any] = None, dropna: bool = True) -> Series:
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

    def insert(self, loc: int, item: Tuple[Any, ...]) -> 'MultiIndex':
        ...

    def item(self) -> Tuple[Any, ...]:
        ...

    def intersection(self, other: Any) -> 'MultiIndex':
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

    def factorize(self, sort: bool = True, na_sentinel: int = -1) -> Tuple[Any, Any]:
        ...

    def __iter__(self) -> Any:
        ...