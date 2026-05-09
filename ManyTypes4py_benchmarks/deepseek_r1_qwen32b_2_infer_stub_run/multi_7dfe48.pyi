from typing import Any, Optional, Sequence, Tuple, Union, List, Callable, Dict, Iterable, TypeVar, Generic, overload
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import StructType
from databricks.koalas import DataFrame, Series, Index
from pandas import MultiIndex as PandasMultiIndex
import pandas as pd

T = TypeVar('T')

class MultiIndex(Index):
    """
    Koalas MultiIndex that corresponds to pandas MultiIndex logically.
    """
    def __new__(cls, levels: Optional[Sequence[Any]] = None, codes: Optional[Sequence[Any]] = None, 
                sortorder: Optional[int] = None, names: Optional[Sequence[Optional[str]]] = None, 
                dtype: Optional[Any] = None, copy: Optional[bool] = None, 
                name: Optional[Union[str, Tuple[str, ...]]] = None, 
                verify_integrity: Optional[bool] = None) -> 'MultiIndex':
        ...

    @property
    def _internal(self) -> 'InternalFrame':
        ...

    @property
    def _column_label(self) -> None:
        ...

    def __abs__(self) -> None:
        ...

    def _with_new_scol(self, scol: Any, dtype: Optional[Any] = None) -> None:
        ...

    def _align_and_column_op(self, f: Callable, *args: Any) -> None:
        ...

    def any(self, *args: Any, **kwargs: Any) -> None:
        ...

    def all(self, *args: Any, **kwargs: Any) -> None:
        ...

    @staticmethod
    def from_tuples(tuples: Sequence[Tuple[Any, ...]], sortorder: Optional[int] = None, 
                   names: Optional[Sequence[Optional[str]]] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_arrays(arrays: Sequence[Any], sortorder: Optional[int] = None, 
                   names: Optional[Sequence[Optional[str]]] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_product(iterables: Sequence[Iterable[Any]], sortorder: Optional[int] = None, 
                    names: Optional[Sequence[Optional[str]]] = None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_frame(df: DataFrame, names: Optional[Sequence[Optional[str]]] = None) -> 'MultiIndex':
        ...

    @property
    def name(self) -> None:
        ...

    @name.setter
    def name(self, name: Any) -> None:
        ...

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1) -> 'MultiIndex':
        ...

    @property
    def levshape(self) -> Tuple[int, ...]:
        ...

    def to_frame(self, index: bool = True, name: Optional[Sequence[Optional[str]]] = None) -> DataFrame:
        ...

    def to_pandas(self) -> pd.MultiIndex:
        ...

    def copy(self, deep: Optional[bool] = None) -> 'MultiIndex':
        ...

    def symmetric_difference(self, other: 'MultiIndex', result_name: Optional[Sequence[Optional[str]]] = None, 
                            sort: Optional[bool] = None) -> 'MultiIndex':
        ...

    def drop(self, codes: Any, level: Optional[Union[int, str]] = None) -> 'MultiIndex':
        ...

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False, 
                    bins: Optional[int] = None, dropna: bool = True) -> Series:
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

    def insert(self, loc: int, item: Any) -> 'MultiIndex':
        ...

    def item(self) -> Tuple[Any, ...]:
        ...

    def intersection(self, other: 'MultiIndex') -> 'MultiIndex':
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

    def __iter__(self) -> Iterable[Tuple[Any, ...]]:
        ...