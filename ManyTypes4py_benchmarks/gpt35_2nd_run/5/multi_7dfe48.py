from distutils.version import LooseVersion
from typing import Any, Optional, Tuple, Union
import warnings
import pandas as pd
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeMultiIndex

class MultiIndex(Index):
    def __new__(cls, levels=None, codes=None, sortorder=None, names=None, dtype=None, copy=False, name=None, verify_integrity=True) -> 'MultiIndex':
        ...

    @property
    def _internal(self) -> Any:
        ...

    @property
    def _column_label(self) -> Any:
        ...

    def __abs__(self) -> None:
        ...

    def _with_new_scol(self, scol, *, dtype=None) -> None:
        ...

    def _align_and_column_op(self, f, *args) -> None:
        ...

    def any(self, *args, **kwargs) -> None:
        ...

    def all(self, *args, **kwargs) -> None:
        ...

    @staticmethod
    def from_tuples(tuples, sortorder=None, names=None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_arrays(arrays, sortorder=None, names=None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_product(iterables, sortorder=None, names=None) -> 'MultiIndex':
        ...

    @staticmethod
    def from_frame(df, names=None) -> 'MultiIndex':
        ...

    @property
    def name(self) -> None:
        ...

    @name.setter
    def name(self, name) -> None:
        ...

    def _verify_for_rename(self, name) -> Any:
        ...

    def swaplevel(self, i=-2, j=-1) -> 'MultiIndex':
        ...

    @property
    def levshape(self) -> Tuple[int, int]:
        ...

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type) -> Any:
        ...

    def _is_monotonic(self, order) -> bool:
        ...

    def _is_monotonic_increasing(self) -> Any:
        ...

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type) -> Any:
        ...

    def _is_monotonic_decreasing(self) -> Any:
        ...

    def to_frame(self, index=True, name=None) -> DataFrame:
        ...

    def to_pandas(self) -> pd.MultiIndex:
        ...

    def toPandas(self) -> pd.MultiIndex:
        ...

    def nunique(self, dropna=True) -> None:
        ...

    def copy(self, deep=None) -> 'MultiIndex':
        ...

    def symmetric_difference(self, other, result_name=None, sort=None) -> 'MultiIndex':
        ...

    def drop(self, codes, level=None) -> 'MultiIndex':
        ...

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True) -> Any:
        ...

    def argmax(self) -> None:
        ...

    def argmin(self) -> None:
        ...

    def asof(self, label) -> None:
        ...

    @property
    def is_all_dates(self) -> bool:
        ...

    def __getattr__(self, item) -> Any:
        ...

    def _get_level_number(self, level) -> Optional[int]:
        ...

    def get_level_values(self, level) -> 'Index':
        ...

    def insert(self, loc, item) -> 'MultiIndex':
        ...

    def item(self) -> Tuple:
        ...

    def intersection(self, other) -> 'MultiIndex':
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

    def factorize(self, sort=True, na_sentinel=-1) -> Tuple[Any, Any]:
        ...

    def __iter__(self) -> Any:
        ...
