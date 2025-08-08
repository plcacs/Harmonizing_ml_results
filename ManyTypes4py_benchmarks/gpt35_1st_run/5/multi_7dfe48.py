from distutils.version import LooseVersion
from typing import Any, Optional, Tuple, Union
import pandas as pd
import pyspark
from databricks import koalas as ks
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeMultiIndex
from databricks.koalas.series import Series
from databricks.koalas.utils import compare_disallow_null, default_session, is_name_like_tuple, name_like_string, scol_for, verify_temp_column_name
from databricks.koalas.internal import InternalFrame, NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT

class MultiIndex(Index):
    def __new__(cls, levels=None, codes=None, sortorder=None, names=None, dtype=None, copy=False, name=None, verify_integrity=True) -> 'MultiIndex':
        ...

    @property
    def _internal(self) -> InternalFrame:
        ...

    @property
    def _column_label(self) -> None:
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

    def _verify_for_rename(self, name) -> None:
        ...

    def swaplevel(self, i=-2, j=-1) -> 'MultiIndex':
        ...

    @property
    def levshape(self) -> Tuple[int, ...]:
        ...

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type) -> Any:
        ...

    def _is_monotonic(self, order) -> bool:
        ...

    def _is_monotonic_increasing(self) -> Series:
        ...

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type) -> Any:
        ...

    def _is_monotonic_decreasing(self) -> Series:
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

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True) -> Series:
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

    def get_level_values(self, level) -> Index:
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

    def factorize(self, sort=True, na_sentinel=-1) -> Tuple[Series, Series]:
        ...

    def __iter__(self) -> Any:
        ...
