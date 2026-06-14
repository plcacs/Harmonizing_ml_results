from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from pyspark.sql import Column

from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.internal import InternalFrame
from databricks.koalas.series import Series


class MultiIndex(Index):
    def __new__(
        cls,
        levels: Any = ...,
        codes: Any = ...,
        sortorder: Optional[int] = ...,
        names: Optional[Any] = ...,
        dtype: Any = ...,
        copy: bool = ...,
        name: Any = ...,
        verify_integrity: bool = ...,
    ) -> MultiIndex: ...

    @property
    def _internal(self) -> InternalFrame: ...

    @property
    def _column_label(self) -> None: ...

    def __abs__(self) -> None: ...

    def _with_new_scol(self, scol: Column, *, dtype: Any = ...) -> None: ...

    def _align_and_column_op(self, f: Any, *args: Any) -> None: ...

    def any(self, *args: Any, **kwargs: Any) -> None: ...

    def all(self, *args: Any, **kwargs: Any) -> None: ...

    @staticmethod
    def from_tuples(
        tuples: Any,
        sortorder: Optional[int] = ...,
        names: Optional[Any] = ...,
    ) -> MultiIndex: ...

    @staticmethod
    def from_arrays(
        arrays: Any,
        sortorder: Optional[int] = ...,
        names: Optional[Any] = ...,
    ) -> MultiIndex: ...

    @staticmethod
    def from_product(
        iterables: Any,
        sortorder: Optional[int] = ...,
        names: Optional[Any] = ...,
    ) -> MultiIndex: ...

    @staticmethod
    def from_frame(
        df: DataFrame,
        names: Optional[Any] = ...,
    ) -> MultiIndex: ...

    @property
    def name(self) -> None: ...

    @name.setter
    def name(self, name: Any) -> None: ...

    def _verify_for_rename(self, name: Any) -> List[Tuple]: ...

    def swaplevel(self, i: Union[int, str] = ..., j: Union[int, str] = ...) -> MultiIndex: ...

    @property
    def levshape(self) -> Tuple[int, ...]: ...

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type: Any) -> Any: ...

    def _is_monotonic(self, order: str) -> bool: ...

    def _is_monotonic_increasing(self) -> Series: ...

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type: Any) -> Any: ...

    def _is_monotonic_decreasing(self) -> Series: ...

    def to_frame(self, index: bool = ..., name: Optional[Any] = ...) -> DataFrame: ...

    def to_pandas(self) -> pd.MultiIndex: ...

    def toPandas(self) -> pd.MultiIndex: ...

    def nunique(self, dropna: bool = ...) -> None: ...

    def copy(self, deep: Optional[Any] = ...) -> MultiIndex: ...

    def symmetric_difference(
        self,
        other: Any,
        result_name: Optional[Any] = ...,
        sort: Optional[bool] = ...,
    ) -> MultiIndex: ...

    def drop(self, codes: Any, level: Optional[Union[int, str]] = ...) -> MultiIndex: ...

    def value_counts(
        self,
        normalize: bool = ...,
        sort: bool = ...,
        ascending: bool = ...,
        bins: Optional[Any] = ...,
        dropna: bool = ...,
    ) -> Series: ...

    def argmax(self) -> None: ...

    def argmin(self) -> None: ...

    def asof(self, label: Any) -> None: ...

    @property
    def is_all_dates(self) -> bool: ...

    def __getattr__(self, item: str) -> Any: ...

    def _get_level_number(self, level: Union[int, str]) -> int: ...

    def get_level_values(self, level: Union[int, str]) -> Index: ...

    def insert(self, loc: int, item: Any) -> Index: ...

    def item(self) -> tuple: ...

    def intersection(self, other: Any) -> MultiIndex: ...

    @property
    def hasnans(self) -> None: ...

    @property
    def inferred_type(self) -> str: ...

    @property
    def asi8(self) -> None: ...

    def factorize(self, sort: bool = ..., na_sentinel: int = ...) -> Any: ...

    def __iter__(self) -> Any: ...