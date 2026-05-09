from __future__ import annotations
import numpy as np
import pandas as pd
from pandas._libs.reshape import libreshape
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index, MultiIndex
from typing import ArrayLike, Level, npt

class _Unstacker:
    def __init__(self, index: MultiIndex, level: int | str, constructor: type[DataFrame], sort: bool = True) -> None:
        ...
    @cache_readonly
    def _indexer_and_to_sort(self) -> tuple[np.ndarray, tuple[np.ndarray]]:
        ...
    @cache_readonly
    def sorted_labels(self) -> tuple[np.ndarray]:
        ...
    def _make_sorted_values(self, values: ArrayLike) -> np.ndarray:
        ...
    def _make_selectors(self) -> None:
        ...
    @cache_readonly
    def mask_all(self) -> bool:
        ...
    @cache_readonly
    def arange_result(self) -> tuple[np.ndarray, bool]:
        ...

def _unstack_multiple(data: DataFrame, clocs: list[int], fill_value: Any = None, sort: bool = True) -> DataFrame:
    ...

@overload
def unstack(obj: DataFrame, level: int | str, fill_value: Any = ..., sort: bool = ...) -> DataFrame:
    ...

def unstack(obj: DataFrame, level: int | str, fill_value: Any = None, sort: bool = True) -> DataFrame:
    ...

def stack(frame: DataFrame, level: int | str, dropna: bool = True, sort: bool = True) -> DataFrame:
    ...

def _stack_multi_columns(frame: DataFrame, level_num: int, dropna: bool = True, sort: bool = True) -> DataFrame:
    ...

def _reorder_for_extension_array_stack(arr: np.ndarray, n_rows: int, n_columns: int) -> np.ndarray:
    ...

def stack_v3(frame: DataFrame, level: list[int]) -> DataFrame:
    ...

def stack_reshape(frame: DataFrame, level: list[int], set_levels: set[int], stack_cols: Index) -> DataFrame:
    ...
