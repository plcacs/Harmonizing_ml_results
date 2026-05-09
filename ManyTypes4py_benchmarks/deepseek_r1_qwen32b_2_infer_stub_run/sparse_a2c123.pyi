import numpy as np
import scipy.sparse
import pandas as pd
from pandas import MultiIndex, Series, date_range
from pandas.arrays import SparseArray
from __future__ import annotations

def make_array(size: int, dense_proportion: float, fill_value: Any, dtype: np.dtype) -> np.ndarray:
    ...

class SparseSeriesToFrame:
    def setup(self) -> None:
        ...
    
    def time_series_to_frame(self) -> None:
        ...

class SparseArrayConstructor:
    params: list[float | Any | np.dtype]
    param_names: list[str]
    
    def setup(self, dense_proportion: float, fill_value: Any, dtype: np.dtype) -> None:
        ...
    
    def time_sparse_array(self, dense_proportion: float, fill_value: Any, dtype: np.dtype) -> None:
        ...

class SparseDataFrameConstructor:
    def setup(self) -> None:
        ...
    
    def time_from_scipy(self) -> None:
        ...

class FromCoo:
    def setup(self) -> None:
        ...
    
    def time_sparse_series_from_coo(self) -> None:
        ...

class ToCoo:
    params: list[bool]
    param_names: list[str]
    
    def setup(self, sort_labels: bool) -> None:
        ...
    
    def time_sparse_series_to_coo(self, sort_labels: bool) -> None:
        ...
    
    def time_sparse_series_to_coo_single_level(self, sort_labels: bool) -> None:
        ...

class ToCooFrame:
    def setup(self) -> None:
        ...
    
    def time_to_coo(self) -> None:
        ...

class Arithmetic:
    params: list[float | Any]
    param_names: list[str]
    
    def setup(self, dense_proportion: float, fill_value: Any) -> None:
        ...
    
    def time_make_union(self, dense_proportion: float, fill_value: Any) -> None:
        ...
    
    def time_intersect(self, dense_proportion: float, fill_value: Any) -> None:
        ...
    
    def time_add(self, dense_proportion: float, fill_value: Any) -> None:
        ...
    
    def time_divide(self, dense_proportion: float, fill_value: Any) -> None:
        ...

class ArithmeticBlock:
    params: list[Any]
    param_names: list[str]
    
    def setup(self, fill_value: Any) -> None:
        ...
    
    def make_block_array(self, length: int, num_blocks: int, block_size: int, fill_value: Any) -> SparseArray:
        ...
    
    def time_make_union(self, fill_value: Any) -> None:
        ...
    
    def time_intersect(self, fill_value: Any) -> None:
        ...
    
    def time_addition(self, fill_value: Any) -> None:
        ...
    
    def time_division(self, fill_value: Any) -> None:
        ...

class MinMax:
    params: list[str | Any]
    param_names: list[str]
    
    def setup(self, func: str, fill_value: Any) -> None:
        ...
    
    def time_min_max(self, func: str, fill_value: Any) -> None:
        ...

class Take:
    params: list[np.ndarray | bool]
    param_names: list[str]
    
    def setup(self, indices: np.ndarray, allow_fill: bool) -> None:
        ...
    
    def time_take(self, indices: np.ndarray, allow_fill: bool) -> None:
        ...

class GetItem:
    def setup(self) -> None:
        ...
    
    def time_integer_indexing(self) -> None:
        ...
    
    def time_slice(self) -> None:
        ...

class GetItemMask:
    params: list[Any]
    param_names: list[str]
    
    def setup(self, fill_value: Any) -> None:
        ...
    
    def time_mask(self, fill_value: Any) -> None:
        ...