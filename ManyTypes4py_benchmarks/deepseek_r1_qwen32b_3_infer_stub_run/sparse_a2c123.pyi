import numpy as np
import scipy.sparse
import pandas as pd
from pandas import MultiIndex, Series, date_range
from pandas.arrays import SparseArray
from typing import Any, Dict, List, Optional, Tuple, Union

def make_array(size: int, dense_proportion: float, fill_value: Any, dtype: Any) -> np.ndarray[Any, Any]:
    ...

class SparseSeriesToFrame:
    def setup(self) -> None:
        ...

    def time_series_to_frame(self) -> pd.DataFrame:
        ...

class SparseArrayConstructor:
    params: List[Tuple[float, Any, Any]]
    param_names: List[str]

    def setup(self, dense_proportion: float, fill_value: Any, dtype: Any) -> None:
        ...

    def time_sparse_array(self, dense_proportion: float, fill_value: Any, dtype: Any) -> SparseArray:
        ...

class SparseDataFrameConstructor:
    def setup(self) -> None:
        ...

    def time_from_scipy(self) -> pd.DataFrame:
        ...

class FromCoo:
    def setup(self) -> None:
        ...

    def time_sparse_series_from_coo(self) -> Series:
        ...

class ToCoo:
    params: List[bool]
    param_names: List[str]

    def setup(self, sort_labels: bool) -> None:
        ...

    def time_sparse_series_to_coo(self, sort_labels: bool) -> scipy.sparse.coo_matrix:
        ...

    def time_sparse_series_to_coo_single_level(self, sort_labels: bool) -> scipy.sparse.coo_matrix:
        ...

class ToCooFrame:
    def setup(self) -> None:
        ...

    def time_to_coo(self) -> scipy.sparse.coo_matrix:
        ...

class Arithmetic:
    params: List[Tuple[float, Any]]
    param_names: List[str]

    def setup(self, dense_proportion: float, fill_value: Any) -> None:
        ...

    def time_make_union(self, dense_proportion: float, fill_value: Any) -> Any:
        ...

    def time_intersect(self, dense_proportion: float, fill_value: Any) -> Any:
        ...

    def time_add(self, dense_proportion: float, fill_value: Any) -> SparseArray:
        ...

    def time_divide(self, dense_proportion: float, fill_value: Any) -> SparseArray:
        ...

class ArithmeticBlock:
    params: List[Any]
    param_names: List[str]

    def setup(self, fill_value: Any) -> None:
        ...

    def make_block_array(self, length: int, num_blocks: int, block_size: int, fill_value: Any) -> SparseArray:
        ...

    def time_make_union(self, fill_value: Any) -> Any:
        ...

    def time_intersect(self, fill_value: Any) -> Any:
        ...

    def time_addition(self, fill_value: Any) -> SparseArray:
        ...

    def time_division(self, fill_value: Any) -> SparseArray:
        ...

class MinMax:
    params: List[Tuple[str, Any]]
    param_names: List[str]

    def setup(self, func: str, fill_value: Any) -> None:
        ...

    def time_min_max(self, func: str, fill_value: Any) -> Any:
        ...

class Take:
    params: List[Tuple[np.ndarray[Any, Any], bool]]
    param_names: List[str]

    def setup(self, indices: np.ndarray[Any, Any], allow_fill: bool) -> None:
        ...

    def time_take(self, indices: np.ndarray[Any, Any], allow_fill: bool) -> SparseArray:
        ...

class GetItem:
    def setup(self) -> None:
        ...

    def time_integer_indexing(self) -> Any:
        ...

    def time_slice(self) -> SparseArray:
        ...

class GetItemMask:
    params: List[Any]
    param_names: List[str]

    def setup(self, fill_value: Any) -> None:
        ...

    def time_mask(self, fill_value: Any) -> SparseArray:
        ...