import numpy as np
import scipy.sparse
import pandas as pd
from pandas import MultiIndex, Series, date_range
from pandas.arrays import SparseArray

def make_array(size: int, dense_proportion: float, fill_value: np.generic, dtype: np.dtype) -> np.ndarray[Any, np.dtype[Any]]:
    ...

class SparseSeriesToFrame:
    def setup(self) -> None:
        ...

    def time_series_to_frame(self) -> None:
        ...

class SparseArrayConstructor:
    params: list[float | np.generic | type[np.generic]]
    param_names: list[str]

    def setup(self, dense_proportion: float, fill_value: np.generic, dtype: type[np.generic]) -> None:
        ...

    def time_sparse_array(self, dense_proportion: float, fill_value: np.generic, dtype: type[np.generic]) -> None:
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

    def time_to_coo(self) -> scipy.sparse.coo_matrix:
        ...

class Arithmetic:
    params: list[float | np.generic]
    param_names: list[str]

    def setup(self, dense_proportion: float, fill_value: np.generic) -> None:
        ...

    def time_make_union(self, dense_proportion: float, fill_value: np.generic) -> None:
        ...

    def time_intersect(self, dense_proportion: float, fill_value: np.generic) -> None:
        ...

    def time_add(self, dense_proportion: float, fill_value: np.generic) -> None:
        ...

    def time_divide(self, dense_proportion: float, fill_value: np.generic) -> None:
        ...

class ArithmeticBlock:
    params: list[np.generic]
    param_names: list[str]

    def setup(self, fill_value: np.generic) -> None:
        ...

    def make_block_array(self, length: int, num_blocks: int, block_size: int, fill_value: np.generic) -> SparseArray:
        ...

    def time_make_union(self, fill_value: np.generic) -> None:
        ...

    def time_intersect(self, fill_value: np.generic) -> None:
        ...

    def time_addition(self, fill_value: np.generic) -> None:
        ...

    def time_division(self, fill_value: np.generic) -> None:
        ...

class MinMax:
    params: list[str | np.generic]
    param_names: list[str]

    def setup(self, func: str, fill_value: np.generic) -> None:
        ...

    def time_min_max(self, func: str, fill_value: np.generic) -> float:
        ...

class Take:
    params: list[np.ndarray[Any, np.dtype[Any]] | bool]
    param_names: list[str]

    def setup(self, indices: np.ndarray[Any, np.dtype[Any]], allow_fill: bool) -> None:
        ...

    def time_take(self, indices: np.ndarray[Any, np.dtype[Any]], allow_fill: bool) -> None:
        ...

class GetItem:
    def setup(self) -> None:
        ...

    def time_integer_indexing(self) -> float:
        ...

    def time_slice(self) -> SparseArray:
        ...

class GetItemMask:
    params: list[bool | np.generic]
    param_names: list[str]

    def setup(self, fill_value: bool | np.generic) -> None:
        ...

    def time_mask(self, fill_value: bool | np.generic) -> float:
        ...