import numpy as np
import scipy.sparse
import pandas as pd
from pandas import MultiIndex, Series, date_range
from pandas.arrays import SparseArray
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy.typing as npt

def make_array(
    size: int,
    dense_proportion: float,
    fill_value: Union[int, float, object],
    dtype: npt.DTypeLike
) -> npt.NDArray[Any]: ...

class SparseSeriesToFrame:
    series: Dict[int, Series]

    def setup(self) -> None: ...
    def time_series_to_frame(self) -> pd.DataFrame: ...

class SparseArrayConstructor:
    params: List[Tuple[float, Union[int, float], npt.DTypeLike]]
    param_names: List[str]

    def setup(
        self,
        dense_proportion: float,
        fill_value: Union[int, float],
        dtype: npt.DTypeLike
    ) -> None: ...
    def time_sparse_array(
        self,
        dense_proportion: float,
        fill_value: Union[int, float],
        dtype: npt.DTypeLike
    ) -> SparseArray: ...

class SparseDataFrameConstructor:
    sparse: scipy.sparse.spmatrix

    def setup(self) -> None: ...
    def time_from_scipy(self) -> pd.DataFrame: ...

class FromCoo:
    matrix: scipy.sparse.coo_matrix

    def setup(self) -> None: ...
    def time_sparse_series_from_coo(self) -> Series: ...

class ToCoo:
    params: List[bool]
    param_names: List[str]
    ss_mult_lvl: Series
    ss_two_lvl: Series

    def setup(self, sort_labels: bool) -> None: ...
    def time_sparse_series_to_coo(self, sort_labels: bool) -> scipy.sparse.coo_matrix: ...
    def time_sparse_series_to_coo_single_level(self, sort_labels: bool) -> scipy.sparse.coo_matrix: ...

class ToCooFrame:
    df: pd.DataFrame

    def setup(self) -> None: ...
    def time_to_coo(self) -> scipy.sparse.coo_matrix: ...

class Arithmetic:
    params: List[Tuple[float, Union[int, float]]]
    param_names: List[str]
    array1: SparseArray
    array2: SparseArray

    def setup(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> None: ...
    def time_make_union(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> Any: ...
    def time_intersect(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> Any: ...
    def time_add(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> SparseArray: ...
    def time_divide(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> SparseArray: ...

class ArithmeticBlock:
    params: List[Union[float, int]]
    param_names: List[str]
    arr1: SparseArray
    arr2: SparseArray

    def setup(self, fill_value: Union[float, int]) -> None: ...
    def make_block_array(
        self,
        length: int,
        num_blocks: int,
        block_size: int,
        fill_value: Union[float, int]
    ) -> SparseArray: ...
    def time_make_union(self, fill_value: Union[float, int]) -> Any: ...
    def time_intersect(self, fill_value: Union[float, int]) -> Any: ...
    def time_addition(self, fill_value: Union[float, int]) -> SparseArray: ...
    def time_division(self, fill_value: Union[float, int]) -> SparseArray: ...

class MinMax:
    params: List[Tuple[str, float]]
    param_names: List[str]
    sp_arr: SparseArray

    def setup(self, func: str, fill_value: float) -> None: ...
    def time_min_max(self, func: str, fill_value: float) -> float: ...

class Take:
    params: List[Tuple[npt.NDArray[np.int64], bool]]
    param_names: List[str]
    sp_arr: SparseArray

    def setup(
        self,
        indices: npt.NDArray[np.int64],
        allow_fill: bool
    ) -> None: ...
    def time_take(
        self,
        indices: npt.NDArray[np.int64],
        allow_fill: bool
    ) -> SparseArray: ...

class GetItem:
    sp_arr: SparseArray

    def setup(self) -> None: ...
    def time_integer_indexing(self) -> float: ...
    def time_slice(self) -> SparseArray: ...

class GetItemMask:
    params: List[Union[bool, float]]
    param_names: List[str]
    sp_arr: SparseArray
    sp_b_arr: SparseArray

    def setup(self, fill_value: Union[bool, float]) -> None: ...
    def time_mask(self, fill_value: Union[bool, float]) -> SparseArray: ...