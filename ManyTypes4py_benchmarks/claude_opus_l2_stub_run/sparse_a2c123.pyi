import numpy as np
import numpy.typing as npt
from typing import Any

from pandas import Series
from pandas.arrays import SparseArray


def make_array(
    size: int,
    dense_proportion: float,
    fill_value: Any,
    dtype: npt.DTypeLike,
) -> np.ndarray: ...


class SparseSeriesToFrame:
    series: dict[int, Series]

    def setup(self) -> None: ...
    def time_series_to_frame(self) -> None: ...


class SparseArrayConstructor:
    params: list[list[Any]]
    param_names: list[str]
    array: np.ndarray

    def setup(self, dense_proportion: float, fill_value: Any, dtype: npt.DTypeLike) -> None: ...
    def time_sparse_array(self, dense_proportion: float, fill_value: Any, dtype: npt.DTypeLike) -> None: ...


class SparseDataFrameConstructor:
    sparse: Any

    def setup(self) -> None: ...
    def time_from_scipy(self) -> None: ...


class FromCoo:
    matrix: Any

    def setup(self) -> None: ...
    def time_sparse_series_from_coo(self) -> None: ...


class ToCoo:
    params: list[bool]
    param_names: list[str]
    ss_mult_lvl: Series
    ss_two_lvl: Series

    def setup(self, sort_labels: bool) -> None: ...
    def time_sparse_series_to_coo(self, sort_labels: bool) -> None: ...
    def time_sparse_series_to_coo_single_level(self, sort_labels: bool) -> None: ...


class ToCooFrame:
    df: Any

    def setup(self) -> None: ...
    def time_to_coo(self) -> None: ...


class Arithmetic:
    params: list[list[Any]]
    param_names: list[str]
    array1: SparseArray
    array2: SparseArray

    def setup(self, dense_proportion: float, fill_value: Any) -> None: ...
    def time_make_union(self, dense_proportion: float, fill_value: Any) -> None: ...
    def time_intersect(self, dense_proportion: float, fill_value: Any) -> None: ...
    def time_add(self, dense_proportion: float, fill_value: Any) -> None: ...
    def time_divide(self, dense_proportion: float, fill_value: Any) -> None: ...


class ArithmeticBlock:
    params: list[Any]
    param_names: list[str]
    arr1: SparseArray
    arr2: SparseArray

    def setup(self, fill_value: Any) -> None: ...
    def make_block_array(
        self,
        length: int,
        num_blocks: int,
        block_size: int,
        fill_value: Any,
    ) -> SparseArray: ...
    def time_make_union(self, fill_value: Any) -> None: ...
    def time_intersect(self, fill_value: Any) -> None: ...
    def time_addition(self, fill_value: Any) -> None: ...
    def time_division(self, fill_value: Any) -> None: ...


class MinMax:
    params: list[list[Any]]
    param_names: list[str]
    sp_arr: SparseArray

    def setup(self, func: str, fill_value: float) -> None: ...
    def time_min_max(self, func: str, fill_value: float) -> None: ...


class Take:
    params: list[list[Any]]
    param_names: list[str]
    sp_arr: SparseArray

    def setup(self, indices: np.ndarray, allow_fill: bool) -> None: ...
    def time_take(self, indices: np.ndarray, allow_fill: bool) -> None: ...


class GetItem:
    sp_arr: SparseArray

    def setup(self) -> None: ...
    def time_integer_indexing(self) -> None: ...
    def time_slice(self) -> None: ...


class GetItemMask:
    params: list[Any]
    param_names: list[str]
    sp_arr: SparseArray
    sp_b_arr: SparseArray

    def setup(self, fill_value: Any) -> None: ...
    def time_mask(self, fill_value: Any) -> None: ...


def setup(*args: Any, **kwargs: Any) -> Any: ...