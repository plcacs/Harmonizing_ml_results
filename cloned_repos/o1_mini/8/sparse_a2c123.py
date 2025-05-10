import numpy as np
import scipy.sparse
import pandas as pd
from pandas import MultiIndex, Series, date_range
from pandas.arrays import SparseArray
from typing import Any, Dict, Union, List, Tuple
from pandas._typing import Dtype

def make_array(
    size: int,
    dense_proportion: float,
    fill_value: Union[int, float],
    dtype: Any
) -> np.ndarray:
    dense_size = int(size * dense_proportion)
    arr = np.full(size, fill_value, dtype=dtype)
    indexer = np.random.choice(np.arange(size), dense_size, replace=False)
    arr[indexer] = np.random.choice(np.arange(100, dtype=dtype), dense_size)
    return arr

class SparseSeriesToFrame:
    series: Dict[int, Series]

    def setup(self) -> None:
        K: int = 50
        N: int = 50001
        rng: pd.DatetimeIndex = date_range('1/1/2000', periods=N, freq='min')
        self.series = {}
        for i in range(1, K):
            data: np.ndarray = np.random.randn(N)[:-i]
            idx: pd.DatetimeIndex = rng[:-i]
            data[100:] = np.nan
            self.series[i] = Series(SparseArray(data), index=idx)

    def time_series_to_frame(self) -> None:
        pd.DataFrame(self.series)

class SparseArrayConstructor:
    params: Tuple[List[float], List[Union[int, float]], List[Any]] = (
        [0.1, 0.01],
        [0, np.nan],
        [np.int64, np.float64, object]
    )
    param_names: List[str] = ['dense_proportion', 'fill_value', 'dtype']
    array: np.ndarray

    def setup(
        self,
        dense_proportion: float,
        fill_value: Union[int, float],
        dtype: Any
    ) -> None:
        N: int = 10 ** 6
        self.array = make_array(N, dense_proportion, fill_value, dtype)

    def time_sparse_array(
        self,
        dense_proportion: float,
        fill_value: Union[int, float],
        dtype: Any
    ) -> None:
        SparseArray(self.array, fill_value=fill_value, dtype=dtype)

class SparseDataFrameConstructor:
    sparse: scipy.sparse.spmatrix

    def setup(self) -> None:
        N: int = 1000
        self.sparse = scipy.sparse.rand(N, N, 0.005)

    def time_from_scipy(self) -> None:
        pd.DataFrame.sparse.from_spmatrix(self.sparse)

class FromCoo:
    matrix: scipy.sparse.coo_matrix

    def setup(self) -> None:
        self.matrix = scipy.sparse.coo_matrix(
            (
                [3.0, 1.0, 2.0],
                (
                    [1, 0, 0],
                    [0, 2, 3]
                )
            ),
            shape=(100, 100)
        )

    def time_sparse_series_from_coo(self) -> None:
        Series.sparse.from_coo(self.matrix)

class ToCoo:
    params: List[bool] = [True, False]
    param_names: List[str] = ['sort_labels']
    ss_mult_lvl: Series
    ss_two_lvl: Series

    def setup(self, sort_labels: bool) -> None:
        s = Series([np.nan] * 10000)
        s[0] = 3.0
        s[100] = -1.0
        s[999] = 12.1
        s_mult_lvl = s.set_axis(MultiIndex.from_product([range(10)] * 4))
        self.ss_mult_lvl = s_mult_lvl.astype('Sparse')
        s_two_lvl = s.set_axis(MultiIndex.from_product([range(100)] * 2))
        self.ss_two_lvl = s_two_lvl.astype('Sparse')

    def time_sparse_series_to_coo(self, sort_labels: bool) -> None:
        self.ss_mult_lvl.sparse.to_coo(
            row_levels=[0, 1],
            column_levels=[2, 3],
            sort_labels=sort_labels
        )

    def time_sparse_series_to_coo_single_level(self, sort_labels: bool) -> None:
        self.ss_two_lvl.sparse.to_coo(sort_labels=sort_labels)

class ToCooFrame:
    df: pd.DataFrame

    def setup(self) -> None:
        N: int = 10000
        k: int = 10
        arr: np.ndarray = np.zeros((N, k), dtype=float)
        arr[0, 0] = 3.0
        arr[12, 7] = -1.0
        arr[0, 9] = 11.2
        self.df = pd.DataFrame(
            arr,
            dtype=pd.SparseDtype('float', fill_value=0.0)
        )

    def time_to_coo(self) -> None:
        self.df.sparse.to_coo()

class Arithmetic:
    params: Tuple[List[float], List[Union[int, float]]] = (
        [0.1, 0.01],
        [0, np.nan]
    )
    param_names: List[str] = ['dense_proportion', 'fill_value']
    array1: SparseArray
    array2: SparseArray

    def setup(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> None:
        N: int = 10 ** 6
        arr1: np.ndarray = make_array(N, dense_proportion, fill_value, np.int64)
        self.array1 = SparseArray(arr1, fill_value=fill_value)
        arr2: np.ndarray = make_array(N, dense_proportion, fill_value, np.int64)
        self.array2 = SparseArray(arr2, fill_value=fill_value)

    def time_make_union(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> None:
        self.array1.sp_index.make_union(self.array2.sp_index)

    def time_intersect(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> None:
        self.array1.sp_index.intersect(self.array2.sp_index)

    def time_add(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> None:
        self.array1 + self.array2

    def time_divide(
        self,
        dense_proportion: float,
        fill_value: Union[int, float]
    ) -> None:
        self.array1 / self.array2

class ArithmeticBlock:
    params: List[Union[float, int]] = [np.nan, 0]
    param_names: List[str] = ['fill_value']
    arr1: SparseArray
    arr2: SparseArray

    def setup(self, fill_value: Union[int, float, np.nan]) -> None:
        N: int = 10 ** 6
        self.arr1 = self.make_block_array(
            length=N,
            num_blocks=1000,
            block_size=10,
            fill_value=fill_value
        )
        self.arr2 = self.make_block_array(
            length=N,
            num_blocks=1000,
            block_size=10,
            fill_value=fill_value
        )

    def make_block_array(
        self,
        length: int,
        num_blocks: int,
        block_size: int,
        fill_value: Union[int, float, np.nan]
    ) -> SparseArray:
        arr: np.ndarray = np.full(length, fill_value)
        indices: np.ndarray = np.random.choice(
            np.arange(0, length, block_size),
            num_blocks,
            replace=False
        )
        for ind in indices:
            arr[ind:ind + block_size] = np.random.randint(0, 100, block_size)
        return SparseArray(arr, fill_value=fill_value)

    def time_make_union(
        self,
        fill_value: Union[int, float, np.nan]
    ) -> None:
        self.arr1.sp_index.make_union(self.arr2.sp_index)

    def time_intersect(
        self,
        fill_value: Union[int, float, np.nan]
    ) -> None:
        self.arr2.sp_index.intersect(self.arr2.sp_index)

    def time_addition(
        self,
        fill_value: Union[int, float, np.nan]
    ) -> None:
        self.arr1 + self.arr2

    def time_division(
        self,
        fill_value: Union[int, float, np.nan
    ) -> None:
        self.arr1 / self.arr2

class MinMax:
    params: Tuple[List[str], List[Union[float, np.nan]]] = (
        ['min', 'max'],
        [0.0, np.nan]
    )
    param_names: List[str] = ['func', 'fill_value']
    sp_arr: SparseArray

    def setup(
        self,
        func: str,
        fill_value: Union[float, np.nan]
    ) -> None:
        N: int = 1000000
        arr: np.ndarray = make_array(N, 1e-05, fill_value, np.float64)
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_min_max(
        self,
        func: str,
        fill_value: Union[float, np.nan]
    ) -> None:
        getattr(self.sp_arr, func)()

class Take:
    params: Tuple[List[np.ndarray], List[bool]] = (
        [np.array([0]), np.arange(100000), np.full(100000, -1)],
        [True, False]
    )
    param_names: List[str] = ['indices', 'allow_fill']
    sp_arr: SparseArray

    def setup(
        self,
        indices: np.ndarray,
        allow_fill: bool
    ) -> None:
        N: int = 1000000
        fill_value: float = 0.0
        arr: np.ndarray = make_array(N, 1e-05, fill_value, np.float64)
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_take(
        self,
        indices: np.ndarray,
        allow_fill: bool
    ) -> None:
        self.sp_arr.take(indices, allow_fill=allow_fill)

class GetItem:
    sp_arr: SparseArray

    def setup(self) -> None:
        N: int = 1000000
        d: float = 1e-05
        arr: np.ndarray = make_array(N, d, np.nan, np.float64)
        self.sp_arr = SparseArray(arr)

    def time_integer_indexing(self) -> None:
        self.sp_arr[78]

    def time_slice(self) -> None:
        self.sp_arr[1:]

class GetItemMask:
    params: List[Union[bool, float]] = [True, False, np.nan]
    param_names: List[str] = ['fill_value']
    sp_arr: SparseArray
    sp_b_arr: SparseArray

    def setup(
        self,
        fill_value: Union[bool, float, np.nan]
    ) -> None:
        N: int = 1000000
        d: float = 1e-05
        arr: np.ndarray = make_array(N, d, np.nan, np.float64)
        self.sp_arr = SparseArray(arr)
        b_arr: np.ndarray = np.full(shape=N, fill_value=fill_value, dtype=np.bool_)
        fv_inds: np.ndarray = np.unique(
            np.random.randint(low=0, high=N - 1, size=int(N * d), dtype=np.int32)
        )
        b_arr[fv_inds] = True if pd.isna(fill_value) else not fill_value
        self.sp_b_arr = SparseArray(
            b_arr,
            dtype=np.bool_,
            fill_value=fill_value
        )

    def time_mask(
        self,
        fill_value: Union[bool, float, np.nan]
    ) -> None:
        self.sp_arr[self.sp_b_arr]

from .pandas_vb_common import setup
