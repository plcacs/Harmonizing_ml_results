import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, spmatrix
import pandas as pd
from pandas import MultiIndex, Series, date_range, DataFrame
from pandas.arrays import SparseArray
from typing import Any, ClassVar, Dict, List, Tuple, Union
from numpy.typing import NDArray, DTypeLike


def make_array(size: int, dense_proportion: float, fill_value: Any, dtype: DTypeLike) -> NDArray[Any]:
    dense_size = int(size * dense_proportion)
    arr: NDArray[Any] = np.full(size, fill_value, dtype)
    indexer: NDArray[np.int_] = np.random.choice(np.arange(size), dense_size, replace=False)
    arr[indexer] = np.random.choice(np.arange(100, dtype=dtype), dense_size)
    return arr


class SparseSeriesToFrame:
    series: Dict[int, Series]

    def setup(self) -> None:
        K = 50
        N = 50001
        rng = date_range('1/1/2000', periods=N, freq='min')
        self.series = {}
        for i in range(1, K):
            data: NDArray[np.float64] = np.random.randn(N)[:-i]
            idx = rng[:-i]
            data[100:] = np.nan
            self.series[i] = Series(SparseArray(data), index=idx)

    def time_series_to_frame(self) -> None:
        pd.DataFrame(self.series)


class SparseArrayConstructor:
    params: ClassVar[Tuple[List[float], List[Union[int, float]], List[DTypeLike]]] = (
        [0.1, 0.01],
        [0, np.nan],
        [np.int64, np.float64, object],
    )
    param_names: ClassVar[List[str]] = ['dense_proportion', 'fill_value', 'dtype']

    array: NDArray[Any]

    def setup(self, dense_proportion: float, fill_value: Any, dtype: DTypeLike) -> None:
        N = 10 ** 6
        self.array = make_array(N, dense_proportion, fill_value, dtype)

    def time_sparse_array(self, dense_proportion: float, fill_value: Any, dtype: DTypeLike) -> None:
        SparseArray(self.array, fill_value=fill_value, dtype=dtype)


class SparseDataFrameConstructor:
    sparse: spmatrix

    def setup(self) -> None:
        N = 1000
        self.sparse = scipy.sparse.rand(N, N, 0.005)

    def time_from_scipy(self) -> None:
        pd.DataFrame.sparse.from_spmatrix(self.sparse)


class FromCoo:
    matrix: coo_matrix

    def setup(self) -> None:
        self.matrix = scipy.sparse.coo_matrix(([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(100, 100))

    def time_sparse_series_from_coo(self) -> None:
        Series.sparse.from_coo(self.matrix)


class ToCoo:
    params: ClassVar[List[bool]] = [True, False]
    param_names: ClassVar[List[str]] = ['sort_labels']

    ss_mult_lvl: Series
    ss_two_lvl: Series

    def setup(self, sort_labels: bool) -> None:
        s: Series = Series([np.nan] * 10000)
        s.iloc[0] = 3.0
        s.iloc[100] = -1.0
        s.iloc[999] = 12.1
        s_mult_lvl: Series = s.set_axis(MultiIndex.from_product([range(10)] * 4))
        self.ss_mult_lvl = s_mult_lvl.astype('Sparse')
        s_two_lvl: Series = s.set_axis(MultiIndex.from_product([range(100)] * 2))
        self.ss_two_lvl = s_two_lvl.astype('Sparse')

    def time_sparse_series_to_coo(self, sort_labels: bool) -> None:
        self.ss_mult_lvl.sparse.to_coo(row_levels=[0, 1], column_levels=[2, 3], sort_labels=sort_labels)

    def time_sparse_series_to_coo_single_level(self, sort_labels: bool) -> None:
        self.ss_two_lvl.sparse.to_coo(sort_labels=sort_labels)


class ToCooFrame:
    df: DataFrame

    def setup(self) -> None:
        N = 10000
        k = 10
        arr: NDArray[np.float64] = np.zeros((N, k), dtype=float)
        arr[0, 0] = 3.0
        arr[12, 7] = -1.0
        arr[0, 9] = 11.2
        self.df = pd.DataFrame(arr, dtype=pd.SparseDtype('float', fill_value=0.0))

    def time_to_coo(self) -> None:
        self.df.sparse.to_coo()


class Arithmetic:
    params: ClassVar[Tuple[List[float], List[Union[int, float]]]] = ([0.1, 0.01], [0, np.nan])
    param_names: ClassVar[List[str]] = ['dense_proportion', 'fill_value']

    array1: SparseArray
    array2: SparseArray

    def setup(self, dense_proportion: float, fill_value: Union[int, float]) -> None:
        N = 10 ** 6
        arr1 = make_array(N, dense_proportion, fill_value, np.int64)
        self.array1 = SparseArray(arr1, fill_value=fill_value)
        arr2 = make_array(N, dense_proportion, fill_value, np.int64)
        self.array2 = SparseArray(arr2, fill_value=fill_value)

    def time_make_union(self, dense_proportion: float, fill_value: Union[int, float]) -> None:
        self.array1.sp_index.make_union(self.array2.sp_index)

    def time_intersect(self, dense_proportion: float, fill_value: Union[int, float]) -> None:
        self.array1.sp_index.intersect(self.array2.sp_index)

    def time_add(self, dense_proportion: float, fill_value: Union[int, float]) -> None:
        self.array1 + self.array2

    def time_divide(self, dense_proportion: float, fill_value: Union[int, float]) -> None:
        self.array1 / self.array2


class ArithmeticBlock:
    params: ClassVar[List[Union[int, float]]] = [np.nan, 0]
    param_names: ClassVar[List[str]] = ['fill_value']

    arr1: SparseArray
    arr2: SparseArray

    def setup(self, fill_value: Union[int, float]) -> None:
        N = 10 ** 6
        self.arr1 = self.make_block_array(length=N, num_blocks=1000, block_size=10, fill_value=fill_value)
        self.arr2 = self.make_block_array(length=N, num_blocks=1000, block_size=10, fill_value=fill_value)

    def make_block_array(
        self,
        length: int,
        num_blocks: int,
        block_size: int,
        fill_value: Union[int, float],
    ) -> SparseArray:
        arr: NDArray[Any] = np.full(length, fill_value)
        indices: NDArray[np.int_] = np.random.choice(np.arange(0, length, block_size), num_blocks, replace=False)
        for ind in indices:
            arr[ind:ind + block_size] = np.random.randint(0, 100, block_size)
        return SparseArray(arr, fill_value=fill_value)

    def time_make_union(self, fill_value: Union[int, float]) -> None:
        self.arr1.sp_index.make_union(self.arr2.sp_index)

    def time_intersect(self, fill_value: Union[int, float]) -> None:
        self.arr2.sp_index.intersect(self.arr2.sp_index)

    def time_addition(self, fill_value: Union[int, float]) -> None:
        self.arr1 + self.arr2

    def time_division(self, fill_value: Union[int, float]) -> None:
        self.arr1 / self.arr2


class MinMax:
    params: ClassVar[Tuple[List[str], List[float]]] = (['min', 'max'], [0.0, np.nan])
    param_names: ClassVar[List[str]] = ['func', 'fill_value']

    sp_arr: SparseArray

    def setup(self, func: str, fill_value: float) -> None:
        N = 1000000
        arr = make_array(N, 1e-05, fill_value, np.float64)
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_min_max(self, func: str, fill_value: float) -> None:
        getattr(self.sp_arr, func)()


class Take:
    params: ClassVar[Tuple[List[NDArray[Any]], List[bool]]] = (
        [np.array([0]), np.arange(100000), np.full(100000, -1)],
        [True, False],
    )
    param_names: ClassVar[List[str]] = ['indices', 'allow_fill']

    sp_arr: SparseArray

    def setup(self, indices: NDArray[Any], allow_fill: bool) -> None:
        N = 1000000
        fill_value = 0.0
        arr = make_array(N, 1e-05, fill_value, np.float64)
        self.sp_arr = SparseArray(arr, fill_value=fill_value)

    def time_take(self, indices: NDArray[Any], allow_fill: bool) -> None:
        self.sp_arr.take(indices, allow_fill=allow_fill)


class GetItem:
    sp_arr: SparseArray

    def setup(self) -> None:
        N = 1000000
        d = 1e-05
        arr = make_array(N, d, np.nan, np.float64)
        self.sp_arr = SparseArray(arr)

    def time_integer_indexing(self) -> None:
        self.sp_arr[78]

    def time_slice(self) -> None:
        self.sp_arr[1:]


class GetItemMask:
    params: ClassVar[List[Union[bool, float]]] = [True, False, np.nan]
    param_names: ClassVar[List[str]] = ['fill_value']

    sp_arr: SparseArray
    sp_b_arr: SparseArray

    def setup(self, fill_value: Union[bool, float]) -> None:
        N = 1000000
        d = 1e-05
        arr = make_array(N, d, np.nan, np.float64)
        self.sp_arr = SparseArray(arr)
        b_arr: NDArray[np.bool_] = np.full(shape=N, fill_value=fill_value, dtype=np.bool_)  # type: ignore[arg-type]
        fv_inds: NDArray[np.int32] = np.unique(
            np.random.randint(low=0, high=N - 1, size=int(N * d), dtype=np.int32)
        )
        b_arr[fv_inds] = True if pd.isna(fill_value) else not bool(fill_value)
        self.sp_b_arr = SparseArray(b_arr, dtype=np.bool_, fill_value=fill_value)

    def time_mask(self, fill_value: Union[bool, float]) -> None:
        self.sp_arr[self.sp_b_arr]


from .pandas_vb_common import setup