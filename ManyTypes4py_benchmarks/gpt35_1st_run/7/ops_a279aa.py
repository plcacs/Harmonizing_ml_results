from __future__ import annotations
import collections
import functools
from typing import TYPE_CHECKING, Generic
import numpy as np
from pandas._libs import NaT, lib
import pandas._libs.groupby as libgroupby
from pandas._typing import ArrayLike, AxisInt, NDFrameT, Shape, npt
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import maybe_cast_pointwise_result, maybe_downcast_to_dtype
from pandas.core.dtypes.common import ensure_float64, ensure_int64, ensure_platform_int, ensure_uint64, is_1d_only_ea_dtype
from pandas.core.dtypes.missing import isna, maybe_fill
from pandas.core.arrays import Categorical
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import CategoricalIndex, Index, MultiIndex, ensure_index
from pandas.core.series import Series
from pandas.core.sorting import compress_group_index, decons_obs_group_ids, get_group_index, get_group_index_sorter, get_indexer_dict

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable, Iterator
    from pandas.core.generic import NDFrame

def check_result_array(obj: np.ndarray, dtype: np.dtype) -> None:
    if isinstance(obj, np.ndarray):
        if dtype != object:
            raise ValueError('Must produce aggregated value')

def extract_result(res) -> np.ndarray:
    if hasattr(res, '_values'):
        res = res._values
        if res.ndim == 1 and len(res) == 1:
            res = res[0]
    return res

class WrappedCythonOp:
    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None:
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na

    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        if how in cls._CYTHON_FUNCTIONS['aggregate']:
            return 'aggregate'
        return 'transform'

    @classmethod
    @functools.cache
    def _get_cython_function(cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool) -> Callable:
        ...

    def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
        ...

    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Shape:
        ...

    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
        ...

    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        ...

    def _cython_op_ndim_compat(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask=None, result_mask=None, **kwargs) -> np.ndarray:
        ...

    def _call_cython_op(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask, result_mask, **kwargs) -> np.ndarray:
        ...

    def _validate_axis(self, axis: AxisInt, values: np.ndarray) -> None:
        ...

    def cython_operation(self, *, values: np.ndarray, axis: AxisInt, min_count: int = -1, comp_ids: np.ndarray, ngroups: int, **kwargs) -> np.ndarray:
        ...

class BaseGrouper:
    def __init__(self, axis: Index, groupings: Sequence[Grouping], sort: bool = True, dropna: bool = True) -> None:
        ...

    @property
    def groupings(self) -> Sequence[Grouping]:
        ...

    def __iter__(self) -> Iterator:
        ...

    @property
    def nkeys(self) -> int:
        ...

    def get_iterator(self, data: NDFrame) -> Generator:
        ...

    def indices(self) -> dict:
        ...

    def result_ilocs(self) -> np.ndarray:
        ...

    @property
    def codes(self) -> list:
        ...

    @property
    def levels(self) -> list:
        ...

    @property
    def names(self) -> list:
        ...

    def size(self) -> Series:
        ...

    @property
    def groups(self) -> dict:
        ...

    @property
    def is_monotonic(self) -> bool:
        ...

    @property
    def has_dropped_na(self) -> bool:
        ...

    @property
    def codes_info(self) -> np.ndarray:
        ...

    @property
    def ngroups(self) -> int:
        ...

    @property
    def result_index(self) -> Index:
        ...

    @property
    def ids(self) -> np.ndarray:
        ...

    @property
    def result_index_and_ids(self) -> tuple:
        ...

    @property
    def observed_grouper(self) -> BaseGrouper:
        ...

    def _ob_index_and_ids(self, levels: list, codes: list, names: list, sorts: list) -> tuple:
        ...

    def _unob_index_and_ids(self, levels: list, codes: list, names: list) -> tuple:
        ...

    def get_group_levels(self) -> Generator:
        ...

    def _cython_operation(self, kind: str, values: np.ndarray, how: str, axis: AxisInt, min_count: int = -1, **kwargs) -> np.ndarray:
        ...

    def agg_series(self, obj: Series, func: Callable, preserve_dtype: bool = False) -> np.ndarray:
        ...

    def _aggregate_series_pure_python(self, obj: Series, func: Callable) -> np.ndarray:
        ...

    def apply_groupwise(self, f: Callable, data: NDFrame) -> tuple:
        ...

    @property
    def _sorted_ids(self) -> np.ndarray:
        ...

class BinGrouper(BaseGrouper):
    def __init__(self, bins: ArrayLike, binlabels: Index, indexer: np.ndarray = None) -> None:
        ...

    @property
    def groups(self) -> dict:
        ...

    @property
    def nkeys(self) -> int:
        ...

    @property
    def codes_info(self) -> np.ndarray:
        ...

    def get_iterator(self, data: NDFrame) -> Generator:
        ...

    @property
    def indices(self) -> dict:
        ...

    @property
    def codes(self) -> list:
        ...

    @property
    def result_index_and_ids(self) -> tuple:
        ...

    @property
    def levels(self) -> list:
        ...

    @property
    def names(self) -> list:
        ...

    @property
    def groupings(self) -> list:
        ...

    @property
    def observed_grouper(self) -> BinGrouper:
        ...

class DataSplitter(Generic[NDFrameT]):
    def __init__(self, data: NDFrameT, ngroups: int, *, sort_idx: np.ndarray, sorted_ids: np.ndarray) -> None:
        ...

    def __iter__(self) -> Iterator:
        ...

    def _sorted_data(self) -> NDFrameT:
        ...

    def _chop(self, sdata: NDFrameT, slice_obj: slice) -> NDFrameT:
        ...

class SeriesSplitter(DataSplitter):
    def _chop(self, sdata: Series, slice_obj: slice) -> Series:
        ...

class FrameSplitter(DataSplitter):
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
        ...
