from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, cast, overload
import warnings
import numpy as np
from pandas._config.config import get_option
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import find_common_type, maybe_promote
from pandas.core.dtypes.common import ensure_platform_int, is_1d_only_ea_dtype, is_integer, needs_i8_conversion
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import factorize, unique
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index, MultiIndex, default_index
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import compress_group_index, decons_obs_group_ids, get_compressed_ids, get_group_index, get_group_index_sorter
if TYPE_CHECKING:
    from pandas._typing import ArrayLike, Level, npt
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.frozen import FrozenList

class _Unstacker:
    def __init__(self, index: MultiIndex, level: int, constructor: object, sort: bool = True) -> None:
        ...

    @cache_readonly
    def _indexer_and_to_sort(self) -> tuple:
        ...

    @cache_readonly
    def sorted_labels(self) -> list:
        ...

    def _make_sorted_values(self, values) -> np.ndarray:
        ...

    def _make_selectors(self) -> None:
        ...

    @cache_readonly
    def mask_all(self) -> bool:
        ...

    @cache_readonly
    def arange_result(self) -> tuple:
        ...

    def get_result(self, obj, value_columns, fill_value) -> DataFrame:
        ...

    def get_new_values(self, values, fill_value=None) -> tuple:
        ...

    def get_new_columns(self, value_columns) -> MultiIndex:
        ...

    @cache_readonly
    def _repeater(self) -> np.ndarray:
        ...

    @cache_readonly
    def new_index(self) -> MultiIndex:
        ...

def _unstack_multiple(data, clocs, fill_value=None, sort=True) -> DataFrame:
    ...

@overload
def unstack(obj, level, fill_value=..., sort=...) -> DataFrame:
    ...

@overload
def unstack(obj, level, fill_value=..., sort=...) -> DataFrame:
    ...

def unstack(obj, level, fill_value=None, sort=True) -> DataFrame:
    ...

def _unstack_frame(obj, level, fill_value=None, sort=True) -> DataFrame:
    ...

def _unstack_extension_series(series, level, fill_value, sort) -> DataFrame:
    ...

def stack(frame, level=-1, dropna=True, sort=True) -> Union[Series, DataFrame]:
    ...

def stack_multiple(frame, level, dropna=True, sort=True) -> DataFrame:
    ...

def _stack_multi_column_index(columns) -> MultiIndex:
    ...

def _stack_multi_columns(frame, level_num=-1, dropna=True, sort=True) -> DataFrame:
    ...

def _reorder_for_extension_array_stack(arr, n_rows, n_columns) -> ExtensionArray:
    ...

def stack_v3(frame, level) -> Union[Series, DataFrame]:
    ...

def stack_reshape(frame, level, set_levels, stack_cols) -> DataFrame:
    ...
