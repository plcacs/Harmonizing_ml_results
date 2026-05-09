from __future__ import annotations
from collections import abc
import types
from typing import TYPE_CHECKING, Literal, cast, overload, Sequence, Mapping, Callable, Hashable, Iterable
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_bool, is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import factorize_from_iterable, factorize_from_iterables
import pandas.core.common as com
from pandas.core.indexes.api import Index, MultiIndex, all_indexes_same, default_index, ensure_index, get_objs_combined_axis, get_unanimous_names
from pandas.core.internals import concatenate_managers
if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping
    from pandas._typing import Axis, AxisInt, HashableT
    from pandas import DataFrame, Series

@overload
def concat(objs: Iterable[ABCSeries | ABCDataFrame], *, 
           axis: AxisInt = ..., 
           join: Literal['inner', 'outer'] = ..., 
           ignore_index: bool = ..., 
           keys: Sequence[Hashable] | None = ..., 
           levels: Sequence[Sequence[Hashable]] | None = ..., 
           names: Sequence[Hashable] | None = ..., 
           verify_integrity: bool = ..., 
           sort: bool = ..., 
           copy: bool = ...) -> ABCSeries | ABCDataFrame: ...

@overload
def concat(objs: Mapping[Hashable, ABCSeries | ABCDataFrame], *, 
           axis: AxisInt = ..., 
           join: Literal['inner', 'outer'] = ..., 
           ignore_index: bool = ..., 
           keys: Sequence[Hashable] | None = ..., 
           levels: Sequence[Sequence[Hashable]] | None = ..., 
           names: Sequence[Hashable] | None = ..., 
           verify_integrity: bool = ..., 
           sort: bool = ..., 
           copy: bool = ...) -> ABCSeries | ABCDataFrame: ...

@overload
def concat(objs: Iterable[ABCSeries | ABCDataFrame], *, 
           axis: AxisInt, 
           join: Literal['inner', 'outer'] = ..., 
           ignore_index: bool = ..., 
           keys: Sequence[Hashable] | None = ..., 
           levels: Sequence[Sequence[Hashable]] | None = ..., 
           names: Sequence[Hashable] | None = ..., 
           verify_integrity: bool = ..., 
           sort: bool = ..., 
           copy: bool = ...) -> ABCSeries | ABCDataFrame: ...

@set_module('pandas')
def concat(objs: Iterable[ABCSeries | ABCDataFrame] | Mapping[Hashable, ABCSeries | ABCDataFrame], *, 
           axis: AxisInt = 0, 
           join: Literal['inner', 'outer'] = 'outer', 
           ignore_index: bool = False, 
           keys: Sequence[Hashable] | None = None, 
           levels: Sequence[Sequence[Hashable]] | None = None, 
           names: Sequence[Hashable] | None = None, 
           verify_integrity: bool = False, 
           sort: bool = False, 
           copy: bool = lib.no_default) -> ABCSeries | ABCDataFrame:
    ...

def _sanitize_mixed_ndim(objs: Iterable[ABCSeries | ABCDataFrame], 
                          sample: ABCSeries | ABCDataFrame, 
                          ignore_index: bool, 
                          axis: AxisInt) -> list[ABCSeries | ABCDataFrame]:
    ...

def _get_result(objs: Iterable[ABCSeries | ABCDataFrame], 
                 is_series: bool, 
                 bm_axis: AxisInt, 
                 ignore_index: bool, 
                 intersect: bool, 
                 sort: bool, 
                 keys: Sequence[Hashable] | None, 
                 levels: Sequence[Sequence[Hashable]] | None, 
                 verify_integrity: bool, 
                 names: Sequence[Hashable] | None, 
                 axis: AxisInt) -> ABCSeries | ABCDataFrame:
    ...

def new_axes(objs: Iterable[ABCSeries | ABCDataFrame], 
              bm_axis: AxisInt, 
              intersect: bool, 
              sort: bool, 
              keys: Sequence[Hashable] | None, 
              names: Sequence[Hashable] | None, 
              axis: AxisInt, 
              levels: Sequence[Sequence[Hashable]] | None, 
              verify_integrity: bool, 
              ignore_index: bool) -> list[Index]:
    ...

def _get_concat_axis_series(objs: Iterable[ABCSeries], 
                             ignore_index: bool, 
                             bm_axis: AxisInt, 
                             keys: Sequence[Hashable] | None, 
                             levels: Sequence[Sequence[Hashable]] | None, 
                             verify_integrity: bool, 
                             names: Sequence[Hashable] | None) -> Index:
    ...

def _get_concat_axis_dataframe(objs: Iterable[ABCDataFrame], 
                                axis: AxisInt, 
                                ignore_index: bool, 
                                keys: Sequence[Hashable] | None, 
                                names: Sequence[Hashable] | None, 
                                levels: Sequence[Sequence[Hashable]] | None, 
                                verify_integrity: bool) -> Index:
    ...

def _clean_keys_and_objs(objs: Iterable[ABCSeries | ABCDataFrame] | Mapping[Hashable, ABCSeries | ABCDataFrame], 
                          keys: Sequence[Hashable] | None) -> tuple[list[ABCSeries | ABCDataFrame], Index | None, set[int]]:
    ...

def _get_sample_object(objs: Iterable[ABCSeries | ABCDataFrame], 
                        ndims: set[int], 
                        keys: Sequence[Hashable] | None, 
                        names: Sequence[Hashable] | None, 
                        levels: Sequence[Sequence[Hashable]] | None, 
                        intersect: bool) -> tuple[ABCSeries | ABCDataFrame, list[ABCSeries | ABCDataFrame]]:
    ...

def _concat_indexes(indexes: Iterable[Index]) -> Index:
    ...

def validate_unique_levels(levels: Sequence[Index]) -> None:
    ...

def _make_concat_multiindex(indexes: Iterable[Index], 
                             keys: Sequence[Hashable], 
                             levels: Sequence[Index] | None = None, 
                             names: Sequence[Hashable] | None = None) -> MultiIndex:
    ...
