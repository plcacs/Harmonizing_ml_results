from __future__ import annotations
from collections import abc
import types
from typing import TYPE_CHECKING, Literal, cast, overload, List, Tuple, Union
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
def concat(objs: Union[List[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]]], *, axis: AxisInt = ..., join: Literal['inner', 'outer'] = ..., ignore_index: bool = ..., keys: Union[None, List[Hashable]] = ..., levels: Union[None, List[Iterable[Hashable]]] = ..., names: Union[None, List[Hashable]] = ..., verify_integrity: bool = ..., sort: bool = ..., copy: Union[bool, lib.no_default] = ...):
    ...

@set_module('pandas')
def concat(objs: Union[List[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]]], *, axis: AxisInt = 0, join: Literal['inner', 'outer'] = 'outer', ignore_index: bool = False, keys: Union[None, List[Hashable]] = None, levels: Union[None, List[Iterable[Hashable]]] = None, names: Union[None, List[Hashable]] = None, verify_integrity: bool = False, sort: bool = False, copy: Union[bool, lib.no_default] = lib.no_default) -> Union[Series, DataFrame]:
    ...

def _sanitize_mixed_ndim(objs: List[Union[Series, DataFrame]], sample: Union[Series, DataFrame], ignore_index: bool, axis: int) -> List[Union[Series, DataFrame]]:
    ...

def _get_result(objs: List[Union[Series, DataFrame]], is_series: bool, bm_axis: int, ignore_index: bool, intersect: bool, sort: bool, keys: Union[None, List[Hashable]], levels: Union[None, List[Iterable[Hashable]]], verify_integrity: bool, names: Union[None, List[Hashable]], axis: int) -> Union[Series, DataFrame]:
    ...

def new_axes(objs: List[Union[Series, DataFrame]], bm_axis: int, intersect: bool, sort: bool, keys: Union[None, List[Hashable]], names: Union[None, List[Hashable]], axis: int, levels: Union[None, List[Iterable[Hashable]]], verify_integrity: bool, ignore_index: bool) -> List[Index]:
    ...

def _get_concat_axis_series(objs: List[Union[Series, DataFrame]], ignore_index: bool, bm_axis: int, keys: Union[None, List[Hashable]], levels: Union[None, List[Iterable[Hashable]]], verify_integrity: bool, names: Union[None, List[Hashable]]) -> Index:
    ...

def _get_concat_axis_dataframe(objs: List[Union[Series, DataFrame]], axis: int, ignore_index: bool, keys: Union[None, List[Hashable]], names: Union[None, List[Hashable]], levels: Union[None, List[Iterable[Hashable]]], verify_integrity: bool) -> Index:
    ...

def _clean_keys_and_objs(objs: Union[List[Union[Series, DataFrame]], Mapping[Hashable, Union[Series, DataFrame]]], keys: Union[None, List[Hashable]]) -> Tuple[List[Union[Series, DataFrame]], Union[None, Index], set[int]]:
    ...

def _get_sample_object(objs: List[Union[Series, DataFrame]], ndims: set[int], keys: Union[None, List[Hashable]], names: Union[None, List[Hashable]], levels: Union[None, List[Iterable[Hashable]]], intersect: bool) -> Tuple[Union[Series, DataFrame], List[Union[Series, DataFrame]]]:
    ...

def _concat_indexes(indexes: List[Index]) -> Index:
    ...

def validate_unique_levels(levels: List[Index]) -> None:
    ...

def _make_concat_multiindex(indexes: List[Index], keys: List[Hashable], levels: Union[None, List[Iterable[Hashable]]], names: Union[None, List[Hashable]]) -> MultiIndex:
    ...
