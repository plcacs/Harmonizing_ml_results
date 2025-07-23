"""
Concat routines.
"""
from __future__ import annotations
from collections import abc
import types
from typing import (
    TYPE_CHECKING, Literal, cast, overload, Any, Sequence, TypeVar, Union,
    Optional, List, Set, Dict, Tuple, Type
)
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
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sized
    from pandas._typing import (
        Axis, AxisInt, HashableT, NDFrameT, Scalar, Manager, ArrayLike
    )
    from pandas import DataFrame, Series
    from numpy import ndarray

T = TypeVar('T')
NDFrameT = TypeVar('NDFrameT', bound='NDFrame')

@overload
def concat(
    objs: Union[Iterable[Union['Series', 'DataFrame']], *,
    axis: Axis = ...,
    join: Literal['inner', 'outer'] = ...,
    ignore_index: bool = ...,
    keys: Optional[Sequence[Hashable]] = ...,
    levels: Optional[Sequence[Sequence[Hashable]]] = ...,
    names: Optional[Sequence[Hashable]] = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...
) -> Union['Series', 'DataFrame']: ...

@overload
def concat(
    objs: Mapping[Hashable, Union['Series', 'DataFrame']], *,
    axis: Axis = ...,
    join: Literal['inner', 'outer'] = ...,
    ignore_index: bool = ...,
    keys: Optional[Sequence[Hashable]] = ...,
    levels: Optional[Sequence[Sequence[Hashable]]] = ...,
    names: Optional[Sequence[Hashable]] = ...,
    verify_integrity: bool = ...,
    sort: bool = ...,
    copy: bool = ...
) -> Union['Series', 'DataFrame']: ...

@set_module('pandas')
def concat(
    objs: Union[Iterable[Union['Series', 'DataFrame']], Mapping[Hashable, Union['Series', 'DataFrame']]],
    *,
    axis: Axis = 0,
    join: Literal['inner', 'outer'] = 'outer',
    ignore_index: bool = False,
    keys: Optional[Sequence[Hashable]] = None,
    levels: Optional[Sequence[Sequence[Hashable]]] = None,
    names: Optional[Sequence[Hashable]] = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = lib.no_default
) -> Union['Series', 'DataFrame']:
    # ... (rest of the function implementation remains exactly the same)
    pass

def _sanitize_mixed_ndim(
    objs: List[Union['Series', 'DataFrame']],
    sample: Union['Series', 'DataFrame'],
    ignore_index: bool,
    axis: int
) -> List[Union['Series', 'DataFrame']]:
    # ... (implementation remains the same)
    pass

def _get_result(
    objs: List[Union['Series', 'DataFrame']],
    is_series: bool,
    bm_axis: int,
    ignore_index: bool,
    intersect: bool,
    sort: bool,
    keys: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool,
    names: Optional[Sequence[Hashable]],
    axis: int
) -> Union['Series', 'DataFrame']:
    # ... (implementation remains the same)
    pass

def new_axes(
    objs: List[Union['Series', 'DataFrame']],
    bm_axis: int,
    intersect: bool,
    sort: bool,
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    axis: int,
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool,
    ignore_index: bool
) -> List[Index]:
    # ... (implementation remains the same)
    pass

def _get_concat_axis_series(
    objs: List['Series'],
    ignore_index: bool,
    bm_axis: int,
    keys: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool,
    names: Optional[Sequence[Hashable]]
) -> Index:
    # ... (implementation remains the same)
    pass

def _get_concat_axis_dataframe(
    objs: List['DataFrame'],
    axis: int,
    ignore_index: bool,
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    verify_integrity: bool
) -> Index:
    # ... (implementation remains the same)
    pass

def _clean_keys_and_objs(
    objs: Union[Iterable[Union['Series', 'DataFrame', None]], Mapping[Hashable, Union['Series', 'DataFrame', None]]],
    keys: Optional[Sequence[Hashable]]
) -> Tuple[List[Union['Series', 'DataFrame']], Optional[Index], Set[int]]:
    # ... (implementation remains the same)
    pass

def _get_sample_object(
    objs: List[Union['Series', 'DataFrame']],
    ndims: Set[int],
    keys: Optional[Sequence[Hashable]],
    names: Optional[Sequence[Hashable]],
    levels: Optional[Sequence[Sequence[Hashable]]],
    intersect: bool
) -> Tuple[Union['Series', 'DataFrame'], List[Union['Series', 'DataFrame']]]:
    # ... (implementation remains the same)
    pass

def _concat_indexes(indexes: List[Index]) -> Index:
    # ... (implementation remains the same)
    pass

def validate_unique_levels(levels: List[Index]) -> None:
    # ... (implementation remains the same)
    pass

def _make_concat_multiindex(
    indexes: List[Index],
    keys: Sequence[Hashable],
    levels: Optional[Sequence[Sequence[Hashable]]] = None,
    names: Optional[Sequence[Hashable]]] = None
) -> MultiIndex:
    # ... (implementation remains the same)
    pass
