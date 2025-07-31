from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import datetime
from functools import partial, wraps
from textwrap import dedent
from typing import Any, Optional, Union, List, Sequence, cast
import warnings
import numpy as np
from pandas._libs import Timestamp, lib
from pandas._libs.algos import rank_1d
import pandas._libs.groupby as libgroupby
from pandas._libs.missing import NA
from pandas._typing import AnyArrayLike, ArrayLike, DtypeObj, IndexLabel, IntervalClosedType, NDFrameT, PositionalIndexer, RandomState, npt
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError, DataError
from pandas.util._decorators import Appender, Substitution, cache_readonly, doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype, ensure_dtype_can_hold_na
from pandas.core.dtypes.common import is_bool_dtype, is_float_dtype, is_hashable, is_integer, is_integer_dtype, is_list_like, is_numeric_dtype, is_object_dtype, is_scalar, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna
from pandas.core import algorithms, sample
from pandas.core._numba import executor
from pandas.core.arrays import ArrowExtensionArray, BaseMaskedArray, ExtensionArray, FloatingArray, IntegerArray, SparseArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.string_arrow import ArrowStringArray, ArrowStringArrayNumpySemantics
from pandas.core.base import PandasObject, SelectionMixin
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby import base, numba_, ops
from pandas.core.groupby.grouper import get_grouper
from pandas.core.groupby.indexing import GroupByIndexingMixin, GroupByNthSelector
from pandas.core.indexes.api import Index, MultiIndex, default_index
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.series import Series
from pandas.core.sorting import get_group_index_sorter
from pandas.core.util.numba_ import get_jit_arguments, maybe_use_numba, prepare_function_arguments

# ... many methods and classes within the GroupBy implementation ...

@doc(GroupBy)
def get_groupby(obj: Union[Series, DataFrame], 
                by: Optional[Any] = None, 
                grouper: Optional[Any] = None, 
                group_keys: bool = True) -> Union[SeriesGroupBy, DataFrameGroupBy]:
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy
        klass = SeriesGroupBy
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy
        klass = DataFrameGroupBy
    else:
        raise TypeError(f'invalid type: {obj}')
    return klass(obj=obj, keys=by, grouper=grouper, group_keys=group_keys)

def _insert_quantile_level(idx: Index, qs: np.ndarray) -> MultiIndex:
    """
    Insert the sequence 'qs' of quantiles as the inner-most level of a MultiIndex.

    The quantile level in the MultiIndex is a repeated copy of 'qs'.

    Parameters
    ----------
    idx : Index
    qs : np.ndarray[float64]

    Returns
    -------
    MultiIndex
    """
    nqs: int = len(qs)
    lev_codes, lev = Index(qs).factorize()
    lev_codes = coerce_indexer_dtype(lev_codes, lev)
    if idx._is_multi:
        idx = cast(MultiIndex, idx)
        levels: List[Any] = list(idx.levels) + [lev]
        codes: List[np.ndarray] = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(lev_codes, len(idx))]
        mi: MultiIndex = MultiIndex(levels=levels, codes=codes, names=idx.names + [None])
    else:
        nidx: int = len(idx)
        idx_codes = coerce_indexer_dtype(np.arange(nidx), idx)
        levels = [idx, lev]
        codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
        mi: MultiIndex = MultiIndex(levels=levels, codes=codes, names=[idx.name, None])
    return mi