from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from pandas._libs import Timedelta, Timestamp, lib
from pandas.core.dtypes.common import ensure_platform_int, is_bool_dtype, is_integer, is_list_like, is_numeric_dtype, is_scalar
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas import Categorical, Index, IntervalIndex
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import DtypeObj, IntervalLeftRight

def cut(x: Any, bins: Any, right: bool = True, labels: Any = None, retbins: bool = False, precision: int = 3, include_lowest: bool = False, duplicates: Literal['raise', 'drop'] = 'raise', ordered: bool = True) -> Any:
    ...

def qcut(x: Any, q: Any, labels: Any = None, retbins: bool = False, precision: int = 3, duplicates: Literal['raise', 'drop'] = 'raise') -> Any:
    ...

def _nbins_to_bins(x_idx: Any, nbins: int, right: bool) -> IntervalIndex:
    ...

def _bins_to_cuts(x_idx: Any, bins: IntervalIndex, right: bool, labels: Any, precision: int, include_lowest: bool, duplicates: str, ordered: bool) -> Tuple[Any, IntervalIndex]:
    ...

def _coerce_to_type(x: Any) -> Tuple[Index, Any]:
    ...

def _is_dt_or_td(dtype: DtypeObj) -> bool:
    ...

def _format_labels(bins: IntervalIndex, precision: int, right: bool, include_lowest: bool) -> IntervalIndex:
    ...

def _preprocess_for_cut(x: Any) -> Index:
    ...

def _postprocess_for_cut(fac: Any, bins: IntervalIndex, retbins: bool, original: Any) -> Any:
    ...
