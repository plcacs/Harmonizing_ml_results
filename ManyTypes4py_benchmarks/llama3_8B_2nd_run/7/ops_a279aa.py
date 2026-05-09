from __future__ import annotations
import collections
import functools
from typing import TYPE_CHECKING, Generic, final, Iterator, Hashable, Callable, Sequence
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

class WrappedCythonOp:
    ...

class BaseGrouper(Generic[NDFrameT]):
    ...

class BinGrouper(BaseGrouper):
    ...

class DataSplitter(Generic[NDFrameT]):
    ...

class SeriesSplitter(DataSplitter):
    ...

class FrameSplitter(DataSplitter):
    ...
