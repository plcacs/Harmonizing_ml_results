from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable
import itertools
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs import missing as libmissing
from pandas._libs.sparse import IntIndex
from pandas.core.dtypes.common import is_integer_dtype, is_list_like, is_object_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype
from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.arrays.string_ import StringDtype
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index, default_index
from pandas.core.series import Series
if TYPE_CHECKING:
    from pandas._typing import NpDtype

def get_dummies(data: DataFrame | Series | Iterable, prefix: str | list[str] | dict[str, str] | None = None, prefix_sep: str = '_', dummy_na: bool = False, columns: list | None = None, sparse: bool = False, drop_first: bool = False, dtype: np.dtype | None = None) -> DataFrame:
    ...

def _get_dummies_1d(data: Series | Iterable, prefix: str | None, prefix_sep: str = '_', dummy_na: bool = False, sparse: bool = False, drop_first: bool = False, dtype: np.dtype | None = None) -> DataFrame:
    ...

def from_dummies(data: DataFrame, sep: str | None = None, default_category: Hashable | dict[Hashable, Hashable] | None = None) -> DataFrame:
    ...
