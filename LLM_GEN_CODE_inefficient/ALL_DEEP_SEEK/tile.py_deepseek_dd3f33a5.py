from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
    Tuple,
    List,
    Sequence,
    Callable,
)

import numpy as np

from pandas._libs import (
    Timedelta,
    Timestamp,
    lib,
)

from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna

from pandas import (
    Categorical,
    Index,
    IntervalIndex,
)
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        DtypeObj,
        IntervalLeftRight,
    )


def cut(
    x: Union[np.ndarray, ABCSeries],
    bins: Union[int, Sequence[float],
    right: bool = True,
    labels: Optional[Union[Sequence, bool]] = None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
) -> Union[Categorical, ABCSeries, np.ndarray, Tuple[Union[Categorical, ABCSeries, np.ndarray], Index]]:
    # Function implementation remains the same
    pass


def qcut(
    x: Union[np.ndarray, ABCSeries],
    q: Union[int, Sequence[float]],
    labels: Optional[Union[Sequence, bool]] = None,
    retbins: bool = False,
    precision: int = 3,
    duplicates: str = "raise",
) -> Union[Categorical, ABCSeries, np.ndarray, Tuple[Union[Categorical, ABCSeries, np.ndarray], np.ndarray]]:
    # Function implementation remains the same
    pass


def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
    # Function implementation remains the same
    pass


def _bins_to_cuts(
    x_idx: Index,
    bins: Index,
    right: bool = True,
    labels: Optional[Union[Sequence, bool]] = None,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
) -> Tuple[Union[Categorical, np.ndarray], Index]:
    # Function implementation remains the same
    pass


def _coerce_to_type(x: Index) -> Tuple[Index, Optional[DtypeObj]]:
    # Function implementation remains the same
    pass


def _is_dt_or_td(dtype: DtypeObj) -> bool:
    # Function implementation remains the same
    pass


def _format_labels(
    bins: Index,
    precision: int,
    right: bool = True,
    include_lowest: bool = False,
) -> IntervalIndex:
    # Function implementation remains the same
    pass


def _preprocess_for_cut(x: Union[np.ndarray, ABCSeries]) -> Index:
    # Function implementation remains the same
    pass


def _postprocess_for_cut(
    fac: Union[Categorical, np.ndarray],
    bins: Index,
    retbins: bool,
    original: Union[np.ndarray, ABCSeries],
) -> Union[Union[Categorical, ABCSeries, np.ndarray], Tuple[Union[Categorical, ABCSeries, np.ndarray], Index]]:
    # Function implementation remains the same
    pass


def _round_frac(x: float, precision: int) -> float:
    # Function implementation remains the same
    pass


def _infer_precision(base_precision: int, bins: Index) -> int:
    # Function implementation remains the same
    pass
