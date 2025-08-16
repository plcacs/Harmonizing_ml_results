"""
Quantilization functions and related stuff
"""

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
    cast,
)

import numpy as np

from pandas._libs import (
    Timedelta,
    Timestamp,
    lib,
)
from pandas._libs.tslibs import BaseOffset

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
    Series,
)
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        DtypeObj,
        IntervalLeftRight,
        NumpyIndexT,
        Scalar,
        ArrayLike,
        npt,
    )


def cut(
    x: Union[ArrayLike, Series],
    bins: Union[int, Sequence[Scalar], IntervalIndex],
    right: bool = True,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
) -> Union[Categorical, Series, np.ndarray, Tuple[Union[Categorical, Series, np.ndarray], np.ndarray]]:
    """
    Bin values into discrete intervals.
    [Rest of the docstring remains the same]
    """
    # [Function implementation remains the same]


def qcut(
    x: Union[ArrayLike, Series],
    q: Union[int, Sequence[float]],
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    duplicates: str = "raise",
) -> Union[Categorical, Series, np.ndarray, Tuple[Union[Categorical, Series, np.ndarray], np.ndarray]]:
    """
    Quantile-based discretization function.
    [Rest of the docstring remains the same]
    """
    # [Function implementation remains the same]


def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _bins_to_cuts(
    x_idx: Index,
    bins: Index,
    right: bool = True,
    labels=None,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
) -> Tuple[Union[Categorical, np.ndarray], Index]:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _coerce_to_type(x: Index) -> Tuple[Index, Optional[DtypeObj]]:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _is_dt_or_td(dtype: DtypeObj) -> bool:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _format_labels(
    bins: Index,
    precision: int,
    right: bool = True,
    include_lowest: bool = False,
) -> IntervalIndex:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _preprocess_for_cut(x: Union[ArrayLike, Series]) -> Index:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _postprocess_for_cut(
    fac: Union[Categorical, np.ndarray],
    bins: Index,
    retbins: bool,
    original: Union[ArrayLike, Series],
) -> Union[Union[Categorical, Series, np.ndarray], Tuple[Union[Categorical, Series, np.ndarray], np.ndarray]]:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _round_frac(x: float, precision: int) -> float:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]


def _infer_precision(base_precision: int, bins: Index) -> int:
    """
    [Docstring remains the same]
    """
    # [Function implementation remains the same]
