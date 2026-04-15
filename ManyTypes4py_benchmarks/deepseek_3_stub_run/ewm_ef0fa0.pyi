from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, overload

import numpy as np
import pandas._libs.window.aggregations as window_aggregations
from pandas._libs.tslibs import Timedelta
from pandas.core.indexers.objects import BaseIndexer, ExponentialMovingWindowIndexer, GroupbyIndexer
from pandas.core.window.online import EWMMeanState
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        NDFrameT,
        NumpyFunc,
        Scalar,
        TimedeltaConvertibleTypes,
        npt,
    )
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame

def get_center_of_mass(
    comass: Optional[float],
    span: Optional[float],
    halflife: Optional[float],
    alpha: Optional[float],
) -> float: ...

def _calculate_deltas(
    times: np.ndarray | Series,
    halflife: float | str | datetime.timedelta | np.timedelta64,
) -> np.ndarray: ...

class ExponentialMovingWindow(BaseWindow):
    _attributes: list[str] = ...
    com: Optional[float]
    span: Optional[float]
    halflife: Optional[float | str | datetime.timedelta | np.timedelta64]
    alpha: Optional[float]
    adjust: bool
    ignore_na: bool
    times: Optional[np.ndarray | Series]
    method: str
    _deltas: np.ndarray
    _com: float

    def __init__(
        self,
        obj: NDFrame,
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[float | str | datetime.timedelta | np.timedelta64] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: Optional[np.ndarray | Series] = None,
        method: str = "single",
        *,
        selection: Optional[Any] = None,
    ) -> None: ...

    def _check_window_bounds(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_vals: int,
    ) -> None: ...

    def _get_window_indexer(self) -> ExponentialMovingWindowIndexer: ...

    def online(
        self,
        engine: str = "numba",
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> OnlineExponentialMovingWindow: ...

    @overload
    def aggregate(
        self,
        func: str | list[str] | dict[str, str | list[str]],
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: None = ...,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    def aggregate(
        self,
        func: Optional[str | list[str] | dict[str, str | list[str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...

    agg = aggregate

    def mean(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["numba", "cython"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> DataFrame | Series: ...

    def sum(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["numba", "cython"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> DataFrame | Series: ...

    def std(
        self,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> DataFrame | Series: ...

    def var(
        self,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> DataFrame | Series: ...

    def cov(
        self,
        other: Optional[DataFrame | Series] = None,
        pairwise: Optional[bool] = None,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> DataFrame | Series: ...

    def corr(
        self,
        other: Optional[DataFrame | Series] = None,
        pairwise: Optional[bool] = None,
        numeric_only: bool = False,
    ) -> DataFrame | Series: ...

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes: list[str] = ...

    def __init__(
        self,
        obj: NDFrame,
        *args: Any,
        _grouper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None: ...

    def _get_window_indexer(self) -> GroupbyIndexer: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    _mean: EWMMeanState
    engine: str
    engine_kwargs: Optional[dict[str, bool]]

    def __init__(
        self,
        obj: NDFrame,
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[float | str | datetime.timedelta | np.timedelta64] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: Optional[np.ndarray | Series] = None,
        engine: str = "numba",
        engine_kwargs: Optional[dict[str, bool]] = None,
        *,
        selection: Optional[Any] = None,
    ) -> None: ...

    def reset(self) -> None: ...

    def aggregate(self, func: Any = None, *args: Any, **kwargs: Any) -> Any: ...

    def std(self, bias: bool = False, *args: Any, **kwargs: Any) -> Any: ...

    def corr(
        self,
        other: Optional[DataFrame | Series] = None,
        pairwise: Optional[bool] = None,
        numeric_only: bool = False,
    ) -> Any: ...

    def cov(
        self,
        other: Optional[DataFrame | Series] = None,
        pairwise: Optional[bool] = None,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> Any: ...

    def var(self, bias: bool = False, numeric_only: bool = False) -> Any: ...

    def mean(
        self,
        *args: Any,
        update: Optional[DataFrame | Series] = None,
        update_times: Optional[Series | np.ndarray] = None,
        **kwargs: Any,
    ) -> DataFrame | Series: ...