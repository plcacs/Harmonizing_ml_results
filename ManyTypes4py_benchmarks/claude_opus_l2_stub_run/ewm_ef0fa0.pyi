from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from pandas.core.indexers.objects import (
    ExponentialMovingWindowIndexer,
    GroupbyIndexer,
)
from pandas.core.window.online import EWMMeanState
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby

if TYPE_CHECKING:
    from pandas._typing import TimedeltaConvertibleTypes, npt
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame


def get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float: ...


def _calculate_deltas(
    times: np.ndarray | Series,
    halflife: float | str | datetime.timedelta | np.timedelta64,
) -> np.ndarray: ...


class ExponentialMovingWindow(BaseWindow):
    _attributes: list[str]
    com: float | None
    span: float | None
    halflife: float | str | datetime.timedelta | np.timedelta64 | None
    alpha: float | None
    adjust: bool
    ignore_na: bool
    times: Series | np.ndarray | None
    _com: float
    _deltas: np.ndarray

    def __init__(
        self,
        obj: NDFrame,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | str | datetime.timedelta | np.timedelta64 | None = ...,
        alpha: float | None = ...,
        min_periods: int = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        times: Series | np.ndarray | None = ...,
        method: str = ...,
        *,
        selection: str | None = ...,
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
        engine: str = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> OnlineExponentialMovingWindow: ...
    def aggregate(self, func: Any = ..., *args: Any, **kwargs: Any) -> DataFrame | Series: ...
    agg = aggregate
    def mean(
        self,
        numeric_only: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> DataFrame | Series: ...
    def sum(
        self,
        numeric_only: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, bool] | None = ...,
    ) -> DataFrame | Series: ...
    def std(
        self,
        bias: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame | Series: ...
    def var(
        self,
        bias: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame | Series: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame | Series: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        numeric_only: bool = ...,
    ) -> DataFrame | Series: ...


class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes: list[str]

    def __init__(
        self,
        obj: NDFrame,
        *args: Any,
        _grouper: Any = ...,
        **kwargs: Any,
    ) -> None: ...
    def _get_window_indexer(self) -> GroupbyIndexer: ...


class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    _mean: EWMMeanState
    engine: str
    engine_kwargs: dict[str, bool] | None

    def __init__(
        self,
        obj: NDFrame,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | str | datetime.timedelta | np.timedelta64 | None = ...,
        alpha: float | None = ...,
        min_periods: int = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        times: Series | np.ndarray | None = ...,
        engine: str = ...,
        engine_kwargs: dict[str, bool] | None = ...,
        *,
        selection: str | None = ...,
    ) -> None: ...
    def reset(self) -> None: ...
    def aggregate(self, func: Any = ..., *args: Any, **kwargs: Any) -> Any: ...
    def std(self, bias: bool = ..., *args: Any, **kwargs: Any) -> Any: ...
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        numeric_only: bool = ...,
    ) -> Any: ...
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        numeric_only: bool = ...,
    ) -> Any: ...
    def var(self, bias: bool = ..., numeric_only: bool = ...) -> Any: ...
    def mean(
        self,
        *args: Any,
        update: DataFrame | Series | None = ...,
        update_times: Series | np.ndarray | None = ...,
        **kwargs: Any,
    ) -> DataFrame | Series: ...