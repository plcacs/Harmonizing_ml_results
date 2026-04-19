from __future__ import annotations

import datetime
from typing import Literal, NoReturn, Optional, Union

import numpy as np
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from pandas.core.indexers.objects import (
    BaseIndexer,
    ExponentialMovingWindowIndexer,
    GroupbyIndexer,
)
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby


def get_center_of_mass(
    comass: Optional[float],
    span: Optional[float],
    halflife: Optional[float],
    alpha: Optional[float],
) -> float: ...


def _calculate_deltas(
    times: Union[Series, np.ndarray],
    halflife: Union[float, str, datetime.timedelta, np.timedelta64],
) -> np.ndarray: ...


class ExponentialMovingWindow(BaseWindow):
    _attributes: list[str]

    def __init__(
        self,
        obj: NDFrame,
        com: Optional[float] = ...,
        span: Optional[float] = ...,
        halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]] = ...,
        alpha: Optional[float] = ...,
        min_periods: Optional[int] = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        times: Optional[Union[Series, np.ndarray]] = ...,
        method: Literal["single", "table"] = ...,
        *,
        selection: Optional[object] = ...,
    ) -> None: ...

    def _check_window_bounds(self, start: int, end: int, num_vals: int) -> None: ...

    def _get_window_indexer(self) -> ExponentialMovingWindowIndexer: ...

    def online(
        self, engine: str = "numba", engine_kwargs: Optional[dict[str, bool]] = ...
    ) -> OnlineExponentialMovingWindow: ...

    def aggregate(
        self, func: Optional[object] = ..., *args: object, **kwargs: object
    ) -> Union[Series, DataFrame]: ...

    def agg(
        self, func: Optional[object] = ..., *args: object, **kwargs: object
    ) -> Union[Series, DataFrame]: ...

    def mean(
        self,
        numeric_only: bool = ...,
        engine: Optional[str] = ...,
        engine_kwargs: Optional[dict[str, bool]] = ...,
    ) -> Union[Series, DataFrame]: ...

    def sum(
        self,
        numeric_only: bool = ...,
        engine: Optional[str] = ...,
        engine_kwargs: Optional[dict[str, bool]] = ...,
    ) -> Union[Series, DataFrame]: ...

    def std(self, bias: bool = ..., numeric_only: bool = ...) -> Union[Series, DataFrame]: ...

    def var(self, bias: bool = ..., numeric_only: bool = ...) -> Union[Series, DataFrame]: ...

    def cov(
        self,
        other: Optional[Union[Series, DataFrame]] = ...,
        pairwise: Optional[bool] = ...,
        bias: bool = ...,
        numeric_only: bool = ...,
    ) -> Union[Series, DataFrame]: ...

    def corr(
        self,
        other: Optional[Union[Series, DataFrame]] = ...,
        pairwise: Optional[bool] = ...,
        numeric_only: bool = ...,
    ) -> Union[Series, DataFrame]: ...


class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes: list[str]

    def __init__(
        self, obj: NDFrame, *args: object, _grouper: Optional[object] = ..., **kwargs: object
    ) -> None: ...

    def _get_window_indexer(self) -> GroupbyIndexer: ...


class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    def __init__(
        self,
        obj: NDFrame,
        com: Optional[float] = ...,
        span: Optional[float] = ...,
        halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]] = ...,
        alpha: Optional[float] = ...,
        min_periods: Optional[int] = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        times: Optional[Union[Series, np.ndarray]] = ...,
        engine: str = "numba",
        engine_kwargs: Optional[dict[str, bool]] = ...,
        *,
        selection: Optional[object] = ...,
    ) -> None: ...

    def reset(self) -> None: ...

    def aggregate(self, func: Optional[object] = ..., *args: object, **kwargs: object) -> NoReturn: ...

    def std(self, bias: bool = ..., *args: object, **kwargs: object) -> NoReturn: ...

    def corr(
        self, other: Optional[Union[Series, DataFrame]] = ..., pairwise: Optional[bool] = ..., numeric_only: bool = ...
    ) -> NoReturn: ...

    def cov(
        self,
        other: Optional[Union[Series, DataFrame]] = ...,
        pairwise: Optional[bool] = ...,
        bias: bool = ...,
        numeric_only: bool = ...,
    ) -> NoReturn: ...

    def var(self, bias: bool = ..., numeric_only: bool = ...) -> NoReturn: ...

    def mean(
        self,
        *args: object,
        update: Optional[Union[DataFrame, Series]] = ...,
        update_times: Optional[Union[Series, np.ndarray]] = ...,
        **kwargs: object,
    ) -> Union[Series, DataFrame]: ...