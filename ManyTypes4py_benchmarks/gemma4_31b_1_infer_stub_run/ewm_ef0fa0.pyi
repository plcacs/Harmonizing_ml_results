from __future__ import annotations
import datetime
from typing import Optional, Union, overload, Any, Literal, overload
import numpy as np
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from pandas.core.indexers.objects import ExponentialMovingWindowIndexer, GroupbyIndexer
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby

def get_center_of_mass(
    comass: Optional[float],
    span: Optional[float],
    halflife: Optional[float],
    alpha: Optional[float]
) -> float: ...

def _calculate_deltas(
    times: Union[np.ndarray, Series],
    halflife: Union[float, str, datetime.timedelta, np.timedelta64]
) -> np.ndarray: ...

class ExponentialMovingWindow(BaseWindow):
    com: Optional[float]
    span: Optional[float]
    halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]]
    alpha: Optional[float]
    min_periods: int
    adjust: bool
    ignore_na: bool
    times: Optional[Union[np.ndarray, Series]]
    method: str
    _com: float
    _deltas: np.ndarray

    def __init__(
        self,
        obj: NDFrame,
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: Optional[Union[np.ndarray, Series]] = None,
        method: str = 'single',
        *,
        selection: Optional[Any] = None,
    ) -> None: ...

    def _check_window_bounds(self, start: Any, end: Any, num_vals: int) -> None: ...

    def _get_window_indexer(self) -> ExponentialMovingWindowIndexer: ...

    def online(
        self,
        engine: str = 'numba',
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> OnlineExponentialMovingWindow: ...

    def aggregate(self, func: Optional[Any] = None, *args: Any, **kwargs: Any) -> NDFrame: ...
    agg: Any  # Alias for aggregate

    def mean(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal['numba', 'cython']] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
    ) -> NDFrame: ...

    def sum(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal['numba', 'cython']] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
    ) -> NDFrame: ...

    def std(self, bias: bool = False, numeric_only: bool = False) -> NDFrame: ...

    def var(self, bias: bool = False, numeric_only: bool = False) -> NDFrame: ...

    def cov(
        self,
        other: Optional[Union[Series, DataFrame]] = None,
        pairwise: Optional[bool] = None,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> NDFrame: ...

    def corr(
        self,
        other: Optional[Union[Series, DataFrame]] = None,
        pairwise: Optional[bool] = None,
        numeric_only: bool = False,
    ) -> NDFrame: ...

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    def __init__(self, obj: NDFrame, *args: Any, _grouper: Any = None, **kwargs: Any) -> None: ...

    def _get_window_indexer(self) -> GroupbyIndexer: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    engine: str
    engine_kwargs: Optional[dict[str, Any]]

    def __init__(
        self,
        obj: NDFrame,
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: Optional[Union[np.ndarray, Series]] = None,
        engine: str = 'numba',
        engine_kwargs: Optional[dict[str, Any]] = None,
        *,
        selection: Optional[Any] = None,
    ) -> None: ...

    def reset(self) -> None: ...

    def aggregate(self, func: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any: ...

    def std(self, bias: bool = False, *args: Any, **kwargs: Any) -> Any: ...

    def corr(self, other: Optional[Union[Series, DataFrame]] = None, pairwise: Optional[bool] = None, numeric_only: bool = False) -> Any: ...

    def cov(self, other: Optional[Union[Series, DataFrame]] = None, pairwise: Optional[bool] = None, bias: bool = False, numeric_only: bool = False) -> Any: ...

    def var(self, bias: bool = False, numeric_only: bool = False) -> Any: ...

    def mean(
        self,
        *args: Any,
        update: Optional[Union[Series, DataFrame]] = None,
        update_times: Optional[Union[Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[Series, DataFrame]: ...