```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import datetime
import numpy as np
import pandas._libs.window.aggregations as window_aggregations

if TYPE_CHECKING:
    from pandas._typing import TimedeltaConvertibleTypes, npt
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame
    from pandas.core.indexers.objects import BaseIndexer, ExponentialMovingWindowIndexer, GroupbyIndexer

def get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None
) -> float: ...

def _calculate_deltas(
    times: np.ndarray | Series,
    halflife: TimedeltaConvertibleTypes
) -> np.ndarray: ...

class ExponentialMovingWindow(BaseWindow):
    _attributes: list[str] = ...
    com: float | None
    span: float | None
    halflife: float | str | datetime.timedelta | np.timedelta64 | None
    alpha: float | None
    adjust: bool
    ignore_na: bool
    times: np.ndarray | Series | None
    method: str
    _com: float
    _deltas: np.ndarray
    
    def __init__(
        self,
        obj: NDFrame,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | str | datetime.timedelta | np.timedelta64 | None = ...,
        alpha: float | None = ...,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: np.ndarray | Series | None = ...,
        method: str = "single",
        *,
        selection: Any = ...
    ) -> None: ...
    
    def _check_window_bounds(
        self,
        start: Any,
        end: Any,
        num_vals: Any
    ) -> None: ...
    
    def _get_window_indexer(self) -> ExponentialMovingWindowIndexer: ...
    
    def online(
        self,
        engine: str = "numba",
        engine_kwargs: dict[str, bool] | None = ...
    ) -> OnlineExponentialMovingWindow: ...
    
    @overload
    def aggregate(
        self,
        func: str | Callable[..., Any] | list[str | Callable[..., Any]],
        *args: Any,
        **kwargs: Any
    ) -> DataFrame: ...
    
    @overload
    def aggregate(
        self,
        func: None = ...,
        *args: Any,
        **kwargs: Any
    ) -> DataFrame: ...
    
    def aggregate(
        self,
        func: str | Callable[..., Any] | list[str | Callable[..., Any]] | None = ...,
        *args: Any,
        **kwargs: Any
    ) -> DataFrame: ...
    
    agg = aggregate
    
    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["numba", "cython"] | None = ...,
        engine_kwargs: dict[str, bool] | None = ...
    ) -> DataFrame | Series: ...
    
    def sum(
        self,
        numeric_only: bool = False,
        engine: Literal["numba", "cython"] | None = ...,
        engine_kwargs: dict[str, bool] | None = ...
    ) -> DataFrame | Series: ...
    
    def std(
        self,
        bias: bool = False,
        numeric_only: bool = False
    ) -> DataFrame | Series: ...
    
    def var(
        self,
        bias: bool = False,
        numeric_only: bool = False
    ) -> DataFrame | Series: ...
    
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = False,
        numeric_only: bool = False
    ) -> DataFrame | Series: ...
    
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        numeric_only: bool = False
    ) -> DataFrame | Series: ...

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes: list[str] = ...
    
    def __init__(
        self,
        obj: NDFrame,
        *args: Any,
        _grouper: Any = ...,
        **kwargs: Any
    ) -> None: ...
    
    def _get_window_indexer(self) -> GroupbyIndexer: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    engine: str
    engine_kwargs: dict[str, bool] | None
    _mean: EWMMeanState
    
    def __init__(
        self,
        obj: NDFrame,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | str | datetime.timedelta | np.timedelta64 | None = ...,
        alpha: float | None = ...,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: np.ndarray | Series | None = ...,
        engine: str = "numba",
        engine_kwargs: dict[str, bool] | None = ...,
        *,
        selection: Any = ...
    ) -> None: ...
    
    def reset(self) -> None: ...
    
    def aggregate(
        self,
        func: Any = ...,
        *args: Any,
        **kwargs: Any
    ) -> Any: ...
    
    def std(
        self,
        bias: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> Any: ...
    
    def corr(
        self,
        other: Any = ...,
        pairwise: Any = ...,
        numeric_only: bool = False
    ) -> Any: ...
    
    def cov(
        self,
        other: Any = ...,
        pairwise: Any = ...,
        bias: bool = False,
        numeric_only: bool = False
    ) -> Any: ...
    
    def var(
        self,
        bias: bool = False,
        numeric_only: bool = False
    ) -> Any: ...
    
    def mean(
        self,
        *args: Any,
        update: DataFrame | Series | None = ...,
        update_times: Series | np.ndarray | None = ...,
        **kwargs: Any
    ) -> DataFrame | Series: ...
```