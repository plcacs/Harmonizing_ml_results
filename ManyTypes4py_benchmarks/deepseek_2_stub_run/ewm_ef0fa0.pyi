```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, overload
import datetime
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import TimedeltaConvertibleTypes, npt
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame
    import pandas._libs.window.aggregations as window_aggregations

def get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None
) -> float: ...

def _calculate_deltas(
    times: np.ndarray | Any,
    halflife: float | str | datetime.timedelta | np.timedelta64
) -> np.ndarray: ...

class ExponentialMovingWindow(BaseWindow):
    _attributes: list[str] = ...
    
    def __init__(
        self,
        obj: Any,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | str | datetime.timedelta | np.timedelta64 | None = ...,
        alpha: float | None = ...,
        min_periods: int = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        times: np.ndarray | Any | None = ...,
        method: str = ...,
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
        engine: str = ...,
        engine_kwargs: dict[str, Any] | None = ...
    ) -> OnlineExponentialMovingWindow: ...
    
    @overload
    def aggregate(
        self,
        func: str | list[str] | dict[str, str | list[str]],
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
        func: Any = ...,
        *args: Any,
        **kwargs: Any
    ) -> Any: ...
    
    agg = aggregate
    
    def mean(
        self,
        numeric_only: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, Any] | None = ...
    ) -> DataFrame | Series: ...
    
    def sum(
        self,
        numeric_only: bool = ...,
        engine: str | None = ...,
        engine_kwargs: dict[str, Any] | None = ...
    ) -> DataFrame | Series: ...
    
    def std(
        self,
        bias: bool = ...,
        numeric_only: bool = ...
    ) -> DataFrame | Series: ...
    
    def var(
        self,
        bias: bool = ...,
        numeric_only: bool = ...
    ) -> DataFrame | Series: ...
    
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        numeric_only: bool = ...
    ) -> DataFrame | Series: ...
    
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        numeric_only: bool = ...
    ) -> DataFrame | Series: ...

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes: list[str] = ...
    
    def __init__(
        self,
        obj: Any,
        *args: Any,
        _grouper: Any = ...,
        **kwargs: Any
    ) -> None: ...
    
    def _get_window_indexer(self) -> GroupbyIndexer: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    def __init__(
        self,
        obj: Any,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | str | datetime.timedelta | np.timedelta64 | None = ...,
        alpha: float | None = ...,
        min_periods: int = ...,
        adjust: bool = ...,
        ignore_na: bool = ...,
        times: np.ndarray | Any | None = ...,
        engine: str = ...,
        engine_kwargs: dict[str, Any] | None = ...,
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
        bias: bool = ...,
        *args: Any,
        **kwargs: Any
    ) -> Any: ...
    
    def corr(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        numeric_only: bool = ...
    ) -> Any: ...
    
    def cov(
        self,
        other: DataFrame | Series | None = ...,
        pairwise: bool | None = ...,
        bias: bool = ...,
        numeric_only: bool = ...
    ) -> Any: ...
    
    def var(
        self,
        bias: bool = ...,
        numeric_only: bool = ...
    ) -> Any: ...
    
    def mean(
        self,
        *args: Any,
        update: DataFrame | Series | None = ...,
        update_times: Any | None = ...,
        **kwargs: Any
    ) -> DataFrame | Series: ...

class BaseWindow: ...
class BaseWindowGroupby: ...
class ExponentialMovingWindowIndexer: ...
class GroupbyIndexer: ...
class EWMMeanState: ...
```