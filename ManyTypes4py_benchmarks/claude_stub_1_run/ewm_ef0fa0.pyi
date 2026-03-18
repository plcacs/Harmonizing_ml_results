```python
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
from pandas._libs.tslibs import Timedelta
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.indexers.objects import BaseIndexer
from pandas.core.window.rolling import BaseWindow, BaseWindowGroupby

if TYPE_CHECKING:
    from pandas._typing import TimedeltaConvertibleTypes, npt
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame

def get_center_of_mass(
    comass: Optional[float],
    span: Optional[float],
    halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]],
    alpha: Optional[float],
) -> float: ...

def _calculate_deltas(
    times: Union[np.ndarray, Series],
    halflife: Union[float, str, datetime.timedelta, np.timedelta64],
) -> np.ndarray: ...

class ExponentialMovingWindow(BaseWindow):
    _attributes: list[str]
    com: Optional[float]
    span: Optional[float]
    halflife: Optional[Union[float, str, datetime.timedelta, np.timedelta64]]
    alpha: Optional[float]
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
        method: str = "single",
        *,
        selection: Optional[Any] = None,
    ) -> None: ...

    def _check_window_bounds(self, start: Any, end: Any, num_vals: Any) -> None: ...
    def _get_window_indexer(self) -> BaseIndexer: ...
    def online(
        self,
        engine: str = "numba",
        engine_kwargs: Optional[dict[str, Any]] = None,
    ) -> OnlineExponentialMovingWindow: ...
    def aggregate(
        self,
        func: Optional[Union[str, Callable[..., Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]: ...
    def agg(
        self,
        func: Optional[Union[str, Callable[..., Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]: ...
    def mean(
        self,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[DataFrame, Series]: ...
    def sum(
        self,
        numeric_only: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union[DataFrame, Series]: ...
    def std(
        self,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]: ...
    def var(
        self,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]: ...
    def cov(
        self,
        other: Optional[Union[Series, DataFrame]] = None,
        pairwise: Optional[bool] = None,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]: ...
    def corr(
        self,
        other: Optional[Union[Series, DataFrame]] = None,
        pairwise: Optional[bool] = None,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]: ...

class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    _attributes: list[str]

    def __init__(
        self,
        obj: NDFrame,
        *args: Any,
        _grouper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None: ...
    def _get_window_indexer(self) -> BaseIndexer: ...

class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    engine: str
    engine_kwargs: Optional[dict[str, Any]]
    _mean: Any

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
        engine: str = "numba",
        engine_kwargs: Optional[dict[str, Any]] = None,
        *,
        selection: Optional[Any] = None,
    ) -> None: ...
    def reset(self) -> None: ...
    def aggregate(
        self,
        func: Optional[Union[str, Callable[..., Any]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
    def std(
        self,
        bias: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
    def corr(
        self,
        other: Optional[Union[Series, DataFrame]] = None,
        pairwise: Optional[bool] = None,
        numeric_only: bool = False,
    ) -> Any: ...
    def cov(
        self,
        other: Optional[Union[Series, DataFrame]] = None,
        pairwise: Optional[bool] = None,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> Any: ...
    def var(
        self,
        bias: bool = False,
        numeric_only: bool = False,
    ) -> Any: ...
    def mean(
        self,
        *args: Any,
        update: Optional[Union[DataFrame, Series]] = None,
        update_times: Optional[Union[Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]: ...
```