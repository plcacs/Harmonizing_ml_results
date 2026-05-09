from __future__ import annotations
from contextlib import contextmanager
from typing import (
    Any,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    Dict,
    TypeVar,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.table import Table
from matplotlib.colors import Colormap
from pandas import DataFrame, Series
from numpy import ndarray

class _Options(dict):
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def reset(self) -> None: ...
    def _get_canonical_key(self, key: str) -> str: ...
    @contextmanager
    def use(self, key: str, value: Any) -> Generator[_Options, None, None]: ...

def table(ax: Axes, data: Union[DataFrame, Series], **kwargs: Any) -> Table: ...

def register() -> None: ...

def deregister() -> None: ...

def scatter_matrix(
    frame: DataFrame,
    alpha: float = ...,
    figsize: Optional[Tuple[float, float]] = ...,
    ax: Optional[Axes] = ...,
    grid: bool = ...,
    diagonal: str = ...,
    marker: str = ...,
    density_kwds: Optional[Dict[str, Any]] = ...,
    hist_kwds: Optional[Dict[str, Any]] = ...,
    range_padding: float = ...,
    **kwargs: Any
) -> ndarray[Axes]: ...

def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Optional[Axes] = ...,
    color: Optional[Union[List[str], Tuple[str, ...]]] = ...,
    colormap: Optional[Union[str, Colormap]] = ...,
    **kwds: Any
) -> Axes: ...

def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: Optional[Axes] = ...,
    samples: int = ...,
    color: Optional[Union[List[str], Tuple[str, ...]]] = ...,
    colormap: Optional[Union[str, Colormap]] = ...,
    **kwargs: Any
) -> Axes: ...

def bootstrap_plot(
    series: Series,
    fig: Optional[Figure] = ...,
    size: int = ...,
    samples: int = ...,
    **kwds: Any
) -> Figure: ...

def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: Optional[List[str]] = ...,
    ax: Optional[Axes] = ...,
    color: Optional[Union[List[str], Tuple[str, ...]]] = ...,
    use_columns: bool = ...,
    xticks: Optional[Union[List[Any], Tuple[Any, ...]]] = ...,
    colormap: Optional[Union[str, Colormap]] = ...,
    axvlines: bool = ...,
    axvlines_kwds: Optional[Dict[str, Any]] = ...,
    sort_labels: bool = ...,
    **kwargs: Any
) -> Axes: ...

def lag_plot(
    series: Series,
    lag: int = ...,
    ax: Optional[Axes] = ...,
    **kwds: Any
) -> Axes: ...

def autocorrelation_plot(
    series: Series,
    ax: Optional[Axes] = ...,
    **kwargs: Any
) -> Axes: ...

plot_params: _Options = ...