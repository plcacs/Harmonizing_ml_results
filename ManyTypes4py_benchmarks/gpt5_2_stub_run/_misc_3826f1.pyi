from typing import Any, ContextManager, Mapping
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.table import Table
from pandas import DataFrame, Series

def table(ax: Axes, data: DataFrame | Series, **kwargs: Any) -> Table: ...
def register() -> None: ...
def deregister() -> None: ...
def scatter_matrix(
    frame: DataFrame,
    alpha: float = ...,
    figsize: tuple[float, float] | None = ...,
    ax: Axes | None = ...,
    grid: bool = ...,
    diagonal: str = ...,
    marker: str = ...,
    density_kwds: Mapping[str, Any] | None = ...,
    hist_kwds: Mapping[str, Any] | None = ...,
    range_padding: float = ...,
    **kwargs: Any
) -> Any: ...
def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = ...,
    color: list[str] | tuple[str, ...] | None = ...,
    colormap: str | Colormap | None = ...,
    **kwds: Any
) -> Axes: ...
def andrews_curves(
    frame: DataFrame,
    class_column: Any,
    ax: Axes | None = ...,
    samples: int = ...,
    color: str | list[str] | tuple[str, ...] | None = ...,
    colormap: str | Colormap | None = ...,
    **kwargs: Any
) -> Axes: ...
def bootstrap_plot(
    series: Series,
    fig: Figure | None = ...,
    size: int = ...,
    samples: int = ...,
    **kwds: Any
) -> Figure: ...
def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: list[Any] | None = ...,
    ax: Axes | None = ...,
    color: list[Any] | tuple[Any, ...] | None = ...,
    use_columns: bool = ...,
    xticks: list[Any] | tuple[Any, ...] | None = ...,
    colormap: str | Colormap | None = ...,
    axvlines: bool = ...,
    axvlines_kwds: Mapping[str, Any] | None = ...,
    sort_labels: bool = ...,
    **kwargs: Any
) -> Axes: ...
def lag_plot(series: Series, lag: int = ..., ax: Axes | None = ..., **kwds: Any) -> Axes: ...
def autocorrelation_plot(series: Series, ax: Axes | None = ..., **kwargs: Any) -> Axes: ...

class _Options(dict[str, Any]):
    _ALIASES: dict[str, str]
    _DEFAULT_KEYS: list[str]
    def __init__(self) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def reset(self) -> None: ...
    def _get_canonical_key(self, key: str) -> str: ...
    def use(self, key: str, value: Any) -> ContextManager[_Options]: ...

plot_params: _Options = ...