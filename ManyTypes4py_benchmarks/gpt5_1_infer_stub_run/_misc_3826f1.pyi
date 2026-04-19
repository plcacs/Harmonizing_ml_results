from typing import Any, ClassVar, ContextManager, Mapping, Optional, Sequence, Tuple, Union, Literal
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.table import Table
from pandas import DataFrame, Series

def table(ax: Axes, data: Union[DataFrame, Series], **kwargs: Any) -> Table: ...
def register() -> None: ...
def deregister() -> None: ...
def scatter_matrix(
    frame: DataFrame,
    alpha: float = ...,
    figsize: Optional[Tuple[float, float]] = ...,
    ax: Optional[Axes] = ...,
    grid: bool = ...,
    diagonal: Literal["hist", "kde"] = ...,
    marker: str = ...,
    density_kwds: Optional[Mapping[str, Any]] = ...,
    hist_kwds: Optional[Mapping[str, Any]] = ...,
    range_padding: float = ...,
    **kwargs: Any
) -> np.ndarray: ...
def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Optional[Axes] = ...,
    color: Optional[Sequence[str]] = ...,
    colormap: Optional[Union[str, Colormap]] = ...,
    **kwds: Any
) -> Axes: ...
def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: Optional[Axes] = ...,
    samples: int = ...,
    color: Optional[Union[str, Sequence[str]]] = ...,
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
    cols: Optional[Sequence[str]] = ...,
    ax: Optional[Axes] = ...,
    color: Optional[Sequence[str]] = ...,
    use_columns: bool = ...,
    xticks: Optional[Sequence[Any]] = ...,
    colormap: Optional[Union[str, Colormap]] = ...,
    axvlines: bool = ...,
    axvlines_kwds: Optional[Mapping[str, Any]] = ...,
    sort_labels: bool = ...,
    **kwargs: Any
) -> Axes: ...
def lag_plot(series: Series, lag: int = ..., ax: Optional[Axes] = ..., **kwds: Any) -> Axes: ...
def autocorrelation_plot(series: Series, ax: Optional[Axes] = ..., **kwargs: Any) -> Axes: ...

class _Options(dict[str, Any]):
    _ALIASES: ClassVar[dict[str, str]]
    _DEFAULT_KEYS: ClassVar[Sequence[str]]
    def __init__(self) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def reset(self) -> None: ...
    def _get_canonical_key(self, key: str) -> str: ...
    def use(self, key: str, value: Any) -> ContextManager[_Options]: ...

plot_params: _Options = ...