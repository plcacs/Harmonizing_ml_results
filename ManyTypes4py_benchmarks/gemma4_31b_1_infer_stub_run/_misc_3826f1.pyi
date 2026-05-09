from __future__ import annotations
from typing import Any, Generator, Optional, Union, overload
from collections.abc import Mapping
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.table import Table
import numpy as np
from pandas import DataFrame, Series

def table(ax: Axes, data: Union[DataFrame, Series], **kwargs: Any) -> Table: ...

def register() -> None: ...

def deregister() -> None: ...

def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    grid: bool = False,
    diagonal: literal['hist', 'kde'] = 'hist',
    marker: str = '.',
    density_kwds: Optional[Mapping[str, Any]] = None,
    hist_kwds: Optional[Mapping[str, Any]] = None,
    range_padding: float = 0.05,
    **kwargs: Any,
) -> np.ndarray: ...

def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Optional[Axes] = None,
    color: Optional[Union[list[str], tuple[str, ...]]] = None,
    colormap: Optional[Union[str, Colormap]] = None,
    **kwds: Any,
) -> Axes: ...

def andrews_curves(
    frame: DataFrame,
    class_column: Any,
    ax: Optional[Axes] = None,
    samples: int = 200,
    color: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    colormap: Optional[Union[str, Colormap]] = None,
    **kwargs: Any,
) -> Axes: ...

def bootstrap_plot(
    series: Series,
    fig: Optional[Figure] = None,
    size: int = 50,
    samples: int = 500,
    **kwds: Any,
) -> Figure: ...

def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: Optional[list[Any]] = None,
    ax: Optional[Axes] = None,
    color: Optional[Union[list[Any], tuple[Any, ...]]] = None,
    use_columns: bool = False,
    xticks: Optional[Union[list[Any], tuple[Any, ...]]] = None,
    colormap: Optional[Union[str, Colormap]] = None,
    axvlines: bool = True,
    axvlines_kwds: Optional[Mapping[str, Any]] = None,
    sort_labels: bool = False,
    **kwargs: Any,
) -> Axes: ...

def lag_plot(series: Series, lag: int = 1, ax: Optional[Axes] = None, **kwds: Any) -> Axes: ...

def autocorrelation_plot(series: Series, ax: Optional[Axes] = None, **kwargs: Any) -> Axes: ...

class _Options(dict[str, Any]):
    _ALIASES: dict[str, str]
    _DEFAULT_KEYS: list[str]

    def reset(self) -> None: ...

    def _get_canonical_key(self, key: str) -> str: ...

    def use(self, key: str, value: Any) -> Generator[_Options, None, None]: ...

plot_params: _Options = ...