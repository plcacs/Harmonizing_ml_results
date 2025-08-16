from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from pandas.plotting._core import _get_plot_backend

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.table import Table
    import numpy as np
    from pandas import DataFrame, Series

def table(ax: Axes, data: DataFrame or Series, **kwargs: Any) -> Table:
    ...

def register() -> None:
    ...

def deregister() -> None:
    ...

def scatter_matrix(frame: DataFrame, alpha: float = 0.5, figsize: tuple[float, float] = None, ax: Axes = None, grid: bool = False, diagonal: str = 'hist', marker: str = '.', density_kwds: dict = None, hist_kwds: dict = None, range_padding: float = 0.05, **kwargs: Any) -> np.ndarray:
    ...

def radviz(frame: DataFrame, class_column: str, ax: Axes = None, color: list[str] = None, colormap: Colormap = None, **kwds: Any) -> Axes:
    ...

def andrews_curves(frame: DataFrame, class_column: str, ax: Axes = None, samples: int = 200, color: list[str] = None, colormap: Colormap = None, **kwargs: Any) -> Axes:
    ...

def bootstrap_plot(series: Series, fig: Figure = None, size: int = 50, samples: int = 500, **kwds: Any) -> Figure:
    ...

def parallel_coordinates(frame: DataFrame, class_column: str, cols: list = None, ax: Axes = None, color: list or tuple = None, use_columns: bool = False, xticks: list or tuple = None, colormap: Colormap = None, axvlines: bool = True, axvlines_kwds: dict = None, sort_labels: bool = False, **kwargs: Any) -> Axes:
    ...

def lag_plot(series: Series, lag: int = 1, ax: Axes = None, **kwds: Any) -> Axes:
    ...

def autocorrelation_plot(series: Series, ax: Axes = None, **kwargs: Any) -> Axes:
    ...

class _Options(dict):
    def __init__(self) -> None:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    def __contains__(self, key: str) -> bool:
        ...

    def reset(self) -> None:
        ...

    def _get_canonical_key(self, key: str) -> str:
        ...

    @contextmanager
    def use(self, key: str, value: Any):
        ...
