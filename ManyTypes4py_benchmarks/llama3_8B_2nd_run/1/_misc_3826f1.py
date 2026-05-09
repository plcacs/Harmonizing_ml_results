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

def table(ax: Axes, data: DataFrame | Series, **kwargs: Any) -> Table:
    ...

def register() -> None:
    ...

def deregister() -> None:
    ...

def scatter_matrix(frame: DataFrame, alpha: float = 0.5, figsize: tuple[float, float] | None = None, ax: Axes | None = None, grid: bool = False, diagonal: str = 'hist', marker: str = '.', density_kwds: dict[str, Any] | None = None, hist_kwds: dict[str, Any] | None = None, range_padding: float = 0.05, **kwargs: Any) -> numpy.ndarray:
    ...

def radviz(frame: DataFrame, class_column: str, ax: Axes | None = None, color: list[str] | tuple[str] | None = None, colormap: str | Colormap | None = None, **kwargs: Any) -> Axes:
    ...

def andrews_curves(frame: DataFrame, class_column: str, ax: Axes | None = None, samples: int = 200, color: list[str] | tuple[str] | None = None, colormap: str | Colormap | None = None, **kwargs: Any) -> Axes:
    ...

def bootstrap_plot(series: Series, fig: Figure | None = None, size: int = 50, samples: int = 500, **kwargs: Any) -> Figure:
    ...

def parallel_coordinates(frame: DataFrame, class_column: str, cols: list[str] | None = None, ax: Axes | None = None, color: list[str] | tuple[str] | None = None, use_columns: bool = False, xticks: list[str] | tuple[str] | None = None, colormap: str | Colormap | None = None, axvlines: bool = True, axvlines_kwds: dict[str, Any] | None = None, sort_labels: bool = False, **kwargs: Any) -> Axes:
    ...

def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwargs: Any) -> Axes:
    ...

def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwargs: Any) -> Axes:
    ...

class _Options(dict):
    _ALIASES: dict[str, str] = {'x_compat': 'xaxis.compat'}
    _DEFAULT_KEYS: list[str] = ['xaxis.compat']

    def __init__(self) -> None:
        super().__setitem__('xaxis.compat', False)

    def __getitem__(self, key: str) -> Any:
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(f'{key} is not a valid pandas plotting option')
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        key = self._get_canonical_key(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        key = self._get_canonical_key(key)
        if key in self._DEFAULT_KEYS:
            raise ValueError(f'Cannot remove default parameter {key}')
        super().__delitem__(key)

    def __contains__(self, key: str) -> bool:
        key = self._get_canonical_key(key)
        return super().__contains__(key)

    def reset(self) -> None:
        self.__init__()

    def _get_canonical_key(self, key: str) -> str:
        return self._ALIASES.get(key, key)

    @contextmanager
    def use(self, key: str, value: Any) -> None:
        old_value = self[key]
        try:
            self[key] = value
            yield self
        finally:
            self[key] = old_value

plot_params = _Options()
