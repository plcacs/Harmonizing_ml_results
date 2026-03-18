```python
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.table import Table
    import numpy as np
    from pandas import DataFrame, Series

def table(ax: Axes, data: DataFrame | Series, **kwargs: Any) -> Table: ...
def register() -> None: ...
def deregister() -> None: ...
def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds: dict[str, Any] | None = None,
    hist_kwds: dict[str, Any] | None = None,
    range_padding: float = 0.05,
    **kwargs: Any,
) -> np.ndarray: ...
def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = None,
    color: list[str] | tuple[str, ...] | None = None,
    colormap: str | Colormap | None = None,
    **kwds: Any,
) -> Axes: ...
def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = None,
    samples: int = 200,
    color: str | list[str] | tuple[str, ...] | None = None,
    colormap: str | Colormap | None = None,
    **kwargs: Any,
) -> Axes: ...
def bootstrap_plot(
    series: Series,
    fig: Figure | None = None,
    size: int = 50,
    samples: int = 500,
    **kwds: Any,
) -> Figure: ...
def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: list[str] | None = None,
    ax: Axes | None = None,
    color: list[str] | tuple[str, ...] | None = None,
    use_columns: bool = False,
    xticks: list[Any] | tuple[Any, ...] | None = None,
    colormap: str | Colormap | None = None,
    axvlines: bool = True,
    axvlines_kwds: dict[str, Any] | None = None,
    sort_labels: bool = False,
    **kwargs: Any,
) -> Axes: ...
def lag_plot(
    series: Series,
    lag: int = 1,
    ax: Axes | None = None,
    **kwds: Any,
) -> Axes: ...
def autocorrelation_plot(
    series: Series,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes: ...

class _Options(dict[str, Any]):
    _ALIASES: dict[str, str]
    _DEFAULT_KEYS: list[str]
    def __init__(self) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def reset(self) -> None: ...
    def _get_canonical_key(self, key: str) -> str: ...
    @contextmanager
    def use(self, key: str, value: Any) -> Generator[_Options, None, None]: ...

plot_params: _Options
```