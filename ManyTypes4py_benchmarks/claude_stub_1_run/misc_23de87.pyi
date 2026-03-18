```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Index, Series

def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    grid: bool = False,
    diagonal: str = 'hist',
    marker: str = '.',
    density_kwds: dict[str, Any] | None = None,
    hist_kwds: dict[str, Any] | None = None,
    range_padding: float = 0.05,
    **kwds: Any,
) -> Any: ...
def _get_marker_compat(marker: str) -> str: ...
def radviz(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = None,
    color: str | None = None,
    colormap: str | None = None,
    **kwds: Any,
) -> Axes: ...
def andrews_curves(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = None,
    samples: int = 200,
    color: str | None = None,
    colormap: str | None = None,
    **kwds: Any,
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
    class_column: Hashable,
    cols: list[Hashable] | None = None,
    ax: Axes | None = None,
    color: str | None = None,
    use_columns: bool = False,
    xticks: list[Any] | None = None,
    colormap: str | None = None,
    axvlines: bool = True,
    axvlines_kwds: dict[str, Any] | None = None,
    sort_labels: bool = False,
    **kwds: Any,
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
    **kwds: Any,
) -> Axes: ...
def unpack_single_str_list(keys: Any) -> Any: ...
```