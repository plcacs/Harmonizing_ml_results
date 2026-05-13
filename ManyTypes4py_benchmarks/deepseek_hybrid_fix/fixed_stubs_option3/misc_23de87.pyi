from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Hashable, List, Literal, Sequence, overload
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Index, Series

def scatter_matrix(
    frame: DataFrame,
    alpha: float = ...,
    figsize: tuple[float, float] | None = ...,
    ax: Axes | None = ...,
    grid: bool = ...,
    diagonal: Literal['hist', 'kde', 'density'] = ...,
    marker: str = ...,
    density_kwds: dict[str, Any] | None = ...,
    hist_kwds: dict[str, Any] | None = ...,
    range_padding: float = ...,
    **kwds: Any,
) -> np.ndarray: ...

def _get_marker_compat(marker: str) -> str: ...

def radviz(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = ...,
    color: str | Sequence[str] | None = ...,
    colormap: str | mpl.colors.Colormap | None = ...,
    **kwds: Any,
) -> Axes: ...

def andrews_curves(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = ...,
    samples: int = ...,
    color: str | Sequence[str] | None = ...,
    colormap: str | mpl.colors.Colormap | None = ...,
    **kwds: Any,
) -> Axes: ...

def bootstrap_plot(
    series: Series,
    fig: Figure | None = ...,
    size: int = ...,
    samples: int = ...,
    **kwds: Any,
) -> Figure: ...

def parallel_coordinates(
    frame: DataFrame,
    class_column: Hashable,
    cols: Sequence[Hashable] | None = ...,
    ax: Axes | None = ...,
    color: str | Sequence[str] | None = ...,
    use_columns: bool = ...,
    xticks: Sequence[float] | None = ...,
    colormap: str | mpl.colors.Colormap | None = ...,
    axvlines: bool = ...,
    axvlines_kwds: dict[str, Any] | None = ...,
    sort_labels: bool = ...,
    **kwds: Any,
) -> Axes: ...

def lag_plot(
    series: Series,
    lag: int = ...,
    ax: Axes | None = ...,
    **kwds: Any,
) -> Axes: ...

def autocorrelation_plot(
    series: Series,
    ax: Axes | None = ...,
    **kwds: Any,
) -> Axes: ...

def unpack_single_str_list(keys: list[str] | str) -> list[str] | str: ...