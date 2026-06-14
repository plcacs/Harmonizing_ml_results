from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Series


def scatter_matrix(
    frame: DataFrame,
    alpha: float = ...,
    figsize: tuple[float, float] | None = ...,
    ax: Axes | None = ...,
    grid: bool = ...,
    diagonal: str = ...,
    marker: str = ...,
    density_kwds: dict | None = ...,
    hist_kwds: dict | None = ...,
    range_padding: float = ...,
    **kwds,
) -> np.ndarray: ...

def _get_marker_compat(marker: str) -> str: ...

def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = ...,
    color: list[str] | tuple[str, ...] | None = ...,
    colormap: str | None = ...,
    **kwds,
) -> Axes: ...

def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = ...,
    samples: int = ...,
    color: list[str] | tuple[str, ...] | None = ...,
    colormap: str | None = ...,
    **kwds,
) -> Axes: ...

def bootstrap_plot(
    series: Series,
    fig: Figure | None = ...,
    size: int = ...,
    samples: int = ...,
    **kwds,
) -> Figure: ...

def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: list[str] | None = ...,
    ax: Axes | None = ...,
    color: list[str] | tuple[str, ...] | None = ...,
    use_columns: bool = ...,
    xticks: Sequence[float] | None = ...,
    colormap: str | None = ...,
    axvlines: bool = ...,
    axvlines_kwds: dict | None = ...,
    sort_labels: bool = ...,
    **kwds,
) -> Axes: ...

def lag_plot(
    series: Series,
    lag: int = ...,
    ax: Axes | None = ...,
    **kwds,
) -> Axes: ...

def autocorrelation_plot(
    series: Series,
    ax: Axes | None = ...,
    **kwds,
) -> Axes: ...

def unpack_single_str_list(keys: list[Hashable] | Hashable) -> list[Hashable] | Hashable: ...