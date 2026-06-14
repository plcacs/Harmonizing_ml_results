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
    density_kwds: dict[str, object] | None = ...,
    hist_kwds: dict[str, object] | None = ...,
    range_padding: float = ...,
    **kwds: object,
) -> np.ndarray: ...

def _get_marker_compat(marker: str) -> str: ...

def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = ...,
    color: str | list[str] | Sequence[str] | None = ...,
    colormap: object | None = ...,
    **kwds: object,
) -> Axes: ...

def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: Axes | None = ...,
    samples: int = ...,
    color: str | list[str] | Sequence[str] | None = ...,
    colormap: object | None = ...,
    **kwds: object,
) -> Axes: ...

def bootstrap_plot(
    series: Series,
    fig: Figure | None = ...,
    size: int = ...,
    samples: int = ...,
    **kwds: object,
) -> Figure: ...

def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: list[str] | None = ...,
    ax: Axes | None = ...,
    color: str | list[str] | Sequence[str] | None = ...,
    use_columns: bool = ...,
    xticks: list[float] | None = ...,
    colormap: object | None = ...,
    axvlines: bool = ...,
    axvlines_kwds: dict[str, object] | None = ...,
    sort_labels: bool = ...,
    **kwds: object,
) -> Axes: ...

def lag_plot(
    series: Series,
    lag: int = ...,
    ax: Axes | None = ...,
    **kwds: object,
) -> Axes: ...

def autocorrelation_plot(
    series: Series,
    ax: Axes | None = ...,
    **kwds: object,
) -> Axes: ...

def unpack_single_str_list(keys: list[Hashable] | Hashable) -> list[Hashable] | Hashable: ...