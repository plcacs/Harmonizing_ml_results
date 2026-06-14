from typing import Any, Mapping, Optional
from collections.abc import Hashable, Sequence
import numpy as np
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
    marker: Hashable = ...,
    density_kwds: Mapping[str, Any] | None = ...,
    hist_kwds: Mapping[str, Any] | None = ...,
    range_padding: float = ...,
    **kwds: Any
) -> Sequence[Sequence[Axes]]: ...

def _get_marker_compat(marker: Hashable) -> Hashable: ...

def radviz(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = ...,
    color: Any = ...,
    colormap: Any = ...,
    **kwds: Any
) -> Axes: ...

def andrews_curves(
    frame: DataFrame,
    class_column: Hashable,
    ax: Axes | None = ...,
    samples: int = ...,
    color: Any = ...,
    colormap: Any = ...,
    **kwds: Any
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
    class_column: Hashable,
    cols: Sequence[Hashable] | None = ...,
    ax: Axes | None = ...,
    color: Any = ...,
    use_columns: bool = ...,
    xticks: Sequence[int | float] | None = ...,
    colormap: Any = ...,
    axvlines: bool = ...,
    axvlines_kwds: Mapping[str, Any] | None = ...,
    sort_labels: bool = ...,
    **kwds: Any
) -> Axes: ...

def lag_plot(
    series: Series,
    lag: int = ...,
    ax: Axes | None = ...,
    **kwds: Any
) -> Axes: ...

def autocorrelation_plot(
    series: Sequence[float] | np.ndarray | Series,
    ax: Axes | None = ...,
    **kwds: Any
) -> Axes: ...

def unpack_single_str_list(keys: object) -> object: ...