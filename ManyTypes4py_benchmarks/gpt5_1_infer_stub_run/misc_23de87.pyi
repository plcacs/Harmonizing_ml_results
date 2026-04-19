from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal, Optional, TypeVar, Union

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pandas import DataFrame, Series


def scatter_matrix(
    frame: DataFrame,
    alpha: float = ...,
    figsize: Optional[tuple[float, float]] = ...,
    ax: Optional[Union[Axes, Sequence[Axes], NDArray[Any]]] = ...,
    grid: bool = ...,
    diagonal: Literal["hist", "kde", "density"] = ...,
    marker: Hashable = ...,
    density_kwds: Optional[dict[str, Any]] = ...,
    hist_kwds: Optional[dict[str, Any]] = ...,
    range_padding: float = ...,
    **kwds: Any
) -> NDArray[Any]: ...


def _get_marker_compat(marker: Hashable) -> Hashable: ...


def radviz(
    frame: DataFrame,
    class_column: Hashable,
    ax: Optional[Axes] = ...,
    color: Optional[Union[str, Sequence[str]]] = ...,
    colormap: Optional[Union[Colormap, str]] = ...,
    **kwds: Any
) -> Axes: ...


def andrews_curves(
    frame: DataFrame,
    class_column: Hashable,
    ax: Optional[Axes] = ...,
    samples: int = ...,
    color: Optional[Union[str, Sequence[str]]] = ...,
    colormap: Optional[Union[Colormap, str]] = ...,
    **kwds: Any
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
    class_column: Hashable,
    cols: Optional[Sequence[Hashable]] = ...,
    ax: Optional[Axes] = ...,
    color: Optional[Union[str, Sequence[str]]] = ...,
    use_columns: bool = ...,
    xticks: Optional[Sequence[Union[float, int]]] = ...,
    colormap: Optional[Union[Colormap, str]] = ...,
    axvlines: bool = ...,
    axvlines_kwds: Optional[dict[str, Any]] = ...,
    sort_labels: bool = ...,
    **kwds: Any
) -> Axes: ...


def lag_plot(
    series: Series,
    lag: int = ...,
    ax: Optional[Axes] = ...,
    **kwds: Any
) -> Axes: ...


def autocorrelation_plot(
    series: Union[Series, Sequence[float], NDArray[Any]],
    ax: Optional[Axes] = ...,
    **kwds: Any
) -> Axes: ...


T = TypeVar("T")
def unpack_single_str_list(keys: Union[list[T], T]) -> Union[T, list[T]]: ...