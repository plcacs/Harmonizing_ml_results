from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Sequence, overload
from collections.abc import Hashable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Index, Series

def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    grid: bool = False,
    diagonal: str = 'hist',
    marker: str = '.',
    density_kwds: Optional[dict[str, Any]] = None,
    hist_kwds: Optional[dict[str, Any]] = None,
    range_padding: float = 0.05,
    **kwds: Any,
) -> Any: ...

def _get_marker_compat(marker: str) -> str: ...

def radviz(
    frame: DataFrame,
    class_column: Hashable,
    ax: Optional[Axes] = None,
    color: Optional[Union[str, Sequence[str]]] = None,
    colormap: Optional[str] = None,
    **kwds: Any,
) -> Axes: ...

def andrews_curves(
    frame: DataFrame,
    class_column: Hashable,
    ax: Optional[Axes] = None,
    samples: int = 200,
    color: Optional[Union[str, Sequence[str]]] = None,
    colormap: Optional[str] = None,
    **kwds: Any,
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
    class_column: Hashable,
    cols: Optional[Union[str, Sequence[str]]] = None,
    ax: Optional[Axes] = None,
    color: Optional[Union[str, Sequence[str]]] = None,
    use_columns: bool = False,
    xticks: Optional[Sequence[float]] = None,
    colormap: Optional[str] = None,
    axvlines: bool = True,
    axvlines_kwds: Optional[dict[str, Any]] = None,
    sort_labels: bool = False,
    **kwds: Any,
) -> Axes: ...

def lag_plot(
    series: Series,
    lag: int = 1,
    ax: Optional[Axes] = None,
    **kwds: Any,
) -> Axes: ...

def autocorrelation_plot(
    series: Series,
    ax: Optional[Axes] = None,
    **kwds: Any,
) -> Axes: ...

def unpack_single_str_list(keys: Union[str, list[str]]) -> Union[str, list[str]]: ...