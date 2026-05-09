from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    overload,
    Tuple,
    Union,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pandas._typing import MatplotlibColor

if TYPE_CHECKING:
    from collections.abc import Collection
    from pandas._libs import lib
    from pandas.core.dtypes.generic import ABCSeries
    from pandas.plotting._matplotlib.core import LinePlot
    from pandas.plotting._matplotlib.style import get_standard_colors

class BoxPlot(LinePlot):
    _kind: str
    _layout_type: str
    _valid_return_types: Tuple[Optional[str], ...]
    BP: Callable[..., NamedTuple]

    def __init__(self, data: pd.DataFrame, return_type: Optional[str] = None, **kwargs: Any) -> None:
        ...

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        y: np.ndarray,
        column_num: Optional[int] = None,
        return_type: Optional[str] = None,
        **kwds: Any
    ) -> Union[Tuple[Axes, Dict[str, Any]], Tuple['BoxPlot.BP', Dict[str, Any]]]:
        ...

    def _validate_color_args(
        self, color: Union[Dict[str, MatplotlibColor], MatplotlibColor, None], colormap: Optional[str]
    ) -> Union[Dict[str, MatplotlibColor], MatplotlibColor, None]:
        ...

    @property
    def _color_attrs(self) -> List[MatplotlibColor]:
        ...

    @property
    def _boxes_c(self) -> MatplotlibColor:
        ...

    @property
    def _whiskers_c(self) -> MatplotlibColor:
        ...

    @property
    def _medians_c(self) -> MatplotlibColor:
        ...

    @property
    def _caps_c(self) -> MatplotlibColor:
        ...

    def _get_colors(self, num_colors: Optional[int] = None, color_kwds: str = 'color') -> None:
        ...

    def maybe_color_bp(self, bp: Dict[str, Any]) -> None:
        ...

    def _make_plot(self, fig: Figure) -> None:
        ...

    def _make_legend(self) -> None:
        ...

    def _post_plot_logic(self, ax: Axes, data: pd.DataFrame) -> None:
        ...

    @property
    def orientation(self) -> str:
        ...

    @property
    def result(self) -> Union[Axes, Dict[str, Any], pd.Series]:
        ...

def _set_ticklabels(
    ax: Axes,
    labels: List[str],
    is_vertical: bool,
    **kwargs: Any
) -> None:
    ...

def maybe_color_bp(
    bp: Dict[str, Any],
    color_tup: Tuple[MatplotlibColor, MatplotlibColor, MatplotlibColor, MatplotlibColor],
    **kwds: Any
) -> None:
    ...

def _grouped_plot_by_column(
    plotf: Callable[..., Any],
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    by: Optional[List[str]] = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[str] = None,
    **kwargs: Any
) -> pd.Series:
    ...

@overload
def boxplot(
    data: pd.DataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Literal['axes'] = ...,
    **kwds: Any
) -> Axes:
    ...

@overload
def boxplot(
    data: pd.DataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Literal['dict'],
    **kwds: Any
) -> Dict[str, Any]:
    ...

@overload
def boxplot(
    data: pd.DataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Literal['both'],
    **kwds: Any
) -> Tuple[Axes, Dict[str, Any]]:
    ...

def boxplot(
    data: Union[pd.DataFrame, ABCSeries],
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[str] = None,
    **kwds: Any
) -> Union[Axes, Dict[str, Any], Tuple[Axes, Dict[str, Any]]]:
    ...

def boxplot_frame(
    self: pd.DataFrame,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[str] = None,
    **kwds: Any
) -> Axes:
    ...

def boxplot_frame_groupby(
    grouped: pd.core.groupby.GroupBy,
    subplots: bool = True,
    column: Optional[Union[str, List[str]]] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    layout: Optional[Tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any
) -> pd.Series:
    ...