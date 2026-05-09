from __future__ import annotations
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from typing import NamedTuple, TypeVar, overload
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas._typing import MatplotlibColor
from pandas.core.dtypes.generic import ABCSeries
from pandas.plotting._matplotlib.core import LinePlot

T = TypeVar('T')

def _set_ticklabels(ax: Axes, labels: List[str], is_vertical: bool, **kwargs: Any) -> None: ...

class BoxPlot(LinePlot):
    _kind: str
    _layout_type: str
    _valid_return_types: Tuple[Optional[str], ...]
    BP: NamedTuple

    def __init__(self, data: Any, return_type: str, **kwargs: Any) -> None: ...

    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, column_num: int, return_type: str, **kwds: Any) -> Tuple[Any, Any]: ...

    def _validate_color_args(self, color: Any, colormap: Any) -> Any: ...

    @property
    def _color_attrs(self) -> Tuple[MatplotlibColor, ...]: ...

    @property
    def _boxes_c(self) -> MatplotlibColor: ...
    @property
    def _whiskers_c(self) -> MatplotlibColor: ...
    @property
    def _medians_c(self) -> MatplotlibColor: ...
    @property
    def _caps_c(self) -> MatplotlibColor: ...

    def _get_colors(self, num_colors: Optional[int] = None, color_kwds: str = 'color') -> None: pass

    def maybe_color_bp(self, bp: Any) -> None: ...

    def _make_plot(self, fig: Figure) -> None: ...

    def _make_legend(self) -> None: pass

    def _post_plot_logic(self, ax: Axes, data: Any) -> None: ...

    @property
    def orientation(self) -> str: ...

    @property
    def result(self) -> Union[Axes, pd.Series[Axes]]: ...

def maybe_color_bp(bp: Any, color_tup: Tuple[MatplotlibColor, MatplotlibColor, MatplotlibColor, MatplotlibColor], **kwds: Any) -> None: ...

def _grouped_plot_by_column(
    plotf: Callable,
    data: Any,
    columns: Optional[List[str]] = None,
    by: Optional[Any] = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[str] = None,
    **kwargs: Any
) -> pd.Series[Axes]: ...

def boxplot(
    data: Any,
    column: Optional[Union[str, List[str]]] = None,
    by: Optional[Any] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    layout: Optional[Tuple[int, int]] = None,
    return_type: Optional[str] = None,
    **kwds: Any
) -> Union[Axes, Dict[str, Any], BoxPlot.BP]: ...

def boxplot_frame(self: ABCSeries, **kwargs: Any) -> Axes: ...

def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: Optional[Any] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    layout: Optional[Tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any
) -> Union[pd.Series[Axes], Axes]: ...