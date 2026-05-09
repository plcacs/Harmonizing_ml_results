from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal, NamedTuple, Sequence, TYPE_CHECKING
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.colors import Color
from pandas._typing import MatplotlibColor
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike
from pandas.core.common import flatten_axes
from pandas.plotting._matplotlib.core import LinePlot, MPLPlot
from pandas.plotting._matplotlib.style import get_standard_colors

if TYPE_CHECKING:
    from collections.abc import Collection
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from pandas import DataFrame, Series
    from pandas.core.groupby import GroupBy

class BoxPlot(LinePlot):
    _kind: str
    _layout_type: str
    _valid_return_types: Tuple[Optional[str], ...]
    
    class BP(NamedTuple):
        ax: Axes
        lines: Dict[str, List[Line2D]]
    
    def __init__(self, data: Any, return_type: Optional[str] = 'axes', **kwargs: Any) -> None:
        ...
    
    @classmethod
    def _plot(cls, ax: Axes, y: Union[ABCSeries, np.ndarray], column_num: Optional[int] = None, return_type: Optional[str] = 'axes', **kwds: Any) -> Tuple[Axes, Dict[str, List[Line2D]]]:
        ...
    
    def _validate_color_args(self, color: Any, colormap: Any) -> None:
        ...
    
    @cache_readonly
    def _color_attrs(self) -> List[MatplotlibColor]:
        ...
    
    @cache_readonly
    def _boxes_c(self) -> MatplotlibColor:
        ...
    
    @cache_readonly
    def _whiskers_c(self) -> MatplotlibColor:
        ...
    
    @cache_readonly
    def _medians_c(self) -> MatplotlibColor:
        ...
    
    @cache_readonly
    def _caps_c(self) -> MatplotlibColor:
        ...
    
    def _get_colors(self, num_colors: Optional[int] = None, color_kwds: str = 'color') -> None:
        ...
    
    def maybe_color_bp(self, bp: Dict[str, List[Line2D]]) -> None:
        ...
    
    def _make_plot(self, fig: Figure) -> None:
        ...
    
    def _make_legend(self) -> None:
        ...
    
    def _post_plot_logic(self, ax: Axes, data: Any) -> None:
        ...
    
    @property
    def orientation(self) -> str:
        ...
    
    @property
    def result(self) -> Any:
        ...

def _set_ticklabels(ax: Axes, labels: List[str], is_vertical: bool, **kwargs: Any) -> None:
    ...

def maybe_color_bp(bp: Dict[str, List[Line2D]], color_tup: Tuple[MatplotlibColor, ...], **kwds: Any) -> None:
    ...

def _grouped_plot_by_column(
    plotf: Callable,
    data: DataFrame,
    columns: Optional[List[str]] = None,
    by: Optional[Any] = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    layout: Optional[Any] = None,
    return_type: Optional[str] = None,
    **kwargs: Any
) -> Series:
    ...

def boxplot(
    data: Any,
    column: Optional[Any] = None,
    by: Optional[Any] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[float] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    layout: Optional[Any] = None,
    return_type: Optional[str] = None,
    **kwds: Any
) -> Union[Axes, Dict[str, List[Line2D]], Tuple[Axes, Dict[str, List[Line2D]]]]:
    ...

def boxplot_frame(self: DataFrame, column: Optional[Any] = None, by: Optional[Any] = None, ax: Optional[Axes] = None, fontsize: Optional[float] = None, rot: int = 0, grid: bool = True, figsize: Optional[Tuple[int, int]] = None, layout: Optional[Any] = None, return_type: Optional[str] = None, **kwds: Any) -> Axes:
    ...

def boxplot_frame_groupby(
    grouped: GroupBy,
    subplots: bool = True,
    column: Optional[Any] = None,
    fontsize: Optional[float] = None,
    rot: int = 0,
    grid: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    layout: Optional[Any] = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any
) -> Union[Series, Axes]:
    ...