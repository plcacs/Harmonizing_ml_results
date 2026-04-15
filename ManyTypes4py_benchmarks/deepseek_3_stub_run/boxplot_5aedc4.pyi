from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    overload,
)
import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.plotting._matplotlib.core import LinePlot, MPLPlot
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.tools import create_subplots, flatten_axes

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from pandas._typing import MatplotlibColor
    from pandas.core.groupby.generic import DataFrameGroupBy

def _set_ticklabels(
    ax: Axes,
    labels: Sequence[str],
    is_vertical: bool,
    **kwargs: Any,
) -> None: ...

class BoxPlot(LinePlot):
    _layout_type: str
    _valid_return_types: tuple[None, Literal["axes", "dict", "both"]]
    
    class BP(NamedTuple):
        ax: Axes
        lines: dict[str, list[Line2D]]
    
    def __init__(
        self,
        data: pd.DataFrame,
        return_type: Literal["axes", "dict", "both"] = "axes",
        **kwargs: Any,
    ) -> None: ...
    
    @property
    def _kind(self) -> str: ...
    
    @classmethod
    def _plot(
        cls,
        ax: Axes,
        y: np.ndarray,
        column_num: Optional[int] = None,
        return_type: Literal["axes", "dict", "both"] = "axes",
        **kwds: Any,
    ) -> tuple[
        Union[Axes, dict[str, list[Line2D]], BoxPlot.BP],
        dict[str, list[Line2D]],
    ]: ...
    
    def _validate_color_args(
        self,
        color: Union[dict[str, str], lib.NoDefault],
        colormap: Optional[str],
    ) -> Optional[Union[dict[str, str], str]]: ...
    
    @cache_readonly
    def _color_attrs(self) -> list[str]: ...
    
    @cache_readonly
    def _boxes_c(self) -> str: ...
    
    @cache_readonly
    def _whiskers_c(self) -> str: ...
    
    @cache_readonly
    def _medians_c(self) -> str: ...
    
    @cache_readonly
    def _caps_c(self) -> str: ...
    
    def _get_colors(
        self,
        num_colors: Optional[int] = None,
        color_kwds: str = "color",
    ) -> None: ...
    
    def maybe_color_bp(self, bp: dict[str, list[Line2D]]) -> None: ...
    
    def _make_plot(self, fig: Figure) -> None: ...
    
    def _make_legend(self) -> None: ...
    
    def _post_plot_logic(self, ax: Axes, data: pd.DataFrame) -> None: ...
    
    @property
    def orientation(self) -> Literal["vertical", "horizontal"]: ...
    
    @property
    def result(self) -> Union[pd.Series[Any], Axes, dict[str, list[Line2D]], BoxPlot.BP]: ...

def maybe_color_bp(
    bp: dict[str, list[Line2D]],
    color_tup: tuple[str, str, str, str],
    **kwds: Any,
) -> None: ...

def _grouped_plot_by_column(
    plotf: Any,
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    layout: Optional[tuple[int, int]] = None,
    return_type: Optional[Literal["axes", "dict", "both"]] = None,
    **kwargs: Any,
) -> Union[pd.Series[Any], Axes]: ...

@overload
def boxplot(
    data: pd.DataFrame,
    column: None = None,
    by: None = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[tuple[int, int]] = None,
    return_type: Optional[Literal["axes", "dict", "both"]] = None,
    **kwds: Any,
) -> Axes: ...

@overload
def boxplot(
    data: pd.DataFrame,
    column: Union[str, Sequence[str]],
    by: None = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[tuple[int, int]] = None,
    return_type: Optional[Literal["axes", "dict", "both"]] = None,
    **kwds: Any,
) -> Axes: ...

@overload
def boxplot(
    data: pd.DataFrame,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Union[str, Sequence[str]],
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[tuple[int, int]] = None,
    return_type: Optional[Literal["axes", "dict", "both"]] = None,
    **kwds: Any,
) -> Union[pd.Series[Any], Axes]: ...

def boxplot(
    data: pd.DataFrame,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[tuple[int, int]] = None,
    return_type: Optional[Literal["axes", "dict", "both"]] = None,
    **kwds: Any,
) -> Union[pd.Series[Any], Axes, dict[str, list[Line2D]], BoxPlot.BP]: ...

def boxplot_frame(
    self: pd.DataFrame,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[tuple[int, int]] = None,
    return_type: Optional[Literal["axes", "dict", "both"]] = None,
    **kwds: Any,
) -> Axes: ...

def boxplot_frame_groupby(
    grouped: DataFrameGroupBy,
    subplots: bool = True,
    column: Optional[Union[str, Sequence[str]]] = None,
    fontsize: Optional[int] = None,
    rot: int = 0,
    grid: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[tuple[int, int]] = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any,
) -> Union[pd.Series[Any], Axes]: ...