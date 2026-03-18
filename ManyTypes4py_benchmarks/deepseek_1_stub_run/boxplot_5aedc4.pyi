```python
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    overload,
    Union,
    Sequence,
    Mapping,
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
    from collections.abc import Collection
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from pandas._typing import MatplotlibColor
    import numpy.typing as npt

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
        data: Any,
        return_type: Literal["axes", "dict", "both"] = "axes",
        **kwargs: Any,
    ) -> None: ...
    
    @property
    def _kind(self) -> str: ...
    
    @classmethod
    def _plot(
        cls,
        ax: Axes,
        y: npt.NDArray[Any],
        column_num: int | None = None,
        return_type: Literal["axes", "dict", "both"] = "axes",
        **kwds: Any,
    ) -> tuple[Any, dict[str, list[Line2D]]]: ...
    
    def _validate_color_args(
        self,
        color: Any,
        colormap: Any,
    ) -> Any: ...
    
    @cache_readonly
    def _color_attrs(self) -> list[Any]: ...
    
    @cache_readonly
    def _boxes_c(self) -> Any: ...
    
    @cache_readonly
    def _whiskers_c(self) -> Any: ...
    
    @cache_readonly
    def _medians_c(self) -> Any: ...
    
    @cache_readonly
    def _caps_c(self) -> Any: ...
    
    def _get_colors(
        self,
        num_colors: int | None = None,
        color_kwds: str = "color",
    ) -> None: ...
    
    def maybe_color_bp(self, bp: dict[str, list[Line2D]]) -> None: ...
    
    def _make_plot(self, fig: Figure) -> None: ...
    
    def _make_legend(self) -> None: ...
    
    def _post_plot_logic(self, ax: Axes, data: Any) -> None: ...
    
    @property
    def orientation(self) -> Literal["vertical", "horizontal"]: ...
    
    @property
    def result(self) -> Any: ...

def maybe_color_bp(
    bp: dict[str, list[Line2D]],
    color_tup: tuple[Any, Any, Any, Any],
    **kwds: Any,
) -> None: ...

def _grouped_plot_by_column(
    plotf: Any,
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    by: str | Sequence[str] | None = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    layout: tuple[int, int] | None = None,
    return_type: Literal["axes", "dict", "both"] | None = None,
    **kwargs: Any,
) -> pd.Series[Any]: ...

@overload
def boxplot(
    data: pd.DataFrame,
    column: str | Sequence[str] | None = ...,
    by: str | Sequence[str] | None = ...,
    ax: Axes | None = ...,
    fontsize: int | None = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    return_type: Literal["axes", "dict", "both"] | None = ...,
    **kwds: Any,
) -> Axes | dict[str, list[Line2D]] | BoxPlot.BP: ...

@overload
def boxplot(
    data: pd.Series,
    column: str | Sequence[str] | None = ...,
    by: str | Sequence[str] | None = ...,
    ax: Axes | None = ...,
    fontsize: int | None = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    return_type: Literal["axes", "dict", "both"] | None = ...,
    **kwds: Any,
) -> Axes | dict[str, list[Line2D]] | BoxPlot.BP: ...

def boxplot(
    data: pd.DataFrame | pd.Series,
    column: str | Sequence[str] | None = None,
    by: str | Sequence[str] | None = None,
    ax: Axes | None = None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: Literal["axes", "dict", "both"] | None = None,
    **kwds: Any,
) -> Axes | dict[str, list[Line2D]] | BoxPlot.BP: ...

def boxplot_frame(
    self: pd.DataFrame,
    column: str | Sequence[str] | None = None,
    by: str | Sequence[str] | None = None,
    ax: Axes | None = None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: Literal["axes", "dict", "both"] | None = None,
    **kwds: Any,
) -> Axes: ...

def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: str | Sequence[str] | None = None,
    fontsize: int | None = None,
    rot: int = 0,
    grid: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any,
) -> pd.Series[Any]: ...
```