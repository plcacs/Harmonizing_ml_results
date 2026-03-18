```pyi
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
import matplotlib as mpl
import numpy as np
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import ABCSeries
import pandas as pd
from pandas.plotting._matplotlib.core import LinePlot, MPLPlot

if TYPE_CHECKING:
    from collections.abc import Collection
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from pandas._typing import MatplotlibColor

def _set_ticklabels(ax: Axes, labels: list[Any], is_vertical: bool, **kwargs: Any) -> None: ...

class BoxPlot(LinePlot):
    _kind: str
    _layout_type: str
    _valid_return_types: tuple[None | Literal['axes'] | Literal['dict'] | Literal['both'], ...]
    return_type: None | Literal['axes'] | Literal['dict'] | Literal['both']

    class BP(NamedTuple):
        pass

    def __init__(self, data: Any, return_type: None | Literal['axes'] | Literal['dict'] | Literal['both'] = ..., **kwargs: Any) -> None: ...
    @classmethod
    def _plot(cls, ax: Axes, y: Any, column_num: int | None = ..., return_type: None | Literal['axes'] | Literal['dict'] | Literal['both'] = ..., **kwds: Any) -> tuple[Any, Any]: ...
    def _validate_color_args(self, color: Any, colormap: Any) -> Any: ...
    @cache_readonly
    def _color_attrs(self) -> Any: ...
    @cache_readonly
    def _boxes_c(self) -> Any: ...
    @cache_readonly
    def _whiskers_c(self) -> Any: ...
    @cache_readonly
    def _medians_c(self) -> Any: ...
    @cache_readonly
    def _caps_c(self) -> Any: ...
    def _get_colors(self, num_colors: int | None = ..., color_kwds: str = ...) -> None: ...
    def maybe_color_bp(self, bp: Any) -> None: ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _make_legend(self) -> None: ...
    def _post_plot_logic(self, ax: Axes, data: Any) -> None: ...
    @property
    def orientation(self) -> Literal['vertical', 'horizontal']: ...
    @property
    def result(self) -> Any: ...

def maybe_color_bp(bp: Any, color_tup: tuple[Any, Any, Any, Any], **kwds: Any) -> None: ...

def _grouped_plot_by_column(
    plotf: Any,
    data: Any,
    columns: Any = ...,
    by: Any = ...,
    numeric_only: bool = ...,
    grid: bool = ...,
    figsize: Any = ...,
    ax: Axes | None = ...,
    layout: Any = ...,
    return_type: None | Literal['axes'] | Literal['dict'] | Literal['both'] = ...,
    **kwargs: Any
) -> Any: ...

def boxplot(
    data: Any,
    column: Any = ...,
    by: Any = ...,
    ax: Axes | None = ...,
    fontsize: int | None = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: Any = ...,
    layout: Any = ...,
    return_type: None | Literal['axes'] | Literal['dict'] | Literal['both'] = ...,
    **kwds: Any
) -> Any: ...

def boxplot_frame(
    self: Any,
    column: Any = ...,
    by: Any = ...,
    ax: Axes | None = ...,
    fontsize: int | None = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: Any = ...,
    layout: Any = ...,
    return_type: None | Literal['axes'] | Literal['dict'] | Literal['both'] = ...,
    **kwds: Any
) -> Any: ...

def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = ...,
    column: Any = ...,
    fontsize: int | None = ...,
    rot: int = ...,
    grid: bool = ...,
    ax: Axes | None = ...,
    figsize: Any = ...,
    layout: Any = ...,
    sharex: bool = ...,
    sharey: bool = ...,
    **kwds: Any
) -> Any: ...
```