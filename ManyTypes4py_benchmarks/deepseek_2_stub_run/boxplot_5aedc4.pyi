```python
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Sequence,
    overload,
    Union,
    Optional,
    Tuple,
    Dict,
    List,
)
import matplotlib as mpl
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Collection
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from pandas._typing import MatplotlibColor
    import pandas.core.groupby.generic

def _set_ticklabels(
    ax: Any,
    labels: Any,
    is_vertical: bool,
    **kwargs: Any
) -> None: ...

class BoxPlot(LinePlot):
    class BP(NamedTuple):
        ax: Any
        lines: Any
    
    _layout_type: str
    _valid_return_types: Tuple[Any, ...]
    
    def __init__(
        self,
        data: Any,
        return_type: str = "axes",
        **kwargs: Any
    ) -> None: ...
    
    @property
    def _kind(self) -> str: ...
    
    @classmethod
    def _plot(
        cls,
        ax: Any,
        y: Any,
        column_num: Optional[int] = None,
        return_type: str = "axes",
        **kwds: Any
    ) -> Tuple[Any, Any]: ...
    
    def _validate_color_args(
        self,
        color: Any,
        colormap: Any
    ) -> Any: ...
    
    @property
    def _color_attrs(self) -> Any: ...
    
    @property
    def _boxes_c(self) -> Any: ...
    
    @property
    def _whiskers_c(self) -> Any: ...
    
    @property
    def _medians_c(self) -> Any: ...
    
    @property
    def _caps_c(self) -> Any: ...
    
    def _get_colors(
        self,
        num_colors: Optional[int] = None,
        color_kwds: str = "color"
    ) -> None: ...
    
    def maybe_color_bp(self, bp: Any) -> None: ...
    
    def _make_plot(self, fig: Any) -> None: ...
    
    def _make_legend(self) -> None: ...
    
    def _post_plot_logic(self, ax: Any, data: Any) -> None: ...
    
    @property
    def orientation(self) -> str: ...
    
    @property
    def result(self) -> Any: ...

def maybe_color_bp(
    bp: Any,
    color_tup: Any,
    **kwds: Any
) -> None: ...

def _grouped_plot_by_column(
    plotf: Any,
    data: Any,
    columns: Optional[Any] = None,
    by: Optional[Any] = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Optional[Any] = None,
    ax: Optional[Any] = None,
    layout: Optional[Any] = None,
    return_type: Optional[str] = None,
    **kwargs: Any
) -> Any: ...

def boxplot(
    data: Any,
    column: Optional[Any] = None,
    by: Optional[Any] = None,
    ax: Optional[Any] = None,
    fontsize: Optional[Any] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Any] = None,
    layout: Optional[Any] = None,
    return_type: Optional[str] = None,
    **kwds: Any
) -> Any: ...

def boxplot_frame(
    self: Any,
    column: Optional[Any] = None,
    by: Optional[Any] = None,
    ax: Optional[Any] = None,
    fontsize: Optional[Any] = None,
    rot: int = 0,
    grid: bool = True,
    figsize: Optional[Any] = None,
    layout: Optional[Any] = None,
    return_type: Optional[str] = None,
    **kwds: Any
) -> Any: ...

def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: Optional[Any] = None,
    fontsize: Optional[Any] = None,
    rot: int = 0,
    grid: bool = True,
    ax: Optional[Any] = None,
    figsize: Optional[Any] = None,
    layout: Optional[Any] = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any
) -> Any: ...
```