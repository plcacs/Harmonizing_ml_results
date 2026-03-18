```python
from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING
import matplotlib as mpl
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Index, Series

def scatter_matrix(
    frame: Any,
    alpha: float = ...,
    figsize: Any = ...,
    ax: Any = ...,
    grid: bool = ...,
    diagonal: Literal["hist", "kde", "density"] = ...,
    marker: str = ...,
    density_kwds: dict[str, Any] | None = ...,
    hist_kwds: dict[str, Any] | None = ...,
    range_padding: float = ...,
    **kwds: Any
) -> Any: ...

def _get_marker_compat(marker: str) -> str: ...

def radviz(
    frame: Any,
    class_column: Hashable,
    ax: Any = ...,
    color: Any = ...,
    colormap: Any = ...,
    **kwds: Any
) -> Any: ...

def andrews_curves(
    frame: Any,
    class_column: Hashable,
    ax: Any = ...,
    samples: int = ...,
    color: Any = ...,
    colormap: Any = ...,
    **kwds: Any
) -> Any: ...

def bootstrap_plot(
    series: Any,
    fig: Any = ...,
    size: int = ...,
    samples: int = ...,
    **kwds: Any
) -> Any: ...

def parallel_coordinates(
    frame: Any,
    class_column: Hashable,
    cols: Any = ...,
    ax: Any = ...,
    color: Any = ...,
    use_columns: bool = ...,
    xticks: Any = ...,
    colormap: Any = ...,
    axvlines: bool = ...,
    axvlines_kwds: dict[str, Any] | None = ...,
    sort_labels: bool = ...,
    **kwds: Any
) -> Any: ...

def lag_plot(
    series: Any,
    lag: int = ...,
    ax: Any = ...,
    **kwds: Any
) -> Any: ...

def autocorrelation_plot(
    series: Any,
    ax: Any = ...,
    **kwds: Any
) -> Any: ...

def unpack_single_str_list(keys: Any) -> Any: ...
```