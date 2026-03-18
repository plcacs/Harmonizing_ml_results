```python
from __future__ import annotations
from typing import Any, Literal
from typing_extensions import TypeAlias

if False:
    from collections.abc import Hashable
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame, Index, Series

_DataFrame: TypeAlias = Any
_Series: TypeAlias = Any
_Index: TypeAlias = Any
_Axes: TypeAlias = Any
_Figure: TypeAlias = Any

def scatter_matrix(
    frame: _DataFrame,
    alpha: float = ...,
    figsize: tuple[float, float] | None = ...,
    ax: _Axes | None = ...,
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
    frame: _DataFrame,
    class_column: str,
    ax: _Axes | None = ...,
    color: Any = ...,
    colormap: Any = ...,
    **kwds: Any
) -> _Axes: ...

def andrews_curves(
    frame: _DataFrame,
    class_column: str,
    ax: _Axes | None = ...,
    samples: int = ...,
    color: Any = ...,
    colormap: Any = ...,
    **kwds: Any
) -> _Axes: ...

def bootstrap_plot(
    series: _Series,
    fig: _Figure | None = ...,
    size: int = ...,
    samples: int = ...,
    **kwds: Any
) -> _Figure: ...

def parallel_coordinates(
    frame: _DataFrame,
    class_column: str,
    cols: list[str] | None = ...,
    ax: _Axes | None = ...,
    color: Any = ...,
    use_columns: bool = ...,
    xticks: list[float] | None = ...,
    colormap: Any = ...,
    axvlines: bool = ...,
    axvlines_kwds: dict[str, Any] | None = ...,
    sort_labels: bool = ...,
    **kwds: Any
) -> _Axes: ...

def lag_plot(
    series: _Series,
    lag: int = ...,
    ax: _Axes | None = ...,
    **kwds: Any
) -> _Axes: ...

def autocorrelation_plot(
    series: _Series,
    ax: _Axes | None = ...,
    **kwds: Any
) -> _Axes: ...

def unpack_single_str_list(keys: list[str] | str) -> str: ...
```