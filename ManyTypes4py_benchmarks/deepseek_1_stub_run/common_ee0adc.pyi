```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, overload
import numpy as np
import pandas as pd
from pandas import Series
import pandas._testing as tm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist
    from matplotlib.collections import Collection, LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

def _check_legend_labels(
    axes: Axes | Sequence[Axes],
    labels: Sequence[str] | None = ...,
    visible: bool = ...
) -> None: ...

def _check_legend_marker(
    ax: Axes,
    expected_markers: Sequence[Any] | None = ...,
    visible: bool = ...
) -> None: ...

def _check_data(xp: Axes, rs: Axes) -> None: ...

def _check_visible(
    collections: Artist | Sequence[Artist],
    visible: bool = ...
) -> None: ...

def _check_patches_all_filled(
    axes: Axes | Sequence[Axes],
    filled: bool = ...
) -> None: ...

def _get_colors_mapped(series: Series, colors: Sequence[Any]) -> list[Any]: ...

def _check_colors(
    collections: Sequence[Artist],
    linecolors: Sequence[Any] | None = ...,
    facecolors: Sequence[Any] | None = ...,
    mapping: Series | None = ...
) -> None: ...

def _check_text_labels(
    texts: Text | Sequence[Text],
    expected: str | Sequence[str]
) -> None: ...

def _check_ticks_props(
    axes: Axes | Sequence[Axes],
    xlabelsize: float | None = ...,
    xrot: float | None = ...,
    ylabelsize: float | None = ...,
    yrot: float | None = ...
) -> None: ...

def _check_ax_scales(
    axes: Axes | Sequence[Axes],
    xaxis: str = ...,
    yaxis: str = ...
) -> None: ...

def _check_axes_shape(
    axes: Axes | Sequence[Axes],
    axes_num: int | None = ...,
    layout: tuple[int, int] | None = ...,
    figsize: tuple[float, float] | None = ...
) -> None: ...

def _flatten_visible(axes: Axes | Sequence[Axes]) -> list[Axes]: ...

def _check_has_errorbars(
    axes: Axes | Sequence[Axes],
    xerr: int = ...,
    yerr: int = ...
) -> None: ...

@overload
def _check_box_return_type(
    returned: dict[str, Any],
    return_type: str,
    expected_keys: Sequence[str] | None = ...,
    check_ax_title: bool = ...
) -> None: ...

@overload
def _check_box_return_type(
    returned: Axes,
    return_type: str,
    expected_keys: Sequence[str] | None = ...,
    check_ax_title: bool = ...
) -> None: ...

@overload
def _check_box_return_type(
    returned: tuple[Any, Any],
    return_type: str,
    expected_keys: Sequence[str] | None = ...,
    check_ax_title: bool = ...
) -> None: ...

@overload
def _check_box_return_type(
    returned: Series,
    return_type: str,
    expected_keys: Sequence[str] | None = ...,
    check_ax_title: bool = ...
) -> None: ...

def _check_box_return_type(
    returned: Any,
    return_type: str,
    expected_keys: Sequence[str] | None = ...,
    check_ax_title: bool = ...
) -> None: ...

def _check_grid_settings(
    obj: Any,
    kinds: Sequence[str],
    kws: dict[str, Any] | None = ...
) -> None: ...

def _unpack_cycler(rcParams: dict[str, Any], field: str = ...) -> list[Any]: ...

def get_x_axis(ax: Axes) -> Any: ...

def get_y_axis(ax: Axes) -> Any: ...

def assert_is_valid_plot_return_object(objs: Any) -> None: ...

def _check_plot_works(
    f: Any,
    default_axes: bool = ...,
    **kwargs: Any
) -> Any: ...

def _gen_default_plot(f: Any, fig: Any, **kwargs: Any) -> Any: ...

def _gen_two_subplots(f: Any, fig: Any, **kwargs: Any) -> Any: ...
```