"""
Module consolidating common testing functions for checking plotting.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union, Optional, Sequence
import numpy as np
import pandas as pd
from pandas import Series
import pandas._testing as tm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist
    from matplotlib.collections import Collection, LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    import matplotlib.pyplot as plt

def _check_legend_labels(
    axes: Union[Axes, Sequence[Axes]],
    labels: Optional[Sequence[str]] = None,
    visible: bool = True
) -> None: ...

def _check_legend_marker(
    ax: Axes,
    expected_markers: Optional[Sequence[Any]] = None,
    visible: bool = True
) -> None: ...

def _check_data(xp: Axes, rs: Axes) -> None: ...

def _check_visible(
    collections: Union[Artist, Sequence[Artist]],
    visible: bool = True
) -> None: ...

def _check_patches_all_filled(
    axes: Union[Axes, Sequence[Axes]],
    filled: bool = True
) -> None: ...

def _get_colors_mapped(series: Series, colors: Sequence[Any]) -> list[Any]: ...

def _check_colors(
    collections: Sequence[Artist],
    linecolors: Optional[Sequence[Any]] = None,
    facecolors: Optional[Sequence[Any]] = None,
    mapping: Optional[Series] = None
) -> None: ...

def _check_text_labels(
    texts: Union[Text, Sequence[Text]],
    expected: Union[str, Sequence[str]]
) -> None: ...

def _check_ticks_props(
    axes: Union[Axes, Sequence[Axes]],
    xlabelsize: Optional[float] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[float] = None,
    yrot: Optional[float] = None
) -> None: ...

def _check_ax_scales(
    axes: Union[Axes, Sequence[Axes]],
    xaxis: str = "linear",
    yaxis: str = "linear"
) -> None: ...

def _check_axes_shape(
    axes: Union[Axes, Sequence[Axes]],
    axes_num: Optional[int] = None,
    layout: Optional[tuple[int, int]] = None,
    figsize: Optional[tuple[float, float]] = None
) -> None: ...

def _flatten_visible(axes: Union[Axes, Sequence[Axes]]) -> list[Axes]: ...

def _check_has_errorbars(
    axes: Union[Axes, Sequence[Axes]],
    xerr: int = 0,
    yerr: int = 0
) -> None: ...

def _check_box_return_type(
    returned: Any,
    return_type: Optional[str],
    expected_keys: Optional[Sequence[str]] = None,
    check_ax_title: bool = True
) -> None: ...

def _check_grid_settings(
    obj: Any,
    kinds: Sequence[str],
    kws: Optional[dict[str, Any]] = None
) -> None: ...

def _unpack_cycler(rcParams: dict[str, Any], field: str = "color") -> list[Any]: ...

def get_x_axis(ax: Axes) -> Any: ...

def get_y_axis(ax: Axes) -> Any: ...

def assert_is_valid_plot_return_object(objs: Any) -> None: ...

def _check_plot_works(
    f: Any,
    default_axes: bool = False,
    **kwargs: Any
) -> Any: ...

def _gen_default_plot(f: Any, fig: Any, **kwargs: Any) -> Iterable[Any]: ...

def _gen_two_subplots(f: Any, fig: Any, **kwargs: Any) -> Iterable[Any]: ...