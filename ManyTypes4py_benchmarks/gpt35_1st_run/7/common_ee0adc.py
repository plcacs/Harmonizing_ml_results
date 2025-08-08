from __future__ import annotations
from typing import TYPE_CHECKING, List, Union
import numpy as np
import pandas as pd
from pandas import Series

if TYPE_CHECKING:
    from collections.abc import Sequence
    from matplotlib.axes import Axes

def _check_legend_labels(axes: Union[Axes, Sequence[Axes]], labels: List[str] = None, visible: bool = True) -> None:
    ...

def _check_legend_marker(ax: Axes, expected_markers: List[str] = None, visible: bool = True) -> None:
    ...

def _check_data(xp: Axes, rs: Axes) -> None:
    ...

def _check_visible(collections: Union[Collection, Sequence[Artist]], visible: bool = True) -> None:
    ...

def _check_patches_all_filled(axes: Union[Axes, Sequence[Axes]], filled: bool = True) -> None:
    ...

def _get_colors_mapped(series: Series, colors: List[str]) -> List[str]:
    ...

def _check_colors(collections: List[Artist], linecolors: List[str] = None, facecolors: List[str] = None, mapping: Series = None) -> None:
    ...

def _check_text_labels(texts: Union[Text, Sequence[Text]], expected: Union[str, List[str]]) -> None:
    ...

def _check_ticks_props(axes: Union[Axes, Sequence[Axes]], xlabelsize: int = None, xrot: int = None, ylabelsize: int = None, yrot: int = None) -> None:
    ...

def _check_ax_scales(axes: Union[Axes, Sequence[Axes]], xaxis: str = 'linear', yaxis: str = 'linear') -> None:
    ...

def _check_axes_shape(axes: Union[Axes, Sequence[Axes]], axes_num: int = None, layout: tuple = None, figsize: tuple = None) -> None:
    ...

def _flatten_visible(axes: Union[Axes, Sequence[Axes]]) -> List[Axes]:
    ...

def _check_has_errorbars(axes: Union[Axes, Sequence[Axes]], xerr: int = 0, yerr: int = 0) -> None:
    ...

def _check_box_return_type(returned: object, return_type: str, expected_keys: List[str] = None, check_ax_title: bool = True) -> None:
    ...

def _check_grid_settings(obj, kinds, kws=None):
    ...

def _unpack_cycler(rcParams, field='color'):
    ...

def get_x_axis(ax: Axes) -> Axes:
    ...

def get_y_axis(ax: Axes) -> Axes:
    ...

def assert_is_valid_plot_return_object(objs: Union[Series, np.ndarray]) -> None:
    ...

def _check_plot_works(f, default_axes: bool = False, **kwargs) -> object:
    ...

def _gen_default_plot(f, fig, **kwargs):
    ...

def _gen_two_subplots(f, fig, **kwargs):
    ...
