from typing import Tuple, List, Union

def _check_axes_shape(axes: Union[Tuple[mpl.axes.Axes], Tuple[Tuple[mpl.axes.Axes]]], axes_num: int, layout: Tuple[int, int]) -> None:
def _check_colors(patches: List[mpl.patches.Patch], facecolors: List[str]) -> None:
def _check_legend_labels(ax: mpl.axes.Axes, label: str) -> None:
def _check_patches_all_filled(ax: mpl.axes.Axes, filled: bool) -> None:
def _check_plot_works(func, **kwargs) -> None:
def _check_text_labels(labels, expected_labels) -> None:
def _check_ticks_props(axes: Union[Tuple[mpl.axes.Axes], Tuple[Tuple[mpl.axes.Axes]]], xlabelsize: int, xrot: int, ylabelsize: int, yrot: int) -> None:
def get_x_axis(ax: mpl.axes.Axes) -> mpl.axis.Axis:
def get_y_axis(ax: mpl.axes.Axes) -> mpl.axis.Axis:
