from __future__ import annotations
from math import ceil
from typing import Any, Dict, Generator, Iterable, Optional, Tuple, Union
import warnings
import matplotlib as mpl
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
if False:  # TYPE_CHECKING
    from collections.abc import Generator, Iterable
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.table import Table
    from pandas import DataFrame, Series

def do_adjust_figure(fig: Any) -> bool:
    """Whether fig has constrained_layout enabled."""
    if not hasattr(fig, 'get_constrained_layout'):
        return False
    return not fig.get_constrained_layout()

def maybe_adjust_figure(fig: Any, *args: Any, **kwargs: Any) -> None:
    """Call fig.subplots_adjust unless fig has constrained_layout enabled."""
    if do_adjust_figure(fig):
        fig.subplots_adjust(*args, **kwargs)

def format_date_labels(ax: Any, rot: float) -> None:
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
        label.set_rotation(rot)
    fig = ax.get_figure()
    if fig is not None:
        maybe_adjust_figure(fig, bottom=0.2)

def table(ax: Any, data: Union[ABCDataFrame, ABCSeries],
          rowLabels: Optional[Any] = None,
          colLabels: Optional[Any] = None,
          **kwargs: Any) -> Any:
    if isinstance(data, ABCSeries):
        data = data.to_frame()
    elif isinstance(data, ABCDataFrame):
        pass
    else:
        raise ValueError('Input data must be DataFrame or Series')
    if rowLabels is None:
        rowLabels = data.index
    if colLabels is None:
        colLabels = data.columns
    cellText = data.values
    return mpl.table.table(ax, cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, **kwargs)

def _get_layout(nplots: int, layout: Optional[Tuple[int, int]] = None, layout_type: str = 'box') -> Tuple[int, int]:
    if layout is not None:
        if not isinstance(layout, (tuple, list)) or len(layout) != 2:
            raise ValueError('Layout must be a tuple of (rows, columns)')
        nrows, ncols = layout
        if nrows == -1 and ncols > 0:
            layout = (ceil(nplots / ncols), ncols)
        elif ncols == -1 and nrows > 0:
            layout = (nrows, ceil(nplots / nrows))
        elif ncols <= 0 and nrows <= 0:
            msg = 'At least one dimension of layout must be positive'
            raise ValueError(msg)
        nrows, ncols = layout
        if nrows * ncols < nplots:
            raise ValueError(f'Layout of {nrows}x{ncols} must be larger than required size {nplots}')
        return layout
    if layout_type == 'single':
        return (1, 1)
    elif layout_type == 'horizontal':
        return (1, nplots)
    elif layout_type == 'vertical':
        return (nplots, 1)
    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        return layouts[nplots]
    except KeyError:
        k = 1
        while k ** 2 < nplots:
            k += 1
        if (k - 1) * k >= nplots:
            return (k, k - 1)
        else:
            return (k, k)

def create_subplots(naxes: int,
                    sharex: bool = False,
                    sharey: bool = False,
                    squeeze: bool = True,
                    subplot_kw: Optional[Dict[str, Any]] = None,
                    ax: Optional[Any] = None,
                    layout: Optional[Tuple[int, int]] = None,
                    layout_type: str = 'box',
                    **fig_kw: Any) -> Tuple[Any, Union[Any, np.ndarray]]:
    """
    Create a figure with a set of subplots already made.
    """
    import matplotlib.pyplot as plt
    if subplot_kw is None:
        subplot_kw = {}
    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        if is_list_like(ax):
            if squeeze:
                ax = np.fromiter(flatten_axes(ax), dtype=object)
            if layout is not None:
                warnings.warn('When passing multiple axes, layout keyword is ignored.', UserWarning, stacklevel=find_stack_level())
            if sharex or sharey:
                warnings.warn('When passing multiple axes, sharex and sharey are ignored. These settings must be specified when creating axes.', UserWarning, stacklevel=find_stack_level())
            if ax.size == naxes:
                fig = ax.flat[0].get_figure()
                return (fig, ax)
            else:
                raise ValueError(f'The number of passed axes must be {naxes}, the same as the output plot')
        fig = ax.get_figure()
        if naxes == 1:
            if squeeze:
                return (fig, ax)
            else:
                return (fig, np.fromiter(flatten_axes(ax), dtype=object))
        else:
            warnings.warn('To output multiple subplots, the figure containing the passed axes is being cleared.', UserWarning, stacklevel=find_stack_level())
            fig.clear()
    nrows, ncols = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = nrows * ncols
    axarr = np.empty(nplots, dtype=object)
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)
    if sharex:
        subplot_kw['sharex'] = ax0
    if sharey:
        subplot_kw['sharey'] = ax0
    axarr[0] = ax0
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        if i >= naxes:
            kwds['sharex'] = None
            kwds['sharey'] = None
        ax_instance = fig.add_subplot(nrows, ncols, i + 1, **kwds)
        axarr[i] = ax_instance
    if naxes != nplots:
        for ax_instance in axarr[naxes:]:
            ax_instance.set_visible(False)
    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)
    if squeeze:
        if nplots == 1:
            axes_out = axarr[0]
        else:
            axes_out = axarr.reshape(nrows, ncols).squeeze()
    else:
        axes_out = axarr.reshape(nrows, ncols)
    return (fig, axes_out)

def _remove_labels_from_axis(axis: Any) -> None:
    for t in axis.get_majorticklabels():
        t.set_visible(False)
    if isinstance(axis.get_minor_locator(), mpl.ticker.NullLocator):
        axis.set_minor_locator(mpl.ticker.AutoLocator())
    if isinstance(axis.get_minor_formatter(), mpl.ticker.NullFormatter):
        axis.set_minor_formatter(mpl.ticker.FormatStrFormatter(''))
    for t in axis.get_minorticklabels():
        t.set_visible(False)
    axis.get_label().set_visible(False)

def _has_externally_shared_axis(ax1: Any, compare_axis: str) -> bool:
    """
    Return whether an axis is externally shared.
    """
    if compare_axis == 'x':
        axes = ax1.get_shared_x_axes()
    elif compare_axis == 'y':
        axes = ax1.get_shared_y_axes()
    else:
        raise ValueError("_has_externally_shared_axis() needs 'x' or 'y' as a second parameter")
    axes_siblings = axes.get_siblings(ax1)
    ax1_points = ax1.get_position().get_points()
    for ax2 in axes_siblings:
        if not np.array_equal(ax1_points, ax2.get_position().get_points()):
            return True
    return False

def handle_shared_axes(axarr: np.ndarray[Any, Any],
                       nplots: int,
                       naxes: int,
                       nrows: int,
                       ncols: int,
                       sharex: bool,
                       sharey: bool) -> None:
    if nplots > 1:
        row_num = lambda x: x.get_subplotspec().rowspan.start
        col_num = lambda x: x.get_subplotspec().colspan.start
        is_first_col = lambda x: x.get_subplotspec().is_first_col()
        if nrows > 1:
            try:
                layout = np.zeros((nrows + 1, ncols + 1), dtype=np.bool_)
                for ax in axarr:
                    layout[row_num(ax), col_num(ax)] = ax.get_visible()
                for ax in axarr:
                    if not layout[row_num(ax) + 1, col_num(ax)]:
                        continue
                    if sharex or _has_externally_shared_axis(ax, 'x'):
                        _remove_labels_from_axis(ax.xaxis)
            except IndexError:
                is_last_row = lambda x: x.get_subplotspec().is_last_row()
                for ax in axarr:
                    if is_last_row(ax):
                        continue
                    if sharex or _has_externally_shared_axis(ax, 'x'):
                        _remove_labels_from_axis(ax.xaxis)
        if ncols > 1:
            for ax in axarr:
                if is_first_col(ax):
                    continue
                if sharey or _has_externally_shared_axis(ax, 'y'):
                    _remove_labels_from_axis(ax.yaxis)

def flatten_axes(axes: Any) -> Generator[Any, None, None]:
    if not is_list_like(axes):
        yield axes
    elif isinstance(axes, (np.ndarray, ABCIndex)):
        yield from np.asarray(axes).reshape(-1)
    else:
        yield from axes

def set_ticks_props(axes: Any,
                    xlabelsize: Optional[float] = None,
                    xrot: Optional[float] = None,
                    ylabelsize: Optional[float] = None,
                    yrot: Optional[float] = None) -> Any:
    for ax in flatten_axes(axes):
        if xlabelsize is not None:
            mpl.artist.setp(ax.get_xticklabels(), fontsize=xlabelsize)
        if xrot is not None:
            mpl.artist.setp(ax.get_xticklabels(), rotation=xrot)
        if ylabelsize is not None:
            mpl.artist.setp(ax.get_yticklabels(), fontsize=ylabelsize)
        if yrot is not None:
            mpl.artist.setp(ax.get_yticklabels(), rotation=yrot)
    return axes

def get_all_lines(ax: Any) -> list[Any]:
    lines = ax.get_lines()
    if hasattr(ax, 'right_ax'):
        lines += ax.right_ax.get_lines()
    if hasattr(ax, 'left_ax'):
        lines += ax.left_ax.get_lines()
    return lines

def get_xlim(lines: Iterable[Any]) -> Tuple[float, float]:
    left, right = (np.inf, -np.inf)
    for line in lines:
        x = line.get_xdata(orig=False)
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    return (left, right)