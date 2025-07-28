from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING, Literal
import warnings
import matplotlib as mpl
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike
import pandas as pd
import pandas.core.common as com
from pandas.util.version import Version
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import LinePlot, MPLPlot
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import create_subplots, flatten_axes, maybe_adjust_figure
if TYPE_CHECKING:
    from collections.abc import Collection
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from pandas._typing import MatplotlibColor

def _set_ticklabels(ax: "Axes", labels: Sequence[str], is_vertical: bool, **kwargs: Any) -> None:
    """Set the tick labels of a given axis.

    Due to https://github.com/matplotlib/matplotlib/pull/17266, we need to handle the
    case of repeated ticks (due to `FixedLocator`) and thus we duplicate the number of
    labels.
    """
    ticks = ax.get_xticks() if is_vertical else ax.get_yticks()
    if len(ticks) != len(labels):
        i, remainder = divmod(len(ticks), len(labels))
        if Version(mpl.__version__) < Version('3.10'):
            assert remainder == 0, remainder
        labels = list(labels) * i
    if is_vertical:
        ax.set_xticklabels(labels, **kwargs)
    else:
        ax.set_yticklabels(labels, **kwargs)

class BoxPlot(LinePlot):

    @property
    def _kind(self) -> Literal['box']:
        return 'box'

    _layout_type: str = 'horizontal'
    _valid_return_types: Tuple[Optional[Literal['axes', 'dict', 'both']], ...] = (None, 'axes', 'dict', 'both')

    class BP(NamedTuple):
        ax: Any = None
        lines: Any = None

    def __init__(
        self,
        data: Any,
        return_type: Union[None, Literal['axes'], Literal['dict'], Literal['both']] = 'axes',
        **kwargs: Any
    ) -> None:
        if return_type not in self._valid_return_types:
            raise ValueError("return_type must be {None, 'axes', 'dict', 'both'}")
        self.return_type: Union[None, Literal['axes'], Literal['dict'], Literal['both']] = return_type
        MPLPlot.__init__(self, data, **kwargs)
        if self.subplots:
            if self.orientation == 'vertical':
                self.sharex = False
            else:
                self.sharey = False

    @classmethod
    def _plot(
        cls,
        ax: "Axes",
        y: np.ndarray,
        column_num: Optional[int] = None,
        return_type: Union[None, Literal['axes'], Literal['dict'], Literal['both']] = 'axes',
        **kwds: Any
    ) -> Tuple[Any, Any]:
        if y.ndim == 2:
            ys = [remove_na_arraylike(v) for v in y]
            ys = [v if v.size > 0 else np.array([np.nan]) for v in ys]
        else:
            ys = remove_na_arraylike(y)
        bp = ax.boxplot(ys, **kwds)
        if return_type == 'dict':
            return (bp, bp)
        elif return_type == 'both':
            return (cls.BP(ax=ax, lines=bp), bp)
        else:
            return (ax, bp)

    def _validate_color_args(self, color: Any, colormap: Optional[Any]) -> Any:
        if color is lib.no_default:
            return None
        if colormap is not None:
            warnings.warn("'color' and 'colormap' cannot be used simultaneously. Using 'color'", stacklevel=find_stack_level())
        if isinstance(color, dict):
            valid_keys = ['boxes', 'whiskers', 'medians', 'caps']
            for key in color:
                if key not in valid_keys:
                    raise ValueError(f"color dict contains invalid key '{key}'. The key must be either {valid_keys}")
        return color

    @cache_readonly
    def _color_attrs(self) -> Tuple[Any, ...]:
        return get_standard_colors(num_colors=3, colormap=self.colormap, color=None)

    @cache_readonly
    def _boxes_c(self) -> Any:
        return self._color_attrs[0]

    @cache_readonly
    def _whiskers_c(self) -> Any:
        return self._color_attrs[0]

    @cache_readonly
    def _medians_c(self) -> Any:
        return self._color_attrs[2]

    @cache_readonly
    def _caps_c(self) -> Any:
        return self._color_attrs[0]

    def _get_colors(self, num_colors: Optional[int] = None, color_kwds: str = 'color') -> Any:
        pass

    def maybe_color_bp(self, bp: Dict[str, Any]) -> None:
        if isinstance(self.color, dict):
            boxes = self.color.get('boxes', self._boxes_c)
            whiskers = self.color.get('whiskers', self._whiskers_c)
            medians = self.color.get('medians', self._medians_c)
            caps = self.color.get('caps', self._caps_c)
        else:
            boxes = self.color or self._boxes_c
            whiskers = self.color or self._whiskers_c
            medians = self.color or self._medians_c
            caps = self.color or self._caps_c
        color_tup: Tuple[Any, Any, Any, Any] = (boxes, whiskers, medians, caps)
        maybe_color_bp(bp, color_tup=color_tup, **self.kwds)

    def _make_plot(self, fig: "Figure") -> None:
        if self.subplots:
            self._return_obj = pd.Series(dtype=object)
            data = create_iter_data_given_by(self.data, self._kind) if self.by is not None else self.data
            for i, (label, y) in enumerate(self._iter_data(data=data)):
                ax = self._get_ax(i)
                kwds = self.kwds.copy()
                if self.by is not None:
                    y = y.T
                    ax.set_title(pprint_thing(label))
                    levels = self.data.columns.levels
                    ticklabels = [pprint_thing(col) for col in levels[0]]
                else:
                    ticklabels = [pprint_thing(label)]
                ret, bp = self._plot(ax, y, column_num=i, return_type=self.return_type, **kwds)
                self.maybe_color_bp(bp)
                self._return_obj[label] = ret
                _set_ticklabels(ax=ax, labels=ticklabels, is_vertical=self.orientation == 'vertical')
        else:
            y = self.data.values.T
            ax = self._get_ax(0)
            kwds = self.kwds.copy()
            ret, bp = self._plot(ax, y, column_num=0, return_type=self.return_type, **kwds)
            self.maybe_color_bp(bp)
            self._return_obj = ret
            labels = [pprint_thing(left) for left in self.data.columns]
            if not self.use_index:
                labels = [pprint_thing(key) for key in range(len(labels))]
            _set_ticklabels(ax=ax, labels=labels, is_vertical=self.orientation == 'vertical')

    def _make_legend(self) -> None:
        pass

    def _post_plot_logic(self, ax: "Axes", data: Any) -> None:
        if self.xlabel:
            ax.set_xlabel(pprint_thing(self.xlabel))
        if self.ylabel:
            ax.set_ylabel(pprint_thing(self.ylabel))

    @property
    def orientation(self) -> Literal['vertical', 'horizontal']:
        if self.kwds.get('vert', True):
            return 'vertical'
        else:
            return 'horizontal'

    @property
    def result(self) -> Any:
        if self.return_type is None:
            return super().result
        else:
            return self._return_obj

def maybe_color_bp(bp: Dict[str, Any], color_tup: Tuple[Any, Any, Any, Any], **kwds: Any) -> None:
    if not kwds.get('boxprops'):
        mpl.artist.setp(bp['boxes'], color=color_tup[0], alpha=1)
    if not kwds.get('whiskerprops'):
        mpl.artist.setp(bp['whiskers'], color=color_tup[1], alpha=1)
    if not kwds.get('medianprops'):
        mpl.artist.setp(bp['medians'], color=color_tup[2], alpha=1)
    if not kwds.get('capprops'):
        mpl.artist.setp(bp['caps'], color=color_tup[3], alpha=1)

def _grouped_plot_by_column(
    plotf: Any,
    data: pd.DataFrame,
    columns: Optional[Sequence[Any]] = None,
    by: Optional[Any] = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Optional[Any] = None,
    ax: Optional["Axes"] = None,
    layout: Optional[Any] = None,
    return_type: Optional[Union[None, Literal['axes'], Literal['dict'], Literal['both']]] = None,
    **kwargs: Any
) -> Union["Axes", pd.Series]:
    grouped = data.groupby(by, observed=False)
    if columns is None:
        if not isinstance(by, (list, tuple)):
            by = [by]
        columns = data._get_numeric_data().columns.difference(by)
    naxes = len(columns)
    fig, axes = create_subplots(
        naxes=naxes,
        sharex=kwargs.pop('sharex', True),
        sharey=kwargs.pop('sharey', True),
        figsize=figsize,
        ax=ax,
        layout=layout
    )
    xlabel: Any
    ylabel: Any
    xlabel, ylabel = (kwargs.pop('xlabel', None), kwargs.pop('ylabel', None))
    if kwargs.get('vert', True):
        xlabel = xlabel or by
    else:
        ylabel = ylabel or by
    ax_values: List[Any] = []
    for ax_item, col in zip(flatten_axes(axes), columns):
        gp_col = grouped[col]
        keys, values = zip(*gp_col)
        re_plotf = plotf(keys, values, ax_item, xlabel=xlabel, ylabel=ylabel, **kwargs)
        ax_item.set_title(col)
        ax_values.append(re_plotf)
        ax_item.grid(grid)
    result: Union[pd.Series, "Axes"] = pd.Series(ax_values, index=columns, copy=False)
    if return_type is None:
        result = axes
    byline = by[0] if len(by) == 1 else by
    fig.suptitle(f'Boxplot grouped by {byline}')
    maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    return result

def boxplot(
    data: Union[pd.DataFrame, ABCSeries],
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Any] = None,
    ax: Optional["Axes"] = None,
    fontsize: Optional[Union[int, float]] = None,
    rot: Union[int, float] = 0,
    grid: bool = True,
    figsize: Optional[Any] = None,
    layout: Optional[Any] = None,
    return_type: Optional[Union[None, Literal['axes'], Literal['dict'], Literal['both']]] = None,
    **kwds: Any
) -> Any:
    import matplotlib.pyplot as plt
    if return_type not in BoxPlot._valid_return_types:
        raise ValueError("return_type must be {'axes', 'dict', 'both'}")
    if isinstance(data, ABCSeries):
        data = data.to_frame('x')
        column = 'x'

    def _get_colors() -> np.ndarray:
        result_list = get_standard_colors(num_colors=3)
        result = np.take(result_list, [0, 0, 2])
        result = np.append(result, 'k')
        colors = kwds.pop('color', None)
        if colors:
            if is_dict_like(colors):
                valid_keys = ['boxes', 'whiskers', 'medians', 'caps']
                key_to_index = dict(zip(valid_keys, range(4)))
                for key, value in colors.items():
                    if key in valid_keys:
                        result[key_to_index[key]] = value
                    else:
                        raise ValueError(f"color dict contains invalid key '{key}'. The key must be either {valid_keys}")
            else:
                result.fill(colors)
        return result

    def plot_group(
        keys: Sequence[Any],
        values: Sequence[Any],
        ax: "Axes",
        **kwds_inner: Any
    ) -> Any:
        xlabel_inner, ylabel_inner = (kwds_inner.pop('xlabel', None), kwds_inner.pop('ylabel', None))
        if xlabel_inner:
            ax.set_xlabel(pprint_thing(xlabel_inner))
        if ylabel_inner:
            ax.set_ylabel(pprint_thing(ylabel_inner))
        keys = [pprint_thing(x) for x in keys]
        values = [np.asarray(remove_na_arraylike(v), dtype=object) for v in values]
        bp = ax.boxplot(values, **kwds_inner)
        if fontsize is not None:
            ax.tick_params(axis='both', labelsize=fontsize)
        _set_ticklabels(ax=ax, labels=keys, is_vertical=kwds_inner.get('vert', True), rotation=rot)
        maybe_color_bp(bp, color_tup=colors, **kwds_inner)
        if return_type == 'dict':
            return bp
        elif return_type == 'both':
            return BoxPlot.BP(ax=ax, lines=bp)
        else:
            return ax

    colors: np.ndarray = _get_colors()
    if column is None:
        columns: Optional[Sequence[Any]] = None
    elif isinstance(column, (list, tuple)):
        columns = column
    else:
        columns = [column]
    if by is not None:
        result = _grouped_plot_by_column(
            plot_group, data, columns=columns, by=by, grid=grid, figsize=figsize, ax=ax, layout=layout, return_type=return_type, **kwds
        )
    else:
        if return_type is None:
            return_type = 'axes'
        if layout is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")
        if ax is None:
            rc: Dict[str, Any] = {'figure.figsize': figsize} if figsize is not None else {}
            with mpl.rc_context(rc):
                ax = plt.gca()
        data = data._get_numeric_data()
        naxes = len(data.columns)
        if naxes == 0:
            raise ValueError('boxplot method requires numerical columns, nothing to plot.')
        if columns is None:
            columns = data.columns
        else:
            data = data[columns]
        result = plot_group(columns, data.values.T, ax, **kwds)
        ax.grid(grid)
    return result

def boxplot_frame(
    self: pd.DataFrame,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Any] = None,
    ax: Optional["Axes"] = None,
    fontsize: Optional[Union[int, float]] = None,
    rot: Union[int, float] = 0,
    grid: bool = True,
    figsize: Optional[Any] = None,
    layout: Optional[Any] = None,
    return_type: Optional[Union[None, Literal['axes'], Literal['dict'], Literal['both']]] = None,
    **kwds: Any
) -> Any:
    import matplotlib.pyplot as plt
    ax_out = boxplot(self, column=column, by=by, ax=ax, fontsize=fontsize, grid=grid, rot=rot, figsize=figsize, layout=layout, return_type=return_type, **kwds)
    plt.draw_if_interactive()
    return ax_out

def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: Optional[Union[str, Sequence[str]]] = None,
    fontsize: Optional[Union[int, float]] = None,
    rot: Union[int, float] = 0,
    grid: bool = True,
    ax: Optional["Axes"] = None,
    figsize: Optional[Any] = None,
    layout: Optional[Any] = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any
) -> Any:
    if subplots is True:
        naxes = len(grouped)
        fig, axes = create_subplots(naxes=naxes, squeeze=False, ax=ax, sharex=sharex, sharey=sharey, figsize=figsize, layout=layout)
        data: Dict[Any, Any] = {}
        for (key, group), ax_item in zip(grouped, flatten_axes(axes)):
            d = group.boxplot(ax=ax_item, column=column, fontsize=fontsize, rot=rot, grid=grid, **kwds)
            ax_item.set_title(pprint_thing(key))
            data[key] = d
        ret: pd.Series = pd.Series(data)
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    else:
        keys, frames = zip(*grouped)
        df = pd.concat(frames, keys=keys, axis=1)
        if column is not None:
            column = com.convert_to_list_like(column)
            multi_key = pd.MultiIndex.from_product([keys, column])
            column = list(multi_key.values)
        ret = df.boxplot(column=column, fontsize=fontsize, rot=rot, grid=grid, ax=ax, figsize=figsize, layout=layout, **kwds)
    return ret